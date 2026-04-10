"""
Agent 3 — Critic (Bonus +5 pts).
Verifies citations against retrieved sources, flags unsupported claims.
"""

import json
import os
import re
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(override=True)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

llm_fast = ChatCerebras(
    model="llama3.1-8b",
    api_key=CEREBRAS_API_KEY,
    temperature=0,
)

SUPPORT_CHECK_PROMPT = """You are a fact-checking assistant. Determine if the given excerpt from a document could support the claim. The excerpt may be a portion of a larger document. If the excerpt contains relevant information that relates to the claim's topic and doesn't contradict it, consider it supported.

Claim: "{claim}"
Excerpt from {doc_id}: "{excerpt}"

Does the excerpt support or relate to the claim? Answer in JSON only:
{{"supported": true or false, "explanation": "one sentence"}}"""


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found: {text[:200]}")


def _extract_cited_sentences(answer: str) -> list[dict]:
    """
    Extract sentences with citations from the answer.
    Returns list of {"sentence": ..., "doc_id": "DOC-XXX"}.
    """
    results = []
    # Split on sentence boundaries, keeping citations attached
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    for sentence in sentences:
        citations = re.findall(r"\[DOC-(\d{3})\]", sentence)
        for doc_num in citations:
            doc_id = f"DOC-{doc_num}"
            # Clean the sentence for claim checking
            clean_sentence = re.sub(r"\[DOC-\d{3}\]", "", sentence).strip()
            if clean_sentence:
                results.append({"sentence": clean_sentence, "doc_id": doc_id})
    return results


def _extract_stats_from_text(text: str) -> set[str]:
    """Extract all numbers, percentages, and statistics from text."""
    stats = set()
    # Percentages: 74.4%, 76%, 3.6%, ~78%
    for m in re.findall(r'~?(\d+(?:[\.,]\d+)?)\s*%', text):
        stats.add(m.replace(',', '.') + '%')
    # Numbers with commas: 1,016  2,400  950
    for m in re.findall(r'\b(\d{1,3}(?:,\d{3})+)\b', text):
        stats.add(m)
    # Plain significant numbers (3+ digits or contextually important)
    for m in re.findall(r'\b(\d{3,})\b', text):
        stats.add(m)
    # Dollar amounts: $2.1B, $500M
    for m in re.findall(r'\$[\d,.]+[BMK]?', text):
        stats.add(m)
    return stats


def _verify_stats_against_source(answer: str, chunks: list[dict], question: str = "") -> list[str]:
    """
    For each statistic in the answer that has a [DOC-XXX] citation,
    verify that the cited document's chunks actually contain that number.
    Returns list of mismatch warnings.
    """
    mismatches = []
    question_stats = _extract_stats_from_text(question) if question else set()
    # Build a map: doc_id -> all text from that doc's chunks
    doc_texts = {}
    for chunk in chunks:
        did = chunk["doc_id"]
        if did not in doc_texts:
            doc_texts[did] = ""
        doc_texts[did] += " " + chunk["text"]

    # Split answer into sentences and find stats + their citations
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    for sentence in sentences:
        citations_in_sentence = re.findall(r'\[DOC-(\d{3})\]', sentence)
        if not citations_in_sentence:
            continue
        # Remove citation markers to get clean text
        clean = re.sub(r'\[?DOC-\d{3}\]?', '', sentence).strip()
        stats_in_sentence = _extract_stats_from_text(clean)
        if not stats_in_sentence:
            continue

        for doc_num in citations_in_sentence:
            doc_id = f"DOC-{doc_num}"
            doc_text = doc_texts.get(doc_id, "")
            if not doc_text:
                continue
            doc_text_lower = doc_text.lower()
            for stat in stats_in_sentence:
                # Whitelist stats that the user explicitly provided
                if stat in question_stats:
                    continue
                # Check if the stat appears in the cited doc's chunks
                if stat.lower() not in doc_text_lower:
                    # Also check without % sign in case formatting differs
                    # And check without commas
                    bare = stat.replace('%', '')
                    no_comma = stat.replace(',', '')
                    if bare.lower() not in doc_text_lower and no_comma.lower() not in doc_text_lower:
                        mismatches.append(
                            f"Stat '{stat}' in claim citing {doc_id} not found in {doc_id} chunks: "
                            f"'{clean}'"
                        )
    return mismatches


class CriticAgent:
    """
    Agent 3: Verifies each citation in the answer against retrieved context.
    Flags unsupported claims and computes confidence score.
    Includes stat-level verification to catch number misattribution.
    """

    def __init__(self, retriever=None):
        """
        Args:
            retriever: Optional HybridRetriever for re-retrieval on flagged claims
        """
        self.llm = llm_fast
        self.retriever = retriever

    def _find_chunk_for_doc(self, doc_id: str, chunks: list[dict]) -> str | None:
        """Find all relevant chunk texts for a given doc_id and concatenate them."""
        texts = [chunk["text"] for chunk in chunks if chunk["doc_id"] == doc_id]
        return "\n...\n".join(texts) if texts else None

    def _check_support(self, claim: str, doc_id: str, excerpt: str) -> dict:
        """Ask Llama 8B if the excerpt supports the claim."""
        prompt = SUPPORT_CHECK_PROMPT.format(
            claim=claim,
            doc_id=doc_id,
            excerpt=excerpt[:8000],  # Limit excerpt length but wide enough to fit chunks
        )
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a precise fact-checker. Output JSON only."),
                HumanMessage(content=prompt),
            ])
            result = _parse_json_response(response.content)
            return result
        except Exception as e:
            print(f"[Critic] Support check failed for {doc_id}: {e}")
            return {"supported": True, "explanation": "Check failed — assuming supported"}

    def run(
        self,
        question: str,
        reasoner_output: dict,
        context_package: dict,
    ) -> dict:
        """
        Verify the reasoner's answer against retrieved sources.

        Args:
            question: Original user question
            reasoner_output: Output from Agent 2 (chain_of_thought, answer, citations)
            context_package: Output from Agent 1 (route, retrieved_chunks, source_doc_ids)

        Returns:
            {
                "verified_answer": "...",
                "flagged_claims": [...],
                "confidence_score": 0.0-1.0,
                "needs_rerun": bool
            }
        """
        answer = reasoner_output.get("answer", "")
        retrieved_chunks = context_package.get("retrieved_chunks", [])
        valid_doc_ids = set(context_package.get("source_doc_ids", []))

        # Extract all cited sentences
        cited_sentences = _extract_cited_sentences(answer)
        if not cited_sentences:
            return {
                "verified_answer": answer,
                "flagged_claims": [],
                "confidence_score": 1.0,
                "needs_rerun": False,
            }

        print(f"[Critic] Checking {len(cited_sentences)} cited claims...")

        flagged_claims = []
        total_claims = len(cited_sentences)
        supported_claims = 0

        for item in cited_sentences:
            doc_id = item["doc_id"]
            claim = item["sentence"]

            # Step 1: Existence check — is doc_id in retrieved context?
            if doc_id not in valid_doc_ids:
                msg = f"Claim '{claim}' cites {doc_id} which was NOT retrieved — not in context"
                print(f"[Critic] {msg}")
                flagged_claims.append(msg)
                continue

            # Step 2: Support check — does the chunk actually support the claim?
            excerpt = self._find_chunk_for_doc(doc_id, retrieved_chunks)
            if excerpt is None:
                msg = f"Claim '{claim}' cites {doc_id} but no chunk text found for verification"
                print(f"[Critic] {msg}")
                flagged_claims.append(msg)
                continue

            check = self._check_support(claim, doc_id, excerpt)
            if check.get("supported", True):
                supported_claims += 1
            else:
                msg = f"Claim '{claim}' cites {doc_id} — UNSUPPORTED: {check.get('explanation', 'N/A')}"
                print(f"[Critic] {msg}")
                flagged_claims.append(msg)

        # Compute confidence
        confidence = supported_claims / total_claims if total_claims > 0 else 1.0

        # Step 3: Stat-level verification — check numbers are in the right source doc
        stat_mismatches = _verify_stats_against_source(answer, retrieved_chunks, question)
        if stat_mismatches:
            print(f"[Critic] Stat mismatches found: {len(stat_mismatches)}")
            for mm in stat_mismatches:
                print(f"  [Critic] {mm}")
            flagged_claims.extend(stat_mismatches)
            # Reduce confidence proportionally
            mismatch_penalty = len(stat_mismatches) / max(total_claims, 1)
            confidence = max(0.0, confidence - mismatch_penalty)

        # Mark unsupported citations in the answer
        verified_answer = answer
        for flag in flagged_claims:
            # Find the doc_id mentioned in the flag
            flag_doc_match = re.search(r"cites (DOC-\d{3})", flag)
            if flag_doc_match and "NOT retrieved" in flag:
                bad_id = flag_doc_match.group(1)
                verified_answer = verified_answer.replace(
                    f"[{bad_id}]", f"[{bad_id}] [UNSUPPORTED]"
                )

        needs_rerun = len(flagged_claims) > 1

        # Optional re-retrieval for flagged claims
        if needs_rerun and self.retriever is not None:
            print(f"[Critic] {len(flagged_claims)} claims flagged — could trigger re-retrieval")

        print(f"[Critic] Confidence: {confidence:.2f} — {len(flagged_claims)} flagged out of {total_claims} claims")

        return {
            "verified_answer": verified_answer,
            "flagged_claims": flagged_claims,
            "confidence_score": round(confidence, 3),
            "needs_rerun": needs_rerun,
        }
