"""
Agent 2 — Reasoner & Synthesizer.
Produces chain-of-thought reasoning + cited answers from retrieved context.
"""

import json
import os
import re
import time
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

llm_fast = ChatCerebras(
    model="llama3.1-8b",
    api_key=CEREBRAS_API_KEY,
    temperature=0,
)

llm_heavy = ChatCerebras(
    model="qwen-3-235b-a22b-instruct-2507",
    api_key=CEREBRAS_API_KEY,
    temperature=0.1,
)

# ── System prompt ────────────────────────────────────────────────────────────

REASONER_SYSTEM_PROMPT = """You are a medical AI research analyst. You answer questions strictly based on the retrieved documents provided.

RULES:
1. Every sentence containing a factual claim MUST end with the citation [DOC-XXX] where XXX is the document number.
2. ONLY cite doc_ids that appear in the RETRIEVED CONTEXT below. Never invent or guess doc_ids.
3. Your chain_of_thought field must show step-by-step reasoning:
   - Step 1: Identify ALL relevant documents and what key information each provides
   - Step 2: Extract specific facts, statistics, and findings from EACH relevant document
   - Step 3: Synthesize across documents to build a comprehensive answer covering ALL parts of the question
   - Step 4: Verify every factual claim has a supporting citation
4. The answer MUST be comprehensive — address ALL parts of the question using evidence from as many relevant documents as possible.
5. When the question asks for examples, impacts, or comparisons, include specific data points from EVERY relevant document in the context. Do NOT stop at 2-3 sources if more relevant data exists.
6. The citations array must list every unique doc_id cited in the answer with a relevant excerpt.
7. If the retrieved context does not contain enough information to answer a part of the question, explicitly state that.

OUTPUT FORMAT (JSON only, no other text):
{
  "chain_of_thought": "Step 1: ...\\nStep 2: ...\\nStep 3: ...\\nStep 4: ...",
  "answer": "Comprehensive answer with [DOC-001] citations for every factual claim.",
  "citations": [
    {"doc_id": "DOC-001", "title": "document title", "relevant_excerpt": "key text from the chunk"}
  ]
}"""

# Prompt for Llama 8B — matches original blueprint spec, proven to work well
REASONER_SYSTEM_PROMPT_LITE = """You are a medical AI research analyst. You answer questions strictly based on the retrieved documents provided.

RULES:
1. Every sentence containing a factual claim MUST end with the citation [DOC-XXX] where XXX is the document number.
2. ONLY cite doc_ids that appear in the RETRIEVED CONTEXT below. Never invent or guess doc_ids.
3. Your chain_of_thought field must show your step-by-step reasoning with numbered steps (Step 1, Step 2, Step 3) BEFORE writing the answer. Each step should be 1-2 sentences identifying relevant documents and key facts.
4. The answer field must be comprehensive, covering ALL parts of the question. Cite evidence from as many relevant documents as possible — do not stop at 1-2 sources when more exist.
5. The citations array must list every unique doc_id cited in the answer with a relevant excerpt.

CRITICAL: You MUST respond with valid JSON only. No text before or after the JSON object.

{
  "chain_of_thought": "Step 1: ...\nStep 2: ...\nStep 3: ...",
  "answer": "Factual claim here [DOC-001]. Another claim [DOC-018].",
  "citations": [
    {"doc_id": "DOC-001", "title": "...", "relevant_excerpt": "..."}
  ]
}"""


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, with recovery for truncated/malformed JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Attempt 1: Full JSON extraction
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Attempt 2: Extract individual fields via regex (handles truncated JSON)
    result = {}

    # Extract "answer" field
    answer_match = re.search(
        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL
    )
    if answer_match:
        raw = answer_match.group(1)
        result["answer"] = raw.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')

    # Extract chain_of_thought field
    cot_match = re.search(
        r'"chain_of_thought"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL
    )
    if cot_match:
        raw = cot_match.group(1)
        result["chain_of_thought"] = raw.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')[:1500]

    # Extract citations array
    cit_match = re.search(r'"citations"\s*:\s*(\[.*?\])', text, re.DOTALL)
    if cit_match:
        try:
            result["citations"] = json.loads(cit_match.group(1))
        except json.JSONDecodeError:
            pass

    if result.get("answer"):
        if "chain_of_thought" not in result:
            result["chain_of_thought"] = "Recovered from partial response."
        if "citations" not in result:
            doc_ids = list(set(re.findall(r'DOC-\d{3}', result["answer"])))
            result["citations"] = [{"doc_id": did, "title": "", "relevant_excerpt": ""} for did in doc_ids]
        return result

    # Attempt 3: Recover from raw text with DOC citations (repetitive/looping responses)
    doc_ids = list(set(re.findall(r'DOC-\d{3}', text)))
    if doc_ids and len(text) > 100:
        # Deduplicate repetitive lines
        lines = text.split('\n')
        seen = set()
        unique_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique_lines.append(line)
        clean = '\n'.join(unique_lines)
        # Remove JSON scaffolding
        clean = re.sub(r'^\s*\{?\s*"chain_of_thought"\s*:\s*"?', '', clean)
        clean = re.sub(r'\\n', '\n', clean)
        clean = re.sub(r'\\"', '"', clean)
        # Ensure DOC refs have brackets
        for did in doc_ids:
            clean = re.sub(rf'(?<!\[){re.escape(did)}(?!\])', f'[{did}]', clean)

        # Separate CoT steps from factual answer content
        cot_part = ""
        answer_part = clean
        # Detect "Step N:" pattern at start — split CoT from answer
        step_pattern = re.compile(r'((?:Step\s*\d+\s*:.*?\n?)+)', re.IGNORECASE | re.DOTALL)
        step_match = step_pattern.match(clean.strip())
        if step_match:
            cot_part = step_match.group(1).strip()
            remainder = clean[step_match.end():].strip()
            # Remove "answer": prefix if present
            remainder = re.sub(r'^"?answer"?\s*:?\s*"?', '', remainder).strip()
            if remainder and re.search(r'\[DOC-\d{3}\]', remainder):
                answer_part = remainder
            elif re.search(r'\[DOC-\d{3}\]', cot_part):
                # Steps contain citations — use the entire text as answer too
                answer_part = clean
            else:
                answer_part = clean

        if not cot_part:
            cot_part = "Recovered from malformed response."

        return {
            "chain_of_thought": cot_part[:1500],
            "answer": answer_part[:3000],
            "citations": [{"doc_id": did, "title": "", "relevant_excerpt": ""} for did in doc_ids],
        }

    raise ValueError(f"No JSON found in LLM response: {text[:200]}")


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"--- CHUNK {i} ---\n"
            f"doc_id: {chunk['doc_id']}\n"
            f"title: {chunk['title']}\n"
            f"source: {chunk['source']}\n"
            f"date: {chunk['date']}\n"
            f"relevance_score: {chunk['relevance_score']:.3f}\n"
            f"text: {chunk['text']}\n"
        )
    return "\n".join(parts)


class ReasonerAgent:
    """
    Agent 2: Reasons over retrieved context with explicit chain-of-thought
    and produces a fully cited answer.
    """

    def run(
        self,
        question: str,
        context_package: dict,
        conversation_history: list | None = None,
    ) -> dict:
        """
        Synthesize an answer from the context package.

        Args:
            question: The user's question
            context_package: Output from Agent 1 (route, sub_queries, retrieved_chunks, source_doc_ids)
            conversation_history: List of {"user": ..., "assistant_answer": ...} dicts

        Returns:
            {
                "chain_of_thought": "...",
                "answer": "...",
                "citations": [{"doc_id": ..., "title": ..., "relevant_excerpt": ...}]
            }
        """
        if conversation_history is None:
            conversation_history = []

        # Model selection: Qwen 235B for complex, Llama 8B for simple
        route = context_package.get("route", "complex")
        if route == "simple":
            llm = llm_fast
            model_name = "llama3.1-8b"
            system_prompt = REASONER_SYSTEM_PROMPT_LITE
            max_chunks = 3
        else:
            llm = llm_heavy
            model_name = "qwen-3-235b-a22b-instruct-2507"
            system_prompt = REASONER_SYSTEM_PROMPT
            max_chunks = 8

        using_llama = (llm == llm_fast)
        print(f"[Agent2] Using {model_name} for {route} route")

        # Sort chunks by relevance
        chunks = context_package.get("retrieved_chunks", [])
        chunks_sorted = sorted(chunks, key=lambda c: c.get("relevance_score", 0), reverse=True)

        def _select_diverse_chunks(sorted_chunks, max_k):
            """Select top-K chunks with doc_id diversity: pick best chunk per doc first, then fill."""
            selected = []
            seen_docs = set()
            # Pass 1: best chunk per unique doc_id
            for chunk in sorted_chunks:
                if chunk["doc_id"] not in seen_docs:
                    selected.append(chunk)
                    seen_docs.add(chunk["doc_id"])
                    if len(selected) >= max_k:
                        break
            # Pass 2: if still under max_k, fill with remaining top-scored chunks
            if len(selected) < max_k:
                for chunk in sorted_chunks:
                    if chunk not in selected:
                        selected.append(chunk)
                        if len(selected) >= max_k:
                            break
            return selected

        def _build_messages(prompt, chunk_list):
            """Build LLM messages with given prompt and chunk list."""
            ctx = _format_context(chunk_list)
            avail_ids = sorted(set(c["doc_id"] for c in chunk_list))
            msgs = [SystemMessage(content=prompt)]
            for turn in conversation_history:
                msgs.append(HumanMessage(content=turn["user"]))
                if turn.get("assistant_answer"):
                    msgs.append(AIMessage(content=turn["assistant_answer"]))
            msgs.append(HumanMessage(
                content=f"QUESTION: {question}\n\n"
                        f"AVAILABLE DOC_IDS (only cite these): {', '.join(avail_ids)}\n\n"
                        f"RETRIEVED CONTEXT:\n{ctx}"
            ))
            return msgs, avail_ids

        chunks_to_use = _select_diverse_chunks(chunks_sorted, max_chunks)
        messages, available_doc_ids = _build_messages(system_prompt, chunks_to_use)

        # Call LLM with retry, fallback, and context reduction
        max_retries = 8
        last_response = None
        parse_failures = 0
        for attempt in range(max_retries):
            try:
                response = llm.invoke(messages)
                last_response = response
                result = _parse_json_response(response.content)

                # Validate required fields
                if "answer" not in result:
                    result["answer"] = response.content
                if "chain_of_thought" not in result:
                    result["chain_of_thought"] = "No explicit chain of thought generated."
                if "citations" not in result:
                    result["citations"] = []

                # Post-process: strip any hallucinated doc_ids from the answer
                cited_ids = set(re.findall(r"\[DOC-\d{3}\]", result["answer"]))
                valid_ids = set(f"[{did}]" for did in available_doc_ids)
                hallucinated = cited_ids - valid_ids
                if hallucinated:
                    print(f"[Agent2] WARNING: Stripping hallucinated citations: {hallucinated}")
                    for h_id in hallucinated:
                        result["answer"] = result["answer"].replace(h_id, "[UNVERIFIED]")

                # Deduplicate citations list by doc_id
                seen_doc_ids = set()
                unique_citations = []
                for c in result["citations"]:
                    if c.get("doc_id") not in seen_doc_ids:
                        seen_doc_ids.add(c["doc_id"])
                        unique_citations.append(c)
                result["citations"] = unique_citations

                print(f"[Agent2] Answer generated with {len(result['citations'])} citations.")
                return result

            except (json.JSONDecodeError, ValueError) as e:
                parse_failures += 1
                print(f"[Agent2] Parse error on attempt {attempt + 1}: {e}")

                # After 2 parse failures with Llama, reduce context to give more output room
                if parse_failures == 2 and using_llama and max_chunks > 3:
                    max_chunks = 3
                    chunks_to_use = _select_diverse_chunks(chunks_sorted, max_chunks)
                    messages, available_doc_ids = _build_messages(system_prompt, chunks_to_use)
                    print(f"[Agent2] Reducing context to {max_chunks} chunks for Llama")

                if attempt == max_retries - 1:
                    # Last resort: return raw text as answer
                    raw = last_response.content if last_response else "Error generating answer."
                    # Try to extract DOC citations from raw text
                    doc_ids = list(set(re.findall(r'DOC-\d{3}', raw)))
                    valid = [d for d in doc_ids if d in set(c["doc_id"] for c in chunks_to_use)]
                    return {
                        "chain_of_thought": "Failed to generate structured output after retries.",
                        "answer": raw,
                        "citations": [{"doc_id": d, "title": "", "relevant_excerpt": ""} for d in valid],
                    }
            except Exception as e:
                error_str = str(e)
                print(f"[Agent2] API error on attempt {attempt + 1}: {e}")

                # If daily token quota exceeded, fall back to Llama 8B with reduced context
                if "token_quota_exceeded" in error_str or "tokens per day" in error_str.lower():
                    if not using_llama:
                        print("[Agent2] Token quota exceeded for Qwen — falling back to Llama 8B")
                        llm = llm_fast
                        using_llama = True
                        max_chunks = 7
                        system_prompt = REASONER_SYSTEM_PROMPT_LITE
                        chunks_to_use = _select_diverse_chunks(chunks_sorted, max_chunks)
                        messages, available_doc_ids = _build_messages(system_prompt, chunks_to_use)
                        continue  # Retry immediately with fast model + reduced context

                wait = min(2 ** (attempt + 1), 60)
                print(f"[Agent2] Retrying in {wait}s...")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    return {
                        "chain_of_thought": f"API error: {e}",
                        "answer": "An error occurred while generating the answer. Please try again.",
                        "citations": [],
                    }
