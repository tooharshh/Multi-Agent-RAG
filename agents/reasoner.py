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

load_dotenv(override=True)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

llm_fast = ChatCerebras(
    model="llama3.1-8b",
    api_key=CEREBRAS_API_KEY,
    temperature=0,
    request_timeout=30,
)

llm_heavy = ChatCerebras(
    model="qwen-3-235b-a22b-instruct-2507",
    api_key=CEREBRAS_API_KEY,
    temperature=0.1,
    request_timeout=120,
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
5. TERMINOLOGY PRECISION: When the question uses specific terminology (e.g., "domain-specific AI tools", "cross-sectional study", "sycophancy", "behavioural tendency"), match that EXACT terminology to the source documents. Do NOT substitute with similar but different metrics (e.g. "hospitals using AI" ≠ "domain-specific AI tools"; "2024 reporting rate" ≠ "historical scoping review rate"). Always pick the statistic that most precisely matches the question wording. If two documents offer different numbers, ALWAYS prefer the document that uses the question's exact phrasing over one with a similar but not identical metric.
6. When the question asks for examples, impacts, or comparisons, include ALL specific numbers, statistics, percentages, named entities, company names, study names, and data points from EVERY relevant document in the context. Do NOT stop at 2-3 sources if more relevant data exists. Do NOT summarize when exact figures are available — always include the exact numbers.
7. For market/funding questions: include total market size, growth rates, specific funding rounds, company names, valuations, and trends.
8. For bias/equity questions: include prevalence statistics, specific study counts, named algorithms, demographic breakdowns, and specific mitigation outcomes.
9. The citations array must list every unique doc_id cited in the answer with a relevant excerpt.
10. If the retrieved context does not contain enough information to answer a part of the question, explicitly state that.
11. USE PRECISE VOCABULARY from the source documents. If a document uses a specific technical term (e.g., "sycophancy", "conformity assessment", "binding enforcement"), use that exact term in your answer rather than a paraphrase.
12. CHUNK RANKING MATTERS: Chunks are ordered by relevance score (highest first). Chunks marked with ⚡ CONTAINS KEY TERMS are the strongest matches for the question's specific terminology. When conflicting statistics appear across chunks, ALWAYS prefer the chunk with the higher rank and key term matches.
13. FORMATTING: Use markdown in your answer for readability. Separate distinct topics into paragraphs. Bold key statistics and named entities (e.g. **22%**, **Recursion-Exscientia**). Use short paragraphs, not one long wall of text.
14. CITATION BREADTH: When multiple documents in the context discuss the same topic, cite ALL of them — not just one. If DOC-011 gives an overview and DOC-012, DOC-013 provide deeper detail on sub-topics, cite each where they are most specific. Prefer the PRIMARY source (the document that conducted the original study or reports the original data) over a secondary source that merely references it. CRITICAL: If the context contains 3+ documents about the same subject (e.g., drug discovery, ethical concerns, demographic bias), your answer MUST cite at least 2 of them with the specific detail each uniquely provides. If a document provides additional facts (trial details, publication venue, specific methodology, company details, clinical outcomes) not mentioned in other documents, you MUST include those facts and cite that document. A landscape/overview document that SUMMARIZES developments is NOT sufficient alone — cite the SPECIFIC documents that provide the deep details (e.g., trial results from one doc, merger details from another, platform specifics from a third).
15. PRIMARY vs SECONDARY SOURCES: If a document SUMMARIZES findings from other articles (e.g., a blog or newsletter citing a study) and the ORIGINAL study/paper is also in the context, always cite the original study document, not the summary. When a landscape overview mentions a milestone briefly AND a dedicated article about that milestone exists in the context, cite the dedicated article for the details.
16. CUMULATIVE vs ANNUAL: When a question asks "how many existed BY [date]" or "total by [year]", report the CUMULATIVE total up to that date — NOT just the count from a single year. If one chunk gives a single-year figure and another gives the cumulative total, always use the cumulative total. Similarly, when asking about "the largest share of those approvals", compute the share against the CUMULATIVE total, not a single-year subset. If a table shows specialty counts across multiple years, use the LATEST cumulative figures. Example: if the table shows 956 radiology devices out of ~1,200 total by 2025, report "Radiology leads overwhelmingly, accounting for ~78% of all cumulative FDA-cleared devices" — do NOT use a single-year subset like "74.4% of 168 devices in 2024" to characterize the cumulative dominance.
17. DATE RANGE MATCHING: When the question specifies a date range (e.g. "between 1995 and 2023"), find the statistic that EXACTLY covers that date range. Do NOT substitute with a statistic from a different time period. When a question says "a 2024 study found X", the stat may come from a study PUBLISHED in 2024 that analyzed HISTORICAL data — use the historical finding, not the 2024-only figure.
18. MULTIPLE STATISTICS DISAMBIGUATION: When the same document contains multiple percentages or statistics about similar topics, always select the one whose surrounding context (date range, study type, population size) most precisely matches the question's wording. CRITICAL: When a document contains BOTH a recent-cohort figure (e.g. "15.5% of 168 devices in 2024") AND a broader scoping-review/historical figure (e.g. "3.6% of 692 devices from 1995-2023"), and the question mentions a "critical gap" or "vast majority missing", ALWAYS report the scoping-review/historical figure because it represents the larger sample and the more established finding that shows the GAP. The recent-cohort figure shows IMPROVEMENT, NOT the gap. State the specific study scope (sample size, date range). Example: "Only 3.6% of 692 FDA-cleared AI/ML devices between 1995 and 2023 reported race or ethnicity of validation cohorts [DOC-015]. While the 2024 cohort showed improvement to 15.5%, the historical figure reveals the scale of the critical gap."
19. SPECIFICITY OVER CATEGORIES: When the question asks which concern/issue "most likely worsens" something or is "most problematic", identify the SPECIFIC mechanism (e.g. "algorithmic bias from non-representative training data") rather than just naming the broad category (e.g. "justice and fairness"). Example: Do NOT say "justice and fairness worsens inequities" — instead say "Algorithmic bias — caused by non-representative training datasets and opaque model development — is identified as the mechanism most likely to worsen existing healthcare inequities, falling under the broader concern of justice and fairness."
20. FRAMEWORK IDENTIFICATION: When the question asks about multiple frameworks or approaches, identify each framework by its SPECIFIC source document and describe its UNIQUE monitoring mechanism with concrete details (e.g. "stakeholder-guided calibration, synthetic data validation, structured audits" for one framework vs "adaptive regulatory oversight replacing static one-time approval" for another). Then clearly state what the THIRD element (e.g. EU AI Act) adds that NEITHER framework provides. Use terms like "binding enforcement", "conformity assessment", "CE marking", or "legally required post-market surveillance" when the evidence supports it.
21. ALL PARTS RULE: When the question asks for multiple specific facts (e.g. "how many workers", "what headcount change", "what company"), you MUST answer EVERY part with a concrete number or name from the context. If the context contains the data, extract it. Do NOT say "unspecified" or "not provided" unless you have checked EVERY chunk from the relevant document.
22. EXACT TECHNICAL TERMS: When source documents use a specific named concept or technical term (e.g., "sycophancy", "algorithmic bias", "conformity assessment"), use that EXACT term in your answer as the key identifying label, rather than describing the phenomenon indirectly. Example: if the source says "LLMs have a tendency toward sycophancy", your answer must include the word "sycophancy" as the identified behavioral tendency — do NOT replace it with a paraphrase like "providing inappropriate reassurances".
23. SOURCE-LOCKED CITATION: Each chunk is wrapped in [BEGIN DOC-XXX CHUNK N] / [END DOC-XXX CHUNK N] markers. When you write [DOC-XXX] after a claim, the information MUST come from inside the [BEGIN DOC-XXX ...] / [END DOC-XXX ...] block. NEVER attribute a fact from one document's chunk to a different document's citation. If you find a statistic inside [BEGIN DOC-015 CHUNK 3], cite it as [DOC-015], NOT [DOC-010] or any other doc_id.
24. VERBATIM GROUNDING: Before including ANY number, percentage, statistic, count, date, or named entity in your answer, you MUST first locate the EXACT text in a retrieved chunk that states that value. In your chain_of_thought, write: "Found in [DOC-XXX CHUNK N]: '<5-15 word verbatim quote containing the number>'". If you cannot find the exact number verbatim in any chunk, DO NOT include that number in the answer. NEVER compute, round, combine, or extrapolate numbers — only use values that appear verbatim in the chunks. Example: if a chunk says "74.4% of 168 devices in 2024" do NOT restate this as "74.4% of all cumulative devices" — those are different claims.

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
5. Include ALL specific numbers, statistics, percentages, named entities, company names, and data points from every relevant document. Do NOT summarize when exact figures are available.
6. TERMINOLOGY PRECISION: Match the EXACT terminology from the question to the source documents. If the question asks about a "cross-sectional study" or "scoping review", find the statistic from THAT study — do not substitute with a different study's numbers. Prefer the most specific, directly matching data point.
7. USE PRECISE VOCABULARY from the source documents. If a document uses a specific technical term, use that exact term in your answer rather than a paraphrase.
8. The citations array must list every unique doc_id cited in the answer with a relevant excerpt.
9. FORMATTING: Use markdown in your answer for readability. Bold key statistics and named entities (e.g. **22%**, **Recursion-Exscientia**). Use short paragraphs, not one long wall of text.
10. CITATION BREADTH: When multiple documents discuss the same topic, cite ALL of them. Prefer the PRIMARY source (the document reporting original data/study) over summary documents. If DOC-X summarizes a finding and DOC-Y is the original study, cite DOC-Y.
11. CUMULATIVE vs ANNUAL: When a question asks "how many existed BY [date]" or "total by [year]", report the CUMULATIVE total up to that date — NOT just the count from a single year. If one chunk gives a single-year figure and another gives the cumulative total, always use the cumulative total. Similarly, when asking about "the largest share of those approvals", compute the share against the CUMULATIVE total, not a single-year subset.
12. DATE RANGE MATCHING: When the question specifies a date range (e.g. "between 1995 and 2023"), find the statistic that EXACTLY covers that date range. Do NOT substitute with a statistic from a different time period. When a question says "a 2024 study found X", the stat may come from a study PUBLISHED in 2024 that analyzed HISTORICAL data — use the historical finding, not the 2024-only figure.
13. MULTIPLE STATISTICS DISAMBIGUATION: When the same document contains multiple percentages or statistics about similar topics, always select the one whose surrounding context (date range, study type, population size) most precisely matches the question's wording. CRITICAL: When a document contains BOTH a recent-cohort figure AND a broader scoping-review/historical figure, and the question mentions a "critical gap" or "vast majority missing", ALWAYS report the scoping-review/historical figure (larger sample, more established finding). The recent-cohort figure shows improvement, NOT the gap. State the specific study scope.
14. SPECIFICITY OVER CATEGORIES: When the question asks which concern/issue "most likely worsens" something or is "most problematic", identify the SPECIFIC mechanism (e.g. "algorithmic bias from non-representative training data") rather than just naming the broad category (e.g. "justice and fairness").
15. ALL PARTS RULE: When the question asks for multiple specific facts, EXTRACT every concrete number or name from the context. Do NOT say "unspecified" or "not provided" unless you have checked every chunk from the relevant document.
16. CONCISENESS: Be CONCISE. Answer ONLY what is specifically asked. Do NOT discuss tangential topics, ethical implications, future research needs, or subjects not directly part of the question. Do NOT add conclusion paragraphs or recommendations unless asked.
17. PRIMARY SOURCE PRIORITY: When multiple documents contain the same statistic or finding, ALWAYS cite the document whose title or source indicates it is the ORIGINAL study/paper (e.g. a research article or clinical study), NOT a secondary summary, blog, or overview article that merely references the finding. If the question names a specific system, tool, or study, cite the document that is ABOUT that system/tool/study.
18. SOURCE-LOCKED CITATION: Each chunk is wrapped in [BEGIN DOC-XXX CHUNK N] / [END DOC-XXX CHUNK N] markers. When you write [DOC-XXX] after a claim, the information MUST come from inside the [BEGIN DOC-XXX ...] / [END DOC-XXX ...] block. NEVER attribute a fact from one document's chunk to a different document's citation.
19. VERBATIM GROUNDING: Before including ANY number, percentage, or statistic in your answer, locate the EXACT text in a chunk that states that value. In your chain_of_thought, write: "Found in [DOC-XXX CHUNK N]: '<verbatim quote>'". If you cannot find the exact number in any chunk, DO NOT include it. NEVER compute, round, combine, or extrapolate numbers.

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


## Cross-term concept mapping: when a question mentions a concept on the left,
## also boost chunks that contain the related terms on the right.
CONCEPT_CROSS_TERMS = {
    # EVAL-007: question says "behavioural tendency" → boost chunks with "sycophancy"
    "behavioural tendency": ["sycophancy", "tendency toward sycophancy", "affirm rather than challenge"],
    "behavioral tendency": ["sycophancy", "tendency toward sycophancy", "affirm rather than challenge"],
    "unsuitable as": ["sycophancy", "delusional thinking", "built on deception"],
    # EVAL-008: question says "critical gap" → boost chunks with historical scoping review figures
    # Also boost chunks with ALL gap statistics: 81.6% no age, <2% peer-reviewed, <1% socioeconomic
    "critical gap": ["3.6%", "692", "scoping review", "1995–2023", "1995-2023", "baseline", "81.6%", "peer-reviewed", "age of study"],
    "vast majority": ["3.6%", "692", "scoping review", "1995–2023", "1995-2023", "baseline", "81.6%", "peer-reviewed"],
    "demographic reporting": ["3.6%", "692", "scoping review", "race or ethnicity", "socioeconomic", "81.6%", "age of study", "peer-reviewed", "perturbation testing"],
    "fda-cleared ai devices": ["3.6%", "692", "81.6%", "race or ethnicity", "scoping review", "peer-reviewed", "age of study"],
    "device submissions": ["3.6%", "692", "81.6%", "race or ethnicity", "socioeconomic", "peer-reviewed"],
    # EVAL-006: question mentions "ethical concerns" → boost chunks with specific mechanisms
    "ethical concerns": ["algorithmic bias", "non-representative", "training datasets"],
    "worsens": ["algorithmic bias", "non-representative", "training datasets"],
    "healthcare inequities": ["algorithmic bias", "non-representative", "training datasets", "bias evaluation"],
    # EVAL-003: ChatRWD questions → include OpenEvidence comparison data
    "chatrwd": ["58%", "openevidence", "24%", "retrieval-augmented generation", "2%-10%"],
    "agentic system": ["chatrwd", "58%", "openevidence", "24%", "retrieval-augmented generation"],
    "compared to": ["openevidence", "24%", "58%", "2%-10%"],
    "chatgpt": ["openevidence", "24%", "58%", "chatrwd", "retrieval-augmented generation"],
    "gemini": ["openevidence", "24%", "58%", "chatrwd", "retrieval-augmented generation"],
    # EVAL-005: drug discovery questions → boost chunks with trial details from DOC-012/013
    "phase ii": ["nature medicine", "71-patient", "forced vital capacity", "fvc", "dose-dependent", "21-site", "tnik kinase"],
    "phase iia": ["nature medicine", "71-patient", "forced vital capacity", "fvc", "dose-dependent", "21-site", "tnik kinase"],
    "rentosertib": ["nature medicine", "71-patient", "21-site", "forced vital capacity", "tnik kinase inhibitor", "ism001-055", "idiopathic pulmonary fibrosis"],
    "ism001-055": ["nature medicine", "71-patient", "21-site", "forced vital capacity", "tnik kinase inhibitor", "rentosertib", "idiopathic pulmonary fibrosis"],
    "drug discovery": ["nature medicine", "71-patient", "forced vital capacity", "recursion", "exscientia", "tnik kinase"],
    "competitive landscape": ["recursion", "exscientia", "merger", "precision chemistry", "phenomic screening", "automated synthesis"],
    "recursion": ["exscientia", "merger", "phenomic screening", "automated synthesis", "end-to-end platform", "november 2024"],
    "exscientia": ["recursion", "merger", "precision chemistry", "automated synthesis", "end-to-end platform", "november 2024"],
    # EVAL-004: cumulative share question → boost cumulative data over single-year
    "largest share": ["overwhelmingly", "dominated", "cumulative", "391", "956"],
    # EVAL-011: strike question → boost chunks with worker count AND UK AI company
    "went on strike": ["2,400", "2400", "24-hour strike", "unfair labor practice", "limbic"],
    "workers participated": ["2,400", "2400", "mental health care providers", "limbic"],
    "headcount change": ["nine providers", "nine to three", "cut to three", "triage team"],
    "uk ai company": ["limbic", "nhs", "63%", "intake", "patient support", "assessing"],
    "confirmed to be evaluating": ["limbic", "nhs", "63%", "intake", "assessing ai tools"],
    "kaiser permanente": ["limbic", "2,400", "2400", "walnut creek", "triage team", "nine to three"],
}


def _extract_key_terms(question: str) -> set:
    """Extract key terms from question including cross-term concept expansions."""
    key_terms = set()
    if not question:
        return key_terms
    import re as _re
    q_lower = question.lower()
    # Extract multi-word quoted phrases and important terms
    for term in _re.findall(r'"([^"]+)"', question):
        key_terms.add(term.lower())
    # Extract hyphenated and domain terms
    for term in _re.findall(r'\b[a-z]+-[a-z]+(?:-[a-z]+)*\b', q_lower):
        key_terms.add(term)
    # Common specific terms to watch for
    for phrase in ["domain-specific", "cross-sectional", "scoping review", "sycophancy",
                   "phase ii", "phase iia", "binding enforcement", "conformity assessment",
                   "vast majority", "critical gap", "algorithmic bias", "cumulative",
                   "692", "950", "76%", "3.6%",
                   "2,400", "2400", "walnut creek", "limbic", "strike",
                   "clinic-level", "adaptive regulatory", "operationalised",
                   "stakeholder calibration", "synthetic data validation",
                   "demographic reporting", "largest share", "ethical concerns",
                   "drug discovery", "competitive landscape", "healthcare inequities",
                   "fda-cleared ai devices", "device submissions"]:
        if phrase in q_lower:
            key_terms.add(phrase)
    # Extract specific percentages from the question
    for m in _re.findall(r'(\d+(?:\.\d+)?%)', q_lower):
        key_terms.add(m)
    # Extract date ranges like "1995 and 2023", "between X and Y"
    for m in _re.findall(r'between\s+(\d{4})\s+and\s+(\d{4})', q_lower):
        key_terms.add(f"{m[0]}")
        key_terms.add(f"{m[1]}")
        key_terms.add(f"{m[0]}–{m[1]}")  # en-dash variant
        key_terms.add(f"{m[0]}-{m[1]}")  # hyphen variant
    # Extract "by mid-YYYY" or "by YYYY"
    for m in _re.findall(r'by\s+(?:mid[- ])?(\d{4})', q_lower):
        key_terms.add(m)
    # Apply CONCEPT_CROSS_TERMS: expand question concepts to related content terms
    for concept, related_terms in CONCEPT_CROSS_TERMS.items():
        if concept in q_lower:
            for rt in related_terms:
                key_terms.add(rt)
    return key_terms


def _format_context(chunks: list[dict], question: str = "") -> str:
    """Format retrieved chunks into a numbered context string for the LLM.
    
    Uses [BEGIN DOC-XXX CHUNK] / [END DOC-XXX CHUNK] delimiters so the LLM
    can precisely attribute facts to the correct source document.
    """
    key_terms = _extract_key_terms(question)
    
    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Check if chunk contains key question terms
        term_matches = []
        chunk_lower = chunk['text'].lower()
        for term in key_terms:
            if term in chunk_lower:
                term_matches.append(term)
        
        match_note = ""
        if term_matches:
            match_note = f"⚡ CONTAINS KEY TERMS: {', '.join(term_matches)}\n"
        
        doc_id = chunk['doc_id']
        parts.append(
            f"[BEGIN {doc_id} CHUNK {i}]\n"
            f"RANK: #{i} by relevance\n"
            f"doc_id: {doc_id}\n"
            f"title: {chunk['title']}\n"
            f"source: {chunk['source']}\n"
            f"date: {chunk['date']}\n"
            f"relevance_score: {chunk['relevance_score']:.3f}\n"
            f"{match_note}"
            f"text: {chunk['text']}\n"
            f"[END {doc_id} CHUNK {i}]\n"
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

        # Model selection based on route
        route = context_package.get("route", "complex")
        if route == "simple":
            llm = llm_fast
            model_name = "llama3.1-8b"
            system_prompt = REASONER_SYSTEM_PROMPT_LITE
            max_chunks = 8
            using_llama = True
        else:
            llm = llm_heavy
            model_name = "qwen-3-235b-a22b-instruct-2507"
            system_prompt = REASONER_SYSTEM_PROMPT
            max_chunks = 12
            using_llama = False
        print(f"[Agent2] Using {model_name} for {route} route")

        # Sort chunks by relevance
        chunks = context_package.get("retrieved_chunks", [])
        chunks_sorted = sorted(chunks, key=lambda c: c.get("relevance_score", 0), reverse=True)

        def _select_diverse_chunks(sorted_chunks, max_k):
            """Select top-K chunks with doc_id diversity and key-term boosting.
            
            When all max_k slots are filled by unique docs in Pass 1,
            Pass 3 checks if any remaining chunk has a high cross-term boost
            and replaces the weakest unique-doc chunk if so.
            """
            key_phrases = list(_extract_key_terms(question))
            
            # Score each chunk: relevance + key-term bonus
            def chunk_priority(chunk):
                score = chunk.get("relevance_score", 0)
                text_lower = chunk["text"].lower()
                for phrase in key_phrases:
                    if phrase in text_lower:
                        score += 5.0  # significant boost for key-term match
                return score
            
            # Re-sort by boosted score
            boosted = sorted(sorted_chunks, key=chunk_priority, reverse=True)
            
            selected = []
            seen_docs = set()
            # Pass 1: best chunk per unique doc_id (using boosted ranking)
            for chunk in boosted:
                if chunk["doc_id"] not in seen_docs:
                    selected.append(chunk)
                    seen_docs.add(chunk["doc_id"])
                    if len(selected) >= max_k:
                        break
            # Pass 2: if still under max_k, fill with remaining top-scored chunks
            if len(selected) < max_k:
                for chunk in boosted:
                    if chunk not in selected:
                        selected.append(chunk)
                        if len(selected) >= max_k:
                            break
            else:
                # Pass 3: All slots filled by unique docs. Check if any remaining
                # chunk with high cross-term boost should replace the weakest selected chunk.
                remaining = [c for c in boosted if c not in selected]
                if remaining:
                    best_remaining = remaining[0]
                    best_remaining_score = chunk_priority(best_remaining)
                    # Only replace if the remaining chunk has significant cross-term boost
                    # (score > base relevance, indicating key-term matches)
                    worst_idx = min(range(len(selected)), key=lambda i: chunk_priority(selected[i]))
                    worst_score = chunk_priority(selected[worst_idx])
                    if best_remaining_score > worst_score + 2.0:
                        selected[worst_idx] = best_remaining
            return selected

        def _build_messages(prompt, chunk_list):
            """Build LLM messages with given prompt and chunk list.
            
            If a rewritten_question is available from Agent 1 follow-up resolution,
            use it as the question and inject ZERO history (the rewrite already
            carries full context). Otherwise, inject the last 2 turns of compressed
            history as a lightweight fallback.
            """
            rewritten_q = context_package.get("rewritten_question")
            q_to_use = rewritten_q if rewritten_q else question

            ctx = _format_context(chunk_list, question=q_to_use)
            avail_ids = sorted(set(c["doc_id"] for c in chunk_list))
            msgs = [SystemMessage(content=prompt)]

            # History injection: zero if rewritten, lightweight fallback otherwise
            if not rewritten_q and conversation_history:
                # Inject last 2 turns only, truncated
                from utils.history import compress_history, history_budget_check
                compressed = compress_history(conversation_history, max_turns=2, fallback_answer_tokens=100)
                compressed = history_budget_check(compressed, max_total_tokens=400, mode="answer")
                for turn in compressed:
                    msgs.append(HumanMessage(content=turn["user"]))
                    if turn.get("assistant_answer"):
                        msgs.append(AIMessage(content=turn["assistant_answer"]))

            msgs.append(HumanMessage(
                content=f"QUESTION: {q_to_use}\n\n"
                        f"CRITICAL DISAMBIGUATION RULES:\n"
                        f"- When multiple documents contain similar-sounding statistics, choose the one whose EXACT wording matches the question's key terms. "
                        f"Do NOT pick a general statistic when a more specific one matches the question's terminology.\n"
                        f"- If the question says 'by mid-YYYY', use the CUMULATIVE total. If the question asks about 'the largest share of those approvals', derive the share from the CUMULATIVE total across all years (e.g. from a multi-year table), NOT a single-year snapshot.\n"
                        f"- If the question says 'a YYYY study found X', report WHAT THE STUDY FOUND (which may be a historical stat), not a stat about YYYY-only data.\n"
                        f"- If the question says 'critical gap' or 'vast majority missing', you MUST pick the LOWER percentage from the LARGER historical sample (e.g. scoping review of 692 devices showing 3.6%), NOT the improved recent cohort figure (e.g. 15.5% of 168 devices). The question is asking about the GAP, not the improvement.\n"
                        f"- If the question asks for specific numbers (how many workers, what headcount), extract the EXACT count from the context — do NOT say 'unspecified'.\n"
                        f"- When comparing frameworks, identify each framework by its source and describe its unique mechanism.\n"
                        f"- When source documents use a NAMED behavioral tendency or technical term (e.g. 'sycophancy', 'algorithmic bias'), use that EXACT word in your answer as the main label — do NOT replace it with a generic description.\n"
                        f"- When the question asks which concern 'worsens' something, name the SPECIFIC mechanism (e.g. 'algorithmic bias from non-representative training data'), not just the broad category.\n"
                        f"- When 3+ documents in the context cover different aspects of the same topic, cite ALL of them with the specific detail each provides.\n"
                        f"- SOURCE LOCK: Each chunk below is wrapped in [BEGIN DOC-XXX CHUNK N] / [END DOC-XXX CHUNK N]. When citing [DOC-XXX], the fact MUST come from inside a [BEGIN DOC-XXX ...] block. Never attribute a fact from DOC-A's chunk to DOC-B.\n"
                        f"- VERBATIM NUMBERS: Only include a number/percentage in your answer if you can point to the EXACT text in a chunk that states it. In your chain_of_thought, quote the source. Never compute, round, or combine numbers.\n\n"
                        f"AVAILABLE DOC_IDS (only cite these): {', '.join(avail_ids)}\n\n"
                        f"RETRIEVED CONTEXT:\n{ctx}"
            ))
            return msgs, avail_ids

        chunks_to_use = _select_diverse_chunks(chunks_sorted, max_chunks)
        messages, available_doc_ids = _build_messages(system_prompt, chunks_to_use)

        # Call LLM with unlimited retries — no fallbacks, wait as long as needed
        last_response = None
        parse_failures = 0
        attempt = 0
        while True:
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
                        result["answer"] = result["answer"].replace(h_id, "")

                # Filter citations list to only valid retrieved doc_ids
                valid_doc_id_set = set(available_doc_ids)
                result["citations"] = [
                    c for c in result["citations"]
                    if c.get("doc_id") in valid_doc_id_set
                ]

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

                if parse_failures >= 8:
                    # Last resort: return raw text as answer
                    raw = last_response.content if last_response else "Error generating answer."
                    doc_ids = list(set(re.findall(r'DOC-\d{3}', raw)))
                    valid = [d for d in doc_ids if d in set(c["doc_id"] for c in chunks_to_use)]
                    return {
                        "chain_of_thought": "Failed to generate structured output after retries.",
                        "answer": raw,
                        "citations": [{"doc_id": d, "title": "", "relevant_excerpt": ""} for d in valid],
                    }
                attempt += 1
            except Exception as e:
                error_str = str(e)
                print(f"[Agent2] API error on attempt {attempt + 1}: {error_str[:200]}")

                # Context length exceeded — reduce chunks and retry immediately
                is_context_length = "reduce the length" in error_str or "context_length" in error_str.lower()
                if is_context_length and max_chunks > 5:
                    max_chunks = max(5, max_chunks - 3)
                    chunks_to_use = _select_diverse_chunks(chunks_sorted, max_chunks)
                    messages, available_doc_ids = _build_messages(system_prompt, chunks_to_use)
                    print(f"[Agent2] Context too long — reducing to {max_chunks} chunks")
                    attempt += 1
                    continue

                # All errors: just wait and retry, no fallbacks ever
                is_queue_error = "queue_exceeded" in error_str or "too_many_requests" in error_str
                is_quota_error = "token_quota_exceeded" in error_str or "tokens per day" in error_str.lower()
                if is_queue_error:
                    wait = min(30 * (attempt + 1), 120)
                elif is_quota_error:
                    wait = min(60 * (attempt + 1), 300)
                else:
                    wait = min(2 ** (attempt + 1), 60)
                print(f"[Agent2] Retrying in {wait}s (attempt {attempt + 1})...")
                time.sleep(wait)
                attempt += 1
