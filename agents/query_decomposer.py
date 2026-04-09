"""
Agent 1 — Query Decomposer & Retriever.
Routes queries as simple/complex, decomposes complex queries, retrieves context.
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

# ── Router prompt ────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a query classifier for an AI-in-Healthcare knowledge base. First determine if the input is a genuine question about AI in healthcare. Then classify its complexity.

Routes:
- "off_topic": greetings, small talk, or anything not related to AI in healthcare (e.g. "hi", "hello", "thanks", "what's the weather", "tell me a joke")
- "simple": requires looking up 1-2 closely related facts that would appear in the SAME document or study — even if the question has two parts connected by "and". Factual retrieval, specific statistics, single-study results, single-survey results.
- "complex": requires synthesizing information from DIFFERENT topic areas or 3+ unrelated documents, constructing causal mechanisms, building arguments across separate studies, or comparing fundamentally different frameworks/approaches. Also classify as "complex" if the question mentions a "critical gap", "historical review", "scoping review", or requires disambiguating between multiple statistics from different time periods within the same study.

KEY RULE: A question asking for two facts from the SAME source (e.g. a single survey, study, or report) is "simple", even if it uses "and". A question requiring cross-document synthesis across DIFFERENT topics is "complex".

When in doubt between simple and complex, ALWAYS classify as "complex".

Examples:
- "hi" → off_topic
- "hello, how are you?" → off_topic
- "thanks!" → off_topic
- "What was the FDA approval rate for AI devices in radiology?" → simple
- "What percentage of organisations had implemented X, and how does this compare to Y?" → simple (same report/survey has both stats)
- "In a survey, which AI use case had 100% adoption and what success rate?" → simple (same survey)
- "How many FDA-cleared AI devices existed by mid-2024, and which specialty has the most?" → simple (same dataset)
- "What share of clinical questions did ChatRWD answer compared to ChatGPT?" → simple (same study)
- "A study found a gap in demographic reporting. What was missing and what percentage?" → complex (statistical disambiguation within a study)
- "A cross-sectional study found a critical gap in demographic data. What was missing?" → complex (requires distinguishing historical vs recent stats)
- "What sensitivity did the diabetic retinopathy AI achieve?" → simple
- "Compare approaches of FDA vs EU MDR and discuss bias implications" → complex (different regulatory frameworks + bias literature)
- "How does AlphaFold affect drug discovery timelines and what are the ethical implications?" → complex (drug discovery + ethics = different topics)
- "What are the key challenges across clinical AI, regulation, and wearables?" → complex (3 different areas)
- "Construct the causal mechanism for divergence between deployment and success rates" → complex (requires inference across multiple sources)
- "Two frameworks propose different monitoring mechanisms. What does the EU AI Act add?" → complex (3 different frameworks need synthesis)
- "What Phase II milestone did X reach, and what merger reshaped the landscape?" → complex (clinical trial + M&A = different topics)
- "A paper identifies five ethical concerns. What are they and which worsens inequities?" → complex (enumeration + analysis across sources)

Output JSON only, no other text:
{"route": "simple" or "complex" or "off_topic", "reasoning": "one sentence explanation"}"""

# ── Decomposer prompt ────────────────────────────────────────────────────────

DECOMPOSER_SYSTEM_PROMPT = """You are a sub-query generator for a healthcare AI knowledge base containing 21 articles covering market/adoption, clinical AI, drug discovery, regulation/ethics, and breaking news.
Break the user's complex question into 2-4 targeted sub-queries.

CRITICAL RULES:
1. Each sub-query MUST target a DIFFERENT medical domain, specialty, or application area.
2. Sub-queries must be specific and independently searchable — concise, < 15 words each.
3. Do NOT create multiple sub-queries about the same topic — maximize diversity.
4. Think about different areas: radiology, surgery, drug discovery, mental health, neonatal care, oncology, cardiology, ophthalmology, infection control, wearables, genomics, ethics, regulation, etc.
5. If the question asks about "most impactful", "across all", or "top examples", generate sub-queries targeting SPECIFIC measured outcomes in distinct domains.
6. PRESERVE SOURCE REFERENCES: When the question mentions a specific paper, study, journal, or organization (e.g. "A 2025 PLOS Digital Health paper", "a 2024 cross-sectional study", "The Hastings Center"), include that reference in at least ONE sub-query verbatim.
7. PRESERVE DATES AND RANGES: If the question mentions specific dates or date ranges (e.g. "between 1995 and 2023", "by mid-2024"), include them in the relevant sub-query.
8. PRESERVE ENTITY NAMES: If the question mentions specific drugs, companies, or named systems (e.g. "ISM001-055", "Rentosertib", "ChatRWD", "EU AI Act"), keep them in sub-queries.
9. FULL COVERAGE: When a question describes N distinct topics, frameworks, or entities (e.g. "Framework A does X, Framework B does Y, what does C add?"), generate at least ONE sub-query for EACH distinct topic. Do NOT generate multiple overlapping sub-queries about only one of them.
10. STUDY METHODOLOGY QUERIES: When a question references a study's findings (e.g. "a cross-sectional study found a gap"), generate one sub-query for the study's specific findings AND one for the evaluation methodology or framework used to assess those findings.

Examples:
Q: "What are the top AI applications that improved measurable clinical outcomes?"
A: {"sub_queries": ["AI chest X-ray triage mortality reduction", "AI sepsis prediction hospital outcomes mortality", "AI robotic surgery complication rates reduction", "AI mammography cancer detection improvement rates", "AI neonatal intensive care prediction outcomes", "AI mental health suicide risk prediction results"]}

Q: "Two frameworks for governing post-deployment AI propose different monitoring mechanisms. One focuses on clinic-level audit steps; the other calls for adaptive regulatory oversight. What does the EU AI Act add?"
A: {"sub_queries": ["operationalised clinic-level AI audit stakeholder calibration bias evaluation framework", "adaptive regulatory oversight replacing static AI approval continuous monitoring", "EU AI Act high-risk classification binding enforcement conformity assessment"]}

Q: "A 2024 cross-sectional study found a critical gap in demographic reporting for FDA devices. What data was missing and what percentage reported it?"
A: {"sub_queries": ["FDA-cleared AI devices demographic reporting race ethnicity validation cohorts scoping review", "bias evaluation framework methodology for clinical AI devices healthcare settings"]}

Q: "How are algorithms being corrected for healthcare disparities and bias?"
A: {"sub_queries": ["AI dermatology skin tone diversity dataset equity", "genomic risk prediction underrepresented populations accuracy", "social determinants of health AI prediction models", "algorithmic bias audit strategies clinical AI evidence", "AI clinical tools racial gender bias mitigation"]}

Q: "What is the total healthcare AI market and how is funding distributed?"
A: {"sub_queries": ["global healthcare AI market size projections 2030", "venture capital funding healthcare AI categories distribution", "largest healthcare AI startup funding rounds 2025", "AI drug discovery investment cumulative funding"]}

Output JSON only, no other text:
{"sub_queries": ["sub-query 1", "sub-query 2", ...]}"""


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in LLM response: {text[:200]}")


class QueryDecomposerAgent:
    """
    Agent 1: Classifies query complexity, decomposes if complex,
    retrieves relevant chunks via HybridRetriever.
    """

    def __init__(self, retriever):
        """
        Args:
            retriever: HybridRetriever instance with .retrieve(query, top_k) method
        """
        self.retriever = retriever
        self.llm = llm_fast

    def _classify(self, question: str, conversation_history: list) -> dict:
        """Classify query as simple or complex using Llama 8B."""
        messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]

        # Include conversation context for follow-up awareness
        for turn in conversation_history:
            messages.append(HumanMessage(content=turn["user"]))

        messages.append(HumanMessage(content=question))

        response = self.llm.invoke(messages)
        try:
            result = _parse_json_response(response.content)
            if result.get("route") not in ("simple", "complex", "off_topic"):
                result["route"] = "complex"  # default to complex when unclear
            return result
        except (json.JSONDecodeError, ValueError):
            return {"route": "complex", "reasoning": "Failed to parse — defaulting to complex"}

    def _decompose(self, question: str, conversation_history: list) -> list[str]:
        """Break a complex question into 2-4 sub-queries using Llama 8B."""
        messages = [SystemMessage(content=DECOMPOSER_SYSTEM_PROMPT)]

        # Context from conversation
        for turn in conversation_history:
            messages.append(HumanMessage(content=turn["user"]))

        messages.append(HumanMessage(content=question))

        response = self.llm.invoke(messages)
        try:
            result = _parse_json_response(response.content)
            sub_queries = result.get("sub_queries", [question])
            if not sub_queries:
                return [question]
            return sub_queries[:6]  # Cap at 6 for broader coverage
        except (json.JSONDecodeError, ValueError):
            return [question]  # Fallback: use original question

    def run(self, question: str, conversation_history: list | None = None) -> dict:
        """
        Full Agent 1 pipeline:
        1. Classify query complexity
        2. Decompose if complex
        3. Retrieve chunks for each sub-query
        4. Deduplicate and return context package

        Returns:
            {
                "route": "simple" | "complex",
                "sub_queries": [...],
                "retrieved_chunks": [...],
                "source_doc_ids": [...]
            }
        """
        if conversation_history is None:
            conversation_history = []

        # Stage A: Route
        classification = self._classify(question, conversation_history)
        route = classification["route"]
        print(f"[Agent1] Route: {route} — {classification.get('reasoning', '')}")

        # Off-topic guard: greetings and non-healthcare queries skip retrieval
        if route == "off_topic":
            print("[Agent1] Off-topic query detected — skipping retrieval.")
            return {
                "route": "off_topic",
                "sub_queries": [],
                "retrieved_chunks": [],
                "source_doc_ids": [],
            }

        # Stage B: Decompose complex queries; simple queries use single-hop retrieval
        if route == "simple":
            sub_queries = [question]
            print(f"[Agent1] Simple route — single-hop retrieval (no decomposition)")
        else:
            sub_queries = self._decompose(question, conversation_history)
            print(f"[Agent1] Sub-queries: {sub_queries}")

        # Stage C: Retrieve for each sub-query
        all_chunks: list[dict] = []
        seen_chunk_keys: set[str] = set()

        # Simple route: single retrieval with top_k=15 (broad coverage)
        # Complex route: multi-query retrieval with top_k=15 each
        queries_to_search = list(sub_queries)
        if route == "complex" and question not in queries_to_search:
            queries_to_search.append(question)

        for sq in queries_to_search:
            top_k = 15 if route == "simple" else 15
            chunks = self.retriever.retrieve(sq, top_k=top_k)
            for chunk in chunks:
                key = f"{chunk['doc_id']}_{chunk['chunk_index']}"
                if key not in seen_chunk_keys:
                    seen_chunk_keys.add(key)
                    all_chunks.append(chunk)

        # Deduplicated source doc IDs
        source_doc_ids = sorted(set(c["doc_id"] for c in all_chunks))

        print(f"[Agent1] Retrieved {len(all_chunks)} unique chunks from {len(source_doc_ids)} documents: {source_doc_ids}")

        return {
            "route": route,
            "sub_queries": sub_queries,
            "retrieved_chunks": all_chunks,
            "source_doc_ids": source_doc_ids,
        }
