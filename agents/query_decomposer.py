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

load_dotenv()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

llm_fast = ChatCerebras(
    model="llama3.1-8b",
    api_key=CEREBRAS_API_KEY,
    temperature=0,
)

# ── Router prompt ────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a query complexity classifier. Given a user question about AI in Healthcare, classify it as:
- "simple": requires a single document lookup of a specific fact, statistic, name, or date
- "complex": requires synthesizing information from multiple documents, making comparisons, explaining mechanisms, or analyzing trends

When in doubt, classify as "complex".

Examples:
- "What was the FDA approval rate for AI devices in radiology?" → simple
- "Compare approaches of FDA vs EU MDR and discuss bias implications" → complex
- "What sensitivity did the diabetic retinopathy AI achieve?" → simple
- "How does AlphaFold affect drug discovery timelines and what are the ethical implications?" → complex
- "How many AI/ML-enabled medical devices has the FDA cleared?" → simple
- "What are the key challenges across clinical AI, regulation, and wearables?" → complex

Output JSON only, no other text:
{"route": "simple" or "complex", "reasoning": "one sentence explanation"}"""

# ── Decomposer prompt ────────────────────────────────────────────────────────

DECOMPOSER_SYSTEM_PROMPT = """You are a sub-query generator for a healthcare AI knowledge base containing 60 documents covering diverse medical domains.
Break the user's complex question into 3-6 targeted sub-queries.

CRITICAL RULES:
1. Each sub-query MUST target a DIFFERENT medical domain, specialty, or application area.
2. Sub-queries must be specific and independently searchable — concise, < 15 words each.
3. Do NOT create multiple sub-queries about the same topic — maximize diversity.
4. Think about different areas: radiology, surgery, drug discovery, mental health, neonatal care, oncology, cardiology, ophthalmology, infection control, wearables, genomics, ethics, regulation, etc.
5. If the question asks about "most impactful", "across all", or "top examples", generate sub-queries targeting SPECIFIC measured outcomes in distinct domains.

Examples:
Q: "What are the top AI applications that improved measurable clinical outcomes?"
A: {"sub_queries": ["AI chest X-ray triage mortality reduction", "AI sepsis prediction hospital outcomes mortality", "AI robotic surgery complication rates reduction", "AI mammography cancer detection improvement rates", "AI neonatal intensive care prediction outcomes", "AI mental health suicide risk prediction results"]}

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
            if result.get("route") not in ("simple", "complex"):
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

        # Stage B: Decompose (complex only)
        if route == "simple":
            sub_queries = [question]
        else:
            sub_queries = self._decompose(question, conversation_history)
            print(f"[Agent1] Sub-queries: {sub_queries}")

        # Stage C: Retrieve for each sub-query
        all_chunks: list[dict] = []
        seen_chunk_keys: set[str] = set()

        # For complex queries, also retrieve using the original question
        queries_to_search = list(sub_queries)
        if route == "complex" and question not in queries_to_search:
            queries_to_search.append(question)

        for sq in queries_to_search:
            # More chunks for complex, fewer for simple
            top_k = 3 if route == "simple" else 7
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
