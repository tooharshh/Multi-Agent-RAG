"""
Agent 1 — Query Decomposer & Retriever.
Routes queries as simple/complex, decomposes complex queries, retrieves context.
Handles follow-up detection, query rewriting, clarification, and meta routes.
"""

import json
import os
import re
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_core.messages import SystemMessage, HumanMessage

from utils.history import compress_history, history_budget_check

load_dotenv(override=True)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

llm_fast = ChatCerebras(
    model="llama3.1-8b",
    api_key=CEREBRAS_API_KEY,
    temperature=0,
)

# ── Follow-up detection prompt ───────────────────────────────────────────────

FOLLOW_UP_DETECTOR_PROMPT = """You are a follow-up question classifier. Given a question and recent conversation context, classify it.

Types:
- "standalone": A self-contained question with no references to prior conversation. Includes well-formed healthcare AI questions that happen to come after other questions but do not reference them.
- "follow_up": References something from prior conversation (pronouns like "it/that/they", "that study", "tell me more", "what about X", "continue", "the other one", "expand on"). CAN be resolved from the provided conversation context.
- "follow_up_doc": Explicitly references a DOC-XXX identifier (e.g. "tell me more about DOC-015").
- "ambiguous": References prior conversation BUT the referent is unclear even with the context provided.
- "meta": Not a knowledge question — asks about the system, the answer format, why something was cited, or requests a summary of the conversation itself (e.g. "why did you cite that?", "summarize our discussion").

Output JSON only:
{"type": "standalone" | "follow_up" | "follow_up_doc" | "ambiguous" | "meta", "reasoning": "one sentence"}"""

# ── Query rewrite prompt ─────────────────────────────────────────────────────

REWRITE_SYSTEM_PROMPT = """You are a query rewriter for a healthcare AI knowledge base.
The user asked a follow-up question that references prior conversation.
Rewrite it into a fully self-contained, specific question.

RULES:
1. The rewritten question must be understandable WITHOUT any conversation context.
2. Preserve the user's intent exactly — do not add topics they didn't ask about.
3. If the user references a specific study, statistic, or concept from the conversation, name it explicitly in the rewrite.
4. If the user says "tell me more", "continue", "go on", or "what else", rewrite as "Provide additional details about [MAIN TOPIC from last assistant answer]".
5. Keep the rewritten question under 40 words.
6. Do NOT invent facts — only reference entities that appear in the conversation context provided.

Output JSON only:
{"rewritten_question": "the fully self-contained question"}"""

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
    Agent 1: Detects follow-ups, rewrites queries, classifies complexity,
    decomposes if complex, retrieves relevant chunks via HybridRetriever.
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = llm_fast

    # ── Regex fast-paths for follow-up detection ─────────────────────────

    _CONTINUE_RE = re.compile(
        r"^\s*(continue|go on|what else|more|keep going|tell me more|expand on that)\s*[?.!]*\s*$",
        re.IGNORECASE,
    )
    _DOC_REF_RE = re.compile(r"DOC-\d{3}", re.IGNORECASE)

    def _detect_follow_up(self, question: str, compressed_history: list[dict]) -> dict:
        """
        Step 0: Classify whether the question is standalone, a follow-up, ambiguous, or meta.
        Uses regex fast-paths before falling back to LLM.
        """
        # Fast-path: no history → standalone by definition
        if not compressed_history:
            return {"type": "standalone", "reasoning": "No conversation history"}

        # Fast-path: explicit DOC-XXX reference
        doc_match = self._DOC_REF_RE.search(question)
        if doc_match:
            return {"type": "follow_up_doc", "reasoning": f"Explicit reference to {doc_match.group()}"}

        # Fast-path: bare continuation phrases
        if self._CONTINUE_RE.match(question):
            return {"type": "follow_up", "reasoning": "Continuation request detected"}

        # LLM classification with compressed context
        context_block = self._format_history_for_prompt(compressed_history)
        messages = [
            SystemMessage(content=FOLLOW_UP_DETECTOR_PROMPT),
            HumanMessage(
                content=f"CONVERSATION CONTEXT:\n{context_block}\n\nCURRENT QUESTION: {question}"
            ),
        ]

        try:
            response = self.llm.invoke(messages)
            result = _parse_json_response(response.content)
            ftype = result.get("type", "standalone")
            if ftype not in ("standalone", "follow_up", "follow_up_doc", "ambiguous", "meta"):
                ftype = "standalone"
            return {"type": ftype, "reasoning": result.get("reasoning", "")}
        except Exception:
            return {"type": "standalone", "reasoning": "Detection failed — defaulting to standalone"}

    def _rewrite_question(self, question: str, compressed_history: list[dict]) -> str | None:
        """
        Step 1: Rewrite a follow-up question into a self-contained standalone query.
        Returns the rewritten question string, or None if rewrite fails.
        """
        context_block = self._format_history_for_prompt(compressed_history)
        messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(
                content=f"CONVERSATION CONTEXT:\n{context_block}\n\nFOLLOW-UP QUESTION: {question}"
            ),
        ]

        try:
            response = self.llm.invoke(messages)
            result = _parse_json_response(response.content)
            rewritten = result.get("rewritten_question", "").strip()
            if rewritten and len(rewritten) > 5:
                return rewritten
        except Exception:
            pass
        return None

    def _format_history_for_prompt(self, compressed_history: list[dict]) -> str:
        """Format compressed history into a compact text block for LLM prompts."""
        lines = []
        for i, turn in enumerate(compressed_history):
            lines.append(f"Turn {i+1} — User: {turn['user']}")
            if turn.get("topic_line"):
                lines.append(f"  Assistant topics: {turn['topic_line']}")
        return "\n".join(lines)

    def _classify(self, question: str, compressed_history: list[dict]) -> dict:
        """Classify query as simple, complex, or off_topic using Llama 8B."""
        messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]

        # Include compressed conversation context (user + assistant topics)
        for turn in compressed_history:
            messages.append(HumanMessage(content=turn["user"]))
            if turn.get("topic_line"):
                # Inject topic-line as a lightweight assistant summary
                messages.append(HumanMessage(content=f"[Prior answer covered: {turn['topic_line']}]"))

        messages.append(HumanMessage(content=question))

        response = self.llm.invoke(messages)
        try:
            result = _parse_json_response(response.content)
            if result.get("route") not in ("simple", "complex", "off_topic"):
                result["route"] = "complex"  # default to complex when unclear
            return result
        except (json.JSONDecodeError, ValueError):
            return {"route": "complex", "reasoning": "Failed to parse — defaulting to complex"}

    def _decompose(self, question: str, compressed_history: list[dict]) -> list[str]:
        """Break a complex question into 2-4 sub-queries using Llama 8B."""
        messages = [SystemMessage(content=DECOMPOSER_SYSTEM_PROMPT)]

        # Context from compressed conversation
        for turn in compressed_history:
            messages.append(HumanMessage(content=turn["user"]))
            if turn.get("topic_line"):
                messages.append(HumanMessage(content=f"[Prior answer covered: {turn['topic_line']}]"))

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
        0. Compress history + detect follow-ups
        1. Rewrite follow-up into standalone question (if needed)
        2. Classify query complexity
        3. Decompose if complex
        4. Retrieve chunks for each sub-query
        5. Deduplicate and return context package

        Returns:
            {
                "route": "simple" | "complex" | "off_topic" | "clarification" | "meta",
                "sub_queries": [...],
                "retrieved_chunks": [...],
                "source_doc_ids": [...],
                "follow_up_detected": bool,
                "rewritten_question": str | None,
                "original_question": str,
                "clarification_question": str | None,
            }
        """
        if conversation_history is None:
            conversation_history = []

        # ── Step 0: Compress history + detect follow-up ──────────────────
        compressed = compress_history(conversation_history, max_turns=5, topic_tokens=80)
        compressed = history_budget_check(compressed, max_total_tokens=800, mode="topic")

        detection = self._detect_follow_up(question, compressed)
        follow_up_type = detection["type"]
        print(f"[Agent1] Follow-up detection: {follow_up_type} — {detection.get('reasoning', '')}")

        rewritten_question = None
        effective_question = question  # the question used for routing and retrieval

        # ── Handle special follow-up types ───────────────────────────────

        if follow_up_type == "ambiguous":
            # Build clarification from recent topic-lines
            topics = []
            for turn in compressed[-3:]:
                if turn.get("topic_line"):
                    topics.append(turn["topic_line"])
            topic_summary = "; ".join(topics) if topics else "various healthcare AI topics"
            clarification_q = (
                f"Could you clarify what you're referring to? "
                f"In our conversation, we discussed: {topic_summary}. "
                f"Which topic would you like to explore further?"
            )
            print(f"[Agent1] Ambiguous follow-up — requesting clarification.")
            return {
                "route": "clarification",
                "sub_queries": [],
                "retrieved_chunks": [],
                "source_doc_ids": [],
                "follow_up_detected": True,
                "rewritten_question": None,
                "original_question": question,
                "clarification_question": clarification_q,
            }

        if follow_up_type == "meta":
            print(f"[Agent1] Meta question detected — no retrieval needed.")
            return {
                "route": "meta",
                "sub_queries": [],
                "retrieved_chunks": [],
                "source_doc_ids": [],
                "follow_up_detected": True,
                "rewritten_question": None,
                "original_question": question,
                "clarification_question": None,
            }

        if follow_up_type == "follow_up_doc":
            # Extract DOC-XXX and retrieve more chunks from that specific doc
            doc_match = self._DOC_REF_RE.search(question)
            target_doc = doc_match.group().upper() if doc_match else None
            if target_doc:
                rewritten = f"Provide detailed information from {target_doc}"
                rewritten_question = rewritten
                effective_question = rewritten
                print(f"[Agent1] Direct doc reference — rewritten to: {effective_question}")

        elif follow_up_type == "follow_up":
            # ── Step 1: Rewrite follow-up ────────────────────────────────
            rewritten = self._rewrite_question(question, compressed)
            if rewritten:
                rewritten_question = rewritten
                effective_question = rewritten
                print(f"[Agent1] Rewritten follow-up: '{question}' → '{effective_question}'")
            else:
                print(f"[Agent1] Rewrite failed — using original question")

        # ── Step 2: Route (using effective question) ─────────────────────
        classification = self._classify(effective_question, compressed)
        route = classification["route"]
        print(f"[Agent1] Route: {route} — {classification.get('reasoning', '')}")

        # Off-topic guard
        if route == "off_topic":
            print("[Agent1] Off-topic query detected — skipping retrieval.")
            return {
                "route": "off_topic",
                "sub_queries": [],
                "retrieved_chunks": [],
                "source_doc_ids": [],
                "follow_up_detected": follow_up_type != "standalone",
                "rewritten_question": rewritten_question,
                "original_question": question,
                "clarification_question": None,
            }

        # ── Step 3: Decompose + retrieve (using effective question) ──────
        if route == "simple":
            sub_queries = [effective_question]
            print(f"[Agent1] Simple route — single-hop retrieval (no decomposition)")
        else:
            sub_queries = self._decompose(effective_question, compressed)
            print(f"[Agent1] Sub-queries: {sub_queries}")

        # Stage C: Retrieve for each sub-query
        all_chunks: list[dict] = []
        seen_chunk_keys: set[str] = set()

        # Simple route: single retrieval with top_k=15 (broad coverage)
        # Complex route: multi-query retrieval with top_k=15 each
        queries_to_search = list(sub_queries)
        if route == "complex" and effective_question not in queries_to_search:
            queries_to_search.append(effective_question)

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
            "follow_up_detected": follow_up_type != "standalone",
            "rewritten_question": rewritten_question,
            "original_question": question,
            "clarification_question": None,
        }
