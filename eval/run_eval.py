"""
Eval Automation — Runs all 20 eval questions through the pipeline and auto-scores.
Bonus +5 pts.
"""

import json
import os
import re
import sys
import time
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_cerebras import ChatCerebras
from langchain_core.messages import SystemMessage, HumanMessage
from indexing.pipeline import run_indexing_pipeline, load_bm25_index
from retrieval.hybrid import HybridRetriever
from agents.query_decomposer import QueryDecomposerAgent
from agents.reasoner import ReasonerAgent
from agents.critic import CriticAgent

load_dotenv(override=True)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

llm_judge = ChatCerebras(
    model="llama3.1-8b",
    api_key=CEREBRAS_API_KEY,
    temperature=0,
)

EVAL_SET_PATH = "./knowledge_bases/eval_set_ai_healthcare.json"
RESULTS_PATH = "./eval/eval_results.json"

JUDGE_SYSTEM_PROMPT = """You are an evaluation judge. Score the AI answer against the expected answer.
Be strict on citation quality — hallucinated doc_ids lose points.
Output JSON only, no other text."""

JUDGE_USER_TEMPLATE = """Question: {question}
Expected Answer: {expected_answer}
Expected Source Documents (ground truth): {source_docs}
AI Answer: {ai_answer}
AI Citations (documents the AI cited): {citations}
AI Chain of Thought: {chain_of_thought}

Score on these dimensions (JSON only):
{{
  "factual_accuracy": 0-3,
  "citation_quality": 0-3,
  "reasoning_trace": 0-2,
  "completeness": 0-2,
  "reasoning": "brief explanation"
}}

Scoring guide:
- factual_accuracy: 0=wrong/contradicts expected, 1=partially correct (some key facts present, significant gaps), 2=mostly correct (most key facts present), 3=fully correct (all major facts from expected answer are present and accurate)
- citation_quality: 0=no citations or all fabricated doc_ids, 1=some citations correct but most expected source docs missing, 2=most citations reference expected source docs correctly, 3=all key claims cite the expected source documents correctly. IMPORTANT: citing ADDITIONAL valid docs beyond expected sources is NOT a penalty. Only penalize if expected source docs are not cited or if doc_ids are fabricated/hallucinated.
- reasoning_trace: 0=no chain of thought at all, 1=chain of thought exists but is brief or superficial without clear logical steps, 2=chain of thought is detailed with clear numbered steps showing document identification, fact extraction, and synthesis
- completeness: 0=most of the question unanswered, 1=partially addresses the question (some parts answered but key aspects missing), 2=fully addresses ALL parts of the question with sufficient detail"""


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found: {text[:200]}")


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    # Normalize source_docs from "Art. XX" to "DOC-0XX" format for compatibility
    for q in questions:
        normalized = []
        for s in q.get("source_docs", []):
            if s.startswith("Art. "):
                num = s.replace("Art. ", "").zfill(3)
                normalized.append(f"DOC-{num}")
            else:
                normalized.append(s)
        q["source_docs"] = normalized
    return questions


def judge_answer(question: str, expected: str, source_docs: list, ai_answer: str, citations: list, chain_of_thought: str = "") -> dict:
    """Score a single answer using Llama 8B as judge."""
    citations_str = ", ".join(
        f"{c['doc_id']} ({c.get('title', 'N/A')})" for c in citations
    ) if citations else "None"

    # Truncate chain_of_thought to avoid overwhelming the judge
    cot_str = chain_of_thought[:1500] if chain_of_thought else "Not provided"

    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        expected_answer=expected,
        source_docs=", ".join(source_docs),
        ai_answer=ai_answer,
        citations=citations_str,
        chain_of_thought=cot_str,
    )

    try:
        response = llm_judge.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        scores = _parse_json_response(response.content)
        # Clamp values
        scores["factual_accuracy"] = max(0, min(3, int(scores.get("factual_accuracy", 0))))
        scores["citation_quality"] = max(0, min(3, int(scores.get("citation_quality", 0))))
        scores["reasoning_trace"] = max(0, min(2, int(scores.get("reasoning_trace", 0))))
        scores["completeness"] = max(0, min(2, int(scores.get("completeness", 0))))
        scores["total"] = (
            scores["factual_accuracy"]
            + scores["citation_quality"]
            + scores["reasoning_trace"]
            + scores["completeness"]
        )
        return scores
    except Exception as e:
        print(f"  [Judge] Error: {e}")
        return {
            "factual_accuracy": 0,
            "citation_quality": 0,
            "reasoning_trace": 0,
            "completeness": 0,
            "total": 0,
            "reasoning": f"Judge error: {e}",
        }


def run_eval():
    """Run all 20 eval questions through the full pipeline and score."""
    print("=" * 70)
    print("EVAL AUTOMATION — AI in Healthcare Multi-Agent RAG")
    print("=" * 70)

    # Initialize pipeline
    print("\n[1/3] Loading indexing pipeline...")
    vector_store, embeddings = run_indexing_pipeline()
    bm25, bm25_chunks = load_bm25_index()
    retriever = HybridRetriever(vector_store, bm25, bm25_chunks)
    agent1 = QueryDecomposerAgent(retriever)
    agent2 = ReasonerAgent()
    agent3 = CriticAgent(retriever)

    # Load eval set
    print("\n[2/3] Loading eval questions...")
    questions = load_eval_set()
    print(f"  Found {len(questions)} questions.")

    # Reorder: hard first, then medium, then easy — maximizes Qwen usage on hardest questions
    difficulty_order = {"hard": 0, "medium": 1, "easy": 2}
    questions_sorted = sorted(questions, key=lambda q: difficulty_order.get(q.get("difficulty", "medium"), 1))

    # Run each question
    print("\n[3/3] Running eval...\n")
    results = []

    for i, q in enumerate(questions_sorted):
        eval_id = q["eval_id"]
        question = q["question"]
        expected = q["expected_answer"]
        source_docs = q["source_docs"]
        difficulty = q["difficulty"]

        print(f"--- {eval_id} ({difficulty}) ---")
        print(f"  Q: {question[:80]}...")

        start = time.time()

        try:
            # Agent 1
            context_package = agent1.run(question)

            # Agent 2
            reasoner_output = agent2.run(question, context_package)

            # Agent 3
            critic_output = agent3.run(question, reasoner_output, context_package)

            # Auto-escalation: if simple route gets low confidence, re-run with complex
            if (context_package["route"] == "simple"
                    and critic_output["confidence_score"] < 0.90):
                print(f"  [Escalation] Low confidence ({critic_output['confidence_score']:.2f}) on simple route — re-running as complex")
                context_package["route"] = "complex"
                reasoner_output = agent2.run(question, context_package)
                critic_output = agent3.run(question, reasoner_output, context_package)

            elapsed = time.time() - start
            ai_answer = critic_output["verified_answer"]
            citations = reasoner_output.get("citations", [])
            chain_of_thought = reasoner_output.get("chain_of_thought", "")

            # Judge
            scores = judge_answer(question, expected, source_docs, ai_answer, citations, chain_of_thought)

            result = {
                "eval_id": eval_id,
                "question": question,
                "difficulty": difficulty,
                "route": context_package["route"],
                "sub_queries": context_package["sub_queries"],
                "source_doc_ids_retrieved": context_package["source_doc_ids"],
                "source_doc_ids_expected": source_docs,
                "ai_answer": ai_answer,
                "chain_of_thought": reasoner_output.get("chain_of_thought", ""),
                "citations": citations,
                "confidence_score": critic_output["confidence_score"],
                "flagged_claims": critic_output["flagged_claims"],
                "scores": scores,
                "elapsed_seconds": round(elapsed, 2),
            }

            print(f"  Route: {context_package['route']} | Retrieved: {context_package['source_doc_ids']}")
            print(f"  Score: {scores['total']}/10 (FA:{scores['factual_accuracy']} CQ:{scores['citation_quality']} RT:{scores['reasoning_trace']} CO:{scores['completeness']})")
            print(f"  Confidence: {critic_output['confidence_score']:.2f} | Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "eval_id": eval_id,
                "question": question,
                "difficulty": difficulty,
                "error": str(e),
                "scores": {"factual_accuracy": 0, "citation_quality": 0, "reasoning_trace": 0, "completeness": 0, "total": 0},
            }

        results.append(result)
        print()

        # Delay between questions to reduce API rate limiting
        if i < len(questions) - 1:
            time.sleep(5)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_raw = sum(r["scores"]["total"] for r in results)
    max_raw = len(results) * 10
    normalized = (total_raw / max_raw) * 50 if max_raw > 0 else 0

    print(f"\n{'Eval ID':<12} {'Diff':<8} {'FA':>3} {'CQ':>3} {'RT':>3} {'CO':>3} {'Total':>6}")
    print("-" * 42)
    for r in results:
        s = r["scores"]
        print(f"{r['eval_id']:<12} {r.get('difficulty','?'):<8} {s['factual_accuracy']:>3} {s['citation_quality']:>3} {s['reasoning_trace']:>3} {s['completeness']:>3} {s['total']:>6}")

    print("-" * 42)
    print(f"{'RAW TOTAL':<24} {total_raw:>18}/{max_raw}")
    print(f"{'NORMALIZED (50 pts)':<24} {normalized:>18.1f}/50")
    print()

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_questions": len(results),
                "raw_score": total_raw,
                "max_raw": max_raw,
                "normalized_score": round(normalized, 1),
            },
            "results": results,
        }, f, indent=2)

    print(f"Results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_eval()
