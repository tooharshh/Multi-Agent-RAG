"""
FastAPI backend for Multi-Agent RAG.
Streams responses in AI SDK UIMessage stream format for assistant-ui.
"""

import json
import re
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from indexing.pipeline import run_indexing_pipeline, load_bm25_index
from retrieval.hybrid import HybridRetriever
from agents.query_decomposer import QueryDecomposerAgent
from agents.reasoner import ReasonerAgent
from agents.critic import CriticAgent


# ── Globals ──────────────────────────────────────────────────────────────────

retriever = None
agent1 = None
agent2 = None
agent3 = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG pipeline on startup."""
    global retriever, agent1, agent2, agent3
    print("[API] Loading indexing pipeline...")
    vector_store, embeddings = run_indexing_pipeline()
    bm25, bm25_chunks = load_bm25_index()
    retriever = HybridRetriever(vector_store, bm25, bm25_chunks)
    agent1 = QueryDecomposerAgent(retriever)
    agent2 = ReasonerAgent()
    agent3 = CriticAgent(retriever)
    print("[API] Pipeline ready.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── AI SDK stream helpers ────────────────────────────────────────────────────

def encode_text(token: str) -> str:
    """AI SDK text part: 0:"token"\n"""
    return f'0:{json.dumps(token)}\n'


def encode_data(data: list) -> str:
    """AI SDK data part: 2:[...]\n"""
    return f'2:{json.dumps(data)}\n'


def encode_finish() -> str:
    """AI SDK finish message: d:{...}\n"""
    return f'd:{json.dumps({"finishReason": "stop"})}\n'


# ── Chat endpoint ────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # Extract the last user message text
    question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            parts = msg.get("parts", msg.get("content", []))
            if isinstance(parts, str):
                question = parts
            elif isinstance(parts, list):
                for part in parts:
                    if isinstance(part, str):
                        question = part
                        break
                    elif isinstance(part, dict) and part.get("type") == "text":
                        question = part.get("text", "")
                        break
            break

    if not question:
        async def error_stream():
            yield encode_text("Please provide a question.")
            yield encode_finish()
        return StreamingResponse(error_stream(), media_type="text/plain; charset=utf-8")

    # Build conversation history from prior messages
    conversation_history = []
    for i, msg in enumerate(messages[:-1]):
        if msg.get("role") == "user":
            user_text = ""
            parts = msg.get("parts", msg.get("content", []))
            if isinstance(parts, str):
                user_text = parts
            elif isinstance(parts, list):
                for part in parts:
                    if isinstance(part, str):
                        user_text = part
                        break
                    elif isinstance(part, dict) and part.get("type") == "text":
                        user_text = part.get("text", "")
                        break
            # Find next assistant message
            assistant_text = ""
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "assistant":
                    a_parts = messages[j].get("parts", messages[j].get("content", []))
                    if isinstance(a_parts, str):
                        assistant_text = a_parts
                    elif isinstance(a_parts, list):
                        for part in a_parts:
                            if isinstance(part, str):
                                assistant_text = part
                                break
                            elif isinstance(part, dict) and part.get("type") == "text":
                                assistant_text = part.get("text", "")
                                break
                    break
            if user_text:
                conversation_history.append({
                    "user": user_text,
                    "assistant_answer": assistant_text,
                })

    async def generate():
        # ── Agent 1: Research ────────────────────────────────────────────
        yield encode_data([{
            "type": "agent_status",
            "agent": "research",
            "status": "active",
            "message": "Classifying query complexity...",
        }])
        await asyncio.sleep(0)

        # Run Agent 1 in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        context_package = await loop.run_in_executor(
            None, lambda: agent1.run(question, conversation_history)
        )

        route = context_package["route"]
        sub_queries = context_package["sub_queries"]
        num_chunks = len(context_package["retrieved_chunks"])
        num_docs = len(context_package["source_doc_ids"])

        # Off-topic short-circuit: respond without running Agent 2/3
        if route == "off_topic":
            yield encode_data([{
                "type": "agent_status",
                "agent": "research",
                "status": "complete",
                "message": "Off-topic query — no retrieval needed.",
                "route": route,
                "sub_queries": [],
            }])
            await asyncio.sleep(0)

            greeting = (
                "Hi there! I'm a research assistant specializing in AI in Healthcare. "
                "I can answer questions about topics like FDA-cleared AI devices, "
                "clinical AI adoption, drug discovery, regulation, ethics, and more. "
                "Try asking something like:\n\n"
                "- *What percentage of healthcare organisations use AI?*\n"
                "- *How many FDA-cleared AI medical devices exist?*\n"
                "- *What are the ethical concerns around clinical AI?*"
            )
            yield encode_text(greeting)
            yield encode_finish()
            return

        if route == "complex":
            decompose_msg = f"Decomposed into {len(sub_queries)} sub-queries"
        else:
            decompose_msg = "Direct lookup — single query"

        yield encode_data([{
            "type": "agent_status",
            "agent": "research",
            "status": "complete",
            "message": f"{decompose_msg}. Retrieved {num_chunks} passages from {num_docs} documents.",
            "route": route,
            "sub_queries": sub_queries,
        }])
        await asyncio.sleep(0)

        # ── Agent 2: Analysis ────────────────────────────────────────────
        model_name = "Qwen 3 235B" if route == "complex" else "Llama 3.1 8B"
        yield encode_data([{
            "type": "agent_status",
            "agent": "analysis",
            "status": "active",
            "message": f"Reasoning with {model_name}...",
        }])
        await asyncio.sleep(0)

        reasoner_output = await loop.run_in_executor(
            None, lambda: agent2.run(question, context_package, conversation_history)
        )

        chain_of_thought = reasoner_output.get("chain_of_thought", "")
        citations = reasoner_output.get("citations", [])

        yield encode_data([{
            "type": "agent_status",
            "agent": "analysis",
            "status": "complete",
            "message": f"Synthesized answer with {len(citations)} citations.",
        }])
        await asyncio.sleep(0)

        # Send reasoning trace as data
        yield encode_data([{
            "type": "reasoning",
            "text": chain_of_thought,
        }])
        await asyncio.sleep(0)

        # ── Agent 3: Writer (Critic) ─────────────────────────────────────
        yield encode_data([{
            "type": "agent_status",
            "agent": "writer",
            "status": "active",
            "message": "Verifying citations and factual accuracy...",
        }])
        await asyncio.sleep(0)

        critic_output = await loop.run_in_executor(
            None, lambda: agent3.run(question, reasoner_output, context_package)
        )

        confidence = critic_output.get("confidence_score", 0)
        flagged = critic_output.get("flagged_claims", [])

        yield encode_data([{
            "type": "agent_status",
            "agent": "writer",
            "status": "complete",
            "message": f"Verification complete — {confidence:.0%} confidence.",
        }])
        await asyncio.sleep(0)

        # ── Stream answer text ───────────────────────────────────────────
        answer = critic_output.get("verified_answer", reasoner_output.get("answer", ""))

        # Stream word by word for smooth UX
        words = answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield encode_text(token)
            # Small delay every few words for streaming feel
            if i % 8 == 0:
                await asyncio.sleep(0.01)

        # ── Send sources + critic metadata ───────────────────────────────
        yield encode_data([{
            "type": "sources",
            "sources": [
                {
                    "doc_id": c.get("doc_id", ""),
                    "title": c.get("title", ""),
                    "excerpt": c.get("relevant_excerpt", ""),
                }
                for c in citations
            ],
        }])

        yield encode_data([{
            "type": "critic",
            "confidence_score": confidence,
            "flagged_claims": flagged,
        }])

        yield encode_finish()

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8765, reload=False)
