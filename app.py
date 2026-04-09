"""
Streamlit Chat UI — AI in Healthcare Multi-Agent RAG.
"""

import re
import streamlit as st
from indexing.pipeline import run_indexing_pipeline, load_bm25_index, get_embeddings, CHROMA_PATH, COLLECTION_NAME
from retrieval.hybrid import HybridRetriever
from agents.query_decomposer import QueryDecomposerAgent
from agents.reasoner import ReasonerAgent
from agents.critic import CriticAgent
import chromadb
from langchain_chroma import Chroma

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI in Healthcare — Multi-Agent RAG",
    page_icon="🏥",
    layout="wide",
)

# ── Session state init ───────────────────────────────────────────────────────

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "indexer_loaded" not in st.session_state:
    st.session_state.indexer_loaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "agent1" not in st.session_state:
    st.session_state.agent1 = None
if "agent2" not in st.session_state:
    st.session_state.agent2 = None
if "agent3" not in st.session_state:
    st.session_state.agent3 = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    critic_enabled = st.toggle("Enable Critic Agent", value=True)
    st.divider()

    # Indexing status
    if st.session_state.indexer_loaded:
        st.success("✅ Knowledge base indexed")
    else:
        st.warning("⏳ Knowledge base not loaded")
        if st.button("🔄 Load / Index Knowledge Base"):
            with st.spinner("Indexing knowledge base... (this may take a few minutes on first run)"):
                try:
                    vector_store, embeddings = run_indexing_pipeline()
                    bm25, bm25_chunks = load_bm25_index()
                    retriever = HybridRetriever(vector_store, bm25, bm25_chunks)

                    st.session_state.retriever = retriever
                    st.session_state.agent1 = QueryDecomposerAgent(retriever)
                    st.session_state.agent2 = ReasonerAgent()
                    st.session_state.agent3 = CriticAgent(retriever)
                    st.session_state.indexer_loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()
    st.caption("**Models:**")
    st.caption("🟢 Router/Decomposer: Llama 3.1 8B")
    st.caption("🔵 Simple answers: Llama 3.1 8B")
    st.caption("🟣 Complex answers: Qwen 3 235B")
    st.caption("🟢 Critic: Llama 3.1 8B")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.conversation_history = []
        st.session_state.messages = []
        st.rerun()


# ── Helper functions ─────────────────────────────────────────────────────────

def render_answer_with_badges(answer: str, citations: list) -> str:
    """Replace [DOC-XXX] with colored HTML badge spans."""
    # Build lookup for titles
    title_map = {}
    for c in citations:
        title_map[c["doc_id"]] = c.get("title", c["doc_id"])

    def badge_replacer(match):
        doc_id = match.group(1)
        title = title_map.get(doc_id, doc_id)
        return (
            f'<span title="{title}" style="background:#1e88e5;color:white;'
            f'padding:2px 6px;border-radius:4px;font-size:0.8em;margin:0 2px;'
            f'cursor:help">{doc_id}</span>'
        )

    rendered = re.sub(r"\[(DOC-\d{3})\]", badge_replacer, answer)

    # Also handle [UNSUPPORTED] markers
    rendered = rendered.replace(
        "[UNSUPPORTED]",
        '<span style="background:#e53935;color:white;padding:2px 6px;border-radius:4px;'
        'font-size:0.75em;margin:0 2px">UNSUPPORTED</span>',
    )
    rendered = rendered.replace(
        "[UNVERIFIED]",
        '<span style="background:#ff9800;color:white;padding:2px 6px;border-radius:4px;'
        'font-size:0.75em;margin:0 2px">UNVERIFIED</span>',
    )

    return rendered


# ── Main UI ──────────────────────────────────────────────────────────────────

st.title("🏥 AI in Healthcare — Multi-Agent RAG")
st.caption("Ask questions about AI in Healthcare. Every answer is grounded in the 21-article knowledge base with citations.")

# Auto-load index on startup
if not st.session_state.indexer_loaded:
    try:
        embeddings = get_embeddings()
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            vector_store = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
            )
            bm25, bm25_chunks = load_bm25_index()
            retriever = HybridRetriever(vector_store, bm25, bm25_chunks)

            st.session_state.retriever = retriever
            st.session_state.agent1 = QueryDecomposerAgent(retriever)
            st.session_state.agent2 = ReasonerAgent()
            st.session_state.agent3 = CriticAgent(retriever)
            st.session_state.indexer_loaded = True
            st.rerun()
    except Exception as e:
        pass  # Index doesn't exist yet — user needs to click the button

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Render answer with badges
            st.markdown(msg["rendered_answer"], unsafe_allow_html=True)

            # Fact-check badge
            if msg.get("critic"):
                conf = msg["critic"]["confidence_score"]
                if conf >= 0.8:
                    st.markdown(
                        f'<span style="background:#43a047;color:white;padding:4px 10px;'
                        f'border-radius:6px;font-size:0.85em">✅ Verified ({conf:.2f})</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span style="background:#ff9800;color:white;padding:4px 10px;'
                        f'border-radius:6px;font-size:0.85em">⚠️ Review ({conf:.2f})</span>',
                        unsafe_allow_html=True,
                    )

            # Expanders
            with st.expander("🧠 Reasoning Trace"):
                st.markdown(msg.get("chain_of_thought", "N/A"))

            with st.expander("📄 Retrieved Sources"):
                for src in msg.get("citations", []):
                    st.markdown(f"**{src['doc_id']}** — {src.get('title', 'N/A')}")
                    st.caption(src.get("relevant_excerpt", "")[:300])
                    st.divider()

            with st.expander("🔍 Sub-queries used"):
                for sq in msg.get("sub_queries", []):
                    st.markdown(f"- {sq}")

            st.caption(f"Route: `{msg.get('route', 'N/A')}` | Model: `{msg.get('model_used', 'N/A')}`")
        else:
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about AI in Healthcare..."):
    if not st.session_state.indexer_loaded:
        st.error("Please load the knowledge base first using the sidebar button.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process through agents
        with st.chat_message("assistant"):
            with st.spinner("🔍 Agent 1: Decomposing query & retrieving..."):
                context_package = st.session_state.agent1.run(
                    prompt,
                    st.session_state.conversation_history,
                )

            route = context_package["route"]
            model_used = "llama3.1-8b" if route == "simple" else "qwen-3-235b-a22b-instruct-2507"

            with st.spinner(f"🧠 Agent 2: Reasoning with {model_used}..."):
                reasoner_output = st.session_state.agent2.run(
                    prompt,
                    context_package,
                    st.session_state.conversation_history,
                )

            # Critic (if enabled)
            critic_output = None
            display_answer = reasoner_output["answer"]
            if critic_enabled:
                with st.spinner("🔎 Agent 3: Verifying citations..."):
                    critic_output = st.session_state.agent3.run(
                        prompt,
                        reasoner_output,
                        context_package,
                    )
                    display_answer = critic_output["verified_answer"]

                    # Auto-escalation: if simple route gets low confidence, re-run with complex
                    if (context_package.get("route") == "simple"
                            and critic_output["confidence_score"] < 0.90):
                        with st.spinner("⬆️ Escalating to complex model..."):
                            context_package["route"] = "complex"
                            reasoner_output = st.session_state.agent2.run(
                                prompt,
                                context_package,
                                st.session_state.conversation_history,
                            )
                            critic_output = st.session_state.agent3.run(
                                prompt,
                                reasoner_output,
                                context_package,
                            )
                            display_answer = critic_output["verified_answer"]

            # Render
            rendered = render_answer_with_badges(display_answer, reasoner_output.get("citations", []))
            st.markdown(rendered, unsafe_allow_html=True)

            # Fact-check badge
            if critic_output:
                conf = critic_output["confidence_score"]
                if conf >= 0.8:
                    st.markdown(
                        f'<span style="background:#43a047;color:white;padding:4px 10px;'
                        f'border-radius:6px;font-size:0.85em">✅ Verified ({conf:.2f})</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span style="background:#ff9800;color:white;padding:4px 10px;'
                        f'border-radius:6px;font-size:0.85em">⚠️ Review ({conf:.2f})</span>',
                        unsafe_allow_html=True,
                    )

            with st.expander("🧠 Reasoning Trace"):
                st.markdown(reasoner_output.get("chain_of_thought", "N/A"))

            with st.expander("📄 Retrieved Sources"):
                for src in reasoner_output.get("citations", []):
                    st.markdown(f"**{src['doc_id']}** — {src.get('title', 'N/A')}")
                    st.caption(src.get("relevant_excerpt", "")[:300])
                    st.divider()

            with st.expander("🔍 Sub-queries used"):
                for sq in context_package.get("sub_queries", []):
                    st.markdown(f"- {sq}")

            st.caption(f"Route: `{route}` | Model: `{model_used}`")

        # Save to message history
        assistant_msg = {
            "role": "assistant",
            "rendered_answer": rendered,
            "chain_of_thought": reasoner_output.get("chain_of_thought", ""),
            "citations": reasoner_output.get("citations", []),
            "sub_queries": context_package.get("sub_queries", []),
            "route": route,
            "model_used": model_used,
            "critic": critic_output,
        }
        st.session_state.messages.append(assistant_msg)

        # Save to conversation history for context passing
        st.session_state.conversation_history.append({
            "user": prompt,
            "assistant_answer": display_answer,
        })
