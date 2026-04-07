# Multi-Agent RAG — AI in Healthcare: Complete Project Instructions

## Project Goal

Build a fully working multi-agent RAG system over a 60-document healthcare AI knowledge base.
Scored on: System Quality (50 pts) + Eval Set Performance (50 pts) + Bonus (up to 15 pts).
**Scoring priority order: Citation accuracy → Factual accuracy → Reasoning trace → Completeness**

---

## Critical Non-Negotiable Rules

1. **NEVER** include a factual claim in any answer without a `[DOC-XXX]` citation.
2. **NEVER** hallucinate a `doc_id` — only cite IDs that appear in retrieved chunks from the vector store.
3. **Chain-of-thought MUST be a separate JSON field** — not embedded inside the answer text.
4. **Conversation history MUST be passed** to both Agent 1 and Agent 2 on every call.
5. **Indexing is idempotent** — check if `./chroma_db/` exists and has documents before re-embedding.
6. **NEVER** modify `eval_set_ai_healthcare.json` questions or expected answers.
7. **NEVER** scrape external data — use only `knowledge_base_ai_healthcare.json`.

---

## Hardware & Environment

- **OS:** Windows 11
- **GPU:** NVIDIA RTX 3050 Mobile — 6 GB VRAM, CUDA installed
- **Python:** Use `.venv` virtual environment (project root)
- **Activation:** `.venv\Scripts\activate` on Windows
- **CUDA config:** All sentence-transformer and cross-encoder models must use `device="cuda"` when CUDA is available; fall back to CPU gracefully

```python
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## LLM Provider: Cerebras API

```python
# Load from environment
from dotenv import load_dotenv
import os
load_dotenv()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
```

**`.env` file** (at project root, gitignored):
```
CEREBRAS_API_KEY=csk-f5fedh6eyjeprtkmdhvtt6dvm5e8w4d3tjt8xv22cr6tfy54
```

### LangChain-Cerebras Integration

```python
from langchain_cerebras import ChatCerebras

llm_fast = ChatCerebras(
    model="llama3.1-8b",
    api_key=CEREBRAS_API_KEY,
    temperature=0,
)

llm_heavy = ChatCerebras(
    model="qwen-3-235b-a22b-instruct-2507",   # Qwen 3 235B Instruct
    api_key=CEREBRAS_API_KEY,
    temperature=0.1,
)
```

### Model Routing Rules (Strict)

| Task | Model | Reason |
|---|---|---|
| Query complexity classification | `llama3.1-8b` | Near-zero latency classification |
| Sub-query decomposition | `llama3.1-8b` | Structured output, fast |
| **Simple route** — final answer synthesis | `llama3.1-8b` | Single-doc, lightweight |
| **Complex route** — final answer synthesis | **`qwen-3-235b-a22b-instruct-2507`** | Any multi-hop question uses heavy model |
| Critic verification (claim checking) | `llama3.1-8b` | Pattern-matching, not deep reasoning |
| LLM eval judge (all 20 questions) | `llama3.1-8b` | High-volume, cost-sensitive |

**Simple vs Complex routing rule:**
- `simple`: question contains single-hop factual retrieval (e.g., a specific stat, date, name) → Llama 8B
- `complex`: ANY question requiring synthesis, comparison, or 2+ documents → **Qwen 235B**
- When in doubt: route to **complex** (Qwen 235B) — accuracy is more important than speed

---

## Dataset Schema

**File:** `knowledge_base_ai_healthcare.json`

```json
{
  "metadata": { "total_documents": 60, ... },
  "documents": [
    {
      "doc_id": "DOC-001",
      "title": "string",
      "source_type": "research_paper|market_report|blog|newsletter",
      "source": "string",
      "url": "string",
      "date": "YYYY-MM-DD",
      "tags": ["string", ...],
      "text": "~3500 words of content"
    }
  ]
}
```

**File:** `eval_set_ai_healthcare.json`

```json
{
  "questions": [
    {
      "eval_id": "EVAL-001",
      "question": "string",
      "expected_answer": "string",
      "source_docs": ["DOC-001"],
      "difficulty": "easy|medium|hard",
      "reasoning_type": "factual_retrieval|multi_doc_synthesis|cross_corpus_analysis|comparison|evidence_synthesis"
    }
  ]
}
```

---

## Project File Structure

```
Multi-Agent-RAG/
├── .venv/                          # Virtual environment (gitignored)
├── .env                            # Cerebras API key (gitignored)
├── .gitignore
├── requirements.txt
├── chroma_db/                      # ChromaDB persistent store (gitignored, auto-created)
├── bm25_index.pkl                  # BM25 serialized index (auto-created)
│
├── indexing/
│   ├── __init__.py
│   └── pipeline.py
│
├── retrieval/
│   ├── __init__.py
│   └── hybrid.py
│
├── agents/
│   ├── __init__.py
│   ├── query_decomposer.py
│   ├── reasoner.py
│   └── critic.py
│
├── eval/
│   └── run_eval.py
│
├── app.py                          # Streamlit UI entry point
├── knowledge_base_ai_healthcare.json
├── eval_set_ai_healthcare.json
└── task.md
```

---

## Phase 1 — Indexing Pipeline (`indexing/pipeline.py`)

### Chunking Strategy

- **Splitter:** `RecursiveCharacterTextSplitter` from LangChain
- **Chunk size:** 512 tokens (use `tiktoken` with `cl100k_base` encoding for token counting)
- **Overlap:** 128 tokens
- **Metadata preserved per chunk:**
  ```python
  {
      "doc_id": "DOC-001",
      "title": "...",
      "source_type": "research_paper",
      "source": "Nature Medicine",
      "url": "...",
      "date": "2025-08-14",
      "tags": "medical imaging,ophthalmology",   # comma-joined string for ChromaDB compat
      "chunk_index": 0
  }
  ```

### Embedding Model (Local, CUDA)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
)
```

- **Dimension:** 768
- **Normalize:** True (required for cosine similarity in ChromaDB)
- **Batch size:** 32 (fits within 6 GB VRAM)

### ChromaDB Setup

```python
import chromadb
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "healthcare_kb"

client = chromadb.PersistentClient(path=CHROMA_PATH)

vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)
```

**Idempotency check:**
```python
existing = client.get_collection(COLLECTION_NAME)
if existing.count() > 0:
    print(f"Collection exists with {existing.count()} chunks. Skipping re-embedding.")
    return vector_store
```

### BM25 Index

```python
from rank_bm25 import BM25Okapi
import pickle

# Tokenize chunk texts for BM25
tokenized_corpus = [chunk.page_content.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)

with open("./bm25_index.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)
```

---

## Phase 2 — Hybrid Retrieval (`retrieval/hybrid.py`)

### Class: `HybridRetriever`

**Method:** `retrieve(query: str, top_k: int = 5) -> List[Dict]`

**Internal flow:**
1. Dense retrieval: `vector_store.similarity_search_with_score(query, k=20)`
2. Sparse retrieval: `bm25.get_top_n(query.lower().split(), all_chunks, n=20)`
3. RRF merge: combine rank lists with $k=60$
4. Cross-encoder re-rank top-20 → return top-5

### Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings: list[list], k: int = 60) -> list:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Cross-Encoder Re-ranking

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=DEVICE,
    max_length=512
)
pairs = [(query, chunk["text"]) for chunk in candidates]
scores = cross_encoder.predict(pairs)
```

### Output format per chunk:
```python
{
    "text": "chunk content...",
    "doc_id": "DOC-001",
    "title": "Deep Learning for Diabetic Retinopathy...",
    "source": "Nature Medicine",
    "url": "...",
    "date": "2025-08-14",
    "tags": ["medical imaging", "ophthalmology"],
    "chunk_index": 0,
    "relevance_score": 0.94
}
```

---

## Phase 3 — Agent 1: Query Decomposer (`agents/query_decomposer.py`)

### Class: `QueryDecomposerAgent`

**Method:** `run(question: str, conversation_history: list) -> dict`

### Stage A — Router (Llama 8B)

System prompt:
```
You are a query complexity classifier. Given a user question about AI in Healthcare, classify it as:
- "simple": requires a single document lookup of a specific fact, statistic, name, or date
- "complex": requires synthesizing information from multiple documents, making comparisons, explaining mechanisms, or analyzing trends

Output JSON only: {"route": "simple" | "complex", "reasoning": "one sentence"}
```

**Few-shot examples embedded in prompt:**
- "What was the FDA approval rate for AI devices in radiology?" → simple
- "Compare approaches of FDA vs EU MDR and discuss bias implications" → complex
- "What sensitivity did the diabetic retinopathy AI achieve?" → simple
- "How does AlphaFold affect drug discovery timelines and what are the ethical implications?" → complex

### Stage B — Decomposer (Llama 8B, complex only)

System prompt:
```
You are a sub-query generator. Break the user's complex question into 2-4 targeted sub-queries.
Each sub-query should be independently searchable — specific, concise, < 15 words.
Output JSON only: {"sub_queries": ["sub-query 1", "sub-query 2", ...]}
```

### Output schema:
```python
{
    "route": "simple" | "complex",
    "sub_queries": ["..."],      # ["<original question>"] if simple, list of decomposed if complex
    "retrieved_chunks": [        # List of chunk dicts from HybridRetriever
        {
            "text": "...",
            "doc_id": "DOC-001",
            "title": "...",
            "relevance_score": 0.94,
            ...
        }
    ],
    "source_doc_ids": ["DOC-001", "DOC-018"]   # deduplicated
}
```

---

## Phase 4 — Agent 2: Reasoner (`agents/reasoner.py`)

### Class: `ReasonerAgent`

**Method:** `run(question: str, context_package: dict, conversation_history: list) -> dict`

### Model selection:
```python
if context_package["route"] == "simple":
    llm = llm_fast    # Llama 8B
else:
    llm = llm_heavy   # Qwen 3 235B — ALWAYS for complex route
```

### System prompt (used for both models):
```
You are a medical AI research analyst. You answer questions strictly based on the retrieved documents provided.

RULES:
1. Every sentence containing a factual claim MUST end with the citation [DOC-XXX] where XXX is the document number.
2. ONLY cite doc_ids that appear in the RETRIEVED CONTEXT below. Never invent or guess doc_ids.
3. Your chain_of_thought field must show your step-by-step reasoning BEFORE writing the answer.
4. The answer field must be complete, covering all parts of the question.
5. The citations array must list every unique doc_id cited in the answer with a relevant excerpt.

OUTPUT FORMAT (JSON only):
{
  "chain_of_thought": "Step 1: ...\nStep 2: ...\nStep 3: ...",
  "answer": "Factual claim here [DOC-001]. Another claim [DOC-018].",
  "citations": [
    {"doc_id": "DOC-001", "title": "...", "relevant_excerpt": "..."}
  ]
}
```

### Conversation context injection:
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [SystemMessage(content=system_prompt)]
for turn in conversation_history:
    messages.append(HumanMessage(content=turn["user"]))
    messages.append(AIMessage(content=turn["assistant_answer"]))
messages.append(HumanMessage(content=f"QUESTION: {question}\n\nRETRIEVED CONTEXT:\n{formatted_context}"))
```

---

## Phase 5 — Agent 3: Critic (`agents/critic.py`) — Bonus +5 pts

### Class: `CriticAgent`

**Method:** `run(question: str, reasoner_output: dict, context_package: dict) -> dict`

**Model:** Llama 8B (verification is fast, not reasoning-heavy)

### Verification steps:

1. **Structural check:** Parse all `[DOC-XXX]` tags in `answer` field
2. **Existence check:** For each cited `doc_id`, verify it's in `context_package["source_doc_ids"]`
3. **Support check (LLM):** For each citation that passes existence check, ask Llama 8B:
   ```
   Does this excerpt support this claim? Answer YES or NO and one sentence explanation.
   Claim: "..."
   Excerpt from DOC-XXX: "..."
   ```
4. **Flag:** Claims failing existence or support check → append `[UNSUPPORTED]` and log to `flagged_claims`
5. **Re-retrieval trigger:** If >1 claim flagged, build a refined query from flagged claims, retrieve again, and flag for optional re-run of Agent 2

### Output schema:
```python
{
    "verified_answer": "answer text with [UNSUPPORTED] markers where applicable",
    "flagged_claims": [
        "Claim '...' cites DOC-042 which was not retrieved — not in context"
    ],
    "confidence_score": 0.91,   # (supported_claims / total_claims)
    "needs_rerun": False         # True if >1 claim flagged
}
```

---

## Phase 6 — Streamlit UI (`app.py`)

### Session state keys:
```python
st.session_state.conversation_history  # List[{user, assistant_answer, cot, citations, critic}]
st.session_state.indexer_loaded        # bool
```

### Layout requirements:
1. **Header:** "🏥 AI in Healthcare — Multi-Agent RAG"
2. **Chat history area:** renders all past turns in order
3. **Per-response blocks:**
   - Citation badges: `[DOC-001]` rendered as colored pills with doc title on hover
   - Fact-check badge: `✅ Verified (0.94)` in green or `⚠️ Review (0.61)` in yellow based on `confidence_score`
   - `st.expander("🧠 Reasoning Trace")` → shows `chain_of_thought`
   - `st.expander("📄 Retrieved Sources")` → shows each source with title, doc_id, excerpt
   - `st.expander("🔍 Sub-queries used")` → shows `sub_queries` list
4. **Input area:** `st.chat_input()` at bottom
5. **Sidebar:** indexing status, model used (llama/qwen), settings toggle for Critic on/off

### Rendering citation badges:
```python
def render_answer_with_badges(answer: str, citations: list) -> str:
    # Replace [DOC-XXX] with colored HTML badge spans
    for citation in citations:
        doc_id = citation["doc_id"]
        title = citation["title"]
        badge = f'<span title="{title}" style="background:#1e88e5;color:white;padding:2px 6px;border-radius:4px;font-size:0.8em;margin:0 2px">{doc_id}</span>'
        answer = answer.replace(f"[{doc_id}]", badge)
    return answer
```

---

## Phase 7 — Eval Automation (`eval/run_eval.py`) — Bonus +5 pts

### Flow:
1. Load `eval_set_ai_healthcare.json`
2. Ensure vector store is loaded (call indexing pipeline)
3. For each of 20 questions: run full Agent 1 → 2 → 3 pipeline
4. Score with Llama 8B as judge:

**Judge prompt per question:**
```
You are an evaluation judge. Score this AI answer against the expected answer.

Question: {question}
Expected Answer: {expected_answer}
Source Documents: {source_docs}
AI Answer: {ai_answer}
AI Citations: {citations}

Score on these dimensions (JSON only):
{
  "factual_accuracy": 0-3,   // 0=wrong, 1=partially correct, 2=mostly correct, 3=fully correct
  "citation_quality": 0-3,   // 0=hallucinated/missing, 1=some correct, 2=mostly correct, 3=all correct and no hallucinations
  "reasoning_trace": 0-2,    // 0=absent, 1=present but weak, 2=clear and logical
  "completeness": 0-2,       // 0=incomplete, 1=partially complete, 2=fully addresses question
  "reasoning": "brief explanation"
}
```

5. Save results to `eval/eval_results.json`
6. Print summary table with per-question scores and grand total
7. Normalize total to 50 pts

---

## Requirements File (`requirements.txt`)

```
# LLM & LangChain
langchain>=0.3.0
langchain-community>=0.3.0
langchain-chroma>=0.2.0
langchain-cerebras>=0.1.0
langchain-huggingface>=0.1.0

# Vector store
chromadb>=0.6.0

# Embeddings & reranking (local, CUDA)
sentence-transformers>=3.0.0
torch>=2.3.0         # CUDA build installed separately if needed

# Sparse retrieval
rank-bm25>=0.2.2

# Tokenization
tiktoken>=0.7.0

# UI
streamlit>=1.35.0

# Utilities
python-dotenv>=1.0.0
numpy>=1.26.0
```

### CUDA installation note:
If `torch.cuda.is_available()` returns False after `pip install torch`, install the CUDA-enabled build:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## `.gitignore`

```
.venv/
.env
chroma_db/
bm25_index.pkl
__pycache__/
*.pyc
*.pyo
eval/eval_results.json
```

---

## Implementation Order (Execute Phases Sequentially)

1. **Phase 0:** Setup `.env`, `.gitignore`, `.venv`, install requirements
2. **Phase 1:** `indexing/pipeline.py` — run and verify chunk count > 0 in ChromaDB
3. **Phase 2:** `retrieval/hybrid.py` — test with `"diabetic retinopathy AI sensitivity"` → must return DOC-001 in top-3
4. **Phase 3:** `agents/query_decomposer.py` — test simple and complex routing
5. **Phase 4:** `agents/reasoner.py` — test with EVAL-001 (easy) → answer must cite `[DOC-001]`
6. **Phase 5:** `agents/critic.py` — test citation verification
7. **Phase 6:** `app.py` — run `streamlit run app.py`
8. **Phase 7:** `eval/run_eval.py` — run all 20 eval questions

**Validation gate between phases:** Each phase must produce correct output on at least EVAL-001, EVAL-002, EVAL-003 before proceeding.

---

## Known Constraints & Pitfalls

- **ChromaDB metadata:** Only accepts `str`, `int`, `float`, `bool` values — convert `tags` list to comma-joined string
- **Qwen 235B token limits:** Trim context to stay under 8,192 token context window; prefer top-3 chunks for simple, top-5 for complex
- **Cross-encoder VRAM:** With 6 GB VRAM, limit cross-encoder batch to 16 pairs at a time
- **Cerebras API timeouts:** Wrap all LLM calls in retry logic with exponential backoff (max 3 retries)
- **BM25 persistence:** Always pickle both `bm25` object AND `all_chunks` list together — BM25 needs the corpus to reconstruct results
- **Citation regex:** Use `re.findall(r'\[DOC-\d{3}\]', answer)` to extract citations for Critic
- **Streamlit rerun:** Use `st.session_state` carefully — avoid re-running indexing on every Streamlit rerender
