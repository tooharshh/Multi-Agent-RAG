# How to Run the Application

This is a Multi-Agent RAG system that uses highly specialized agents to route, reason, and verify answers. 

## Prerequisites
- Python 3.11+
- Node.js 18+ and pnpm
- NVIDIA GPU with CUDA (recommended for embeddings and re-ranking), but CPUs work as well. 
- A [Cerebras Cloud](https://cloud.cerebras.ai/) API key

## 1. Set up the Python backend

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Configure environment variables

```bash
cp .env.example .env
```
Open `.env` and add your Cerebras API key:
```
CEREBRAS_API_KEY=your_key_here
```

## 3. Start the FastAPI backend

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8765
```

On the first run, the indexing pipeline will:
- Load and chunk articles from the knowledge base
- Embed them and store them in ChromaDB (`chroma_db/`)
- Build a BM25 index (`bm25_index.pkl`)

Subsequent runs will skip re-indexing automatically.

## 4. Set up and start the frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 5. Running the Evaluation

To evaluate the pipeline against expected answers, run:

```bash
python eval/run_eval.py
```

Results are saved to `eval/eval_results.json`.
