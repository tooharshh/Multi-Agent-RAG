"""
Indexing Pipeline — Chunk, embed, and store documents in ChromaDB + BM25.
"""

import json
import os
import pickle
import torch
import tiktoken
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "healthcare_kb"
BM25_PATH = "./bm25_index.pkl"
KB_PATH = "./knowledge_base_ai_healthcare.json"

# Tiktoken-based length function for accurate token counting
_enc = tiktoken.get_encoding("cl100k_base")


def _tiktoken_len(text: str) -> int:
    return len(_enc.encode(text))


def load_knowledge_base(path: str = KB_PATH) -> list[dict]:
    """Load the 60-document knowledge base from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


def chunk_documents(documents: list[dict]) -> list[Document]:
    """
    Split each document into ~512-token chunks with 128-token overlap.
    Preserves all metadata per chunk for citation tracing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=_tiktoken_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    for doc in documents:
        text = doc["text"]
        chunks = splitter.split_text(text)
        for i, chunk_text in enumerate(chunks):
            metadata = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source_type": doc["source_type"],
                "source": doc["source"],
                "url": doc["url"],
                "date": doc["date"],
                "tags": ",".join(doc["tags"]),  # ChromaDB needs str, not list
                "chunk_index": i,
            }
            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))

    return all_chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local sentence-transformer embedding model on CUDA."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def build_vector_store(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embed chunks and store in ChromaDB. Idempotent — skips if collection already populated.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Idempotency check
    try:
        existing = client.get_collection(COLLECTION_NAME)
        if existing.count() > 0:
            print(f"[Indexing] Collection '{COLLECTION_NAME}' exists with {existing.count()} chunks. Skipping re-embedding.")
            return Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
            )
    except Exception:
        pass  # Collection doesn't exist yet — proceed

    print(f"[Indexing] Embedding {len(chunks)} chunks with all-mpnet-base-v2 on {DEVICE}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )
    print(f"[Indexing] ChromaDB collection '{COLLECTION_NAME}' created with {vector_store._collection.count()} chunks.")
    return vector_store


def build_bm25_index(chunks: list[Document]) -> None:
    """
    Build a BM25 sparse index from chunk texts and serialize to disk.
    Skips if pickle already exists.
    """
    if os.path.exists(BM25_PATH):
        print(f"[Indexing] BM25 index already exists at {BM25_PATH}. Skipping.")
        return

    print(f"[Indexing] Building BM25 index over {len(chunks)} chunks...")
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"[Indexing] BM25 index saved to {BM25_PATH}.")


def load_bm25_index() -> tuple:
    """Load the BM25 index and chunk list from disk."""
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def run_indexing_pipeline() -> tuple:
    """
    Full pipeline: load KB → chunk → embed into ChromaDB → build BM25.
    Returns (vector_store, embeddings).
    """
    print(f"[Indexing] Using device: {DEVICE}")

    # Load and chunk
    documents = load_knowledge_base()
    print(f"[Indexing] Loaded {len(documents)} documents from knowledge base.")

    chunks = chunk_documents(documents)
    print(f"[Indexing] Created {len(chunks)} chunks (512 tokens, 128 overlap).")

    # Embeddings
    embeddings = get_embeddings()

    # Vector store
    vector_store = build_vector_store(chunks, embeddings)

    # BM25
    build_bm25_index(chunks)

    return vector_store, embeddings


if __name__ == "__main__":
    vs, emb = run_indexing_pipeline()
    print("[Indexing] Pipeline complete.")
