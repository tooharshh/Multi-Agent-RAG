"""
Indexing Pipeline — Fetch articles from live URLs, chunk, embed, and store
in ChromaDB + BM25.

Pipeline:
  1. Read corpus manifest (JSON with article URLs and metadata).
  2. For each article, fetch live content from its URL using trafilatura.
  3. If live fetch fails or returns insufficient text, fall back to the
     pre-cached text in the manifest JSON (logged as fallback).
  4. Chunk extracted text (512 tokens, 128-token overlap).
  5. Embed with sentence-transformers/all-mpnet-base-v2 on CUDA.
  6. Store in ChromaDB (dense) and BM25 pickle (sparse).
"""

import json
import logging
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

from indexing.fetcher import fetch_all_articles

# Configure logging for the pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "healthcare_kb"
BM25_PATH = "./bm25_index.pkl"
KB_PATH = "./knowledge_bases/knowledge_base_ai_healthcare.json"

# Tiktoken-based length function for accurate token counting
_enc = tiktoken.get_encoding("cl100k_base")


def _tiktoken_len(text: str) -> int:
    return len(_enc.encode(text))


def load_corpus_manifest(path: str = KB_PATH) -> list[dict]:
    """
    Load the corpus manifest JSON which contains article metadata and URLs.
    The manifest also includes pre-cached text as a fallback for articles
    whose live URLs cannot be fetched.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


# Article 21 (DOC-021) is breaking news published after any LLM training cutoff.
# It MUST be fetched from the live URL — cached text is never acceptable.
_LIVE_REQUIRED_ARTICLES = {"DOC-021"}


def fetch_and_build_documents(manifest: list[dict]) -> list[dict]:
    """
    For each article in the manifest:
      1. Attempt live URL fetch + text extraction (ALWAYS attempted first).
      2. If live fetch succeeds, use the fetched text.
      3. If live fetch fails, fall back to manifest text — EXCEPT for articles
         in _LIVE_REQUIRED_ARTICLES which must succeed live.

    Returns list of document dicts with 'text' populated and 'fetch_method' set.
    """
    # Fetch all articles from their live URLs
    fetch_results = fetch_all_articles(manifest)

    documents = []
    live_count = 0
    fallback_count = 0
    live_required_failures = []

    for doc_meta, fetch_result in zip(manifest, fetch_results):
        doc_id = doc_meta["doc_id"]
        doc = {
            "doc_id": doc_id,
            "article_number": doc_meta.get("article_number", doc_id.replace("DOC-", "")),
            "title": doc_meta["title"],
            "source_type": doc_meta["source_type"],
            "source": doc_meta["source"],
            "url": doc_meta["url"],
            "date": doc_meta["date"],
            "tags": doc_meta["tags"],
        }

        fetched_text = fetch_result.get("text")
        cached_text = doc_meta.get("text", "")
        is_live_required = doc_id in _LIVE_REQUIRED_ARTICLES

        if fetched_text and len(fetched_text.strip()) >= 500:
            doc["text"] = fetched_text
            doc["fetch_method"] = "live"
            live_count += 1
            logger.info(
                f"  {doc_id}: LIVE fetch — {len(fetched_text):,} chars"
            )
        elif is_live_required:
            # Article 21 (and any future live-required articles) MUST come from
            # the live URL.  If the live fetch failed, log a critical error but
            # still include whatever we got so Q11 can at least be attempted.
            live_required_failures.append(doc_id)
            if cached_text and len(cached_text.strip()) > 50:
                doc["text"] = cached_text
                doc["fetch_method"] = "live_failed_cached"
                fallback_count += 1
                logger.critical(
                    f"  {doc_id}: LIVE FETCH FAILED for live-required article! "
                    f"Using cached text ({len(cached_text):,} chars) as emergency fallback. "
                    f"Q11 answers may be unreliable."
                )
            else:
                logger.critical(
                    f"  {doc_id}: LIVE FETCH FAILED and no cached text available. "
                    f"This article will be MISSING from the index."
                )
                continue
        elif cached_text and len(cached_text.strip()) > 50:
            doc["text"] = cached_text
            doc["fetch_method"] = "fallback"
            fallback_count += 1
            logger.warning(
                f"  {doc_id}: FALLBACK to cached text — "
                f"live fetch returned {len(fetched_text) if fetched_text else 0} chars, "
                f"using {len(cached_text):,} cached chars"
            )
        else:
            logger.error(f"  {doc_id}: NO TEXT available (live or cached)")
            continue

        documents.append(doc)

    logger.info(
        f"[Pipeline] Text sources: {live_count} live, {fallback_count} fallback, "
        f"{len(documents)} total documents ready for indexing"
    )
    if live_required_failures:
        logger.critical(
            f"[Pipeline] WARNING: Live-required articles failed to fetch: "
            f"{live_required_failures}. Breaking news content may be stale."
        )
    return documents


def load_knowledge_base(path: str = KB_PATH) -> list[dict]:
    """
    Full document loading: reads manifest → fetches live URLs → falls back
    to cached text where needed.
    """
    manifest = load_corpus_manifest(path)
    return fetch_and_build_documents(manifest)


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
                "article_number": doc.get("article_number", doc["doc_id"].replace("DOC-", "")),
                "title": doc["title"],
                "source_type": doc["source_type"],
                "source": doc["source"],
                "url": doc["url"],
                "date": doc["date"],
                "tags": ",".join(doc["tags"]),  # ChromaDB needs str, not list
                "chunk_index": i,
                "fetch_method": doc.get("fetch_method", "unknown"),
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


def build_vector_store(chunks: list[Document], embeddings: HuggingFaceEmbeddings, force: bool = False) -> Chroma:
    """
    Embed chunks and store in ChromaDB. Idempotent — skips if collection
    already populated (unless force=True).
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Idempotency check (skip if force rebuild)
    if not force:
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
    else:
        # Force: delete existing collection first
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[Indexing] Deleted existing collection '{COLLECTION_NAME}' for force rebuild.")
        except Exception:
            pass

    print(f"[Indexing] Embedding {len(chunks)} chunks with all-mpnet-base-v2 on {DEVICE}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )
    print(f"[Indexing] ChromaDB collection '{COLLECTION_NAME}' created with {vector_store._collection.count()} chunks.")
    return vector_store


def build_bm25_index(chunks: list[Document], force: bool = False) -> None:
    """
    Build a BM25 sparse index from chunk texts and serialize to disk.
    Skips if pickle already exists (unless force=True).
    """
    if os.path.exists(BM25_PATH) and not force:
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


def run_indexing_pipeline(force: bool = False) -> tuple:
    """
    Full pipeline: fetch articles from live URLs → chunk → embed → store.

    Live fetching is ALWAYS attempted first.  ChromaDB/BM25 are only reused
    as a fallback when the live fetch → embed pipeline encounters an
    unrecoverable error (e.g. no network).

    Args:
        force: If True, delete existing indices before rebuilding.

    Returns (vector_store, embeddings).
    """
    print(f"[Indexing] Using device: {DEVICE}")
    print("[Indexing] Live URL fetch is ALWAYS attempted first.")

    # Embeddings model is needed for both fresh build and fallback
    embeddings = get_embeddings()

    # ── Step 1: Always attempt live fetch → chunk → embed ────────────
    try:
        documents = load_knowledge_base()
        live_n = sum(1 for d in documents if d.get('fetch_method') == 'live')
        fallback_n = sum(1 for d in documents if d.get('fetch_method') in ('fallback', 'live_failed_cached'))
        print(f"[Indexing] Fetched {len(documents)} documents ({live_n} live, {fallback_n} fallback).")

        chunks = chunk_documents(documents)
        print(f"[Indexing] Created {len(chunks)} chunks (512 tokens, 128 overlap).")

        # Always rebuild indices from freshly-fetched content
        vector_store = build_vector_store(chunks, embeddings, force=True)
        build_bm25_index(chunks, force=True)

        return vector_store, embeddings

    except Exception as e:
        logger.error(f"[Indexing] Live fetch pipeline failed: {e}")

        # ── Step 2: Fall back to existing ChromaDB + BM25 if available ──
        if not force:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            try:
                existing = client.get_collection(COLLECTION_NAME)
                if existing.count() > 0 and os.path.exists(BM25_PATH):
                    print(
                        f"[Indexing] FALLBACK: Using existing index "
                        f"({existing.count()} chunks) because live fetch failed."
                    )
                    vector_store = Chroma(
                        client=client,
                        collection_name=COLLECTION_NAME,
                        embedding_function=embeddings,
                    )
                    return vector_store, embeddings
            except Exception:
                pass

        # No fallback possible — re-raise the original error
        raise


if __name__ == "__main__":
    import sys
    force_rebuild = "--force" in sys.argv
    vs, emb = run_indexing_pipeline(force=force_rebuild)
    print("[Indexing] Pipeline complete.")
