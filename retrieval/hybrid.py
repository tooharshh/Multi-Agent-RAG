"""
Hybrid Retrieval — Dense (ChromaDB) + Sparse (BM25) + RRF + Cross-Encoder Re-ranking.
"""

import torch
import numpy as np
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HybridRetriever:
    """
    Combines dense vector search (ChromaDB) with sparse BM25 retrieval,
    merges via Reciprocal Rank Fusion, and re-ranks with a cross-encoder.
    """

    def __init__(
        self,
        vector_store: Chroma,
        bm25: BM25Okapi,
        bm25_chunks: list[Document],
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.bm25_chunks = bm25_chunks
        print(f"[Retrieval] Loading cross-encoder on {DEVICE}...")
        self.cross_encoder = CrossEncoder(
            cross_encoder_model,
            device=DEVICE,
            max_length=512,
        )
        print("[Retrieval] HybridRetriever ready.")

    def _dense_search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        """ChromaDB similarity search with scores."""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def _sparse_search(self, query: str, n: int = 20) -> list[tuple[Document, float]]:
        """BM25 sparse retrieval."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:n]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.bm25_chunks[idx], float(scores[idx])))
        return results

    @staticmethod
    def _reciprocal_rank_fusion(
        rankings: list[list[tuple[str, float]]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Merge multiple ranked lists using RRF.
        Each ranking is a list of (chunk_id, score) tuples.
        Returns merged list sorted by RRF score descending.
        """
        rrf_scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, (chunk_id, _score) in enumerate(ranking):
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _make_chunk_id(self, doc: Document) -> str:
        """Unique ID for a chunk: doc_id + chunk_index."""
        return f"{doc.metadata['doc_id']}_{doc.metadata['chunk_index']}"

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Full hybrid retrieval pipeline:
        1. Dense search (top-20)
        2. BM25 sparse search (top-20)
        3. RRF merge
        4. Cross-encoder re-rank top-20 → return top-k
        """
        # Step 1 & 2: parallel retrieval
        dense_results = self._dense_search(query, k=20)
        sparse_results = self._sparse_search(query, n=20)

        # Build lookup by chunk_id
        chunk_lookup: dict[str, Document] = {}
        dense_ranking: list[tuple[str, float]] = []
        for doc, score in dense_results:
            cid = self._make_chunk_id(doc)
            chunk_lookup[cid] = doc
            dense_ranking.append((cid, score))

        sparse_ranking: list[tuple[str, float]] = []
        for doc, score in sparse_results:
            cid = self._make_chunk_id(doc)
            chunk_lookup[cid] = doc
            sparse_ranking.append((cid, score))

        # Step 3: RRF merge
        merged = self._reciprocal_rank_fusion([dense_ranking, sparse_ranking])

        # Take top-20 for re-ranking
        candidates = []
        for cid, rrf_score in merged[:20]:
            if cid in chunk_lookup:
                candidates.append((cid, chunk_lookup[cid], rrf_score))

        if not candidates:
            return []

        # Step 4: Cross-encoder re-rank (batch ≤ 16 for 6 GB VRAM)
        pairs = [(query, c[1].page_content) for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs, batch_size=16)

        # Combine and sort
        scored = []
        for i, (cid, doc, rrf_score) in enumerate(candidates):
            scored.append({
                "text": doc.page_content,
                "doc_id": doc.metadata["doc_id"],
                "article_number": doc.metadata.get("article_number", doc.metadata["doc_id"].replace("DOC-", "")),
                "title": doc.metadata["title"],
                "source": doc.metadata["source"],
                "url": doc.metadata["url"],
                "date": doc.metadata["date"],
                "tags": doc.metadata["tags"].split(",") if doc.metadata["tags"] else [],
                "chunk_index": doc.metadata["chunk_index"],
                "relevance_score": float(ce_scores[i]),
            })

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored[:top_k]
