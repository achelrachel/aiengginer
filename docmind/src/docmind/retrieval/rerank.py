"""
Reranking and result processing for retrieval.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from docmind.config import settings
from docmind.logging_config import get_logger
from docmind.embedding.model import get_embedder
from docmind.vectorstore.client import get_vector_store

logger = get_logger("docmind.retrieval.rerank")


async def rerank_results(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Rerank retrieval candidates using cosine similarity between
    query embedding and chunk text embedding.

    For demo, we use the embedder to compute query embedding and
    re-rank by combined RRF + similarity score.

    In production, this would use a cross-encoder model.
    """
    if not candidates:
        return []

    embedder = get_embedder()
    query_embedding = embedder.embed_text(query)
    query_norm = sum(x * x for x in query_embedding) ** 0.5

    reranked = []
    for candidate in candidates:
        doc_id = candidate.get("doc_id")
        if not doc_id:
            continue

        # Get chunk text from vector store metadata
        # For demo, we approximate: use the RRF score + heuristic
        rrf_score = candidate.get("rrf_score", 0.0)

        # Compute cosine similarity using metadata if available
        metadata = candidate.get("metadata", {})
        chunk_text = metadata.get("text", "")

        if chunk_text:
            chunk_embedding = embedder.embed_text(chunk_text)
            chunk_norm = sum(x * x for x in chunk_embedding) ** 0.5

            if query_norm > 0 and chunk_norm > 0:
                dot = sum(a * b for a, b in zip(query_embedding, chunk_embedding))
                similarity = dot / (query_norm * chunk_norm)
            else:
                similarity = 0.0
        else:
            similarity = rrf_score

        # Combined score: weighted RRF + similarity
        combined = 0.4 * rrf_score + 0.6 * similarity

        reranked.append({
            "doc_id": doc_id,
            "combined_score": combined,
            "rrf_score": rrf_score,
            "similarity": similarity,
            "metadata": metadata,
        })

    reranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return reranked[:top_k]


async def retrieve_and_rerank(
    query: str,
    bm25_storage: BM25Storage,
    vector_results: List[Tuple[str, float, Dict[str, Any]]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Hybrid search (RRF fusion)
    2. Reranking by similarity
    3. Return top-k with metadata
    """
    from docmind.retrieval.hybrid import hybrid_search

    fused = hybrid_search(query, bm25_storage, vector_results, top_k=max(20, top_k * 3))
    reranked = await rerank_results(query, fused, top_k=top_k)
    return reranked
