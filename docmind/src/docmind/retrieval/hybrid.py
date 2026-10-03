"""
Hybrid retrieval — RRF fusion of BM25 and vector results.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from docmind.logging_config import get_logger
from docmind.retrieval.bm25 import BM25Storage

logger = get_logger("docmind.retrieval.hybrid")


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[str, float]],
    vector_results: List[Tuple[str, float, Dict[str, Any]]],
    k: float = 60.0,
) -> List[Dict[str, Any]]:
    """
    Combine BM25 and vector search results using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = sum over queries of 1 / (k + rank(d))

    Args:
        bm25_results: List of (doc_id, score) from BM25
        vector_results: List of (doc_id, distance, metadata) from vector search
        k: RRF constant (default 60)

    Returns:
        List of fused results with doc_id, combined score, and metadata.
    """
    # Convert to rank-based scores
    rrf_scores: Dict[str, float] = {}

    # BM25: rank by score descending
    for rank, (doc_id, _) in enumerate(bm25_results, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Vector: rank by distance ascending (lower = more similar = higher rank)
    # Convert distance to similarity: sim = 1 - distance
    vector_ranked = sorted(vector_results, key=lambda x: x[1])
    for rank, (doc_id, distance, metadata) in enumerate(vector_ranked, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Build result list
    metadata_by_id = {doc_id: metadata for doc_id, _, metadata in vector_results}
    results = []
    for doc_id, rrf_score in rrf_scores.items():
        results.append({
            "doc_id": doc_id,
            "rrf_score": rrf_score,
            "metadata": metadata_by_id.get(doc_id, {}),
        })

    # Sort by RRF score descending
    results.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.debug("rrf_fusion", total=len(results), k=k)
    return results


def hybrid_search(
    query: str,
    bm25_storage: BM25Storage,
    vector_results: List[Tuple[str, float, Dict[str, Any]]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search: BM25 + vector, fused with RRF.

    Args:
        query: User query string
        bm25_storage: BM25 index storage
        vector_results: Pre-computed vector search results (doc_id, distance, metadata)
        top_k: Number of final results to return

    Returns:
        Fused top-k results.
    """
    bm25_results = bm25_storage.get_index().search(query, top_k=max(50, top_k * 5))
    fused = reciprocal_rank_fusion(bm25_results, vector_results, k=60.0)
    return fused[:top_k]
