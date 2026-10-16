"""
Reranking and result processing for retrieval.
"""
from __future__ import annotations

import logging
import math
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

    The implementation uses the configured embedder for query and chunk
    embeddings and combines cosine similarity with reciprocal-rank fusion.
    This is not a cross-encoder and makes no cross-encoder quality claim.
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

            # zip() silently truncates incompatible dimensions, producing a
            # plausible but meaningless score. Ignore such candidates rather
            # than ranking them as if both vectors used the same model.
            if (
                not query_embedding
                or len(query_embedding) != len(chunk_embedding)
                or not all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in [*query_embedding, *chunk_embedding])
            ):
                logger.warning("rerank_embedding_invalid", doc_id_length=len(str(doc_id)))
                continue
            if query_norm > 0 and chunk_norm > 0:
                dot = sum(a * b for a, b in zip(query_embedding, chunk_embedding))
                similarity = dot / (query_norm * chunk_norm)
            else:
                similarity = 0.0
        else:
            similarity = rrf_score

        if not isinstance(rrf_score, (int, float)) or not math.isfinite(rrf_score):
            continue
        # Combined score: weighted RRF + similarity
        combined = 0.4 * rrf_score + 0.6 * similarity
        if not math.isfinite(combined):
            continue

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
    allowed_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Hybrid search (RRF fusion)
    2. Reranking by similarity
    3. Return top-k with metadata
    """
    from docmind.retrieval.hybrid import hybrid_search

    from docmind.retrieval.hybrid import reciprocal_rank_fusion
    index = bm25_storage.get_index()
    bm25_results = index.search(query, top_k=len(index))
    if allowed_ids is not None:
        bm25_results = [x for x in bm25_results if x[0] in allowed_ids]
    fused = reciprocal_rank_fusion(bm25_results[:max(50, top_k * 5)], vector_results)
    fused = fused[:max(20, top_k * 3)]
    if fused:
        stored = await get_vector_store().get_chunks(ids=[x["doc_id"] for x in fused])
        metadata = {
            chunk_id: {**(meta or {}), "text": text}
            for chunk_id, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"])
        }
        fused = [{**x, "metadata": metadata[x["doc_id"]]} for x in fused if x["doc_id"] in metadata]
    reranked = await rerank_results(query, fused, top_k=top_k)
    return reranked
