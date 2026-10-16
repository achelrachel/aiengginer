"""Reranking must never turn corrupt vectors into plausible similarity scores."""

import pytest

from docmind.retrieval import rerank


class EmbeddingStub:
    def embed_text(self, value):
        values = {
            "query": [1.0, 0.0, 0.0],
            "valid": [1.0, 0.0, 0.0],
            "wrong-dimension": [1.0, 0.0],
            "nan-vector": [float("nan"), 0.0, 0.0],
            "zero-vector": [0.0, 0.0, 0.0],
        }
        return values[value]


@pytest.mark.asyncio
async def test_rerank_discards_dimension_mismatches_and_nonfinite_vectors(monkeypatch):
    monkeypatch.setattr(rerank, "get_embedder", lambda: EmbeddingStub())
    candidates = [
        {"doc_id": "wrong", "rrf_score": 0.9, "metadata": {"text": "wrong-dimension"}},
        {"doc_id": "nan", "rrf_score": 0.9, "metadata": {"text": "nan-vector"}},
        {"doc_id": "valid", "rrf_score": 0.1, "metadata": {"text": "valid"}},
    ]
    results = await rerank.rerank_results("query", candidates)
    assert [item["doc_id"] for item in results] == ["valid"]
    assert results[0]["similarity"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_zero_norm_is_bounded_and_never_inf(monkeypatch):
    monkeypatch.setattr(rerank, "get_embedder", lambda: EmbeddingStub())
    results = await rerank.rerank_results(
        "query", [{"doc_id": "zero", "rrf_score": 0.4, "metadata": {"text": "zero-vector"}}]
    )
    assert results[0]["similarity"] == 0.0
    assert results[0]["combined_score"] == pytest.approx(0.16)


@pytest.mark.asyncio
async def test_invalid_rank_fusion_score_never_reaches_output(monkeypatch):
    monkeypatch.setattr(rerank, "get_embedder", lambda: EmbeddingStub())
    results = await rerank.rerank_results(
        "query", [{"doc_id": "invalid", "rrf_score": float("inf"), "metadata": {"text": "valid"}}]
    )
    assert results == []
