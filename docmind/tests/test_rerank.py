"""Regression tests for evidence scores sent from retrieval to citations."""
import pytest

from docmind.retrieval import rerank


class ControlledEmbedder:
    def embed_text(self, text):
        return {
            "question": [1.0, 0.0],
            "aligned": [1.0, 0.0],
            "orthogonal": [0.0, 1.0],
            "opposite": [-1.0, 0.0],
        }[text]


@pytest.mark.asyncio
async def test_signed_cosine_is_normalized_for_evidence_api(monkeypatch):
    monkeypatch.setattr(rerank, "get_embedder", lambda: ControlledEmbedder())
    candidates = [
        {
            "doc_id": value,
            "rrf_score": 0.02,
            "metadata": {"text": value, "doc_id": value},
        }
        for value in ("opposite", "orthogonal", "aligned")
    ]

    ranked = await rerank.rerank_results("question", candidates, top_k=3)
    by_id = {item["doc_id"]: item for item in ranked}
    assert [item["doc_id"] for item in ranked] == [
        "aligned", "orthogonal", "opposite"
    ]
    assert by_id["aligned"]["similarity"] == pytest.approx(1.0)
    assert by_id["orthogonal"]["similarity"] == pytest.approx(0.5)
    assert by_id["opposite"]["similarity"] == pytest.approx(0.0)
    assert all(0.0 <= item["combined_score"] <= 1.0 for item in ranked)
    # The old code returned a negative combined_score here, causing
    # valid source citations to be rejected by QueryService.
    assert by_id["opposite"]["combined_score"] == pytest.approx(0.008)


@pytest.mark.asyncio
async def test_nonmatching_embedding_dimensions_are_not_ranked(monkeypatch):
    class WrongDimensions:
        def embed_text(self, text):
            return [1.0, 0.0] if text == "question" else [1.0]

    monkeypatch.setattr(rerank, "get_embedder", lambda: WrongDimensions())
    results = await rerank.rerank_results("question", [
        {"doc_id": "invalid", "rrf_score": 0.02, "metadata": {"text": "chunk"}}
    ])
    assert results == []
