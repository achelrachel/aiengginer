"""RRF must not reward repeated IDs or corrupt vector distances."""

import pytest

from docmind.retrieval.hybrid import reciprocal_rank_fusion


def test_duplicate_retriever_hits_never_inflate_their_vote():
    results = reciprocal_rank_fusion(
        [("a", 5.0), ("a", 5.0), ("b", 4.0)],
        [("a", 0.1, {"filename": "a.txt"}), ("a", 0.1, {"filename": "a.txt"})],
    )
    by_id = {row["doc_id"]: row for row in results}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"]["rrf_score"] == pytest.approx(2 / 61)
    assert by_id["b"]["rrf_score"] == pytest.approx(1 / 63)
    assert by_id["a"]["metadata"]["filename"] == "a.txt"


def test_nonfinite_vector_distances_are_removed_not_assigned_rank():
    results = reciprocal_rank_fusion(
        [("bm25", 1.0)],
        [("nan", float("nan"), {}), ("inf", float("inf"), {}), ("good", 0.2, {})],
    )
    assert {row["doc_id"] for row in results} == {"bm25", "good"}
    assert next(x for x in results if x["doc_id"] == "good")["rrf_score"] == pytest.approx(1 / 61)


@pytest.mark.parametrize("k", [0, -1, float("nan"), float("inf"), True])
def test_invalid_fusion_parameter_fails_closed(k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([("a", 1.0)], [], k=k)


def test_invalid_keyword_score_and_vector_metadata_are_not_merged():
    assert reciprocal_rank_fusion([("a", float("inf"))], [("b", 0.1, None)]) == []
