"""Every hybrid retrieval source must respect the same document allowlist."""

import pytest
from docmind.retrieval import rerank


class _Index:
    def __len__(self):
        return 3

    def search(self, *_args, **_kwargs):
        return [("allowed", 8.0), ("forbidden", 7.0)]


class _BM25:
    def get_index(self):
        return _Index()


class _VectorStore:
    def __init__(self):
        self.requested = []

    async def get_chunks(self, *, ids):
        self.requested.extend(ids)
        return {
            "ids": ids,
            "documents": ["Public excerpt" for _ in ids],
            "metadatas": [{} for _ in ids],
        }


@pytest.mark.asyncio
async def test_vector_candidates_are_filtered_before_chunk_fetch(monkeypatch):
    store = _VectorStore()
    monkeypatch.setattr(rerank, "get_vector_store", lambda: store)

    async def do_not_modify_candidates(_query, candidates, top_k=8):
        return candidates[:top_k]

    monkeypatch.setattr(rerank, "rerank_results", do_not_modify_candidates)
    results = await rerank.retrieve_and_rerank(
        "query", _BM25(),
        [("forbidden", 0.01, {}), ("allowed", 0.2, {})],
        allowed_ids={"allowed"},
    )
    assert [row["doc_id"] for row in results] == ["allowed"]
    assert store.requested == ["allowed"]


@pytest.mark.asyncio
async def test_empty_allowlist_blocks_all_vector_and_keyword_chunks(monkeypatch):
    store = _VectorStore()
    monkeypatch.setattr(rerank, "get_vector_store", lambda: store)
    results = await rerank.retrieve_and_rerank(
        "query", _BM25(), [("forbidden", 0.01, {})], allowed_ids=set()
    )
    assert results == []
    assert store.requested == []
