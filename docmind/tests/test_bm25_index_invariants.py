"""Regression tests for the SQLite-rebuilt in-memory BM25 index."""

from docmind.retrieval.bm25 import BM25Index, BM25Storage


def test_upsert_does_not_duplicate_document_id():
    index = BM25Index()
    index.add_document("chunk-1", "alpha alpha")
    index.add_document("chunk-1", "beta beta")
    assert index.document_count() == 1
    assert index.search("alpha") == []
    assert index.search("beta")[0][0] == "chunk-1"


def test_batch_upsert_and_removal_update_search_terms():
    index = BM25Index()
    index.add_documents([("a", "stock chart"), ("b", "balance sheet")])
    index.add_documents([("a", "API contract"), ("c", "API review")])
    assert index.document_count() == 3
    assert index.search("stock") == []
    assert {doc for doc, _ in index.search("API")} == {"a", "c"}
    index.remove_document("c")
    assert index.document_count() == 2
    assert [doc for doc, _ in index.search("review")] == []


def test_storage_rebuilds_in_memory_index_without_fake_disk_snapshot(tmp_path):
    first = BM25Storage(tmp_path)
    first.get_index().add_document("id-1", "persisted source in SQLite")
    assert first.get_index().document_count() == 1
    first.save()  # Kept for compatibility, not an on-disk BM25 writer.
    second = BM25Storage(tmp_path)
    assert second.get_index().document_count() == 0
