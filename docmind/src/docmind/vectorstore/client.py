"""Persistent Chroma storage using the same embeddings as query retrieval."""
from __future__ import annotations

import asyncio
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from docmind.config import settings


class VectorStore:
    def __init__(self, client=None):
        self._client = client
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_db_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name="docmind_chunks", metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        return self._collection

    async def upsert_chunks(self, chunks, embeddings):
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk requires an embedding")
        await asyncio.to_thread(
            self.collection.upsert,
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata_dict for c in chunks],
            embeddings=embeddings,
        )

    async def delete_document(self, doc_id):
        await asyncio.to_thread(self.collection.delete, where={"doc_id": doc_id})

    async def get_chunks(self, ids=None, where=None):
        return await asyncio.to_thread(
            self.collection.get, ids=ids, where=where, include=["documents", "metadatas"]
        )

    async def query(self, query_embedding, top_k=10, where=None, include=None):
        count = await self.count()
        if count == 0:
            return [], [], [], []
        results = await asyncio.to_thread(
            self.collection.query, query_embeddings=[query_embedding],
            n_results=min(top_k, count), where=where,
            include=include or ["documents", "metadatas", "distances"],
        )
        def first(key):
            rows = results.get(key)
            return rows[0] if rows else []
        return first("ids"), first("documents"), first("metadatas"), first("distances")

    async def count(self):
        return await asyncio.to_thread(self.collection.count)

    async def clean(self):
        await asyncio.to_thread(self.client.delete_collection, "docmind_chunks")
        self._collection = None


_default_store = None


def get_vector_store():
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store


def reset_vector_store():
    global _default_store
    _default_store = None
