"""
ChromaDB vector database client.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api import ChromaBackendAPI
from chromadb.api.client import ChromaClient

from docmind.config import settings
from docmind.db.models import Chunk
from docmind.logging_config import get_logger

logger = get_logger("docmind.vectorstore")


class VectorStore:
    """Wrapper around ChromaDB for document chunk storage and retrieval."""

    def __init__(self, client: Optional[ChromaClient] = None):
        self._client = client
        self._collection: Optional[chromadb.Collection] = None
        self._collection_name = "docmind_chunks"

    @property
    def client(self) -> ChromaClient:
        if self._client is None:
            chroma_settings = ChromaSettings(
                persist_directory=str(settings.chroma_db_dir),
                anonymized_telemetry=False,
            )
            self._client = chromadb.Client(chroma_settings)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """Insert or update chunks in the vector store asynchronously."""
        if not chunks:
            return

        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata_dict for c in chunks]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info("upserted_chunks", count=len(chunks), doc_ids=ids[:10])

    async def delete_document(self, doc_id: str) -> None:
        """Remove all chunks for a document."""
        result = self.collection.get(where={"document_id": doc_id}, include=["ids"])
        if result["ids"]:
            self.collection.delete(ids=result["ids"])
            logger.info("deleted_doc_chunks", doc_id=doc_id, count=len(result["ids"]))

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """Query the vector store and return ids, documents, metadatas, distances."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=include,
        )

        ids = results["ids"][0] if results["ids"] else []
        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        return ids, docs, metadatas, distances

    async def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self.collection.count()

    async def clean(self) -> None:
        """Delete the collection and recreate it."""
        try:
            self.client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = None
        logger.info("vector_store_cleaned")


_default_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store


def reset_vector_store() -> None:
    global _default_store
    _default_store = None
