"""
Ingestion service — orchestrates document upload, parsing, chunking, embedding.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from docmind.config import settings
from docmind.db.models import Document, Chunk, get_session, init_db
from docmind.embedding.model import get_embedder
from docmind.indexing.chunker import chunk_text
from docmind.indexing.parser import parse_file
from docmind.logging_config import get_logger
from docmind.security.validation import (
    clean_filename,
    secure_upload_path,
    validate_file_type,
    MAX_UPLOAD_SIZE,
)
from docmind.vectorstore.client import get_vector_store

logger = get_logger("docmind.services.ingestion")


class IngestionService:
    """Service for document ingestion pipeline."""

    async def ingest_file(self, file: UploadFile, original_filename: str) -> Document:
        """
        Process an uploaded file through the full ingestion pipeline:
        1. Validate and save file
        2. Parse document
        3. Chunk text
        4. Embed chunks
        5. Store in vector database and metadata DB

        Returns the created Document record.
        """
        logger.info("ingest_start", filename=original_filename, content_type=file.content_type)

        # Step 1: Validate file
        if file.size and file.size > MAX_UPLOAD_SIZE:
            raise ValueError(f"File size {file.size} exceeds maximum {MAX_UPLOAD_SIZE} bytes")

        # Save file to secure path
        safe_path = secure_upload_path(original_filename)
        try:
            content = await file.read()
            if len(content) > MAX_UPLOAD_SIZE:
                raise ValueError(f"File content size exceeds maximum")
            safe_path.write_bytes(content)
        except Exception as e:
            logger.error("file_save_failed", error=str(e))
            raise

        # Validate file type by magic bytes
        is_valid, reason = validate_file_type(safe_path)
        if not is_valid:
            safe_path.unlink(missing_ok=True)
            logger.warning("invalid_file_type", filename=original_filename, reason=reason)
            raise ValueError(f"Invalid file type: {reason}")

        # Determine file type
        ext = safe_path.suffix.lower().lstrip(".")
        file_type = ext

        doc_id = None
        try:
            # Step 2: Parse document
            logger.info("parsing", filename=original_filename, path=str(safe_path))
            text = parse_file(safe_path)

            if not text.strip():
                raise ValueError("Document is empty after parsing")

            logger.info("parsed_success", chars=len(text), filename=original_filename)

            # Step 3: Chunk text
            document_id = f"doc_{asyncio.get_event_loop().time()}_{original_filename[:20]}"
            chunks = chunk_text(text, document_id)

            if not chunks:
                raise ValueError("No chunks generated from document")

            logger.info("chunked", count=len(chunks), doc_id=document_id)

            # Step 4: Embed chunks
            embedder = get_embedder()
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_batch(texts)

            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]

            logger.info("embedded", count=len(chunks), dim=len(embeddings[0]) if embeddings else 0)

            # Step 5: Store in databases
            async with get_session() as session:
                # Create document record
                doc = Document(
                    id=document_id,
                    filename=clean_filename(original_filename),
                    original_filename=original_filename,
                    file_size_bytes=len(content),
                    file_type=file_type,
                    status="indexed",
                    chunk_count=len(chunks),
                )
                session.add(doc)

                # Create chunk records
                for chunk in chunks:
                    chunk_obj = Chunk(
                        id=chunk.id,
                        document_id=document_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        metadata_dict={
                            "doc_id": document_id,
                            "filename": original_filename,
                            "chunk_index": chunk.chunk_index,
                            "strategy": chunk.metadata.get("strategy", "unknown"),
                        },
                    )
                    session.add(chunk_obj)

                await session.commit()
                doc_id = document_id
                logger.info("db_persisted", doc_id=document_id, chunks=len(chunks))

            # Store in vector database
            vector_store = get_vector_store()
            chunk_objects = [
                Chunk(
                    id=c.id,
                    document_id=document_id,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    metadata_dict={
                        "doc_id": document_id,
                        "filename": original_filename,
                        "chunk_index": c.chunk_index,
                        "strategy": c.metadata.get("strategy", "unknown"),
                    },
                )
                for c in chunks
            ]
            await vector_store.upsert_chunks(chunk_objects)
            logger.info("vector_stored", doc_id=document_id, chunks=len(chunks))

            return doc

        except Exception as e:
            logger.error("ingest_failed", filename=original_filename, error=str(e), exc_info=True)
            async with get_session() as session:
                doc = Document(
                    id=document_id or f"doc_failed_{original_filename[:20]}",
                    filename=clean_filename(original_filename),
                    original_filename=original_filename,
                    file_size_bytes=len(content) if 'content' in dir() else 0,
                    file_type=file_type,
                    status="failed",
                    error_message=str(e)[:2048],
                )
                session.add(doc)
                await session.commit()
            raise
        finally:
            # Clean up temp file
            if safe_path.exists():
                safe_path.unlink(missing_ok=True)

    async def list_documents(self):
        """List all documents from metadata database."""
        async with get_session() as session:
            from docmind.db.models import Document
            result = await session.execute(
                __import__("sqlalchemy").select(Document).order_by(Document.created_at.desc())
            )
            documents = result.scalars().all()
            return [doc.to_public_dict() for doc in documents]

    async def get_document(self, doc_id: str) -> Optional[dict]:
        """Get a single document by ID."""
        async with get_session() as session:
            from docmind.db.models import Document
            result = await session.execute(
                __import__("sqlalchemy").select(Document).where(Document.id == doc_id)
            )
            doc = result.scalars().first()
            return doc.to_public_dict() if doc else None

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks."""
        async with get_session() as session:
            from docmind.db.models import Document
            result = await session.execute(
                __import__("sqlalchemy").select(Document).where(Document.id == doc_id)
            )
            doc = result.scalars().first()
            if not doc:
                return False
            await session.delete(doc)
            await session.commit()

        # Remove from vector store
        vector_store = get_vector_store()
        await vector_store.delete_document(doc_id)

        logger.info("document_deleted", doc_id=doc_id)
        return True
