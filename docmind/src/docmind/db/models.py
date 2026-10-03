"""
Database models and session management.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional
import uuid

from contextlib import asynccontextmanager
from sqlalchemy import BigInteger, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from docmind.config import settings


class Base(DeclarativeBase):
    """Base for all SQLAlchemy models."""
    pass


class Document(Base):
    """Represents an uploaded document in the knowledge base."""
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    chunk_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.original_filename,
            "file_size_bytes": self.file_size_bytes,
            "file_type": self.file_type,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat(),
        }


class Chunk(Base):
    """A single chunk of a document stored in the vector database."""
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    text: Mapped[str] = mapped_column(String(8192), nullable=False)
    metadata_json: Mapped[str] = mapped_column(String(2048), nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    document = relationship("Document", back_populates="chunks")

    @property
    def metadata_dict(self) -> dict:
        import json
        try:
            return json.loads(self.metadata_json)
        except Exception:
            return {}

    @metadata_dict.setter
    def metadata_dict(self, value: dict) -> None:
        import json
        self.metadata_json = json.dumps(value, ensure_ascii=False)


class QueryAudit(Base):
    """Audit log for each query to the knowledge base."""
    __tablename__ = "query_audit"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_text: Mapped[str] = mapped_column(String(2048), nullable=False)
    query_language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    response_text: Mapped[str] = mapped_column(String(8192), nullable=False)
    answer_status: Mapped[str] = mapped_column(String(32), nullable=False)
    retrieval_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    retrieval_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    llm_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    estimated_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "query_text": self.query_text,
            "query_language": self.query_language,
            "response_text": self.response_text,
            "answer_status": self.answer_status,
            "retrieval_count": self.retrieval_count,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "created_at": self.created_at.isoformat(),
        }


class Engine:
    """Async database engine and session factory."""

    def __init__(self):
        self._engine = None
        self._session_factory = None

    def get_engine(self):
        if self._engine is None:
            self._engine = create_async_engine(
                settings.db_url,
                echo=settings.db_echo,
                future=True,
            )
        return self._engine

    def get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.get_engine(),
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._session_factory

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None


_engine = Engine()


@asynccontextmanager
async def get_session():
    """Yield an async session for dependency injection."""
    factory = _engine.get_session_factory()
    async with factory() as session:
        yield session


async def close_db():
    await _engine.close()


async def init_db():
    """Create all tables."""
    async with _engine.get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
