"""
Pydantic schemas — query.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from docmind.models.common import BaseResponse


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["Bagaimana prosedur pelaporan insiden keamanan?"],
    )
    top_k: int = Field(default=8, ge=1, le=50, description="Number of source chunks to retrieve")
    filters: Optional[dict] = Field(
        default=None,
        description="Optional metadata filters (e.g. {'filename': 'sop.pdf'})",
    )

    @field_validator("query")
    @classmethod
    def normalize_query(cls, v: str) -> str:
        return v.strip()


class Citation(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    excerpt: str = Field(..., max_length=1500, description="Relevant excerpt from the source chunk")
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    answer_status: str = Field(..., pattern="^(answered|refused|partial)$")
    citations: list[Citation] = Field(default_factory=list)
    retrieval_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    database: str = "connected"
    vector_store: str = "connected"
    llm_provider: str = "available"
    document_count: int = 0
