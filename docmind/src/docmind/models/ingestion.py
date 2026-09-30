"""
Pydantic schemas — ingestion.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from docmind.models.common import BaseResponse


class DocumentUploadResponse(BaseResponse):
    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    status: str
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime


class DocumentListResponse(BaseResponse):
    documents: list[DocumentUploadResponse]
    total: int


class DocumentDeleteResponse(BaseResponse):
    success: bool
    document_id: str
    message: Optional[str] = None
