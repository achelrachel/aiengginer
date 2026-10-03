"""
Pydantic schemas — common.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID
from typing import Any, Optional, Optional

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response with from_attributes enabled."""
    model_config = {"from_attributes": True}


class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class ErrorDetail(BaseModel):
    detail: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
    errors: list[ErrorDetail] = Field(default_factory=list)
