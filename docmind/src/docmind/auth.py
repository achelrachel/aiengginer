"""
API key authentication.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from docmind.config import settings
from docmind.logging_config import get_logger

logger = get_logger("docmind.auth")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_header = APIKeyHeader(name="Authorization", auto_error=False)


async def get_api_key(
    request: Request,
    x_api_key: Optional[str] = Security(api_key_header),
    authorization: Optional[str] = Security(bearer_header),
) -> str:
    """
    Validate API key from X-API-Key header or Authorization: Bearer <key>.
    """
    key = x_api_key or (
        authorization.removeprefix("Bearer ").strip() if authorization else None
    )

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header or Authorization: Bearer <key>.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if key != settings.api_key:
        logger.warning("invalid_api_key_attempt", client=request.client.host if request.client else "unknown")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return key
