"""
Request ID generation and logging context binding.
"""
from __future__ import annotations

import uuid
from typing import Optional

from docmind.logging_config import bind_request_id, get_logger

logger = get_logger("docmind.observability.request_id")


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def bind_request_context(
    request_id: str,
    *,
    method: Optional[str] = None,
    path: Optional[str] = None,
    client_host: Optional[str] = None,
) -> None:
    """Bind request context to the logging contextvars."""
    import structlog.contextvars
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=method,
        path=path,
        client_host=client_host,
    )
