"""
Rate limiting setup using slowapi.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from docmind.config import settings
from docmind.logging_config import get_logger

logger = get_logger("docmind.rate_limit")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_requests}/{settings.rate_limit_window_seconds} seconds"],
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # HTTP Retry-After must be delay-seconds or a valid HTTP date (RFC 9110).
    # SlowAPI's exc.detail is a human-readable limit description, not a delay.
    # Use a conservative bound because both the global limit and endpoint
    # overrides must have had time to reset before a client retries.
    retry_after_seconds = max(60, settings.rate_limit_window_seconds)
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded. Please try again later.",
            "retry_after": exc.detail,
            "retry_after_seconds": retry_after_seconds,
        },
        headers={"Retry-After": str(retry_after_seconds)},
    )


def setup_rate_limit(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
