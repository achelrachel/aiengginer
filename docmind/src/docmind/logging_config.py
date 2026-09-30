"""
Structured JSON logging configuration.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from docmind.config import settings


def configure_logging() -> None:
    """Configure structured logging once at startup."""
    shared_ancestor = {
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer() if not settings.log_json else structlog.processors.JSONRenderer(),
                "foreign_pre_chain": [
                    structlog.contextvars.merge_contextvars,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                ],
            },
        },
        "handlers": {
            "default": {
                "level": settings.log_level.upper(),
                "formatter": "json",
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": settings.log_level.upper(),
                "propagate": False,
            },
        },
    }

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.config.dictConfig(shared_ancestor)


def get_logger(name: str = "docmind") -> structlog.stdlib.BoundLogger:
    """Get a bound logger for the given name."""
    logger = structlog.get_logger(name)
    return logger


def bind_request_id(logger: structlog.stdlib.BoundLogger, request_id: str) -> structlog.stdlib.BoundLogger:
    """Bind a request ID to the logger."""
    return logger.bind(request_id=request_id)


def bind_trace_info(
    logger: structlog.stdlib.BoundLogger,
    *,
    stage: Optional[str] = None,
    doc_id: Optional[str] = None,
    query_id: Optional[str] = None,
) -> structlog.stdlib.BoundLogger:
    """Bind trace metadata for observability."""
    return logger.bind(
        stage=stage,
        doc_id=doc_id,
        query_id=query_id,
    )
