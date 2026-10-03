"""
DocMind — KnowledgeBase QA API.

Production-like demo: RAG pipeline with hybrid retrieval,
structured LLM output, observability, security controls.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from docmind.config import settings
from docmind.logging_config import configure_logging, get_logger
from docmind.db.models import init_db, close_db
from docmind.rate_limit import setup_rate_limit
from docmind.api.routes import router as api_router

logger = get_logger("docmind.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    configure_logging()
    logger.info("startup", env=settings.app_env, debug=settings.debug)
    await init_db()
    from sqlalchemy import select
    from docmind.db.models import Chunk, Document, get_session
    from docmind.retrieval.bm25 import get_bm25_storage
    async with get_session() as session:
        rows = (await session.execute(
            select(Chunk.id, Chunk.text).join(Document).where(Document.status == "indexed")
        )).all()
    get_bm25_storage().get_index().add_documents(list(rows))
    setup_rate_limit(app)
    logger.info("startup_complete")
    try:
        yield
    finally:
        from docmind.llm.client import close_llm_client
        await close_llm_client()
        logger.info("shutdown_started")
        await close_db()
        logger.info("shutdown_complete")


app = FastAPI(
    title="DocMind",
    description=(
        "KnowledgeBase QA dengan RAG, structured output, dan retrieval observability. "
        "Portfolio project — production-like demo."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "check": "liveness",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "DocMind",
        "description": "KnowledgeBase QA dengan RAG",
        "docs": "/docs",
        "health": "/health",
    }
