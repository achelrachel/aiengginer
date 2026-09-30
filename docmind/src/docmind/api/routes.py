"""
API routes for DocMind.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse

from docmind.auth import get_api_key
from docmind.models.ingestion import DocumentDeleteResponse, DocumentListResponse, DocumentUploadResponse
from docmind.models.query import HealthResponse, QueryRequest, QueryResponse
from docmind.rate_limit import limiter
from docmind.services.ingestion import IngestionService
from docmind.services.query import QueryService
from docmind.config import settings
from docmind.logging_config import get_logger
from docmind.observability.request_id import generate_request_id, bind_request_context
from docmind.observability.metrics import Metrics

logger = get_logger("docmind.api.routes")

router = APIRouter()

ingestion_service = IngestionService()
query_service = QueryService()


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    responses={
        401: {"model": None},
        403: {"model": None},
        413: {"model": None},
        422: {"model": None},
    },
)
@limiter.limit(f"{settings.rate_limit_requests // 6}/minute")
async def upload_document(
    request: Request,
    file: UploadFile,
    api_key: str = Depends(get_api_key),
):
    """
    Upload and ingest a document into the knowledge base.

    Supported formats: PDF, TXT, MD, DOCX, PPTX
    Maximum file size: {settings.max_upload_size_mb}MB
    """
    request_id = generate_request_id()
    bind_request_context(request_id, method="POST", path="/api/v1/documents/upload")

    logger.info("upload_started", filename=file.filename, request_id=request_id)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type .{ext} not allowed. Allowed: {', '.join(settings.allowed_extensions)}",
        )

    try:
        document = await ingestion_service.ingest_file(file, file.filename)
        logger.info("upload_success", doc_id=document.id, filename=file.filename)
        return DocumentUploadResponse(
            id=document.id,
            filename=document.original_filename,
            file_type=document.file_type,
            file_size_bytes=document.file_size_bytes,
            status=document.status,
            chunk_count=document.chunk_count,
            error_message=document.error_message,
            created_at=document.created_at,
        )
    except ValueError as e:
        logger.warning("upload_validation_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("upload_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get(
    "/documents",
    response_model=DocumentListResponse,
)
async def list_documents(
    api_key: str = Depends(get_api_key),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """List all ingested documents."""
    request_id = generate_request_id()
    bind_request_context(request_id, method="GET", path="/api/v1/documents")

    documents = await ingestion_service.list_documents()
    total = len(documents)
    start = (page - 1) * page_size
    end = start + page_size

    return DocumentListResponse(
        documents=documents[start:end],
        total=total,
    )


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentUploadResponse,
)
async def get_document(
    doc_id: str,
    api_key: str = Depends(get_api_key),
):
    """Get a specific document by ID."""
    request_id = generate_request_id()
    bind_request_context(request_id, method="GET", path=f"/api/v1/documents/{doc_id}")

    doc = await ingestion_service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentUploadResponse(
        id=doc["id"],
        filename=doc["filename"],
        file_type=doc["file_type"],
        file_size_bytes=doc["file_size_bytes"],
        status=doc["status"],
        chunk_count=doc["chunk_count"],
        error_message=doc.get("error_message"),
        created_at=doc["created_at"],
    )


@router.delete(
    "/documents/{doc_id}",
    response_model=DocumentDeleteResponse,
)
async def delete_document(
    doc_id: str,
    api_key: str = Depends(get_api_key),
):
    """Delete a document and all its chunks."""
    request_id = generate_request_id()
    bind_request_context(request_id, method="DELETE", path=f"/api/v1/documents/{doc_id}")

    success = await ingestion_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDeleteResponse(success=True, document_id=doc_id, message="Document deleted")


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        401: {"model": None},
        429: {"model": None},
    },
)
@limiter.limit(f"{settings.rate_limit_requests // 2}/minute")
async def query_knowledge_base(
    request: Request,
    body: QueryRequest,
    api_key: str = Depends(get_api_key),
):
    """
    Query the knowledge base with a natural language question.

    Returns a grounded answer with citations from the ingested documents.
    """
    request_id = generate_request_id()
    bind_request_context(request_id, method="POST", path="/api/v1/query")

    metrics = Metrics(request_id=request_id)
    logger.info("query_started", query_preview=body.query[:100], request_id=request_id)

    try:
        result = await query_service.query(
            query=body.query,
            top_k=body.top_k,
            filters=body.filters,
            metrics=metrics,
        )
        return result
    except ValueError as e:
        logger.warning("query_validation_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("query_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        database="connected",
        vector_store="connected",
        llm_provider="available",
        document_count=0,
    )
