"""
Query service — orchestrates query processing, retrieval, and LLM response.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from docmind.config import settings
from docmind.db.models import QueryAudit, get_session
from docmind.embedding.model import get_embedder
from docmind.llm.client import get_llm_client
from docmind.llm.prompt import (
    build_messages,
    estimate_prompt_tokens,
)
from docmind.llm.schema import QueryAnswer, QUERY_ANSWER_SCHEMA
from docmind.logging_config import get_logger
from docmind.observability.metrics import Metrics
from docmind.retrieval.bm25 import BM25Storage, get_bm25_storage
from docmind.retrieval.rerank import retrieve_and_rerank
from docmind.security.validation import (
    detect_injection_attempt,
    hash_injection_detected,
    validate_query,
)
from docmind.vectorstore.client import get_vector_store

logger = get_logger("docmind.services.query")


class QueryService:
    """Service for processing user queries with RAG."""

    async def query(
        self,
        query: str,
        top_k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Metrics] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline:
        1. Validate and sanitize query
        2. Detect injection attempts
        3. Hybrid retrieval (BM25 + vector)
        4. Rerank results
        5. Build context
        6. LLM generate answer
        7. Parse and validate structured output
        8. Log audit
        """
        if metrics is None:
            metrics = Metrics(request_id="query")

        metrics.record_stage("validation")
        t0 = time.monotonic()

        # Step 1: Validate query
        try:
            query = validate_query(query, max_length=2000)
        except ValueError as e:
            logger.warning("invalid_query", error=str(e))
            raise

        # Step 2: Check for injection attempts (log but don't block)
        injection_hash = hash_injection_detected(query)
        if injection_hash["suspicious"]:
            logger.warning(
                "potential_injection",
                **{k: v for k, v in injection_hash.items() if k != "suspicious"},
            )

        metrics.record_stage("injection_check")

        # Step 3: Embedding query
        embedder = get_embedder()
        query_embedding = embedder.embed_text(query)
        metrics.record_stage("embedding")

        # Step 4: Vector retrieval
        vector_store = get_vector_store()
        where_filter = None
        if filters:
            where_filter = filters

        vector_results = await vector_store.query(
            query_embedding=query_embedding,
            top_k=max(50, top_k * 5),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        # Convert to list of (doc_id, distance, metadata)
        vector_list = []
        for i, doc_id in enumerate(vector_results[0]):
            if i < len(vector_results[0]):
                vector_list.append((
                    doc_id,
                    vector_results[3][i] if i < len(vector_results[3]) else 0.0,
                    vector_results[2][i] if i < len(vector_results[2]) else {},
                ))

        metrics.record_stage("vector_retrieval")

        # Step 5: BM25 retrieval
        bm25_storage = get_bm25_storage()
        bm25_index = bm25_storage.get_index()
        bm25_results = bm25_index.search(query, top_k=max(50, top_k * 5))
        metrics.record_stage("bm25_retrieval")

        # Step 6: Hybrid retrieval + rerank
        retrieval_results = await retrieve_and_rerank(
            query=query,
            bm25_storage=bm25_storage,
            vector_results=vector_list,
            top_k=top_k,
            allowed_ids=set((await vector_store.get_chunks(where=filters))["ids"]) if filters else None,
        )
        metrics.record_stage("rerank")
        metrics.retrieval_count = len(retrieval_results)

        # Step 7: Build context from top results
        context_chunks = []
        for result in retrieval_results[:top_k]:
            doc_id = result.get("doc_id")
            metadata = result.get("metadata", {})

            # Get full chunk text from vector store
            chunk_text = metadata.get("text", "")

            if chunk_text:
                context_chunks.append({
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": metadata.get("doc_id", doc_id),
                        "filename": metadata.get("filename", "unknown"),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "similarity_score": result.get("combined_score", 0.0),
                    },
                })

        logger.info(
            "context_built",
            chunks=len(context_chunks),
            total_chars=sum(len(c["text"]) for c in context_chunks),
        )
        metrics.record_stage("context_build")

        retrieval_latency_ms = round((time.monotonic() - t0) * 1000, 2)
        if not context_chunks:
            return {
                "answer": "Tidak ada sumber yang sesuai dalam basis pengetahuan.",
                "answer_status": "refused", "citations": [],
                "retrieval_latency_ms": retrieval_latency_ms,
                "total_latency_ms": retrieval_latency_ms,
            }

        # Step 8: LLM generation
        llm_client = get_llm_client()
        messages = build_messages(query, context_chunks)

        estimated_tokens = estimate_prompt_tokens(query, context_chunks)
        logger.info(
            "llm_request",
            estimated_input_tokens=estimated_tokens,
            model=llm_client.model,
        )

        try:
            response = await llm_client.chat_completion(
                messages=messages,
                response_format=QUERY_ANSWER_SCHEMA,
                temperature=0.3,
                max_tokens=settings.llm_max_tokens,
            )
        except Exception as e:
            logger.error("llm_failed", error=str(e))
            # Return graceful error response
            return {
                "answer": "Maaf, terjadi kesalahan saat memproses pertanyaan. Silakan coba lagi.",
                "answer_status": "refused",
                "citations": [],

                "retrieval_latency_ms": round((time.monotonic() - t0) * 1000, 2),
                "llm_latency_ms": None,
                "total_latency_ms": round((time.monotonic() - t0) * 1000, 2),
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_usd": None,
            }

        metrics.record_stage("llm")
        metrics.input_tokens = response.input_tokens
        metrics.output_tokens = response.output_tokens

        # Step 9: Parse structured output
        try:
            parsed = QueryAnswer.model_validate_json(response.content)
        except Exception as e:
            logger.error("parse_failed", error=str(e), raw=response.content[:500])
            # Fallback: try to extract JSON from response
            try:
                json_str = response.content
                if "```" in json_str:
                    json_str = json_str.split("```")[1].split("json")[1] if "```json" in json_str else json_str.split("```")[1]
                parsed = QueryAnswer.model_validate_json(json_str)
            except Exception:
                logger.error("fallback_parse_failed")
                parsed = QueryAnswer(
                    answer="Maaf, saya tidak dapat memproses jawaban dengan format yang benar.",
                    answer_status="refused",
                    citations=[],
                    thought="Gagal memparse output LLM",
                )

        metrics.record_stage("parse")

        # Step 10: Build final response
        citations = []
        sources = {
            (c["metadata"]["doc_id"], c["metadata"]["chunk_index"]): c
            for c in context_chunks
        }
        invalid_citation = False
        for cit in parsed.citations:
            source = sources.get((cit.doc_id, cit.chunk_index))
            if (not source or cit.filename != source["metadata"]["filename"]
                    or not cit.excerpt.strip() or cit.excerpt not in source["text"]):
                invalid_citation = True
                break
            citations.append({
                "doc_id": cit.doc_id,
                "filename": cit.filename,
                "chunk_index": cit.chunk_index,
                "excerpt": cit.excerpt[:1500],
                "similarity_score": cit.similarity_score,
            })

        if invalid_citation or (parsed.answer_status.value != "refused" and not citations):
            parsed = QueryAnswer(
                answer="Jawaban tidak dapat diverifikasi terhadap sumber yang tersedia.",
                answer_status="refused", citations=[],
            )
            citations = []

        total_latency = (time.monotonic() - t0) * 1000

        estimated_cost = None  # Provider pricing is not configured.

        response_data = {
            "answer": parsed.answer,
            "answer_status": parsed.answer_status.value,
            "citations": citations,
            "retrieval_latency_ms": retrieval_latency_ms,
            "llm_latency_ms": round(response.latency_ms or 0, 2),
            "total_latency_ms": round(total_latency, 2),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6) if estimated_cost else None,
        }

        # Step 11: Audit log
        await self._audit_log(
            query=query,
            response_data=response_data,
            retrieval_count=metrics.retrieval_count,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=response.latency_ms,
        )

        logger.info(
            "query_completed",
            answer_status=parsed.answer_status.value,
            citations_count=len(citations),
            total_latency_ms=round(total_latency, 2),
        )

        return response_data

    async def _audit_log(
        self,
        query: str,
        response_data: Dict[str, Any],
        retrieval_count: int,
        retrieval_latency_ms: Optional[float],
        llm_latency_ms: Optional[float],
    ) -> None:
        """Save query audit record."""
        async with get_session() as session:
            audit = QueryAudit(
                query_text=query[:2048],
                query_language="id",
                response_text=response_data.get("answer", "")[:8192],
                answer_status=response_data.get("answer_status", "refused"),
                retrieval_count=retrieval_count,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                total_latency_ms=response_data.get("total_latency_ms"),
                input_tokens=response_data.get("input_tokens"),
                output_tokens=response_data.get("output_tokens"),
                estimated_cost_usd=response_data.get("estimated_cost_usd"),
            )
            session.add(audit)
            await session.commit()
