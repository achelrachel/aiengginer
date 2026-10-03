"""
Document chunker — structure-aware chunking with recursive fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from docmind.config import settings
from docmind.logging_config import get_logger

logger = get_logger("docmind.indexing.chunker")


@dataclass
class Chunk:
    """A single document chunk with metadata."""
    id: str
    text: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


# Pattern untuk section headers dalam dokumen (bahasa Indonesia & Inggris)
SECTION_PATTERNS = [
    re.compile(r"^(#+\s|={2,}\s|---\s|KALIMAT\s|Bagian\s|Bagian\s|Pasal\s|Artikel\s|Ayat\s|Point\s|\d+\.\s|[A-Z][a-z]+:\s)", re.MULTILINE),
    re.compile(r"\n#{1,3}\s+.+\n", re.MULTILINE),
    re.compile(r"\n{2,}\s*[A-Z][a-z]+:\s*.+\n", re.MULTILINE),
]


def estimate_tokens(text: str) -> int:
    """Estimate token count (approximate: 1 token ≈ 4 characters for Indo/English mix)."""
    return max(1, len(text) // 4)


def structure_aware_split(text: str, document_id: str) -> list[Chunk]:
    """
    Split text by structural boundaries (headers, paragraphs).
    Prioritizes semantic units over arbitrary character splits.
    """
    chunks = []
    chunk_id = 0

    # Split by double newlines first (paragraph boundaries)
    paragraphs = re.split(r"\n\s*\n", text)

    current_chunk_parts = []
    current_size = 0
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = estimate_tokens(para)

        # If paragraph alone exceeds chunk size, split it
        if para_tokens > settings.chunk_size_tokens:
            # Split long paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                s_tokens = estimate_tokens(sentence)
                if current_size + s_tokens > settings.chunk_size_tokens and current_chunk_parts:
                    # Emit current chunk
                    chunk_text = "\n\n".join(current_chunk_parts).strip()
                    if chunk_text:
                        chunks.append(Chunk(
                            id=f"{document_id}_chunk_{chunk_id}",
                            text=chunk_text,
                            document_id=document_id,
                            chunk_index=chunk_index,
                            start_char=0,
                            end_char=len(chunk_text),
                            metadata={"strategy": "structure_aware"},
                        ))
                        chunk_id += 1
                        chunk_index += 1
                    current_chunk_parts = []
                    current_size = 0
                current_chunk_parts.append(sentence)
                current_size += s_tokens
        else:
            # Normal paragraph
            if current_size + para_tokens > settings.chunk_size_tokens and current_chunk_parts:
                chunk_text = "\n\n".join(current_chunk_parts).strip()
                if chunk_text:
                    chunks.append(Chunk(
                        id=f"{document_id}_chunk_{chunk_id}",
                        text=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        start_char=0,
                        end_char=len(chunk_text),
                        metadata={"strategy": "structure_aware"},
                    ))
                    chunk_id += 1
                    chunk_index += 1
                current_chunk_parts = []
                current_size = 0
            current_chunk_parts.append(para)
            current_size += para_tokens

    # Emit remaining
    if current_chunk_parts:
        chunk_text = "\n\n".join(current_chunk_parts).strip()
        if chunk_text:
            chunks.append(Chunk(
                id=f"{document_id}_chunk_{chunk_id}",
                text=chunk_text,
                document_id=document_id,
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                metadata={"strategy": "structure_aware"},
            ))

    logger.info("structure_chunks", count=len(chunks), doc_id=document_id)
    return chunks


def recursive_character_split(text: str, document_id: str) -> list[Chunk]:
    """
    Fallback: split by character count with overlap.
    Used when structure-aware splitting produces too few/many chunks.
    """
    chunk_size = settings.chunk_size_tokens * 4  # approx characters
    overlap = settings.chunk_overlap_tokens * 4

    chunks = []
    chunk_id = 0
    chunk_index = 0

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(Chunk(
                id=f"{document_id}_chunk_{chunk_id}",
                text=chunk_text,
                document_id=document_id,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
                metadata={"strategy": "recursive_character", "start_char": start, "end_char": end},
            ))
            chunk_id += 1
            chunk_index += 1

        if end >= len(text):
            break

        start = end - overlap

    logger.info("recursive_chunks", count=len(chunks), doc_id=document_id)
    return chunks


def chunk_text(text: str, document_id: str) -> list[Chunk]:
    """
    Main chunking entry point.
    Uses structure-aware splitting, falls back to recursive if needed.
    """
    if not text.strip():
        return []

    strategy_chunks = structure_aware_split(text, document_id)

    # If structure-aware produced 0 chunks or too many (>100), fallback
    if (not strategy_chunks or len(strategy_chunks) > 100
            or any(len(c.text) > settings.chunk_size_tokens * 4 for c in strategy_chunks)):
        logger.info("fallback_to_recursive", doc_id=document_id, reason=len(strategy_chunks))
        return recursive_character_split(text, document_id)

    return strategy_chunks
