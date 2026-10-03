"""
LLM prompt templates and system prompt construction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from docmind.config import settings
from docmind.logging_config import get_logger
from docmind.llm.schema import QueryAnswer

logger = get_logger("docmind.llm.prompt")


def build_system_prompt() -> str:
    """
    Build the system prompt for the RAG query assistant.
    This prompt defines the assistant's behavior, constraints, and output format.
    """
    return f"""<system>
Anda adalah asisten pertanyaan-jawaban yang berbasis pada konteks yang diberikan.
Anda harus merespons pertanyaan pengguna menggunakan HANYA informasi dari konteks yang diberikan.
Tidak boleh menambahkan informasi dari luar konteks.
 Tidak boleh membuat fakta atau angka yang tidak ada di konteks.

<instructions>
1. Baca konteks yang diberikan dengan seksama.
2. Jawab pertanyaan pengguna berdasarkan konteks tersebut.
3. Jika konteks Tidak mengandung informasi yang cukup, jawab dengan jujur bahwa
   informasi tidak tersedia. Jangan membuat jawaban.
4. Jika jawaban hanya sebagian tersedia, berikan yang tersedia dan tunjukkan
   mana yang kurang.
5. Setiap pernyataan yang disertasi harus memiliki citation.
6. Gunakan bahasa yang sama dengan pertanyaan pengguna (Indonesia atau Inggris).

<format>
Output harus mengikuti format JSON yang telah ditentukan:
- answer: jawaban teks (minimal 1 karakter)
- answer_status: "answered" / "refused" / "partial"
- citations: array dari objek citation (doc_id, filename, chunk_index, excerpt, similarity_score)

Output harus VALID JSON, tidak boleh ada teks di luar JSON.
</format>

<constraints>
- Jawaban harus di-grounding pada konteks.
- Tidak boleh hallucinate.
- Jika tidak yakin, jawab "refused" dengan alasan.
- Maximal 3000 karakter untuk jawaban.
- Maksimal 10 citation.
</constraints>
</system>"""


def build_user_prompt(
    query: str,
    context_chunks: List[Dict[str, Any]],
    *,
    include_metadata: bool = True,
) -> str:
    """
    Build the user message with context chunks and the query.

    Args:
        query: User's question
        context_chunks: List of retrieved chunks with metadata
        include_metadata: Whether to include chunk metadata in context

    Returns:
        Formatted user prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        text = chunk.get("text", chunk.get("document", ""))
        metadata = chunk.get("metadata", {})
        doc_id = metadata.get("doc_id", chunk.get("doc_id", "unknown"))
        filename = metadata.get("filename", chunk.get("filename", "unknown"))
        chunk_index = metadata.get("chunk_index", chunk.get("chunk_index", 0))

        context_parts.append(
            f"=== CONTEXT {i} ===\n"
            f"Source: {filename} (doc_id: {doc_id}, chunk: {chunk_index})\n"
            f"Content:\n{text}\n"
            f"=================\n"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""<query>
{query}
</query>

<context>
{context_text}
</context>

<instructions>
Jawablah pertanyaan di atas menggunakan HANYA informasi dari context yang diberikan.
Berikan jawaban yang jelas, akurat, dan disertasi dengan citation yang tepat.
Pastikan output dalam FORMAT JSON yang benar.
</instructions>
"""
    return prompt


def build_messages(
    query: str,
    context_chunks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Build the full message list for LLM chat completion.
    """
    system = build_system_prompt()
    user = build_user_prompt(query, context_chunks)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def estimate_context_tokens(context_chunks: List[Dict[str, Any]]) -> int:
    """Estimate token count for context chunks (rough estimate)."""
    total_chars = sum(len(c.get("text", c.get("document", ""))) for c in context_chunks)
    return max(1, total_chars // 4)


def estimate_prompt_tokens(query: str, context_chunks: List[Dict[str, Any]]) -> int:
    """Estimate total prompt tokens."""
    system_tokens = estimate_context_tokens([{"text": build_system_prompt()}])
    context_tokens = estimate_context_tokens(context_chunks)
    query_tokens = max(1, len(query) // 4)
    return system_tokens + context_tokens + query_tokens
