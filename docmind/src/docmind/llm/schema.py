"""
Structured output schema for LLM responses.
Defines the expected format for query answers.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    REFUSED = "refused"
    PARTIAL = "partial"


class CitationData(BaseModel):
    """A single citation to a source document chunk."""
    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename of the source document")
    chunk_index: int = Field(..., description="Chunk index within the document")
    excerpt: str = Field(..., description="Relevant excerpt supporting the answer")
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity/relevance score"
    )


class QueryAnswer(BaseModel):
    """
    Structured output schema for LLM query responses.
    The LLM must fill this schema for every query.
    """
    answer: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="The answer to the user's question, grounded in the provided context.",
    )
    answer_status: AnswerStatus = Field(
        ...,
        description="Whether the answer is fully answered, refused, or partial.",
    )
    citations: list[CitationData] = Field(
        default_factory=list,
        description="List of citations supporting the answer.",
    )
    thought: str = Field(
        default="",
        max_length=500,
        description="Internal reasoning (not shown to user).",
    )

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, v: list) -> list:
        if not isinstance(v, list):
            return []
        return v[:10]  # Cap at 10 citations

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, v: str) -> str:
        return v.strip()


# JSON Schema for LLM structured output (OpenAI response_format)
QUERY_ANSWER_SCHEMA = {
    "name": "query_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the user's question, grounded in the provided context.",
            },
            "answer_status": {
                "type": "string",
                "enum": ["answered", "refused", "partial"],
                "description": "Answer status.",
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "chunk_index": {"type": "integer"},
                        "excerpt": {"type": "string"},
                        "similarity_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["doc_id", "filename", "chunk_index", "excerpt", "similarity_score"],
                    "additionalProperties": False,
                },
                "description": "List of citations supporting the answer.",
            },

        },
        "required": ["answer", "answer_status", "citations"],
        "additionalProperties": False,
    },
}
