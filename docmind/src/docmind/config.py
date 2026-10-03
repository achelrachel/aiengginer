"""
DocMind application configuration.
Semua setting dibaca dari environment variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    api_key: str = Field(default="dev-key-change-me", alias="API_KEY")

    rate_limit_requests: int = Field(default=60, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")

    max_upload_size_mb: int = Field(default=10, alias="MAX_UPLOAD_SIZE_MB")
    allowed_extensions: list[str] = Field(
        default=["pdf", "txt", "md", "docx", "pptx"],
        alias="ALLOWED_EXTENSIONS",
    )

    db_url: str = Field(default="sqlite+aiosqlite:///./data/docmind.db", alias="DATABASE_URL")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    chroma_db_dir: Path = Field(default=Path("./data/chroma"), alias="CHROMADB_DIR")

    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_MODEL")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")

    bm25_top_k: int = Field(default=50, alias="BM25_TOP_K")
    vector_top_k: int = Field(default=50, alias="VECTOR_TOP_K")
    rerank_top_k: int = Field(default=8, alias="RERANK_TOP_K")
    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")

    chunk_size_tokens: int = Field(default=400, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=60, alias="CHUNK_OVERLAP_TOKENS")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    upload_dir: Path = Field(default=Path("./data/uploads"), alias="UPLOAD_DIR")

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        return v

    @field_validator("chroma_db_dir", "data_dir", "upload_dir", mode="after")
    @classmethod
    def ensure_dirs(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def validate_security_and_chunking(self):
        if not self.api_key.strip() or (self.is_production() and self.api_key == "dev-key-change-me"):
            raise ValueError("Set a non-default API_KEY before production use")
        if self.chunk_size_tokens <= 0 or not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("Chunk overlap must be smaller than the positive chunk size")
        return self

    def is_production(self) -> bool:
        return self.app_env.strip().lower() == "production"

    def is_development(self) -> bool:
        return self.app_env.strip().lower() == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
