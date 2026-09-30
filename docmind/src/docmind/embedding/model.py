"""
Embedding model wrapper using sentence-transformers.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from docmind.config import settings
from docmind.logging_config import get_logger

logger = get_logger("docmind.embedding")


class Embedder:
    """Wraps a sentence-transformer model for embedding generation."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self._model_name = model_name or settings.embedding_model
        self._device = device or settings.embedding_device
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("loading_embedding_model", model=self._model_name, device=self._device)
            self._model = SentenceTransformer(self._model_name, device=self._device)
            logger.info("embedding_model_loaded", model=self._model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        embeddings = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return embeddings[0].tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed multiple texts in batches."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


_default_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder


def reset_embedder() -> None:
    global _default_embedder
    _default_embedder = None
