"""
BM25 text search index.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docmind.config import settings
from docmind.logging_config import get_logger

logger = get_logger("docmind.retrieval.bm25")


@dataclass
class BM25Index:
    """
    In-memory BM25 index for hybrid retrieval.
    Built from chunk documents.
    """

    k1: float = 1.2
    b: float = 0.75

    def __init__(self):
        self._documents: Dict[str, str] = {}  # id -> text
        self._doc_ids: List[str] = []
        self._doc_lengths: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._term_freqs: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._idf: Dict[str, float] = {}
        self._expanded: Dict[str, List[str]] = {}

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        self._documents[doc_id] = text
        self._doc_ids.append(doc_id)
        tokens = self._tokenize(text)
        self._doc_lengths[doc_id] = len(tokens)
        self._term_freqs[doc_id] = self._compute_term_freqs(tokens)

        # Rebuild IDF after each addition (simple approach)
        self._recompute_idf()

        logger.debug("bm25_added", doc_id=doc_id, tokens=len(tokens))

    def add_documents(self, docs: List[Tuple[str, str]]) -> None:
        """Batch add documents."""
        for doc_id, text in docs:
            self.add_document(doc_id, text)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text)
        return tokens

    def _compute_term_freqs(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequencies for a document."""
        freq = defaultdict(float)
        for token in tokens:
            freq[token] += 1.0
        return dict(freq)

    def _recompute_idf(self) -> None:
        """Recompute IDF for all terms."""
        n = len(self._doc_ids)
        if n == 0:
            self._idf = {}
            return

        # Count documents containing each term
        doc_count = defaultdict(int)
        for doc_id in self._doc_ids:
            for term in self._term_freqs[doc_id]:
                doc_count[term] += 1

        self._idf = {}
        for term, count in doc_count.items():
            # BM25 IDF formula
            self._idf[term] = math.log((n - count + 0.5) / (count + 0.5) + 1.0)

        self._avgdl = sum(self._doc_lengths.values()) / n if n > 0 else 0.0

    def expand_query(self, query: str) -> List[str]:
        """
        Simple query expansion: include original tokens.
        In production, could use synonym expansion or translation.
        """
        if query in self._expanded:
            return self._expanded[query]

        tokens = self._tokenize(query)
        self._expanded[query] = tokens
        return tokens

    def score(self, query: str, doc_id: str) -> float:
        """
        Compute BM25 score for a query against a document.
        Returns 0 if document not in index.
        """
        if doc_id not in self._documents:
            return 0.0

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        doc_freqs = self._term_freqs.get(doc_id, {})
        doc_length = self._doc_lengths.get(doc_id, 0.0)
        score = 0.0

        for term in query_tokens:
            tf = doc_freqs.get(term, 0.0)
            if tf == 0.0:
                continue

            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue

            # BM25 scoring formula
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_length / max(self._avgdl, 1.0)))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Search the index and return top-k (doc_id, score) pairs.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[str, float]] = []
        for doc_id in self._doc_ids:
            score = self.score(query, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def __len__(self) -> int:
        return len(self._doc_ids)

    def document_count(self) -> int:
        return len(self._doc_ids)


class BM25Storage:
    """
    Persistent BM25 index storage.
    Uses simple JSON files for demo; production would use a proper search index.
    """

    def __init__(self, storage_dir: Path = Path("./data/bm25")):
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._index: Optional[BM25Index] = None

    def get_index(self) -> BM25Index:
        if self._index is None:
            self._index = BM25Index()
            self._load()
        return self._index

    def _load(self) -> None:
        """Load index from disk (simple implementation)."""
        # For demo, we rebuild from scratch each time
        # Production: serialize index to disk
        pass

    def save(self) -> None:
        """Save index to disk."""
        # For demo, no persistence needed
        pass


_default_bm25: Optional[BM25Storage] = None


def get_bm25_storage() -> BM25Storage:
    global _default_bm25
    if _default_bm25 is None:
        _default_bm25 = BM25Storage()
    return _default_bm25


def reset_bm25() -> None:
    global _default_bm25
    _default_bm25 = None
