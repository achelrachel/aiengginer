"""
Security: input validation, sanitization, file handling.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urlparse

from docmind.config import settings
from docmind.logging_config import get_logger

logger = get_logger("docmind.security")


ALLOWED_EXTENSIONS: Set[str] = set(settings.allowed_extensions)
MAX_UPLOAD_SIZE: int = settings.max_upload_size_mb * 1024 * 1024

SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.[a-zA-Z0-9]+$")

# Patterns that may indicate prompt injection attempts
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|prior|above)\s+(instructions|directions|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"from\s+now\s+on\s+", re.IGNORECASE),
    re.compile(r"system\s*:", re.IGNORECASE),
    re.compile(r"<\s*system", re.IGNORECASE),
    re.compile(r"<\s*directive", re.IGNORECASE),
    re.compile(r"[\s\S]*?ignore\s+the\s+above", re.IGNORECASE),
]


def sanitize_text(text: str, max_length: int = 4000) -> str:
    """
    Sanitize user input text.
    - Normalize unicode
    - Limit length
    - Trim
    """
    if not isinstance(text, str):
        return ""
    # Normalize to NFC
    text = unicodedata.normalize("NFC", text)
    # Trim
    text = text.strip()
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    return text


def validate_query(text: str, max_length: int = 2000) -> str:
    """
    Validate and sanitize a user query.
    Returns sanitized text or raises ValueError.
    """
    text = sanitize_text(text, max_length)
    if not text:
        raise ValueError("Query cannot be empty")
    if len(text) < 2:
        raise ValueError("Query too short")
    return text


def detect_injection_attempt(text: str) -> bool:
    """
    Detect potential prompt injection patterns in text.
    Returns True if suspicious patterns found.
    """
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def hash_injection_detected(text: str) -> dict:
    """
    Return metadata about injection detection without logging the actual text.
    """
    return {
        "suspicious": detect_injection_attempt(text),
        "length": len(text),
        "has_newline": "\n" in text,
        "has_instruction_patterns": any(p.search(text) for p in INJECTION_PATTERNS),
    }


ALLOWED_MIME_TYPES = {
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".pptx": {"application/vnd.openxmlformats-officedocument.presentationml.presentation"},
}

MAGIC_BYTES = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04"],
    ".pptx": [b"PK\x03\x04"],
}


def validate_file_type(file_path: Path) -> tuple[bool, str]:
    """
    Validate file type by both extension and magic bytes.
    Returns (is_valid, reason).
    """
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Extension '{ext}' not allowed"

    if ext in MAGIC_BYTES:
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
            for magic in MAGIC_BYTES[ext]:
                if header.startswith(magic):
                    return True, "OK"
            return False, f"File header does not match {ext} signature"
        except Exception as e:
            return False, f"Error reading file: {e}"

    return True, "OK"


def secure_upload_path(filename: str) -> Path:
    """
    Generate a secure path for uploaded files.
    Uses UUID-based naming to prevent path traversal.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".tmp"
    safe_name = f"{uuid.uuid4()}{ext}"
    return settings.upload_dir / safe_name


def is_safe_path(path: Path, base: Path) -> bool:
    """
    Check if a path is safely within base directory (no path traversal).
    """
    try:
        resolved = path.resolve()
        base_resolved = base.resolve()
        return str(resolved).startswith(str(base_resolved))
    except Exception:
        return False


def clean_filename(filename: str) -> str:
    """
    Sanitize a filename to remove potentially dangerous characters.
    """
    # Remove path components
    filename = os.path.basename(filename)
    # Normalize
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")
    # Keep only safe chars
    filename = re.sub(r"[^\w\s.-]", "", filename)
    filename = re.sub(r"\s+", "_", filename)
    if not filename:
        filename = "unnamed"
    return filename[:255]


import uuid
