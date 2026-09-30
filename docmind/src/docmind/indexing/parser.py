"""
Document parser — supports PDF, TXT, MD, DOCX, PPTX.
Returns extracted text with basic metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("docmind.indexing.parser")


def parse_txt(path: Path) -> str:
    """Parse plain text file."""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_md(path: Path) -> str:
    """Parse markdown file (treated as plain text with structure hints)."""
    return parse_txt(path)


def parse_pdf(path: Path) -> str:
    """Parse PDF using PyMuPDF (fitz)."""
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF parsing. Install with: pip install pymupdf")

    doc = fitz.open(path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append(f"\n--- PAGE {page_num + 1} ---\n{text}")
    doc.close()
    return "\n".join(pages)


def parse_docx(path: Path) -> str:
    """Parse DOCX using python-docx."""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        raise RuntimeError("python-docx is required for DOCX parsing. Install with: pip install python-docx")

    doc = DocxDocument(path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def parse_pptx(path: Path) -> str:
    """Parse PPTX using python-pptx."""
    try:
        from pptx import Presentation
    except ImportError:
        raise RuntimeError("python-pptx is required for PPTX parsing. Install with: pip install python-pptx")

    prs = Presentation(path)
    slides = []
    for slide_num, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    txt = para.text.strip()
                    if txt:
                        texts.append(txt)
        if texts:
            slides.append(f"\n--- SLIDE {slide_num} ---\n" + "\n".join(texts))
    return "\n".join(slides)


PARSERS = {
    ".txt": parse_txt,
    ".md": parse_md,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
}


def parse_file(path: Path) -> str:
    """
    Parse a document file based on its extension.
    Raises ValueError if format not supported.
    Raises RuntimeError if required library missing.
    """
    ext = path.suffix.lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {list(PARSERS.keys())}")

    parser = PARSERS[ext]
    try:
        text = parser(path)
    except Exception as e:
        logger.error("parse_failed", path=str(path), error=str(e), exc_info=True)
        raise RuntimeError(f"Failed to parse {path.name}: {e}")

    if not text.strip():
        logger.warning("empty_document", path=str(path))
        raise ValueError(f"Document {path.name} is empty or contains no extractable text")

    return text
