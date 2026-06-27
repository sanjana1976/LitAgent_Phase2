"""
Download PDFs, cache binaries, and extract plain text with coarse section tags.
"""

from __future__ import annotations

import logging
from typing import Any

from pypdf import PdfReader

from tools.context import get_cache_dir
from tools.file_cache import FileCache
from tools.http_client import rate_limited_get
from tools.text_heuristics import split_sections

logger = logging.getLogger(__name__)


class PDFToolError(RuntimeError):
    """Raised when a PDF cannot be retrieved or parsed."""


def tool_fetch_and_parse_pdf(paper_id: str, url: str) -> dict[str, Any]:
    """
    Download a PDF when ``url`` points to one, extract text, and cache both on disk.

    Args:
        paper_id: Logical identifier reused by analysis tools and caches.
        url: HTTP(S) link to the PDF or HTML landing page (PDF preferred).

    Returns:
        Dictionary with ``full_text`` (str), ``sections`` (dict section_name -> text),
        ``pdf_path`` (optional local path), and ``from_cache`` (bool).

    Raises:
        PDFToolError: for network, non-PDF, or empty extraction failures.
    """
    if not paper_id.strip():
        raise PDFToolError("paper_id must be non-empty")
    if not url.strip():
        raise PDFToolError("url must be non-empty")

    cache = FileCache(get_cache_dir(), namespace="pdf_tool")
    cache_key = f"parse:{paper_id}:{url}"
    cached = cache.get_json(cache_key) or cache.get_json(f"paper:{paper_id}")
    if cached is not None:
        logger.info("PDF parse cache hit for %s", paper_id)
        cached["from_cache"] = True
        return cached

    try:
        resp = rate_limited_get(url, host_key="pdf_fetch")
    except Exception as exc:
        logger.exception("PDF download failed")
        raise PDFToolError(f"Could not download PDF: {exc}") from exc

    content_type = resp.headers.get("content-type", "").lower()
    data = resp.content
    if "pdf" not in content_type and not data.startswith(b"%PDF"):
        raise PDFToolError(f"URL did not return a PDF (content-type={content_type!r})")

    pdf_path = cache.write_bytes_key("pdfs", paper_id + url, data)

    try:
        reader = PdfReader(str(pdf_path))
        text_parts: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text_parts.append(extracted)
        full_text = "\n".join(text_parts)
    except Exception as exc:
        logger.exception("pypdf extraction failed")
        raise PDFToolError(f"PDF parse failed: {exc}") from exc

    if not full_text.strip():
        raise PDFToolError("Extracted PDF text was empty.")

    sections = split_sections(full_text)
    payload = {
        "paper_id": paper_id,
        "source_url": url,
        "full_text": full_text,
        "sections": sections,
        "pdf_path": str(pdf_path),
        "from_cache": False,
    }
    cache.set_json(cache_key, payload)
    # Deterministic alias so analysis tools can resolve without knowing the URL hash key
    cache.set_json(f"paper:{paper_id}", payload)
    return payload
