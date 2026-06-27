"""Resolve extracted paper plain text from the PDF tool cache."""

from __future__ import annotations

from typing import Any

from tools.context import get_cache_dir
from tools.file_cache import FileCache


class PaperTextError(RuntimeError):
    """Raised when no cached extraction exists for a ``paper_id``."""


def load_cached_paper_text(
    paper_id: str,
    full_text_override: str | None,
) -> tuple[str, dict[str, str]]:
    """
    Return ``(full_text, sections)`` from :func:`tools.pdf_tools.tool_fetch_and_parse_pdf`.

    Args:
        paper_id: Same identifier passed to the PDF tool.
        full_text_override: When set (tests), bypass disk cache.
    """
    if full_text_override:
        return full_text_override, {}

    cache = FileCache(get_cache_dir(), namespace="pdf_tool")
    data = cache.get_json(f"paper:{paper_id}")
    if not isinstance(data, dict):
        raise PaperTextError(
            "No cached PDF parse for this paper_id. Run tool_fetch_and_parse_pdf first."
        )
    ft = data.get("full_text")
    sec = data.get("sections") or {}
    if not isinstance(ft, str) or not ft.strip():
        raise PaperTextError("Cached parse missing full_text.")
    return ft, sec if isinstance(sec, dict) else {}
