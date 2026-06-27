"""
Stage 3 of the LitSynth pipeline: fetch and parse paper content.

Takes the deduplicated ``Paper`` list produced by retrieval (stage 2) and
upgrades each one into a ``ScoredPaper`` with as much grounded text as we
can recover:

- If the paper's ``url`` plausibly points to a PDF we delegate to the A3
  ``tool_fetch_and_parse_pdf`` tool, which handles HTTP + pypdf + caching.
- arXiv landing pages (``/abs/<id>``) are rewritten to their PDF mirror
  (``/pdf/<id>``) so we don't have to chase HTML.
- Any failure of the PDF tool degrades gracefully to abstract-only text so
  later stages (ranker, claim extractor) always have *something* to work
  with. We never raise out of this module.

Downstream the ranker (stage 4) will fill in ``relevance_score``; here we
emit ``relevance_score=0.0`` as a placeholder.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from synthesis.schemas import ScoredPaper
from tools.pdf_tools import tool_fetch_and_parse_pdf
from tools.schemas import Paper

logger = logging.getLogger(__name__)


PdfTool = Callable[[str, str], dict[str, Any]]


_ARXIV_ABS_RE = re.compile(r"^(https?://arxiv\.org/abs/)([^?#\s]+)", re.IGNORECASE)
_DEFAULT_SECTION_CHAR_LIMIT = 8_000


def _looks_like_pdf_url(url: str) -> bool:
    """Return True when ``url`` is plausibly a PDF (or an arXiv abs page)."""
    if not url:
        return False
    lowered = url.lower()
    if lowered.endswith(".pdf"):
        return True
    if "/pdf/" in lowered:
        return True
    if _ARXIV_ABS_RE.match(url):
        return True
    return False


def _normalize_pdf_url(url: str) -> str:
    """Rewrite arXiv abstract URLs to their PDF mirror; return others unchanged."""
    match = _ARXIV_ABS_RE.match(url)
    if match:
        return f"https://arxiv.org/pdf/{match.group(2)}"
    return url


def _truncate_full_text(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, snapping back to a paragraph break when possible."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[:limit]
    boundary = head.rfind("\n\n")
    if boundary > limit // 2:
        return head[:boundary]
    return head


def _truncate_sections(sections: dict[str, Any], per_section_limit: int) -> dict[str, str]:
    """Coerce raw section dict to ``dict[str, str]`` with per-section length cap."""
    out: dict[str, str] = {}
    for key, value in sections.items():
        if not isinstance(value, str):
            continue
        out[str(key)] = value if len(value) <= per_section_limit else value[:per_section_limit]
    return out


def fetch_and_parse(
    papers: list[Paper],
    *,
    max_full_text_chars: int = 60_000,
    pdf_tool: PdfTool | None = None,
) -> list[ScoredPaper]:
    """
    Convert retrieved ``Paper`` records into ``ScoredPaper`` records with text content.

    For each paper we try to obtain full text by delegating to the A3 PDF tool
    when the URL looks like a PDF (or an arXiv abstract page, which we rewrite
    to the PDF mirror first). On any failure -- including ``PDFToolError``,
    network errors, or empty extractions -- we fall back to the paper's abstract
    so the downstream ranker still has something to score.

    Args:
        papers: Deduplicated retrieval results from stage 2.
        max_full_text_chars: Hard cap on the ``full_text`` length, truncated to
            the nearest paragraph boundary when feasible.
        pdf_tool: Injectable PDF-fetch callable for tests; defaults to
            :func:`tools.pdf_tools.tool_fetch_and_parse_pdf`.

    Returns:
        One ``ScoredPaper`` per input paper, in input order, with
        ``relevance_score=0.0`` (the ranker fills this in next).
    """
    tool: PdfTool = pdf_tool if pdf_tool is not None else tool_fetch_and_parse_pdf
    scored: list[ScoredPaper] = []

    for paper in papers:
        full_text: str = ""
        sections: dict[str, str] = {}
        has_pdf = False

        url = (paper.url or "").strip()
        if url and _looks_like_pdf_url(url):
            fetch_url = _normalize_pdf_url(url)
            try:
                payload = tool(paper.paper_id, fetch_url)
            except Exception as exc:  # noqa: BLE001 -- module contract is "never raise"
                logger.warning(
                    "PDF fetch/parse failed for %s (%s): %s",
                    paper.paper_id,
                    fetch_url,
                    exc,
                )
            else:
                raw_text = payload.get("full_text") if isinstance(payload, dict) else None
                raw_sections = payload.get("sections") if isinstance(payload, dict) else None
                if isinstance(raw_text, str) and raw_text.strip():
                    full_text = _truncate_full_text(raw_text, max_full_text_chars)
                    has_pdf = True
                    if isinstance(raw_sections, dict):
                        sections = _truncate_sections(raw_sections, _DEFAULT_SECTION_CHAR_LIMIT)
                else:
                    logger.warning(
                        "PDF tool returned empty text for %s; falling back to abstract",
                        paper.paper_id,
                    )

        if not has_pdf:
            full_text = paper.abstract or ""
            sections = {}

        if has_pdf:
            text_tier = "full_text"
        elif paper.abstract and paper.abstract.strip():
            text_tier = "abstract"
        else:
            text_tier = "none"

        year: int | None = None
        if paper.publication_date is not None:
            year = paper.publication_date.year

        scored.append(
            ScoredPaper(
                paper_id=paper.paper_id,
                title=paper.title,
                authors=list(paper.authors or []),
                abstract=paper.abstract,
                year=year,
                venue=paper.venue,
                url=paper.url,
                api_source=paper.api_source,
                sections=sections,
                full_text=full_text,
                has_pdf=has_pdf,
                text_tier=text_tier,
                relevance_score=0.0,
            )
        )

    return scored
