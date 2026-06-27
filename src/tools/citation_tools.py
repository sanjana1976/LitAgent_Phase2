"""
Forward-citation lookup via Semantic Scholar (papers that cite a given work).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from config.config import get_settings
from db.queries import get_paper_by_id, row_to_metadata_dict
from tools.context import get_default_database
from tools.http_client import rate_limited_get
from tools.schemas import Paper, SearchFilters, validate_search_filters
from tools.search_tools import SearchToolError

logger = logging.getLogger(__name__)


class CitationToolError(RuntimeError):
    """Raised when forward-citation lookup cannot be completed."""


def _s2_item_to_paper(item: dict[str, Any]) -> Paper:
    pid = str(item.get("paperId") or "")
    title = str(item.get("title") or "")
    authors_raw = item.get("authors") or []
    authors = [a.get("name", "").strip() for a in authors_raw if isinstance(a, dict)]
    abstract = item.get("abstract")
    url_p = item.get("url")
    venue = item.get("venue")
    cit = item.get("citationCount")
    infl = item.get("influentialCitationCount")
    pub_date_s = item.get("publicationDate")
    pub_dt: date | None = None
    if isinstance(pub_date_s, str) and len(pub_date_s) >= 10:
        try:
            pub_dt = date.fromisoformat(pub_date_s[:10])
        except ValueError:
            pub_dt = None
    ext = item.get("externalIds") or {}
    doi = ext.get("DOI") if isinstance(ext, dict) else None
    return Paper(
        paper_id=f"s2:{pid}",
        title=title,
        authors=[x for x in authors if x],
        abstract=str(abstract) if abstract is not None else None,
        url=str(url_p) if url_p else None,
        publication_date=pub_dt,
        doi=str(doi) if doi else None,
        venue=str(venue) if venue else None,
        citation_count=int(cit) if cit is not None else None,
        influential_citations=int(infl) if infl is not None else None,
        api_source="semantic_scholar",
        metadata={"s2_paper_id": pid, "citation_direction": "forward"},
    )


def resolve_semantic_scholar_paper_id(paper_id: str) -> str:
    """
    Map a local or external ``paper_id`` to a Semantic Scholar API identifier.

    Accepts ``s2:…``, ``arxiv:…``, numeric SQLite ids, DOI strings, and
    ``crossref:…`` prefixes.
    """
    normalized = paper_id.strip()
    if normalized.startswith("s2:"):
        return normalized[3:]
    if normalized.startswith("arxiv:"):
        aid = normalized[6:].split("v")[0]
        return f"ARXIV:{aid}"
    if normalized.startswith("crossref:"):
        return f"DOI:{normalized[9:]}"
    if normalized.lower().startswith("doi:"):
        return f"DOI:{normalized[4:].strip()}"
    if "/" in normalized and normalized.startswith("10."):
        return f"DOI:{normalized}"

    if normalized.isdigit():
        row = get_paper_by_id(get_default_database(), int(normalized))
        if not row:
            raise CitationToolError(f"No paper in database with id={normalized!r}")
        meta = row_to_metadata_dict(row)
        s2_id = meta.get("s2_paper_id")
        if s2_id:
            return str(s2_id)
        raw = meta.get("raw")
        if isinstance(raw, dict) and raw.get("paperId"):
            return str(raw["paperId"])
        doi = row.get("doi")
        if doi:
            return f"DOI:{doi}"
        api_src = str(row.get("api_source") or "")
        if api_src == "arxiv":
            import re

            m = re.search(r"arxiv[:\s]*([0-9]+\.[0-9]+)", json.dumps(meta), re.I)
            if m:
                return f"ARXIV:{m.group(1)}"
        raise CitationToolError(
            f"Paper {normalized} has no Semantic Scholar id or DOI for citation lookup."
        )

    return normalized


def tool_lookup_forward_citations(
    paper_id: str,
    filters: dict[str, Any] | None = None,
) -> list[Paper]:
    """
    Find papers that cite the given work (forward citations / "who built on this?").

  ``paper_id`` may be a Semantic Scholar id (``s2:…``), arXiv id, DOI, Crossref id,
    or a numeric local database id when the row has S2 or DOI metadata.

    ``filters``: ``max_results`` (1–100, default 10).
    """
    f = validate_search_filters(filters)
    s2_id = resolve_semantic_scholar_paper_id(paper_id)
    settings = get_settings()
    fields = ",".join(
        [
            "paperId",
            "title",
            "authors",
            "year",
            "abstract",
            "url",
            "citationCount",
            "influentialCitationCount",
            "publicationDate",
            "externalIds",
            "venue",
        ]
    )
    limit = min(f.max_results, 100)
    url = f"{settings.semantic_scholar_api_base_url}/paper/{s2_id}/citations"
    params: dict[str, Any] = {"fields": fields, "limit": limit}
    logger.info("Semantic Scholar forward citations for %s", s2_id[:80])
    try:
        resp = rate_limited_get(url, params=params, host_key="api.semanticscholar.org")
    except Exception as exc:
        logger.exception("Semantic Scholar citations request failed")
        raise CitationToolError(f"Semantic Scholar citations request failed: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise CitationToolError("Semantic Scholar returned non-JSON") from exc

    data = payload.get("data") or []
    papers: list[Paper] = []
    for edge in data:
        citing = edge.get("citingPaper") if isinstance(edge, dict) else None
        if isinstance(citing, dict) and citing.get("paperId"):
            papers.append(_s2_item_to_paper(citing))
    return papers
