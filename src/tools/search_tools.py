"""
External scholarly search tools: arXiv, DBLP, Semantic Scholar, Crossref.

Each function validates ``filters`` via ``SearchFilters``, applies rate limiting,
and returns normalized ``Paper`` instances. Network failures are logged and
re-raised as ``SearchToolError``.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any
from urllib.parse import quote, urlencode

from config.config import get_settings
from tools.http_client import rate_limited_get
from tools.schemas import Paper, SearchFilters, validate_search_filters

logger = logging.getLogger(__name__)


class SearchToolError(RuntimeError):
    """Raised when a search provider returns an error or unparsable payload."""


_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_OPENSEARCH_NS = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}


def _parse_arxiv_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _arxiv_id_from_entry_id(uri: str) -> str:
    # http://arxiv.org/abs/1234.5678v1 -> 1234.5678v1
    m = re.search(r"arxiv\.org/abs/([^/]+)", uri, re.I)
    return m.group(1) if m else uri


def tool_search_arxiv(query: str, filters: dict[str, Any] | None = None) -> list[Paper]:
    """
    Search arXiv's Atom API by keyword, optional author, category, and submitted date range.

    ``filters`` may include: ``author``, ``category``, ``date_from``, ``date_to``, ``max_results``.

    Returns:
        List of :class:`Paper` entries with ``api_source='arxiv'``.
    """
    f = validate_search_filters(filters)
    settings = get_settings()
    q_parts: list[str] = [f"all:{query}"]
    if f.author:
        q_parts.append(f"au:{f.author}")
    if f.category:
        q_parts.append(f"cat:{f.category}")
    # arXiv uses submittedDateRange for v1 API - format YYYYMMDDTOYYYYMMDDO
    if f.date_from or f.date_to:
        start = f.date_from or date(1991, 1, 1)
        end = f.date_to or date.today()
        ds = start.strftime("%Y%m%d")
        de = end.strftime("%Y%m%d")
        q_parts.append(f"submittedDate:[{ds} TO {de}]")

    search_query = " AND ".join(q_parts)
    if f.date_from or f.date_to:
        sort_by = "submittedDate"
        sort_order = "descending"
    else:
        sort_by = "relevance"
        sort_order = "descending"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": f.max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{settings.arxiv_api_base_url}?{urlencode(params)}"
    logger.info("arXiv search: %s", search_query[:200])

    try:
        resp = rate_limited_get(url, host_key="export.arxiv.org")
    except Exception as exc:
        logger.exception("arXiv request failed")
        raise SearchToolError(f"arXiv request failed: {exc}") from exc

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        raise SearchToolError("arXiv returned invalid XML") from exc

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title_el = entry.find("atom:title", _ATOM_NS)
        title = (title_el.text or "").replace("\n", " ").strip() if title_el is not None else ""
        summary_el = entry.find("atom:summary", _ATOM_NS)
        abstract = (summary_el.text or "").strip() if summary_el is not None else None
        id_el = entry.find("atom:id", _ATOM_NS)
        entry_id = (id_el.text or "").strip() if id_el is not None else ""
        published_el = entry.find("atom:published", _ATOM_NS)
        pub = _parse_arxiv_date(published_el.text if published_el is not None else None)
        authors: list[str] = []
        for a in entry.findall("atom:author", _ATOM_NS):
            n = a.find("atom:name", _ATOM_NS)
            if n is not None and n.text:
                authors.append(n.text.strip())
        pid = _arxiv_id_from_entry_id(entry_id)
        papers.append(
            Paper(
                paper_id=f"arxiv:{pid}",
                title=title,
                authors=authors,
                abstract=abstract,
                url=f"https://arxiv.org/abs/{pid.split('v')[0]}" if pid else entry_id,
                publication_date=pub,
                api_source="arxiv",
                metadata={"arxiv_id": pid, "atom_id": entry_id},
            )
        )
    return papers


def _parse_dblp_date(y: str | None) -> date | None:
    if not y or len(y) < 4:
        return None
    try:
        return date(int(y[:4]), 1, 1)
    except ValueError:
        return None


def tool_search_dblp(query: str, filters: dict[str, Any] | None = None) -> list[Paper]:
    """
    Search DBLP (computer science bibliography) and normalize hits to ``Paper``.

    ``filters``: ``author`` (combined into query), ``max_results`` mapped to ``h`` (hits).
    """
    f = validate_search_filters(filters)
    settings = get_settings()
    q = query
    if f.author:
        q = f"{q} author:{f.author}"
    params = {"q": q, "format": "xml", "h": min(f.max_results, 100)}
    url = f"{settings.dblp_api_base_url}?{urlencode(params)}"
    logger.info("DBLP search: %s", q[:200])
    try:
        resp = rate_limited_get(url, host_key="dblp.org")
    except Exception as exc:
        logger.exception("DBLP request failed")
        raise SearchToolError(f"DBLP request failed: {exc}") from exc

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        raise SearchToolError("DBLP returned invalid XML") from exc

    papers: list[Paper] = []
    for hit in root.findall(".//hit"):
        info = hit.find("info")
        if info is None:
            continue
        title_el = info.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        authors: list[str] = []
        for au in info.findall("authors/author"):
            if au.text:
                authors.append(au.text.strip())
        yr_el = info.find("year")
        year = (yr_el.text or "").strip() if yr_el is not None else ""
        key_el = info.find("key")
        dblp_key = (key_el.text or "").strip() if key_el is not None else ""
        url_el = info.find("ee") or info.find("url")
        link = (url_el.text or "").strip() if url_el is not None else None
        venue_el = info.find("venue")
        venue = (venue_el.text or "").strip() if venue_el is not None else None
        pid = dblp_key or link or title
        papers.append(
            Paper(
                paper_id=f"dblp:{pid}",
                title=title,
                authors=authors,
                abstract=None,
                url=link,
                publication_date=_parse_dblp_date(year or None),
                venue=venue,
                api_source="dblp",
                metadata={"dblp_key": dblp_key},
            )
        )
    return papers


def tool_search_semantic_scholar(
    query: str,
    filters: dict[str, Any] | None = None,
) -> list[Paper]:
    """
    Search Semantic Scholar (free API) and include citation and influential citation counts.

    Optional ``filters``: ``max_results`` (1–100), ``year`` not directly in API — use query text.
    """
    f = validate_search_filters(filters)
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
    url = f"{settings.semantic_scholar_api_base_url}/paper/search"
    params: dict[str, Any] = {"query": query, "limit": limit, "fields": fields}
    logger.info("Semantic Scholar search: %s", query[:120])
    try:
        resp = rate_limited_get(url, params=params, host_key="api.semanticscholar.org")
    except Exception as exc:
        logger.exception("Semantic Scholar request failed")
        raise SearchToolError(f"Semantic Scholar request failed: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise SearchToolError("Semantic Scholar returned non-JSON") from exc

    data = payload.get("data") or []
    papers: list[Paper] = []
    for item in data:
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
        papers.append(
            Paper(
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
                metadata={"s2_paper_id": pid, "raw": item},
            )
        )
    return papers


def _crossref_date(parts: list[int] | None) -> date | None:
    if not parts:
        return None
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        return date(y, m, d)
    except (ValueError, TypeError):
        return None


def _quote_doi_path(doi: str) -> str:
    """Percent-encode a DOI for use in Crossref REST path segments."""
    return quote(doi, safe="")


def tool_search_crossref(query: str, filters: dict[str, Any] | None = None) -> list[Paper]:
    """
    Search Crossref works by free text, DOI, title, or author (via ``filters``).

    ``filters``: ``doi`` (exact lookup shortcut), ``author``, ``max_results``.
    """
    f = validate_search_filters(filters)
    settings = get_settings()
    rows = min(f.max_results, 100)

    if f.doi:
        url = f"{settings.crossref_api_base_url}/works/{_quote_doi_path(f.doi)}"
        try:
            resp = rate_limited_get(url, host_key="api.crossref.org")
            msg = resp.json().get("message") or {}
            return [_crossref_work_to_paper(msg)]
        except Exception as exc:
            logger.exception("Crossref DOI lookup failed")
            raise SearchToolError(f"Crossref DOI lookup failed: {exc}") from exc

    parts = []
    if query:
        parts.append(query)
    if f.author:
        parts.append(f"author:{f.author}")
    q = " ".join(parts) if parts else query
    url = f"{settings.crossref_api_base_url}/works"
    params = {"query": q, "rows": rows}
    logger.info("Crossref search: %s", q[:200])
    try:
        resp = rate_limited_get(url, params=params, host_key="api.crossref.org")
    except Exception as exc:
        logger.exception("Crossref request failed")
        raise SearchToolError(f"Crossref request failed: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise SearchToolError("Crossref returned non-JSON") from exc

    items = (payload.get("message") or {}).get("items") or []
    return [_crossref_work_to_paper(it) for it in items]


def _author_family_given(a: dict[str, Any]) -> str:
    fam = a.get("family") or ""
    giv = a.get("given") or ""
    return f"{giv} {fam}".strip() or fam or giv


def _crossref_work_to_paper(msg: dict[str, Any]) -> Paper:
    title_list = msg.get("title") or []
    title = str(title_list[0]) if title_list else ""
    authors_raw = msg.get("author") or []
    authors = [_author_family_given(x) for x in authors_raw if isinstance(x, dict)]
    issued = (msg.get("issued") or {}).get("date-parts") or []
    pub = _crossref_date(issued[0]) if issued else None
    doi = msg.get("DOI")
    url_list = msg.get("URL") or ""
    link = str(url_list) if url_list else None
    container = msg.get("container-title") or []
    venue = str(container[0]) if container else None
    publisher = msg.get("publisher")
    abstract = msg.get("abstract")
    if isinstance(abstract, str):
        # Crossref sometimes returns JATS tags — strip crudely
        abstract = re.sub(r"<[^>]+>", "", abstract)
    pid = str(doi) if doi else str(msg.get("DOI", title))[:200]
    return Paper(
        paper_id=f"crossref:{pid}",
        title=title,
        authors=authors,
        abstract=str(abstract) if abstract else None,
        url=link,
        publication_date=pub,
        doi=str(doi) if doi else None,
        venue=venue,
        publisher=str(publisher) if publisher else None,
        api_source="crossref",
        metadata={"crossref_type": msg.get("type"), "raw_keys": list(msg.keys())},
    )
