"""
Stage 2 of the LitSynth pipeline: multi-source paper retrieval and dedupe.

For every sub-query produced by :mod:`synthesis.decompose`, this stage fans
out across the configured A3 search providers (arXiv, Semantic Scholar, and
optionally DBLP / Crossref as fallbacks) and merges the results into a single
deduplicated, ranked list of :class:`~tools.schemas.Paper` records.

Design notes:
- Search callables are injected through ``search_callables`` so unit tests
  can plug in deterministic fakes without monkeypatching.
- A single provider raising must not abort retrieval: each call is wrapped in
  a try/except and the failure is logged at WARNING level.
- Deduplication walks a strict priority chain: DOI (lowercased), then the
  arXiv id stripped of any version suffix, then a normalized title.
- Ranking is intentionally a tiny heuristic - the real relevance ranker is a
  later pipeline stage; here we just want a stable, sensible ordering.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import date
from typing import Any

from synthesis.schemas import ResearchQuestion
from tools.schemas import Paper
from tools.search_tools import (
    tool_search_arxiv,
    tool_search_crossref,
    tool_search_dblp,
    tool_search_semantic_scholar,
)

logger = logging.getLogger(__name__)


SearchCallable = Callable[..., list[Paper]]


_DEFAULT_SEARCH_CALLABLES: dict[str, SearchCallable] = {
    "arxiv": tool_search_arxiv,
    "semantic_scholar": tool_search_semantic_scholar,
    "dblp": tool_search_dblp,
    "crossref": tool_search_crossref,
}

_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)
_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _strip_arxiv_version(paper_id: str) -> str:
    """Drop the trailing ``vN`` revision marker from an arXiv-style id."""
    prefix, _, suffix = paper_id.partition(":")
    if not suffix:
        return _ARXIV_VERSION_SUFFIX.sub("", paper_id).lower()
    return f"{prefix.lower()}:{_ARXIV_VERSION_SUFFIX.sub('', suffix)}"


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for dedupe."""
    if not title:
        return ""
    stripped = _PUNCTUATION.sub(" ", title.lower())
    return _WHITESPACE.sub(" ", stripped).strip()


def _dedupe_key(paper: Paper) -> tuple[str, str] | None:
    """
    Compute a dedupe key for ``paper`` using the documented priority chain.

    Returns ``None`` when no usable identifier can be derived; such papers
    are kept as-is (each gets a synthetic unique key).
    """
    if paper.doi:
        doi = paper.doi.strip().lower()
        if doi:
            return ("doi", doi)

    pid = (paper.paper_id or "").strip()
    if pid.lower().startswith("arxiv:") or (paper.api_source or "").lower() == "arxiv":
        stripped = _strip_arxiv_version(pid)
        if stripped:
            return ("arxiv", stripped)

    normalized = _normalize_title(paper.title or "")
    if normalized:
        return ("title", normalized)

    return None


def _abstract_present(paper: Paper) -> bool:
    return bool(paper.abstract and paper.abstract.strip())


def _date_ordinal(d: date | None) -> int:
    return d.toordinal() if isinstance(d, date) else 0


def _sort_key(paper: Paper) -> tuple[int, int, int, str]:
    """Sort: abstract first, citations desc, newest first, title asc."""
    has_abstract = 1 if _abstract_present(paper) else 0
    citations = paper.citation_count if paper.citation_count is not None else 0
    return (
        -has_abstract,
        -int(citations),
        -_date_ordinal(paper.publication_date),
        (paper.title or "").lower(),
    )


def _resolve_callable(
    source: str,
    overrides: dict[str, SearchCallable] | None,
) -> SearchCallable | None:
    """Resolve the search callable for ``source`` from overrides or defaults."""
    if overrides and source in overrides:
        return overrides[source]
    return _DEFAULT_SEARCH_CALLABLES.get(source)


def retrieve_papers(
    rq: ResearchQuestion,
    *,
    per_query_limit: int = 5,
    total_limit: int = 12,
    sources: tuple[str, ...] = ("arxiv", "semantic_scholar"),
    search_callables: dict[str, SearchCallable] | None = None,
) -> list[Paper]:
    """
    Retrieve, deduplicate, and rank papers for every sub-query in ``rq``.

    Args:
        rq: The decomposed research question (see :mod:`synthesis.decompose`).
            Each entry in ``rq.sub_queries`` is issued to every requested
            source.
        per_query_limit: ``max_results`` passed to each provider per call.
        total_limit: Maximum number of papers to keep after dedupe and sort.
        sources: Ordered list of provider keys to query. Unknown keys are
            silently skipped with a warning.
        search_callables: Optional mapping of provider key to callable used
            to override the default A3 search tools (useful for tests).

    Returns:
        A list of :class:`~tools.schemas.Paper` records, deduped and ordered
        by abstract availability, citation count, recency, and title.

    Notes:
        Failures from individual providers are logged and swallowed so that
        one flaky source cannot abort retrieval.
    """
    queries = [q for q in (rq.sub_queries or []) if isinstance(q, str) and q.strip()]
    if not queries and rq.question:
        queries = [rq.question.strip()]

    filters: dict[str, Any] = {"max_results": int(per_query_limit)}

    collected: list[Paper] = []
    for query in queries:
        for source in sources:
            search_fn = _resolve_callable(source, search_callables)
            if search_fn is None:
                logger.warning("No search callable registered for source %r", source)
                continue
            try:
                results = search_fn(query, filters)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Search provider %r failed for sub-query %r: %s",
                    source,
                    query,
                    exc,
                )
                continue
            if results:
                collected.extend(results)

    seen: dict[tuple[str, str], Paper] = {}
    untagged: list[Paper] = []
    for paper in collected:
        key = _dedupe_key(paper)
        if key is None:
            untagged.append(paper)
            continue
        if key not in seen:
            seen[key] = paper

    deduped: list[Paper] = list(seen.values()) + untagged
    deduped.sort(key=_sort_key)
    return deduped[: int(total_limit)]
