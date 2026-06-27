"""
Stage 9 of the LitSynth pipeline: citation validator.

The generator stage is asked - but cannot be forced - to cite only papers
from the supplied set. This module is the deterministic backstop: it
scans the produced prose for bracketed citations, attempts to resolve
each one against the input paper list using a tolerant matcher, and
returns both the per-citation verdicts and the aggregate hallucination /
confidence signals consumed by :class:`~synthesis.schemas.SynthesisResult`.

The validator is a pure function. It performs no I/O, makes no LLM
calls, and never mutates its inputs.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from synthesis.schemas import CitationCheck, ScoredPaper, assign_citation_keys

logger = logging.getLogger(__name__)


_CITATION_RE = re.compile(
    r"\[("
    r"[A-Z][A-Za-z\-\u00C0-\u017F'.]+"
    r"(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z\-\u00C0-\u017F]+))?"
    r"[\s,]+"
    r"\d{4}[a-z]?"
    r")\]"
)

_WHITESPACE_RE = re.compile(r"\s+")


def _relax(text: str) -> str:
    """
    Normalize a citation fragment for tolerant matching.

    The relaxed form is lowercased, comma-stripped, and folds the
    ``et al.`` token (and its punctuation variants) down to ``etal`` so
    that ``[Smith et al. 2023]`` and ``[Smith, et al, 2023]`` collide.
    """
    s = text.strip().lower()
    s = s.replace(",", " ")
    s = s.replace("et al.", "etal").replace("et al", "etal")
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _build_paper_index(
    papers: list[ScoredPaper],
) -> dict[str, list[str]]:
    """
    Build a relaxed-key -> list[paper_id] map from the candidate papers.

    Keys are assigned via :func:`synthesis.schemas.assign_citation_keys`, which
    disambiguates same-author/year collisions (``2025a``/``2025b``) using the
    same paper list the prompt builder saw. A relaxed key can therefore still
    map to multiple ``paper_id`` values only if the generator cited a
    deliberately ambiguous, non-suffixed key; that case is left at length > 1
    so the resolver treats it as ambiguous (and therefore invalid).
    """
    keys = assign_citation_keys(papers)
    index: dict[str, list[str]] = defaultdict(list)
    for paper in papers:
        key = keys.get(paper.paper_id, paper.short_citation_key())
        inner = key.strip().lstrip("[").rstrip("]")
        relaxed = _relax(inner)
        if not relaxed:
            continue
        if paper.paper_id not in index[relaxed]:
            index[relaxed].append(paper.paper_id)
    return dict(index)


def _extract_inner_citations(review_text: str) -> list[str]:
    """Return the inner text of every bracketed citation in input order."""
    return [m.group(1).strip() for m in _CITATION_RE.finditer(review_text or "")]


def validate_citations(
    review_text: str,
    papers: list[ScoredPaper],
) -> tuple[list[CitationCheck], list[str], list[str], float]:
    """
    Validate the citations that appear in the generated review prose.

    Args:
        review_text: The literature review text produced by the generator
            stage. May be empty or contain zero bracketed citations.
        papers: The paper set the model was told it could cite from;
            their ``short_citation_key()`` values define the legal keys.

    Returns:
        A tuple ``(citation_checks, citations_used, hallucinated,
        confidence_score)``:

        - ``citation_checks``: one :class:`~synthesis.schemas.CitationCheck`
          per unique bracketed citation, in first-appearance order.
        - ``citations_used``: deduplicated list of resolved ``paper_id``
          values in first-appearance order.
        - ``hallucinated``: raw bracketed citation strings (with the
          surrounding brackets) that did not resolve to exactly one paper.
        - ``confidence_score``: ``valid_unique / max(1, total_unique)``
          clamped to ``[0.0, 1.0]``. Returns ``0.0`` when the review text
          contains no bracketed citations.

    Notes:
        Pure function. A citation is considered valid iff its relaxed
        form resolves to exactly one paper; ambiguous matches (e.g. two
        papers with the same author surname and year) are flagged as
        invalid and reported via ``hallucinated``.
    """
    inner_citations = _extract_inner_citations(review_text)
    if not inner_citations:
        return ([], [], [], 0.0)

    paper_index = _build_paper_index(papers)

    checks: list[CitationCheck] = []
    seen_inner: set[str] = set()

    citations_used: list[str] = []
    seen_paper_ids: set[str] = set()

    hallucinated: list[str] = []

    valid_unique = 0
    total_unique = 0

    for inner in inner_citations:
        if inner in seen_inner:
            continue
        seen_inner.add(inner)
        total_unique += 1

        raw_key = f"[{inner}]"
        relaxed = _relax(inner)
        matches = paper_index.get(relaxed, [])

        if len(matches) == 1:
            valid_unique += 1
            resolved = matches[0]
            checks.append(
                CitationCheck(
                    citation_key=raw_key,
                    resolved_paper_id=resolved,
                    is_valid=True,
                )
            )
            if resolved not in seen_paper_ids:
                seen_paper_ids.add(resolved)
                citations_used.append(resolved)
        else:
            if len(matches) > 1:
                logger.warning(
                    "Citation %r resolves to %d papers; marking invalid",
                    raw_key,
                    len(matches),
                )
            checks.append(
                CitationCheck(
                    citation_key=raw_key,
                    resolved_paper_id=None,
                    is_valid=False,
                )
            )
            hallucinated.append(raw_key)

    confidence_score = valid_unique / max(1, total_unique)
    confidence_score = max(0.0, min(1.0, confidence_score))

    return (checks, citations_used, hallucinated, confidence_score)
