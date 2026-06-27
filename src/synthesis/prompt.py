"""
Stage 7 of the LitSynth pipeline: synthesis prompt builder.

This module turns the ranked papers, grounded claims, and detected
contradictions into a single deterministic prompt pair (``system`` +
``user``) for the literature-review generator. The builder performs no I/O
and never calls an LLM, which keeps it fully unit-testable and lets
upstream callers reason about cost (prompt size is hard-bounded by
per-field truncation).

The output is a frozen :class:`SynthesisPrompt` so downstream stages cannot
accidentally mutate the prompt while it is being sent to the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from synthesis.schemas import (
    ClaimRecord,
    ContradictionPair,
    ScoredPaper,
    assign_citation_keys,
)

logger = logging.getLogger(__name__)


_ABSTRACT_MAX_CHARS = 400
_CLAIM_MAX_CHARS = 240
_EXPLANATION_MAX_CHARS = 240
_AUTHORS_PREVIEW = 3


@dataclass(frozen=True)
class SynthesisPrompt:
    """Immutable bundle of the system / user prompt and the legal citation keys."""

    system: str
    user: str
    expected_citations: list[str]


def _truncate(text: str | None, limit: int) -> str:
    """Trim ``text`` to at most ``limit`` characters with an ellipsis suffix."""
    if not text:
        return ""
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "\u2026"


def _format_authors(authors: list[str]) -> str:
    """Render ``Authors: A, B, C[ et al.]`` from the first few author names."""
    if not authors:
        return "Unknown"
    head = [a.strip() for a in authors[:_AUTHORS_PREVIEW] if isinstance(a, str) and a.strip()]
    if not head:
        return "Unknown"
    suffix = " et al." if len(authors) > _AUTHORS_PREVIEW else ""
    return ", ".join(head) + suffix


def _expected_citations(papers: list[ScoredPaper], keys: dict[str, str]) -> list[str]:
    """Return the unique, collision-disambiguated citation keys in paper order."""
    seen: set[str] = set()
    out: list[str] = []
    for paper in papers:
        if paper.paper_id in seen:
            continue
        seen.add(paper.paper_id)
        key = keys.get(paper.paper_id, paper.short_citation_key())
        out.append(key)
    return out


def _build_system(word_budget: int) -> str:
    """Return the system instruction describing tone, structure, and rules."""
    return (
        "You are writing the Related Work / Literature Review section of an "
        "academic paper. Your job is to synthesize the provided sources into "
        "a coherent narrative that cites every empirical claim. Follow these "
        "rules strictly:\n"
        "1. Cite every claim using the bracketed [Author et al. YEAR] keys "
        "EXACTLY as provided in the paper list below. Do not invent, rename, "
        "abbreviate, or merge citation keys.\n"
        "2. Do not invent papers, authors, results, or numbers that are not "
        "present in the supplied material. If the evidence is thin, hedge.\n"
        "3. Surface contradictions explicitly: when two papers disagree, name "
        "both with their citation keys and describe the tension in one or two "
        "sentences.\n"
        "4. Group the discussion into 2-4 themes (e.g. methods, datasets, "
        "criticisms, recent advances). Use short topical paragraphs.\n"
        f"5. Target roughly {int(word_budget)} words across 3-5 paragraphs. "
        "Prefer concise, evidence-led prose over filler.\n"
        "6. Tag any UNGROUNDED claims (those marked as such below) with a "
        "hedging phrase such as 'reportedly' or 'according to the authors' "
        "and still cite the source.\n"
        "Return only the literature-review prose. Do not include a title, "
        "preamble, JSON, or a separate references list."
    )


def _format_papers_block(papers: list[ScoredPaper], keys: dict[str, str]) -> str:
    """Render the numbered paper list section of the user prompt."""
    if not papers:
        return "(no papers provided)"
    lines: list[str] = []
    for idx, paper in enumerate(papers, start=1):
        key = keys.get(paper.paper_id, paper.short_citation_key())
        year = paper.year if paper.year is not None else "n.d."
        source = paper.api_source or "unknown source"
        authors = _format_authors(paper.authors)
        header = (
            f"{idx}. {key} - {paper.title} ({source}, {year}). "
            f"Authors: {authors}."
        )
        lines.append(header)
        if paper.abstract:
            abstract = _truncate(paper.abstract, _ABSTRACT_MAX_CHARS)
            if abstract:
                lines.append(f"   Abstract: {abstract}")
    return "\n".join(lines)


def _format_claims_block(
    papers: list[ScoredPaper], claims: list[ClaimRecord], keys: dict[str, str]
) -> str:
    """Render claims grouped by paper, flagging non-grounded entries."""
    if not claims:
        return "(no claims extracted)"

    key_by_paper_id = keys
    grouped: dict[str, list[ClaimRecord]] = {}
    order: list[str] = []
    for claim in claims:
        if claim.paper_id not in grouped:
            grouped[claim.paper_id] = []
            order.append(claim.paper_id)
        grouped[claim.paper_id].append(claim)

    lines: list[str] = []
    for paper_id in order:
        key = key_by_paper_id.get(paper_id, f"[{paper_id}]")
        lines.append(f"{key} ({paper_id}):")
        for claim in grouped[paper_id]:
            body = _truncate(claim.claim, _CLAIM_MAX_CHARS)
            tag = "" if claim.grounded else " (UNGROUNDED)"
            lines.append(f"  - {body}{tag}")
    return "\n".join(lines)


def _format_contradictions_block(
    papers: list[ScoredPaper],
    contradictions: list[ContradictionPair],
    keys: dict[str, str],
) -> str:
    """Render each contradiction with both papers' citation keys."""
    if not contradictions:
        return "(no contradictions detected)"

    key_by_paper_id = keys
    lines: list[str] = []
    for pair in contradictions:
        key_a = key_by_paper_id.get(pair.paper_a, f"[{pair.paper_a}]")
        key_b = key_by_paper_id.get(pair.paper_b, f"[{pair.paper_b}]")
        explanation = _truncate(pair.explanation, _EXPLANATION_MAX_CHARS)
        lines.append(
            f"- {key_a} vs {key_b} [{pair.tension_type}]: {explanation}"
        )
        lines.append(f"    {key_a} claim: {_truncate(pair.claim_a, _CLAIM_MAX_CHARS)}")
        lines.append(f"    {key_b} claim: {_truncate(pair.claim_b, _CLAIM_MAX_CHARS)}")
    return "\n".join(lines)


def _build_user(
    *,
    question: str,
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
    keys: dict[str, str],
    expected_citations: list[str],
    word_budget: int,
) -> str:
    """Assemble the user-side prompt body in the required section order."""
    citation_list = ", ".join(expected_citations) if expected_citations else "(none)"
    parts: list[str] = []
    parts.append(f"Research question:\n{question.strip()}")
    parts.append("Papers (use EXACTLY these citation keys):\n" + _format_papers_block(papers, keys))
    parts.append("Grounded claims by paper:\n" + _format_claims_block(papers, claims, keys))
    parts.append(
        "Contradictions:\n" + _format_contradictions_block(papers, contradictions, keys)
    )
    parts.append(
        "Final instructions:\n"
        f"- Write 3-5 paragraphs totaling approximately {int(word_budget)} words.\n"
        "- Group the discussion into 2-4 themes.\n"
        "- Cite every claim using the bracketed [Author et al. YEAR] keys "
        "EXACTLY as listed above. Do not invent citations not in the list "
        "above.\n"
        f"- Allowed citation keys: {citation_list}.\n"
        "- These are the ONLY citation keys you may use. If a statement cannot "
        "be supported by one of them, omit the statement rather than citing a "
        "key that is not on this list.\n"
        "- Mention every contradiction explicitly, naming both sides.\n"
        "- Hedge any (UNGROUNDED) claim with a phrase like 'reportedly'."
    )
    return "\n\n".join(parts)


def build_synthesis_prompt(
    *,
    question: str,
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
    word_budget: int = 500,
) -> SynthesisPrompt:
    """
    Build a deterministic synthesis prompt for the literature-review LLM.

    Args:
        question: The user's research question (verbatim).
        papers: Ranked papers that constitute the legal citation set; each
            paper's citation key (see :func:`synthesis.schemas.assign_citation_keys`,
            which disambiguates same-author/year collisions) becomes an allowed
            citation.
        claims: Extracted per-paper claims; non-grounded claims are still
            included but tagged ``(UNGROUNDED)`` so the model knows to hedge.
        contradictions: Cross-paper tensions to surface explicitly.
        word_budget: Approximate target length of the generated review.

    Returns:
        A frozen :class:`SynthesisPrompt` with the system prompt, user
        prompt, and the deduplicated list of citation keys the model is
        allowed to use (in paper input order).

    Notes:
        Pure function. No LLM call, no network, no filesystem access.
        Per-field truncation keeps the prompt size bounded regardless of
        input pathologies (over-long abstracts, claims, or explanations).
    """
    keys = assign_citation_keys(papers)
    expected = _expected_citations(papers, keys)
    system = _build_system(word_budget=word_budget)
    user = _build_user(
        question=question,
        papers=papers,
        claims=claims,
        contradictions=contradictions,
        keys=keys,
        expected_citations=expected,
        word_budget=word_budget,
    )
    return SynthesisPrompt(system=system, user=user, expected_citations=expected)
