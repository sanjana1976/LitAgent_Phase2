"""
Critic stage for the writer-critic revision loop.

A second LLM role reads the drafted literature review against the evidence
set (papers, extracted claims, detected contradictions) and raises
*checkable* objections — problems the code can verify, not stylistic taste:

- draft statements not supported by any extracted claim,
- contradictions in the evidence the draft fails to surface,
- citation keys outside the allowed set.

Hallucination-resistance discipline (same as contradiction detection): the
LLM proposes, code disposes. Every objection must quote a verbatim excerpt
from the draft; objections whose excerpt does not appear in the draft are
dropped as fabrications. Hallucinated citations found by the deterministic
validator are appended as auto-objections regardless of what the LLM says,
so the loop's most important signal never depends on the critic model.

Fail-soft contract: if the LLM is unavailable, ``critique_review`` returns
``([], False)`` and the revision loop simply doesn't run — a broken critic
can never block a review.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from synthesis.llm import SynthesisLLMError, call_json
from synthesis.schemas import ClaimRecord, ContradictionPair, ScoredPaper, assign_citation_keys

logger = logging.getLogger(__name__)

_MAX_OBJECTIONS = 4
_EXCERPT_MIN_CHARS = 15
_CLAIM_PREVIEW_CHARS = 200
_WHITESPACE_RE = re.compile(r"\s+")

_SYSTEM_PROMPT = (
    "You are a skeptical peer reviewer checking a drafted literature review "
    "against the ONLY evidence its author was given. Raise at most "
    f"{_MAX_OBJECTIONS} CHECKABLE objections, strictly of these kinds: "
    "(a) a statement in the draft that none of the provided claims supports, "
    "(b) a contradiction listed in the evidence that the draft fails to "
    "mention, (c) a citation key used in the draft that is not in the allowed "
    "list. Each objection MUST include an 'excerpt' copied verbatim from the "
    "draft (at least a full clause) and a one-sentence 'problem'. Do NOT "
    "object to style, length, or ordering. If the draft is faithful to the "
    "evidence, return an empty list. Respond with a single JSON object: "
    '{"objections": [{"excerpt": "...", "problem": "..."}]}'
)


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").lower()).strip()


def _build_user_prompt(
    question: str,
    review_text: str,
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
) -> str:
    keys = assign_citation_keys(papers)
    allowed = ", ".join(keys.get(p.paper_id, p.short_citation_key()) for p in papers) or "(none)"
    claim_lines = [
        f"- [{keys.get(c.paper_id, c.paper_id)}] {c.claim[:_CLAIM_PREVIEW_CHARS]}"
        for c in claims
    ] or ["(no claims extracted)"]
    tension_lines = [
        f"- {keys.get(pair.paper_a, pair.paper_a)} vs {keys.get(pair.paper_b, pair.paper_b)}: "
        f"{pair.explanation[:_CLAIM_PREVIEW_CHARS]}"
        for pair in contradictions
    ] or ["(no contradictions detected)"]

    return (
        f"Research question: {question.strip()}\n\n"
        f"Allowed citation keys: {allowed}\n\n"
        "Evidence claims:\n" + "\n".join(claim_lines) + "\n\n"
        "Known contradictions the draft should surface:\n"
        + "\n".join(tension_lines)
        + "\n\n----- BEGIN DRAFT -----\n"
        f"{review_text}\n"
        "----- END DRAFT -----\n\n"
        'Respond with JSON: {"objections": [{"excerpt": "...", "problem": "..."}]}'
    )


def _validated_objections(payload: dict[str, Any], review_text: str) -> list[str]:
    """Keep only objections whose excerpt verifiably appears in the draft."""
    raw = payload.get("objections") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    normalized_draft = _normalize(review_text)
    out: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        excerpt = str(item.get("excerpt", "") or "").strip()
        problem = str(item.get("problem", "") or "").strip()
        if len(excerpt) < _EXCERPT_MIN_CHARS or not problem:
            continue
        if _normalize(excerpt) not in normalized_draft:
            logger.info("critic: dropping objection with fabricated excerpt")
            continue
        out.append(f'"{excerpt}" — {problem}')
        if len(out) >= _MAX_OBJECTIONS:
            break
    return out


def critique_review(
    question: str,
    review_text: str,
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
    hallucinated_citations: list[str],
    *,
    llm_call: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[str], bool]:
    """
    Critique a drafted review against its evidence set.

    Returns:
        ``(objections, used_llm)``. Objections are human-readable strings the
        writer must address; each is either validated against the draft text
        or generated deterministically from the citation validator. On LLM
        failure returns ``([], False)`` so the revision loop is skipped.
    """
    if not review_text or not review_text.strip():
        return [], False

    # Deterministic auto-objections: the citation validator already proved
    # these are wrong, no model judgment involved.
    auto = [
        f"The citation {key!r} does not resolve to any paper in the evidence "
        "set; remove it or replace it with an allowed citation key."
        for key in hallucinated_citations
    ]

    invoker = llm_call if llm_call is not None else call_json
    try:
        payload = invoker(
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(question, review_text, papers, claims, contradictions),
            temperature=0.0,
            max_tokens=900,
        )
    except SynthesisLLMError as exc:
        logger.warning("critic: LLM call failed (%s); returning auto-objections only.", exc)
        return auto, bool(auto)
    except Exception as exc:  # noqa: BLE001 - the critic must never block a review
        logger.warning("critic: unexpected error (%s); returning auto-objections only.", exc)
        return auto, bool(auto)

    validated = _validated_objections(payload, review_text)
    combined = auto + [o for o in validated if o not in auto]
    return combined[:_MAX_OBJECTIONS + len(auto)], True
