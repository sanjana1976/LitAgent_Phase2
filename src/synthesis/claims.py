"""
Stage 5 of the LitSynth pipeline: per-paper claim extraction.

Role in the pipeline
--------------------
Given the ranked papers from stage 4, this stage asks the LLM to enumerate
the small set of *factual, paper-specific* claims each paper actually makes
(results, methods, datasets, limitations). Those claims become the atomic
units that stage 6 cross-checks for contradictions and that stage 8 cites
in the generated review.

Hallucination-resistance properties
-----------------------------------
- Each LLM call sees a single paper's own text (truncated ``full_text``,
  else ``abstract``, else nothing — in which case the paper is skipped
  rather than hallucinated against), so claims cannot leak between papers.
- Each claim must come with an ``evidence_quote`` that the model claims is
  verbatim from the supplied context. We re-verify that quote by normalised
  substring match against the same text the model saw and flip
  ``grounded=False`` (and halve ``confidence``) when verification fails.
- Per-paper LLM failures are isolated: a :class:`SynthesisLLMError` for one
  paper logs a warning and is skipped; subsequent papers still run.
- The function ``extract_claims`` itself never raises; callers always get a
  (possibly empty) list back.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from synthesis.llm import SynthesisLLMError, call_json
from synthesis.schemas import ClaimRecord, ScoredPaper

logger = logging.getLogger(__name__)


_FULL_TEXT_CHAR_BUDGET = 12_000
_RELAXED_PREFIX_CHARS = 60
_WHITESPACE_RE = re.compile(r"\s+")

_SYSTEM_PROMPT = (
    "You are a careful research-synthesis assistant. You extract specific, "
    "factual claims from one academic paper at a time. A 'claim' is a "
    "concrete assertion the paper itself makes about its results, methods, "
    "datasets, evaluation setup, or limitations. Do NOT include general "
    "background, motivation, or claims attributed to prior work. Every "
    "claim must be backed by a verbatim quote copied character-for-character "
    "from the paper text provided to you. Respond with JSON only."
)


def _normalize(text: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return _WHITESPACE_RE.sub(" ", (text or "").lower()).strip()


def _build_context(paper: ScoredPaper) -> str:
    """Pick the best available text for grounding claims against one paper."""
    if paper.full_text and paper.full_text.strip():
        return paper.full_text[:_FULL_TEXT_CHAR_BUDGET]
    if paper.abstract and paper.abstract.strip():
        return paper.abstract
    return ""


def _context_tier(paper: ScoredPaper) -> str:
    """
    Report which source tier the grounding context comes from.

    Prefers the paper-level ``text_tier`` set during fetch/parse (the source of
    truth). Falls back to inferring from ``has_pdf``/available text for papers
    constructed directly (e.g. in tests) without an explicit ``text_tier``.
    """
    tier = getattr(paper, "text_tier", "none")
    if tier in ("full_text", "abstract"):
        return tier
    if paper.has_pdf and paper.full_text and paper.full_text.strip():
        return "full_text"
    if (paper.full_text and paper.full_text.strip()) or (
        paper.abstract and paper.abstract.strip()
    ):
        return "abstract"
    return "none"


def _build_user_prompt(paper: ScoredPaper, context: str, max_claims: int) -> str:
    """Render the per-paper user prompt with the source text the model must cite."""
    authors = ", ".join(paper.authors[:5]) if paper.authors else "Unknown"
    year = str(paper.year) if paper.year is not None else "n.d."
    venue = paper.venue or ""
    header = (
        f"Paper id: {paper.paper_id}\n"
        f"Title: {paper.title}\n"
        f"Authors: {authors}\n"
        f"Year: {year}\n"
        f"Venue: {venue}\n"
    )
    instructions = (
        f"Extract up to {max_claims} grounded, paper-specific factual claims "
        "from the paper text below. Prefer claims about: empirical results, "
        "methods/architecture, datasets used, evaluation protocol, and stated "
        "limitations. Avoid claims that merely paraphrase prior work or "
        "general background. For each claim, copy a short verbatim quote "
        "(<= 240 characters) from the paper text as 'evidence_quote', and "
        "give a 'confidence' in [0.0, 1.0] reflecting how directly the quote "
        "supports the claim. Respond with strict JSON of the form: "
        '{"claims": [{"claim": "...", "evidence_quote": "...", '
        '"confidence": 0.0}, ...]}'
    )
    return (
        f"{header}\n"
        f"{instructions}\n\n"
        "----- BEGIN PAPER TEXT -----\n"
        f"{context}\n"
        "----- END PAPER TEXT -----\n"
    )


def _coerce_confidence(value: Any) -> float:
    """Coerce an LLM-provided confidence to a float clamped to [0, 1]."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.5
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def quote_is_grounded(evidence_quote: str, source_text: str) -> bool:
    """
    Public grounding check: does ``evidence_quote`` appear in ``source_text``?

    Used by the gap-detection loop to re-ground an unsupported claim against a
    candidate corroborating paper. Normalizes ``source_text`` the same way the
    extractor does so results are consistent across the pipeline.
    """
    return _is_grounded(evidence_quote, _normalize(source_text))


def _is_grounded(evidence_quote: str, normalized_context: str) -> bool:
    """Verify an evidence quote against the normalized source text."""
    normalized_quote = _normalize(evidence_quote)
    if not normalized_quote or not normalized_context:
        return False
    if normalized_quote in normalized_context:
        return True
    prefix = normalized_quote[:_RELAXED_PREFIX_CHARS]
    if len(prefix) >= 10 and prefix in normalized_context:
        return True
    return False


def _parse_claims_payload(
    payload: dict[str, Any],
    paper: ScoredPaper,
    normalized_context: str,
    max_claims: int,
    context_tier: str,
) -> list[ClaimRecord]:
    """Convert one LLM JSON payload into validated, grounded ``ClaimRecord``s."""
    raw_claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(raw_claims, list):
        return []

    records: list[ClaimRecord] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        claim_text = str(item.get("claim", "") or "").strip()
        evidence = str(item.get("evidence_quote", "") or "").strip()
        if not claim_text or not evidence:
            continue
        confidence = _coerce_confidence(item.get("confidence", 0.5))
        try:
            record = ClaimRecord(
                paper_id=paper.paper_id,
                claim=claim_text,
                evidence_quote=evidence,
                confidence=confidence,
                grounded=False,
            )
        except Exception:  # noqa: BLE001 - pydantic validation may reject lengths
            continue

        grounded = _is_grounded(evidence, normalized_context)
        record.grounded = grounded
        if grounded:
            # Self-grounded against the originating paper's own text. The tier
            # reflects how strong that text is (full PDF vs abstract only).
            record.grounding_tier = context_tier if context_tier in ("full_text", "abstract") else "abstract"
            record.supporting_paper_id = paper.paper_id
        else:
            # Unverified for now. The gap-hunting loop may later upgrade this to
            # 'corroborated' by finding a different supporting paper.
            record.grounding_tier = "none"
            record.supporting_paper_id = None
            record.confidence = _coerce_confidence(record.confidence * 0.5)

        records.append(record)
        if len(records) >= max_claims:
            break
    return records


def extract_claims(
    papers: list[ScoredPaper],
    *,
    max_claims_per_paper: int = 4,
    llm_call: Callable[..., dict[str, Any]] | None = None,
) -> list[ClaimRecord]:
    """
    Extract grounded factual claims from each paper.

    Args:
        papers: Ranked papers from stage 4. Papers with no ``full_text`` and
            no ``abstract`` are skipped without invoking the LLM.
        max_claims_per_paper: Hard cap on claims kept per paper after
            grounding. Defaults to 4.
        llm_call: Injectable JSON LLM caller with the same signature as
            :func:`synthesis.llm.call_json`. Defaults to ``call_json``.

    Returns:
        Flat list of ``ClaimRecord`` instances across all input papers.
        The function never raises; LLM failures for an individual paper
        produce a logged warning and that paper is skipped.
    """
    if not papers:
        return []
    caller = llm_call if llm_call is not None else call_json

    all_records: list[ClaimRecord] = []
    for paper in papers:
        context = _build_context(paper)
        if not context:
            logger.warning(
                "claims: skipping %s — no full_text or abstract available.",
                paper.paper_id,
            )
            continue

        normalized_context = _normalize(context)
        context_tier = _context_tier(paper)
        user_prompt = _build_user_prompt(paper, context, max_claims_per_paper)

        try:
            payload = caller(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                temperature=0.1,
                max_tokens=1_200,
            )
        except SynthesisLLMError as exc:
            logger.warning(
                "claims: LLM call failed for %s (%s); skipping paper.",
                paper.paper_id,
                exc,
            )
            continue
        except Exception as exc:  # noqa: BLE001 - never let one paper crash batch
            logger.warning(
                "claims: unexpected LLM error for %s (%s); skipping paper.",
                paper.paper_id,
                exc,
            )
            continue

        records = _parse_claims_payload(
            payload, paper, normalized_context, max_claims_per_paper, context_tier
        )
        if not records:
            logger.warning(
                "claims: LLM returned no usable claims for %s; skipping paper.",
                paper.paper_id,
            )
            continue
        all_records.extend(records)

    return all_records
