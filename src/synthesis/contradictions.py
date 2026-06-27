"""
Stage 6 of the LitSynth pipeline: cross-paper contradiction detection.

Role in the pipeline
--------------------
Given the grounded ``ClaimRecord`` set produced by stage 5, this stage asks
the LLM to surface a small number of *genuine* tensions between papers —
direct contradictions, scope mismatches (claim X holds in setting A but not
in setting B), and methodology disagreements. The resulting
``ContradictionPair`` list is consumed by stages 7-8 so the generated
review can explicitly discuss disagreements rather than silently pick a
side.

Hallucination-resistance properties
-----------------------------------
- The LLM only sees claims that *already* exist in the input set; we then
  re-validate every returned pair against that set:
    * Paper ids must both appear in the input ``claims``' paper_id set —
      pairs that name a paper that wasn't shown are dropped.
    * ``claim_a`` / ``claim_b`` must substring-match real claim text (with
      a 30-character paraphrase tolerance) — pure inventions are dropped.
    * Self-contradictions (``paper_a == paper_b``) are dropped.
- (A, B) and (B, A) duplicates are collapsed by canonicalising the pair as
  ``(min(paper_id), max(paper_id))``.
- pydantic ``model_validate`` enforces the schema (``tension_type`` literal,
  field types) and rejects malformed entries silently.
- If fewer than two distinct papers have claims we return ``[]`` without
  calling the LLM at all, so a degenerate input cannot spawn a hallucinated
  "contradiction" between a paper and itself.
- The function ``detect_contradictions`` itself never raises.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import ValidationError

from synthesis.llm import SynthesisLLMError, call_json
from synthesis.schemas import ClaimRecord, ContradictionPair, ScoredPaper

logger = logging.getLogger(__name__)


_PARAPHRASE_MATCH_CHARS = 30

_SYSTEM_PROMPT = (
    "You are a careful research-synthesis assistant. You are given a "
    "numbered list of factual claims, each tagged with the paper it came "
    "from. Identify genuine tensions BETWEEN DIFFERENT PAPERS: direct "
    "contradictions, scope mismatches (a claim that holds in one regime but "
    "not another), or methodological disagreements. Do NOT invent claims; "
    "use only the wording given. Do NOT compare a paper against itself. "
    "Respond with strict JSON only."
)


def _build_user_prompt(
    claims: list[ClaimRecord],
    papers_by_id: dict[str, ScoredPaper],
    max_pairs: int,
) -> str:
    """Render the numbered claim list and instructions for the model."""
    lines: list[str] = []
    for idx, claim in enumerate(claims, start=1):
        paper = papers_by_id.get(claim.paper_id)
        cite_key = paper.short_citation_key() if paper is not None else ""
        suffix = f" {cite_key}" if cite_key else ""
        lines.append(f"{idx}. paper_id={claim.paper_id}{suffix} :: {claim.claim}")
    numbered = "\n".join(lines)

    schema_hint = (
        '{"contradictions": [{"paper_a": "<paper_id>", '
        '"paper_b": "<paper_id>", "claim_a": "<verbatim claim text>", '
        '"claim_b": "<verbatim claim text>", '
        '"tension_type": "contradiction"|"scope"|"methodology", '
        '"explanation": "<one or two sentences>"}]}'
    )
    return (
        f"Find up to {max_pairs} genuine cross-paper tensions in the claims "
        "below. Only emit a pair when the two claims really do disagree. "
        "Use the exact paper_id values shown and copy the claim text "
        "verbatim from the list. Skip pairs that share a paper_id.\n\n"
        "Claims:\n"
        f"{numbered}\n\n"
        f"Respond with JSON of the form: {schema_hint}\n"
    )


def _claim_text_matches(candidate: str, real_claims_lower: list[str]) -> bool:
    """Allow a 30-character substring match either way to tolerate paraphrase."""
    candidate_lower = (candidate or "").strip().lower()
    if len(candidate_lower) < _PARAPHRASE_MATCH_CHARS:
        return False
    for real in real_claims_lower:
        if not real:
            continue
        if candidate_lower in real or real in candidate_lower:
            return True
        prefix = candidate_lower[:_PARAPHRASE_MATCH_CHARS]
        if prefix and prefix in real:
            return True
        real_prefix = real[:_PARAPHRASE_MATCH_CHARS]
        if real_prefix and real_prefix in candidate_lower:
            return True
    return False


def detect_contradictions(
    claims: list[ClaimRecord],
    papers: list[ScoredPaper],
    *,
    max_pairs: int = 5,
    llm_call: Callable[..., dict[str, Any]] | None = None,
) -> list[ContradictionPair]:
    """
    Detect cross-paper tensions from a grounded claim set.

    Args:
        claims: All ``ClaimRecord``s produced by :func:`extract_claims`.
        papers: The same ranked papers used in stage 5 — only used to render
            readable citation keys inside the LLM prompt; pairs are still
            validated by ``paper_id``.
        max_pairs: Hard cap on returned pairs after deduplication. Default 5.
        llm_call: Injectable JSON LLM caller with the same signature as
            :func:`synthesis.llm.call_json`. Defaults to ``call_json``.

    Returns:
        Validated, deduplicated list of ``ContradictionPair`` objects. The
        function never raises; LLM failures or empty results yield ``[]``.
    """
    if not claims:
        return []

    paper_ids_with_claims = {c.paper_id for c in claims}
    if len(paper_ids_with_claims) < 2:
        return []

    caller = llm_call if llm_call is not None else call_json
    papers_by_id = {p.paper_id: p for p in papers}

    user_prompt = _build_user_prompt(claims, papers_by_id, max_pairs)

    try:
        payload = caller(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.1,
            max_tokens=1_500,
        )
    except SynthesisLLMError as exc:
        logger.warning("contradictions: LLM call failed (%s); returning [].", exc)
        return []
    except Exception as exc:  # noqa: BLE001 - never propagate to pipeline
        logger.warning(
            "contradictions: unexpected LLM error (%s); returning [].", exc
        )
        return []

    raw_pairs = payload.get("contradictions") if isinstance(payload, dict) else None
    if not isinstance(raw_pairs, list) or not raw_pairs:
        logger.warning("contradictions: LLM returned no pairs; returning [].")
        return []

    real_claims_by_paper: dict[str, list[str]] = {}
    for claim in claims:
        real_claims_by_paper.setdefault(claim.paper_id, []).append(
            claim.claim.lower()
        )

    validated: list[ContradictionPair] = []
    seen_keys: set[tuple[str, str]] = set()

    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        try:
            pair = ContradictionPair.model_validate(item)
        except ValidationError:
            continue

        if pair.paper_a == pair.paper_b:
            continue
        if pair.paper_a not in paper_ids_with_claims:
            continue
        if pair.paper_b not in paper_ids_with_claims:
            continue

        if not _claim_text_matches(
            pair.claim_a, real_claims_by_paper.get(pair.paper_a, [])
        ):
            continue
        if not _claim_text_matches(
            pair.claim_b, real_claims_by_paper.get(pair.paper_b, [])
        ):
            continue

        key = tuple(sorted((pair.paper_a, pair.paper_b)))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        validated.append(pair)
        if len(validated) >= max_pairs:
            break

    return validated
