"""Pure formatting helpers for the LitSynth decision-trace UI.

The Streamlit app imports these helpers, and tests exercise them without
requiring Streamlit. Keep this module side-effect free: no network, no LLM, no
filesystem writes.
"""

from __future__ import annotations

from typing import Any

from synthesis.state import SynthesisState
from synthesis.trace import DecisionStep


def state_metrics(state: SynthesisState) -> dict[str, Any]:
    """Return compact run-level numbers for dashboard cards."""
    return {
        "papers": len(state.papers),
        "claims": len(state.claims),
        "grounded_fraction": round(state.grounded_fraction(), 3),
        "citation_validity": round(state.citation_validity(), 3),
        "contradictions": len(state.contradictions),
        "open_gaps": len(state.open_gaps),
        "hallucinated_citations": len(state.hallucinated_citations),
    }


def trace_rows(state: SynthesisState) -> list[dict[str, Any]]:
    """Flatten trace steps into rows suitable for a table/dataframe."""
    rows: list[dict[str, Any]] = []
    for idx, step in enumerate(state.trace, start=1):
        rows.append(
            {
                "#": idx,
                "action": step.action,
                "result": step.result,
                "parent": step.parent_step_id or "",
                "trigger": step.trigger,
                "rationale": step.rationale,
                "result_note": step.result_note,
                "effect": summarize_effect(step),
                "duration_ms": step.duration_ms,
                "llm_calls": step.llm_calls,
            }
        )
    return rows


def summarize_effect(step: DecisionStep) -> str:
    """Render the referenceable state delta in one readable sentence."""
    effect = step.effect
    parts: list[str] = []
    if effect.added_paper_ids:
        parts.append(f"added papers: {', '.join(effect.added_paper_ids)}")
    if effect.claim_refs:
        parts.append(f"claims: {', '.join(effect.claim_refs)}")
    elif effect.claim_ref:
        parts.append(f"claim: {effect.claim_ref}")
    if effect.contradiction_ids:
        parts.append(f"contradictions: {', '.join(effect.contradiction_ids)}")
    if effect.resolved_conflict:
        parts.append(f"resolved conflict: {effect.resolved_conflict}")
    if effect.tier_before or effect.tier_after:
        before = effect.tier_before or "?"
        after = effect.tier_after or "?"
        parts.append(f"tier: {before} -> {after}")
    if effect.gap_ref:
        parts.append(f"gap: {effect.gap_ref}")
    return "; ".join(parts)


def claim_rows(state: SynthesisState) -> list[dict[str, Any]]:
    """Return claims with the grounding fields reviewers need to audit."""
    return [
        {
            "claim_id": claim.claim_id,
            "paper_id": claim.paper_id,
            "claim": claim.claim,
            "grounded": claim.grounded,
            "grounding_tier": claim.grounding_tier,
            "supporting_paper_id": claim.supporting_paper_id or "",
            "confidence": claim.confidence,
        }
        for claim in state.claims
    ]


def paper_rows(state: SynthesisState) -> list[dict[str, Any]]:
    """Return paper evidence rows, including the paper-level text tier."""
    return [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "citation_key": paper.short_citation_key(),
            "text_tier": paper.text_tier,
            "has_pdf": paper.has_pdf,
            "relevance_score": round(paper.relevance_score, 4),
            "url": paper.url or "",
        }
        for paper in state.papers
    ]


def contradiction_rows(state: SynthesisState) -> list[dict[str, Any]]:
    """Return contradiction rows keyed by stable contradiction_id."""
    return [
        {
            "contradiction_id": pair.contradiction_id,
            "paper_a": pair.paper_a,
            "paper_b": pair.paper_b,
            "tension_type": pair.tension_type,
            "claim_a": pair.claim_a,
            "claim_b": pair.claim_b,
            "explanation": pair.explanation,
        }
        for pair in state.contradictions
    ]


def gap_rows(state: SynthesisState) -> list[dict[str, Any]]:
    """Return gap lifecycle rows for the UI."""
    return [
        {
            "gap_id": gap.gap_id,
            "kind": gap.kind,
            "status": gap.status,
            "origin_claim_ref": gap.origin_claim_ref or "",
            "resolved_by_step_id": gap.resolved_by_step_id or "",
            "resolved_by_paper_id": gap.resolved_by_paper_id or "",
            "description": gap.description,
        }
        for gap in state.gaps
    ]
