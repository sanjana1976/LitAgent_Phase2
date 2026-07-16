"""
Working state for the agentic LitSynth controller.

:class:`SynthesisState` is everything the agent knows at a point in time: the
question, the evidence gathered so far, the open gaps, and the full decision
trace. It is a *mutable working object* that the controller updates each step.
The immutable deliverable is produced once, at the end, by :meth:`to_result`.

Two disciplines carried over from the locked layers below it:
- **Never store what you can derive.** Counts, fractions, status, and the
  current step are computed from ``trace``/``claims``/``citation_checks`` rather
  than stored alongside them (which would create drift). Only the irreducible
  raw material is stored.
- **Don't blend distinct signals.** Citation validity and grounded fraction are
  reported separately; the single legacy ``confidence_score`` is only assembled
  inside :meth:`to_result` for backward compatibility with ``synthesis_runs``.
"""

from __future__ import annotations

import operator
from collections import Counter
from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from synthesis.schemas import (
    CitationCheck,
    ClaimRecord,
    ContradictionPair,
    ScoredPaper,
    SynthesisResult,
)
from synthesis.trace import DecisionStep

# ---------------------------------------------------------------------------
# Gaps: one model, one lifecycle, a ``kind`` discriminator for the real
# behavioral difference between an ungrounded claim and a missing theme.
# ---------------------------------------------------------------------------

GapKind = Literal["ungrounded_claim", "coverage"]
GapStatus = Literal["open", "hunting", "resolved", "flagged_unverified"]


class Gap(BaseModel):
    """A hole in the review the agent may try to close, with a lifecycle."""

    model_config = ConfigDict(extra="forbid")

    gap_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: GapKind
    description: str
    origin_claim_ref: str | None = Field(
        default=None,
        description="claim_id this gap is about (ungrounded_claim gaps).",
    )
    status: GapStatus = "open"
    resolved_by_step_id: str | None = None
    resolved_by_paper_id: str | None = None


TerminalReason = Literal[
    "synthesized",
    "no_papers",
    "iteration_cap",
    "reformulation_cap",
    "error",
]


def _merge_by_id(id_attr: str):
    """
    Build a LangGraph reducer that upserts list items by a stable id field.

    A new item whose id matches an existing one *replaces* it in place
    (preserving list order); unseen ids append. This lets nodes update an
    item — e.g. a gap hunt upgrading a claim's grounding tier — by returning
    a modified copy instead of mutating shared state, which stays correct
    under parallel fan-out.
    """

    def _reducer(existing: list, new: list) -> list:
        index_by_id = {getattr(item, id_attr): i for i, item in enumerate(existing)}
        merged = list(existing)
        for item in new:
            key = getattr(item, id_attr)
            if key in index_by_id:
                merged[index_by_id[key]] = item
            else:
                index_by_id[key] = len(merged)
                merged.append(item)
        return merged

    return _reducer


# papers is upsert-by-id like the others: the pool never loses a paper (nodes
# only ever add or refresh entries — e.g. re-ranked copies carrying updated
# relevance_score), while *selection* lives separately in ``active_paper_ids``.
merge_papers = _merge_by_id("paper_id")
merge_claims = _merge_by_id("claim_id")
merge_contradictions = _merge_by_id("contradiction_id")
merge_gaps = _merge_by_id("gap_id")


class SynthesisState(BaseModel):
    """Mutable working memory for one synthesis run."""

    model_config = ConfigDict(extra="forbid")

    # --- identity / inputs ---
    question: str
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Live working set of sub-queries; reformulation history is in the trace.",
    )
    session_id: str | None = None

    # --- accumulating evidence ---
    # The Annotated reducer metadata is read only by the LangGraph orchestration
    # layer (synthesis.graph); pydantic and all direct callers ignore it.
    papers: Annotated[list[ScoredPaper], merge_papers] = Field(default_factory=list)
    active_paper_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered paper_ids currently selected for downstream stages "
            "(ranking/relevance-gate output). Empty means 'all papers' for "
            "backward compatibility with controller-driven runs."
        ),
    )
    claims: Annotated[list[ClaimRecord], merge_claims] = Field(default_factory=list)
    contradictions: Annotated[list[ContradictionPair], merge_contradictions] = Field(
        default_factory=list
    )
    gaps: Annotated[list[Gap], merge_gaps] = Field(default_factory=list)

    # --- generative output (empty until the synthesize step) ---
    review_text: str | None = None
    objections: list[str] = Field(
        default_factory=list,
        description=(
            "Validated checkable objections from the most recent critique pass "
            "(replace semantics; emptied when the critic accepts the draft)."
        ),
    )
    citation_checks: list[CitationCheck] = Field(
        default_factory=list,
        description="Raw per-citation verdicts; citation_validity() is derived from these.",
    )
    hallucinated_citations: list[str] = Field(default_factory=list)

    # --- the legible record ---
    trace: Annotated[list[DecisionStep], operator.add] = Field(default_factory=list)

    # --- termination ---
    terminal_reason: TerminalReason | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------ #
    # Derived views (never stored)
    # ------------------------------------------------------------------ #

    @property
    def current_step(self) -> int:
        return len(self.trace)

    @property
    def status(self) -> str:
        if self.terminal_reason is not None:
            return "done"
        if self.trace:
            return self.trace[-1].action
        return "init"

    @property
    def reformulation_count(self) -> int:
        return sum(1 for s in self.trace if s.action == "reformulate")

    @property
    def open_gaps(self) -> list[Gap]:
        return [g for g in self.gaps if g.status in ("open", "hunting")]

    def grounded_fraction(self) -> float:
        if not self.claims:
            return 0.0
        return sum(1 for c in self.claims if c.grounded) / len(self.claims)

    def grounded_by_tier(self) -> dict[str, int]:
        return dict(Counter(c.grounding_tier for c in self.claims))

    def citation_validity(self) -> float:
        if not self.citation_checks:
            return 0.0
        valid = sum(1 for c in self.citation_checks if c.is_valid)
        return valid / len(self.citation_checks)

    # ------------------------------------------------------------------ #
    # Helpers for the controller
    # ------------------------------------------------------------------ #

    def log(self, step: DecisionStep) -> DecisionStep:
        """Append a decision step to the trace and return it."""
        self.trace.append(step)
        return step

    def get_claim(self, claim_id: str) -> ClaimRecord | None:
        return next((c for c in self.claims if c.claim_id == claim_id), None)

    def get_paper(self, paper_id: str) -> ScoredPaper | None:
        return next((p for p in self.papers if p.paper_id == paper_id), None)

    def active_papers(self) -> list[ScoredPaper]:
        """
        The papers currently selected for downstream stages, in selection order.

        Falls back to all papers when ``active_paper_ids`` is empty so that
        controller-driven runs (which trim ``papers`` in place instead of
        tracking a selection) behave exactly as before.
        """
        if not self.active_paper_ids:
            return list(self.papers)
        by_id = {p.paper_id: p for p in self.papers}
        return [by_id[pid] for pid in self.active_paper_ids if pid in by_id]

    def citations_used(self) -> list[str]:
        """Resolved, deduplicated paper ids from valid citation checks (in order)."""
        used: list[str] = []
        seen: set[str] = set()
        for check in self.citation_checks:
            pid = check.resolved_paper_id
            if check.is_valid and pid and pid not in seen:
                seen.add(pid)
                used.append(pid)
        return used

    # ------------------------------------------------------------------ #
    # Deliverable
    # ------------------------------------------------------------------ #

    def to_result(self) -> SynthesisResult:
        """
        Produce the immutable :class:`SynthesisResult` artifact from state.

        The legacy single ``confidence_score`` is assembled here (and only here)
        as ``citation_validity * grounded_fraction`` for backward compatibility
        with existing persistence and renderers. The honest, unblended numbers
        remain available on the state via the dedicated accessors.
        """
        confidence = round(
            max(0.0, min(1.0, self.citation_validity() * self.grounded_fraction())),
            4,
        )
        return SynthesisResult(
            question=self.question,
            review_text=self.review_text or "",
            citations_used=self.citations_used(),
            hallucinated_citations=list(self.hallucinated_citations),
            contradictions_found=len(self.contradictions),
            confidence_score=confidence,
            papers=self.active_papers(),
            claims=list(self.claims),
            contradictions=list(self.contradictions),
            citation_checks=list(self.citation_checks),
        )
