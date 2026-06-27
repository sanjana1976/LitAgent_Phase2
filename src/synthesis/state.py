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

from collections import Counter
from datetime import datetime, timezone
from typing import Literal
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
    papers: list[ScoredPaper] = Field(default_factory=list)
    claims: list[ClaimRecord] = Field(default_factory=list)
    contradictions: list[ContradictionPair] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)

    # --- generative output (empty until the synthesize step) ---
    review_text: str | None = None
    citation_checks: list[CitationCheck] = Field(
        default_factory=list,
        description="Raw per-citation verdicts; citation_validity() is derived from these.",
    )
    hallucinated_citations: list[str] = Field(default_factory=list)

    # --- the legible record ---
    trace: list[DecisionStep] = Field(default_factory=list)

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
            papers=list(self.papers),
            claims=list(self.claims),
            contradictions=list(self.contradictions),
            citation_checks=list(self.citation_checks),
        )
