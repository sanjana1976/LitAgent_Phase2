"""
Decision-trace layer for the agentic LitSynth controller.

A :class:`DecisionStep` is the atomic unit of the whole project: it is at once
the runtime trace, the demo timeline (Step 6), and the structured eval evidence
(Step 5). It is modelled as a *before -> decide -> after* arc rather than a flat
event, because collapsing that arc is exactly what turns an explanation into a
mere log.

Design decisions (locked):
- **Two-phase mutable**: one object per decision, written in two phases. It is
  created at decision time (``result="pending"``) and updated once the action
  finishes (``complete(...)``). No join is needed anywhere the trace is read.
- **Typed params union**: each :data:`ActionType` carries exactly the params it
  needs via a ``kind``-discriminated union, so the eval harness can aggregate
  over action types without parsing prose.
- **Causal chain**: ``parent_step_id`` links a step to the prior step it
  responds to, so the trace is an explanation, not just a sequence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthesis.schemas import GroundingTier

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

ActionType = Literal[
    "decompose",
    "search",
    "reformulate",
    "fetch_pdf",
    "extract_claims",
    "detect_contradictions",
    "hunt_support",
    "resolve_conflict",
    "synthesize",
]

# 'pending' is the phase-1 state before the action has been executed. The other
# verdicts are written in phase 2 by :meth:`DecisionStep.complete`.
#   ok           -> the action did what it intended
#   insufficient -> it ran but the goal was not met (e.g. still < 4 papers)
#   failed       -> it raised / errored
#   noop         -> there was nothing to do
StepResult = Literal["pending", "ok", "insufficient", "failed", "noop"]


# ---------------------------------------------------------------------------
# Typed params: one model per action, discriminated on ``kind``.
# ---------------------------------------------------------------------------


class _Params(BaseModel):
    """Base for all action params (strict; forbids unknown fields)."""

    model_config = ConfigDict(extra="forbid")


class DecomposeParams(_Params):
    kind: Literal["decompose"] = "decompose"
    question: str
    n: int = 4


class SearchParams(_Params):
    kind: Literal["search"] = "search"
    query: str
    sources: list[str] = Field(default_factory=list)


class ReformulateParams(_Params):
    kind: Literal["reformulate"] = "reformulate"
    original_query: str
    new_query: str


class FetchPdfParams(_Params):
    kind: Literal["fetch_pdf"] = "fetch_pdf"
    paper_id: str
    url: str | None = None


class ExtractClaimsParams(_Params):
    kind: Literal["extract_claims"] = "extract_claims"
    paper_ids: list[str] = Field(default_factory=list)


class DetectContradictionsParams(_Params):
    kind: Literal["detect_contradictions"] = "detect_contradictions"
    claim_ids: list[str] = Field(default_factory=list)


class GapHuntParams(_Params):
    kind: Literal["hunt_support"] = "hunt_support"
    claim_id: str
    search_terms: list[str] = Field(default_factory=list)


class ResolveConflictParams(_Params):
    kind: Literal["resolve_conflict"] = "resolve_conflict"
    paper_a: str
    paper_b: str
    search_terms: list[str] = Field(default_factory=list)


class SynthesizeParams(_Params):
    kind: Literal["synthesize"] = "synthesize"
    word_budget: int = 500


StepParams = Annotated[
    Union[
        DecomposeParams,
        SearchParams,
        ReformulateParams,
        FetchPdfParams,
        ExtractClaimsParams,
        DetectContradictionsParams,
        GapHuntParams,
        ResolveConflictParams,
        SynthesizeParams,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Effect: the referenceable state delta produced by a step.
# ---------------------------------------------------------------------------


class StepEffect(BaseModel):
    """
    What concretely changed in :class:`SynthesisState` as a result of a step.

    Distinct from params: params are *what the action was invoked with*; the
    effect is *what it changed*. Kept referenceable (ids, tier transitions) so
    the UI can render "upgraded claim C from abstract -> corroborated via X".
    """

    model_config = ConfigDict(extra="forbid")

    added_paper_ids: list[str] = Field(default_factory=list)
    claim_ref: str | None = Field(
        default=None,
        description="A single claim this step acted on (e.g. gap-hunt corroboration).",
    )
    claim_refs: list[str] = Field(
        default_factory=list,
        description="Multiple claims produced by this step (e.g. per-paper extraction).",
    )
    tier_before: GroundingTier | None = None
    tier_after: GroundingTier | None = None
    contradiction_ids: list[str] = Field(
        default_factory=list,
        description="contradiction_ids produced by this step (e.g. detection).",
    )
    resolved_conflict: str | None = Field(
        default=None,
        description="contradiction_id of the conflict this step acted on (stable referent).",
    )
    gap_ref: str | None = None


# ---------------------------------------------------------------------------
# DecisionStep: the two-phase arc.
# ---------------------------------------------------------------------------


class DecisionStep(BaseModel):
    """One logged decision: observe (before) -> decide -> act -> result (after)."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(default_factory=lambda: uuid4().hex)
    parent_step_id: str | None = None

    # --- BEFORE: what prompted the decision ---
    trigger: str
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- THE DECISION ---
    action: ActionType
    params: StepParams
    rationale: str

    # --- AFTER: written in phase 2 (see complete) ---
    result: StepResult = "pending"
    result_note: str = ""
    effect: StepEffect = Field(default_factory=StepEffect)
    completed_at: datetime | None = None

    # --- observability (Step 5 baseline comparison) ---
    llm_calls: int = 0
    duration_ms: int | None = None

    @model_validator(mode="after")
    def _action_matches_params(self) -> "DecisionStep":
        if self.params.kind != self.action:
            raise ValueError(
                f"action {self.action!r} does not match params.kind "
                f"{self.params.kind!r}"
            )
        return self

    @classmethod
    def start(
        cls,
        *,
        action: ActionType,
        params: StepParams,
        trigger: str,
        rationale: str,
        parent_step_id: str | None = None,
        step_id: str | None = None,
    ) -> "DecisionStep":
        """Create a phase-1 (pending) step at decision time."""
        kwargs: dict = {
            "action": action,
            "params": params,
            "trigger": trigger,
            "rationale": rationale,
            "parent_step_id": parent_step_id,
        }
        if step_id is not None:
            kwargs["step_id"] = step_id
        return cls(**kwargs)

    def complete(
        self,
        *,
        result: StepResult,
        result_note: str = "",
        effect: StepEffect | None = None,
        llm_calls: int = 0,
        duration_ms: int | None = None,
    ) -> "DecisionStep":
        """Record phase-2 outcome and stamp ``completed_at``. Mutates in place."""
        if result == "pending":
            raise ValueError("complete() requires a terminal result, not 'pending'")
        self.result = result
        self.result_note = result_note
        if effect is not None:
            self.effect = effect
        self.llm_calls = llm_calls
        self.duration_ms = duration_ms
        self.completed_at = datetime.now(timezone.utc)
        return self

    @property
    def is_pending(self) -> bool:
        return self.result == "pending"
