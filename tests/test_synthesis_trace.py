"""Tests for the decision-trace layer (synthesis.trace)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthesis.trace import (
    DecisionStep,
    GapHuntParams,
    ReformulateParams,
    SearchParams,
    StepEffect,
)


def _reformulate_step() -> DecisionStep:
    return DecisionStep.start(
        action="reformulate",
        params=ReformulateParams(
            original_query="long context retrieval",
            new_query="long-context attention efficiency",
        ),
        trigger="only 2 papers after first search",
        rationale="too few results; broadening terms",
    )


def test_start_creates_pending_step_with_ids_and_timestamps() -> None:
    step = _reformulate_step()
    assert step.is_pending
    assert step.result == "pending"
    assert step.completed_at is None
    assert step.decided_at is not None
    assert step.step_id  # auto-generated
    assert step.parent_step_id is None


def test_step_ids_are_unique() -> None:
    assert _reformulate_step().step_id != _reformulate_step().step_id


def test_complete_records_phase_two_outcome() -> None:
    step = _reformulate_step()
    effect = StepEffect(added_paper_ids=["arxiv:1", "arxiv:2", "arxiv:3", "arxiv:4"])
    step.complete(
        result="ok",
        result_note="reformulation returned 6 papers",
        effect=effect,
        llm_calls=1,
        duration_ms=820,
    )
    assert not step.is_pending
    assert step.result == "ok"
    assert step.result_note == "reformulation returned 6 papers"
    assert step.effect.added_paper_ids == ["arxiv:1", "arxiv:2", "arxiv:3", "arxiv:4"]
    assert step.llm_calls == 1
    assert step.duration_ms == 820
    assert step.completed_at is not None


def test_complete_rejects_pending_result() -> None:
    step = _reformulate_step()
    with pytest.raises(ValueError):
        step.complete(result="pending")


def test_insufficient_result_captures_try_and_fail() -> None:
    """The trace must be able to record a failed attempt, not only successes."""
    step = _reformulate_step()
    step.complete(result="insufficient", result_note="still only 3 papers after retry")
    assert step.result == "insufficient"


def test_action_must_match_params_kind() -> None:
    with pytest.raises(ValidationError):
        DecisionStep(
            action="search",
            params=ReformulateParams(original_query="a", new_query="b"),
            trigger="t",
            rationale="r",
        )


def test_parent_step_id_builds_causal_chain() -> None:
    search = DecisionStep.start(
        action="search",
        params=SearchParams(query="long context retrieval", sources=["arxiv"]),
        trigger="initial question",
        rationale="first pass retrieval",
    )
    reformulate = DecisionStep.start(
        action="reformulate",
        params=ReformulateParams(original_query="long context retrieval", new_query="x"),
        trigger="only 2 papers",
        rationale="broaden",
        parent_step_id=search.step_id,
    )
    assert reformulate.parent_step_id == search.step_id


def test_tier_transition_is_referenceable_for_ui() -> None:
    step = DecisionStep.start(
        action="hunt_support",
        params=GapHuntParams(claim_id="c7", search_terms=["scaling laws", "chinchilla"]),
        trigger="claim c7 is ungrounded",
        rationale="search for a corroborating paper",
    )
    step.complete(
        result="ok",
        effect=StepEffect(
            claim_ref="c7",
            tier_before="none",
            tier_after="corroborated",
            added_paper_ids=["arxiv:99"],
        ),
    )
    assert step.effect.tier_before == "none"
    assert step.effect.tier_after == "corroborated"
    assert step.effect.claim_ref == "c7"


def test_discriminated_union_round_trips_through_json() -> None:
    step = _reformulate_step()
    step.complete(result="ok", result_note="6 papers")
    dumped = step.model_dump(mode="json")
    assert dumped["params"]["kind"] == "reformulate"
    restored = DecisionStep.model_validate(dumped)
    assert isinstance(restored.params, ReformulateParams)
    assert restored.params.new_query == "long-context attention efficiency"
    assert restored.result == "ok"
