"""
Persist and reload agentic synthesis runs including the full decision trace.

The ``synthesis_runs.result_json`` column stores the legacy
:class:`~synthesis.schemas.SynthesisResult` fields plus trace metadata so
Streamlit and ``synth-history`` can replay every decision step.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from db.database import Database, DatabaseError
from db.queries import insert_synthesis_run
from synthesis.schemas import SynthesisResult
from synthesis.state import Gap, SynthesisState, TerminalReason
from synthesis.trace import DecisionStep

logger = logging.getLogger(__name__)


def build_persist_payload(state: SynthesisState) -> dict[str, Any]:
    """Merge deliverable fields with trace metadata for SQLite storage."""
    payload = state.to_result().model_dump(mode="json")
    payload["trace"] = [step.model_dump(mode="json") for step in state.trace]
    payload["gaps"] = [gap.model_dump(mode="json") for gap in state.gaps]
    payload["terminal_reason"] = state.terminal_reason
    payload["session_id"] = state.session_id
    payload["sub_queries"] = list(state.sub_queries)
    payload["active_paper_ids"] = list(state.active_paper_ids)
    return payload


def persist_synthesis_state(
    database: Database,
    session_id: str | None,
    state: SynthesisState,
) -> None:
    """Write one ``synthesis_runs`` row. Failures must not crash callers."""
    result = state.to_result()
    try:
        insert_synthesis_run(
            database,
            session_id=session_id or state.session_id,
            question=result.question,
            review_text=result.review_text,
            result_json=json.dumps(build_persist_payload(state), ensure_ascii=False),
            confidence_score=result.confidence_score,
            contradictions_found=result.contradictions_found,
            hallucinated_count=len(result.hallucinated_citations),
        )
    except DatabaseError:
        logger.exception("Failed to persist synthesis run; continuing.")


def synthesis_result_from_json(raw: str) -> SynthesisResult:
    """Parse persisted JSON into a ``SynthesisResult`` (ignores trace extras)."""
    data = json.loads(raw)
    trace_keys = {
        "trace",
        "gaps",
        "terminal_reason",
        "session_id",
        "sub_queries",
        "active_paper_ids",
    }
    result_data = {k: v for k, v in data.items() if k not in trace_keys}
    return SynthesisResult.model_validate(result_data)


def load_synthesis_state_from_json(raw: str) -> SynthesisState | None:
    """
    Reconstruct ``SynthesisState`` when trace metadata was persisted.

    Returns ``None`` for legacy rows that only store the flat result artifact.
    """
    data = json.loads(raw)
    if "trace" not in data:
        return None

    trace = [DecisionStep.model_validate(item) for item in data.get("trace", [])]
    gaps = [Gap.model_validate(item) for item in data.get("gaps", [])]
    terminal = data.get("terminal_reason")
    terminal_reason: TerminalReason | None = terminal if terminal else None

    result = SynthesisResult.model_validate(
        {
            k: v
            for k, v in data.items()
            if k not in {"trace", "gaps", "terminal_reason", "sub_queries", "active_paper_ids"}
        }
    )

    return SynthesisState(
        question=result.question,
        sub_queries=list(data.get("sub_queries") or []),
        session_id=data.get("session_id"),
        papers=list(result.papers),
        active_paper_ids=list(data.get("active_paper_ids") or []),
        claims=list(result.claims),
        contradictions=list(result.contradictions),
        gaps=gaps,
        review_text=result.review_text or None,
        citation_checks=list(result.citation_checks),
        hallucinated_citations=list(result.hallucinated_citations),
        trace=trace,
        terminal_reason=terminal_reason,
        created_at=result.created_at,
    )
