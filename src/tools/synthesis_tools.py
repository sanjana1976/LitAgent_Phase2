"""
Agent-callable tools that expose the agentic LitSynth (A4) controller.

Registered with the existing A3 tool registry so the chat REPL can invoke
literature-review synthesis the same way it invokes search and analysis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from db.queries import get_latest_synthesis_run_for_session
from synthesis.graph import SynthesisConfig, run_graph_synthesis
from synthesis.persistence import synthesis_result_from_json
from synthesis.schemas import SynthesisResult
from tools.context import get_default_database, get_tool_session_id

logger = logging.getLogger(__name__)


def _result_to_compact_dict(result: SynthesisResult) -> dict[str, Any]:
    """
    Trim the full result to a chat-friendly summary the model can quote back.

    The full result is still persisted to SQLite by the controller; this is the
    shape the agent sees in its tool message.
    """
    return {
        "question": result.question,
        "review_text": result.review_text,
        "papers": [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "citation_key": p.short_citation_key(),
                "url": p.url,
                "year": p.year,
                "venue": p.venue,
                "relevance_score": round(p.relevance_score, 4),
                "has_pdf": p.has_pdf,
            }
            for p in result.papers
        ],
        "citations_used": result.citations_used,
        "hallucinated_citations": result.hallucinated_citations,
        "contradictions_found": result.contradictions_found,
        "contradictions": [
            {
                "paper_a": c.paper_a,
                "paper_b": c.paper_b,
                "tension_type": c.tension_type,
                "explanation": c.explanation,
            }
            for c in result.contradictions
        ],
        "confidence_score": result.confidence_score,
        "total_claims": len(result.claims),
        "grounded_claims": sum(1 for c in result.claims if c.grounded),
    }


def tool_synthesize_literature_review(
    question: str,
    *,
    word_budget: int = 500,
    top_n: int = 6,
    sources: list[str] | None = None,
) -> str:
    """
    Generate a structured literature-review section for ``question``.

    Runs the full LitSynth controller (retrieval → claims → contradiction
    detection → gap hunting → conflict resolution → synthesis) and returns a
    JSON string the model can quote back to the user.
    The run is persisted to ``synthesis_runs`` so it can be inspected later via
    ``python main.py synth-history``.

    Args:
        question: Raw research question, e.g. "competing approaches to long-context retrieval".
        word_budget: Approximate prose length for the review (default 500 words).
        top_n: Maximum parsed-paper working set for the controller (default 6).
        sources: Optional list of search sources (``"arxiv"``, ``"semantic_scholar"``,
            ``"dblp"``, ``"crossref"``). Defaults to arxiv + semantic_scholar.

    Returns:
        JSON-encoded summary including the review text, cited papers, contradictions,
        hallucination flags, and confidence score.
    """
    q = (question or "").strip()
    if not q:
        return json.dumps(
            {"error": "question must be a non-empty string", "question": question}
        )

    paper_limit = max(2, min(12, int(top_n)))
    cfg = SynthesisConfig(
        word_budget=max(150, min(2000, int(word_budget))),
        min_relevant_papers=min(4, paper_limit),
        total_paper_limit=paper_limit,
        sources=tuple(sources) if sources else SynthesisConfig().sources,
    )

    try:
        database = get_default_database()
    except Exception:  # noqa: BLE001
        logger.exception("Could not resolve default database; running without persistence")
        database = None

    result = run_graph_synthesis(
        q,
        config=cfg,
        database=database,
        session_id=get_tool_session_id(),
    )
    return json.dumps(_result_to_compact_dict(result), ensure_ascii=False)


def tool_get_review_context() -> str:
    """
    Recall the papers and topic from the most recent literature review in this session.

    Use this when the user asks a follow-up about a previous review — "those
    papers", "the second paper", "which of them evaluate on X" — and the
    paper identifiers are not already visible in the conversation. Returns the
    same JSON shape as ``tool_synthesize_literature_review`` (question,
    review_text, papers with paper_id/citation_key/url, contradictions), or an
    ``error`` field when no review has been generated yet.
    """
    try:
        database = get_default_database()
    except Exception:  # noqa: BLE001
        logger.exception("Could not resolve default database for review recall")
        return json.dumps({"error": "database unavailable; no review context to recall"})

    session_id = get_tool_session_id()
    try:
        row = get_latest_synthesis_run_for_session(database, session_id)
        if row is None and session_id:
            # Fall back to the newest run overall so a fresh session can still
            # pick up the user's last review.
            row = get_latest_synthesis_run_for_session(database, None)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load latest synthesis run")
        return json.dumps({"error": "could not load a previous review from the database"})

    if row is None:
        return json.dumps(
            {"error": "no literature review has been generated yet in any session"}
        )

    try:
        result = synthesis_result_from_json(str(row["result_json"]))
    except Exception:  # noqa: BLE001
        logger.exception("Persisted synthesis run %s is unreadable", row.get("id"))
        return json.dumps({"error": "the stored review could not be parsed"})

    payload = _result_to_compact_dict(result)
    payload["run_id"] = row.get("id")
    payload["created_at"] = row.get("created_at")
    payload["from_session"] = row.get("session_id")
    return json.dumps(payload, ensure_ascii=False)
