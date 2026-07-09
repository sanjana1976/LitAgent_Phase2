"""
LLM relevance gate for the agentic LitSynth controller.

TF-IDF ranking (stage 4) can order papers, but with a small corpus it cannot
*reject* an off-topic paper: generic ML vocabulary ("model", "training",
"benchmark") produces nonzero cosine similarity for almost anything. This
module adds a semantic gate: one JSON LLM call that scores every candidate
paper's title + abstract against the research question on a 0-10 scale, so
papers that merely share vocabulary with the question are dropped before the
expensive claim-extraction and generation stages run.

Fail-soft contract: if the LLM is unavailable or returns garbage, the input
papers are returned unchanged (``used_llm=False``) so a missing API key or a
transient outage degrades to the old TF-IDF-only behavior instead of erasing
the working set.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from synthesis.llm import SynthesisLLMError, call_json
from synthesis.schemas import ScoredPaper

logger = logging.getLogger(__name__)


_ABSTRACT_PREVIEW_CHARS = 350
_DEFAULT_KEEP_THRESHOLD = 6

_SYSTEM_PROMPT = (
    "You are a strict research librarian judging whether candidate papers "
    "belong in a literature review for a specific research question. For "
    "each paper, score topical relevance from 0 to 10: 10 means the paper "
    "directly addresses the question's core topic; 5 means it is related "
    "but tangential; 0 means it is off-topic. Judge only from the title and "
    "abstract given. Sharing generic vocabulary (model, benchmark, training) "
    "is NOT relevance. Respond with a single JSON object of the form "
    '{"scores": [{"paper_id": "...", "score": 0}, ...]} covering every '
    "paper, with no explanations outside the JSON."
)


def _build_user_prompt(question: str, papers: list[ScoredPaper]) -> str:
    lines: list[str] = [f"Research question: {question.strip()}", "", "Candidate papers:"]
    for paper in papers:
        abstract = (paper.abstract or "").strip().replace("\n", " ")
        if len(abstract) > _ABSTRACT_PREVIEW_CHARS:
            abstract = abstract[:_ABSTRACT_PREVIEW_CHARS].rstrip() + "…"
        lines.append(f"- paper_id: {paper.paper_id}")
        lines.append(f"  title: {paper.title}")
        if abstract:
            lines.append(f"  abstract: {abstract}")
    lines.append("")
    lines.append(
        'Return JSON: {"scores": [{"paper_id": "...", "score": 0-10}, ...]} '
        "with one entry per paper."
    )
    return "\n".join(lines)


def _parse_scores(payload: dict[str, Any]) -> dict[str, int]:
    """Extract a paper_id -> clamped-int-score map from the LLM payload."""
    raw = payload.get("scores") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return {}
    scores: dict[str, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("paper_id", "") or "").strip()
        if not pid:
            continue
        try:
            score = int(float(item.get("score", 0)))
        except (TypeError, ValueError):
            continue
        scores[pid] = max(0, min(10, score))
    return scores


def llm_relevance_filter(
    question: str,
    papers: list[ScoredPaper],
    *,
    keep_threshold: int = _DEFAULT_KEEP_THRESHOLD,
    llm_call: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[ScoredPaper], dict[str, int], bool]:
    """
    Keep only papers the LLM scores at or above ``keep_threshold``.

    Args:
        question: The user's research question.
        papers: Candidate papers (title + abstract are what gets judged).
        keep_threshold: Minimum 0-10 relevance score a paper must reach.
        llm_call: Injectable JSON LLM caller with the same signature as
            :func:`synthesis.llm.call_json`. Defaults to ``call_json``.

    Returns:
        ``(kept_papers, scores, used_llm)``. ``scores`` maps paper_id to the
        model's 0-10 judgment. Papers the model failed to score are kept
        (conservative: an omission is not evidence of irrelevance). On any
        LLM failure the input list is returned unchanged with
        ``used_llm=False``.
    """
    if not papers:
        return [], {}, False

    invoker = llm_call if llm_call is not None else call_json
    try:
        payload = invoker(
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(question, papers),
            temperature=0.0,
            max_tokens=1_024,
        )
    except SynthesisLLMError as exc:
        logger.warning("Relevance gate LLM call failed; keeping all papers: %s", exc)
        return list(papers), {}, False
    except Exception as exc:  # noqa: BLE001 - the gate must never abort the run
        logger.warning(
            "Relevance gate raised %s; keeping all papers", type(exc).__name__
        )
        return list(papers), {}, False

    scores = _parse_scores(payload)
    if not scores:
        logger.warning("Relevance gate returned no usable scores; keeping all papers")
        return list(papers), {}, False

    threshold = max(0, min(10, int(keep_threshold)))
    kept = [
        paper
        for paper in papers
        if scores.get(paper.paper_id) is None or scores[paper.paper_id] >= threshold
    ]
    dropped = len(papers) - len(kept)
    if dropped:
        logger.info(
            "Relevance gate dropped %d of %d paper(s) below score %d",
            dropped,
            len(papers),
            threshold,
        )
    return kept, scores, True
