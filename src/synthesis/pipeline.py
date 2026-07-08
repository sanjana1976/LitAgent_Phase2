"""
LitSynth pipeline orchestrator.

Wires together the ten stages defined in the A4 proposal into a single
``run_synthesis`` entry point. Each stage is exposed as a hook so the
orchestrator can be unit-tested by injecting fakes for any stage; in production
the defaults call the real stage modules under :mod:`synthesis`.

The orchestrator is the only synthesis module allowed to:
- mutate SQLite (writes one row to ``synthesis_runs`` per completed run),
- touch the network outside of A3 tools (via the LLM-backed stages).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from db.database import Database, DatabaseError
from db.queries import insert_synthesis_run
from synthesis.claims import extract_claims
from synthesis.contradictions import detect_contradictions
from synthesis.decompose import decompose_question
from synthesis.fetch_parse import fetch_and_parse
from synthesis.generate import generate_literature_review
from synthesis.prompt import SynthesisPrompt, build_synthesis_prompt
from synthesis.rank import rank_papers
from synthesis.retrieve import retrieve_papers
from synthesis.schemas import (
    ClaimRecord,
    ContradictionPair,
    ResearchQuestion,
    ScoredPaper,
    SynthesisResult,
)
from synthesis.validate_cites import validate_citations
from tools.schemas import Paper

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Tunable knobs for one synthesis run."""

    sub_query_count: int = 4
    per_query_limit: int = 5
    total_paper_limit: int = 12
    top_n_ranked: int = 6
    max_claims_per_paper: int = 4
    max_contradiction_pairs: int = 5
    word_budget: int = 500
    sources: tuple[str, ...] = ("arxiv",)


@dataclass
class PipelineHooks:
    """
    Optional callables for testing.

    Each hook defaults to the matching stage function. Tests can swap any of them
    without monkeypatching module globals.
    """

    decompose: Callable[..., ResearchQuestion] = field(default=decompose_question)
    retrieve: Callable[..., list[Paper]] = field(default=retrieve_papers)
    fetch_parse: Callable[..., list[ScoredPaper]] = field(default=fetch_and_parse)
    rank: Callable[..., list[ScoredPaper]] = field(default=rank_papers)
    extract_claims: Callable[..., list[ClaimRecord]] = field(default=extract_claims)
    detect_contradictions: Callable[..., list[ContradictionPair]] = field(
        default=detect_contradictions
    )
    build_prompt: Callable[..., SynthesisPrompt] = field(default=build_synthesis_prompt)
    generate: Callable[..., str] = field(default=generate_literature_review)
    validate_cites: Callable[..., tuple] = field(default=validate_citations)


def _confidence_with_grounding(citation_score: float, claims: list[ClaimRecord]) -> float:
    """
    Combine the citation validity score with the fraction of grounded claims.

    The final score is the product, which means: a review that cites everything
    correctly but used ungrounded claims is still penalized, and vice versa.
    """
    if not claims:
        grounded_fraction = 0.0
    else:
        grounded_fraction = sum(1 for c in claims if c.grounded) / len(claims)
    score = max(0.0, min(1.0, citation_score * grounded_fraction))
    return round(score, 4)


def run_synthesis(
    question: str,
    *,
    config: PipelineConfig | None = None,
    hooks: PipelineHooks | None = None,
    database: Database | None = None,
    session_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> SynthesisResult:
    """
    Execute the ten-stage synthesis pipeline and return a populated result.

    Args:
        question: The user's raw research question.
        config: Tunable per-run knobs; defaults to :class:`PipelineConfig`.
        hooks: Injectable stage functions; defaults to the real implementations.
        database: When provided, the completed run is persisted to ``synthesis_runs``.
        session_id: Optional session id stored alongside the run for resume support.
        progress: Optional callback invoked with a short status string before each
            stage. Useful for surfacing progress in the CLI without coupling to it.

    The pipeline is fail-soft: a stage that returns nothing (e.g. no papers found,
    LLM unavailable) does not abort the pipeline. The final ``SynthesisResult``
    still includes whatever evidence the earlier stages produced, plus a clear
    review_text explaining what was missing.
    """
    cfg = config or PipelineConfig()
    h = hooks or PipelineHooks()

    def _step(label: str) -> None:
        logger.info("synthesis: %s", label)
        if progress is not None:
            try:
                progress(label)
            except Exception:  # noqa: BLE001
                logger.debug("Progress callback raised; ignoring", exc_info=True)

    _step("decomposing question into sub-queries")
    rq = h.decompose(question, n=cfg.sub_query_count)

    _step(f"retrieving papers across {len(cfg.sources)} source(s)")
    raw_papers = h.retrieve(
        rq,
        per_query_limit=cfg.per_query_limit,
        total_limit=cfg.total_paper_limit,
        sources=cfg.sources,
    )

    if not raw_papers:
        return _empty_result(question, reason="no papers returned by search providers")

    _step(f"fetching + parsing {len(raw_papers)} candidate paper(s)")
    parsed = h.fetch_parse(raw_papers)

    _step(f"ranking by relevance and keeping top {cfg.top_n_ranked}")
    ranked = h.rank(parsed, question=question, top_n=cfg.top_n_ranked)
    if not ranked:
        return _empty_result(question, reason="all retrieved papers failed parsing")

    _step("extracting grounded claims (one LLM call per paper)")
    claims = h.extract_claims(ranked, max_claims_per_paper=cfg.max_claims_per_paper)

    _step("detecting cross-paper contradictions")
    contradictions = h.detect_contradictions(
        claims, ranked, max_pairs=cfg.max_contradiction_pairs
    )

    _step("assembling synthesis prompt")
    prompt = h.build_prompt(
        question=question,
        papers=ranked,
        claims=claims,
        contradictions=contradictions,
        word_budget=cfg.word_budget,
    )

    _step("generating literature review (one LLM call, high token budget)")
    review_text = h.generate(prompt, word_budget=cfg.word_budget)

    _step("validating inline citations against source set")
    citation_checks, citations_used, hallucinated, citation_score = h.validate_cites(
        review_text, ranked
    )

    confidence = _confidence_with_grounding(citation_score, claims)

    result = SynthesisResult(
        question=question,
        review_text=review_text,
        citations_used=citations_used,
        hallucinated_citations=hallucinated,
        contradictions_found=len(contradictions),
        confidence_score=confidence,
        papers=ranked,
        claims=claims,
        contradictions=contradictions,
        citation_checks=list(citation_checks),
    )

    if database is not None:
        _persist(database, session_id, result)

    return result


def _empty_result(question: str, *, reason: str) -> SynthesisResult:
    """Construct a no-papers SynthesisResult with a clear explanation."""
    return SynthesisResult(
        question=question,
        review_text=(
            f"No literature review could be produced for: {question!r}. "
            f"Reason: {reason}. "
            "Try rephrasing the question, expanding sources, or relaxing date filters."
        ),
        citations_used=[],
        hallucinated_citations=[],
        contradictions_found=0,
        confidence_score=0.0,
        papers=[],
        claims=[],
        contradictions=[],
        citation_checks=[],
    )


def _persist(database: Database, session_id: str | None, result: SynthesisResult) -> None:
    """
    Write one ``synthesis_runs`` row without trace metadata.

    Prefer :func:`synthesis.persistence.persist_synthesis_state` for agentic runs.
    """
    try:
        payload = result.model_dump(mode="json")
        insert_synthesis_run(
            database,
            session_id=session_id,
            question=result.question,
            review_text=result.review_text,
            result_json=json.dumps(payload, ensure_ascii=False),
            confidence_score=result.confidence_score,
            contradictions_found=result.contradictions_found,
            hallucinated_count=len(result.hallucinated_citations),
        )
    except DatabaseError:
        logger.exception("Failed to persist synthesis run; continuing.")
