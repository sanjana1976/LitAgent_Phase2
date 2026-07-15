"""
LangGraph orchestration for LitSynth.

Phase 1 of the migration described in Design.md: a linear StateGraph over
:class:`~synthesis.state.SynthesisState` whose nodes wrap the existing stage
functions. The graph is the orchestration layer only — every stage body
(decompose, retrieve, rank, relevance gate, claims, contradictions, prompt,
generate, validate) stays where it is and keeps its injectable-LLM testing
seam.

Architecture invariants:

- **State is the schema, the graph is the control flow.** ``SynthesisState``
  fields carry LangGraph reducers (append-only ``papers`` deduped by id,
  upsert-by-id ``claims``/``contradictions``/``gaps``, append ``trace``);
  which papers feed downstream stages is the separate ``active_paper_ids``
  selection, so filtering never destroys evidence.
- **Nodes are pure with respect to state**: they read the typed state, call
  stage functions, and return partial update dicts. Item updates (e.g. a gap
  hunt upgrading a claim) return modified *copies* that the upsert reducer
  merges — no shared-object mutation, which keeps phase-3 parallel fan-out
  safe.
- **Every node emits completed :class:`DecisionStep` records** into the trace
  via its reducer, preserving the decision-trace UI and eval harness. Steps
  are born completed: in a graph, deciding is the edges and executing is the
  nodes, so the old two-phase pending->complete arc collapses.
- **Dependencies flow through the LangGraph config channel**
  (``config["configurable"]["litsynth_deps"]`` / ``"litsynth_config"``), not
  module globals — tests inject stubs the same way production injects real
  stage functions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from synthesis.rank import rank_papers
from synthesis.schemas import (
    ClaimRecord,
    ContradictionPair,
    ResearchQuestion,
    ScoredPaper,
    SynthesisResult,
)
from synthesis.state import Gap, SynthesisState
from synthesis.trace import (
    DecisionStep,
    DecomposeParams,
    DetectContradictionsParams,
    ExtractClaimsParams,
    FilterRelevanceParams,
    GapHuntParams,
    ReformulateParams,
    ResolveConflictParams,
    SearchParams,
    StepEffect,
    SynthesizeParams,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config + dependency injection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SynthesisConfig:
    """Policy knobs for one graph-driven synthesis run."""

    sub_query_count: int = 4
    min_relevant_papers: int = 4
    max_reformulations: int = 2
    per_query_limit: int = 5
    total_paper_limit: int = 12
    max_claims_per_paper: int = 4
    max_contradiction_pairs: int = 5
    word_budget: int = 500
    gap_search_terms: int = 6
    min_relevance_score: float = 0.03
    relevance_keep_threshold: int = 6
    hunt_paper_limit: int = 5
    max_concurrency: int = 4
    sources: tuple[str, ...] = ("arxiv",)


def _real_decompose(*args: Any, **kwargs: Any):
    from synthesis.decompose import decompose_question

    return decompose_question(*args, **kwargs)


def _real_retrieve(*args: Any, **kwargs: Any):
    from synthesis.retrieve import retrieve_papers

    return retrieve_papers(*args, **kwargs)


def _real_fetch_parse(*args: Any, **kwargs: Any):
    from synthesis.fetch_parse import fetch_and_parse

    return fetch_and_parse(*args, **kwargs)


def _real_relevance_filter(*args: Any, **kwargs: Any):
    from synthesis.relevance import llm_relevance_filter

    return llm_relevance_filter(*args, **kwargs)


def _real_extract_claims(*args: Any, **kwargs: Any):
    from synthesis.claims import extract_claims

    return extract_claims(*args, **kwargs)


def _real_detect_contradictions(*args: Any, **kwargs: Any):
    from synthesis.contradictions import detect_contradictions

    return detect_contradictions(*args, **kwargs)


def _real_build_prompt(*args: Any, **kwargs: Any):
    from synthesis.prompt import build_synthesis_prompt

    return build_synthesis_prompt(*args, **kwargs)


def _real_generate(*args: Any, **kwargs: Any):
    from synthesis.generate import generate_literature_review

    return generate_literature_review(*args, **kwargs)


def _real_validate_cites(*args: Any, **kwargs: Any):
    from synthesis.validate_cites import validate_citations

    return validate_citations(*args, **kwargs)


def _real_reformulate(*args: Any, **kwargs: Any):
    from synthesis.reformulate import resolve_reformulated_query

    return resolve_reformulated_query(*args, **kwargs)


@dataclass
class SynthesisDeps:
    """
    Injectable stage functions for the graph nodes.

    Defaults lazily resolve the real stage implementations (mirroring the old
    ``ControllerHooks`` seam) so unit tests can swap any stage without
    monkeypatching or network access.
    """

    decompose: Callable[..., ResearchQuestion] = field(default=_real_decompose)
    retrieve: Callable[..., list[Any]] = field(default=_real_retrieve)
    fetch_parse: Callable[..., list[ScoredPaper]] = field(default=_real_fetch_parse)
    relevance_filter: Callable[..., tuple] = field(default=_real_relevance_filter)
    extract_claims: Callable[..., list[ClaimRecord]] = field(default=_real_extract_claims)
    detect_contradictions: Callable[..., list[ContradictionPair]] = field(
        default=_real_detect_contradictions
    )
    build_prompt: Callable[..., Any] = field(default=_real_build_prompt)
    generate: Callable[..., str] = field(default=_real_generate)
    validate_cites: Callable[..., tuple] = field(default=_real_validate_cites)
    reformulate: Callable[..., str] = field(default=_real_reformulate)


def _ctx(config: RunnableConfig | None) -> tuple[SynthesisConfig, SynthesisDeps]:
    """Pull the run config and dependency bundle out of the LangGraph config channel."""
    configurable = (config or {}).get("configurable", {})
    cfg = configurable.get("litsynth_config") or SynthesisConfig()
    deps = configurable.get("litsynth_deps") or SynthesisDeps()
    return cfg, deps


# --------------------------------------------------------------------------- #
# Trace + keyword helpers
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "is", "are", "be", "by", "that", "this", "these", "those", "we", "our",
        "it", "its", "as", "at", "from", "than", "using", "use", "can", "which",
        "their", "they", "show", "shows", "shown", "achieves", "achieve",
    }
)


def derive_search_terms(claim_text: str, *, limit: int = 6) -> list[str]:
    """Deterministically pick salient keywords from a claim for a targeted hunt."""
    seen: set[str] = set()
    terms: list[str] = []
    for raw in claim_text.lower().split():
        word = "".join(ch for ch in raw if ch.isalnum())
        if len(word) < 4 or word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if len(terms) >= limit:
            break
    return terms


def _completed_step(
    *,
    action: str,
    params: Any,
    trigger: str,
    rationale: str,
    result: str,
    result_note: str = "",
    parent_step_id: str | None = None,
    effect: StepEffect | None = None,
    llm_calls: int = 0,
    started: float | None = None,
) -> DecisionStep:
    """Create a DecisionStep that is born completed (single-phase, graph-native)."""
    step = DecisionStep.start(
        action=action,  # type: ignore[arg-type]
        params=params,
        trigger=trigger,
        rationale=rationale,
        parent_step_id=parent_step_id,
    )
    step.complete(
        result=result,  # type: ignore[arg-type]
        result_note=result_note,
        effect=effect,
        llm_calls=llm_calls,
        duration_ms=_elapsed_ms(started) if started is not None else None,
    )
    return step


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _last_step_id(state: SynthesisState) -> str | None:
    return state.trace[-1].step_id if state.trace else None


def _single_query_rq(terms: list[str], fallback: str) -> ResearchQuestion:
    """One combined hunt query (a term list would fan out per-word searches)."""
    query = " ".join(terms).strip()
    if len(query) < 3:
        query = fallback
    return ResearchQuestion(question=query, sub_queries=[query])


def _stamped(claim: ClaimRecord, paper: ScoredPaper) -> ClaimRecord:
    """Return a grounding-stamped copy of ``claim`` based on its paper's text tier."""
    if claim.grounded:
        tier = paper.text_tier if paper.text_tier in ("full_text", "abstract") else "abstract"
        return claim.model_copy(
            update={"grounding_tier": tier, "supporting_paper_id": paper.paper_id}
        )
    return claim.model_copy(
        update={"grounding_tier": "none", "supporting_paper_id": None}
    )


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def decompose_node(state: SynthesisState, config: RunnableConfig) -> dict[str, Any]:
    """Split the question into keyword sub-queries (LLM with deterministic fallback)."""
    cfg, deps = _ctx(config)
    started = time.perf_counter()
    rq = deps.decompose(state.question, n=cfg.sub_query_count)
    sub_queries = list(rq.sub_queries)
    step = _completed_step(
        action="decompose",
        params=DecomposeParams(question=state.question, n=cfg.sub_query_count),
        trigger="no sub-queries have been generated yet",
        rationale="decompose the question before retrieval so search covers multiple angles",
        result="ok" if sub_queries else "insufficient",
        result_note=f"generated {len(sub_queries)} sub-query(s)",
        parent_step_id=_last_step_id(state),
        started=started,
    )
    return {"sub_queries": sub_queries, "trace": [step]}


def _searched_queries(state: SynthesisState) -> set[str]:
    """Queries already issued, derived from the trace (never stored separately)."""
    return {
        s.params.query
        for s in state.trace
        if s.action == "search" and s.params.kind == "search"
    }


def plan_search_node(state: SynthesisState, config: RunnableConfig) -> dict[str, Any]:
    """No-op dispatch anchor; :func:`fan_out_searches` on its outgoing edge does the work."""
    return {}


def fan_out_searches(state: SynthesisState, config: RunnableConfig):
    """
    Fan out one parallel ``search_one`` branch per not-yet-searched sub-query.

    The searched-set is derived from the trace, so when the reformulation
    cycle re-enters, only the newly added query spawns a branch. With nothing
    pending, control skips straight to the ranking join.
    """
    searched = _searched_queries(state)
    pending = [
        q
        for q in (state.sub_queries or [state.question])
        if q.strip() and q not in searched
    ]
    if not pending:
        return "rank_pool"
    known_ids = [p.paper_id for p in state.papers]
    parent = _last_step_id(state)
    return [
        Send(
            "search_one",
            {
                "query": query,
                "question": state.question,
                "known_ids": known_ids,
                "parent_step_id": parent,
            },
        )
        for query in pending
    ]


def search_one_node(payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """
    One parallel search branch: retrieve + fetch/parse a single sub-query.

    Branches only append (papers via the upsert reducer, one trace step);
    ranking and ``active_paper_ids`` selection happen once in the
    ``rank_pool`` join after every branch has merged. Two branches that find
    the same paper concurrently are deduplicated by the reducer.
    """
    cfg, deps = _ctx(config)
    query: str = payload["query"]
    question: str = payload["question"]
    known_ids = set(payload.get("known_ids") or [])
    started = time.perf_counter()

    try:
        rq = ResearchQuestion(question=question, sub_queries=[query])
        raw = deps.retrieve(
            rq,
            per_query_limit=cfg.per_query_limit,
            total_limit=cfg.total_paper_limit,
            sources=cfg.sources,
        )
        raw_new = [p for p in raw if p.paper_id not in known_ids]
        parsed = deps.fetch_parse(raw_new)
        new_papers = [p for p in parsed if p.paper_id not in known_ids]
    except Exception as exc:  # noqa: BLE001 - one branch must not kill the sweep
        step = _completed_step(
            action="search",
            params=SearchParams(query=query, sources=list(cfg.sources)),
            trigger=f"sub-query {query!r} has not been searched yet",
            rationale="search every decomposed angle before judging coverage",
            result="failed",
            result_note=f"{type(exc).__name__}: {exc}",
            parent_step_id=payload.get("parent_step_id"),
            started=started,
        )
        return {"trace": [step]}

    step = _completed_step(
        action="search",
        params=SearchParams(query=query, sources=list(cfg.sources)),
        trigger=f"sub-query {query!r} has not been searched yet",
        rationale="search every decomposed angle before judging coverage",
        result="ok" if new_papers else "noop",
        result_note=(
            f"retrieved {len(raw)} candidate(s), added {len(new_papers)} new paper(s)"
        ),
        parent_step_id=payload.get("parent_step_id"),
        effect=StepEffect(added_paper_ids=[p.paper_id for p in new_papers]),
        started=started,
    )
    return {"papers": new_papers, "trace": [step]}


def rank_pool_node(state: SynthesisState, config: RunnableConfig) -> dict[str, Any]:
    """
    Join after the search fan-out: rank the merged pool into ``active_paper_ids``.

    Runs exactly once per retrieval round, after every branch has merged, and
    is the sole writer of the selection during retrieval — parallel branches
    never touch replace-semantics fields.
    """
    cfg, _ = _ctx(config)
    if not state.papers:
        return {}
    ranked = rank_papers(
        list(state.papers),
        question=state.question,
        top_n=cfg.total_paper_limit,
        min_score=cfg.min_relevance_score,
    )
    return {"active_paper_ids": [p.paper_id for p in ranked]}


def reformulate_node(state: SynthesisState, config: RunnableConfig) -> dict[str, Any]:
    """Rewrite the latest query when retrieval came back thin (LLM with fallback)."""
    cfg, deps = _ctx(config)
    original = state.sub_queries[-1] if state.sub_queries else state.question
    started = time.perf_counter()

    try:
        new_query = deps.reformulate(state, original)
    except Exception as exc:  # noqa: BLE001 - a failed rewrite must not abort the run
        step = _completed_step(
            action="reformulate",
            params=ReformulateParams(original_query=original, new_query=""),
            trigger=(
                f"retrieval found only {len(state.active_paper_ids)} paper(s), below "
                f"the target of {cfg.min_relevant_papers}"
            ),
            rationale="broaden and retarget the query before another search",
            result="failed",
            result_note=f"{type(exc).__name__}: {exc}",
            parent_step_id=_last_step_id(state),
            started=started,
        )
        return {"trace": [step]}

    trigger = (
        f"retrieval found only {len(state.active_paper_ids)} paper(s), below "
        f"the target of {cfg.min_relevant_papers}"
    )
    if new_query and new_query not in state.sub_queries:
        step = _completed_step(
            action="reformulate",
            params=ReformulateParams(original_query=original, new_query=new_query),
            trigger=trigger,
            rationale="broaden and retarget the query before another search",
            result="ok",
            result_note=f"reformulated query to {new_query!r}",
            parent_step_id=_last_step_id(state),
            started=started,
        )
        return {"sub_queries": list(state.sub_queries) + [new_query], "trace": [step]}

    step = _completed_step(
        action="reformulate",
        params=ReformulateParams(original_query=original, new_query=new_query or ""),
        trigger=trigger,
        rationale="broaden and retarget the query before another search",
        result="noop",
        result_note="reformulation produced no new query",
        parent_step_id=_last_step_id(state),
        started=started,
    )
    return {"trace": [step]}


def route_after_search(state: SynthesisState, config: RunnableConfig) -> str:
    """
    Coverage router: the graph-native successor of the old ``next_action`` policy.

    Enough relevant papers -> proceed to the relevance gate. Thin coverage with
    reformulation budget left -> reformulate (which cycles back into search).
    Budget exhausted -> proceed with whatever evidence exists; the synthesize
    node derives the ``reformulation_cap`` terminal reason from the trace.
    """
    cfg, _ = _ctx(config)
    if len(state.active_paper_ids) >= cfg.min_relevant_papers:
        return "filter_relevance"
    if state.reformulation_count >= cfg.max_reformulations:
        return "filter_relevance"
    return "reformulate"


def route_after_gate(state: SynthesisState, config: RunnableConfig):
    """
    Post-gate coverage check: a gutted working set re-enters the retrieval cycle.

    The gate can legitimately drop most of the pool (tangential retrieval), in
    which case pressing on would synthesize a "review" of one or two papers.
    While reformulation budget remains, go find better evidence instead.
    Otherwise fan out one parallel ``extract_one`` branch per active paper.
    """
    cfg, _ = _ctx(config)
    thin = len(state.active_paper_ids) < cfg.min_relevant_papers
    if thin and state.reformulation_count < cfg.max_reformulations:
        return "reformulate"
    active = state.active_papers()
    if not active:
        return "detect_contradictions"
    return [Send("extract_one", {"paper": paper}) for paper in active]


def filter_relevance_node(state: SynthesisState, config: RunnableConfig) -> dict[str, Any]:
    """Semantically gate the active set against the question (fail-soft)."""
    cfg, deps = _ctx(config)
    active = state.active_papers()
    if not active:
        return {}

    started = time.perf_counter()
    params = FilterRelevanceParams(
        paper_ids=[p.paper_id for p in active],
        keep_threshold=cfg.relevance_keep_threshold,
    )
    trigger = f"retrieval gathered {len(active)} candidate paper(s)"
    rationale = "drop off-topic papers before claim extraction and synthesis"
    parent = _last_step_id(state)

    try:
        kept, _scores, used_llm = deps.relevance_filter(
            state.question,
            list(active),
            keep_threshold=cfg.relevance_keep_threshold,
        )
    except Exception as exc:  # noqa: BLE001 - the gate must not abort the run
        step = _completed_step(
            action="filter_relevance",
            params=params,
            trigger=trigger,
            rationale=rationale,
            result="failed",
            result_note=f"{type(exc).__name__}: {exc}",
            parent_step_id=parent,
            started=started,
        )
        return {"trace": [step]}

    if not used_llm:
        step = _completed_step(
            action="filter_relevance",
            params=params,
            trigger=trigger,
            rationale=rationale,
            result="noop",
            result_note="relevance LLM unavailable; keeping all papers",
            parent_step_id=parent,
            started=started,
        )
        return {"trace": [step]}

    kept_ids = [p.paper_id for p in kept]
    dropped = [p.paper_id for p in active if p.paper_id not in set(kept_ids)]
    step = _completed_step(
        action="filter_relevance",
        params=params,
        trigger=trigger,
        rationale=rationale,
        result="ok",
        result_note=(
            f"kept {len(kept_ids)} paper(s), dropped {len(dropped)} below "
            f"score {cfg.relevance_keep_threshold}: {dropped}"
            if dropped
            else f"all {len(kept_ids)} paper(s) judged relevant"
        ),
        parent_step_id=parent,
        llm_calls=1,
        started=started,
    )
    return {"active_paper_ids": kept_ids, "trace": [step]}


def extract_one_node(payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """One parallel claim-extraction branch: a single paper, a single LLM call."""
    cfg, deps = _ctx(config)
    paper: ScoredPaper = payload["paper"]
    started = time.perf_counter()

    try:
        raw_claims = deps.extract_claims(
            [paper], max_claims_per_paper=cfg.max_claims_per_paper
        )
    except Exception as exc:  # noqa: BLE001 - one branch must not kill the stage
        step = _completed_step(
            action="extract_claims",
            params=ExtractClaimsParams(paper_ids=[paper.paper_id]),
            trigger="paper available for claim extraction",
            rationale="extract grounded, paper-specific claims from this paper",
            result="failed",
            result_note=f"{type(exc).__name__}: {exc}",
            started=started,
        )
        return {"trace": [step]}

    stamped = [_stamped(c, paper) for c in raw_claims]
    step = _completed_step(
        action="extract_claims",
        params=ExtractClaimsParams(paper_ids=[paper.paper_id]),
        trigger="paper available for claim extraction",
        rationale="extract grounded, paper-specific claims from this paper",
        result="ok" if stamped else "noop",
        result_note=f"extracted {len(stamped)} claim(s) from {paper.paper_id}",
        effect=StepEffect(claim_refs=[c.claim_id for c in stamped]),
        llm_calls=1,
        started=started,
    )
    return {"claims": stamped, "trace": [step]}


def detect_contradictions_node(
    state: SynthesisState, config: RunnableConfig
) -> dict[str, Any]:
    """Detect cross-paper tensions over the accumulated claim set."""
    cfg, deps = _ctx(config)
    started = time.perf_counter()
    last_extract = next(
        (s.step_id for s in reversed(state.trace) if s.action == "extract_claims"),
        None,
    )
    params = DetectContradictionsParams(claim_ids=[c.claim_id for c in state.claims])
    trigger = "claims extracted, contradiction detection running"
    rationale = "surface genuine cross-paper tensions before resolution"

    try:
        pairs = deps.detect_contradictions(
            list(state.claims),
            state.active_papers(),
            max_pairs=cfg.max_contradiction_pairs,
        )
    except Exception as exc:  # noqa: BLE001 - detection must not abort the run
        step = _completed_step(
            action="detect_contradictions",
            params=params,
            trigger=trigger,
            rationale=rationale,
            result="failed",
            result_note=f"{type(exc).__name__}: {exc}",
            parent_step_id=last_extract,
            started=started,
        )
        return {"trace": [step]}

    stamped_pairs = [
        pair if pair.contradiction_id else pair.model_copy(update={"contradiction_id": uuid4().hex})
        for pair in pairs
    ]
    step = _completed_step(
        action="detect_contradictions",
        params=params,
        trigger=trigger,
        rationale=rationale,
        result="ok" if stamped_pairs else "noop",
        result_note=f"detected {len(stamped_pairs)} contradiction(s)",
        effect=StepEffect(contradiction_ids=[p.contradiction_id for p in stamped_pairs]),
        parent_step_id=last_extract,
        llm_calls=1,
        started=started,
    )
    return {"contradictions": stamped_pairs, "trace": [step]}


def _claim_extract_step_map(state: SynthesisState) -> dict[str, str]:
    """Map each claim_id to the extract_claims step that produced it."""
    mapping: dict[str, str] = {}
    for s in state.trace:
        if s.action == "extract_claims":
            for cid in s.effect.claim_refs:
                mapping[cid] = s.step_id
    return mapping


def fan_out_hunts(state: SynthesisState, config: RunnableConfig):
    """Fan out one parallel ``hunt_one`` branch per ungrounded claim."""
    ungrounded = [
        c for c in state.claims if not c.grounded and c.grounding_tier == "none"
    ]
    if not ungrounded:
        return "plan_conflicts"
    claim_to_step = _claim_extract_step_map(state)
    known_ids = [p.paper_id for p in state.papers]
    return [
        Send(
            "hunt_one",
            {
                "claim": claim,
                "question": state.question,
                "known_ids": known_ids,
                "parent_step_id": claim_to_step.get(claim.claim_id),
            },
        )
        for claim in ungrounded
    ]


def hunt_one_node(payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """
    One parallel gap hunt: try to corroborate a single ungrounded claim.

    Returns only reducer-managed fields (claim upsert copy, gap, papers,
    trace); the ``consolidate`` join folds any hunt-added papers into
    ``active_paper_ids`` afterward. Two branches adding the same supporting
    paper concurrently are deduplicated by the papers reducer.
    """
    cfg, deps = _ctx(config)
    from synthesis.claims import quote_is_grounded

    claim: ClaimRecord = payload["claim"]
    question: str = payload["question"]
    known_ids = set(payload.get("known_ids") or [])
    parent = payload.get("parent_step_id")

    gap = Gap(
        kind="ungrounded_claim",
        description=f"ungrounded claim: {claim.claim[:80]}",
        origin_claim_ref=claim.claim_id,
        status="hunting",
    )
    terms = derive_search_terms(claim.claim, limit=cfg.gap_search_terms)
    started = time.perf_counter()
    params = GapHuntParams(claim_id=claim.claim_id, search_terms=terms)

    try:
        rq = _single_query_rq(terms, claim.claim)
        raw = deps.retrieve(
            rq,
            per_query_limit=cfg.per_query_limit,
            total_limit=cfg.hunt_paper_limit,
            sources=cfg.sources,
        )
        candidates = rank_papers(
            deps.fetch_parse(raw),
            question=question,
            top_n=cfg.hunt_paper_limit,
            min_score=cfg.min_relevance_score,
        )
    except Exception as exc:  # noqa: BLE001 - one branch must not kill the stage
        gap = gap.model_copy(update={"status": "flagged_unverified"})
        step = _completed_step(
            action="hunt_support",
            params=params,
            trigger="ungrounded claim detected",
            rationale="search for a different paper that grounds this claim",
            result="failed",
            result_note=f"{type(exc).__name__}: {exc}",
            effect=StepEffect(claim_ref=claim.claim_id, gap_ref=gap.gap_id),
            parent_step_id=parent,
            started=started,
        )
        return {"gaps": [gap], "trace": [step]}

    supporting = next(
        (
            cand
            for cand in candidates
            if cand.paper_id != claim.paper_id
            and quote_is_grounded(
                claim.evidence_quote, cand.full_text or cand.abstract or ""
            )
        ),
        None,
    )

    if supporting is None:
        gap = gap.model_copy(update={"status": "flagged_unverified"})
        step = _completed_step(
            action="hunt_support",
            params=params,
            trigger="ungrounded claim detected",
            rationale="search for a different paper that grounds this claim",
            result="insufficient",
            result_note="no candidate paper grounded the claim",
            effect=StepEffect(claim_ref=claim.claim_id, gap_ref=gap.gap_id),
            parent_step_id=parent,
            started=started,
        )
        return {"gaps": [gap], "trace": [step]}

    step_id = uuid4().hex
    upgraded = claim.model_copy(
        update={
            "grounded": True,
            "grounding_tier": "corroborated",
            "supporting_paper_id": supporting.paper_id,
        }
    )
    gap = gap.model_copy(
        update={
            "status": "resolved",
            "resolved_by_paper_id": supporting.paper_id,
            "resolved_by_step_id": step_id,
        }
    )
    added = [supporting.paper_id] if supporting.paper_id not in known_ids else []
    step = _completed_step(
        action="hunt_support",
        params=params,
        trigger="ungrounded claim detected",
        rationale="search for a different paper that grounds this claim",
        result="ok",
        result_note=f"claim corroborated by {supporting.paper_id}",
        effect=StepEffect(
            claim_ref=claim.claim_id,
            tier_before="none",
            tier_after="corroborated",
            added_paper_ids=added,
            gap_ref=gap.gap_id,
        ),
        parent_step_id=parent,
        started=started,
    )
    step.step_id = step_id
    update: dict[str, Any] = {"claims": [upgraded], "gaps": [gap], "trace": [step]}
    if added:
        update["papers"] = [supporting]
    return update


def plan_conflicts_node(state: SynthesisState, config: RunnableConfig) -> dict[str, Any]:
    """No-op join between the hunt fan-in and the conflict fan-out."""
    return {}


def fan_out_conflicts(state: SynthesisState, config: RunnableConfig):
    """Fan out one parallel ``resolve_one`` branch per detected contradiction."""
    if not state.contradictions:
        return "consolidate"
    known_ids = [p.paper_id for p in state.papers]
    return [
        Send(
            "resolve_one",
            {"pair": pair, "question": state.question, "known_ids": known_ids},
        )
        for pair in state.contradictions
    ]


def resolve_one_node(payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """One parallel conflict resolution: hunt a third paper for a single pair."""
    cfg, deps = _ctx(config)
    pair: ContradictionPair = payload["pair"]
    question: str = payload["question"]
    known_ids = set(payload.get("known_ids") or [])

    terms = derive_search_terms(
        f"{pair.claim_a} {pair.claim_b}", limit=cfg.gap_search_terms
    )
    started = time.perf_counter()
    params = ResolveConflictParams(
        paper_a=pair.paper_a, paper_b=pair.paper_b, search_terms=terms
    )
    trigger = f"papers {pair.paper_a} and {pair.paper_b} disagree"
    rationale = "hunt a third paper that contextualizes the disagreement"

    try:
        rq = _single_query_rq(terms, pair.claim_a)
        raw = deps.retrieve(
            rq,
            per_query_limit=cfg.per_query_limit,
            total_limit=cfg.hunt_paper_limit,
            sources=cfg.sources,
        )
        candidates = rank_papers(
            deps.fetch_parse(raw),
            question=question,
            top_n=cfg.hunt_paper_limit,
            min_score=cfg.min_relevance_score,
        )
    except Exception as exc:  # noqa: BLE001 - one branch must not kill the stage
        step = _completed_step(
            action="resolve_conflict",
            params=params,
            trigger=trigger,
            rationale=rationale,
            result="failed",
            result_note=f"{type(exc).__name__}: {exc}",
            effect=StepEffect(resolved_conflict=pair.contradiction_id),
            started=started,
        )
        return {"trace": [step]}

    resolution: ClaimRecord | None = None
    paper_c: ScoredPaper | None = None
    for cand in candidates:
        if cand.paper_id in (pair.paper_a, pair.paper_b):
            continue
        try:
            cand_claims = deps.extract_claims(
                [cand], max_claims_per_paper=cfg.max_claims_per_paper
            )
        except Exception:  # noqa: BLE001
            continue
        stamped = [_stamped(c, cand) for c in cand_claims]
        resolution = next((c for c in stamped if c.grounded), None)
        if resolution is not None:
            paper_c = cand
            break

    if resolution is None or paper_c is None:
        step = _completed_step(
            action="resolve_conflict",
            params=params,
            trigger=trigger,
            rationale=rationale,
            result="insufficient",
            result_note="no third paper grounded a resolution",
            effect=StepEffect(resolved_conflict=pair.contradiction_id),
            started=started,
        )
        return {"trace": [step]}

    added = [paper_c.paper_id] if paper_c.paper_id not in known_ids else []
    step = _completed_step(
        action="resolve_conflict",
        params=params,
        trigger=trigger,
        rationale=rationale,
        result="ok",
        result_note=f"conflict contextualized by {paper_c.paper_id}",
        effect=StepEffect(
            resolved_conflict=pair.contradiction_id,
            added_paper_ids=added,
            claim_ref=resolution.claim_id,
            claim_refs=[resolution.claim_id],
        ),
        started=started,
    )
    update: dict[str, Any] = {"claims": [resolution], "trace": [step]}
    if added:
        update["papers"] = [paper_c]
    return update


def consolidate_node(state: SynthesisState, config: RunnableConfig) -> dict[str, Any]:
    """
    Join after the hunt/conflict fan-outs: fold branch-added papers into the
    active set.

    Which papers the branches added is derived from their trace effects, so
    this node is the single writer of ``active_paper_ids`` for the stage —
    parallel branches never touch replace-semantics fields.
    """
    added: list[str] = []
    for step in state.trace:
        if step.action in ("hunt_support", "resolve_conflict"):
            added.extend(step.effect.added_paper_ids)
    if not added:
        return {}
    pool_ids = {p.paper_id for p in state.papers}
    active = list(state.active_paper_ids)
    for pid in added:
        if pid in pool_ids and pid not in active:
            active.append(pid)
    if active == list(state.active_paper_ids):
        return {}
    return {"active_paper_ids": active}


def synthesize_node(state: SynthesisState, config: RunnableConfig) -> dict[str, Any]:
    """Generate the final review from the active evidence set and validate citations."""
    cfg, deps = _ctx(config)
    started = time.perf_counter()
    params = SynthesizeParams(word_budget=cfg.word_budget)
    trigger = "retrieval, claims, gaps, and conflicts have been processed"
    rationale = "generate the final literature review from accumulated state"
    parent = _last_step_id(state)

    try:
        papers = rank_papers(
            state.active_papers(),
            question=state.question,
            top_n=cfg.total_paper_limit,
            min_score=cfg.min_relevance_score,
        )
        paper_ids = {p.paper_id for p in papers}
        claims = [c for c in state.claims if c.paper_id in paper_ids]
        contradictions = [
            pair
            for pair in state.contradictions
            if pair.paper_a in paper_ids and pair.paper_b in paper_ids
        ]
        prompt = deps.build_prompt(
            question=state.question,
            papers=papers,
            claims=claims,
            contradictions=contradictions,
            word_budget=cfg.word_budget,
        )
        review_text = deps.generate(prompt, word_budget=cfg.word_budget)
        citation_checks, _used, hallucinated, _score = deps.validate_cites(
            review_text, papers
        )
    except Exception as exc:  # noqa: BLE001 - preserve trace; result stays inspectable
        step = _completed_step(
            action="synthesize",
            params=params,
            trigger=trigger,
            rationale=rationale,
            result="failed",
            result_note=f"{type(exc).__name__}: {exc}",
            parent_step_id=parent,
            started=started,
        )
        return {"trace": [step], "terminal_reason": "error"}

    step = _completed_step(
        action="synthesize",
        params=params,
        trigger=trigger,
        rationale=rationale,
        result="ok" if review_text.strip() else "insufficient",
        result_note=(
            f"generated review with {len(list(citation_checks))} citation check(s) "
            f"and {len(list(hallucinated))} hallucinated citation(s)"
        ),
        effect=StepEffect(claim_refs=[c.claim_id for c in state.claims]),
        parent_step_id=parent,
        llm_calls=1,
        started=started,
    )
    # Retrieval-phase outcome, derived from the trace: if the run burned its
    # whole reformulation budget and still sits under the coverage target, the
    # honest terminal reason is the cap, even though a review was generated.
    retrieval_thin = (
        len(papers) < cfg.min_relevant_papers
        and state.reformulation_count >= cfg.max_reformulations
    )
    return {
        "review_text": review_text,
        "citation_checks": list(citation_checks),
        "hallucinated_citations": list(hallucinated),
        # Ranked copies carry fresh relevance_score values; the upsert reducer
        # refreshes the matching pool entries so the result reports real scores.
        "papers": papers,
        "active_paper_ids": [p.paper_id for p in papers],
        "terminal_reason": "reformulation_cap" if retrieval_thin else "synthesized",
        "trace": [step],
    }


# --------------------------------------------------------------------------- #
# Graph assembly + entry point
# --------------------------------------------------------------------------- #


def _build_graph_definition() -> StateGraph:
    """
    Assemble the synthesis graph: adaptive-retrieval cycle + parallel fan-outs.

    Map-reduce stages (via the Send API, concurrency-bounded at runtime):

    - ``plan_search``  fans out one ``search_one`` branch per pending sub-query,
      joined by ``rank_pool`` (sole writer of ``active_paper_ids``).
    - the relevance gate fans out one ``extract_one`` branch per active paper.
    - ``detect_contradictions`` fans out ``hunt_one`` per ungrounded claim.
    - ``plan_conflicts`` fans out ``resolve_one`` per contradiction, joined by
      ``consolidate`` which folds branch-added papers into the active set.

    Coverage routers re-enter the reformulation cycle from both the post-search
    rank and the post-gate check.
    """
    g = StateGraph(SynthesisState)
    g.add_node("decompose", decompose_node)
    g.add_node("plan_search", plan_search_node)
    g.add_node("search_one", search_one_node)
    g.add_node("rank_pool", rank_pool_node)
    g.add_node("reformulate", reformulate_node)
    g.add_node("filter_relevance", filter_relevance_node)
    g.add_node("extract_one", extract_one_node)
    g.add_node("detect_contradictions", detect_contradictions_node)
    g.add_node("hunt_one", hunt_one_node)
    g.add_node("plan_conflicts", plan_conflicts_node)
    g.add_node("resolve_one", resolve_one_node)
    g.add_node("consolidate", consolidate_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "decompose")
    g.add_edge("decompose", "plan_search")
    g.add_conditional_edges("plan_search", fan_out_searches)
    g.add_edge("search_one", "rank_pool")
    g.add_conditional_edges(
        "rank_pool",
        route_after_search,
        {"reformulate": "reformulate", "filter_relevance": "filter_relevance"},
    )
    g.add_edge("reformulate", "plan_search")
    g.add_conditional_edges("filter_relevance", route_after_gate)
    g.add_edge("extract_one", "detect_contradictions")
    g.add_conditional_edges("detect_contradictions", fan_out_hunts)
    g.add_edge("hunt_one", "plan_conflicts")
    g.add_conditional_edges("plan_conflicts", fan_out_conflicts)
    g.add_edge("resolve_one", "consolidate")
    g.add_edge("consolidate", "synthesize")
    g.add_edge("synthesize", END)
    return g


def build_synthesis_graph(checkpointer: Any | None = None):
    """Compile the synthesis graph, optionally with a checkpointer for resume."""
    return _build_graph_definition().compile(checkpointer=checkpointer)


def open_checkpointer(checkpoint_path: Any):
    """
    Open a :class:`SqliteSaver` against ``checkpoint_path``.

    ``check_same_thread=False`` is required because the Send fan-outs execute
    branches on worker threads that all share the saver's connection.
    """
    import sqlite3
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def run_graph_synthesis_state(
    question: str | None,
    *,
    config: SynthesisConfig | None = None,
    deps: SynthesisDeps | None = None,
    database: Any | None = None,
    session_id: str | None = None,
    progress: Callable[[str], None] | None = None,
    checkpoint_path: Any | None = None,
    thread_id: str | None = None,
    resume: bool = False,
) -> SynthesisState:
    """
    Run the synthesis graph and return the full traced :class:`SynthesisState`.

    This is the entry point for callers that render the decision trace (the
    Streamlit UI); :func:`run_graph_synthesis` wraps it for callers that only
    need the result artifact.

    Checkpointing: when ``checkpoint_path`` is given, every superstep is
    persisted under ``thread_id`` so an interrupted run (crash, quota
    exhaustion, Ctrl-C) can be continued with ``resume=True`` — completed
    stages, including finished parallel branches, are not re-executed.
    ``question`` may be ``None`` on resume; it is recovered from the
    checkpoint.
    """
    cfg = config or SynthesisConfig()
    checkpointer = open_checkpointer(checkpoint_path) if checkpoint_path else None
    graph = build_synthesis_graph(checkpointer=checkpointer)

    run_config: RunnableConfig = {
        # Bound the Send fan-outs so parallel LLM/API branches respect
        # provider rate limits; headroom on recursion for the retrieval cycle.
        "max_concurrency": max(1, cfg.max_concurrency),
        "recursion_limit": 50,
        "configurable": {
            "litsynth_config": cfg,
            "litsynth_deps": deps or SynthesisDeps(),
        },
    }
    if checkpointer is not None:
        run_config["configurable"]["thread_id"] = thread_id or uuid4().hex[:12]

    if resume:
        if checkpointer is None or thread_id is None:
            raise ValueError("resume requires checkpoint_path and thread_id")
        graph_input: Any = None  # continue from the last checkpoint
    else:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        graph_input = SynthesisState(question=question, session_id=session_id)

    try:
        if progress is not None:
            final_values: dict[str, Any] | None = None
            reported: int | None = None
            for values in graph.stream(graph_input, run_config, stream_mode="values"):
                final_values = values
                trace = values.get("trace") or []
                if reported is None:
                    # First event: on resume it is the checkpointed state —
                    # skip past the historical steps instead of replaying them.
                    reported = len(trace) if resume else 0
                # Only report steps that are new since the last update; a node
                # that changes state without adding steps must not repeat the
                # last line.
                for step in trace[reported:]:
                    try:
                        progress(f"{step.action}: {step.result_note}")
                    except Exception:  # noqa: BLE001
                        logger.debug("Progress callback raised; ignoring", exc_info=True)
                reported = len(trace)
            raw = final_values or {}
        else:
            raw = graph.invoke(graph_input, run_config)
    finally:
        if checkpointer is not None:
            try:
                checkpointer.conn.close()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to close checkpointer connection", exc_info=True)

    state = SynthesisState.model_validate(raw)
    if state.terminal_reason is None:
        state.terminal_reason = "synthesized" if state.review_text else "error"

    if database is not None:
        from synthesis.persistence import persist_synthesis_state

        persist_synthesis_state(database, session_id or state.session_id, state)
    return state


def run_graph_synthesis(
    question: str | None,
    *,
    config: SynthesisConfig | None = None,
    deps: SynthesisDeps | None = None,
    database: Any | None = None,
    session_id: str | None = None,
    progress: Callable[[str], None] | None = None,
    checkpoint_path: Any | None = None,
    thread_id: str | None = None,
    resume: bool = False,
) -> SynthesisResult:
    """
    Run the LangGraph synthesis path end-to-end and return the result artifact.

    Same contract the old controller entry point had (persistence into
    ``synthesis_runs``, optional coarse progress callback) so callers can swap
    orchestrators without changing anything else. See
    :func:`run_graph_synthesis_state` for checkpoint/resume semantics.
    """
    state = run_graph_synthesis_state(
        question,
        config=config,
        deps=deps,
        database=database,
        session_id=session_id,
        progress=progress,
        checkpoint_path=checkpoint_path,
        thread_id=thread_id,
        resume=resume,
    )
    return state.to_result()
