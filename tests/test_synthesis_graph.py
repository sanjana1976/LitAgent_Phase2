"""Tests for the LangGraph synthesis orchestration (:mod:`synthesis.graph`)."""

from __future__ import annotations

from datetime import date
from typing import Any

from synthesis.graph import (
    SynthesisConfig,
    SynthesisDeps,
    build_synthesis_graph,
    route_after_gate,
    route_after_search,
    run_graph_synthesis,
)
from synthesis.schemas import ClaimRecord, ContradictionPair, ResearchQuestion, ScoredPaper
from synthesis.state import SynthesisState
from tools.schemas import Paper

_MARKER = "unique marker phrase for corroboration"


def _paper(pid: str, title: str = "Paper") -> Paper:
    return Paper(
        paper_id=pid,
        title=title,
        authors=["Ada Lovelace"],
        abstract="A useful abstract.",
        publication_date=date(2024, 1, 1),
        api_source="arxiv",
    )


def _scored(pid: str, title: str = "Paper", **kw: Any) -> ScoredPaper:
    defaults: dict[str, Any] = {
        "paper_id": pid,
        "title": title,
        "authors": ["Ada Lovelace"],
        "abstract": "long context retrieval methods abstract",
        "text_tier": "abstract",
    }
    defaults.update(kw)
    return ScoredPaper(**defaults)


def _claim(
    paper_id: str,
    *,
    grounded: bool,
    quote: str = "a verbatim evidence quote",
    text: str = "a sufficiently long paper-specific claim",
) -> ClaimRecord:
    return ClaimRecord(
        paper_id=paper_id,
        claim=text,
        evidence_quote=quote,
        grounded=grounded,
        grounding_tier="abstract" if grounded else "none",
    )


def _stub_deps(**overrides: Any) -> SynthesisDeps:
    """Deps where every stage is a deterministic no-network stub."""

    def decompose(question: str, *, n: int) -> ResearchQuestion:
        return ResearchQuestion(question=question, sub_queries=["angle one", "angle two"])

    corpus = {
        "angle one": [_paper("arxiv:1", "Long Context Retrieval Survey")],
        "angle two": [_paper("arxiv:2", "Retrieval Benchmarks for LLMs")],
    }

    def retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        return list(corpus.get(rq.sub_queries[0], []))

    def fetch_parse(papers: list[Paper]) -> list[ScoredPaper]:
        return [
            _scored(p.paper_id, p.title, abstract="long context retrieval methods")
            for p in papers
        ]

    def relevance_filter(question: str, papers: list[ScoredPaper], **kw: Any):
        return list(papers), {p.paper_id: 9 for p in papers}, True

    def extract(papers: list[ScoredPaper], **kw: Any) -> list[ClaimRecord]:
        return [_claim(papers[0].paper_id, grounded=True)]

    def detect(claims: list[ClaimRecord], papers: list[ScoredPaper], **kw: Any):
        return []

    defaults = dict(
        decompose=decompose,
        retrieve=retrieve,
        fetch_parse=fetch_parse,
        relevance_filter=relevance_filter,
        extract_claims=extract,
        detect_contradictions=detect,
        build_prompt=lambda **kw: object(),
        generate=lambda prompt, **kw: "A grounded review.",
        validate_cites=lambda text, papers: ([], [], [], 1.0),
        # No-op rewrite so no test ever reaches the real (network) reformulator.
        reformulate=lambda state, original: "",
        # Clean critic: accepts every draft (tests opt into objections).
        critique=lambda question, review_text, *a, **kw: ([], True),
    )
    defaults.update(overrides)
    return SynthesisDeps(**defaults)


def _run(deps: SynthesisDeps, cfg: SynthesisConfig | None = None) -> SynthesisState:
    graph = build_synthesis_graph()
    raw = graph.invoke(
        SynthesisState(question="long context retrieval"),
        {
            "configurable": {
                # The stub corpus yields 2 papers; keep the coverage target at 2
                # so tests that are not about the reformulation cycle skip it.
                "litsynth_config": cfg
                or SynthesisConfig(min_relevance_score=0.0, min_relevant_papers=2),
                "litsynth_deps": deps,
            }
        },
    )
    return SynthesisState.model_validate(raw)


def test_graph_compiles() -> None:
    assert build_synthesis_graph() is not None


def test_linear_run_reaches_review_with_full_trace() -> None:
    state = _run(_stub_deps())

    assert state.review_text == "A grounded review."
    assert state.terminal_reason == "synthesized"
    assert {p.paper_id for p in state.papers} == {"arxiv:1", "arxiv:2"}
    assert set(state.active_paper_ids) == {"arxiv:1", "arxiv:2"}
    assert len(state.claims) == 2
    assert all(c.grounded for c in state.claims)

    # The synthesize node upserts ranked copies, so scores are real, not 0.0.
    assert all(p.relevance_score > 0.0 for p in state.to_result().papers)

    actions = [s.action for s in state.trace]
    assert actions == [
        "decompose",
        "search",
        "search",
        "filter_relevance",
        "extract_claims",
        "extract_claims",
        "detect_contradictions",
        "synthesize",
        "critique",
    ]
    assert all(s.result != "pending" for s in state.trace)
    # A clean critique accepts the draft without a revision.
    assert state.trace[-1].result == "ok"
    assert state.objections == []


def test_trace_steps_branch_causally_through_retrieval() -> None:
    """Parallel search branches share the decompose step as their causal parent."""
    state = _run(_stub_deps())
    decompose, search1, search2, gate = state.trace[:4]
    assert decompose.parent_step_id is None
    # Fan-out: both branches point at the dispatching decompose step.
    assert search1.parent_step_id == decompose.step_id
    assert search2.parent_step_id == decompose.step_id
    # The gate follows the fan-in; its parent is one of the search branches.
    assert gate.parent_step_id in {search1.step_id, search2.step_id}


def test_relevance_gate_shrinks_active_set_but_not_papers() -> None:
    def gate(question: str, papers: list[ScoredPaper], **kw: Any):
        kept = [p for p in papers if p.paper_id == "arxiv:1"]
        return kept, {"arxiv:2": 2}, True

    state = _run(_stub_deps(relevance_filter=gate))

    # Evidence is append-only; selection shrank.
    assert {p.paper_id for p in state.papers} == {"arxiv:1", "arxiv:2"}
    assert state.active_paper_ids == ["arxiv:1"]
    # Claims were only extracted for the active paper.
    assert {c.paper_id for c in state.claims} == {"arxiv:1"}
    # The result artifact reports the active set only.
    assert [p.paper_id for p in state.to_result().papers] == ["arxiv:1"]


def test_gate_noop_keeps_active_set_when_llm_unavailable() -> None:
    def gate(question: str, papers: list[ScoredPaper], **kw: Any):
        return list(papers), {}, False

    state = _run(_stub_deps(relevance_filter=gate))
    gate_step = next(s for s in state.trace if s.action == "filter_relevance")
    assert gate_step.result == "noop"
    assert set(state.active_paper_ids) == {"arxiv:1", "arxiv:2"}


def test_hunt_upgrades_claim_via_upsert_not_duplicate() -> None:
    ungrounded = _claim("arxiv:1", grounded=False, quote=_MARKER)

    def extract(papers: list[ScoredPaper], **kw: Any) -> list[ClaimRecord]:
        if papers[0].paper_id == "arxiv:1":
            return [ungrounded]
        return [_claim(papers[0].paper_id, grounded=True)]

    hunt_calls: list[str] = []

    def retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        query = rq.sub_queries[0]
        if query == "angle one":
            return [_paper("arxiv:1")]
        if query == "angle two":
            return [_paper("arxiv:2")]
        hunt_calls.append(query)
        return [_paper("arxiv:support", "Supporting Paper")]

    def fetch_parse(papers: list[Paper]) -> list[ScoredPaper]:
        out = []
        for p in papers:
            if p.paper_id == "arxiv:support":
                out.append(
                    _scored(
                        p.paper_id,
                        p.title,
                        full_text=f"long context retrieval text with {_MARKER}",
                        text_tier="full_text",
                    )
                )
            else:
                out.append(_scored(p.paper_id, p.title, abstract="long context retrieval"))
        return out

    state = _run(_stub_deps(extract_claims=extract, retrieve=retrieve, fetch_parse=fetch_parse))

    assert hunt_calls, "the hunt should have issued a targeted search"
    # The upsert reducer replaced the claim in place — same id, no duplicate.
    matching = [c for c in state.claims if c.claim_id == ungrounded.claim_id]
    assert len(matching) == 1
    assert matching[0].grounded is True
    assert matching[0].grounding_tier == "corroborated"
    assert matching[0].supporting_paper_id == "arxiv:support"
    # The supporting paper joined both the evidence pool and the active set.
    assert state.get_paper("arxiv:support") is not None
    assert "arxiv:support" in state.active_paper_ids
    gap = next(g for g in state.gaps if g.origin_claim_ref == ungrounded.claim_id)
    assert gap.status == "resolved"
    assert gap.resolved_by_paper_id == "arxiv:support"
    hunt_step = next(s for s in state.trace if s.action == "hunt_support")
    extract_step = next(
        s
        for s in state.trace
        if s.action == "extract_claims" and ungrounded.claim_id in s.effect.claim_refs
    )
    assert hunt_step.parent_step_id == extract_step.step_id


def test_conflict_resolution_adds_third_paper_and_claim() -> None:
    pair = ContradictionPair(
        paper_a="arxiv:1",
        paper_b="arxiv:2",
        claim_a="scaling improves accuracy on retrieval benchmark datasets",
        claim_b="scaling degrades accuracy on retrieval benchmark datasets",
    )

    def detect(claims: list[ClaimRecord], papers: list[ScoredPaper], **kw: Any):
        return [pair]

    def retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        query = rq.sub_queries[0]
        if query == "angle one":
            return [_paper("arxiv:1")]
        if query == "angle two":
            return [_paper("arxiv:2")]
        return [_paper("arxiv:3", "Third Paper")]

    def fetch_parse(papers: list[Paper]) -> list[ScoredPaper]:
        return [
            _scored(p.paper_id, p.title, abstract="long context retrieval scaling")
            for p in papers
        ]

    def extract(papers: list[ScoredPaper], **kw: Any) -> list[ClaimRecord]:
        return [_claim(papers[0].paper_id, grounded=True)]

    state = _run(
        _stub_deps(
            detect_contradictions=detect,
            retrieve=retrieve,
            fetch_parse=fetch_parse,
            extract_claims=extract,
        )
    )

    assert state.get_paper("arxiv:3") is not None
    assert "arxiv:3" in state.active_paper_ids
    resolve_step = next(s for s in state.trace if s.action == "resolve_conflict")
    assert resolve_step.result == "ok"
    assert resolve_step.effect.resolved_conflict == pair.contradiction_id
    resolution_id = resolve_step.effect.claim_ref
    assert any(c.claim_id == resolution_id and c.paper_id == "arxiv:3" for c in state.claims)


def _router_config(cfg: SynthesisConfig) -> dict:
    return {"configurable": {"litsynth_config": cfg, "litsynth_deps": _stub_deps()}}


def test_router_proceeds_when_coverage_is_met() -> None:
    state = SynthesisState(
        question="q", active_paper_ids=["a", "b", "c", "d"]
    )
    cfg = SynthesisConfig(min_relevant_papers=4)
    assert route_after_search(state, _router_config(cfg)) == "filter_relevance"


def test_router_reformulates_when_thin_and_budget_remains() -> None:
    state = SynthesisState(question="q", active_paper_ids=["a"])
    cfg = SynthesisConfig(min_relevant_papers=4, max_reformulations=2)
    assert route_after_search(state, _router_config(cfg)) == "reformulate"


def test_router_proceeds_when_reformulation_budget_exhausted() -> None:
    from synthesis.trace import DecisionStep, ReformulateParams

    state = SynthesisState(question="q", active_paper_ids=["a"])
    for i in range(2):
        state.trace.append(
            DecisionStep.start(
                action="reformulate",
                params=ReformulateParams(original_query=f"q{i}", new_query=f"q{i + 1}"),
                trigger="thin",
                rationale="retry",
            )
        )
    cfg = SynthesisConfig(min_relevant_papers=4, max_reformulations=2)
    assert route_after_search(state, _router_config(cfg)) == "filter_relevance"


def test_router_after_gate_reformulates_when_gate_guts_the_set() -> None:
    state = SynthesisState(question="q", active_paper_ids=["only-survivor"])
    cfg = SynthesisConfig(min_relevant_papers=4, max_reformulations=2)
    assert route_after_gate(state, _router_config(cfg)) == "reformulate"


def test_router_after_gate_fans_out_when_coverage_met_or_budget_gone() -> None:
    from langgraph.types import Send

    cfg = SynthesisConfig(min_relevant_papers=2, max_reformulations=2)
    ok_state = SynthesisState(
        question="q",
        papers=[_scored("a"), _scored("b")],
        active_paper_ids=["a", "b"],
    )
    result = route_after_gate(ok_state, _router_config(cfg))
    assert isinstance(result, list)
    assert all(isinstance(s, Send) and s.node == "extract_one" for s in result)
    assert [s.arg["paper"].paper_id for s in result] == ["a", "b"]

    from synthesis.trace import DecisionStep, ReformulateParams

    exhausted = SynthesisState(
        question="q", papers=[_scored("a")], active_paper_ids=["a"]
    )
    for i in range(2):
        exhausted.trace.append(
            DecisionStep.start(
                action="reformulate",
                params=ReformulateParams(original_query=f"q{i}", new_query=f"q{i + 1}"),
                trigger="thin",
                rationale="retry",
            )
        )
    result = route_after_gate(exhausted, _router_config(cfg))
    assert isinstance(result, list) and len(result) == 1

    # No active papers AND no budget left -> skip extraction entirely.
    empty = SynthesisState(question="q")
    for i in range(2):
        empty.trace.append(
            DecisionStep.start(
                action="reformulate",
                params=ReformulateParams(original_query=f"e{i}", new_query=f"e{i + 1}"),
                trigger="thin",
                rationale="retry",
            )
        )
    assert route_after_gate(empty, _router_config(cfg)) == "detect_contradictions"


def test_gate_gutting_active_set_triggers_new_retrieval() -> None:
    """Gate keeps 1 of 2 -> reformulate -> new search -> gate passes -> synthesized."""
    retrieve_calls: list[str] = []

    def retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        query = rq.sub_queries[0]
        retrieve_calls.append(query)
        if query == "angle one":
            return [_paper("arxiv:1"), _paper("arxiv:offtopic")]
        if query == "angle two":
            return []
        return [_paper("arxiv:good1"), _paper("arxiv:good2")]

    def gate(question: str, papers: list[ScoredPaper], **kw: Any):
        kept = [p for p in papers if "offtopic" not in p.paper_id]
        return kept, {}, True

    def reformulate(state: SynthesisState, original: str) -> str:
        return "better query"

    deps = _stub_deps(retrieve=retrieve, relevance_filter=gate)
    deps.reformulate = reformulate
    state = _run(deps, SynthesisConfig(min_relevant_papers=2, min_relevance_score=0.0))

    assert "better query" in retrieve_calls
    # After the second sweep the gate-passing set meets the target.
    assert set(state.active_paper_ids) >= {"arxiv:1", "arxiv:good1", "arxiv:good2"}
    assert "arxiv:offtopic" not in state.active_paper_ids
    assert state.terminal_reason == "synthesized"
    gate_steps = [s for s in state.trace if s.action == "filter_relevance"]
    assert len(gate_steps) == 2, "the gate re-runs after the retrieval cycle"


def test_reformulation_cycle_searches_only_the_new_query() -> None:
    """Thin first sweep -> reformulate -> only the new query searched -> synthesized."""
    retrieve_calls: list[str] = []

    def decompose(question: str, *, n: int) -> ResearchQuestion:
        return ResearchQuestion(question=question, sub_queries=["initial query"])

    def retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        query = rq.sub_queries[0]
        retrieve_calls.append(query)
        if query == "initial query":
            return [_paper("arxiv:1"), _paper("arxiv:2")]
        return [_paper(f"arxiv:{i}") for i in range(1, 6)]

    def reformulate(state: SynthesisState, original: str) -> str:
        return "broadened query"

    deps = _stub_deps(decompose=decompose, retrieve=retrieve)
    deps.reformulate = reformulate
    state = _run(deps, SynthesisConfig(min_relevant_papers=4, min_relevance_score=0.0))

    assert retrieve_calls == ["initial query", "broadened query"]
    assert state.terminal_reason == "synthesized"
    assert len(state.active_paper_ids) == 5
    prefix = [s.action for s in state.trace[:4]]
    assert prefix == ["decompose", "search", "reformulate", "search"]
    reform_step = state.trace[2]
    assert reform_step.params.new_query == "broadened query"
    assert "broadened query" in state.sub_queries


def test_reformulation_cap_terminal_reason_when_still_thin() -> None:
    def decompose(question: str, *, n: int) -> ResearchQuestion:
        return ResearchQuestion(question=question, sub_queries=["only query"])

    def retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        return [_paper("arxiv:1")]

    attempts: list[str] = []

    def reformulate(state: SynthesisState, original: str) -> str:
        attempts.append(original)
        return f"retry {len(attempts)}"

    deps = _stub_deps(decompose=decompose, retrieve=retrieve)
    deps.reformulate = reformulate
    state = _run(
        deps,
        SynthesisConfig(min_relevant_papers=4, max_reformulations=2, min_relevance_score=0.0),
    )

    assert len(attempts) == 2
    assert state.terminal_reason == "reformulation_cap"
    # The run still produced a review from what little evidence it had.
    assert state.review_text == "A grounded review."


def test_gate_failure_is_logged_and_active_set_unchanged() -> None:
    def gate(question: str, papers: list[ScoredPaper], **kw: Any):
        raise RuntimeError("gate exploded")

    state = _run(_stub_deps(relevance_filter=gate))
    gate_step = next(s for s in state.trace if s.action == "filter_relevance")
    assert gate_step.result == "failed"
    assert "gate exploded" in gate_step.result_note
    assert set(state.active_paper_ids) == {"arxiv:1", "arxiv:2"}
    assert state.terminal_reason == "synthesized"


def test_gate_threshold_comes_from_config() -> None:
    seen: dict[str, Any] = {}

    def gate(question: str, papers: list[ScoredPaper], **kw: Any):
        seen.update(kw)
        return list(papers), {}, True

    state = _run(
        _stub_deps(relevance_filter=gate),
        SynthesisConfig(relevance_keep_threshold=8, min_relevance_score=0.0),
    )
    assert seen.get("keep_threshold") == 8
    gate_step = next(s for s in state.trace if s.action == "filter_relevance")
    assert gate_step.params.keep_threshold == 8


def test_conflict_resolution_skips_the_two_conflicting_papers() -> None:
    pair = ContradictionPair(
        paper_a="arxiv:1",
        paper_b="arxiv:2",
        claim_a="scaling improves accuracy on retrieval benchmark datasets",
        claim_b="scaling degrades accuracy on retrieval benchmark datasets",
    )
    extracted_from: list[str] = []

    def detect(claims: list[ClaimRecord], papers: list[ScoredPaper], **kw: Any):
        return [pair]

    def retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        query = rq.sub_queries[0]
        if query == "angle one":
            return [_paper("arxiv:1")]
        if query == "angle two":
            return [_paper("arxiv:2")]
        # The hunt returns the two conflicting papers plus nothing else usable.
        return [_paper("arxiv:1"), _paper("arxiv:2")]

    def fetch_parse(papers: list[Paper]) -> list[ScoredPaper]:
        return [
            _scored(p.paper_id, p.title, abstract="long context retrieval scaling")
            for p in papers
        ]

    def extract(papers: list[ScoredPaper], **kw: Any) -> list[ClaimRecord]:
        extracted_from.append(papers[0].paper_id)
        return [_claim(papers[0].paper_id, grounded=True)]

    state = _run(
        _stub_deps(
            detect_contradictions=detect,
            retrieve=retrieve,
            fetch_parse=fetch_parse,
            extract_claims=extract,
        )
    )

    resolve_step = next(s for s in state.trace if s.action == "resolve_conflict")
    assert resolve_step.result == "insufficient"
    # During the conflict hunt, neither conflicting paper was re-extracted:
    # only the two initial per-paper extractions happened.
    assert extracted_from.count("arxiv:1") == 1
    assert extracted_from.count("arxiv:2") == 1


def test_failed_search_query_is_logged_and_run_continues() -> None:
    def retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        query = rq.sub_queries[0]
        if query == "angle one":
            raise RuntimeError("provider down")
        return [_paper("arxiv:2")]

    state = _run(_stub_deps(retrieve=retrieve))

    search_steps = [s for s in state.trace if s.action == "search"]
    # Parallel branches merge in nondeterministic order; compare as a multiset.
    assert sorted(s.result for s in search_steps) == ["failed", "ok"]
    failed_step = next(s for s in search_steps if s.result == "failed")
    assert "provider down" in failed_step.result_note
    # The no-op reformulator burns the budget; the run still produces a review
    # from the single surviving paper and reports the honest terminal reason.
    reform_steps = [s for s in state.trace if s.action == "reformulate"]
    assert [s.result for s in reform_steps] == ["noop", "noop"]
    assert state.review_text == "A grounded review."
    assert state.terminal_reason == "reformulation_cap"


def test_synthesize_failure_sets_error_terminal_reason() -> None:
    def boom(prompt: Any, **kw: Any) -> str:
        raise RuntimeError("generation exploded")

    state = _run(_stub_deps(generate=boom))

    assert state.terminal_reason == "error"
    assert state.review_text is None
    synth_step = next(s for s in state.trace if s.action == "synthesize")
    assert synth_step.result == "failed"
    assert "generation exploded" in synth_step.result_note


def test_claim_extraction_branches_run_concurrently() -> None:
    """The Send fan-out overlaps per-paper LLM calls instead of serializing them."""
    import threading
    import time as _time

    tracker = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def slow_extract(papers: list[ScoredPaper], **kw: Any) -> list[ClaimRecord]:
        with lock:
            tracker["now"] += 1
            tracker["peak"] = max(tracker["peak"], tracker["now"])
        _time.sleep(0.1)
        with lock:
            tracker["now"] -= 1
        return [_claim(papers[0].paper_id, grounded=True)]

    started = _time.perf_counter()
    state = _run(_stub_deps(extract_claims=slow_extract))
    elapsed = _time.perf_counter() - started

    assert len(state.claims) == 2
    assert tracker["peak"] >= 2, "both extraction branches should overlap in time"
    # Two 0.1s extractions in parallel should finish well under the serial 0.2s+.
    assert elapsed < 5.0


def test_papers_found_by_parallel_branches_deduplicate() -> None:
    """Two search branches returning the same paper merge to one pool entry."""

    def retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        # Both sub-queries surface the same paper plus one unique hit.
        return [_paper("arxiv:shared"), _paper(f"arxiv:{rq.sub_queries[0][-3:]}")]

    state = _run(_stub_deps(retrieve=retrieve))

    shared = [p for p in state.papers if p.paper_id == "arxiv:shared"]
    assert len(shared) == 1
    assert len(state.papers) == 3


def test_critic_objection_triggers_one_revision() -> None:
    """An objecting critic sends the draft back to the writer exactly once."""
    critiques: list[str] = []
    drafts: list[str] = []

    def critic(question: str, review_text: str, *a: Any, **kw: Any):
        critiques.append(review_text)
        if len(critiques) == 1:
            return ["\"weak sentence\" — not supported by any claim"], True
        return [], True

    def generate(prompt: Any, **kw: Any) -> str:
        drafts.append("draft")
        if len(drafts) == 1:
            return "First draft with a weak sentence."
        # The revision prompt must carry the previous draft and the objection.
        assert "First draft with a weak sentence." in prompt.user
        assert "weak sentence" in prompt.user
        return "Revised draft, fully grounded."

    state = _run(_stub_deps(critique=critic, generate=generate))

    assert state.review_text == "Revised draft, fully grounded."
    assert len(critiques) == 2  # draft critiqued, revision re-critiqued
    assert state.objections == []  # final pass came back clean
    actions = [s.action for s in state.trace]
    assert actions[-4:] == ["synthesize", "critique", "synthesize", "critique"]
    revise_step = state.trace[-2]
    assert revise_step.params.revision == 1
    critique_steps = [s for s in state.trace if s.action == "critique"]
    assert [s.result for s in critique_steps] == ["insufficient", "ok"]


def test_revision_budget_caps_persistent_critic() -> None:
    """A critic that never approves stops at max_revisions, keeping the last draft."""
    calls = {"critic": 0, "writer": 0}

    def stubborn_critic(question: str, review_text: str, *a: Any, **kw: Any):
        calls["critic"] += 1
        return ["\"grounded review\" — reviewer is never satisfied"], True

    def generate(prompt: Any, **kw: Any) -> str:
        calls["writer"] += 1
        return "A grounded review."

    state = _run(
        _stub_deps(critique=stubborn_critic, generate=generate),
        SynthesisConfig(min_relevance_score=0.0, max_revisions=2),
    )

    # Draft + 2 revisions; critic ran after each; loop then ended despite objections.
    assert calls["writer"] == 3
    assert calls["critic"] == 3
    assert state.objections  # last critique still objected — honestly recorded
    assert state.review_text == "A grounded review."
    revisions = [
        s.params.revision
        for s in state.trace
        if s.action == "synthesize" and s.params.revision > 0
    ]
    assert revisions == [1, 2]


def test_critic_failure_accepts_draft() -> None:
    def broken_critic(question: str, review_text: str, *a: Any, **kw: Any):
        raise RuntimeError("critic exploded")

    state = _run(_stub_deps(critique=broken_critic))

    assert state.review_text == "A grounded review."
    critique_step = next(s for s in state.trace if s.action == "critique")
    assert critique_step.result == "failed"
    assert state.objections == []


def test_failed_revision_keeps_previous_draft() -> None:
    """If the rewrite LLM call fails, the original draft survives."""
    seen = {"n": 0}

    def critic(question: str, review_text: str, *a: Any, **kw: Any):
        seen["n"] += 1
        if seen["n"] == 1:
            return ["\"grounded review\" — objection"], True
        return [], True

    def generate(prompt: Any, **kw: Any) -> str:
        if "Your previous draft" in str(getattr(prompt, "user", "")):
            raise RuntimeError("rewrite exploded")
        return "A grounded review."

    state = _run(_stub_deps(critique=critic, generate=generate))

    assert state.review_text == "A grounded review."
    revise_step = next(
        s for s in state.trace if s.action == "synthesize" and s.params.revision == 1
    )
    assert revise_step.result == "failed"


def test_run_graph_synthesis_persists_and_reports_progress(tmp_path: Any) -> None:
    from db.database import Database
    from db.init_db import initialize_schema
    from db.queries import list_recent_synthesis_runs

    db = Database(tmp_path / "graph.sqlite3")
    initialize_schema(db)
    seen: list[str] = []

    result = run_graph_synthesis(
        "long context retrieval",
        config=SynthesisConfig(min_relevance_score=0.0),
        deps=_stub_deps(),
        database=db,
        session_id="graph-session",
        progress=seen.append,
    )

    assert result.review_text == "A grounded review."
    assert [p.paper_id for p in result.papers] == ["arxiv:1", "arxiv:2"] or [
        p.paper_id for p in result.papers
    ] == ["arxiv:2", "arxiv:1"]
    assert seen, "progress callback should have fired"
    rows = list_recent_synthesis_runs(db, limit=5)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "graph-session"


def test_crashed_run_resumes_from_checkpoint_without_rerunning_stages(
    tmp_path: Any,
) -> None:
    """A generate-stage crash resumes at synthesize; retrieval/claims are not redone."""
    from synthesis.graph import run_graph_synthesis_state

    retrieve_calls: list[str] = []
    extract_calls: list[str] = []
    generate_attempts = {"n": 0}

    def counting_retrieve(rq: ResearchQuestion, **kw: Any) -> list[Paper]:
        retrieve_calls.append(rq.sub_queries[0])
        corpus = {
            "angle one": [_paper("arxiv:1", "Long Context Retrieval Survey")],
            "angle two": [_paper("arxiv:2", "Retrieval Benchmarks for LLMs")],
        }
        return list(corpus.get(rq.sub_queries[0], []))

    def counting_extract(papers: list[ScoredPaper], **kw: Any) -> list[ClaimRecord]:
        extract_calls.append(papers[0].paper_id)
        return [_claim(papers[0].paper_id, grounded=True)]

    def flaky_generate(prompt: Any, **kw: Any) -> str:
        generate_attempts["n"] += 1
        if generate_attempts["n"] == 1:
            raise KeyboardInterrupt("simulated mid-run crash")
        return "A resumed review."

    deps = _stub_deps(
        retrieve=counting_retrieve,
        extract_claims=counting_extract,
        generate=flaky_generate,
    )
    checkpoint = tmp_path / "checkpoints.sqlite3"
    cfg = SynthesisConfig(min_relevance_score=0.0, min_relevant_papers=2)

    try:
        run_graph_synthesis_state(
            "long context retrieval",
            config=cfg,
            deps=deps,
            checkpoint_path=checkpoint,
            thread_id="crash-run",
        )
        raise AssertionError("expected the simulated crash to propagate")
    except KeyboardInterrupt:
        pass

    searches_before = len(retrieve_calls)
    extracts_before = len(extract_calls)
    assert searches_before == 2 and extracts_before == 2

    state = run_graph_synthesis_state(
        None,
        config=cfg,
        deps=deps,
        checkpoint_path=checkpoint,
        thread_id="crash-run",
        resume=True,
    )

    # Retrieval and claim extraction were NOT re-executed on resume.
    assert len(retrieve_calls) == searches_before
    assert len(extract_calls) == extracts_before
    assert generate_attempts["n"] == 2
    assert state.review_text == "A resumed review."
    assert state.terminal_reason == "synthesized"
    assert state.question == "long context retrieval"
    # The full trace survived the crash/resume boundary.
    assert [s.action for s in state.trace[:1]] == ["decompose"]
    assert any(s.action == "synthesize" for s in state.trace)


def test_resume_requires_checkpoint_and_thread_id() -> None:
    from synthesis.graph import run_graph_synthesis_state
    import pytest

    with pytest.raises(ValueError, match="resume requires"):
        run_graph_synthesis_state(None, deps=_stub_deps(), resume=True)


def test_persistence_roundtrips_active_paper_ids(tmp_path: Any) -> None:
    from db.database import Database
    from db.init_db import initialize_schema
    from db.queries import list_recent_synthesis_runs, get_synthesis_run_result_json
    from synthesis.persistence import load_synthesis_state_from_json

    def gate(question: str, papers: list[ScoredPaper], **kw: Any):
        return [p for p in papers if p.paper_id == "arxiv:1"], {}, True

    db = Database(tmp_path / "graph.sqlite3")
    initialize_schema(db)
    run_graph_synthesis(
        "long context retrieval",
        config=SynthesisConfig(min_relevance_score=0.0),
        deps=_stub_deps(relevance_filter=gate),
        database=db,
    )

    row_id = list_recent_synthesis_runs(db, limit=1)[0]["id"]
    raw = get_synthesis_run_result_json(db, row_id)
    assert raw is not None
    restored = load_synthesis_state_from_json(raw)
    assert restored is not None
    assert restored.active_paper_ids == ["arxiv:1"]
    assert [p.paper_id for p in restored.active_papers()] == ["arxiv:1"]
