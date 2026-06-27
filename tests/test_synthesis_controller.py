"""Tests for the first agentic synthesis controller layer."""

from __future__ import annotations

from datetime import date

from synthesis.controller import ControllerConfig, ControllerHooks, SynthesisController, next_action
from synthesis.schemas import CitationCheck, ClaimRecord, ContradictionPair, ResearchQuestion, ScoredPaper
from synthesis.state import Gap, SynthesisState
from synthesis.trace import DecisionStep, DetectContradictionsParams, ReformulateParams, SearchParams
from tools.schemas import Paper


def _cfg(**kwargs: object) -> ControllerConfig:
    kwargs.setdefault("min_relevance_score", 0.0)
    return ControllerConfig(**kwargs)


def _paper(pid: str, title: str = "Paper") -> Paper:
    return Paper(
        paper_id=pid,
        title=title,
        authors=["Ada Lovelace"],
        abstract="A useful abstract.",
        publication_date=date(2024, 1, 1),
        api_source="arxiv",
    )


def _scored(pid: str, title: str = "Paper") -> ScoredPaper:
    return ScoredPaper(
        paper_id=pid,
        title=title,
        authors=["Ada Lovelace"],
        abstract="A useful abstract.",
        text_tier="abstract",
    )


def test_next_action_starts_with_decompose() -> None:
    state = SynthesisState(question="long context retrieval")
    step = next_action(state)
    assert step.action == "decompose"
    assert step.params.kind == "decompose"
    assert step.result == "pending"


def test_next_action_searches_after_decompose_or_reformulation() -> None:
    state = SynthesisState(question="q", sub_queries=["first query"])
    step = next_action(state)
    assert step.action == "search"
    assert isinstance(step.params, SearchParams)
    assert step.params.query == "first query"

    state.log(
        DecisionStep.start(
            action="reformulate",
            params=ReformulateParams(original_query="first query", new_query="second query"),
            trigger="too few papers",
            rationale="retry",
        )
    )
    state.sub_queries.append("second query")
    step = next_action(state)
    assert step.action == "search"
    assert isinstance(step.params, SearchParams)
    assert step.params.query == "second query"
    assert step.parent_step_id == state.trace[-1].step_id


def test_next_action_reformulates_when_retrieval_is_thin() -> None:
    state = SynthesisState(
        question="q",
        sub_queries=["first query"],
        papers=[_scored("arxiv:1"), _scored("arxiv:2")],
    )
    step = next_action(state, _cfg(min_relevant_papers=4))
    assert step.action == "reformulate"
    assert isinstance(step.params, ReformulateParams)
    assert step.params.original_query == "first query"
    assert "below the target" in step.trigger


def test_next_action_stops_after_capped_retry_search_is_still_thin() -> None:
    state = SynthesisState(
        question="q",
        sub_queries=["q1", "q2"],
        papers=[_scored("arxiv:1")],
    )
    state.log(
        DecisionStep.start(
            action="reformulate",
            params=ReformulateParams(original_query="q1", new_query="q2"),
            trigger="thin retrieval",
            rationale="retry",
        )
    )
    state.log(
        DecisionStep.start(
            action="search",
            params=SearchParams(query="q2", sources=["arxiv"]),
            trigger="retry after reformulation",
            rationale="perform capped retry",
        )
    )
    step = next_action(state, _cfg(min_relevant_papers=4, max_reformulations=1))
    assert step.action == "synthesize"
    assert "cap has been reached" in step.rationale


def test_controller_reformulates_and_retries_until_enough_papers() -> None:
    retrieve_calls: list[str] = []

    def decompose(question: str, *, n: int) -> ResearchQuestion:
        return ResearchQuestion(question=question, sub_queries=["initial query"])

    def retrieve(rq: ResearchQuestion, **kwargs: object) -> list[Paper]:
        query = rq.sub_queries[0]
        retrieve_calls.append(query)
        if len(retrieve_calls) == 1:
            return [_paper("arxiv:1"), _paper("arxiv:2")]
        return [
            _paper("arxiv:1"),
            _paper("arxiv:2"),
            _paper("arxiv:3"),
            _paper("arxiv:4"),
            _paper("arxiv:5"),
        ]

    def fetch_parse(papers: list[Paper]) -> list[ScoredPaper]:
        return [_scored(p.paper_id, p.title) for p in papers]

    controller = SynthesisController(
        config=_cfg(min_relevant_papers=4, max_reformulations=2),
        hooks=ControllerHooks(decompose=decompose, retrieve=retrieve, fetch_parse=fetch_parse),
    )

    state = controller.run_retrieval_loop("long context retrieval")

    assert state.terminal_reason == "synthesized"
    assert retrieve_calls[0] == "initial query"
    assert len(retrieve_calls) == 2
    assert len(state.papers) == 5
    assert [s.action for s in state.trace] == [
        "decompose",
        "search",
        "reformulate",
        "search",
        "synthesize",
    ]
    assert state.trace[1].result == "insufficient"
    assert state.trace[2].result == "ok"
    assert state.trace[3].result == "ok"
    assert state.trace[3].effect.added_paper_ids == ["arxiv:3", "arxiv:4", "arxiv:5"]
    assert state.trace[2].parent_step_id == state.trace[1].step_id
    assert state.trace[3].parent_step_id == state.trace[2].step_id


def _claim(
    paper_id: str,
    *,
    grounded: bool,
    tier: str = "none",
    quote: str = "a verbatim evidence quote",
    text: str = "a sufficiently long paper-specific claim",
) -> ClaimRecord:
    return ClaimRecord(
        paper_id=paper_id,
        claim=text,
        evidence_quote=quote,
        grounded=grounded,
        grounding_tier=tier,
    )


def _make_extract(mapping: dict[str, object], calls: list[str] | None = None):
    def _extract(papers: list[ScoredPaper], *, max_claims_per_paper: int = 4):
        pid = papers[0].paper_id
        if calls is not None:
            calls.append(pid)
        outcome = mapping.get(pid)
        if isinstance(outcome, Exception):
            raise outcome
        return list(outcome or [])

    return _extract


# --------------------------------------------------------------------------- #
# run_claims_loop
# --------------------------------------------------------------------------- #


def test_claims_loop_appends_claims_for_every_paper() -> None:
    state = SynthesisState(question="q", papers=[_scored("arxiv:1"), _scored("arxiv:2")])
    mapping = {
        "arxiv:1": [_claim("arxiv:1", grounded=True)],
        "arxiv:2": [_claim("arxiv:2", grounded=False)],
    }
    controller = SynthesisController(hooks=ControllerHooks(extract_claims=_make_extract(mapping)))
    controller.run_claims_loop(state)
    assert len(state.claims) == 2
    assert {c.paper_id for c in state.claims} == {"arxiv:1", "arxiv:2"}


def test_claims_loop_stamps_grounding_tier_from_paper_text_tier() -> None:
    paper = ScoredPaper(paper_id="arxiv:1", title="A", full_text="body", text_tier="full_text")
    # extractor returns grounded claim but with an unstamped tier
    mapping = {"arxiv:1": [_claim("arxiv:1", grounded=True, tier="none")]}
    state = SynthesisState(question="q", papers=[paper])
    controller = SynthesisController(hooks=ControllerHooks(extract_claims=_make_extract(mapping)))
    controller.run_claims_loop(state)
    claim = state.claims[0]
    assert claim.grounding_tier == "full_text"
    assert claim.supporting_paper_id == "arxiv:1"


def test_claims_loop_failed_extraction_is_logged_not_raised() -> None:
    state = SynthesisState(question="q", papers=[_scored("arxiv:1"), _scored("arxiv:2")])
    mapping = {
        "arxiv:1": RuntimeError("boom"),
        "arxiv:2": [_claim("arxiv:2", grounded=True)],
    }
    controller = SynthesisController(hooks=ControllerHooks(extract_claims=_make_extract(mapping)))
    controller.run_claims_loop(state)  # must not raise
    assert state.trace[0].result == "failed"
    assert "boom" in state.trace[0].result_note
    assert state.trace[1].result == "ok"
    assert [c.paper_id for c in state.claims] == ["arxiv:2"]


def test_claims_loop_populates_step_effect_claim_refs() -> None:
    c1 = _claim("arxiv:1", grounded=True)
    c2 = _claim("arxiv:1", grounded=False)
    state = SynthesisState(question="q", papers=[_scored("arxiv:1")])
    controller = SynthesisController(hooks=ControllerHooks(extract_claims=_make_extract({"arxiv:1": [c1, c2]})))
    controller.run_claims_loop(state)
    assert state.trace[0].effect.claim_refs == [c1.claim_id, c2.claim_id]


def test_claims_loop_preserves_trace_order() -> None:
    papers = [_scored("arxiv:1"), _scored("arxiv:2"), _scored("arxiv:3")]
    mapping = {p.paper_id: [_claim(p.paper_id, grounded=True)] for p in papers}
    calls: list[str] = []
    state = SynthesisState(question="q", papers=papers)
    controller = SynthesisController(hooks=ControllerHooks(extract_claims=_make_extract(mapping, calls)))
    controller.run_claims_loop(state)
    assert calls == ["arxiv:1", "arxiv:2", "arxiv:3"]
    assert [s.action for s in state.trace] == ["extract_claims"] * 3
    assert [s.params.paper_ids[0] for s in state.trace] == ["arxiv:1", "arxiv:2", "arxiv:3"]


def test_claims_loop_ungrounded_claim_is_stamped_none() -> None:
    paper = ScoredPaper(paper_id="arxiv:1", title="A", full_text="body", text_tier="full_text")
    # extractor wrongly stamps a tier on an ungrounded claim; loop must reset it
    mapping = {"arxiv:1": [_claim("arxiv:1", grounded=False, tier="full_text")]}
    state = SynthesisState(question="q", papers=[paper])
    controller = SynthesisController(hooks=ControllerHooks(extract_claims=_make_extract(mapping)))
    controller.run_claims_loop(state)
    claim = state.claims[0]
    assert claim.grounding_tier == "none"
    assert claim.supporting_paper_id is None


# --------------------------------------------------------------------------- #
# run_gap_detection_loop
# --------------------------------------------------------------------------- #

_MARKER = "special marker phrase grounding the claim"


def _make_retrieve(papers: list[Paper]):
    def _retrieve(rq: ResearchQuestion, **kwargs: object) -> list[Paper]:
        return papers

    return _retrieve


def _make_fetch(scored: list[ScoredPaper]):
    def _fetch(raw: list[Paper]) -> list[ScoredPaper]:
        return scored

    return _fetch


def _state_with_ungrounded_claim(
    *,
    quote: str = _MARKER,
    claim_text: str = "transformer retrieval accuracy benchmark datasets improves",
) -> tuple[SynthesisState, ClaimRecord, SynthesisController]:
    paper = ScoredPaper(
        paper_id="arxiv:1",
        title="A",
        abstract="an abstract without the supporting sentence",
        text_tier="abstract",
    )
    claim = _claim("arxiv:1", grounded=False, quote=quote, text=claim_text)
    state = SynthesisState(question="q", papers=[paper])
    controller = SynthesisController(
        hooks=ControllerHooks(extract_claims=_make_extract({"arxiv:1": [claim]}))
    )
    controller.run_claims_loop(state)
    return state, claim, controller


def _with_hunt_hooks(controller: SynthesisController, candidates: list[ScoredPaper]) -> None:
    controller.hooks.retrieve = _make_retrieve([Paper(paper_id="arxiv:2", title="B", api_source="arxiv")])
    controller.hooks.fetch_parse = _make_fetch(candidates)


def _supporting_paper() -> ScoredPaper:
    return ScoredPaper(
        paper_id="arxiv:2",
        title="Supporting",
        full_text=f"intro ... {_MARKER} ... conclusion",
        text_tier="full_text",
    )


def test_gap_loop_creates_gap_for_ungrounded_claim() -> None:
    state, claim, controller = _state_with_ungrounded_claim()
    _with_hunt_hooks(controller, candidates=[])  # nothing found
    controller.run_gap_detection_loop(state)
    assert len(state.gaps) == 1
    gap = state.gaps[0]
    assert gap.kind == "ungrounded_claim"
    assert gap.origin_claim_ref == claim.claim_id


def test_gap_loop_successful_hunt_upgrades_to_corroborated() -> None:
    state, claim, controller = _state_with_ungrounded_claim()
    _with_hunt_hooks(controller, candidates=[_supporting_paper()])
    controller.run_gap_detection_loop(state)
    assert claim.grounding_tier == "corroborated"
    assert claim.supporting_paper_id == "arxiv:2"
    assert claim.grounded is True
    gap = state.gaps[0]
    assert gap.status == "resolved"
    assert gap.resolved_by_paper_id == "arxiv:2"


def test_gap_loop_failed_hunt_flags_unverified() -> None:
    state, _claim_obj, controller = _state_with_ungrounded_claim()
    non_matching = ScoredPaper(paper_id="arxiv:2", title="B", full_text="unrelated content", text_tier="full_text")
    _with_hunt_hooks(controller, candidates=[non_matching])
    controller.run_gap_detection_loop(state)
    gap = state.gaps[0]
    assert gap.status == "flagged_unverified"
    hunt = [s for s in state.trace if s.action == "hunt_support"][0]
    assert hunt.result == "insufficient"


def test_gap_loop_parent_links_to_origin_extract_step() -> None:
    state, _claim_obj, controller = _state_with_ungrounded_claim()
    extract_step = [s for s in state.trace if s.action == "extract_claims"][0]
    _with_hunt_hooks(controller, candidates=[_supporting_paper()])
    controller.run_gap_detection_loop(state)
    hunt = [s for s in state.trace if s.action == "hunt_support"][0]
    assert hunt.parent_step_id == extract_step.step_id


def test_gap_loop_records_tier_before_and_after() -> None:
    state, claim, controller = _state_with_ungrounded_claim()
    _with_hunt_hooks(controller, candidates=[_supporting_paper()])
    controller.run_gap_detection_loop(state)
    hunt = [s for s in state.trace if s.action == "hunt_support"][0]
    assert hunt.effect.tier_before == "none"
    assert hunt.effect.tier_after == "corroborated"
    assert hunt.effect.claim_ref == claim.claim_id


def test_gap_loop_skips_grounded_claims() -> None:
    paper = ScoredPaper(paper_id="arxiv:1", title="A", full_text="body", text_tier="full_text")
    state = SynthesisState(question="q", papers=[paper])
    controller = SynthesisController(
        hooks=ControllerHooks(extract_claims=_make_extract({"arxiv:1": [_claim("arxiv:1", grounded=True)]}))
    )
    controller.run_claims_loop(state)
    _with_hunt_hooks(controller, candidates=[_supporting_paper()])
    controller.run_gap_detection_loop(state)
    assert state.gaps == []
    assert not any(s.action == "hunt_support" for s in state.trace)


def test_gap_loop_ignores_coverage_gaps() -> None:
    paper = ScoredPaper(paper_id="arxiv:1", title="A", full_text="body", text_tier="full_text")
    state = SynthesisState(question="q", papers=[paper])
    coverage = Gap(kind="coverage", description="missing theme", status="open")
    state.gaps.append(coverage)
    controller = SynthesisController(
        hooks=ControllerHooks(extract_claims=_make_extract({"arxiv:1": [_claim("arxiv:1", grounded=True)]}))
    )
    controller.run_claims_loop(state)
    _with_hunt_hooks(controller, candidates=[_supporting_paper()])
    controller.run_gap_detection_loop(state)
    assert state.gaps == [coverage]
    assert coverage.status == "open"
    assert not any(s.action == "hunt_support" for s in state.trace)


def test_gap_loop_adds_supporting_paper_and_derives_search_terms() -> None:
    state, _claim_obj, controller = _state_with_ungrounded_claim()
    _with_hunt_hooks(controller, candidates=[_supporting_paper()])
    controller.run_gap_detection_loop(state)
    assert "arxiv:2" in [p.paper_id for p in state.papers]
    hunt = [s for s in state.trace if s.action == "hunt_support"][0]
    assert hunt.params.search_terms  # non-empty, derived from claim text
    assert "transformer" in hunt.params.search_terms


# --------------------------------------------------------------------------- #
# run_detect_contradictions_loop
# --------------------------------------------------------------------------- #


def _state_with_two_claims() -> tuple[SynthesisState, SynthesisController]:
    state = SynthesisState(question="q", papers=[_scored("arxiv:1"), _scored("arxiv:2")])
    controller = SynthesisController(
        hooks=ControllerHooks(
            extract_claims=_make_extract(
                {
                    "arxiv:1": [_claim("arxiv:1", grounded=True)],
                    "arxiv:2": [_claim("arxiv:2", grounded=True)],
                }
            )
        )
    )
    controller.run_claims_loop(state)
    return state, controller


def test_detect_loop_appends_contradictions_with_stable_ids() -> None:
    state, controller = _state_with_two_claims()
    pair = _conflict_pair()
    controller.hooks.detect_contradictions = lambda c, p, **kw: [pair]
    controller.run_detect_contradictions_loop(state)
    assert state.contradictions == [pair]
    assert state.contradictions[0].contradiction_id == pair.contradiction_id


def test_detect_loop_noop_on_empty_result() -> None:
    state, controller = _state_with_two_claims()
    controller.hooks.detect_contradictions = lambda c, p, **kw: []
    controller.run_detect_contradictions_loop(state)
    step = [s for s in state.trace if s.action == "detect_contradictions"][0]
    assert step.result == "noop"
    assert state.contradictions == []


def test_detect_loop_failure_is_logged_not_raised() -> None:
    state, controller = _state_with_two_claims()

    def _boom(claims, papers, **kw):
        raise RuntimeError("detector exploded")

    controller.hooks.detect_contradictions = _boom
    controller.run_detect_contradictions_loop(state)  # must not raise
    step = [s for s in state.trace if s.action == "detect_contradictions"][0]
    assert step.result == "failed"
    assert "detector exploded" in step.result_note
    assert state.contradictions == []


def test_detect_loop_parent_links_to_last_extract_step() -> None:
    state, controller = _state_with_two_claims()
    last_extract = [s for s in state.trace if s.action == "extract_claims"][-1]
    controller.hooks.detect_contradictions = lambda c, p, **kw: [_conflict_pair()]
    controller.run_detect_contradictions_loop(state)
    step = [s for s in state.trace if s.action == "detect_contradictions"][0]
    assert step.parent_step_id == last_extract.step_id
    # params carry the claim ids that were passed to the detector
    assert set(step.params.claim_ids) == {c.claim_id for c in state.claims}


def test_detect_loop_effect_lists_contradiction_ids() -> None:
    state, controller = _state_with_two_claims()
    p1 = _conflict_pair()
    p2 = _conflict_pair()
    controller.hooks.detect_contradictions = lambda c, p, **kw: [p1, p2]
    controller.run_detect_contradictions_loop(state)
    step = [s for s in state.trace if s.action == "detect_contradictions"][0]
    assert step.effect.contradiction_ids == [p1.contradiction_id, p2.contradiction_id]


def test_run_orders_detection_after_claims_and_before_conflict() -> None:
    def decompose(question: str, *, n: int) -> ResearchQuestion:
        return ResearchQuestion(question=question, sub_queries=["initial query"])

    def retrieve(rq: ResearchQuestion, **kwargs: object) -> list[Paper]:
        return [_paper("arxiv:1"), _paper("arxiv:2")]

    def fetch_parse(papers: list[Paper]) -> list[ScoredPaper]:
        return [_scored(p.paper_id) for p in papers]

    pair = _conflict_pair()
    controller = SynthesisController(
        config=_cfg(min_relevant_papers=2),
        hooks=ControllerHooks(
            decompose=decompose,
            retrieve=retrieve,
            fetch_parse=fetch_parse,
            extract_claims=_make_extract(
                {
                    "arxiv:1": [_claim("arxiv:1", grounded=True)],
                    "arxiv:2": [_claim("arxiv:2", grounded=True)],
                    "arxiv:3": [_claim("arxiv:3", grounded=True)],
                }
            ),
            detect_contradictions=lambda c, p, **kw: [pair],
            generate=lambda prompt, **kw: "Review.",
            validate_cites=lambda text, papers: ([], [], [], 0.0),
        ),
    )
    state = controller.run("long context retrieval")
    actions = [s.action for s in state.trace]
    detect_i = actions.index("detect_contradictions")
    last_extract_i = max(i for i, a in enumerate(actions) if a == "extract_claims")
    conflict_i = actions.index("resolve_conflict")
    assert last_extract_i < detect_i < conflict_i


# --------------------------------------------------------------------------- #
# run_conflict_resolution_loop
# --------------------------------------------------------------------------- #


def _conflict_pair() -> ContradictionPair:
    return ContradictionPair(
        paper_a="arxiv:1",
        paper_b="arxiv:2",
        claim_a="scaling improves accuracy on the retrieval benchmark datasets",
        claim_b="scaling degrades accuracy on the retrieval benchmark datasets",
    )


def _conflict_controller(
    candidates: list[ScoredPaper], extract_mapping: dict[str, object]
) -> SynthesisController:
    return SynthesisController(
        hooks=ControllerHooks(
            extract_claims=_make_extract(extract_mapping),
            retrieve=_make_retrieve([Paper(paper_id="arxiv:3", title="C", api_source="arxiv")]),
            fetch_parse=_make_fetch(candidates),
        )
    )


def test_contradiction_id_is_auto_and_unique() -> None:
    a = _conflict_pair()
    b = _conflict_pair()
    assert a.contradiction_id and b.contradiction_id
    assert a.contradiction_id != b.contradiction_id


def test_conflict_loop_grounds_resolution_from_third_paper() -> None:
    pair = _conflict_pair()
    state = SynthesisState(question="q", contradictions=[pair])
    third = ScoredPaper(paper_id="arxiv:3", title="C", full_text="body", text_tier="full_text")
    controller = _conflict_controller(
        candidates=[third],
        extract_mapping={"arxiv:3": [_claim("arxiv:3", grounded=True)]},
    )
    controller.run_conflict_resolution_loop(state)

    resolution = state.claims[-1]
    assert resolution.paper_id == "arxiv:3"
    assert resolution.grounded is True
    assert resolution.grounding_tier == "full_text"  # self-grounded at C's tier
    assert resolution.supporting_paper_id == "arxiv:3"


def test_conflict_loop_resolution_tier_follows_third_paper_text_tier() -> None:
    pair = _conflict_pair()
    state = SynthesisState(question="q", contradictions=[pair])
    third = ScoredPaper(paper_id="arxiv:3", title="C", abstract="abs", text_tier="abstract")
    controller = _conflict_controller(
        candidates=[third],
        extract_mapping={"arxiv:3": [_claim("arxiv:3", grounded=True)]},
    )
    controller.run_conflict_resolution_loop(state)
    assert state.claims[-1].grounding_tier == "abstract"


def test_conflict_loop_adds_third_paper_and_links_contradiction_id() -> None:
    pair = _conflict_pair()
    state = SynthesisState(question="q", contradictions=[pair])
    third = ScoredPaper(paper_id="arxiv:3", title="C", full_text="body", text_tier="full_text")
    controller = _conflict_controller(
        candidates=[third],
        extract_mapping={"arxiv:3": [_claim("arxiv:3", grounded=True)]},
    )
    controller.run_conflict_resolution_loop(state)

    step = [s for s in state.trace if s.action == "resolve_conflict"][0]
    assert step.result == "ok"
    assert step.effect.resolved_conflict == pair.contradiction_id
    assert "arxiv:3" in [p.paper_id for p in state.papers]
    assert step.effect.added_paper_ids == ["arxiv:3"]
    assert step.effect.claim_refs == [state.claims[-1].claim_id]


def test_conflict_loop_skips_the_two_conflicting_papers() -> None:
    pair = _conflict_pair()
    state = SynthesisState(question="q", contradictions=[pair])
    # candidate list includes the conflicting papers first; they must be skipped
    cand_a = ScoredPaper(paper_id="arxiv:1", title="A", full_text="body", text_tier="full_text")
    cand_b = ScoredPaper(paper_id="arxiv:2", title="B", full_text="body", text_tier="full_text")
    third = ScoredPaper(paper_id="arxiv:3", title="C", full_text="body", text_tier="full_text")
    controller = _conflict_controller(
        candidates=[cand_a, cand_b, third],
        extract_mapping={
            "arxiv:1": [_claim("arxiv:1", grounded=True)],
            "arxiv:2": [_claim("arxiv:2", grounded=True)],
            "arxiv:3": [_claim("arxiv:3", grounded=True)],
        },
    )
    controller.run_conflict_resolution_loop(state)
    assert state.claims[-1].paper_id == "arxiv:3"


def test_conflict_loop_insufficient_when_no_third_paper_grounds() -> None:
    pair = _conflict_pair()
    state = SynthesisState(question="q", contradictions=[pair])
    third = ScoredPaper(paper_id="arxiv:3", title="C", full_text="body", text_tier="full_text")
    controller = _conflict_controller(
        candidates=[third],
        extract_mapping={"arxiv:3": [_claim("arxiv:3", grounded=False)]},
    )
    controller.run_conflict_resolution_loop(state)
    step = [s for s in state.trace if s.action == "resolve_conflict"][0]
    assert step.result == "insufficient"
    assert step.effect.resolved_conflict == pair.contradiction_id
    assert state.claims == []


def test_conflict_loop_derives_search_terms_from_both_claims() -> None:
    pair = _conflict_pair()
    state = SynthesisState(question="q", contradictions=[pair])
    third = ScoredPaper(paper_id="arxiv:3", title="C", full_text="body", text_tier="full_text")
    controller = _conflict_controller(
        candidates=[third],
        extract_mapping={"arxiv:3": [_claim("arxiv:3", grounded=True)]},
    )
    controller.run_conflict_resolution_loop(state)
    step = [s for s in state.trace if s.action == "resolve_conflict"][0]
    assert "scaling" in step.params.search_terms
    assert step.params.paper_a == "arxiv:1"
    assert step.params.paper_b == "arxiv:2"


def test_conflict_loop_failed_retrieve_is_logged_not_raised() -> None:
    pair = _conflict_pair()
    state = SynthesisState(question="q", contradictions=[pair])

    def _boom(rq: ResearchQuestion, **kwargs: object) -> list[Paper]:
        raise RuntimeError("network down")

    controller = SynthesisController(
        hooks=ControllerHooks(
            extract_claims=_make_extract({}),
            retrieve=_boom,
            fetch_parse=_make_fetch([]),
        )
    )
    controller.run_conflict_resolution_loop(state)  # must not raise
    step = [s for s in state.trace if s.action == "resolve_conflict"][0]
    assert step.result == "failed"
    assert "network down" in step.result_note
    assert step.effect.resolved_conflict == pair.contradiction_id


# --------------------------------------------------------------------------- #
# run_synthesize_loop
# --------------------------------------------------------------------------- #


def test_synthesize_loop_populates_review_and_citation_state() -> None:
    paper = ScoredPaper(
        paper_id="arxiv:1",
        title="A",
        authors=["Ada Lovelace"],
        year=2024,
        abstract="abstract",
        text_tier="abstract",
    )
    claim = _claim("arxiv:1", grounded=True)
    check = CitationCheck(
        citation_key="[Lovelace 2024]",
        resolved_paper_id="arxiv:1",
        is_valid=True,
    )
    state = SynthesisState(question="long context retrieval", papers=[paper], claims=[claim])
    controller = SynthesisController(
        hooks=ControllerHooks(
            generate=lambda prompt, **kw: "Review body [Lovelace 2024].",
            validate_cites=lambda text, papers: ([check], ["arxiv:1"], [], 1.0),
        )
    )
    controller.run_synthesize_loop(state)
    assert state.review_text == "Review body [Lovelace 2024]."
    assert state.citation_checks == [check]
    assert state.hallucinated_citations == []
    assert state.to_result().citations_used == ["arxiv:1"]


def test_synthesize_loop_uses_relevant_unique_citation_set() -> None:
    primary = ScoredPaper(
        paper_id="arxiv:1",
        title="Long context retrieval benchmarks",
        authors=["Alice Liu", "Ada Lovelace"],
        year=2025,
        abstract="long context retrieval benchmarks for large language models",
        text_tier="abstract",
    )
    duplicate_key = ScoredPaper(
        paper_id="arxiv:2",
        title="Long context retrieval datasets",
        authors=["Bob Liu", "Grace Hopper"],
        year=2025,
        abstract="long context retrieval datasets for large language models",
        text_tier="abstract",
    )
    off_topic = ScoredPaper(
        paper_id="arxiv:3",
        title="Low light video enhancement",
        authors=["Carol Zhang"],
        year=2025,
        abstract="image restoration and low light video enhancement",
        text_tier="abstract",
    )
    captured: dict[str, object] = {}

    def _build_prompt(**kwargs: object):
        captured["papers"] = kwargs["papers"]
        captured["claims"] = kwargs["claims"]
        from synthesis.prompt import build_synthesis_prompt

        prompt = build_synthesis_prompt(**kwargs)
        captured["expected_citations"] = prompt.expected_citations
        return prompt

    def _validate(text: str, papers: list[ScoredPaper]):
        captured["validated_papers"] = papers
        return ([], [], [], 0.0)

    state = SynthesisState(
        question="long context retrieval large language models",
        papers=[off_topic, duplicate_key, primary],
        claims=[
            _claim("arxiv:1", grounded=True),
            _claim("arxiv:2", grounded=True),
            _claim("arxiv:3", grounded=True),
        ],
    )
    controller = SynthesisController(
        config=ControllerConfig(min_relevance_score=0.03),
        hooks=ControllerHooks(
            build_prompt=_build_prompt,
            generate=lambda prompt, **kw: "Review.",
            validate_cites=_validate,
        ),
    )

    controller.run_synthesize_loop(state)

    papers = captured["papers"]
    kept_ids = {p.paper_id for p in papers}
    # The off-topic paper is dropped by the relevance floor, but the two papers
    # that share a [Liu et al. 2025] key are both kept (disambiguated), not one
    # silently discarded.
    assert kept_ids == {"arxiv:1", "arxiv:2"}
    assert "arxiv:3" not in kept_ids
    assert {p.paper_id for p in captured["validated_papers"]} == kept_ids
    assert {c.paper_id for c in captured["claims"]} == kept_ids
    # Both surviving papers receive distinct, resolvable citation keys.
    assert sorted(captured["expected_citations"]) == [
        "[Liu et al. 2025a]",
        "[Liu et al. 2025b]",
    ]


def test_synthesize_loop_logs_step_with_parent_and_claim_refs() -> None:
    claim = _claim("arxiv:1", grounded=True)
    state = SynthesisState(question="long context retrieval", papers=[_scored("arxiv:1")], claims=[claim])
    parent = state.log(
        DecisionStep.start(
            action="detect_contradictions",
            params=DetectContradictionsParams(claim_ids=[claim.claim_id]),
            trigger="prior step",
            rationale="prior rationale",
        )
    )
    parent.complete(result="noop")
    controller = SynthesisController(
        hooks=ControllerHooks(
            generate=lambda prompt, **kw: "Review.",
            validate_cites=lambda text, papers: ([], [], [], 0.0),
        )
    )
    controller.run_synthesize_loop(state)
    synth = state.trace[-1]
    assert synth.action == "synthesize"
    assert synth.parent_step_id == parent.step_id
    assert synth.params.word_budget == 500
    assert synth.effect.claim_refs == [claim.claim_id]
    assert synth.llm_calls == 1


def test_synthesize_loop_failed_stage_is_logged_not_raised() -> None:
    state = SynthesisState(question="long context retrieval", papers=[_scored("arxiv:1")])

    def _boom(**kwargs):
        raise RuntimeError("prompt broke")

    controller = SynthesisController(hooks=ControllerHooks(build_prompt=_boom))
    controller.run_synthesize_loop(state)  # must not raise
    synth = state.trace[-1]
    assert synth.action == "synthesize"
    assert synth.result == "failed"
    assert "prompt broke" in synth.result_note
    assert state.review_text is None


def test_synthesize_loop_to_result_carries_final_artifact() -> None:
    paper = ScoredPaper(
        paper_id="arxiv:1",
        title="A",
        authors=["Ada Lovelace"],
        year=2024,
        abstract="abstract",
        text_tier="abstract",
    )
    claim = _claim("arxiv:1", grounded=True)
    check = CitationCheck(
        citation_key="[Lovelace 2024]",
        resolved_paper_id="arxiv:1",
        is_valid=True,
    )
    state = SynthesisState(question="long context retrieval", papers=[paper], claims=[claim])
    controller = SynthesisController(
        hooks=ControllerHooks(
            generate=lambda prompt, **kw: "Grounded review [Lovelace 2024].",
            validate_cites=lambda text, papers: ([check], ["arxiv:1"], [], 1.0),
        )
    )
    controller.run_synthesize_loop(state)
    result = state.to_result()
    assert result.review_text == "Grounded review [Lovelace 2024]."
    assert result.confidence_score == 1.0
    assert result.claims == [claim]


# --------------------------------------------------------------------------- #
# integrated controller run
# --------------------------------------------------------------------------- #


def test_run_wires_existing_loops_with_unbroken_causal_trace() -> None:
    pair = _conflict_pair()
    calls: list[str] = []

    def decompose(question: str, *, n: int) -> ResearchQuestion:
        return ResearchQuestion(question=question, sub_queries=["initial query"])

    def retrieve(rq: ResearchQuestion, **kwargs: object) -> list[Paper]:
        calls.append(rq.sub_queries[0])
        if len(calls) == 1:
            return [_paper("arxiv:1"), _paper("arxiv:2")]
        if len(calls) == 2:
            return [_paper("arxiv:4", "Gap Support")]
        return [_paper("arxiv:3", "Conflict Resolver")]

    scored = {
        "arxiv:1": ScoredPaper(
            paper_id="arxiv:1",
            title="A",
            abstract="base abstract",
            text_tier="abstract",
        ),
        "arxiv:2": ScoredPaper(
            paper_id="arxiv:2",
            title="B",
            abstract="base abstract",
            text_tier="abstract",
        ),
        "arxiv:3": ScoredPaper(
            paper_id="arxiv:3",
            title="Conflict Resolver",
            full_text="resolution body",
            text_tier="full_text",
        ),
        "arxiv:4": ScoredPaper(
            paper_id="arxiv:4",
            title="Gap Support",
            full_text=f"paper four contains {_MARKER}",
            text_tier="full_text",
        ),
    }

    def fetch_parse(papers: list[Paper]) -> list[ScoredPaper]:
        return [scored[p.paper_id] for p in papers]

    ungrounded = _claim(
        "arxiv:2",
        grounded=False,
        quote=_MARKER,
        text="transformer retrieval accuracy benchmark datasets improves",
    )
    extract_mapping = {
        "arxiv:1": [_claim("arxiv:1", grounded=True)],
        "arxiv:2": [ungrounded],
        "arxiv:3": [_claim("arxiv:3", grounded=True, text="third paper contextualizes the conflict")],
    }

    def detect(claims: list[ClaimRecord], papers: list[ScoredPaper], **kwargs: object):
        return [pair]

    controller = SynthesisController(
        config=_cfg(min_relevant_papers=2),
        hooks=ControllerHooks(
            decompose=decompose,
            retrieve=retrieve,
            fetch_parse=fetch_parse,
            extract_claims=_make_extract(extract_mapping),
            detect_contradictions=detect,
            generate=lambda prompt, **kw: "Review.",
            validate_cites=lambda text, papers: ([], [], [], 0.0),
        ),
    )

    state = controller.run("long context retrieval")

    assert state.terminal_reason == "synthesized"
    assert [s.action for s in state.trace] == [
        "decompose",
        "search",
        "synthesize",
        "extract_claims",
        "extract_claims",
        "detect_contradictions",
        "hunt_support",
        "resolve_conflict",
        "synthesize",
    ]

    # Detect step links back to the last extract step and lists the new ids.
    last_extract = [s for s in state.trace if s.action == "extract_claims"][-1]
    detect_step = next(s for s in state.trace if s.action == "detect_contradictions")
    assert detect_step.parent_step_id == last_extract.step_id
    assert detect_step.effect.contradiction_ids == [pair.contradiction_id]
    assert state.contradictions[0].contradiction_id == pair.contradiction_id

    # Gap step causally points back to the extract step that produced the claim.
    extract_for_ungrounded = next(
        s for s in state.trace if ungrounded.claim_id in s.effect.claim_refs
    )
    gap_step = next(s for s in state.trace if s.action == "hunt_support")
    assert gap_step.parent_step_id == extract_for_ungrounded.step_id
    assert gap_step.effect.claim_ref == ungrounded.claim_id
    assert gap_step.effect.tier_before == "none"
    assert gap_step.effect.tier_after == "corroborated"

    # Conflict step is linked by stable contradiction_id, not by list position.
    conflict_step = next(s for s in state.trace if s.action == "resolve_conflict")
    assert conflict_step.effect.resolved_conflict == pair.contradiction_id
    assert conflict_step.effect.claim_refs

    paper_ids = {p.paper_id for p in state.papers}
    assert {"arxiv:1", "arxiv:2", "arxiv:3", "arxiv:4"}.issubset(paper_ids)
    assert ungrounded.grounding_tier == "corroborated"
    assert ungrounded.supporting_paper_id == "arxiv:4"
    assert state.get_claim(conflict_step.effect.claim_refs[0]) is not None
    assert state.review_text == "Review."


def test_controller_records_reformulation_cap_terminal_reason() -> None:
    def decompose(question: str, *, n: int) -> ResearchQuestion:
        return ResearchQuestion(question=question, sub_queries=["initial query"])

    def retrieve(rq: ResearchQuestion, **kwargs: object) -> list[Paper]:
        return [_paper("arxiv:1")]

    def fetch_parse(papers: list[Paper]) -> list[ScoredPaper]:
        return [_scored(p.paper_id, p.title) for p in papers]

    controller = SynthesisController(
        config=_cfg(min_relevant_papers=4, max_reformulations=1),
        hooks=ControllerHooks(decompose=decompose, retrieve=retrieve, fetch_parse=fetch_parse),
    )

    state = controller.run_retrieval_loop("long context retrieval")

    assert state.terminal_reason == "reformulation_cap"
    assert state.reformulation_count == 1
    assert [s.action for s in state.trace] == [
        "decompose",
        "search",
        "reformulate",
        "search",
        "synthesize",
    ]
    assert state.trace[-2].result == "insufficient"


# --------------------------------------------------------------------------- #
# retrieval relevance
# --------------------------------------------------------------------------- #


def test_gap_hunt_issues_single_combined_query_not_per_word() -> None:
    state, _claim_rec, controller = _state_with_ungrounded_claim()
    captured: list[ResearchQuestion] = []

    def _retrieve(rq: ResearchQuestion, **kwargs: object) -> list[Paper]:
        captured.append(rq)
        return [Paper(paper_id="arxiv:2", title="B", api_source="arxiv")]

    controller.hooks.retrieve = _retrieve
    controller.hooks.fetch_parse = _make_fetch([_supporting_paper()])
    controller.run_gap_detection_loop(state)

    assert len(captured) == 1
    rq = captured[0]
    # One combined phrase query, not one single-word query per derived keyword.
    assert len(rq.sub_queries) == 1
    assert len(rq.sub_queries[0].split()) > 1


def test_conflict_hunt_issues_single_combined_query_not_per_word() -> None:
    pair = _conflict_pair()
    state = SynthesisState(question="q", contradictions=[pair])
    captured: list[ResearchQuestion] = []

    def _retrieve(rq: ResearchQuestion, **kwargs: object) -> list[Paper]:
        captured.append(rq)
        return [Paper(paper_id="arxiv:3", title="C", api_source="arxiv")]

    controller = SynthesisController(
        hooks=ControllerHooks(
            extract_claims=_make_extract({}),
            retrieve=_retrieve,
            fetch_parse=_make_fetch([]),
        )
    )
    controller.run_conflict_resolution_loop(state)

    assert len(captured) == 1
    assert len(captured[0].sub_queries) == 1
    assert len(captured[0].sub_queries[0].split()) > 1


def test_execute_search_relevance_ranks_and_trims_working_set() -> None:
    def decompose(question: str, *, n: int) -> ResearchQuestion:
        return ResearchQuestion(question=question, sub_queries=["graph neural networks"])

    def retrieve(rq: ResearchQuestion, **kwargs: object) -> list[Paper]:
        return [_paper("arxiv:1"), _paper("arxiv:2"), _paper("arxiv:3")]

    on_topic = ScoredPaper(
        paper_id="arxiv:3",
        title="Graph neural networks for retrieval",
        abstract="graph neural networks improve graph retrieval",
        text_tier="abstract",
    )
    off_topic_a = ScoredPaper(
        paper_id="arxiv:1",
        title="Supernova survey",
        abstract="astrophysical supernova spectra and stellar remnants",
        text_tier="abstract",
    )
    off_topic_b = ScoredPaper(
        paper_id="arxiv:2",
        title="Pasta recipes",
        abstract="boiling water timing for fresh pasta",
        text_tier="abstract",
    )

    def fetch_parse(papers: list[Paper]) -> list[ScoredPaper]:
        return [off_topic_a, off_topic_b, on_topic]

    controller = SynthesisController(
        config=_cfg(min_relevant_papers=1, total_paper_limit=2),
        hooks=ControllerHooks(decompose=decompose, retrieve=retrieve, fetch_parse=fetch_parse),
    )

    state = controller.run_retrieval_loop("graph neural networks")

    # Working set is trimmed to the limit, and the on-topic paper wins on
    # relevance even though the providers returned it last.
    assert len(state.papers) == 2
    assert state.papers[0].paper_id == "arxiv:3"
    assert state.papers[0].relevance_score > 0


def test_synthesis_inputs_keeps_papers_that_share_a_citation_key() -> None:
    controller = SynthesisController()
    liu_a = ScoredPaper(
        paper_id="arxiv:liu_a",
        title="Long context survey",
        authors=["Hao Liu", "Wei Chen"],
        year=2025,
        abstract="long context retrieval survey",
        text_tier="abstract",
    )
    liu_b = ScoredPaper(
        paper_id="arxiv:liu_b",
        title="Retrieval benchmark",
        authors=["Hao Liu", "Mei Wang"],
        year=2025,
        abstract="long context retrieval benchmark",
        text_tier="abstract",
    )
    state = SynthesisState(
        question="long context retrieval",
        papers=[liu_a, liu_b],
        claims=[
            _claim("arxiv:liu_a", grounded=True),
            _claim("arxiv:liu_b", grounded=True),
        ],
    )

    papers, claims, _contradictions = controller._synthesis_inputs(state)

    # Both same-key papers are kept (disambiguated downstream), not dropped.
    assert {p.paper_id for p in papers} == {"arxiv:liu_a", "arxiv:liu_b"}
    assert {c.paper_id for c in claims} == {"arxiv:liu_a", "arxiv:liu_b"}
