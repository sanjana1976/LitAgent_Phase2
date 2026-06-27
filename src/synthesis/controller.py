"""
Agentic controller spine for LitSynth.

This module is the first replacement layer for the old linear synthesis
pipeline. It does not try to implement every downstream feature at once; it
establishes the decision loop that makes those features possible:

1. read :class:`SynthesisState`
2. choose the next typed :class:`DecisionStep`
3. execute it
4. write the result back to state and complete the trace step

The first real behavior implemented here is adaptive retrieval: if the agent
does not have enough parsed papers, it reformulates the query and tries again,
with every decision recorded as structured trace data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from synthesis.decompose import decompose_question
from synthesis.rank import rank_papers
from synthesis.schemas import (
    ClaimRecord,
    ContradictionPair,
    ResearchQuestion,
    ScoredPaper,
    SynthesisResult,
)
from synthesis.reformulate import default_reformulate, resolve_reformulated_query
from synthesis.state import Gap, SynthesisState
from synthesis.trace import (
    DecisionStep,
    DecomposeParams,
    DetectContradictionsParams,
    ExtractClaimsParams,
    GapHuntParams,
    ReformulateParams,
    ResolveConflictParams,
    SearchParams,
    SynthesizeParams,
    StepEffect,
)

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "is", "are", "be", "by", "that", "this", "these", "those", "we", "our",
        "it", "its", "as", "at", "from", "than", "using", "use", "can", "which",
        "their", "they", "show", "shows", "shown", "achieves", "achieve",
    }
)


@dataclass(frozen=True)
class ControllerConfig:
    """Policy knobs for the first agentic controller loop."""

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
    sources: tuple[str, ...] = ("arxiv",)


@dataclass
class ControllerHooks:
    """Injectable stage functions for tests and later live execution."""

    decompose: Callable[..., ResearchQuestion] = field(default=decompose_question)
    retrieve: Callable[..., list[Any]] | None = None
    fetch_parse: Callable[..., list[ScoredPaper]] | None = None
    extract_claims: Callable[..., list[ClaimRecord]] | None = None
    detect_contradictions: Callable[..., list[ContradictionPair]] | None = None
    build_prompt: Callable[..., Any] | None = None
    generate: Callable[..., str] | None = None
    validate_cites: Callable[..., tuple] | None = None
    reformulate: Callable[[SynthesisState, str], str] | None = None


def _last_step_id(state: SynthesisState) -> str | None:
    return state.trace[-1].step_id if state.trace else None


def _paper_ids(papers: list[ScoredPaper]) -> set[str]:
    return {p.paper_id for p in papers}


def _derive_search_terms(claim_text: str, *, limit: int = 6) -> list[str]:
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


def _stamp_grounding(claim: ClaimRecord, paper: ScoredPaper) -> None:
    """
    Normalize a claim's grounding fields from its originating paper's tier.

    Idempotent with the extractor's own stamping; applied controller-side so the
    loop is correct even when an injected ``extract_claims`` hook does not stamp.
    """
    if claim.grounded:
        tier = paper.text_tier if paper.text_tier in ("full_text", "abstract") else "abstract"
        claim.grounding_tier = tier
        claim.supporting_paper_id = paper.paper_id
    else:
        claim.grounding_tier = "none"
        claim.supporting_paper_id = None


def next_action(state: SynthesisState, config: ControllerConfig | None = None) -> DecisionStep:
    """
    Decide the next controller action from the current state.

    The policy is intentionally small and explicit:
    - no sub-queries -> decompose
    - no papers or last action was a reformulation -> search
    - fewer than ``min_relevant_papers`` -> reformulate until cap, then stop
    - enough papers -> no-op synthesize placeholder (downstream controller work)
    """
    cfg = config or ControllerConfig()
    parent = _last_step_id(state)

    if not state.sub_queries:
        return DecisionStep.start(
            action="decompose",
            params=DecomposeParams(question=state.question, n=cfg.sub_query_count),
            trigger="no sub-queries have been generated yet",
            rationale="decompose the question before retrieval so search covers multiple angles",
            parent_step_id=parent,
        )

    if not state.papers or (state.trace and state.trace[-1].action == "reformulate"):
        query = state.sub_queries[-1]
        return DecisionStep.start(
            action="search",
            params=SearchParams(query=query, sources=list(cfg.sources)),
            trigger=f"working set has {len(state.papers)} parsed paper(s)",
            rationale="search the current query and add any newly parsed papers to state",
            parent_step_id=parent,
        )

    if len(state.papers) < cfg.min_relevant_papers:
        if state.reformulation_count >= cfg.max_reformulations:
            return DecisionStep.start(
                action="synthesize",
                params=SynthesizeParams(word_budget=0),
                trigger=(
                    f"only {len(state.papers)} paper(s) after "
                    f"{state.reformulation_count} reformulation(s)"
                ),
                rationale="stop retrieval because the reformulation cap has been reached",
                parent_step_id=parent,
            )
        current_query = state.sub_queries[-1]
        new_query = default_reformulate(state)
        return DecisionStep.start(
            action="reformulate",
            params=ReformulateParams(original_query=current_query, new_query=new_query),
            trigger=(
                f"retrieval found only {len(state.papers)} paper(s), below "
                f"the target of {cfg.min_relevant_papers}"
            ),
            rationale="broaden and retarget the query before another search",
            parent_step_id=parent,
        )

    return DecisionStep.start(
        action="synthesize",
        params=SynthesizeParams(word_budget=0),
        trigger=f"working set has {len(state.papers)} parsed paper(s)",
        rationale="enough evidence has been gathered for downstream synthesis",
        parent_step_id=parent,
    )


class SynthesisController:
    """Execute the first agentic synthesis loop over :class:`SynthesisState`."""

    def __init__(
        self,
        *,
        config: ControllerConfig | None = None,
        hooks: ControllerHooks | None = None,
    ) -> None:
        self.config = config or ControllerConfig()
        self.hooks = hooks or ControllerHooks()

    # ------------------------------------------------------------------ #
    # Lazy stage resolvers (avoid the tools -> pipeline import cycle)
    # ------------------------------------------------------------------ #

    def _retrieve_fn(self) -> Callable[..., list[Any]]:
        if self.hooks.retrieve is not None:
            return self.hooks.retrieve
        from synthesis.retrieve import retrieve_papers

        return retrieve_papers

    def _fetch_parse_fn(self) -> Callable[..., list[ScoredPaper]]:
        if self.hooks.fetch_parse is not None:
            return self.hooks.fetch_parse
        from synthesis.fetch_parse import fetch_and_parse

        return fetch_and_parse

    def _extract_claims_fn(self) -> Callable[..., list[ClaimRecord]]:
        if self.hooks.extract_claims is not None:
            return self.hooks.extract_claims
        from synthesis.claims import extract_claims

        return extract_claims

    def _detect_contradictions_fn(self) -> Callable[..., list[ContradictionPair]]:
        if self.hooks.detect_contradictions is not None:
            return self.hooks.detect_contradictions
        from synthesis.contradictions import detect_contradictions

        return detect_contradictions

    def _build_prompt_fn(self) -> Callable[..., Any]:
        if self.hooks.build_prompt is not None:
            return self.hooks.build_prompt
        from synthesis.prompt import build_synthesis_prompt

        return build_synthesis_prompt

    def _generate_fn(self) -> Callable[..., str]:
        if self.hooks.generate is not None:
            return self.hooks.generate
        from synthesis.generate import generate_literature_review

        return generate_literature_review

    def _validate_cites_fn(self) -> Callable[..., tuple]:
        if self.hooks.validate_cites is not None:
            return self.hooks.validate_cites
        from synthesis.validate_cites import validate_citations

        return validate_citations

    def run_retrieval_loop(self, question: str) -> SynthesisState:
        """
        Run through adaptive retrieval and return the traced state.

        This intentionally stops before generation. It is the demoable spine:
        decompose -> search -> maybe reformulate/search again -> terminal reason.
        """
        state = SynthesisState(question=question)
        while state.terminal_reason is None:
            step = next_action(state, self.config)
            state.log(step)
            self._execute(state, step)
            if step.action == "synthesize":
                if len(state.papers) >= self.config.min_relevant_papers:
                    state.terminal_reason = "synthesized"
                else:
                    state.terminal_reason = "reformulation_cap"
        return state

    def run(self, question: str) -> SynthesisState:
        """
        Run the currently-built controller loops as one traced workflow.

        Thin orchestration over the isolated loops, in dependency order:
        retrieval -> claims -> detect contradictions -> gap detection ->
        conflict resolution. Contradiction detection must precede conflict
        resolution (which reads ``state.contradictions``); gap detection is
        independent of both, so it simply runs after detection.
        """
        state = self.run_retrieval_loop(question)
        if state.terminal_reason == "error":
            return state

        retrieval_reason = state.terminal_reason
        state.terminal_reason = None
        self.run_claims_loop(state)
        self.run_detect_contradictions_loop(state)
        self.run_gap_detection_loop(state)
        self.run_conflict_resolution_loop(state)
        self.run_synthesize_loop(state)

        state.terminal_reason = retrieval_reason or "synthesized"
        return state

    def run_claims_loop(self, state: SynthesisState) -> SynthesisState:
        """
        Extract grounded claims, one traced step per paper.

        Each paper gets its own ``extract_claims`` :class:`DecisionStep`. Claims
        are grounding-stamped from the paper's ``text_tier`` and appended to
        ``state.claims``. A failed extraction for one paper is logged with
        ``result="failed"`` and does not abort the loop.
        """
        extract = self._extract_claims_fn()
        for paper in list(state.papers):
            step = state.log(
                DecisionStep.start(
                    action="extract_claims",
                    params=ExtractClaimsParams(paper_ids=[paper.paper_id]),
                    trigger="paper available for claim extraction",
                    rationale="extract grounded, paper-specific claims from this paper",
                )
            )
            start = time.perf_counter()
            try:
                claims = extract(
                    [paper], max_claims_per_paper=self.config.max_claims_per_paper
                )
            except Exception as exc:  # noqa: BLE001 - one paper must not kill the loop
                step.complete(
                    result="failed",
                    result_note=f"{type(exc).__name__}: {exc}",
                    duration_ms=_elapsed_ms(start),
                )
                continue

            for claim in claims:
                _stamp_grounding(claim, paper)
            state.claims.extend(claims)
            step.complete(
                result="ok" if claims else "noop",
                result_note=f"extracted {len(claims)} claim(s) from {paper.paper_id}",
                effect=StepEffect(claim_refs=[c.claim_id for c in claims]),
                duration_ms=_elapsed_ms(start),
            )
        return state

    def run_detect_contradictions_loop(self, state: SynthesisState) -> SynthesisState:
        """
        Detect cross-paper contradictions over the accumulated claims.

        Emits one ``detect_contradictions`` step, causally linked to the last
        ``extract_claims`` step. Returned pairs (each with a stable
        ``contradiction_id``) are appended to ``state.contradictions`` and their
        ids recorded on the step effect. Empty -> ``noop``; error -> ``failed``.
        """
        detect = self._detect_contradictions_fn()
        claim_ids = [c.claim_id for c in state.claims]
        last_extract = next(
            (s.step_id for s in reversed(state.trace) if s.action == "extract_claims"),
            None,
        )
        step = state.log(
            DecisionStep.start(
                action="detect_contradictions",
                params=DetectContradictionsParams(claim_ids=claim_ids),
                trigger="claims extracted, contradiction detection running",
                rationale="surface genuine cross-paper tensions before resolution",
                parent_step_id=last_extract,
            )
        )
        start = time.perf_counter()

        try:
            pairs = detect(
                state.claims,
                state.papers,
                max_pairs=self.config.max_contradiction_pairs,
            )
        except Exception as exc:  # noqa: BLE001 - detection must not abort the run
            step.complete(
                result="failed",
                result_note=f"{type(exc).__name__}: {exc}",
                duration_ms=_elapsed_ms(start),
            )
            return state

        new_ids: list[str] = []
        for pair in pairs:
            if not pair.contradiction_id:
                pair.contradiction_id = uuid4().hex
            state.contradictions.append(pair)
            new_ids.append(pair.contradiction_id)

        step.complete(
            result="ok" if new_ids else "noop",
            result_note=f"detected {len(new_ids)} contradiction(s)",
            effect=StepEffect(contradiction_ids=new_ids),
            duration_ms=_elapsed_ms(start),
        )
        return state

    def run_gap_detection_loop(self, state: SynthesisState) -> SynthesisState:
        """
        For each ungrounded claim, autonomously hunt a corroborating paper.

        An ungrounded claim (``grounded=False`` and ``grounding_tier="none"``)
        opens an ``ungrounded_claim`` :class:`Gap` and triggers a targeted
        ``hunt_support`` step, causally linked to the ``extract_claims`` step
        that produced the claim. On success the claim is upgraded to
        ``"corroborated"``; otherwise the gap is flagged ``flagged_unverified``.
        """
        claim_to_extract_step = self._claim_extract_step_map(state)
        retrieve = self._retrieve_fn()
        fetch_parse = self._fetch_parse_fn()

        for claim in list(state.claims):
            if claim.grounded or claim.grounding_tier != "none":
                continue

            gap = Gap(
                kind="ungrounded_claim",
                description=f"ungrounded claim: {claim.claim[:80]}",
                origin_claim_ref=claim.claim_id,
                status="hunting",
            )
            state.gaps.append(gap)

            terms = _derive_search_terms(claim.claim, limit=self.config.gap_search_terms)
            step = state.log(
                DecisionStep.start(
                    action="hunt_support",
                    params=GapHuntParams(claim_id=claim.claim_id, search_terms=terms),
                    trigger="ungrounded claim detected",
                    rationale="search for a different paper that grounds this claim",
                    parent_step_id=claim_to_extract_step.get(claim.claim_id),
                )
            )
            start = time.perf_counter()

            try:
                rq = self._single_query_rq(terms, claim.claim)
                raw = retrieve(
                    rq,
                    per_query_limit=self.config.per_query_limit,
                    total_limit=self.config.total_paper_limit,
                    sources=self.config.sources,
                )
                candidates = self._rank_candidates(fetch_parse(raw), state.question)
            except Exception as exc:  # noqa: BLE001 - one hunt must not kill the loop
                gap.status = "flagged_unverified"
                step.complete(
                    result="failed",
                    result_note=f"{type(exc).__name__}: {exc}",
                    effect=StepEffect(claim_ref=claim.claim_id, gap_ref=gap.gap_id),
                    duration_ms=_elapsed_ms(start),
                )
                continue

            supporting = self._first_supporting_paper(claim, candidates)
            if supporting is not None:
                claim.grounded = True
                claim.grounding_tier = "corroborated"
                claim.supporting_paper_id = supporting.paper_id
                gap.status = "resolved"
                gap.resolved_by_paper_id = supporting.paper_id
                gap.resolved_by_step_id = step.step_id
                added: list[str] = []
                if state.get_paper(supporting.paper_id) is None:
                    state.papers.append(supporting)
                    added = [supporting.paper_id]
                step.complete(
                    result="ok",
                    result_note=f"claim corroborated by {supporting.paper_id}",
                    effect=StepEffect(
                        claim_ref=claim.claim_id,
                        tier_before="none",
                        tier_after="corroborated",
                        added_paper_ids=added,
                        gap_ref=gap.gap_id,
                    ),
                    duration_ms=_elapsed_ms(start),
                )
            else:
                gap.status = "flagged_unverified"
                step.complete(
                    result="insufficient",
                    result_note="no candidate paper grounded the claim",
                    effect=StepEffect(claim_ref=claim.claim_id, gap_ref=gap.gap_id),
                    duration_ms=_elapsed_ms(start),
                )
        return state

    def run_conflict_resolution_loop(self, state: SynthesisState) -> SynthesisState:
        """
        For each contradiction, hunt a third paper that contextualizes it.

        Two papers in tension trigger a ``resolve_conflict`` step that searches
        for a *different* third paper. If a candidate yields a grounded claim,
        that claim becomes the resolution: it is sourced from and self-grounded
        in the third paper (tier-tagged from its ``text_tier``, exactly like any
        other claim), appended to ``state.claims``, and linked to the
        contradiction via ``StepEffect.resolved_conflict``. If no third paper
        grounds a resolution, the step is logged ``insufficient``.
        """
        extract = self._extract_claims_fn()
        retrieve = self._retrieve_fn()
        fetch_parse = self._fetch_parse_fn()

        for pair in list(state.contradictions):
            terms = _derive_search_terms(
                f"{pair.claim_a} {pair.claim_b}", limit=self.config.gap_search_terms
            )
            step = state.log(
                DecisionStep.start(
                    action="resolve_conflict",
                    params=ResolveConflictParams(
                        paper_a=pair.paper_a, paper_b=pair.paper_b, search_terms=terms
                    ),
                    trigger=f"papers {pair.paper_a} and {pair.paper_b} disagree",
                    rationale="hunt a third paper that contextualizes the disagreement",
                )
            )
            start = time.perf_counter()

            try:
                rq = self._single_query_rq(terms, pair.claim_a)
                raw = retrieve(
                    rq,
                    per_query_limit=self.config.per_query_limit,
                    total_limit=self.config.total_paper_limit,
                    sources=self.config.sources,
                )
                candidates = self._rank_candidates(fetch_parse(raw), state.question)
            except Exception as exc:  # noqa: BLE001 - one conflict must not kill the loop
                step.complete(
                    result="failed",
                    result_note=f"{type(exc).__name__}: {exc}",
                    effect=StepEffect(resolved_conflict=pair.contradiction_id),
                    duration_ms=_elapsed_ms(start),
                )
                continue

            resolution, paper_c = self._first_grounded_resolution(pair, candidates, extract)
            if resolution is not None and paper_c is not None:
                state.claims.append(resolution)
                added: list[str] = []
                if state.get_paper(paper_c.paper_id) is None:
                    state.papers.append(paper_c)
                    added = [paper_c.paper_id]
                step.complete(
                    result="ok",
                    result_note=f"conflict contextualized by {paper_c.paper_id}",
                    effect=StepEffect(
                        resolved_conflict=pair.contradiction_id,
                        added_paper_ids=added,
                        claim_ref=resolution.claim_id,
                        claim_refs=[resolution.claim_id],
                    ),
                    duration_ms=_elapsed_ms(start),
                )
            else:
                step.complete(
                    result="insufficient",
                    result_note="no third paper grounded a resolution",
                    effect=StepEffect(resolved_conflict=pair.contradiction_id),
                    duration_ms=_elapsed_ms(start),
                )
        return state

    def run_synthesize_loop(self, state: SynthesisState) -> SynthesisState:
        """
        Generate the final review and validate citations against gathered papers.

        This is the terminal state mutation before callers ask ``state.to_result()``
        for the immutable artifact. The stage reuses the existing prompt,
        generation and citation-validation modules through hooks so tests can
        stay deterministic while production can still call the real LLM stage.
        """
        build_prompt = self._build_prompt_fn()
        generate = self._generate_fn()
        validate_cites = self._validate_cites_fn()
        step = state.log(
            DecisionStep.start(
                action="synthesize",
                params=SynthesizeParams(word_budget=self.config.word_budget),
                trigger="retrieval, claims, gaps, and conflicts have been processed",
                rationale="generate the final literature review from accumulated state",
                parent_step_id=_last_step_id(state),
            )
        )
        start = time.perf_counter()

        try:
            papers, claims, contradictions = self._synthesis_inputs(state)
            prompt = build_prompt(
                question=state.question,
                papers=papers,
                claims=claims,
                contradictions=contradictions,
                word_budget=self.config.word_budget,
            )
            review_text = generate(prompt, word_budget=self.config.word_budget)
            citation_checks, _citations_used, hallucinated, _citation_score = validate_cites(
                review_text, papers
            )
        except Exception as exc:  # noqa: BLE001 - preserve trace; result can still be inspected
            step.complete(
                result="failed",
                result_note=f"{type(exc).__name__}: {exc}",
                duration_ms=_elapsed_ms(start),
            )
            return state

        state.review_text = review_text
        state.citation_checks = list(citation_checks)
        state.hallucinated_citations = list(hallucinated)
        step.complete(
            result="ok" if review_text.strip() else "insufficient",
            result_note=(
                f"generated review with {len(state.citation_checks)} citation check(s) "
                f"and {len(state.hallucinated_citations)} hallucinated citation(s)"
            ),
            effect=StepEffect(claim_refs=[c.claim_id for c in state.claims]),
            llm_calls=1,
            duration_ms=_elapsed_ms(start),
        )
        return state

    def _synthesis_inputs(
        self, state: SynthesisState
    ) -> tuple[list[ScoredPaper], list[ClaimRecord], list[ContradictionPair]]:
        """
        Select the evidence set used by the final generator.

        Gap/conflict hunts may temporarily add supporting papers that are useful
        for verification but only weakly related to the user's question. Rank and
        threshold the accumulated set again before generation so tangential hunt
        results do not dominate the final review. Papers that share a short
        citation key (e.g. two ``[Liu et al. 2025]`` papers) are kept rather than
        dropped: ``assign_citation_keys`` disambiguates them downstream
        (``2025a``/``2025b``) so both remain citable and resolvable, instead of
        silently discarding a relevant source.
        """
        papers = rank_papers(
            state.papers,
            question=state.question,
            top_n=self.config.total_paper_limit,
            min_score=self.config.min_relevance_score,
        )

        paper_ids = {p.paper_id for p in papers}
        claims = [c for c in state.claims if c.paper_id in paper_ids]
        contradictions = [
            pair
            for pair in state.contradictions
            if pair.paper_a in paper_ids and pair.paper_b in paper_ids
        ]
        return papers, claims, contradictions

    def _first_grounded_resolution(
        self,
        pair: ContradictionPair,
        candidates: list[ScoredPaper],
        extract: Callable[..., list[ClaimRecord]],
    ) -> tuple[ClaimRecord | None, ScoredPaper | None]:
        """Return the first grounded claim from a *third* candidate paper."""
        for cand in candidates:
            if cand.paper_id in (pair.paper_a, pair.paper_b):
                continue
            cand_claims = extract(
                [cand], max_claims_per_paper=self.config.max_claims_per_paper
            )
            for claim in cand_claims:
                _stamp_grounding(claim, cand)
            grounded = next((c for c in cand_claims if c.grounded), None)
            if grounded is not None:
                return grounded, cand
        return None, None

    @staticmethod
    def _claim_extract_step_map(state: SynthesisState) -> dict[str, str]:
        """Map each claim_id to the extract_claims step that produced it."""
        mapping: dict[str, str] = {}
        for s in state.trace:
            if s.action == "extract_claims":
                for cid in s.effect.claim_refs:
                    mapping[cid] = s.step_id
        return mapping

    @staticmethod
    def _first_supporting_paper(
        claim: ClaimRecord, candidates: list[ScoredPaper]
    ) -> ScoredPaper | None:
        """Return the first *different* candidate whose text grounds the claim."""
        from synthesis.claims import quote_is_grounded

        for cand in candidates:
            if cand.paper_id == claim.paper_id:
                continue
            source_text = cand.full_text or cand.abstract or ""
            if quote_is_grounded(claim.evidence_quote, source_text):
                return cand
        return None

    @staticmethod
    def _single_query_rq(terms: list[str], fallback: str) -> ResearchQuestion:
        """
        Build a research question that issues exactly one combined search.

        ``retrieve_papers`` fans out one provider call per ``sub_query``. Passing
        the derived keywords as a *list* therefore turns a single claim into a
        burst of single-word queries (``all:video``, ``all:data``) that pull in
        unrelated recent papers. Collapsing the terms into one phrase keeps the
        hunt anchored to the claim's actual topic.
        """
        query = " ".join(terms).strip()
        if len(query) < 3:
            query = fallback
        return ResearchQuestion(question=query, sub_queries=[query])

    def _rank_candidates(
        self, candidates: list[ScoredPaper], question: str
    ) -> list[ScoredPaper]:
        """Order hunt candidates by relevance to the question (ties preserve order)."""
        if not candidates:
            return candidates
        return rank_papers(
            candidates,
            question=question,
            top_n=len(candidates),
            min_score=self.config.min_relevance_score,
        )

    def _resolve_reformulated_query(
        self, state: SynthesisState, original_query: str
    ) -> str:
        if self.hooks.reformulate is not None:
            return self.hooks.reformulate(state, original_query)
        return resolve_reformulated_query(state, original_query)

    def _execute(self, state: SynthesisState, step: DecisionStep) -> None:
        start = time.perf_counter()
        try:
            if step.action == "decompose":
                self._execute_decompose(state, step)
            elif step.action == "search":
                self._execute_search(state, step)
            elif step.action == "reformulate":
                self._execute_reformulate(state, step)
            elif step.action == "synthesize":
                self._execute_synthesize_placeholder(step)
            else:
                step.complete(
                    result="noop",
                    result_note=f"action {step.action!r} is not implemented in this controller layer",
                )
            if not step.is_pending and step.duration_ms is None:
                step.duration_ms = _elapsed_ms(start)
        except Exception as exc:  # noqa: BLE001 - preserve trace instead of crashing silently
            step.complete(
                result="failed",
                result_note=f"{type(exc).__name__}: {exc}",
                duration_ms=_elapsed_ms(start),
            )
            state.terminal_reason = "error"

    def _execute_decompose(self, state: SynthesisState, step: DecisionStep) -> None:
        params = step.params
        if params.kind != "decompose":
            raise TypeError("decompose step received non-decompose params")
        rq = self.hooks.decompose(params.question, n=params.n)
        state.sub_queries = list(rq.sub_queries)
        step.complete(
            result="ok" if state.sub_queries else "insufficient",
            result_note=f"generated {len(state.sub_queries)} sub-query(s)",
        )

    def _execute_search(self, state: SynthesisState, step: DecisionStep) -> None:
        params = step.params
        if params.kind != "search":
            raise TypeError("search step received non-search params")

        before = _paper_ids(state.papers)
        rq = ResearchQuestion(question=state.question, sub_queries=[params.query])
        raw = self._retrieve_fn()(
            rq,
            per_query_limit=self.config.per_query_limit,
            total_limit=self.config.total_paper_limit,
            sources=tuple(params.sources) or self.config.sources,
        )
        parsed = self._fetch_parse_fn()(raw)
        new_papers = [p for p in parsed if p.paper_id not in before]
        state.papers.extend(new_papers)

        if state.papers:
            state.papers = rank_papers(
                state.papers,
                question=state.question,
                top_n=self.config.total_paper_limit,
                min_score=self.config.min_relevance_score,
            )

        result = "ok" if len(state.papers) >= self.config.min_relevant_papers else "insufficient"
        step.complete(
            result=result,
            result_note=(
                f"retrieved {len(raw)} candidate(s), parsed {len(parsed)}, "
                f"added {len(new_papers)} new paper(s); working set now has {len(state.papers)}"
            ),
            effect=StepEffect(added_paper_ids=[p.paper_id for p in new_papers]),
        )

    def _execute_reformulate(self, state: SynthesisState, step: DecisionStep) -> None:
        params = step.params
        if params.kind != "reformulate":
            raise TypeError("reformulate step received non-reformulate params")

        new_query = self._resolve_reformulated_query(state, params.original_query)
        if new_query and new_query not in state.sub_queries:
            state.sub_queries.append(new_query)
            result = "ok"
            note = f"reformulated query to {new_query!r}"
        else:
            result = "noop"
            note = "reformulation produced no new query"
        step.complete(result=result, result_note=note)

    def _execute_synthesize_placeholder(self, step: DecisionStep) -> None:
        if step.result != "pending":
            return
        step.complete(
            result="ok",
            result_note="retrieval controller reached a terminal handoff point",
        )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def run_agentic_synthesis(
    question: str,
    *,
    config: ControllerConfig | None = None,
    hooks: ControllerHooks | None = None,
    database: Any | None = None,
    session_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> SynthesisResult:
    """
    Public entry point for the agentic LitSynth path.

    Runs the controller end-to-end, converts the final state into the legacy
    ``SynthesisResult`` artifact, and optionally persists the result using the
    same ``synthesis_runs`` table as the original linear pipeline.
    """
    if progress is not None:
        progress("running agentic controller")
    controller = SynthesisController(config=config, hooks=hooks)
    state = controller.run(question)
    result = state.to_result()
    if database is not None:
        from synthesis.pipeline import _persist

        _persist(database, session_id, result)
    if progress is not None:
        progress("agentic synthesis complete")
    return result
