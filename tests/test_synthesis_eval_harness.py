"""Tests for the LitSynth eval harness (claim faithfulness, hallucination, coverage)."""

from __future__ import annotations

import json
from pathlib import Path

from synthesis.eval_harness import (
    EvalCase,
    EvalReport,
    aggregate,
    load_cases,
    resolve_case_result,
    score_synthesis,
    write_results,
)
from synthesis.schemas import (
    CitationCheck,
    ClaimRecord,
    ContradictionPair,
    ScoredPaper,
    SynthesisResult,
)


def _result(
    *,
    claims: list[ClaimRecord] | None = None,
    citation_checks: list[CitationCheck] | None = None,
    hallucinated: list[str] | None = None,
    contradictions: list[ContradictionPair] | None = None,
) -> SynthesisResult:
    return SynthesisResult(
        question="q",
        review_text="...",
        citations_used=[],
        hallucinated_citations=list(hallucinated or []),
        contradictions_found=len(contradictions or []),
        confidence_score=0.0,
        papers=[],
        claims=list(claims or []),
        contradictions=list(contradictions or []),
        citation_checks=list(citation_checks or []),
    )


def test_score_all_grounded_and_resolved() -> None:
    claims = [
        ClaimRecord(paper_id="arxiv:1", claim="first claim", evidence_quote="first quote", confidence=0.9, grounded=True),
        ClaimRecord(paper_id="arxiv:2", claim="second claim", evidence_quote="second quote", confidence=0.6, grounded=True),
    ]
    checks = [
        CitationCheck(citation_key="[Smith 2023]", resolved_paper_id="arxiv:1", is_valid=True),
        CitationCheck(citation_key="[Doe 2024]", resolved_paper_id="arxiv:2", is_valid=True),
    ]
    res = _result(claims=claims, citation_checks=checks)
    case = EvalCase(question="q", expected_contradiction_keys=[])
    rep = score_synthesis(res, case)
    assert rep.claim_faithfulness == 1.0
    assert rep.citation_hallucination_rate == 0.0
    assert rep.contradiction_coverage == 0.0


def test_score_partial_grounding_and_hallucination() -> None:
    claims = [
        ClaimRecord(paper_id="arxiv:1", claim="first claim", evidence_quote="first quote", confidence=0.9, grounded=True),
        ClaimRecord(paper_id="arxiv:2", claim="second claim", evidence_quote="second quote", confidence=0.6, grounded=False),
    ]
    checks = [
        CitationCheck(citation_key="[Smith 2023]", resolved_paper_id="arxiv:1", is_valid=True),
        CitationCheck(citation_key="[Ghost 2099]", resolved_paper_id=None, is_valid=False),
    ]
    res = _result(claims=claims, citation_checks=checks, hallucinated=["[Ghost 2099]"])
    rep = score_synthesis(res, EvalCase(question="q", expected_contradiction_keys=[]))
    assert rep.claim_faithfulness == 0.5
    assert rep.citation_hallucination_rate == 0.5


def test_contradiction_coverage_canonical_keys() -> None:
    contradictions = [
        ContradictionPair(
            paper_a="arxiv:2",
            paper_b="arxiv:1",  # order reversed; harness must canonicalize
            claim_a="some claim text",
            claim_b="other claim text",
            tension_type="contradiction",
            explanation="",
        )
    ]
    res = _result(contradictions=contradictions)
    case = EvalCase(
        question="q",
        expected_contradiction_keys=["arxiv:1|arxiv:2", "arxiv:1|arxiv:3"],
    )
    rep = score_synthesis(res, case)
    assert rep.contradiction_coverage == 0.5
    assert rep.expected_contradictions == 2
    assert rep.found_contradictions == 1


def test_aggregate_macro_average() -> None:
    a = EvalReport(
        question="q1",
        claim_faithfulness=1.0,
        citation_hallucination_rate=0.0,
        contradiction_coverage=1.0,
        total_claims=2,
        grounded_claims=2,
        total_citations=2,
        hallucinated_citations=0,
        expected_contradictions=1,
        found_contradictions=1,
    )
    b = EvalReport(
        question="q2",
        claim_faithfulness=0.0,
        citation_hallucination_rate=1.0,
        contradiction_coverage=0.0,
        total_claims=1,
        grounded_claims=0,
        total_citations=1,
        hallucinated_citations=1,
        expected_contradictions=1,
        found_contradictions=0,
    )
    agg = aggregate([a, b])
    assert agg["claim_faithfulness"] == 0.5
    assert agg["citation_hallucination_rate"] == 0.5
    assert agg["contradiction_coverage"] == 0.5
    assert agg["case_count"] == 2.0


def test_aggregate_empty() -> None:
    agg = aggregate([])
    assert agg == {
        "claim_faithfulness": 0.0,
        "citation_hallucination_rate": 0.0,
        "contradiction_coverage": 0.0,
        "case_count": 0.0,
    }


def test_write_results_round_trip(tmp_path: Path) -> None:
    rep = EvalReport(
        question="q",
        claim_faithfulness=1.0,
        citation_hallucination_rate=0.0,
        contradiction_coverage=1.0,
        total_claims=1,
        grounded_claims=1,
        total_citations=1,
        hallucinated_citations=0,
        expected_contradictions=1,
        found_contradictions=1,
    )
    out = tmp_path / "results.json"
    written = write_results([rep], output_path=out)
    assert written == out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["per_case"]) == 1
    assert payload["aggregate"]["case_count"] == 1.0


def test_load_cases_from_json(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "question": "What are RAG tradeoffs?",
                    "expected_contradiction_keys": ["arxiv:1|arxiv:2"],
                    "fixture": "rag.json",
                    "notes": "demo",
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_cases(cases_path)

    assert cases == [
        EvalCase(
            question="What are RAG tradeoffs?",
            expected_contradiction_keys=["arxiv:1|arxiv:2"],
            notes="demo",
            fixture="rag.json",
        )
    ]


def test_resolve_case_result_prefers_fixture(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    result = _result(
        claims=[
                ClaimRecord(
                    paper_id="arxiv:1",
                    claim="demo claim",
                    evidence_quote="demo quote",
                    confidence=0.9,
                    grounded=True,
                )
        ]
    )
    (fixtures_dir / "demo.json").write_text(
        json.dumps(result.model_dump(mode="json")),
        encoding="utf-8",
    )
    case = EvalCase(
        question="q",
        expected_contradiction_keys=[],
        fixture="demo.json",
    )
    resolved = resolve_case_result(case, fixtures_dir=fixtures_dir, db_raw=None)
    assert resolved is not None
    assert len(resolved.claims) == 1
