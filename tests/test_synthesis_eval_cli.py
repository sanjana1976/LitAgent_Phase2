"""CLI tests for the LitSynth eval command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

import main as cli_module
from db.database import Database
from db.init_db import initialize_schema
from db.queries import insert_synthesis_run
from synthesis.schemas import CitationCheck, ClaimRecord, SynthesisResult


def _sample_result(question: str) -> SynthesisResult:
    return SynthesisResult(
        question=question,
        review_text="Grounded review [Smith 2024].",
        citations_used=["arxiv:1"],
        hallucinated_citations=[],
        contradictions_found=0,
        confidence_score=1.0,
        papers=[],
        claims=[
            ClaimRecord(
                paper_id="arxiv:1",
                claim="A grounded claim.",
                evidence_quote="A grounded claim.",
                grounded=True,
            )
        ],
        contradictions=[],
        citation_checks=[
            CitationCheck(
                citation_key="[Smith 2024]",
                resolved_paper_id="arxiv:1",
                is_valid=True,
            )
        ],
    )


def test_eval_synthesis_scores_persisted_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    question = "What are RAG tradeoffs?"
    cases = tmp_path / "cases.json"
    output = tmp_path / "results.json"
    db_path = tmp_path / "agent.sqlite"
    cases.write_text(
        json.dumps(
            [
                {
                    "question": question,
                    "expected_contradiction_keys": [],
                    "notes": "test case",
                }
            ]
        ),
        encoding="utf-8",
    )

    db = Database(db_path)
    initialize_schema(db)
    result = _sample_result(question)
    insert_synthesis_run(
        db,
        session_id="test",
        question=question,
        review_text=result.review_text,
        result_json=result.model_dump_json(),
        confidence_score=result.confidence_score,
        contradictions_found=result.contradictions_found,
        hallucinated_count=len(result.hallucinated_citations),
    )

    monkeypatch.setattr(
        cli_module,
        "get_settings",
        lambda: SimpleNamespace(
            project_root=tmp_path,
            database_path=db_path,
            openai_api_key="test-key",
        ),
    )

    runner = CliRunner()
    completed = runner.invoke(
        cli_module.cli,
        [
            "eval-synthesis",
            "--cases",
            str(cases),
            "--output",
            str(output),
        ],
    )

    assert completed.exit_code == 0, completed.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["aggregate"]["case_count"] == 1.0
    assert payload["aggregate"]["claim_faithfulness"] == 1.0
    assert payload["aggregate"]["citation_hallucination_rate"] == 0.0
    assert "[eval] wrote" in completed.output
