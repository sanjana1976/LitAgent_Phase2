"""
Generate committed offline eval fixtures for ``eval/cases.json``.

Run once after changing cases:

    python eval/seed_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
for path in (SRC, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from synthesis.schemas import (  # noqa: E402
    CitationCheck,
    ClaimRecord,
    ContradictionPair,
    ScoredPaper,
    SynthesisResult,
)

FIXTURES_DIR = PROJECT_ROOT / "eval" / "fixtures"

# (fixture filename, grounded of 8, hallucinated of 5, contradiction pair or None)
SPECS: list[tuple[str, int, int, tuple[str, str] | None]] = [
    ("long-context-retrieval.json", 7, 1, None),
    ("rag-vs-finetuning.json", 7, 0, ("arxiv:rag1", "arxiv:ft1")),
    ("instruction-retrieval.json", 8, 0, None),
    ("hallucination-reduction.json", 6, 2, None),
    ("chain-of-thought-reasoning.json", 7, 1, None),
    ("multi-document-summarization.json", 8, 0, ("arxiv:sum1", "arxiv:sum2")),
    ("sparse-vs-dense-retrieval.json", 7, 1, None),
    ("citation-accuracy-assistants.json", 6, 1, None),
    ("long-document-transformers.json", 8, 0, None),
    ("climate-impact-ml.json", 7, 1, ("arxiv:cli1", "arxiv:cli2")),
    ("deforestation-remote-sensing.json", 7, 0, None),
    ("carbon-sequestration-sensing.json", 6, 2, None),
    ("weather-ensemble-deep-learning.json", 8, 0, None),
    ("symbolic-vs-neural-knowledge-graphs.json", 7, 1, None),
    ("agent-tool-selection-errors.json", 7, 1, None),
    ("faithfulness-summarization-metrics.json", 8, 0, None),
    ("open-vs-proprietary-retrieval.json", 6, 1, None),
    ("contradictory-claims-detection.json", 7, 0, ("arxiv:con1", "arxiv:con2")),
    ("pdf-parsing-claim-extraction.json", 7, 1, None),
    ("structured-json-prompting.json", 8, 0, None),
    ("multi-api-rate-limits.json", 7, 1, None),
    ("agent-filesystem-guardrails.json", 8, 0, None),
]


def _paper(paper_id: str, title: str) -> ScoredPaper:
    return ScoredPaper(
        paper_id=paper_id,
        title=title,
        authors=["Smith, A."],
        year=2024,
        venue="Demo",
        url=f"https://example.org/{paper_id}",
        relevance_score=0.8,
        has_pdf=True,
    )


def _build(
    question: str,
    *,
    grounded: int,
    hallucinated: int,
    contradiction: tuple[str, str] | None,
) -> SynthesisResult:
    total_claims = 8
    claims: list[ClaimRecord] = []
    for i in range(total_claims):
        claims.append(
            ClaimRecord(
                paper_id=f"arxiv:demo{i % 3 + 1}",
                claim=f"Claim {i + 1} about the literature.",
                evidence_quote=f"Evidence sentence {i + 1} from the source text.",
                confidence=0.8 if i < grounded else 0.4,
                grounded=i < grounded,
            )
        )

    total_citations = 5
    checks: list[CitationCheck] = []
    halluc_keys: list[str] = []
    for i in range(total_citations):
        is_halluc = i < hallucinated
        key = f"[Ghost {2099 + i}]" if is_halluc else f"[Author{i} 2024]"
        if is_halluc:
            halluc_keys.append(key)
        checks.append(
            CitationCheck(
                citation_key=key,
                resolved_paper_id=None if is_halluc else f"arxiv:demo{i % 3 + 1}",
                is_valid=not is_halluc,
            )
        )

    contradictions: list[ContradictionPair] = []
    papers = [
        _paper("arxiv:demo1", "Demo paper one"),
        _paper("arxiv:demo2", "Demo paper two"),
        _paper("arxiv:demo3", "Demo paper three"),
    ]
    if contradiction is not None:
        a, b = contradiction
        papers.extend(
            [
                _paper(a, f"Paper {a}"),
                _paper(b, f"Paper {b}"),
            ]
        )
        contradictions.append(
            ContradictionPair(
                paper_a=a,
                paper_b=b,
                claim_a="Method A reports higher accuracy.",
                claim_b="Method B reports higher accuracy.",
                tension_type="contradiction",
                explanation="Competing performance claims.",
            )
        )

    citations_used = [c.resolved_paper_id for c in checks if c.is_valid and c.resolved_paper_id]

    return SynthesisResult(
        question=question,
        review_text=f"Synthesized review for: {question}",
        citations_used=list(dict.fromkeys(citations_used)),
        hallucinated_citations=halluc_keys,
        contradictions_found=len(contradictions),
        confidence_score=round((grounded / total_claims) * ((total_citations - hallucinated) / total_citations), 4),
        papers=papers,
        claims=claims,
        contradictions=contradictions,
        citation_checks=checks,
    )


def main() -> None:
    cases_path = PROJECT_ROOT / "eval" / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if len(cases) != len(SPECS):
        raise SystemExit(f"cases.json has {len(cases)} entries; seed script expects {len(SPECS)}")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for case, (filename, grounded, hallucinated, contradiction) in zip(cases, SPECS, strict=True):
        if case.get("fixture") != filename:
            raise SystemExit(f"Fixture mismatch for {case['question']!r}: {case.get('fixture')} != {filename}")
        result = _build(
            case["question"],
            grounded=grounded,
            hallucinated=hallucinated,
            contradiction=contradiction,
        )
        path = FIXTURES_DIR / filename
        path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
        print(f"wrote {path.name}")

    print(f"Generated {len(SPECS)} fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
