"""
LitSynth (A4) eval harness.

Stage 10 of the pipeline. Computes three measurable scores against a hand-
labeled ground-truth file so the synthesis quality can be tracked over time:

- **claim faithfulness** — fraction of generated claims whose ``grounded`` flag
  is True (the claim extractor's verbatim-quote check passed).
- **citation hallucination rate** — fraction of inline citations the validator
  flagged as not resolving to any retrieved paper.
- **contradiction coverage** — fraction of the expected contradictions
  (identified by ``contradiction_key``s) that the agent surfaced.

The harness is intentionally self-contained: it takes already-computed
``SynthesisResult`` objects so the CI/eval path never depends on the network.
Live runs are produced by the regular pipeline; the harness only scores them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from synthesis.schemas import SynthesisResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalCase:
    """One labeled research question with reference answers."""

    question: str
    expected_contradiction_keys: list[str]
    notes: str = ""


@dataclass(frozen=True)
class EvalReport:
    """Per-case scores produced by :func:`score_synthesis`."""

    question: str
    claim_faithfulness: float
    citation_hallucination_rate: float
    contradiction_coverage: float
    total_claims: int
    grounded_claims: int
    total_citations: int
    hallucinated_citations: int
    expected_contradictions: int
    found_contradictions: int


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def load_cases(path: Path) -> list[EvalCase]:
    """
    Load labeled eval cases from JSON.

    Expected shape:

    ``[{"question": "...", "expected_contradiction_keys": ["a|b"], "notes": "..."}]``
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("eval cases file must contain a JSON list")

    cases: list[EvalCase] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"eval case {index} must be an object")
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"eval case {index} is missing question")
        keys = item.get("expected_contradiction_keys", [])
        if not isinstance(keys, list):
            raise ValueError(
                f"eval case {index} expected_contradiction_keys must be a list"
            )
        cases.append(
            EvalCase(
                question=question,
                expected_contradiction_keys=[str(k).strip() for k in keys if str(k).strip()],
                notes=str(item.get("notes", "") or ""),
            )
        )
    return cases


def _contradiction_keys(result: SynthesisResult) -> set[str]:
    """Build canonical ``"paper_a|paper_b"`` keys with sorted endpoints."""
    out: set[str] = set()
    for pair in result.contradictions:
        a, b = sorted((pair.paper_a, pair.paper_b))
        out.add(f"{a}|{b}")
    return out


def score_synthesis(result: SynthesisResult, case: EvalCase) -> EvalReport:
    """Score one ``SynthesisResult`` against one labeled case."""
    total_claims = len(result.claims)
    grounded = sum(1 for c in result.claims if c.grounded)

    total_citations = len(result.citation_checks)
    halluc = len(result.hallucinated_citations)

    expected_keys = {k.strip() for k in case.expected_contradiction_keys if k.strip()}
    found = _contradiction_keys(result)
    overlap = len(expected_keys & found)

    return EvalReport(
        question=case.question,
        claim_faithfulness=_safe_div(grounded, total_claims),
        citation_hallucination_rate=_safe_div(halluc, total_citations),
        contradiction_coverage=_safe_div(overlap, len(expected_keys)),
        total_claims=total_claims,
        grounded_claims=grounded,
        total_citations=total_citations,
        hallucinated_citations=halluc,
        expected_contradictions=len(expected_keys),
        found_contradictions=len(found),
    )


def aggregate(reports: Iterable[EvalReport]) -> dict[str, float]:
    """Return macro-averaged metrics across a sequence of per-case reports."""
    items = list(reports)
    if not items:
        return {
            "claim_faithfulness": 0.0,
            "citation_hallucination_rate": 0.0,
            "contradiction_coverage": 0.0,
            "case_count": 0.0,
        }
    n = len(items)
    return {
        "claim_faithfulness": sum(r.claim_faithfulness for r in items) / n,
        "citation_hallucination_rate": sum(r.citation_hallucination_rate for r in items) / n,
        "contradiction_coverage": sum(r.contradiction_coverage for r in items) / n,
        "case_count": float(n),
    }


def write_results(
    reports: Iterable[EvalReport],
    *,
    output_path: Path,
) -> Path:
    """
    Write a JSON dump of per-case reports plus an aggregate block.

    The output schema is stable so external dashboards can consume it.
    """
    items = [asdict(r) for r in reports]
    payload = {
        "per_case": items,
        "aggregate": aggregate(EvalReport(**r) for r in items),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
