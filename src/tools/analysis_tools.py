"""
Deep paper analysis via OpenAI plus citation extraction from reference blocks.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from config.config import get_settings
from pydantic import ValidationError

from tools.paper_text import PaperTextError, load_cached_paper_text
from tools.schemas import (
    Citation,
    CodeAvailability,
    KeyEquation,
    MethodologyBlock,
    PaperAnalysis,
    ResultsBlock,
)
from tools.text_heuristics import (
    extract_equation_like_lines,
    extract_github_urls,
    extract_reference_block,
    guess_title_from_text,
)

logger = logging.getLogger(__name__)


class AnalysisToolError(RuntimeError):
    """Raised when analysis cannot be completed."""


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _resolve_full_text(paper_id: str, full_text_override: str | None) -> tuple[str, dict[str, str]]:
    try:
        return load_cached_paper_text(paper_id, full_text_override)
    except PaperTextError as exc:
        raise AnalysisToolError(str(exc)) from exc


def _heuristic_analysis(paper_id: str, text: str) -> PaperAnalysis:
    title = guess_title_from_text(text, fallback=paper_id)
    equations = [KeyEquation(**x) for x in extract_equation_like_lines(text)]
    repos = extract_github_urls(text)
    repro = 0.25
    if repos:
        repro += 0.35
    if "dataset" in text.lower():
        repro += 0.15
    repro = min(1.0, repro)
    return PaperAnalysis(
        paper_id=paper_id,
        title=title,
        key_contributions=[],
        methodology=MethodologyBlock(approach="(heuristic) See full text for details."),
        key_equations=equations,
        results=ResultsBlock(),
        limitations=[],
        future_work=[],
        related_work_categories={},
        reproducibility_score=repro,
        code_availability=CodeAvailability(has_code=bool(repos), repo_links=repos),
        created_at=datetime.now(timezone.utc),
    )


def _openai_structure(paper_id: str, text: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise AnalysisToolError("OpenAI API key not configured")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AnalysisToolError("openai package is not installed") from exc

    client = OpenAI(api_key=settings.openai_api_key)
    excerpt = text[:120_000]
    system = (
        "You extract structured research metadata from academic paper plain text. "
        "Reply with ONLY valid JSON matching the requested schema. "
        "Use empty arrays/objects when information is missing."
    )
    user = f"""Paper identifier: {paper_id}

Plain text excerpt:
{excerpt}

Return JSON with keys:
- title (string)
- key_contributions (array of short strings)
- methodology: {{ "approach", "datasets" (array of strings), "experimental_setup", "baselines_compared" (array) }}
- key_equations: [ {{ "equation": string, "description": string }} ]
- results: {{ "main_metrics": object string->number, "improvements": object string->string, "failure_cases": array }}
- limitations (array of strings)
- future_work (array of strings)
- related_work_categories: object mapping category name to array of references, each reference {{ "cited_title", "cited_authors" (array), "year", "venue", "raw" }}
- reproducibility_score: number 0-1
- code_availability: {{ "has_code": bool, "repo_links": array of strings }}
"""

    msg = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw_text = msg.choices[0].message.content or ""
    m = _JSON_BLOCK.search(raw_text)
    if not m:
        raise AnalysisToolError("OpenAI model response did not contain JSON")
    return json.loads(m.group(0))


def _merge_openai_and_heuristics(
    paper_id: str,
    text: str,
    openai_obj: dict[str, Any],
) -> PaperAnalysis:
    meth = openai_obj.get("methodology") or {}
    methodology = MethodologyBlock(
        approach=str(meth.get("approach", "") or ""),
        datasets=[str(x) for x in (meth.get("datasets") or [])],
        experimental_setup=str(meth.get("experimental_setup", "") or ""),
        baselines_compared=[str(x) for x in (meth.get("baselines_compared") or [])],
    )
    res = openai_obj.get("results") or {}
    main_metrics: dict[str, float] = {}
    for k, v in (res.get("main_metrics") or {}).items():
        try:
            main_metrics[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    results = ResultsBlock(
        main_metrics=main_metrics,
        improvements={str(k): str(v) for k, v in (res.get("improvements") or {}).items()},
        failure_cases=[str(x) for x in (res.get("failure_cases") or [])],
    )
    keq: list[KeyEquation] = []
    for item in openai_obj.get("key_equations") or []:
        if isinstance(item, dict):
            keq.append(
                KeyEquation(
                    equation=str(item.get("equation", "")),
                    description=str(item.get("description", "") or ""),
                )
            )
    if not keq:
        keq = [KeyEquation(**x) for x in extract_equation_like_lines(text)]

    rw: dict[str, list[Citation]] = {}
    raw_rw = openai_obj.get("related_work_categories") or {}
    if isinstance(raw_rw, dict):
        for cat, refs in raw_rw.items():
            if not isinstance(refs, list):
                continue
            cits: list[Citation] = []
            for r in refs:
                if not isinstance(r, dict):
                    continue
                y = r.get("year")
                cits.append(
                    Citation(
                        cited_title=str(r.get("cited_title", "Unknown")),
                        cited_authors=[str(a) for a in (r.get("cited_authors") or [])],
                        year=int(y) if y is not None and str(y).isdigit() else None,
                        venue=str(r["venue"]) if r.get("venue") else None,
                        raw=str(r["raw"]) if r.get("raw") else None,
                    )
                )
            rw[str(cat)] = cits

    ca = openai_obj.get("code_availability") or {}
    repos = [str(x) for x in (ca.get("repo_links") or [])]
    extra = extract_github_urls(text)
    for u in extra:
        if u not in repos:
            repos.append(u)

    score = float(openai_obj.get("reproducibility_score") or 0.0)
    try:
        return PaperAnalysis(
            paper_id=paper_id,
            title=str(openai_obj.get("title") or guess_title_from_text(text)),
            key_contributions=[str(x) for x in (openai_obj.get("key_contributions") or [])],
            methodology=methodology,
            key_equations=keq,
            results=results,
            limitations=[str(x) for x in (openai_obj.get("limitations") or [])],
            future_work=[str(x) for x in (openai_obj.get("future_work") or [])],
            related_work_categories=rw,
            reproducibility_score=score,
            code_availability=CodeAvailability(
                has_code=bool(ca.get("has_code")) or bool(repos),
                repo_links=repos,
            ),
            created_at=datetime.now(timezone.utc),
        )
    except ValidationError as exc:
        raise AnalysisToolError(f"Structured merge validation failed: {exc}") from exc


def tool_deep_analyze_paper(paper_id: str, *, full_text: str | None = None) -> PaperAnalysis:
    """
    Run the deep-analysis engine: OpenAI structures content; heuristics fill gaps.

    Requires prior ``tool_fetch_and_parse_pdf`` unless ``full_text`` is passed explicitly
    (mainly for tests).
    """
    if not paper_id.strip():
        raise AnalysisToolError("paper_id must be non-empty")

    text, _sec = _resolve_full_text(paper_id, full_text)

    settings = get_settings()
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY unset; using heuristic analysis only")
        return _heuristic_analysis(paper_id, text)

    try:
        openai_obj = _openai_structure(paper_id, text)
        return _merge_openai_and_heuristics(paper_id, text, openai_obj)
    except AnalysisToolError as exc:
        logger.warning("OpenAI analysis unavailable (%s); using heuristics", exc)
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        logger.warning("OpenAI output unusable (%s); using heuristics", exc)

    return _heuristic_analysis(paper_id, text)


def tool_extract_citations(paper_id: str, *, full_text: str | None = None) -> list[Citation]:
    """
    Parse bibliography-style lines from the references section of extracted text.

    Uses :func:`text_heuristics.extract_reference_block` plus lightweight line splitting.
    """
    text, _ = _resolve_full_text(paper_id, full_text)
    ref_block = extract_reference_block(text) or text
    citations: list[Citation] = []
    for line in ref_block.splitlines():
        s = line.strip()
        if len(s) < 12:
            continue
        year_m = re.search(r"\b(19|20)\d{2}\b", s)
        year = int(year_m.group(0)) if year_m else None
        parts = re.split(r"\.(?=\s[A-Z])", s, maxsplit=1)
        title = parts[0][:400]
        rest = parts[1] if len(parts) > 1 else ""
        venue = None
        if rest:
            venue_m = re.search(r"(In |IEEE |ACM |Proceedings|Journal)[^\n]+", rest)
            venue = venue_m.group(0).strip() if venue_m else None
        authors_guess: list[str] = []
        if "," in s[:120]:
            authors_guess = [a.strip() for a in s.split(",")[:3]]
        citations.append(
            Citation(
                cited_title=title if len(title) > 5 else s[:300],
                cited_authors=authors_guess,
                year=year,
                venue=venue,
                raw=s,
            )
        )
        if len(citations) >= 500:
            break
    return citations
