"""
Multi-paper comparison matrix (methodology, datasets, metrics, cost, novelty).
"""

from __future__ import annotations

import json
import logging
import re

from config.config import get_settings
from pydantic import BaseModel, Field

from tools.paper_text import PaperTextError, load_cached_paper_text
from tools.schemas import ComparisonMatrix

logger = logging.getLogger(__name__)


class _CompareJSON(BaseModel):
    """Intermediate schema for LLM JSON."""

    methodology: dict[str, str] = Field(default_factory=dict)
    datasets_used: dict[str, str] = Field(default_factory=dict)
    results_metrics: dict[str, str] = Field(default_factory=dict)
    computational_cost: dict[str, str] = Field(default_factory=dict)
    novelty: dict[str, str] = Field(default_factory=dict)


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _fallback_compare(paper_ids: list[str]) -> ComparisonMatrix:
    out = ComparisonMatrix(paper_ids=paper_ids)
    for pid in paper_ids:
        text, _ = load_cached_paper_text(pid, None)
        snippet = " ".join(text.split())[:400]
        out.methodology[pid] = snippet or "(no text)"
        low = text.lower()
        ds_hits = []
        for key in ("mnist", "imagenet", "coco", "wikitext", "glue", "pubmed"):
            if key in low:
                ds_hits.append(key)
        out.datasets_used[pid] = ", ".join(ds_hits) if ds_hits else "(keywords not detected)"
        out.results_metrics[pid] = "(extract metrics manually — heuristic mode)"
        out.computational_cost[pid] = (
            "GPU mentioned" if "gpu" in low or "cuda" in low else "Not inferred from text"
        )
        out.novelty[pid] = "See introduction / related work sections in cached PDF text."
    return out


def _openai_compare(paper_ids: list[str]) -> ComparisonMatrix:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("no openai key")
    from openai import OpenAI

    parts: list[str] = []
    for pid in paper_ids:
        text, _ = load_cached_paper_text(pid, None)
        parts.append(f"### {pid}\n{text[:24_000]}")
    bundle = "\n\n".join(parts)

    client = OpenAI(api_key=settings.openai_api_key)
    system = (
        "You compare academic papers. Reply ONLY with JSON containing five objects "
        "each keyed by paper id string with short comparative prose values: "
        "methodology, datasets_used, results_metrics, computational_cost, novelty."
    )
    user = f"Papers:\n{bundle}\n\nReturn JSON matching schema with keys exactly as above."
    msg = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = msg.choices[0].message.content or ""
    m = _JSON_BLOCK.search(raw)
    if not m:
        raise ValueError("no json")
    data = json.loads(m.group(0))
    parsed = _CompareJSON.model_validate(data)
    return ComparisonMatrix(
        paper_ids=paper_ids,
        methodology=parsed.methodology,
        datasets_used=parsed.datasets_used,
        results_metrics=parsed.results_metrics,
        computational_cost=parsed.computational_cost,
        novelty=parsed.novelty,
    )


def tool_compare_papers(paper_ids: list[str]) -> ComparisonMatrix:
    """
    Compare 2–5 papers across methodology, datasets, headline metrics, compute, and novelty.

    Uses OpenAI when ``OPENAI_API_KEY`` is set; otherwise a lightweight keyword fallback.

    Raises:
        ValueError: invalid paper id count or missing cached PDF text for any id.
    """
    ids = [p.strip() for p in paper_ids if p.strip()]
    if not 2 <= len(ids) <= 5:
        raise ValueError("tool_compare_papers requires between 2 and 5 paper ids")

    # Ensure cache hits up-front for clearer errors
    for pid in ids:
        try:
            load_cached_paper_text(pid, None)
        except PaperTextError as exc:
            raise ValueError(str(exc)) from exc

    settings = get_settings()
    if settings.openai_api_key:
        try:
            return _openai_compare(ids)
        except Exception as exc:
            logger.warning("OpenAI compare failed (%s); using fallback", exc)

    return _fallback_compare(ids)
