"""
Stage 1 of the LitSynth pipeline: research-question decomposition.

Given a single user research question, this module asks the LLM to split it into
3-5 narrower sub-queries that cover *different angles* (e.g. methods,
benchmarks, criticisms, recent work). Those sub-queries become the inputs to
the retriever stage so we can cast a wider net than a single bag-of-words call
would allow.

Design notes:
- The LLM is called through :mod:`synthesis.llm` and is required to return a
  strict ``{"sub_queries": [str]}`` JSON object.
- The LLM dependency is *injected* so unit tests can stub it without touching
  the network or the OpenAI SDK.
- A deterministic templated fallback is used whenever the LLM is unavailable
  or returns garbage; the fallback never raises so this stage cannot break the
  rest of the pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from synthesis.llm import SynthesisLLMError, call_json
from synthesis.schemas import ResearchQuestion

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a research librarian helping decompose a literature-review "
    "question into focused sub-queries for scholarly search. Respond with a "
    "single JSON object of the form {\"sub_queries\": [\"...\", \"...\"]}. "
    "Each sub-query should target a different angle of the question, such as "
    "core methods, benchmarks/datasets, criticisms or limitations, and recent "
    "work. Keep each sub-query under 18 words, do not number them, and never "
    "include explanations outside the JSON object."
)


def _default_llm_call(
    *, system: str, user: str, temperature: float, max_tokens: int
) -> dict[str, Any]:
    """Default LLM adapter (wraps :func:`synthesis.llm.call_json`)."""
    return call_json(
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _fallback_sub_queries(question: str, n: int) -> list[str]:
    """Deterministic templated decomposition used when the LLM is unusable."""
    base = question.strip() or "research question"
    candidates = [
        base,
        f"{base} methods",
        f"{base} benchmarks",
        f"{base} limitations",
        f"{base} recent work",
    ]
    return _clean_and_clamp(candidates, n=n, min_count=1)


def _clean_and_clamp(
    raw: list[Any], *, n: int, min_count: int = 2
) -> list[str]:
    """Strip, drop empties, case-insensitively dedupe, then clamp to ``n``."""
    upper_bound = max(n, min_count)
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
        if len(cleaned) >= upper_bound:
            break
    return cleaned


def _extract_sub_queries(payload: dict[str, Any]) -> list[str]:
    """Pull a list-of-strings out of the LLM JSON payload, defensively."""
    if not isinstance(payload, dict):
        return []
    value = payload.get("sub_queries")
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def decompose_question(
    question: str,
    *,
    n: int = 4,
    llm_call: Callable[..., dict[str, Any]] | None = None,
) -> ResearchQuestion:
    """
    Decompose ``question`` into 2-``n`` focused sub-queries.

    Args:
        question: The raw user research question.
        n: Upper bound on the number of returned sub-queries (default 4).
            The returned count is clamped to the inclusive range ``[2, n]``
            whenever enough distinct candidates exist.
        llm_call: Optional injectable LLM adapter with the same keyword
            signature as :func:`synthesis.llm.call_json`. Defaults to the
            real OpenAI-backed implementation.

    Returns:
        A :class:`~synthesis.schemas.ResearchQuestion` carrying the original
        question plus the cleaned, deduplicated sub-query list.

    Notes:
        Falls back to a deterministic templated decomposition if the LLM
        raises :class:`~synthesis.llm.SynthesisLLMError` or returns fewer than
        two usable sub-queries. The fallback never raises.
    """
    invoker = llm_call if llm_call is not None else _default_llm_call

    target_n = max(2, int(n))
    user_prompt = (
        f"Research question: {question.strip()}\n\n"
        f"Produce between 3 and {target_n} sub-queries as JSON: "
        "{\"sub_queries\": [\"...\"]}."
    )

    sub_queries: list[str] = []
    try:
        payload = invoker(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.1,
            max_tokens=512,
        )
        sub_queries = _clean_and_clamp(
            _extract_sub_queries(payload), n=target_n
        )
    except SynthesisLLMError as exc:
        logger.warning("Decompose LLM call failed; using fallback: %s", exc)
        sub_queries = []
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Decompose LLM call raised %s; using fallback", type(exc).__name__
        )
        sub_queries = []

    if len(sub_queries) < 2:
        logger.info(
            "Decompose produced %d usable sub-queries; using fallback split.",
            len(sub_queries),
        )
        sub_queries = _fallback_sub_queries(question, n=target_n)

    return ResearchQuestion(question=question.strip(), sub_queries=sub_queries)
