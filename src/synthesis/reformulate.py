"""
LLM-backed query reformulation for the agentic retrieval loop.

When retrieval returns too few on-topic papers, this module asks the model to
rewrite the current sub-query for scholarly search. A deterministic fallback
keeps the controller testable without network access.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from synthesis.llm import SynthesisLLMError, call_json
from synthesis.state import SynthesisState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research librarian improving a scholarly search query. "
    "The previous query returned too few relevant papers. Respond with a single "
    "JSON object: {\"new_query\": \"...\"}. Keep the rewritten query under 18 "
    "words, stay focused on the original research question, use concrete "
    "technical terms scholars would use in paper titles/abstracts, and do not "
    "append generic suffixes like 'methods benchmarks limitations'."
)


def default_reformulate(state: SynthesisState) -> str:
    """
    Deterministic reformulation fallback used when the LLM is unavailable.

    Keeps the controller testable and preserves the legacy trace shape.
    """
    base = state.sub_queries[-1] if state.sub_queries else state.question
    attempt = state.reformulation_count + 1
    suffixes = [
        "methods benchmarks limitations",
        "recent survey empirical comparison",
        "failure modes evaluation datasets",
    ]
    suffix = suffixes[min(attempt - 1, len(suffixes) - 1)]
    return f"{base} {suffix}".strip()


def _default_llm_call(
    *, system: str, user: str, temperature: float, max_tokens: int
) -> dict[str, Any]:
    return call_json(
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def llm_reformulate_query(
    state: SynthesisState,
    original_query: str,
    *,
    llm_call: Callable[..., dict[str, Any]] | None = None,
) -> str | None:
    """
    Ask the LLM for a tighter search query anchored to ``state.question``.

    Returns ``None`` when the LLM fails or returns an unusable payload.
    """
    invoker = llm_call if llm_call is not None else _default_llm_call
    user_prompt = (
        f"Research question: {state.question.strip()}\n"
        f"Current search query: {original_query.strip()}\n"
        f"Reformulation attempt: {state.reformulation_count + 1}\n\n"
        'Return JSON: {"new_query": "..."}'
    )
    try:
        payload = invoker(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.2,
            max_tokens=256,
        )
    except SynthesisLLMError as exc:
        logger.warning("Reformulate LLM call failed; using fallback: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Reformulate LLM call raised %s; using fallback", type(exc).__name__
        )
        return None

    if not isinstance(payload, dict):
        return None
    value = payload.get("new_query")
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if len(cleaned) < 3:
        return None
    if cleaned.casefold() == original_query.strip().casefold():
        return None
    return cleaned


def resolve_reformulated_query(
    state: SynthesisState,
    original_query: str,
    *,
    llm_call: Callable[..., dict[str, Any]] | None = None,
) -> str:
    """Prefer an LLM rewrite; fall back to the deterministic suffix strategy."""
    candidate = llm_reformulate_query(
        state, original_query, llm_call=llm_call
    )
    if candidate:
        return candidate
    return default_reformulate(state)
