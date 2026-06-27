"""
Stage 8 of the LitSynth pipeline: literature-review prose generator.

Given a fully-formed :class:`~synthesis.prompt.SynthesisPrompt`, this stage
asks the LLM for free-form prose (no JSON) at a low-but-not-zero
temperature. The function is intentionally crash-proof: any failure in the
LLM wrapper is converted into a short deterministic fallback string that
still lists the sources the model was meant to review, so the rest of the
pipeline (citation validation, contradiction counting, persistence) can
continue without special-casing this stage.

The LLM adapter is injected via the ``llm_call`` parameter so unit tests
can exercise both the happy path and the error path without monkey
patching the network layer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from synthesis.llm import SynthesisLLMError, call_text
from synthesis.prompt import SynthesisPrompt

logger = logging.getLogger(__name__)


_DEFAULT_TEMPERATURE = 0.3
_MIN_MAX_TOKENS = 800
_MAX_MAX_TOKENS = 4_000
_TOKEN_HEADROOM = 256
_TOKENS_PER_WORD = 4


def _compute_max_tokens(word_budget: int) -> int:
    """Translate a word budget into a clamped ``max_tokens`` request."""
    raw = int(word_budget) * _TOKENS_PER_WORD + _TOKEN_HEADROOM
    return max(_MIN_MAX_TOKENS, min(_MAX_MAX_TOKENS, raw))


def _fallback_text(prompt: SynthesisPrompt, reason: str) -> str:
    """Render the deterministic safe-fallback string when the LLM fails."""
    citations = ", ".join(prompt.expected_citations) if prompt.expected_citations else "none"
    safe_reason = (reason or "unknown error").strip().replace("\n", " ")
    return (
        f"Literature review generation failed: {safe_reason}. "
        f"Sources reviewed: {citations}."
    )


def generate_literature_review(
    prompt: SynthesisPrompt,
    *,
    word_budget: int = 500,
    llm_call: Callable[..., str] | None = None,
) -> str:
    """
    Generate the literature-review prose for ``prompt`` via the LLM.

    Args:
        prompt: The deterministic prompt produced by
            :func:`synthesis.prompt.build_synthesis_prompt`.
        word_budget: Approximate target length; used only to size the
            ``max_tokens`` request to the LLM (clamped to a sane range).
        llm_call: Optional injectable callable with the same keyword
            signature as :func:`synthesis.llm.call_text`. Defaults to the
            real OpenAI-backed implementation.

    Returns:
        The generated prose with leading and trailing whitespace stripped.
        If the LLM raises :class:`~synthesis.llm.SynthesisLLMError` (or any
        other unexpected exception), a deterministic fallback string of the
        form ``"Literature review generation failed: <reason>. Sources
        reviewed: <comma-joined expected_citations>."`` is returned instead.

    Notes:
        This function NEVER raises. A failure to call the LLM is converted
        into the fallback text so the downstream citation validator and
        pipeline orchestrator can keep running.
    """
    invoker = llm_call if llm_call is not None else call_text
    max_tokens = _compute_max_tokens(word_budget)

    try:
        raw = invoker(
            system=prompt.system,
            user=prompt.user,
            temperature=_DEFAULT_TEMPERATURE,
            max_tokens=max_tokens,
        )
    except SynthesisLLMError as exc:
        logger.warning("Literature review LLM call failed: %s", exc)
        return _fallback_text(prompt, str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Literature review LLM call raised %s: %s",
            type(exc).__name__,
            exc,
        )
        return _fallback_text(prompt, f"{type(exc).__name__}: {exc}")

    if not isinstance(raw, str):
        logger.warning("LLM returned non-string payload of type %s", type(raw).__name__)
        return _fallback_text(prompt, "LLM returned a non-string payload")

    return raw.strip()
