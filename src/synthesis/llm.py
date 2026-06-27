"""
Thin wrapper around the OpenAI Chat Completions API for the synthesis layer.

Why a separate wrapper instead of using LangChain's ``ChatOpenAI`` everywhere?
- Synthesis stages want raw JSON-only responses (``response_format={"type": "json_object"}``),
  retried once with stricter wording on parse failure.
- They never need tool-binding; the agent loop in :mod:`agent.agent` handles that.
- Keeping this isolated makes it trivial to monkeypatch in unit tests.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from config.config import get_settings

logger = logging.getLogger(__name__)


class SynthesisLLMError(RuntimeError):
    """Raised when the LLM cannot return parseable JSON for a synthesis stage."""


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _coerce_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise SynthesisLLMError("Empty LLM response.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(raw)
        if not match:
            raise SynthesisLLMError("LLM response contained no JSON object.")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise SynthesisLLMError(f"LLM JSON malformed: {exc}") from exc


def call_json(
    *,
    system: str,
    user: str,
    temperature: float = 0.1,
    max_tokens: int = 2_048,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Call OpenAI Chat Completions and return a parsed JSON object.

    Args:
        system: System instruction for the model.
        user: User-side prompt (already filled with all data the stage needs).
        temperature: Sampling temperature (synthesis prefers low temps).
        max_tokens: Response token budget.
        model: Override ``AppSettings.openai_model``.

    Raises:
        SynthesisLLMError: When the API is unconfigured or the response is unparseable.
    """
    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        raise SynthesisLLMError("OPENAI_API_KEY is not configured.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SynthesisLLMError("openai package is not installed") from exc

    client = OpenAI(api_key=api_key)
    chosen_model = (model or settings.openai_model).strip() or "gpt-4o"

    try:
        resp = client.chat.completions.create(
            model=chosen_model,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI synthesis call failed")
        raise SynthesisLLMError(f"OpenAI call failed: {exc}") from exc

    raw = resp.choices[0].message.content or ""
    return _coerce_json(raw)


def call_text(
    *,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 3_500,
    model: str | None = None,
) -> str:
    """
    Call OpenAI Chat Completions for plain free-form text (no JSON parsing).

    Used by the literature-review generator where the artifact is prose.
    """
    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        raise SynthesisLLMError("OPENAI_API_KEY is not configured.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SynthesisLLMError("openai package is not installed") from exc

    client = OpenAI(api_key=api_key)
    chosen_model = (model or settings.openai_model).strip() or "gpt-4o"

    try:
        resp = client.chat.completions.create(
            model=chosen_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI synthesis text call failed")
        raise SynthesisLLMError(f"OpenAI call failed: {exc}") from exc

    return (resp.choices[0].message.content or "").strip()
