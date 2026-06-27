"""Unit tests for :mod:`synthesis.generate` (literature-review generator)."""

from __future__ import annotations

from typing import Any

import pytest

from synthesis.generate import generate_literature_review
from synthesis.llm import SynthesisLLMError
from synthesis.prompt import SynthesisPrompt


def _make_prompt(citations: list[str] | None = None) -> SynthesisPrompt:
    return SynthesisPrompt(
        system="SYSTEM",
        user="USER",
        expected_citations=citations if citations is not None else [
            "[Smith et al. 2023]",
            "[Doe 2024]",
        ],
    )


def test_happy_path_returns_stripped_text() -> None:
    captured: dict[str, Any] = {}

    def fake_llm(*, system: str, user: str, temperature: float, max_tokens: int) -> str:
        captured["system"] = system
        captured["user"] = user
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return "   Generated review prose.   \n"

    prompt = _make_prompt()
    result = generate_literature_review(prompt, llm_call=fake_llm)

    assert result == "Generated review prose."
    assert captured["system"] == "SYSTEM"
    assert captured["user"] == "USER"
    assert captured["temperature"] == pytest.approx(0.3)
    assert 800 <= captured["max_tokens"] <= 4_000


def test_llm_error_returns_fallback_with_expected_citations() -> None:
    def boom(**_: Any) -> str:
        raise SynthesisLLMError("rate limit")

    prompt = _make_prompt(["[Smith et al. 2023]", "[Doe 2024]"])
    result = generate_literature_review(prompt, llm_call=boom)

    assert result.startswith("Literature review generation failed: rate limit")
    assert "[Smith et al. 2023]" in result
    assert "[Doe 2024]" in result


def test_never_raises_on_unexpected_exception() -> None:
    def explode(**_: Any) -> str:
        raise RuntimeError("totally unexpected")

    prompt = _make_prompt()
    result = generate_literature_review(prompt, llm_call=explode)

    assert "Literature review generation failed" in result
    assert "totally unexpected" in result


def test_fallback_when_no_expected_citations() -> None:
    def boom(**_: Any) -> str:
        raise SynthesisLLMError("api offline")

    prompt = _make_prompt(citations=[])
    result = generate_literature_review(prompt, llm_call=boom)

    assert "Literature review generation failed" in result
    assert "Sources reviewed: none." in result


def test_max_tokens_scales_with_word_budget() -> None:
    seen: dict[str, int] = {}

    def capture(*, system: str, user: str, temperature: float, max_tokens: int) -> str:
        seen["max_tokens"] = max_tokens
        return "ok"

    prompt = _make_prompt()
    generate_literature_review(prompt, word_budget=2000, llm_call=capture)
    assert seen["max_tokens"] <= 4_000

    generate_literature_review(prompt, word_budget=10, llm_call=capture)
    assert seen["max_tokens"] >= 800


def test_non_string_payload_is_converted_to_fallback() -> None:
    def returns_none(**_: Any) -> Any:
        return None

    prompt = _make_prompt()
    result = generate_literature_review(prompt, llm_call=returns_none)
    assert "Literature review generation failed" in result
