from __future__ import annotations

import re

import pytest

from guardrails.output import (
    EMPTY_SEARCH_SAFE_REPLY,
    TurnToolTracker,
    _tool_output_is_empty_list,
    apply_output_guardrails,
    response_contains_bibtex_blocks,
)


def test_response_detects_bibtex() -> None:
    assert response_contains_bibtex_blocks("@article{smith2024, title={X}}")
    assert response_contains_bibtex_blocks("@techreport{tr1, title={T}}")
    assert not response_contains_bibtex_blocks("No papers found.")


def test_tool_output_is_empty_list_variants() -> None:
    assert _tool_output_is_empty_list("[]")
    assert _tool_output_is_empty_list("  []  ")
    assert _tool_output_is_empty_list("")
    assert not _tool_output_is_empty_list("[1]")
    assert not _tool_output_is_empty_list("not a list")


def test_apply_guardrail_replaces_fabricated_bibtex_after_empty_searches() -> None:
    tracker = TurnToolTracker()
    tracker.record("tool_search_arxiv", "[]")
    tracker.record("tool_search_crossref", "[]")
    raw = "@inproceedings{fake2024,\n  title = {Made Up},\n}"
    out = apply_output_guardrails(raw, tracker)
    assert out == EMPTY_SEARCH_SAFE_REPLY
    assert not response_contains_bibtex_blocks(out)


def test_apply_guardrail_mixed_search_results_allows_bibtex() -> None:
    tracker = TurnToolTracker()
    tracker.record("tool_search_arxiv", "[]")
    tracker.record("tool_search_dblp", '[{"paper_id": "dblp:1", "title": "Hit"}]')
    bib = "@article{real2024, title = {{Real}}}"
    assert apply_output_guardrails(bib, tracker) == bib


def test_apply_guardrail_single_empty_search_triggers() -> None:
    tracker = TurnToolTracker()
    tracker.record("tool_search_semantic_scholar", "[]")
    assert apply_output_guardrails("@article{x,}", tracker) == EMPTY_SEARCH_SAFE_REPLY


def test_apply_guardrail_allows_bibtex_when_search_had_hits() -> None:
    tracker = TurnToolTracker()
    tracker.record("tool_search_arxiv", "[1]")
    bib = "@article{real2024, title = {{Real}}}"
    assert apply_output_guardrails(bib, tracker) == bib


def test_apply_guardrail_no_op_without_search_tools() -> None:
    tracker = TurnToolTracker()
    tracker.record("tool_generate_bibtex", "@article{fromtool,}")
    bib = "@article{maybe2024, title = {{X}}}"
    assert apply_output_guardrails(bib, tracker) == bib


def test_apply_guardrail_no_op_on_empty_assistant_text() -> None:
    tracker = TurnToolTracker()
    tracker.record("tool_search_arxiv", "[]")
    assert apply_output_guardrails("   ", tracker) == "   "


def test_empty_search_safe_reply_has_no_bibtex() -> None:
    assert not re.search(r"@\w+\{", EMPTY_SEARCH_SAFE_REPLY)
