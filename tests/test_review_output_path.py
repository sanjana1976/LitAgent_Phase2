"""Unit tests for ``main._resolve_review_output_path``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# main.py is at repo root, not a package; load it via spec so tests stay path-agnostic.
_MAIN_SPEC = importlib.util.spec_from_file_location(
    "_litsynth_main",
    Path(__file__).resolve().parents[1] / "main.py",
)
assert _MAIN_SPEC is not None and _MAIN_SPEC.loader is not None
_main = importlib.util.module_from_spec(_MAIN_SPEC)
_MAIN_SPEC.loader.exec_module(_main)

REVIEWS_DIR_NAME = _main.REVIEWS_DIR_NAME
_resolve_review_output_path = _main._resolve_review_output_path
_slugify_question = _main._slugify_question


def test_relative_output_anchored_under_reviews_dir(tmp_path: Path) -> None:
    target = _resolve_review_output_path(
        Path("review.md"),
        question="long-context retrieval",
        project_root=tmp_path,
    )
    assert target == tmp_path / REVIEWS_DIR_NAME / "review.md"


def test_relative_output_already_in_reviews_dir_not_double_prefixed(tmp_path: Path) -> None:
    target = _resolve_review_output_path(
        Path(REVIEWS_DIR_NAME) / "custom.md",
        question="q",
        project_root=tmp_path,
    )
    assert target == tmp_path / REVIEWS_DIR_NAME / "custom.md"


def test_absolute_output_used_as_is(tmp_path: Path) -> None:
    abs_target = (tmp_path / "elsewhere" / "out.md").resolve()
    target = _resolve_review_output_path(
        abs_target,
        question="q",
        project_root=tmp_path,
    )
    assert target == abs_target


def test_no_output_generates_timestamped_file_in_reviews_dir(tmp_path: Path) -> None:
    target = _resolve_review_output_path(
        None,
        question="What about RAG vs fine-tuning?",
        project_root=tmp_path,
    )
    assert target.parent == tmp_path / REVIEWS_DIR_NAME
    assert target.suffix == ".md"
    assert target.name.startswith("review-")
    # Slug has no spaces / weird chars and is lowercased.
    assert "rag-vs-fine-tuning" in target.name


@pytest.mark.parametrize(
    "raw, expected_substring",
    [
        ("Hello, World!", "hello-world"),
        ("RAG vs Fine-tuning", "rag-vs-fine-tuning"),
        ("   ", "review"),
        ("", "review"),
    ],
)
def test_slugify_question(raw: str, expected_substring: str) -> None:
    assert _slugify_question(raw) == expected_substring or expected_substring in _slugify_question(raw)
