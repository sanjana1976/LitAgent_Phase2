from __future__ import annotations

from pathlib import Path

import pytest

from config.config import get_settings
from guardrails.permissions import GuardrailError, PermissionManager
from guardrails.validators import validate_user_message


@pytest.fixture
def pm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PermissionManager:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    settings = get_settings(reload=True)
    return PermissionManager(settings)


def test_permission_manager_autonomous_search(pm: PermissionManager) -> None:
    auto = pm.check_permission("tool_search_arxiv", "execute")
    assert auto.allowed and not auto.needs_confirmation
    forward = pm.check_permission("tool_lookup_forward_citations", "execute")
    assert forward.allowed and not forward.needs_confirmation


def test_permission_manager_confirmation_writes(pm: PermissionManager) -> None:
    gated = pm.check_permission("tool_save_summary", "execute")
    assert gated.allowed and gated.needs_confirmation
    export = pm.check_permission("tool_export_list_to_bibtex", "execute")
    assert export.allowed and export.needs_confirmation


def test_permission_manager_unknown_tool_needs_confirmation(pm: PermissionManager) -> None:
    unknown = pm.check_permission("tool_nonexistent_future", "execute")
    assert unknown.allowed and unknown.needs_confirmation
    assert unknown.reason


def test_permission_manager_blocked_tool_from_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("GUARDRAILS_BLOCKED_TOOLS", '["tool_delete_everything"]')
    settings = get_settings(reload=True)
    pm = PermissionManager(settings)
    decision = pm.check_permission("tool_delete_everything", "execute")
    assert not decision.allowed
    assert not decision.needs_confirmation
    assert decision.reason


@pytest.mark.parametrize(
    ("user_text", "reason_key"),
    [
        ("please delete reading list 3", "tool_delete_reading_list"),
        ("delete list 3 now", "tool_delete_reading_list"),
        ("delete paper 9", "tool_delete_paper"),
        ("help me bypass paywall", "bypass_paywall"),
        ("remove paywall from pdf", "bypass_paywall"),
        ("modify pdf headers", "modify_pdf"),
        ("edit pdf content", "modify_pdf"),
        ("fabricate metadata for this", "fabricate_metadata"),
    ],
)
def test_check_blocked_intent_phrases(
    pm: PermissionManager,
    user_text: str,
    reason_key: str,
) -> None:
    decision = pm.check_blocked_intent(user_text)
    assert decision is not None
    assert not decision.allowed
    assert decision.reason == pm._BLOCK_REASONS[reason_key]


@pytest.mark.parametrize(
    "user_text",
    [
        "search arxiv for transformers",
        "delete outdated notes from my summary",
        "generate bibtex for paper 1",
    ],
)
def test_check_blocked_intent_allows_benign_phrases(pm: PermissionManager, user_text: str) -> None:
    assert pm.check_blocked_intent(user_text) is None


def test_validate_filesystem_target_inside_project(pm: PermissionManager, tmp_path: Path) -> None:
    target = tmp_path / "exports" / "list.bib"
    pm.validate_filesystem_target(target)  # no raise


def test_validate_filesystem_target_outside_project(pm: PermissionManager, tmp_path: Path) -> None:
    outside = tmp_path.parent.parent / "outside_export.bib"
    with pytest.raises(GuardrailError, match="outside the project"):
        pm.validate_filesystem_target(outside)


def test_validate_user_message_rejects_empty() -> None:
    with pytest.raises(GuardrailError, match="cannot be empty"):
        validate_user_message("   ")


def test_validate_user_message_rejects_oversized() -> None:
    huge = "x" * 32_001
    with pytest.raises(GuardrailError, match="maximum length"):
        validate_user_message(huge)


def test_validate_user_message_strips() -> None:
    assert validate_user_message("  hello  ") == "hello"
