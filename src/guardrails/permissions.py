"""Guardrail and permission policy for tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.config import AppSettings


class GuardrailError(PermissionError):
    """Raised when a disallowed operation is requested."""


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission check."""

    allowed: bool
    needs_confirmation: bool
    reason: str | None = None


class PermissionManager:
    """
    Three-layer guardrail policy with optional config-based overrides.

    Layer 1 (autonomous): execute immediately.
    Layer 2 (confirmation): ask user before executing.
    Layer 3 (blocked): refuse and explain.
    """

    _BLOCK_REASONS: dict[str, str] = {
        "tool_delete_paper": "Deleting papers is blocked due to data-loss risk.",
        "tool_delete_reading_list": "Deleting reading lists is blocked due to data-loss risk.",
        "fabricate_metadata": "Fabricating paper metadata is blocked. API unavailable, try another source.",
        "bypass_paywall": "Bypassing paywalls is blocked. Please use your university library access.",
        "modify_pdf": "Modifying PDF files is blocked to keep originals pristine.",
        "access_outside_project": "Accessing files outside the project directory is blocked.",
    }

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._autonomous = set(settings.guardrails_autonomous_tools)
        self._needs_confirmation = set(settings.guardrails_confirmation_tools)
        self._blocked = set(settings.guardrails_blocked_tools)

    def check_permission(self, tool_name: str, action: str) -> PermissionDecision:
        """
        Return whether a tool call is allowed and if confirmation is required.
        """
        _ = action  # reserved for future action-specific policies
        if tool_name in self._blocked:
            reason = self._BLOCK_REASONS.get(tool_name, "This operation is blocked by policy.")
            return PermissionDecision(allowed=False, needs_confirmation=False, reason=reason)

        if tool_name in self._needs_confirmation:
            return PermissionDecision(allowed=True, needs_confirmation=True)

        if tool_name in self._autonomous:
            return PermissionDecision(allowed=True, needs_confirmation=False)

        # Conservative default for unknown tools.
        return PermissionDecision(
            allowed=True,
            needs_confirmation=True,
            reason="Unrecognized tool requires explicit confirmation.",
        )

    def validate_filesystem_target(self, candidate: str | Path) -> None:
        """Block writes that escape the configured project root."""
        resolved = Path(candidate).expanduser().resolve()
        root = self._settings.project_root
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise GuardrailError(self._BLOCK_REASONS["access_outside_project"]) from exc

    def check_blocked_intent(self, user_text: str) -> PermissionDecision | None:
        """
        Detect clearly blocked intents before tool selection.

        This is a lightweight phrase matcher, not a full intent classifier.
        """
        low = user_text.lower()
        if "delete reading list" in low or "delete list" in low:
            return PermissionDecision(False, False, self._BLOCK_REASONS["tool_delete_reading_list"])
        if "delete paper" in low:
            return PermissionDecision(False, False, self._BLOCK_REASONS["tool_delete_paper"])
        if "bypass paywall" in low or "remove paywall" in low:
            return PermissionDecision(False, False, self._BLOCK_REASONS["bypass_paywall"])
        if "modify pdf" in low or "edit pdf" in low:
            return PermissionDecision(False, False, self._BLOCK_REASONS["modify_pdf"])
        if "fabricate" in low and "metadata" in low:
            return PermissionDecision(False, False, self._BLOCK_REASONS["fabricate_metadata"])
        return None

    def needs_confirmation_for_write(self, tool_name: str, kwargs: dict[str, Any]) -> bool:
        """
        Additional write checks for potentially overwriting operations.
        """
        if tool_name == "tool_export_list_to_bibtex":
            return True
        if tool_name == "tool_save_summary":
            return True
        _ = kwargs
        return False
