"""User-confirmation contract for mutating tools."""

from __future__ import annotations


class ConfirmationRequired(RuntimeError):
    """
    Raised when a tool that modifies persistent state or files needs explicit approval.

    The agent should obtain user consent, then re-invoke the same tool with
    ``user_confirmed=True``.
    """

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        super().__init__(message)
        self.tool_name = tool_name
