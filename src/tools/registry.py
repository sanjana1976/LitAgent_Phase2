"""Central registry for LangChain-compatible tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tools.tools_registry import TOOL_SPECS


def get_registered_tools() -> list[Callable[..., Any]]:
    """
    Return tool callables (plain Python functions) for binding to LangChain/LCEL.

    Full metadata is exposed on ``tools.tools_registry.TOOL_SPECS``.
    """
    return [spec.function for spec in TOOL_SPECS]
