"""
Agent tools: scholarly search (×4), PDF ingest, analysis, forward citations, compare, bibliography, lists, export.

Registry: ``tools_registry.TOOL_SPECS`` (19 tools). See README.md and design.md.
"""

from tools.registry import get_registered_tools
from tools.tools_registry import TOOL_SPECS, build_tool_specs, list_tool_signatures

__all__ = [
    "TOOL_SPECS",
    "build_tool_specs",
    "get_registered_tools",
    "list_tool_signatures",
]
