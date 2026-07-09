"""
Machine-readable registry of all agent tools (names, callables, return types, descriptions).

LangChain and other orchestration layers can introspect ``TOOL_SPECS`` to bind tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Annotated, Any, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class ToolSpec:
    """Static metadata describing a single tool."""

    name: str
    function: Callable[..., Any]
    doc_first_line: str
    return_annotation: str


def _format_annotation(tp: Any) -> str:
    if tp is None:
        return "None"
    if hasattr(tp, "__name__"):
        if getattr(tp, "__module__", "").startswith("builtins"):
            return tp.__name__
    s = str(tp).replace("typing.", "")
    return s


def _callable_return_str(fn: Callable[..., Any]) -> str:
    hints = get_type_hints(fn)
    if "return" in hints:
        return _format_annotation(hints["return"])
    return "Any"


def build_tool_specs() -> tuple[ToolSpec, ...]:
    """Import tool callables lazily to avoid circular imports at package import time."""
    from tools.analysis_tools import tool_deep_analyze_paper, tool_extract_citations
    from tools.bibliography_tools import (
        tool_generate_apa,
        tool_generate_bibtex,
        tool_generate_chicago,
    )
    from tools.citation_tools import tool_lookup_forward_citations
    from tools.compare_tools import tool_compare_papers
    from tools.pdf_tools import tool_fetch_and_parse_pdf
    from tools.reading_list_tools import (
        tool_add_paper_to_list,
        tool_create_reading_list,
        tool_get_list_contents,
        tool_list_all_lists,
        tool_remove_paper_from_list,
    )
    from tools.search_tools import (
        tool_search_arxiv,
        tool_search_crossref,
        tool_search_dblp,
        tool_search_semantic_scholar,
    )
    from tools.storage_tools import tool_export_list_to_bibtex, tool_save_summary
    from tools.synthesis_tools import (
        tool_get_review_context,
        tool_synthesize_literature_review,
    )

    definitions: list[tuple[str, Callable[..., Any]]] = [
        ("tool_search_arxiv", tool_search_arxiv),
        ("tool_search_dblp", tool_search_dblp),
        ("tool_search_semantic_scholar", tool_search_semantic_scholar),
        ("tool_search_crossref", tool_search_crossref),
        ("tool_fetch_and_parse_pdf", tool_fetch_and_parse_pdf),
        ("tool_deep_analyze_paper", tool_deep_analyze_paper),
        ("tool_extract_citations", tool_extract_citations),
        ("tool_lookup_forward_citations", tool_lookup_forward_citations),
        ("tool_compare_papers", tool_compare_papers),
        ("tool_generate_bibtex", tool_generate_bibtex),
        ("tool_generate_apa", tool_generate_apa),
        ("tool_generate_chicago", tool_generate_chicago),
        ("tool_create_reading_list", tool_create_reading_list),
        ("tool_add_paper_to_list", tool_add_paper_to_list),
        ("tool_remove_paper_from_list", tool_remove_paper_from_list),
        ("tool_list_all_lists", tool_list_all_lists),
        ("tool_get_list_contents", tool_get_list_contents),
        ("tool_save_summary", tool_save_summary),
        ("tool_export_list_to_bibtex", tool_export_list_to_bibtex),
        ("tool_synthesize_literature_review", tool_synthesize_literature_review),
        ("tool_get_review_context", tool_get_review_context),
    ]

    specs: list[ToolSpec] = []
    for name, fn in definitions:
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        specs.append(
            ToolSpec(
                name=name,
                function=fn,
                doc_first_line=doc,
                return_annotation=_callable_return_str(fn),
            )
        )
    return tuple(specs)


TOOL_SPECS: tuple[ToolSpec, ...] = build_tool_specs()


def list_tool_signatures() -> dict[str, str]:
    """
    Map tool name → '(args) -> return_type' style string for agent prompts.

    Note:
        Uses introspection and may omit complex ``Annotated`` metadata.
    """
    out: dict[str, str] = {}
    for spec in TOOL_SPECS:
        fn = spec.function
        name = spec.name
        try:
            hints = get_type_hints(fn, include_extras=True)
        except Exception:
            hints = getattr(fn, "__annotations__", {})
        arg_parts: list[str] = []
        code = getattr(fn, "__code__", None)
        varnames = list(code.co_varnames[: code.co_argcount]) if code else []
        for vn in varnames:
            if vn in ("self", "cls"):
                continue
            ann = hints.get(vn, Any)
            if get_origin(ann) is Annotated:
                ann = get_args(ann)[0]
            arg_parts.append(f"{vn}: {_format_annotation(ann)}")
        ret = hints.get("return", Any)
        if get_origin(ret) is Annotated:
            ret = get_args(ret)[0]
        out[name] = f"({', '.join(arg_parts)}) -> {_format_annotation(ret)}"
    return out
