"""
LitSynth (A4): research synthesis layer built on top of the A3 tool/agent base.

The synthesis pipeline turns a single research question into a structured,
citation-grounded literature review section with explicit contradiction
detection and hallucination-resistant citation validation.

Submodules (one per pipeline stage; intentionally narrow surfaces):

- :mod:`synthesis.schemas`         — pydantic data shapes shared across stages
- :mod:`synthesis.decompose`       — query decomposer (stage 1)
- :mod:`synthesis.retrieve`        — multi-source paper retriever + dedup (stage 2)
- :mod:`synthesis.fetch_parse`     — PDF / abstract fetcher wrapping A3 tools (stage 3)
- :mod:`synthesis.rank`            — TF-IDF cosine relevance ranker (stage 4)
- :mod:`synthesis.claims`          — per-paper claim extractor (stage 5)
- :mod:`synthesis.contradictions`  — cross-paper contradiction detector (stage 6)
- :mod:`synthesis.prompt`          — synthesis prompt builder (stage 7)
- :mod:`synthesis.generate`        — literature review generator (stage 8)
- :mod:`synthesis.validate_cites`  — citation validator (stage 9)
- :mod:`synthesis.eval_harness`    — eval harness scaffolding (stage 10)
- :mod:`synthesis.pipeline`        — orchestrator stitching stages 1..9 together
- :mod:`synthesis.llm`             — small wrapper for JSON-strict OpenAI calls

Design rule:
    Every stage takes typed inputs and returns typed outputs (see ``schemas``)
    so each stage can be unit-tested in isolation. The pipeline orchestrator
    is the only module that mutates SQLite or touches the network outside of
    A3 tools.
"""

from __future__ import annotations
