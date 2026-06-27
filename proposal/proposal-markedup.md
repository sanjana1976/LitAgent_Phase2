# A4 Proposal (Marked-Up): Research Synthesis Agent

This is a marked-up copy of my original proposal. For each item I mark it as:

- [IMPLEMENTED] built as written, with where it lives in the code
- [CHANGED] built but different from what I proposed, with what changed
- [DROPPED] no longer planned

The original is in proposal/proposal.md.

## Planned Technologies

- Python 3.11+: [IMPLEMENTED] used throughout.
- OpenAI LLM via OpenAI SDK and langchain-openai.ChatOpenAI: [IMPLEMENTED] LLM calls live in src/synthesis/llm.py and are used by the claim, contradiction, and generation steps.
- LangChain agent framework: [IMPLEMENTED] the chat agent is in src/agent/agent.py; synthesis is exposed as a tool in src/tools/synthesis_tools.py.
- PDF parsing from A3 (pypdf): [IMPLEMENTED] src/tools/pdf_tools.py plus src/synthesis/fetch_parse.py.
- SQLite storage and synthesis cache: [IMPLEMENTED] src/db/database.py, src/db/queries.py; synthesis runs are persisted and can be reloaded.
- Custom pytest eval harness: [IMPLEMENTED] src/synthesis/eval_harness.py, cases in eval/cases.json, results in eval/results.json, run via the eval-synthesis CLI command in main.py.
- Terminal REPL and synthesize CLI subcommand: [IMPLEMENTED] main.py. Also [CHANGED] added a Streamlit UI (streamlit_app.py) on top of the proposed terminal-only interface.

## First Deliverable

Single user story: user asks a research question and gets a structured literature review with themes, contradictions, and inline citations.

- Find 5 to 8 relevant papers: [IMPLEMENTED] src/synthesis/retrieve.py and src/tools/search_tools.py. [CHANGED] retrieval now relevance-ranks and trims the working set (src/synthesis/rank.py) and defaults to arXiv-only for speed.
- Read abstracts and key sections, PDF with abstract fallback: [IMPLEMENTED] src/synthesis/fetch_parse.py, abstract fallback handled via the text tier.
- Group into themes: [IMPLEMENTED] handled in the generation prompt (src/synthesis/prompt.py, src/synthesis/generate.py).
- Identify at least one contradiction: [IMPLEMENTED] src/synthesis/contradictions.py, driven by the controller.
- Structured argument with inline citations: [IMPLEMENTED] src/synthesis/generate.py and citation keys from src/synthesis/schemas.py.

[CHANGED] The biggest change from the proposal: the first deliverable was a fixed sequence of steps, but it is now run by an agentic controller (src/synthesis/controller.py) that decides what to do next from its own state (src/synthesis/state.py) and logs each decision (src/synthesis/trace.py).

## Rough Architecture (the 10 components)

1. Query Decomposer: [IMPLEMENTED] src/synthesis/decompose.py.
2. Paper Retriever: [IMPLEMENTED] src/synthesis/retrieve.py and src/tools/search_tools.py.
3. Paper Fetcher and Parser: [IMPLEMENTED] src/synthesis/fetch_parse.py, caching in src/tools/pdf_tools.py and src/tools/file_cache.py.
4. Relevance Ranker: [IMPLEMENTED] src/synthesis/rank.py (TF-IDF cosine). [CHANGED] now also applied mid-run by the controller and with a minimum relevance floor, not just once.
5. Claim Extractor: [IMPLEMENTED] src/synthesis/claims.py.
6. Contradiction Detector: [IMPLEMENTED] src/synthesis/contradictions.py.
7. Synthesis Prompt Builder: [IMPLEMENTED] src/synthesis/prompt.py.
8. Literature Review Generator: [IMPLEMENTED] src/synthesis/generate.py.
9. Citation Validator: [IMPLEMENTED] src/synthesis/validate_cites.py. [CHANGED] added citation-key disambiguation in src/synthesis/schemas.py so two papers with the same author/year are not dropped or falsely flagged.
10. Eval Harness: [IMPLEMENTED] src/synthesis/eval_harness.py, run via eval-synthesis in main.py, results in eval/results.json.

[CHANGED] The 10 steps still exist as actions, but they are no longer a fixed pipeline. The controller orchestrates them: retrieval, claims, contradiction detection, gap hunting, conflict resolution, then synthesis.

## After-First-Deliverable Goals

- Hallucination eval suite: [CHANGED] eval harness scores claim faithfulness and citation hallucination rate (src/synthesis/eval_harness.py, eval/results.json). The labeled set is smaller than the proposed 20+ questions (eval/cases.json has a starter set).
- Contradiction detection eval: [CHANGED] contradiction coverage is scored by the harness, but I did not curate the full set of 10 known disagreements; coverage is only counted when a run is labeled.
- Multi-round refinement: [CHANGED] partially covered: the controller does adaptive retrieval and reformulates queries on its own. There is no explicit user "go deeper on X" command.
- Gap detection: [IMPLEMENTED] run_gap_detection_loop in src/synthesis/controller.py hunts for support for ungrounded claims.
- Export to markdown or BibTeX: [DROPPED] not implemented.
- Session persistence: [IMPLEMENTED] synthesis runs are stored and reloadable via src/db/queries.py.
- Confidence scoring: [IMPLEMENTED] confidence and grounding tiers on claims (src/synthesis/schemas.py, src/synthesis/state.py).
- Comparison mode (two competing methods): [CHANGED] delivered as contradiction detection plus conflict resolution (the agent compares two papers' claims and hunts a third paper to contextualize the disagreement) rather than a separate pro/con command.
