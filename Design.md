# DESIGN

This document focuses on design decisions for the Research Paper Analyzer Agent and why they were made. It is organized as the system actually evolved: A3 (the chat assistant), A4 (the LitSynth synthesis pipeline), A4.5 (the agentic refactor with an explicit decision trace), and finally the LangGraph migration that the whole system runs on today. The historical sections are kept on purpose — the current architecture only makes sense as the answer to problems the earlier versions hit.

## Core architecture (current)

Two cooperating LangGraph systems share one tool layer, one SQLite store, and one guardrail policy.

### Runtime 1: the chat assistant (`main.py chat`)

```
User (terminal)
  -> main.py chat
      ├── PermissionManager.check_blocked_intent(user text)     (before any LLM call)
      ├── ConversationManager (history + SQLite session rows)
      └── AgentManager.respond()
              ├── langchain.agents.create_agent graph (LangGraph prebuilt)
              │     └── 21 permission-guarded tools:
              │           blocked      -> tool returns its policy reason, never executes
              │           confirmation -> interrupt() pauses the WHOLE graph;
              │                           the CLI asks yes/no; Command(resume=...)
              │                           re-enters the tool with the decision
              │           autonomous   -> executes
              ├── permission_audit rows in SQLite (once per decision)
              └── TurnToolTracker + apply_output_guardrails on the final reply
```

The important property: **permission policy lives inside the tools, not in an orchestration loop.** Each registered tool is wrapped in a guard that enforces the three tiers at execution time. Confirmation is a real graph interrupt — the run genuinely pauses, survives in the checkpointer, and resumes with the user's decision — not a callback threaded through a hand-rolled loop. `AgentManager.respond()` kept its original synchronous contract through the migration, so the conversation layer, the CLI, and the transcripts never changed.

Step budget: `_max_steps = 8` maps to the graph's `recursion_limit`; a model that calls tools forever raises `AgentError`, same contract as the original loop. Unknown tool names come back as error `ToolMessage`s and the model recovers. Tool failures are stringified into the transcript so the REPL never exits.

### Runtime 2: the synthesis engine (`main.py synthesize`, the chat tool, Streamlit)

`synthesis/graph.py` is a `StateGraph` over `SynthesisState`:

```
decompose                       1 LLM call -> keyword sub-queries (deterministic fallback)
    v
plan_search --Send--> search_one x N        parallel: one branch per unsearched sub-query
    v
rank_pool                                   join: TF-IDF ranks the pool -> active_paper_ids
    v
coverage router ----thin?----> reformulate -> plan_search        (adaptive-retrieval cycle,
    v                                                              max_reformulations budget)
filter_relevance                            LLM gate drops off-topic papers (fail-soft)
    v            \---gutted?--> reformulate (the gate can also re-enter the cycle)
extract_one x N                             parallel: one claim-extraction branch per paper
    v
detect_contradictions                       1 LLM call, code-validated pairs
    v
hunt_one x N                                parallel: one branch per ungrounded claim
    v
resolve_one x N                             parallel: one branch per contradiction
    v
consolidate                                 join: folds hunt-added papers into the active set
    v
synthesize (writer) <------> critique (critic)      writer-critic loop, max_revisions=2
    v
review + full decision trace, persisted to synthesis_runs
```

Five structural rules hold everything together:

1. **State is the schema, the graph is the control flow.** `SynthesisState` fields carry LangGraph reducers: `papers`, `claims`, `contradictions`, and `gaps` upsert by stable id; `trace` appends. Nodes return partial update dicts with modified *copies* — a gap hunt upgrading a claim returns a new `ClaimRecord` with the same `claim_id`, and the reducer replaces the old one in place. Nothing ever mutates a shared object, which is exactly the property that makes the parallel fan-outs safe.
2. **Evidence vs. selection.** `papers` is append-only provenance — once a paper is fetched it is never deleted. Which papers feed downstream stages is the separate `active_paper_ids` list (plain replace semantics). Only the two join nodes (`rank_pool`, `consolidate`) ever write it, so a filtering step can never destroy evidence and two parallel branches can never collide on it.
3. **Map-reduce parallelism with a budget.** Dispatcher conditionals return `Send` batches; worker nodes receive small payload dicts, not full state; joins run once after all branches merge. `max_concurrency` (default 4) bounds the branches so parallel LLM/arXiv calls respect rate limits. Measured live: the flagship demo question dropped from ~2.5–3 minutes to ~70 seconds wall clock.
4. **Every superstep is checkpointed.** A `SqliteSaver` writes to `data/checkpoints.sqlite3` under the run id printed at start. `synthesize --resume RUN_ID` continues an interrupted run — crash, quota exhaustion, Ctrl-C — without re-executing completed stages, including finished parallel branches. Verified by killing a live run mid-flight and resuming it to a finished review.
5. **Steps are born completed.** The old two-phase pending→complete `DecisionStep` arc existed because one loop interleaved deciding and executing. In the graph, edges decide and nodes execute, so each node emits finished `DecisionStep`s. The trace schema, the Streamlit UI, and the eval harness were unchanged by the migration.

Dependency injection flows through the LangGraph config channel (`config["configurable"]["litsynth_deps"]` / `litsynth_config`) — the same seam production uses to wire real stage functions is the seam tests use to inject stubs, so the whole graph runs network-free under pytest.

Entry points: `run_graph_synthesis_state(...)` returns the full traced state (Streamlit needs the trace); `run_graph_synthesis(...)` wraps it for callers that only need the result artifact (CLI, chat tool). Both accept `checkpoint_path` / `thread_id` / `resume`.

### The writer-critic loop

The last two nodes turn hallucination resistance from *detect-and-report* into *detect-and-fix*. After the writer drafts the review, a **critic** — a second LLM role in `synthesis/critic.py` — reads the draft against the only evidence the writer was given and raises at most four **checkable objections**: statements no extracted claim supports, contradictions in the evidence the draft failed to surface, citation keys outside the allowed list.

The critic obeys the same discipline as every LLM judgment in this codebase — **the model proposes, code disposes**:

- every objection must quote a verbatim excerpt from the draft; excerpts that do not appear in the draft (normalized substring check) are dropped as fabrications;
- hallucinated citations found by the deterministic validator become auto-objections regardless of what the critic model says, so the loop's most important signal never depends on model judgment;
- style, length, and ordering complaints are prompted away and, if they slip through without an excerpt, die at validation.

Objections send the draft back to the writer with the previous draft and the objection list appended to the synthesis prompt; the rewrite is re-validated and re-critiqued, up to `max_revisions` (default 2). Fail-soft in both directions: a broken critic accepts the draft, and a failed rewrite keeps the previous draft. Unresolved objections at the budget cap stay honestly recorded in `state.objections`.

### What runs where

| Surface | Path |
|---------|------|
| `python main.py synthesize "..."` | `run_graph_synthesis` with checkpointing + run id |
| `python main.py synthesize --resume ID` | same graph, resumed from the checkpoint thread |
| chat: "write me a literature review on X" | `tool_synthesize_literature_review` -> same graph |
| chat follow-ups: "which of those papers..." | `tool_get_review_context` recalls the session's last review |
| `streamlit run streamlit_app.py` | `run_graph_synthesis_state` -> review + decision-trace panels |
| `python main.py eval-synthesis` | scores persisted `SynthesisResult`s; `--live` runs the graph |

---

## Summary

- **Framework**: LangGraph end to end — a `StateGraph` workflow for synthesis, a `create_agent` prebuilt graph for chat — over LangChain `ChatOpenAI` as the model layer.
- **Outside world**: HTTP APIs (arXiv, DBLP, Semantic Scholar, Crossref), PDF fetch + cache, SQLite (data + run checkpoints), BibTeX export under project root.
- **Multi-turn**: `ConversationManager` + SQLite sessions; `chat` / `resume` / `status`; synthesis runs resumable via per-run checkpoints.
- **Guardrails: autonomous** — Layer 1 tool list (search, read, analyze, format citations, synthesis).
- **Guardrails: confirm** — Layer 2 writes and exports, enforced as graph interrupts.
- **Guardrails: disallow** — Layer 3 blocked intents, optional blocked tools, filesystem boundary; **plus** output guardrail for fabricated BibTeX after empty searches; **plus** the critic loop on generated reviews.

Scope stays **narrowly research-assistant** (literature search, PDF-backed analysis, reading lists, citations, forward-citation discovery, literature-review synthesis) rather than a general coding agent.

### Which agent class is used? (REVIEW CHANGE 1/3)

**`AgentManager`** is the primary runtime. It owns the LangGraph agent graph, the guarded tools, interrupt confirmations, tool tracking, and output sanitization. `main.py` constructs this directly.
**`ResearchPaperAgent`** is a legacy wrapper around `AgentManager` exposing only `generate_reply(history)` for older call sites. No permission manager, no output guardrail hook, no confirmation callback.
**`ConversationManager`**: history + persistence only.

### Data and external systems

- **SQLite**: papers, reading lists, summaries, conversation history, permission audit, synthesis runs. Schema version **4** (`src/db/init_db.py`); lightweight `ALTER TABLE` migrations upgrade older files. Run checkpoints live in a separate `data/checkpoints.sqlite3` managed by LangGraph's `SqliteSaver`.
- **HTTP APIs:** arXiv Atom, DBLP XML, Semantic Scholar Graph API, Crossref REST (base URLs in `config/config.py`, overridable via env). TLS uses the OS certificate store when the optional `truststore` package is present, so corporate/AV-proxied machines work out of the box.
- **Local cache:** PDF text and parsed sections under `{database_parent}/cache/` (`tools/context.get_cache_dir`, `tools/file_cache.py`, `tools/paper_text.py`). Keys are SHA-256 hashed filenames to avoid path issues.
- **Transcripts (`transcripts/`):** **not** written by the CLI. JSON files there are **manual documentation**.

### CLI bootstrap and packaging

- **`main.py`** prepends `src/` and the repo root to `sys.path` so `python main.py` works without an editable install; **`pip install -e .`** is still recommended for IDE imports and pytest `pythonpath`.
- **Click** command group: `init-db`, `chat`, `resume`, `status`, `synthesize` (with `--resume`), `synth-history`, `eval-synthesis`; global `--log-level` overrides `LOG_LEVEL`.
- **`chat --no-persist`:** skips SQLite writes for conversation turns and permission audit (AgentManager gets `database=None`); schema init still runs so reading-list tools work.
- **`chat --reading-list-context`:** optional string stored on each persisted turn for archival/tagging.
- **Confirmation UX:** write tools prompt `Confirm execution of …? (yes/no)` with default **`no`**; only `y`, `yes`, or `confirm` approve. Under the hood this is a LangGraph interrupt, not a mid-loop callback.

---

## Three Key Design Decisions

### 1) Graph-orchestrated agent with tool calling, not a one-shot chatbot

**Decision:**
Use a dedicated `AgentManager` around a LangGraph agent graph (`langchain.agents.create_agent` over `ChatOpenAI`) with permission-guarded tools, instead of a one-shot chatbot response model.

Paper analysis requests are rarely single-step — users ask discovery + comparison + citations in one flow. The agent needs to decide tools, inspect outputs, then continue reasoning before the final response.

**How it works:**

- Tools are registered in `src/tools/tools_registry.py`. Each is wrapped in a permission guard and bound into the agent graph; the graph runs model → tools → model until the model returns text with no tool calls or the step limit is hit.
- **Max 8 tool-iteration steps** per user message (`AgentManager._max_steps`, mapped to the graph's `recursion_limit`); exceeding raises `AgentError`.
- **Unknown tool names** from the model produce an error `ToolMessage` and the loop continues: no process crash.
- **Tool failures** (`GuardrailError`, provider errors, etc.) are stringified into `ToolMessage` content so the model can recover; the REPL does not exit.
- **Confirmation-tier tools** pause the entire graph with `interrupt()`; the CLI's yes/no answer resumes it with `Command(resume=...)`. Audit logging sits *after* the interrupt point in the wrapper so it fires exactly once per decision even though the wrapper re-executes on resume.
- **`context_note`** (from `ConversationManager.context_snapshot()`) is injected as an extra `SystemMessage` after the main system prompt, not mixed into user history. It now also carries the session's last review topic so follow-up turns stay anchored.
- The system prompt in `src/agent/agent.py` lists tool-use policy; **tool signatures** from `list_tool_signatures()` are appended so the model sees parameter shapes.
- **Test seams are explicit constructor arguments**, not internal patching: `model_instance` injects a scripted `BaseChatModel` (tests/helpers_llm.py) and `tool_overrides` swaps a tool's callable while keeping its schema.

**Trade-offs:**

- Better multi-step reasoning for research workflows
- Flexible tool composition without hardcoding every user intent path
- Easier future extension for new tools
- More runtime complexity than a direct single LLM call
- Tool loops need step limits and careful error handling

**Ownership split (me vs assistant):**

- I defined the high-level behavior and requirements (OpenAI backbone, deep analysis expectations, multi-turn continuity)
- Assistant implemented the code, slacked off about edge cases in the beginning till i forced it

---

### 2) Guardrails: input, tool permissions, output, and filesystem

**Decision:**
Safety is split across **four mechanisms** (not only the three course tiers for tools):

1. **Input intent blocking**: phrase match on raw user text before the LLM runs.
2. **Tool permission tiers**: autonomous vs confirmation vs blocked tool names, enforced inside the tools themselves.
3. **Output guardrail**: post-process the assistant's final message when searches were empty but BibTeX was invented.
4. **Filesystem boundary**: exports must stay under `PROJECT_ROOT`.

**Why this matters:**
This assistant runs workflows touching local persistence and export paths. Explicit policy boundaries are needed for trust, repeatability, and safety review. Empty search results are a common case where models hallucinate citations; the output layer catches that even if the prompt is ignored.

#### 2a) Input intent (`check_blocked_intent`) and message validation

**Pre-LLM intent**: lightweight substring match in `PermissionManager` (not a full classifier).

#### 2b) Tool permission tiers (`check_permission`)

Values come from `AppSettings` in `config/config.py` (`guardrails_*_tools`) and `PermissionManager` in `src/guardrails/permissions.py`. Since the LangGraph migration the check runs **inside each tool's guard wrapper** at execution time, so no orchestration path can skip it.

**Layer 1: autonomous (no confirmation):**

| Category | Tools |
|----------|-------|
| Search | `tool_search_arxiv`, `tool_search_dblp`, `tool_search_semantic_scholar`, `tool_search_crossref` |
| PDF & analysis | `tool_fetch_and_parse_pdf`, `tool_deep_analyze_paper`, `tool_extract_citations`, `tool_lookup_forward_citations`, `tool_compare_papers` |
| Bibliography formatting | `tool_generate_bibtex`, `tool_generate_apa`, `tool_generate_chicago` |
| Reading list read | `tool_list_all_lists`, `tool_get_list_contents` |
| Synthesis | `tool_synthesize_literature_review`, `tool_get_review_context` |

**Layer 2: confirmation required** (graph interrupt; CLI prompts `yes` / `confirm`):

- `tool_create_reading_list`, `tool_add_paper_to_list`, `tool_remove_paper_from_list`
- `tool_save_summary`, `tool_export_list_to_bibtex`

**Layer 3: blocked or conservative defaults:**

- **`guardrails_blocked_tools`:** empty by default; extra tool names can be added via environment. A blocked tool returns its policy reason as the tool result — the model sees why, nothing executes.
- **Unknown tool names:** confirmation-required (not silently auto-allowed).
- **Filesystem:** `validate_filesystem_target` for export paths (resolved path must be under `PROJECT_ROOT`).

**Dual confirmation path for writes:**

1. **Graph layer**: the guard wrapper pauses the run with `interrupt({"tool": name, "args": ...})`; `main.py` supplies the yes/no; on approval the wrapper injects **`user_confirmed=True`** into the tool args — but only when the tool's schema actually accepts that flag (the old blind injection was silently dropped by pydantic for tools without the parameter).
2. **Tool layer**: mutating tools (`tool_save_summary`, `tool_export_list_to_bibtex`) raise **`ConfirmationRequired`** when overwriting existing summaries or clobbering an on-disk `.bib` file unless `user_confirmed=True` (defense if the model retries without going through the CLI prompt).

**Export path check:** `validate_filesystem_target` runs on `tool_export_list_to_bibtex`'s `filename` argument inside the guard, immediately before execution (in addition to tool-internal checks).

**Permission audit:** rows in `permission_audit` are written only when `AgentManager` is constructed with a `Database` (i.e. `chat` with default `--persist`). Logging is positioned after the interrupt point so a confirmed call audits exactly once.

**System prompt:** `src/agent/agent.py` repeats complementary rules (no deletion, no paywall bypass, no writes outside the project, confirm mutations, refuse fabrication on API failure, do not invent BibTeX when all searches return nothing).

#### 2c) Output guardrail (`apply_output_guardrails`)

Implemented in `src/guardrails/output.py`, invoked at the end of `AgentManager.respond()` over the tool outputs recorded during the run. This is **defense in depth** beyond the system prompt. It does not block legitimate BibTeX from `tool_generate_bibtex` when searches returned papers; it targets BibTeX-shaped text in the model's free-form reply after failed discovery.

#### Tool signatures and OpenAI JSON schema

Agent-bound tools must expose only JSON-serializable argument types. Reading-list, bibliography, and storage tools use `get_default_database()` internally instead of a `Database` parameter the model could not fill safely.

**Trade-offs:**

- Strong safety defaults and reviewability
- Clear user control for mutations
- Slightly slower interaction when many writes are requested
- Output guardrail may occasionally replace a borderline reply; tuning is possible via regex or tracker rules

**Ownership split (me vs assistant):**

- I drove the policy shape and blocked behaviors
- Assistant implemented guardrail classes, integration points, DB audit plumbing, output guardrail, and tests

---

### 3) Session-aware conversation persistence + resume-first CLI UX

**Decision:**
Persist each turn with `session_id` in SQLite and expose conversational controls directly in CLI (`help`, `status`, `list my lists`, `clear history`, `resume`). Synthesis runs get the same treatment: every run persists to `synthesis_runs`, checkpoints per step, and can be resumed by run id.

**Why this matters:**
Research analysis is long-running and incremental. Users need continuity across restarts, not only in-memory context — and a 70-second, dozen-LLM-call synthesis run should never be lost to a crash at second 60.

#### CLI presentation (`src/cli/welcome.py`) (REVIEW CHANGE 2/3)

On `python main.py chat`, the REPL prints a **bordered welcome banner** (Unicode box, short feature lines, emojis) plus session id and model name. Resume shows turn count in the banner.

- **`help`**: formatted command reference via `echo_chat_help()`
- **`list my lists`**: bypasses the LLM; calls `tool_list_all_lists` directly
- **Quiet default**: unless `--verbose` or a parent `--log-level` is set, chat reconfigures logging to **WARNING** so tool/API `INFO` lines do not clutter the terminal during searches
- **`python main.py chat --verbose`**: keeps parent/root log level (does not force WARNING); use with `python main.py --log-level INFO chat` to see tool traffic
- **`setup_logging`** also sets **`httpx`** and **`openai`** loggers to WARNING to reduce SDK noise
- After each assistant reply, guardrail error, or special command, a blank line precedes the next `You>` prompt for readability

#### Chat memory for synthesis follow-ups

After a review is generated in chat, follow-up questions ("which of those papers…", "the second paper…") work through two mechanisms: the mandatory "Papers used" block the agent prints after every review (paper ids stay in the visible transcript), and `tool_get_review_context`, which reloads the session's most recent persisted review — question, text, and full paper set — even after a `resume` into a fresh process. `context_snapshot()` also injects the last review topic into every turn's context note.

**Trade-offs:**

- Strong continuity for multi-turn technical discussions
- Easy operational visibility in terminal-only use
- More schema + command-surface maintenance
- Session data management grows over time without pruning strategy

**Ownership split (me vs assistant):**

- I prioritized resume-capable workflows and usability commands
- Assistant implemented session query helpers, command wiring, and welcome banner

---

## Tool catalog (21 tools)

Registry: `src/tools/tools_registry.py` · implementations under `src/tools/`.

| Tool | Purpose |
|------|---------|
| `tool_search_arxiv` | Keyword/author/category/date search via arXiv Atom API; free text is sanitized into `all:term AND …` boolean queries with an automatic OR-broadening retry when the precise query finds nothing |
| `tool_search_dblp` | CS bibliography search (DBLP XML) |
| `tool_search_semantic_scholar` | Search with citation counts (S2 Graph API) |
| `tool_search_crossref` | Works search or DOI lookup (Crossref REST) |
| `tool_fetch_and_parse_pdf` | Download PDF, extract text, cache sections |
| `tool_deep_analyze_paper` | Structured analysis (OpenAI when key present; heuristics fallback) |
| `tool_extract_citations` | Parse reference block from cached PDF text (**backward** citations) |
| `tool_lookup_forward_citations` | Papers that **cite** a given work via S2 `/paper/{id}/citations` (**forward** citations) |
| `tool_compare_papers` | Side-by-side comparison of multiple cached/analyzed papers |
| `tool_generate_bibtex` | BibTeX string from DB row or cached text |
| `tool_generate_apa` | APA-style reference string |
| `tool_generate_chicago` | Chicago author-date string |
| `tool_create_reading_list` | New list (confirmation) |
| `tool_add_paper_to_list` | Add paper to list (confirmation) |
| `tool_remove_paper_from_list` | Remove mapping (confirmation; not full paper delete) |
| `tool_list_all_lists` | List reading list names/ids |
| `tool_get_list_contents` | Papers in a list with reading status |
| `tool_save_summary` | Persist analysis summary (confirmation) |
| `tool_export_list_to_bibtex` | Write `.bib` under project root (confirmation + path guard) |
| `tool_synthesize_literature_review` | Run the full LitSynth graph from chat; returns review + paper set as JSON |
| `tool_get_review_context` | Recall the session's most recent review (question, text, papers) for follow-up turns |

### Forward vs backward citations (NEW FEATURE: REVIEW CHANGE 3/3)

- **`tool_extract_citations`**: from PDF full text: what *this paper cites*.
- **`tool_lookup_forward_citations`**: from Semantic Scholar: what *cites this paper* ("who built on this?"). Accepts `s2:…`, `arxiv:…`, DOI, `crossref:…`, or numeric SQLite `paper_id` when S2/DOI metadata exists (`src/tools/citation_tools.py`).

---

## A4 LitSynth: research synthesis pipeline

A4 (`proposal.md`) builds on A3. The point of A4 is to actually generate something: take one research question and write a structured literature-review section, with contradiction detection and citation checking so the model can't just make citations up. All of this lives under `src/synthesis/`.

The system went through three orchestration generations, all sharing the same stage modules and `schemas`:

1. **Linear pipeline** (`synthesis.pipeline.run_synthesis`) — the original 10-stage straight-line function. Deleted after the migration; described below because the stages it defined are still the graph's nodes.
2. **Agentic controller** (`synthesis.controller.SynthesisController`) — a hand-rolled ReAct-style loop with explicit state and a logged decision trace. Also deleted; its design decisions (state shape, trace schema, decision policy) live on unchanged in the graph. See [§ A4.5](#a45-agentic-refactor-controller-state-and-decision-trace-design-history).
3. **LangGraph workflow** (`synthesis.graph`) — the current and only runtime, described in [Core architecture](#core-architecture-current) and in the [migration log](#langgraph-migration-2026-07) below.

The controller was my answer to the main review note. The first version looked like "agentic programming," but the hard part (deciding what to do next when you don't know what you will find) was buried inside a straight-line script. The refactor pulled that decision-making out into data you can actually look at — and the LangGraph migration then made the control flow itself declarative.

### Stage modules (one responsibility each)

These are the graph's building blocks; each is independently unit-tested with an injectable LLM caller.

| Module | Responsibility |
|--------|----------------|
| `synthesis.decompose` | LLM splits the question into 3-5 **keyword-style search queries** (terse technical terms, never sentences — question words poison scholarly search APIs); deterministic keyword-extraction fallback never raises |
| `synthesis.retrieve` | Fan out across A3 search tools; dedupe by DOI → version-stripped arXiv id → normalized title; **round-robin interleave preserving each provider's relevance order** so every sub-query contributes its top hits before the cap |
| `synthesis.fetch_parse` | Wrap `tool_fetch_and_parse_pdf` (auto-rewrites arXiv `/abs/` → `/pdf/`); abstract-only fallback when PDF fetch fails (`has_pdf=False`); truncates body to a paragraph boundary |
| `synthesis.rank` | Dependency-free TF-IDF + cosine ranks papers vs the question; `min_score` floor; returns new `ScoredPaper` copies (immutable inputs) |
| `synthesis.relevance` | **LLM relevance gate**: one JSON call scores every paper's title+abstract 0-10 against the question; papers below `relevance_keep_threshold` are dropped from the active set. TF-IDF can order papers but cannot *reject* one that merely shares generic ML vocabulary; this can. Fail-soft: LLM failure keeps all papers |
| `synthesis.claims` | One JSON-strict LLM call per paper for `{claim, evidence_quote, confidence}`; re-verifies each quote as a normalized substring of the source text; ungrounded claims get `grounded=False` and confidence halved |
| `synthesis.contradictions` | One LLM call across all claims; pydantic-validates each pair, drops unknown paper ids, drops self-contradictions, canonicalizes (A,B)/(B,A) duplicates; skips the LLM entirely when fewer than 2 papers have claims |
| `synthesis.prompt` | Deterministic builder: enforces citation format (`[Author et al. YEAR]`), word budget, abstract/claim/explanation length caps; returns the deduped `expected_citations` list the model is allowed to use |
| `synthesis.generate` | One LLM call (free-form text); returns a deterministic fallback string on any failure: never raises |
| `synthesis.validate_cites` | Regex extracts inline `[…]` citations; resolves each against `ScoredPaper.short_citation_key()`; emits `CitationCheck`s and a `valid_unique / total_unique` score |
| `synthesis.critic` | **Writer-critic reviewer**: raises checkable objections against a draft (verbatim-excerpt validated; hallucinated citations become deterministic auto-objections); powers the revision loop |
| `synthesis.reformulate` | LLM query rewrite when retrieval is thin (deterministic suffix fallback); feeds the adaptive-retrieval cycle |
| `synthesis.eval_harness` | Loads hand-labeled `EvalCase`s from `eval/cases.json`, scores saved or live `SynthesisResult`s on **claim faithfulness**, **citation hallucination rate**, and **contradiction coverage**, then writes `eval/results.json`. No LLM judge anywhere — all three metrics are computed by code |

### How it resists hallucination

Four layers, each tested on its own:

1. **Per-paper grounding (claims).** Every claim has to come with an exact `evidence_quote`. The extractor checks that the quote really shows up in the source text (it normalizes the text and also allows a 60-character prefix match). A claim that fails this check still survives, but its `confidence` is cut in half and it gets tagged `(UNGROUNDED)` in the prompt so the writer hedges it.
2. **Allowed citations only (prompt + validator).** The prompt lists the `expected_citations` and tells the model not to use anything else. The validator then pulls every `[...]` out of the generated text and checks it against that same list. Anything that does not match goes into `hallucinated_citations`, the score drops, and `SynthesisResult.to_markdown()` adds a warning note.
3. **The critic loop (fix, not just flag).** Validated objections — including deterministic auto-objections for every hallucinated citation — send the draft back to the writer with explicit instructions to fix each one, up to `max_revisions`. Detection alone leaves the error in the output; the loop removes it.
4. **Score penalty.** `confidence_score = citation_validity × grounded_claim_fraction`, clamped to [0, 1]. If any layer fails, the score goes down. The eval harness averages this across a labeled dataset. The CLI command is `python main.py eval-synthesis`.

### Integration with A3

- SQLite table `synthesis_runs` (schema **v4**): `id, session_id, question, review_text, result_json, confidence_score, contradictions_found, hallucinated_count, created_at`. Helpers in `db.queries` (`insert_synthesis_run`, `list_recent_synthesis_runs`, `get_latest_synthesis_run_for_session`).
- Agent-callable tools `tool_synthesize_literature_review` and `tool_get_review_context` in `src/tools/synthesis_tools.py`, registered in `tools.tools_registry.TOOL_SPECS` (count **21**) and listed in `GUARDRAILS_AUTONOMOUS_TOOLS`.
- CLI commands in `main.py`: `synthesize "..."` (with `--resume`, `--word-budget`, `--top-n`, `--source` repeatable, `--output`, `--session-id`, `--verbose`), `synth-history`, and `eval-synthesis` (`--live` to generate missing cases).
- A pydantic-only module `synthesis.schemas` for cross-stage types (`ResearchQuestion`, `ScoredPaper`, `ClaimRecord`, `ContradictionPair`, `CitationCheck`, `SynthesisResult`) so each stage can be unit-tested in complete isolation.
- A tiny OpenAI wrapper `synthesis.llm.call_json` / `call_text` (uses the existing `OPENAI_API_KEY` + `OPENAI_MODEL`: no new config). Always asks for `response_format={"type":"json_object"}` for structured stages; raises `SynthesisLLMError` on parse failure so the calling stage can fail soft.

### Trade-offs

- **More moving parts** than a single "summarize these abstracts" prompt. The upside is that each stage is small, type-checked, and easy to swap out.
- **Several LLM calls** on the normal path (1 decompose, 1 relevance gate per retrieval round, N claim extractions, 1 contradictions, 1 generate, 1+ critique). That is the real cost. The parallel fan-outs hide most of the latency, checkpointing makes an interrupted run resumable instead of lost, and the per-paper claim calls fail soft, so a flaky provider only loses one paper's claims instead of the whole run.
- **TF-IDF + an LLM gate instead of embeddings** for relevance. TF-IDF is deterministic and dependency-free for *ordering*; the gate adds semantic *rejection* for one cheap JSON call per retrieval round. An embedding ranker could replace both later without changing the public surface.

### Ownership split (me vs assistant)

- I defined the proposal, the staged architecture, and the hallucination-resistance properties (verbatim grounding + whitelisted citations + confidence penalty).
- The assistant implemented the stage modules, the orchestration, the agent-callable tools, the CLI commands, the SQLite migration, and the mocked tests.

---

## A4.5 Agentic refactor: controller, state, and decision trace (design history)

> The `controller.py` described here was **deleted in the LangGraph migration** — the graph subsumed it at behavior parity. This section stays because the decisions made here (the state shape, the trace schema, the decision policy, the relevance fixes) are exactly what the graph runs on today; only the loop mechanics changed.

The linear pipeline gives an answer but it can't explain itself. It runs the same stages in the same order no matter what it finds. The agentic layer swapped that straight line for a loop that reads its own state, picks the next action, and writes down every decision as data you can read back.

### First decision: keep "text source" and "claim verdict" separate

Before writing any controller code, I locked one schema decision because everything else depends on it. The source of a paper's text and whether a claim checks out are two different things, so they get two different fields. If I had jammed them into one enum (like `grounding_source` being one of `full_text`, `abstract`, `unverified`), I would lose the most important signal: "we had the full PDF and the quote still did not verify."

- `ScoredPaper.text_tier: Literal["full_text", "abstract", "none"]`. A fact about the paper: the best text we actually have for it.
- `ClaimRecord.grounded: bool`. The verdict: did the exact quote check out against the source text.
- `ClaimRecord.grounding_tier: Literal["full_text", "abstract", "corroborated", "none"]`. How the claim is supported right now. This can go up during a run (for example `none` to `corroborated` after a successful gap hunt).
- `ClaimRecord.supporting_paper_id`. Which paper backs the claim (its own paper normally, or a different one after corroboration).

I added stable ids for the same reason: it is easier to point at data by id than by list position. So `ClaimRecord.claim_id` and `ContradictionPair.contradiction_id` (both default to `uuid4().hex`). In the LangGraph migration those same stable ids became the upsert keys for the state reducers — the decision paid for itself twice.

### `synthesis.trace`: the decision trace

A `DecisionStep` is the main building block. The same object is the runtime trace, the demo timeline, and the evidence the eval harness reads.

- **Typed params.** `StepParams` is a union keyed by `kind`, one model per action type (`DecomposeParams`, `SearchParams`, `ReformulateParams`, `FilterRelevanceParams`, `ExtractClaimsParams`, `DetectContradictionsParams`, `GapHuntParams`, `ResolveConflictParams`, `SynthesizeParams` — now with a `revision` counter — and `CritiqueParams`). This lets the eval harness group by action type without parsing text, and a `model_validator` checks that `params.kind == action`.
- **Cause links.** `parent_step_id` points a step back at the step it was reacting to. A gap hunt points back at the `extract_claims` step that produced the ungrounded claim, and a conflict resolution carries the `contradiction_id` it worked on. Under the parallel fan-outs, causality is branch-shaped: all branches of one stage share the dispatching step as their parent.
- **Effect.** `StepEffect` records what changed (`added_paper_ids`, `claim_ref`/`claim_refs`, `contradiction_ids`, `tier_before`/`tier_after`, `resolved_conflict`, `gap_ref`), so the UI can say "upgraded claim C from none to corroborated using X".
- **Numbers.** `llm_calls` and `duration_ms` per step feed the baseline comparison.
- **Phases, then and now.** Originally each step was created `pending` at decision time and completed after execution — necessary because one loop interleaved deciding and executing. In the graph, edges decide and nodes execute, so steps are born completed. The schema kept both phases for backward compatibility with persisted traces.

### `synthesis.state`: the working memory

`SynthesisState` is everything the agent knows at one moment: `question`, `sub_queries`, `papers`, `active_paper_ids`, `claims`, `contradictions`, `gaps`, `objections`, `review_text`, `citation_checks`, and the full `trace`. Two rules, both carried over from the schema work:

- **Don't store what you can compute.** `current_step`, `status`, `reformulation_count`, `open_gaps`, `grounded_fraction()`, `grounded_by_tier()`, and `citation_validity()` are all derived from the raw lists, not stored next to them. The graph extends this: which queries have been searched and how many revisions have run are derived from the trace, and the `reformulation_cap` terminal reason is computed at synthesis time rather than stored mid-run.
- **Don't mix separate signals.** Citation validity and grounded fraction are reported on their own. The old single `confidence_score` is only built inside `to_result()`, and only so the `synthesis_runs` table keeps working.

`Gap` is one model with a `kind` field (`ungrounded_claim` or `coverage`) and a simple lifecycle (`open`, `hunting`, then `resolved` or `flagged_unverified`). A failed hunt is just the existing fields combined, not a new redundant flag. `SynthesisState` is the mutable thing the workflow edits while it runs; `SynthesisResult` is the frozen output you hand off. Different uses, different lifetimes.

### The decision policy (formerly `next_action`, now the graph's routers)

The controller's `next_action(state, config)` policy function read the state and returned the next step: decompose, then search every angle, then reformulate until a cap, then synthesize. That exact policy survives as the graph's conditional-edge routers (`route_after_search`, `route_after_gate`, `route_after_critique`) — same decisions, native idiom. The feature loops became nodes and fan-outs:

| Behavior | Then (controller method) | Now (graph) |
|----------|--------------------------|-------------|
| Adaptive retrieval | `run_retrieval_loop` — search, rank, reformulate up to `max_reformulations` | `plan_search` fan-out → `rank_pool` join → coverage router → `reformulate` cycle |
| Relevance gating | `run_relevance_gate_loop` | `filter_relevance` node; a gutted active set re-enters the reformulate cycle |
| Claim extraction | `run_claims_loop`, one traced step per paper | `extract_one` parallel branches, one per active paper |
| Contradiction detection | `run_detect_contradictions_loop` | `detect_contradictions` node |
| Gap hunting | `run_gap_detection_loop` | `hunt_one` parallel branches, one per ungrounded claim |
| Conflict resolution | `run_conflict_resolution_loop` | `resolve_one` parallel branches, one per contradiction |
| Synthesis | `run_synthesize_loop` | `synthesize` node + the writer-critic loop |

### `synthesis.trace_view` + `streamlit_app.py`: decision trace UI

`trace_view` is a plain formatting layer with no Streamlit imports (`state_metrics`, `trace_rows`, `summarize_effect`, `claim_rows`, `paper_rows`, `contradiction_rows`, `gap_rows`), so the display logic can be tested without running a UI. `streamlit_app.py` runs the graph and shows two panels: the final review on the left, and the decision trace as a colored timeline on the right (color-coded result badges, the cause and effect of each step, and timing). Below that are tabs for papers, claims, contradictions, and gaps. This is the demo: a reviewer reads the right panel and sees exactly why the left panel came out the way it did.

### Retrieval relevance: fixing it at every layer

An early run gave a bad result: a long-context-retrieval question returned an astrophysics paper. When I dug in, relevance was being lost in several places, so the fix is layered too:

1. **Query construction** (`synthesis.decompose` + `tools.search_tools`). The decomposer produces keyword queries, never sentences — arXiv's parser matches every raw token, so "What are the competing approaches to…" matched *what* and *competing* instead of the topic. The arXiv tool then sanitizes free text into `all:term AND all:term …` boolean queries and automatically broadens to `OR` when the precise query returns nothing.
2. **The provider sort** (`tools.search_tools`). arXiv sorts by `relevance`, falling back to `submittedDate` only when the caller sets a date range.
3. **The merge** (`synthesis.retrieve`). Results interleave round-robin across sub-queries preserving each provider's relevance order — the old citation-count/recency re-sort let the newest papers win regardless of topic.
4. **Coverage** (the graph's routers). Every decomposed angle gets searched — the original controller only ever searched the *last* sub-query, which was the single biggest source of off-topic reviews.
5. **Semantic rejection** (`synthesis.relevance`). The LLM gate drops papers that share vocabulary but not topic; a gutted post-gate set re-enters the reformulation cycle instead of producing a thin review silently.
6. **The synthesis input** (the `synthesize` node). Right before writing, the active papers get ranked one more time with a `min_relevance_score` floor, and claims/contradictions are cut down to the survivors. A weak result turns into an honest `reformulation_cap` instead of an off-topic review.

### Making citation keys unique

Two different papers by the same first author in the same year produce the same `short_citation_key` (two `[Liu et al. 2025]` papers). This used to hurt twice: evidence selection would drop one of the papers, and the validator would mark any key that pointed to more than one paper as hallucinated, so a real citation got called fake. `synthesis.schemas.assign_citation_keys(papers)` fixes this by looking at the whole list at once: when keys collide, it adds a letter to the year (`[Liu et al. 2025a]`, `[Liu et al. 2025b]`) in order. The prompt builder and the validator both call it on the same paper list, so they always agree on the keys.

### Ownership split (me vs assistant)

- I ran the design review and made the hard calls: splitting text source from claim verdict, the typed `DecisionStep`, stable ids, the `SynthesisState` shape, how conflict resolution stays grounded, and the call to wire the loops together with an integration test before adding synthesize.
- The assistant wrote the `trace`, `state`, `controller`, and `trace_view` modules, each feature loop, the Streamlit UI, and the mocked tests.

---

## LangGraph migration (2026-07)

The hand-rolled controller proved out the architecture; the migration swapped its orchestration for LangGraph while keeping every stage module, the typed trace, and the eval harness. Each phase landed with the full suite green and was verified against live APIs, not just stubs.

### Phase log

**Phase 1 — linear parity graph.** `synthesis/graph.py`: a `StateGraph` whose nodes wrap the existing stage functions, linear edges, per-node completed `DecisionStep`s, dependency injection via the config channel. `SynthesisState` gained its reducers (upsert-by-id via the stable ids from A4.5) and the `active_paper_ids` evidence/selection split. The controller stayed the production path until parity was proven live.

**Phase 2 — the graph takes over.** Coverage routers and the reformulation cycle (the old `next_action` policy as conditional edges); a post-gate router so a relevance-gutted working set re-enters retrieval while budget remains — live effect on the RAG-vs-fine-tuning question: 1 relevant paper and 0 contradictions became 3 papers and 2 contradictions. All three entry points (CLI, chat tool, Streamlit) rewired to `run_graph_synthesis*`; `controller.py` and `pipeline.py` deleted.

**Phase 3 — parallel fan-out.** Send-API map-reduce for searches, claim extractions, gap hunts, and conflict resolutions; dispatcher conditionals return `Send` batches; join nodes (`rank_pool`, `consolidate`) are the sole writers of `active_paper_ids`; `max_concurrency` (default 4) bounds the branches. Live timing: the flagship question dropped from ~2.5-3 minutes to ~70 seconds wall clock, with all four arXiv searches firing within 3 ms of each other.

**Phase 4 — checkpointing + resume.** `SqliteSaver` persists every superstep to `data/checkpoints.sqlite3` under a printed run id; `synthesize --resume RUN_ID` continues an interrupted run without re-executing completed stages, including finished parallel branches. Verified live by force-killing a run 45 seconds in and resuming it to a finished review. Progress streaming skips checkpointed history on resume.

**Phase 5 — chat agent on the prebuilt graph.** `AgentManager` rebuilt on `langchain.agents.create_agent` (the maintained successor to `create_react_agent`). Permission policy moved inside guarded tool wrappers; confirmation-tier tools pause the graph with `interrupt()` and resume via `Command`; audit logging fires exactly once per decision, positioned after the interrupt point. `respond()` kept its synchronous contract — conversation layer, CLI, and transcripts unchanged. Tests inject a scripted `BaseChatModel` (`model_instance`) and stub tools (`tool_overrides`) instead of patching internals. Verified live: parallel multi-source search with graceful per-tool failure, and an interrupt-confirmed reading-list write through the real CLI.

**Phase 6 — writer-critic loop.** `synthesis/critic.py` plus `critique`/`revise` nodes cycling after `synthesize`, capped by `max_revisions`. Objections are validated by code (verbatim-excerpt check; deterministic auto-objections for hallucinated citations) before they can trigger a revision; fail-soft in both directions. Verified live: the critic ran in production and accepted a clean draft ("draft is faithful to the evidence", 0 hallucinated citations).

### What deliberately did not change

- The stage modules and their injectable-LLM test seams — the graph orchestrates them, it does not absorb them.
- The `DecisionStep` trace schema, the Streamlit UI, and the eval harness — graph runs produce the same legible record the controller did.
- The guardrail policy surface (`AppSettings` tool tiers, blocked intents, output guardrail, filesystem boundary) — only the enforcement point moved (into the tools).
- `AgentManager.respond()`'s contract and the CLI UX.

### Trade-offs of the migration

- **A framework dependency where there was none.** Accepted for what it bought: checkpoint/resume, bounded parallelism, and native interrupts are all things the hand-rolled controller would have had to grow bespoke versions of.
- **Nondeterministic within-stage trace order.** Parallel branches merge in arrival order; tests compare per-stage results as multisets, and causality is branch-shaped rather than chain-shaped inside a fan-out.
- **Reducer discipline is now load-bearing.** Any new state field must choose replace vs. upsert semantics consciously, and parallel writers of replace-semantics fields are a graph-construction error. The evidence/selection split exists precisely to keep that rule easy to follow.
