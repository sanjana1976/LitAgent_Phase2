# Post-review feedback — Research Paper Analyzer




## Review change 1/3 — Runtime architecture and pre-LLM guardrails

**What changed**

- **`AgentManager`** is the only path used by `main.py chat` (LangChain tool loop, permissions, output guardrails).
- **`ResearchPaperAgent`** documented as a **legacy** `generate_reply()` wrapper without guardrails or confirmations.
- **`ConversationManager`** only loads/persists history; it does not call the LLM.
- **Input intent blocking** (`check_blocked_intent`) runs on user text **before** any model call (delete list/paper, paywall bypass, modify PDF, fabricate metadata).
- **Compact session context** (`context_snapshot`) injected as a short `SystemMessage` instead of replaying full metadata dumps each turn.

**Why**

- Review prep and code reading showed confusion about which class actually ran the REPL.
- Blocked requests should not consume model budget or add latency for a refusal the policy already knows.

**Trade-off:** Slightly more code surface (`AgentManager` + `ConversationManager`) for maintainability, not for raw speed.

---

## Review change 2/3 — CLI UX, quiet terminal, and output guardrail

**What changed**

- **`src/cli/welcome.py`:** bordered welcome banner, `help` text, session/model display.
- **Default quiet chat:** logging forced to **WARNING** unless `--verbose` or parent `--log-level` is set; `httpx` / `openai` loggers capped at WARNING.
- **`list my lists`:** REPL command calls `tool_list_all_lists` **directly** (no LLM).
- **Output guardrail** (`src/guardrails/output.py`): if all search tools in a turn returned empty and the model still emits BibTeX blocks in free text, reply is replaced with a fixed **`EMPTY_SEARCH_SAFE_REPLY`** (local check, no extra model call).

**Why**

- Review demo: terminal “random lines” (tool `INFO` logs) hurt readability.
- Reviewer 2 / empty-search probes: models invent `@article{...}` when APIs return nothing; prompt-only rules are insufficient.

**Trade-off:** Confirmation prompts for writes add one user interaction per mutating tool—intentional safety, not latency optimization.

---

## Review change 3/3 — Forward citations, tests, docs, and I/O cleanup

**What changed**

- **`tool_lookup_forward_citations`** (Semantic Scholar): structured “who cited this paper?” instead of guessing from the model.
- **Test suite expanded to 78 mocked tests** (guardrails, search providers, citations, agent loop, DB/sessions)—no live OpenAI in CI.
- **`design.md` / README / review docs** aligned to one architecture story.
- **CLI no longer writes `transcripts/*.json` each turn**; chat history stays in **SQLite**; `transcripts/` is manual/Cursor documentation only.

**Why**

- Review demo step 2 suggested forward-citation follow-up after picking a paper.
- Course/review readiness: documented guardrails, oracles for empty-search BibTeX, reproducible pytest.
- Author request: transcripts are personal agent logs, not duplicate REPL mirrors.

**Latency / cost / optimization**

**Trade-off:** Using `tool_lookup_forward_citations` adds an API call when the user asks for forward citations—that is **feature cost**, not a reduction—but it is cheaper and more reliable than hallucinating citation graphs in prose.


## What did not improve costs:

- **Forward citations** and **multi-source search** add external API calls when used—they improve **correctness**, not minimal API usage.
- **Confirmation gates** add human latency by design.
- **Deep analysis with OpenAI** still uses a separate completion when the key is set (with heuristic fallback only when appropriate).

---

## Summary table

| Review phase | Main deliverable | Primary win |
|--------------|------------------|-------------|
| **1/3** | `AgentManager`-first + pre-LLM blocks | Zero-token refusals; clearer architecture |
| **2/3** | Quiet CLI, banner, output guardrail, `list my lists` shortcut | Fewer tokens/no LLM on simple paths; shorter safe replies |
| **3/3** | Forward citations, 78 tests, doc alignment, no CLI transcript mirror | Targeted APIs vs. hallucination; cheaper CI; less per-turn I/O |


