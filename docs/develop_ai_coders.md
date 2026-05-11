# AI-coder contributor onboarding

Companion to [`docs/develop.md`](develop.md) (humans, general) and [`AGENTS.md`](../AGENTS.md) (vendor-neutral rules for AI agents). Read develop.md first if new to EMHASS.

This file teaches the **human driving the AI agent**. AGENTS.md teaches the agent. develop.md teaches the human contributing without an agent. Three audiences, three docs, no overlap.

EMHASS landmines AI tools won't flag without explicit prompt: sign conventions, SOC scaling, MILP infeasibility, `q_input_start=0`, dual logger, `OptimizationCacheKey`, source-resolve discipline. Each section below addresses one.

## 1. AI-tool setup

### 1a. Claude Code (primary, tested-against)

Native tools cover context-loading on-demand. No pre-pack step.

| Action | Approach |
|---|---|
| Load file | `Read` tool |
| Find files | `Glob` pattern |
| Search content | `Grep` regex |
| Run tests | `Bash`: `pytest tests/` |
| Lint | `Bash`: `uvx ruff check .` |
| PR ops | `Bash`: `gh pr create`, `gh pr view`, `gh pr edit` |
| Format | `Bash`: `uvx ruff format .` |

Per-task agent choice:
- Refactor / implement → general agent with `Edit` + `Write`
- Code review → `code-reviewer` agent (read-only)
- Codebase exploration → `Explore` agent or `Glob` + `Grep` direct

Public EMHASS-specific skill plugin: not yet published. Until one ships, hand-instruct the agent from AGENTS.md.

### 1b. Cursor (untested — contribution welcome)

Conventions seen in community:
- `@file` mention to add context to chat
- `pytest tests/` passes locally before commit
- `uvx ruff check .` clean before push

PR adding a tested Cursor-setup recipe for EMHASS welcome.

### 1c. Aider (untested — contribution welcome)

Conventions seen in community:
- `/add <file>` to add context
- `/test` and `/lint` if configured in `.aider.conf.yml`
- Same pytest + ruff baseline

PR adding a tested Aider-setup recipe welcome.

## 2. Decision-tree: issue-first vs PR-direct

Default mental model:

```
What kind of change?
│
├── Pure docs / typo / wording                  → PR direct
├── Doc reorg / new doc / structural rename     → Issue or Discussion first
├── Bug fix < 10 lines, no behavior change      → PR direct (reproducer in body)
├── Bug fix ≥ 10 lines OR behavior change       → Issue first → then PR
├── New feature / new endpoint / new param      → Issue first (always)
└── Refactor (no behavior change)               → Issue first if > 50 LOC, else PR direct
```

Real-PR examples:

- **PR-direct (small docs / typo):** [#817](https://github.com/davidusb-geek/emhass/pull/817) (regression_model typo, 2 lines), [#814](https://github.com/davidusb-geek/emhass/pull/814) (broken doc link).
- **Issue-first (structural new doc):** [#835](https://github.com/davidusb-geek/emhass/pull/835) filed [issue #828](https://github.com/davidusb-geek/emhass/issues/828) first for the plan-output schema doc.
- **Discussion-first variant (corridor-aligned):** [#836](https://github.com/davidusb-geek/emhass/pull/836) (Cookbook scaffold, [Discussion #824](https://github.com/davidusb-geek/emhass/discussions/824) approval first); [#831](https://github.com/davidusb-geek/emhass/pull/831) (AGENTS.md introduction, [Discussion #808](https://github.com/davidusb-geek/emhass/discussions/808) corridor first).
- **Cautionary (no issue first, got pushback):** [#830](https://github.com/davidusb-geek/emhass/pull/830) param_definitions defaults. No issue filed. Maintainer review revealed direction disagreement (`config_defaults.json` as source vs `param_definitions.json` as source). Result: review round-trip. One-issue-comment first would have surfaced the direction-decision before code edits.
