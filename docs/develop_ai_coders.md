# AI-coder contributor onboarding

Companion to [`docs/develop.md`](develop.md) (humans, general) and [`AGENTS.md`](../AGENTS.md) (vendor-neutral rules for AI agents). Read develop.md first if new to EMHASS.

This file teaches the **human driving the AI agent**. AGENTS.md teaches the agent. develop.md teaches the human contributing without an agent. Three audiences, three docs, no overlap.

EMHASS landmines AI tools won't flag without an explicit prompt: sign conventions, SOC scaling, MILP infeasibility, `q_input_start=0`, dual logger, `OptimizationCacheKey`, source-resolve discipline. Each section below addresses one.

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

PR adding a tested Cursor-setup recipe for EMHASS is welcome.

### 1c. Aider (untested — contribution welcome)

Conventions seen in community:
- `/add <file>` to add context
- `/test` and `/lint` if configured in `.aider.conf.yml`
- Same pytest + ruff baseline

PR adding a tested Aider-setup recipe is welcome.

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

## 3. What AI won't tell you about EMHASS

Seven landmines. AI tools won't flag any of these without explicit prompt.

### Sign conventions
`P_grid`, `P_batt`, `P_PV` are unsigned in DataFrame columns. AI tools guess direction from the variable name. The MILP constraint form locks the sign — variable name does not. Verify against the power-balance constraint in `src/emhass/optimization.py` (battery and grid blocks around the `param_soc_init` plumbing). Trace constraint form, not name.

### SOC scaling trap
`SOC_opt` is fraction (0..1) in the DataFrame and the CSV export. Multiplied by 100 only at the MQTT publish step (`src/emhass/command_line.py:2329`). Home Assistant entity shows percent; raw CSV row shows fraction. Symmetric trap on input: `soc_init` runtime param expects fraction; most sensors publish percent. See `docs/plan_output_schema.md` once PR [#835](https://github.com/davidusb-geek/emhass/pull/835) merges for the full output-side story.

### MILP infeasibility
"Infeasible" status means the constraint set has no solution. AI tools propose fixes by relaxing arbitrary constraints. Don't. The symptom (infeasibility report) hides the actual wrong constraint. Bisect the constraint set, find the contradiction. Real case: PR [#785](https://github.com/davidusb-geek/emhass/pull/785) traced `q_input_start=0` thermal-battery boundary that made the heat-input schedule unsolvable.

### `q_input_start=0` thermal-battery landmine
Heat-input variable at the first timestep `q_input_start=0` creates infeasibility when thermal-battery state requires non-zero input at t0. Symptom: solve fails with "Infeasible". Cause: initial-state constraint placement. PR [#785](https://github.com/davidusb-geek/emhass/pull/785) carries the fix and the discussion. AI tools that skim `optimization.py` won't catch this — read PR #785's diff before adding thermal-battery-adjacent code.

### Dual logger subsystems
EMHASS has two logger setups: CLI (`src/emhass/command_line.py`) and Web (`src/emhass/web_server.py`). Both substantive (90+ and 70+ logger calls — grep `logger\.`). Touch both or none. AI tools that "improve logging" in one file break log-format parity with the other.

### `OptimizationCacheKey` 4-step add-a-param workflow
Adding a new optimisation parameter requires 4 edits per `docs/develop.md`: (1) `src/emhass/data/config_defaults.json`, (2) `src/emhass/static/data/param_definitions.json`, (3) optim helper signature in `command_line.py`, (4) the cache-key tuple (`OptimizationCacheKey` dataclass declared at `src/emhass/command_line.py:108`). AI tools regularly forget step 4 — silent cache-miss-explosion on every solve. Grep `OptimizationCacheKey` in `src/emhass/` before adding params; verify your new param is in the tuple.

### Source-resolve discipline
Ambiguous types / signs / units / conventions: trace upstream code first. "Ask maintainer" is last resort. Real precedent: PR [#835](https://github.com/davidusb-geek/emhass/pull/835)'s sign-convention questions self-resolved from `optimization.py` MILP constraints rather than punting to the maintainer. Audit-source-ambiguity does not have to propagate into the PR.

## 4. Self-check pre-PR

Run through before opening. Each item has a *why* — skip the item only if the *why* genuinely does not apply.

- [ ] Issue filed if behavior change? *Why: surfaces direction-disagreement before code edits; see §2 cautionary #830.*
- [ ] `pytest tests/` passes locally? *Why: CI runs the same suite; local-fail = CI-fail = wasted review cycle.*
- [ ] `uvx ruff check .` clean? *Why: ruff is enforced via `.github/workflows/code-quality.yml`; a red lint blocks merge.*
- [ ] Sign conventions verified (if PR touches power / SOC / cost variables)? *Why: column names do not encode sign; see §3 sign-conventions landmine.*
- [ ] One concern per PR (scope discipline)? *Why: bundled PRs invite scope-objection on one part and block the whole PR.*
- [ ] Issue or Discussion linked in PR body if applicable? *Why: makes review context one-click; saves maintainer time.*
- [ ] Reproducer in body if behavior-change fix? *Why: lets the maintainer confirm the bug, not just the patch.*
- [ ] Maintainer-scope-corridors checked? ([Discussion #808](https://github.com/davidusb-geek/emhass/discussions/808) Layers, [Discussion #789](https://github.com/davidusb-geek/emhass/discussions/789) MILP scope) *Why: out-of-corridor PRs land in indefinite review limbo.*

## 5. Red flags: stop and ask

Patterns that mean stop, file an issue, do not PR:

- "AI says this is a bug but I can't explain why in my own words." → Don't trust. File issue with the question. Maintainer or community will explain or correct.
- "I can't tell what unit this number is in (W vs kW, fraction vs percent, UTC vs local)." → File issue. Document the ambiguity for the next person.
- "I don't know which subsystem owns this concern (optim vs forecast vs retrieve_hass)." → File issue. Don't pick blindly — cross-subsystem PRs carry heavy review-friction.
- "Test passes but I don't trust the test." → Add a counter-example. If still ambiguous, file issue with the counter-example.
- "AI generated 200+ LOC and I haven't read it line-by-line." → Stop. Read every line before commit. Unreviewed AI-generated code is a future-bug source.
- "AI proposed relaxing a constraint to make the solve feasible." → Stop. See §3 MILP infeasibility — relaxation masks the wrong-constraint cause. File issue with the infeasible model dump.
- "The fix is in a subsystem I don't normally touch and the diff feels speculative." → File issue describing the symptom and the speculative diff. Let the subsystem owner weigh in before you push.

## 6. Help resources

- [GitHub Discussions](https://github.com/davidusb-geek/emhass/discussions) — questions, ideas, use-case sharing
- [`docs/develop.md`](develop.md) — general developer guide (Method 1 venv with `uv`, Method 2 DevContainer, Method 3 Docker)
- [`AGENTS.md`](../AGENTS.md) — agent-side rules (read by AI tools; humans read to understand agent constraints)
- Maintainer scope corridors: [Discussion #808](https://github.com/davidusb-geek/emhass/discussions/808) (Layers 1-3, zero-config default), [Discussion #789](https://github.com/davidusb-geek/emhass/discussions/789) (EMHASS = MILP core, glue layer separate)
- Issue templates: `.github/ISSUE_TEMPLATE/`

---

Cross-references: [`docs/develop.md`](develop.md), [`AGENTS.md`](../AGENTS.md), [`CONTRIBUTING.md`](../CONTRIBUTING.md).

AI-tool-coverage gaps: Cursor (§1b) and Aider (§1c) are untested stubs. PRs adding tested-against patterns welcome.
