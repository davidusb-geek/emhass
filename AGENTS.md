---
name: emhass-agents
description: Vendor-neutral rules for AI coding agents working on EMHASS source.
---

<!-- Last verified against upstream/master @ 6537c47, 2026-04-30 -->

Rules for AI coders (Claude Code, Cursor, Aider, Copilot, Codex) on EMHASS source. Complements `docs/develop.md` (human canon); no duplication. Where `develop.md` covers topic, this links + adds AI-specific constraints.

*Humans driving agent: see [`docs/develop_ai_coders.md`](docs/develop_ai_coders.md) for contributor companion.*

## Repository layout

- `src/emhass/` — `optimization.py`, `forecast.py`, `retrieve_hass.py`, `web_server.py`, `command_line.py`, `utils.py`.
- `tests/` — pytest suite.
- `docs/` — Sphinx source. Start: `develop.md`; examples: `study_cases/`.
- `data/` — `config_defaults.json`, `associations.csv`.
- `src/emhass/static/` — web UI assets incl. `param_definitions.json`.

## Section 1 — Canonical commands

Full setup + workflow in `docs/develop.md` (Method 1: `uv` venv, Method 2: DevContainer, Method 3: Docker). Read first.

Quick-recall:

| Action | Command |
|---|---|
| Run tests | `pytest tests/` |
| Sync dev deps | `uv sync --extra test` |
| Build docs | `sphinx-build -b html docs docs/_build` (configured in `docs/conf.py`) |
| Lint | `uvx ruff check .` (enforced via `.github/workflows/code-quality.yml` on every PR) |

Tech stack (verify versions in `pyproject.toml` before assuming API):

| Component | Version source |
|---|---|
| Python | `pyproject.toml` `requires-python` |
| Optimisation | CVXPY (pin in `pyproject.toml`) |
| Web | Quart + Uvicorn |
| Tests | pytest |

## Section 2 — Pipeline map

Three optimisation modes via `command_line.py`. Shared: input-prep + publish. Body differs by mode. Core: one method on `Optimization`.

| Phase | Symbol |
|---|---|
| Input preparation | `command_line.py::set_input_data_dict` |
| Mode entry, perfect forecast | `command_line.py::perfect_forecast_optim` |
| Mode entry, day-ahead | `command_line.py::dayahead_forecast_optim` |
| Mode entry, rolling MPC | `command_line.py::naive_mpc_optim` |
| Optimisation core (LP build and CVXPY solve) | `optimization.py::Optimization.perform_optimization` |
| Publish | `command_line.py::publish_data` |

Stage instrumentation: `stage_timer(stage_times, "<label>", logger)`. Five active labels:

- `"pv_forecast"`
- `"load_forecast"`
- `"price_prep"`
- `"optim_solve"`
- `"publish"`

Grep `'stage_timer.*"<label>"'` for live call site. Labels stable; line numbers not.

## Section 3 — Don't-touch rules

Five invariants. Easy to break, hard to detect in CI.

1. **`action_logs.txt` line format.** `web_server.py` parses lines by splitting on first whitespace, comparing leading token to `"ERROR"`. Format change (prefix, JSON, structured log) silently breaks UI error reporting.

2. **Logger handler accumulation in `utils.get_logger`.** Attaches handler unconditionally every call. Duplicate calls → duplicate log lines → masked failures. Avoid. Guard changes need maintainer coordination (CLI + web both call this).

3. **Two parallel logging subsystems.** CLI: `utils.get_logger`. Web: `app.logger`. Touch both or neither. Partial migrations break downstream log consumers.

4. **`param_definitions.json` is a structured surface.** Additive only. Rename/remove/type-change breaks config UI + external tooling. New entries OK; mutations need migration plan + maintainer review.

5. **Optimisation-result DataFrame columns and units.** `opt_res_*` → `opt_res_latest.csv`, HA sensors, external bridges. Column renames + unit changes = breaking. Additive columns OK. Current `opt_res_latest.csv` columns = de-facto schema.

## Section 4 — Maintainer scope corridors

From public maintainer statements. Cite source if questioned.

- **Threat model** (#808): security scope = code injection, not auth bypass/data leakage. In-memory-read endpoints OK; filesystem/DB/shell endpoints need maintainer sign-off.
- **EMHASS scope** (#789): MILP optimiser. Vehicle APIs, OCPP, EVCC, charger modulation → integration layer, not core.
- **Glue layer agnostic.** Node-RED, MQTT, HA, generic automations = equivalent. No HA-specific code in core.
- **Zero-config default must keep working.** Add-on starts + produces sensible optimisation on defaults after every change.

## Section 5 — Limits and gotchas (read this if you are an AI coder or working with one)

AI finds code + candidates. Domain experts decide bug vs design. 2026-04-26 audit: 8 findings, 4 PR-able, 4 issue-first. Skip human-in-loop → ~50% noise.

**Issue first, not PR, when:**

- Behaviour changes visibly (output values, log format, error messages).
- Magic constant/sentinel might be intentional (`=0` = no constraint? negative = disabled?).
- Condition looks wrong but may encode domain convention (AC/DC power; charge/discharge sign).
- Change touches `optimization.py`, `retrieve_hass.py`, or `forecast.py` beyond ~3 lines.

**Verify before done:**

- Sign conventions (`P_grid > 0` = import or export? Check, don't assume).
- Units (HA scales SOC by 100; CSV uses 0..1).
- Test reproducer present for any behaviour-change PR.
- Smoke-test (`docker compose up` + browser config page) if schema or `web_server.py` changed.

**No refactor without issue:**

- Restructuring `optimization.py` (3000+ lines) without RFC issue → rejected.
- Renaming public API params breaks downstream; needs migration path.
- New dependencies: issue first.

**Add parameter:** four-step workflow in `docs/develop.md` (`associations.csv` + `config_defaults.json` + `param_definitions.json` + `OptimizationCacheKey`, optional `check_def_loads`). Skip step → breaks something.

**Change default value (existing param):** `src/emhass/static/data/param_definitions.json` first — source of truth. Align `src/emhass/data/config_defaults.json` to match. See `docs/develop.md` § Changing default values.

**External forecast feed alignment.** `runtimeparams` handlers for `pv_power_forecast`, `load_power_forecast`, `load_cost_forecast`, `prod_price_forecast` tolerate length/frequency/timezone differences — day-ahead feeds publish 24h not 48h, padded gracefully. Tolerant ≠ silent: log clearly when alignment happens. Silent shifts → wrong-but-valid-looking plans (`optim_status: Optimal`, every timestep offset by N).

**Common AI hallucinations:**

- Confusing `param_definitions.json` (GUI schema, SoT for defaults) with `config_defaults.json` (headless loader fallback).
- Inventing CVXPY APIs absent in pinned version.
- Forgetting `command_line.py` entry points (`set_input_data_dict`, `perfect_forecast_optim`, `dayahead_forecast_optim`, `naive_mpc_optim`, `publish_data`) are `async def` → writing sync wrappers.

## Section 6 — Behavioral guardrails
Four rules. Reduce wrong-assumption drift, scope creep, dead-code regression.
1. **Think first.** Surface assumptions before code. Multiple interpretations → ask, never silently pick. Simpler path exists → say so. Unclear → stop, name the confusion.
2. **Simplicity.** Min code that solves stated problem. No bonus features. No abstractions for single-use. No error handling for impossible cases. 200 lines where 50 work → rewrite.
3. **Surgical.** Touch only requested code. No "while I'm here" refactors. Match existing style. Remove only imports/vars YOUR change orphaned, never pre-existing dead code unless asked.
4. **Goal-driven.** Task → verifiable criteria. "Add validation" → "tests for invalid inputs pass". "Fix bug" → "reproduce-test passes". State plan as numbered steps, each with verify check.

## Section 7 — Conventions

- **Docs:** soft Diátaxis (tutorials, how-tos, reference, explanation). Not strict four-quadrant. Worked examples in `docs/study_cases/`.
- **Commits:** prefix `fix`/`docs`/`feat`/`chore` per maintainer practice.

## Section 8 — Where to find more

- [`docs/develop.md`](docs/develop.md) — canonical dev guide (fork, venv, DevContainer, Docker, adding params, PR). Read first.
- [`llms.txt`](https://emhass.readthedocs.io/en/latest/llms.txt) — Sphinx routing manifest. Not in source tree; built per Sphinx run, served from Read the Docs.
- [`docs/study_cases/`](docs/study_cases/) — worked examples per persona.
- [Project board](https://github.com/users/davidusb-geek/projects/2) — coordination + scope corridors.
