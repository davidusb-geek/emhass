---
name: emhass-agents
description: Vendor-neutral rules for AI coding agents working on EMHASS source.
---

<!-- Last verified against upstream/master @ 6537c47, 2026-04-30 -->

This file documents rules for AI coding agents (Claude Code, Cursor, Aider, Copilot, Codex) working on EMHASS source. It complements `docs/develop.md` (canonical for humans) and does not duplicate its content. Where `docs/develop.md` already covers a topic, this file links and adds AI-specific constraints on top.

*Humans driving an agent on this codebase: see [`docs/develop_ai_coders.md`](docs/develop_ai_coders.md) for the contributor-side companion to this file.*

## Repository layout

- `src/emhass/` — core module: `optimization.py`, `forecast.py`, `retrieve_hass.py`, `web_server.py`, `command_line.py`, `utils.py`.
- `tests/` — pytest suite.
- `docs/` — Sphinx source. Start with `develop.md`; worked examples in `study_cases/`.
- `data/` — config defaults and schema (`config_defaults.json`, `associations.csv`).
- `src/emhass/static/` — web UI assets, including `param_definitions.json`.

## Section 1 — Canonical commands

Setup, environment variables, and the full developer workflow live in `docs/develop.md` (Method 1 Python venv with `uv`, Method 2 DevContainer, Method 3 Docker). Read that first.

Quick-recall for AI tools:

| Action | Command |
|---|---|
| Run tests | `pytest tests/` |
| Sync dev deps | `uv sync --extra test` |
| Build docs | `sphinx-build -b html docs docs/_build` (configured in `docs/conf.py`) |
| Lint | `uvx ruff check .` (enforced via `.github/workflows/code-quality.yml` on every PR) |

Tech stack (verify versions in `pyproject.toml` before assuming an API):

| Component | Version source |
|---|---|
| Python | `pyproject.toml` `requires-python` |
| Optimisation | CVXPY (pin in `pyproject.toml`) |
| Web | Quart + Uvicorn |
| Tests | pytest |

## Section 2 — Pipeline map

EMHASS exposes three optimisation modes from `command_line.py`. All share an input-prep step and a publish step; the body differs by mode. The optimisation core is one method on the `Optimization` class.

| Phase | Symbol |
|---|---|
| Input preparation | `command_line.py::set_input_data_dict` |
| Mode entry, perfect forecast | `command_line.py::perfect_forecast_optim` |
| Mode entry, day-ahead | `command_line.py::dayahead_forecast_optim` |
| Mode entry, rolling MPC | `command_line.py::naive_mpc_optim` |
| Optimisation core (LP build and CVXPY solve) | `optimization.py::Optimization.perform_optimization` |
| Publish | `command_line.py::publish_data` |

For finer-grained stage instrumentation, the codebase uses `stage_timer(stage_times, "<label>", logger)` blocks. Five labels are in active use:

- `"pv_forecast"`
- `"load_forecast"`
- `"price_prep"`
- `"optim_solve"`
- `"publish"`

Grep `'stage_timer.*"<label>"'` for the live call site at any time. The labels are stable across refactors; the line numbers are not.

## Section 3 — Don't-touch rules

These five invariants are easy to break by accident and hard to detect in CI.

1. **`action_logs.txt` line format.** The web server's error-detection logic in `src/emhass/web_server.py` parses each line by splitting on the first whitespace and comparing the leading token to `"ERROR"`. Any change to the log line format (extra prefix, structured-logging migration, JSON envelope) silently breaks error reporting in the UI.

2. **Logger handler accumulation in `utils.get_logger`.** The function attaches a handler unconditionally on every call. Calling it twice for the same logger name produces duplicated log lines, which has historically masked real failures by hiding them in scroll-back. Avoid duplicate calls; if a guard becomes appropriate, coordinate the change with the maintainer because both the CLI and the web path call into this function.

3. **Two parallel logging subsystems.** The CLI path uses `utils.get_logger`. The web path uses `app.logger` (the Flask logger). Logging changes touch both consistently or land in neither. Partial migrations leave the two paths emitting different formats and break log consumers downstream.

4. **`param_definitions.json` is a structured surface.** Additive changes only. Renaming a key, removing one, or changing its type contract breaks the configuration UI and any external tooling that reads the schema. New entries are fine; mutations need a migration plan and a maintainer-led review.

5. **Optimisation-result DataFrame columns and units.** The `opt_res_*` DataFrames are written to `opt_res_latest.csv`, published as Home Assistant sensors (`sensor.p_pv_forecast`, `sensor.p_load_forecast`, `sensor.p_batt_forecast`, etc., per `docs/publish_data.md`), and consumed by external bridges (Node-RED flows, third-party forecasters, regional market integrations). Column renames and unit changes are breaking changes; additive columns are fine. Treat the current `opt_res_latest.csv` columns as the de-facto schema until a formal schema doc lands.

## Section 4 — Maintainer scope corridors

These corridors come from public maintainer statements. Cite the source if a contributor questions them.

- **Threat model** (Discussion #808): the project's security envelope is code injection, not auth bypass or data leakage. Endpoints that read in-memory state are inside the corridor; endpoints that touch the filesystem, a database, or shell out are not, and need explicit maintainer sign-off.
- **EMHASS scope** (Issue #789): EMHASS is a MILP optimiser. Vehicle APIs, OCPP, EVCC, and direct charger modulation belong in the integration layer, not in core.
- **Glue layer is agnostic.** Node-RED, MQTT, Home Assistant, and generic automations are equivalent integration paths. Do not wire Home-Assistant-specific code paths into core.
- **Zero-config default must keep working.** The add-on must continue to start and produce a sensible optimisation with default configuration after every change.

## Section 5 — Limits and gotchas (read this if you are an AI coder or working with one)

AI coders find code locations and produce candidate changes. Domain experts decide whether something is a bug or design. A 2026-04-26 schema audit illustrates the split: of eight candidate findings, four were confirmed bugs and PR-able, four needed maintainer judgment and went issue-first. Skipping the human-in-the-loop step produces roughly fifty percent noise.

**File an issue, not a PR, when:**

- Behaviour changes in any visible way (output values, log format, error messages).
- A magic constant or sentinel might be intentional ("`=0` means no constraint?", "negative value treated as disabled?").
- A condition looks wrong but might encode a domain convention you do not know (AC vs DC stack power; charge vs discharge sign conventions).
- The change touches `optimization.py`, `retrieve_hass.py`, or `forecast.py` beyond about three lines.

**Always verify before claiming done:**

- Sign conventions (`P_grid > 0` means import? export? Check; do not assume).
- Units in the wild (Home Assistant scales SOC by 100; CSV uses 0..1; they differ).
- A test reproducer is present for any behaviour-change PR.
- Container or UI smoke-test (`docker compose up` plus the browser config page) if schema or `web_server.py` changed.

**Do not refactor without an issue:**

- Restructuring `optimization.py` (3000+ lines) without an architecture-RFC issue gets rejected.
- Renaming public API parameters breaks downstream consumers; needs a migration path.
- Adding new dependencies is coordinated via issue first.

**Adding a parameter:** follow the four-step workflow documented in `docs/develop.md` (`associations.csv` plus `config_defaults.json` plus `param_definitions.json` plus `OptimizationCacheKey`, optionally `check_def_loads`). Skipping any step breaks something.

**Changing a default value (existing parameter):** edit `src/emhass/static/data/param_definitions.json` first — it is the source of truth. Then align `src/emhass/data/config_defaults.json` to match. See `docs/develop.md` § Changing default values.

**External forecast feed alignment.** The `runtimeparams` handlers for `pv_power_forecast`, `load_power_forecast`, `load_cost_forecast`, and `prod_price_forecast` are intentionally tolerant of length, frequency, and timezone differences — day-ahead price feeds typically publish 24h, not 48h, and get padded gracefully. Tolerant ≠ silent: if you change resample/align logic, log clearly when alignment happens. Silent shifts have historically produced wrong-but-valid-looking plans (`optim_status: Optimal` with every timestep offset by N).

**Things AI tools commonly hallucinate or get wrong here:**

- Confusing `param_definitions.json` (GUI schema, source of truth for default values) with `config_defaults.json` (runtime loader fallback for headless installs).
- Inventing solver or CVXPY APIs that do not exist in the pinned version.
- Forgetting that the public `command_line.py` entry points (`set_input_data_dict`, `perfect_forecast_optim`, `dayahead_forecast_optim`, `naive_mpc_optim`, `publish_data`) are `async def` and writing synchronous wrappers around them.

**Token and context limits:** the largest source files (`optimization.py`, `command_line.py`, both 3000+ lines) exceed comfortable context for many models. Use `repomix` (`npx repomix`) to flatten the repo for full-context tools that support it; otherwise scope reading to specific functions.

## Section 6 — Behavioral guardrails
Four rules. Reduce wrong-assumption drift, scope creep, dead-code regression.
1. **Think first.** Surface assumptions before code. Multiple interpretations → ask, never silently pick. Simpler path exists → say so. Unclear → stop, name the confusion.
2. **Simplicity.** Min code that solves stated problem. No bonus features. No abstractions for single-use. No error handling for impossible cases. 200 lines where 50 work → rewrite.
3. **Surgical.** Touch only requested code. No "while I'm here" refactors. Match existing style. Remove only imports/vars YOUR change orphaned, never pre-existing dead code unless asked.
4. **Goal-driven.** Task → verifiable criteria. "Add validation" → "tests for invalid inputs pass". "Fix bug" → "reproduce-test passes". State plan as numbered steps, each with verify check.
Source: Andrej Karpathy LLM-coding-pitfalls observations (https://x.com/karpathy/status/2015883857489522876), distilled.

## Section 7 — Conventions

- **Documentation style:** soft Diátaxis (https://diataxis.fr/): tutorials, how-tos, reference, explanation. Pragmatic, not strictly four-quadrant. The `docs/study_cases/` directory holds worked examples.
- **Commit messages:** prefix with type (`fix`, `docs`, `feat`, `chore`) per recent maintainer practice.

## Section 8 — Where to find more

- [`docs/develop.md`](docs/develop.md) — canonical EMHASS development guide (fork, venv, DevContainer, Docker, adding a parameter, PR process). Read this first.
- [`llms.txt`](https://emhass.readthedocs.io/en/latest/llms.txt) — Sphinx-generated routing manifest. The file does not exist in the source tree; it is built per Sphinx run and served from Read the Docs.
- [`docs/study_cases/`](docs/study_cases/) — Diátaxis-soft worked examples per persona.
- [Project board](https://github.com/users/davidusb-geek/projects/2) — coordination and scope corridors visible per card.
