---
name: emhass-agents
description: Vendor-neutral rules for AI coding agents working on EMHASS source.
---

<!-- Last verified against upstream/master @ 6537c47, 2026-04-30 -->

This file documents rules for AI coding agents (Claude Code, Cursor, Aider, Copilot, Codex) working on EMHASS source. It complements `docs/develop.md` (canonical for humans) and does not duplicate its content. Where `docs/develop.md` already covers a topic, this file links and adds AI-specific constraints on top.

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
| Lint | No enforced linter at the time of writing. |

Tech stack (verify versions in `pyproject.toml` before assuming an API):

| Component | Version source |
|---|---|
| Python | `pyproject.toml` `requires-python` |
| Pydantic | v1 at the time of writing |
| Optimisation | CVXPY (pin in `pyproject.toml`) |
| Web | Flask |
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

(filled in Task 5)

## Section 4 — Maintainer scope corridors

(filled in Task 6)

## Section 5 — Limits and gotchas

(filled in Task 7)

## Section 6 — Conventions

(filled in Task 8)

## Section 7 — Where to find more

(filled in Task 9)
