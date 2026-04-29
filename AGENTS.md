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

(filled in Task 3)

## Section 2 — Pipeline map

(filled in Task 4)

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
