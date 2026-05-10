---
orphan: true
---

<!--
Cookbook recipe template.

To contribute a recipe:
  1. Copy this file:  cp _template.md <category>_<pattern>.md   (e.g. ev_daily_commute.md)
  2. Fill the sections below.
  3. Add a link to your new recipe in `index.md` under the matching category.
  4. Open a PR.

Contributor rules (see DOC-cookbook design notes):
  - Config snippets MUST be source-verified against `src/emhass/utils.py` (treat_runtimeparams)
    or `src/emhass/optimization.py`. Include `<!-- source: <file>:<line> -->` HTML comments
    above each Config / Snippet code block.
  - Transport snippets (HA rest_command / Node-RED / EVCC / AppDaemon / vendor-native) must mark
    which stack the recipe was tested against, or mark untested variants as such.
  - Keep total length under ~200 lines including code blocks.
-->

# Recipe Title

## Goal

One sentence — what does this recipe achieve?

## Prerequisites

- EMHASS version: e.g. ≥ X.Y
- Config flags / runtime env required (one per line)
- Transport stack tested against: e.g. Node-RED 3+, Home Assistant Core ≥ 2024.x, EVCC ≥ 0.x, etc.

## Config

<!-- source: src/emhass/path.py:LINE -->

```yaml
# EMHASS config keys, runnable as-is
```

## Snippet

<!-- source: src/emhass/path.py:LINE (if applicable) -->
<!-- transport: Node-RED 3.1 (tested) / HA rest_command (untested — community contribution welcome) -->

```js
// integration code, runnable
```

## Caveats

- Known limit one
- Known limit two
- Edge case when X

## Credits

- Pattern from Discussion #NNN (@handle)
- Prior art: `docs/study_cases/...md`
