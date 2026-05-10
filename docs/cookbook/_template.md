---
orphan: true
---

<!--
Cookbook recipe template (stepwise walkthrough format).

To contribute a recipe:
  1. Copy this file:  cp _template.md <category>_<pattern>.md   (e.g. ev_daily_commute.md)
  2. Replace the placeholders below.
  3. Add a link to your new recipe in `index.md` under the matching category.
  4. Open a PR.

Contributor rules:

  A. Source verification (mandatory)
     - Every `<!-- source: <file>:<line> -->` comment must cite a REAL line number in
       the current `src/emhass/...` tree. Do NOT leave `path.py:LINE` placeholders in
       a merged recipe — reviewers will reject.
     - Anchor recipe Config + Snippet content against `src/emhass/utils.py`
       (`treat_runtimeparams`), `src/emhass/optimization.py`, or
       `src/emhass/data/config_defaults.json`.

  B. Length and array-size discipline
     - Whenever a Snippet builds an array passed to EMHASS (`load_cost_forecast`,
       `prod_price_forecast`, etc.), include either a runtime length-check OR an
       in-code comment stating the expected length (= `horizon_steps`). EMHASS
       silently pads / truncates mismatched arrays.

  C. Transport tagging
     - Mark every Snippet with which transport stack it was tested on (e.g.
       Node-RED 3.x, Home Assistant 2024.x, AppDaemon Y.Z). Mark untested
       variants explicitly: "untested — contribution welcome".

  D. Stepwise structure
     - Use the Step-1 / Step-2 / ... format below. Each step is small (one node /
       one config block / one validation pass), has narrative explaining WHY,
       shows the code, and ends with "Expected: ..." stating what the reader
       should see after this step.

  E. Length cap
     - Keep total length under ~250 lines including code blocks. If you need more,
       consider splitting into two recipes (e.g. basic + advanced variant).
-->

# Recipe Title

## Goal

One sentence — what does this recipe achieve?

## Prerequisites

- EMHASS version: e.g. ≥ X.Y
- Config flags / runtime env required (one per line)
- Transport stack tested against: e.g. Node-RED 3+, Home Assistant Core ≥ 2024.x, EVCC ≥ 0.x, etc.

## Step 1: Verify your static EMHASS config

<!-- source: src/emhass/data/config_defaults.json:<line> -->

Narrative — what static keys must be set before this recipe makes sense.

```yaml
# EMHASS config keys, runnable as-is
```

Expected: your EMHASS instance restarts cleanly with these keys present.

## Step 2: <first integration step>

<!-- source: src/emhass/<file>.py:<line> (if applicable) -->
<!-- transport: Node-RED 3.x (tested) -->

Narrative — what this step adds and why.

```js
// integration code, runnable, source-verified
```

Expected: `msg.payload` (or equivalent) now contains `<x>` after this step.

## Step 3: <next integration step>

Narrative + code + Expected. Repeat as needed (Step 4, Step 5, ...).

## Caveats

- Known limit one
- Known limit two
- Edge case when X

## Credits

- Pattern from Discussion #NNN (@handle)
- Prior art: `docs/study_cases/...md`
- Field names verified against `src/emhass/utils.py:treat_runtimeparams` on YYYY-MM-DD
