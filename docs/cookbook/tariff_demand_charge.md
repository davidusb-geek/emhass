# Demand / capacity charge on peak grid import

## Goal

Make EMHASS shave your **peak grid import** (the kW demand / capacity charge many utilities bill on top of energy) in the *same* optimization that minimizes energy cost — no core changes, no second solver. You set one config key to your tariff's demand rate; EMHASS adds a peak-power term to the objective and spreads deferrable load to hold the single highest import point down. Optionally you feed it the peak already locked in this billing period so it never wastes flexibility fighting a peak it can't beat.

## Prerequisites

- EMHASS **≥ 0.17.7** (the opt-in capacity / demand charge with billing-period peak floor landed in that release — CHANGELOG `#623`).
- No special config block — this rides on your existing grid-import setup. If `capacity_cost_per_kw` is `0` (the default) the feature is a true no-op: the peak variable is not even created.
- Transport-agnostic. The static rate is a config key; the optional incurred-peak floor is a runtime param on `naive-mpc-optim` calls (any orchestrator that POSTs runtime params — Node-RED, AppDaemon, HA `rest_command`, a cron `curl`).

## Step 1: Turn on the demand charge

<!-- source: src/emhass/data/config_defaults.json:139 (default 0.0 = off) -->
<!-- source: src/emhass/data/associations.csv:99 (runtime-overridable, same name) -->

Set `capacity_cost_per_kw` in your `optim_conf` to your utility's demand-charge rate, in **your currency per kW** of billed peak. This is not a tuning weight you invent — it is the real tariff number, so the optimizer trades a €/kW peak reduction against €/kWh energy arbitrage on the correct economic footing.

```yaml
optim_conf:
  # Demand / capacity charge on the single peak grid import over the horizon.
  # Currency per kW. 0.0 = feature off (default). Example: €8/kW/month tariff.
  capacity_cost_per_kw: 8.0
```

It is also runtime-overridable under the same name (associations.csv), so a time-of-use utility with different demand rates per season can pass it per call instead of hard-coding it.

Expected: with `capacity_cost_per_kw > 0`, a day-ahead or MPC run still solves and returns `optim_status: Optimal`; the resulting `P_grid` plan has a **lower maximum import** than the same run with the key at `0` (verified in Step 4).

## Step 2: Understand what the solver is doing (the model)

<!-- source: src/emhass/optimization.py:1408-1419 (peak_import epigraph + incurred-peak floor) -->
<!-- source: src/emhass/optimization.py:1646 (objective term) -->

The feature adds one scalar variable `peak_import` (Watts) and the epigraph constraint that pins it to the highest grid-import timestep:

```
peak_import ≥ p_grid_pos[t]        for every timestep t     (epigraph → peak = max import)
peak_import ≥ current_period_peak                           (floor at already-incurred peak, Step 3)
```

and one term to the maximization objective (EMHASS maximizes −cost):

```
maximize:  −Σ_t ( unit_load_cost[t] · p_grid_pos[t] · Δt )        # energy cost, per-timestep
           − capacity_cost_per_kw · ( peak_import / 1000 )        # demand charge, ONE-TIME on the peak
```

That is exactly a weighted-sum of the two objectives: energy cost plus a peak-power penalty. Because the demand term is a **power** charge it is *not* multiplied by the timestep the way the energy terms are — `peak_import` is in W and divided by 1000 to price it in kW. The epigraph is what linearizes `max(p_grid_pos)` into an LP the existing solver handles, so nothing about your solver choice changes.

Expected: no action this step — this is the mental model for why raising `capacity_cost_per_kw` flattens the import profile instead of just shifting it to the cheapest hour.

## Step 3 (MPC only): Feed the peak already incurred this billing period

<!-- source: src/emhass/utils.py:1637-1638 (treat_runtimeparams reads current_period_peak on the prediction_horizon path) -->
<!-- source: src/emhass/optimization.py:281-303, 3889-3906 (scalar Watts param, coerced/validated) -->

A demand charge is billed on the **month's** peak, but one optimization only sees its own horizon. If you already hit, say, 6 kW earlier this month, there is no point spending battery/deferrable flexibility to keep this afternoon under 6 kW — that peak is already paid for. Pass the running monthly peak as `current_period_peak` (in **Watts**) so the solver floors `peak_import` there and only fights *new* peaks above it.

This is a runtime param, honored on the `naive-mpc-optim` (prediction-horizon) path. POST it to `/action/naive-mpc-optim` in the `runtime_params` body (strictly valid JSON, copy-paste as-is):

```json
{
  "prediction_horizon": 24,
  "capacity_cost_per_kw": 8.0,
  "current_period_peak": 6000
}
```

- `current_period_peak` is in **Watts** — the highest grid import measured so far this billing month.
- `capacity_cost_per_kw` here is an optional per-call override of the Step 1 config value.
- Both are **scalars, not arrays**, so the template's array-length / `horizon_steps` sizing rule does not apply — nothing to pad or truncate.

Your orchestrator maintains the running peak: on each cycle, `current_period_peak = max(previous_stored_peak, latest_measured_grid_import_W)`, reset to `0` at the start of each billing period.

Expected: with a non-zero `current_period_peak`, the plan stops shaving below that floor — deferrable loads relax up to (but not past) the incurred peak, recovering energy-cost savings the charge would otherwise forfeit.

## Step 4: Verify the shave

<!-- source: docs/plan_output_schema.md — `P_grid` (W, positive = import); P_grid = P_grid_pos + P_grid_neg at optimization.py:2299 -->

`peak_import` is an internal solver variable, not a published column, so read the effect off the published `P_grid` series: its maximum positive value is the planned peak import. Run the same inputs twice and compare:

```python
# From the EMHASS optimization result DataFrame `opt_res`:
peak_off = opt_res["P_grid"].clip(lower=0).max()   # capacity_cost_per_kw = 0.0
peak_on  = opt_res["P_grid"].clip(lower=0).max()   # capacity_cost_per_kw = 8.0
# Expect peak_on <= peak_off, at the cost of a small rise in the energy-only cost_fun term.
```

Expected: `peak_on ≤ peak_off`. The gap is your planned peak reduction; if it is zero, either your load has no shiftable headroom in this horizon or the demand rate is too small relative to the energy spread to justify moving anything.

## Caveats

- **Horizon peak ≠ calendar-month peak.** Each run only prices the peak *within its own horizon*. True monthly demand-charge behavior requires the MPC path plus `current_period_peak` (Step 3) carrying the month's running peak; a bare day-ahead run resets the notion of "peak" every solve.
- **`current_period_peak` is MPC-only.** It is read from runtime params on the `prediction_horizon` path (`utils.py:1637`) and defaults to `None` on the day-ahead path (`utils.py:1675`) — passing it to `dayahead-optim` has no effect.
- **Opt-in, fail-open on bad input.** Default `0.0` skips the variable entirely (`optimization.py:1408`). A negative or non-finite `capacity_cost_per_kw` / `current_period_peak` is *ignored with a warning*, not an error (`optimization.py:1260-1270`, `3894-3906`) — so a bad value silently disables the charge; check your logs if a shave you expected does not appear.
- **Units.** `capacity_cost_per_kw` is per **kW**; `current_period_peak` is in **Watts** (matches `P_grid`). Mixing them up (e.g. passing 6 instead of 6000) sets a 6 W floor, effectively no floor.

## Credits

- Feature: opt-in capacity / demand charge with billing-period peak floor — **#623**, implemented by @LesIT1, requested by @matti-oss.
- Weighted-sum peak/cost LP formulation from the #623 discussion (@Whatsonyourmind).
- Field names and line numbers verified against `src/emhass/utils.py:treat_runtimeparams`, `src/emhass/optimization.py`, and `src/emhass/data/config_defaults.json` on 2026-07-18 (EMHASS 0.17.9 tree).
