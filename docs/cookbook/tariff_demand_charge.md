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

## Step 3b (MPC only): Mask the charge to your tariff's demand window

<!-- source: src/emhass/optimization.py (param_capacity_window vector param, masked epigraph) -->

Many tariffs assess demand only inside a **window** (e.g. 16:00-20:00, business days). Without a mask, the epigraph prices *every* timestep, so a deliberate off-window import — midday EV charging is the classic case — pins `peak_import` and the optimizer both pays a phantom charge on it and sees zero marginal value in shaving the actual window. Pass `capacity_charge_window`, a `prediction_horizon`-length list of `0`/`1` weights aligned to the horizon timesteps, and the epigraph becomes `peak_import ≥ mask[t] · p_grid_pos[t]`: only in-window import can set the priced peak.

```json
{
  "prediction_horizon": 24,
  "capacity_cost_per_kw": 8.0,
  "current_period_peak": 6000,
  "capacity_charge_window": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
}
```

Your orchestrator owns the calendar (business days, public holidays, seasons) and recomputes the mask each cycle; EMHASS stays tariff-agnostic. Omit the key for the previous full-horizon behaviour. An all-zero mask (horizon entirely outside the window) makes the demand term a constant — the plan is then identical to a run without the charge, which is exactly right when no window timestep is reachable. If your billing period is monthly, zero the slots that fall in the *next* month and reset `current_period_peak` at the boundary. See `passing_data.md` for an HA template that builds the mask.

Expected: with the spike-owning timesteps masked out, off-window imports run unshaved at full power while in-window import is shaved toward the `current_period_peak` floor.

## Step 3c: Price the tariff's measurement interval, not the raw timestep

<!-- source: src/emhass/optimization.py:_build_capacity_interval_arrays (issue #540) -->

Most demand charges actually bill on the **average** import power over a fixed clocked interval (commonly 30 minutes), not a single optimisation timestep's instantaneous power. With a 5-minute `optimization_time_step`, a bare epigraph over each timestep (Steps 1-3b) prices a brief 5-minute spike as if it held for the whole interval — overstating the charge and giving the optimiser an artificially strong incentive to shave short spikes that would barely move the real 30-minute average.

Set `capacity_charge_interval_timesteps` to the number of native timesteps that make up one tariff measurement interval:

```yaml
optim_conf:
  capacity_cost_per_kw: 8.0
  # 30-minute tariff interval at a 5-minute optimization_time_step:
  capacity_charge_interval_timesteps: 6
```

`1` (the default) reproduces the same Steps 1-3b epigraph semantics: `peak_import >= cp.multiply(capacity_charge_window, p_grid_pos)`, no aggregation matrix is built at all. `N > 1` switches the epigraph to price the average import power over each *completed* N-timestep interval instead: a single 5-minute 6000 W spike inside an otherwise-idle 30-minute block is priced as its 1000 W (1/6-weighted) average, not the raw 6000 W.

`capacity_charge_interval_timesteps` is a structural property of the shared capacity-charge model (`optim_conf`, `associations.csv`) — like `capacity_cost_per_kw`, it follows normal EMHASS config/runtime-override behaviour (no special restriction), and it applies to **every** optimisation mode that uses the capacity charge: `naive-mpc-optim`, `dayahead-optim` and `perfect-optim` alike, since they all share the same `perform_optimization` constraint build. Changing its value changes the optimisation cache key, so it rebuilds the problem rather than reusing a warm-started one.

**Only the runtime-history mechanism below is `naive-mpc-optim`-only.** `dayahead-optim`/`perfect-optim` have no elapsed-interval history input at all — the epigraph still aggregates in N-timestep blocks for them when `N > 1`, but always with an empty history (equivalent to `m = 0`), so **the horizon start (t0) is implicitly assumed to already sit exactly on a tariff interval boundary** on those paths. If your horizon start does not align to a real interval boundary there, the first "completed" interval reported by the day-ahead/perfect solve will not match the real billing interval it is meant to represent.

On `naive-mpc-optim`, t0 is rarely exactly on an interval boundary. Pass `capacity_charge_current_interval_history` — the **average** positive import power (Watts, oldest → newest) for each native timestep already elapsed in the currently open interval, or an equivalent energy-derived average (`elapsed_energy_Wh / (optimization_time_step_minutes / 60)`), never an instantaneous snapshot — so the first completed interval prices correctly instead of assuming t0 is a boundary:

```json
{
  "prediction_horizon": 24,
  "capacity_cost_per_kw": 8.0,
  "capacity_charge_current_interval_history": [0, 0, 0, 6000]
}
```

Its length (`0` to `capacity_charge_interval_timesteps - 1`) tells EMHASS how far t0 sits into the open interval — your orchestrator owns that clock alignment, EMHASS does not compute timezone/season/wall-clock rules.

`capacity_charge_window` (Step 3b) still applies, but its semantics shift with aggregation on: for `N > 1` the mask is read only at each *completed interval's endpoint*, so this assumes your demand-window boundaries align with the tariff measurement-interval boundaries (the mask is effectively constant across any one N-timestep interval); EMHASS does not validate this alignment.

**Horizon-tail:** a tariff interval still incomplete at the far end of the current horizon has no row in the epigraph and is simply not priced by this solve — a known receding-horizon boundary condition, intentionally left without terminal-continuation machinery; a later solve prices it once enough of the horizon advances for its completion to become visible.

**`current_period_peak` and aggregation:** when `capacity_charge_interval_timesteps > 1`, `current_period_peak` (Step 3) must be expressed in the **same metric the epigraph now prices** - the highest already-*completed* clocked tariff-interval **average** import power (in Watts), not the maximum instantaneous or per-native-timestep import. For the N=6 / 5-minute MPC / 30-minute tariff example above, that is the highest completed 30-minute average positive import power seen so far this billing period. The currently *open* (not-yet-completed) interval belongs in `capacity_charge_current_interval_history`, not in `current_period_peak` - do not fold it into the incumbent floor until its tariff interval actually completes. There is no API change here: `current_period_peak` remains the same single scalar Watts parameter as before; only what value is the *correct* one to pass changes when aggregation is on.

Expected: with `capacity_charge_interval_timesteps > 1`, a brief single-timestep spike inside an otherwise-quiet interval barely moves `peak_import`, while a genuinely sustained import across the whole interval prices at close to its raw power, same as before.

## Step 4: Verify the shave

<!-- source: docs/plan_output_schema.md — `P_grid` (W, positive = import); P_grid = P_grid_pos + P_grid_neg at optimization.py:2299 -->

`peak_import` is an internal solver variable, not a published column, so read the effect off the published `P_grid` series: its maximum positive value is the planned peak import. Run the same inputs twice and compare:

```python
# From the EMHASS optimization result DataFrame `opt_res`:
peak_off = opt_res["P_grid"].clip(lower=0).max()  # capacity_cost_per_kw = 0.0
peak_on = opt_res["P_grid"].clip(lower=0).max()  # capacity_cost_per_kw = 8.0
# Expect peak_on <= peak_off, at the cost of a small rise in the energy-only cost_fun term.
```

Expected: `peak_on ≤ peak_off`. The gap is your planned peak reduction; if it is zero, either your load has no shiftable headroom in this horizon or the demand rate is too small relative to the energy spread to justify moving anything.

## Caveats

- **Horizon peak ≠ calendar-month peak.** Each run only prices the peak *within its own horizon*. True monthly demand-charge behavior requires the MPC path plus `current_period_peak` (Step 3) carrying the month's running peak; a bare day-ahead run resets the notion of "peak" every solve.
- **`current_period_peak` is MPC-only.** It is read from runtime params on the `prediction_horizon` path (`utils.py:1637`) and defaults to `None` on the day-ahead path (`utils.py:1675`) — passing it to `dayahead-optim` has no effect.
- **Opt-in, fail-open on bad input.** Default `0.0` skips the variable entirely (`optimization.py:1408`). A negative or non-finite `capacity_cost_per_kw` / `current_period_peak` is *ignored with a warning*, not an error (`optimization.py:1260-1270`, `3894-3906`) — so a bad value silently disables the charge; check your logs if a shave you expected does not appear.
- **Units.** `capacity_cost_per_kw` is per **kW**; `current_period_peak` is in **Watts** (matches `P_grid`). Mixing them up (e.g. passing 6 instead of 6000) sets a 6 W floor, effectively no floor.
- **`capacity_charge_window` is MPC-only and fail-open too.** Like `current_period_peak` it is read on the `prediction_horizon` path only. An invalid mask (non-numeric, NaN/inf, shorter than the horizon) is ignored with a warning — the full horizon gets priced again, which *overstates* the charge on off-window imports; check logs if the window seems to have no effect.
- **`capacity_charge_current_interval_history` is MPC-only and fail-open; `capacity_charge_interval_timesteps` applies everywhere.** The interval-aggregation math applies on `dayahead-optim`/`perfect-optim` too when `N > 1` (it is structural, shared by all three modes) — but with no history to consult there, so the horizon start must itself sit on an interval boundary for those runs to price the correct billing intervals. An invalid history (non-numeric, NaN/inf, negative, or longer than `capacity_charge_interval_timesteps - 1`) is ignored with a warning, falling back to an empty history — never an error. At `capacity_charge_interval_timesteps = 1` the aggregation matrix is never constructed at all — the plain per-timestep epigraph applies exactly as before this feature existed, and the history is never inspected or validated.

## Credits

- Feature: opt-in capacity / demand charge with billing-period peak floor — **#623**, implemented by @LesIT1, requested by @matti-oss.
- Weighted-sum peak/cost LP formulation from the #623 discussion (@Whatsonyourmind).
- Feature: per-timestep demand-window mask — **#1066**, implemented by @hossamnagy.
- Feature: tariff measurement-interval aggregation (`capacity_charge_interval_timesteps` / `capacity_charge_current_interval_history`) — **#540** discussion; this implementation covers single-component aggregation only, while multiple independent capacity components remain a separate follow-up.
- Field names and line numbers verified against `src/emhass/utils.py:treat_runtimeparams`, `src/emhass/optimization.py`, and `src/emhass/data/config_defaults.json` on 2026-07-18, using the EMHASS 0.17.9 source tree (the then-current release — a verification snapshot only; the feature itself requires just **≥ 0.17.7**, per Prerequisites).
