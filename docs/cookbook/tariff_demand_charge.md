# Demand / capacity charge on peak grid import

## Goal

Price a tariff's billed demand peak inside the same EMHASS optimization that prices energy. EMHASS remains tariff-agnostic: the caller supplies the rate, billing-period incumbent, demand window and (for rolling MPC) any realised part of the open tariff interval.

## Prerequisites

- Base capacity charging (`capacity_cost_per_kw` and `current_period_peak`) is available from EMHASS 0.17.7.
- Demand-window use requires a build exposing `capacity_charge_window`.
- Excluding a tariff-eligible occurrence from the current MPC solve's peak requires a build exposing `capacity_charge_consideration`.
- Tariff-interval aggregation requires a build exposing `capacity_charge_interval_timesteps` and `capacity_charge_current_interval_history`.
- Multiple independent capacity/demand components in one solve (`capacity_cost_per_kw` as a list of K rates) requires a build with issue #540 Part B.
- Transport: examples below are direct EMHASS config/runtime payloads. Adapter-specific Node-RED, Home Assistant and AppDaemon transport is not claimed as tested here.

## Step 1: Set the marginal capacity-charge rate

<!-- source: src/emhass/data/config_defaults.json:140 -->
<!-- source: src/emhass/data/associations.csv:99 -->
<!-- transport: direct EMHASS configuration; adapter-specific transport untested -->

Set `capacity_cost_per_kw` to the marginal billing cost of increasing the applicable billed peak by 1 kW. Do not blindly copy a daily tariff number unless the tariff applies one billing-period peak across those billable days; if it does, convert the daily rate to the corresponding billing-period marginal cost.

```yaml
optim_conf:
  capacity_cost_per_kw: 8.0
```

`0` (default) disables capacity charging.

Expected: the optimization still solves normally; a positive rate gives the solver an economic reason to reduce the applicable demand peak when flexibility is available.

## Step 2: Understand the default N=1 model

<!-- source: src/emhass/optimization.py:1936-1952 -->
<!-- source: src/emhass/optimization.py:2232 -->

At the default `capacity_charge_interval_timesteps = 1`, `peak_import` is constrained by each eligible positive-import timestep and floored by `current_period_peak`. The objective prices that scalar peak once in currency/kW.

```text
peak_import >= capacity_charge_window[t] * p_grid_pos[t]
peak_import >= current_period_peak

capacity term = capacity_cost_per_kw * peak_import / 1000
```

This is a power charge, so it is not multiplied by the optimization timestep.

Expected: `N=1` preserves the pre-aggregation capacity-charge semantics.

## Step 3: Feed the incumbent billing-period peak (MPC)

<!-- source: src/emhass/utils.py:1771 -->
<!-- transport: direct EMHASS naive-mpc-optim runtime JSON; adapter-specific transport untested -->

Pass `current_period_peak` in Watts. With `N=1`, use the highest eligible positive-import timestep already incurred in the current billing period. With `N>1`, use the highest eligible completed tariff-interval average instead.

```json
{
  "prediction_horizon": 24,
  "capacity_cost_per_kw": 8.0,
  "current_period_peak": 6000
}
```

The currently open tariff interval is not part of this incumbent until it completes.

Expected: EMHASS does not spend flexibility trying to reduce the planned peak below a billed peak that is already locked in.

## Step 3b: Apply the tariff demand window (MPC)

<!-- source: src/emhass/utils.py:1780 -->
<!-- transport: direct EMHASS naive-mpc-optim runtime JSON; adapter-specific transport untested -->

`capacity_charge_window` is a `prediction_horizon`-length list aligned with the horizon. The example below has `prediction_horizon: 24`, so the mask contains exactly 24 values.

```json
{
  "prediction_horizon": 24,
  "capacity_cost_per_kw": 8.0,
  "current_period_peak": 6000,
  "capacity_charge_window": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
}
```

The caller owns business-day, holiday, season and timezone rules. With `N>1`, EMHASS uses the mask value at each completed tariff interval endpoint, so demand-window boundaries should align with tariff measurement-interval boundaries.

Expected: only eligible demand-window timesteps/intervals can raise the priced peak.

## Step 3c: Exclude a tariff-eligible occurrence from the current MPC solve's peak

<!-- source: src/emhass/utils.py:1780 -->
<!-- transport: direct EMHASS naive-mpc-optim runtime JSON; adapter-specific transport untested -->

A rolling-MPC horizon can contain two occurrences of the same recurring demand window: a nearer one about to be executed and committed as billing history, and a later one still fully replannable next cycle. `capacity_charge_window` cannot express "still tariff-eligible, but do not let this occurrence affect the peak in THIS solve" - both occurrences are genuinely eligible, so masking the later one there would overload eligibility with a second meaning (issue [#540](https://github.com/davidusb-geek/emhass/issues/540)).

`capacity_charge_consideration` is a separate `prediction_horizon`-length list, aligned like `capacity_charge_window`, for exactly this case. It can only narrow what `capacity_charge_window` already allows - never widen it - and it never touches `current_period_peak`.

```json
{
  "prediction_horizon": 24,
  "capacity_cost_per_kw": 8.0,
  "current_period_peak": 6000,
  "capacity_charge_window": [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  "capacity_charge_consideration": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}
```

Here `capacity_charge_window` marks two separate occurrences of the same recurring demand window: indices 2-4 and indices 14-16. Both remain fully tariff-eligible. `capacity_charge_consideration` excludes only the second occurrence (indices 14-16 set to `0`), so only the first participates in THIS solve's prospective capacity peak; the second stays tariff-eligible but is capacity-unpriced in this solve. The `1`s elsewhere in `capacity_charge_consideration` are inert - `effective = capacity_charge_window * capacity_charge_consideration`, so consideration only ever matters where the window already allows it. A caller may instead consider one, several, or all occurrences each cycle; EMHASS exposes the mechanism only and does not choose the policy.

Defaults to unset (`None`) = every tariff-eligible timestep/interval considered, identical to today's behaviour without this key. Ordinary usage is `0`/`1` (considered or not); it is not a fractional billing discount.

```{warning}
Excluding a later, genuinely eligible occurrence does two things. It removes the MPC's incentive to pre-position the battery for that occurrence in the current solve; and, because only its capacity contribution was removed, it makes that occurrence comparatively attractive as a place to put grid import or battery charging. EMHASS exposes the mechanism only - it does not decide, or default to, a "nearest occurrence only" policy. Choose which occurrence(s) to consider deliberately; the consequence of that choice is caller-owned.

An excluded timestep is **capacity-unpriced in this solve, not free**: its energy price, the battery/grid/deferrable-load constraints and every other objective term still apply, and it stays fully tariff-eligible. `peak_import` is an internal optimisation variable only - it is not returned by `naive-mpc-optim` and is not published. To verify what a plan actually does, use the returned `P_grid_pos` and the Step 4 tariff-metric helper below, passing the *effective* `capacity_charge_window * capacity_charge_consideration` weight as its `window` argument when consideration is active - a plan can still schedule a large import at an excluded but still tariff-eligible timestep that never raised this solve's priced peak.
```

Expected: only the considered, tariff-eligible timestep/interval(s) can raise the priced peak; excluded-but-still-eligible timesteps/intervals do not participate in this solve's prospective capacity peak, and their tariff eligibility remains unchanged.

## Step 3d: Price a tariff measurement interval

<!-- source: src/emhass/data/config_defaults.json:141 -->
<!-- source: src/emhass/data/associations.csv:100 -->
<!-- source: src/emhass/optimization.py:1683 -->
<!-- source: src/emhass/utils.py:1788 -->
<!-- transport: direct EMHASS configuration/runtime JSON; adapter-specific transport untested -->

Set `capacity_charge_interval_timesteps` to the number of native optimization timesteps in one tariff measurement interval:

```yaml
optim_conf:
  capacity_charge_interval_timesteps: 6
```

For a 5-minute optimizer and a clocked 30-minute demand interval, `N=6`. `N` must be a positive integer; invalid values warn and fall back to `1`. A ratio such as 30/20 = 1.5 cannot be represented exactly by this model.

With `N>1`, EMHASS prices completed N-timestep average positive import instead of the largest raw native timestep. A single 6000 W 5-minute spike in an otherwise-zero 30-minute block therefore contributes 1000 W to the billed 30-minute average.

Rolling MPC may start inside an already-open interval. Pass the average positive-import power for each elapsed native timestep, oldest to newest:

```json
{
  "prediction_horizon": 24,
  "capacity_cost_per_kw": 8.0,
  "capacity_charge_interval_timesteps": 6,
  "capacity_charge_current_interval_history": [2000, 4000, 0, 6000]
}
```

At 17:20 in a 17:00-17:30 interval, four 5-minute averages are already realised. If the remaining planned averages are 3000 W and 1000 W, the completed 30-minute average is `(2000 + 4000 + 0 + 6000 + 3000 + 1000) / 6 = 2666.7 W`.

History entries are native-interval averages, not instantaneous snapshots. An equivalent energy measurement is valid after conversion to mean Watts.

Expected: the first tariff interval combines realised history with the remaining planned timesteps, then subsequent completed intervals use planned data only.

## Step 3e: Multiple independent capacity/demand components (K>1)

<!-- source: src/emhass/data/config_defaults.json:141 -->
<!-- source: src/emhass/static/data/param_definitions.json (capacity_cost_per_kw) -->
<!-- source: src/emhass/optimization.py (_init_capacity_multi_params) -->
<!-- transport: direct EMHASS configuration/runtime JSON; adapter-specific transport untested -->

Some tariffs bill more than one demand peak on the same connection at the same time — for example a seasonal-demand charge and an off-peak-demand charge, each with its own rate, eligibility window and billing-period incumbent. Pass `capacity_cost_per_kw` as a **list of two or more rates** to price K such components in the same single optimisation:

```json
{
  "prediction_horizon": 24,
  "capacity_cost_per_kw": [8.0, 3.0],
  "capacity_charge_interval_timesteps": 6,
  "current_period_peak": [6000, 4000],
  "capacity_charge_window": [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  ]
}
```

Semantics:

- **Canonical form.** `capacity_cost_per_kw` is normalised before anything structural is decided: `8.0`, `[8.0]` (what the config form serialises for a single value) and `[]` all collapse to the legacy scalar K=1 form (`[]` = disabled) — same model, same cache key. Only a list of two or more rates selects K>1, with `K = len(list)`. `capacity_charge_interval_timesteps` normalises the same way (`6` ≡ `[6]` = shared basis; `[6, 3]` = per-component bases).
- Every capacity runtime input becomes a **list of exactly K entries**, one per component in `capacity_cost_per_kw` order: `current_period_peak`, `capacity_charge_window`, `capacity_charge_consideration` and `capacity_charge_current_interval_history`. `capacity_charge_interval_timesteps` may be a bare value (shared measurement basis) or a list of exactly K for independent per-component intervals — **any other list length is a configuration error**, refused rather than silently reinterpreting the tariff measurement basis.
- Components are fully independent: separate rate, measurement interval, eligibility window, MPC consideration, realised open-interval history and incumbent peak. A peak seen only in component A's window never raises component B's peak; component A's consideration mask never affects component B; each incumbent floor binds only its own component.
- There is still **one optimisation, one solver call, one physical dispatch**. Only the capacity portion of the objective changes, from `capacity_cost_per_kw * peak_import / 1000` to `sum over k of capacity_cost_per_kw[k] * peak_import[k] / 1000`.
- A component whose rate is `0` (or invalid) is economically inactive — no `peak_import` variable, no epigraph, no objective term — exactly as a scalar `capacity_cost_per_kw` of `0` is a no-op. `[0.0, 0.0]` is a valid inert K=2 configuration.
- A non-list or wrong-length **runtime** value (e.g. 3 masks for a K=2 charge, or a bare scalar `current_period_peak`) is dropped for **every** component with one warning — never partially applied.

Verify each component separately with the Step 4 helper, passing that component's own window/consideration/history/incumbent and (when a list) that component's own `N`.

Expected: raising one component's rate changes the dispatch; the other components' peaks, windows and incumbents are unaffected.

## Step 4: Verify the tariff metric

<!-- source: src/emhass/optimization.py:1683 -->
<!-- transport: local Python helper; untested adapter transport - contribution welcome -->

Do not verify `N>1` with the raw maximum of `P_grid`; that would compare a native-timestep peak with a tariff-interval average. Use the same completed-interval metric, passing the effective `capacity_charge_window * capacity_charge_consideration` weight as `window` whenever consideration is active (unset consideration reduces to the raw window, matching EMHASS's own fail-open default):

```python
def billed_peak_w(p_grid_w, n=1, history=(), window=None, incumbent_w=0):
    imports = [max(float(p), 0.0) for p in p_grid_w]
    weights = [1.0] * len(imports) if window is None else list(window)
    if n == 1:
        candidates = [w * p for w, p in zip(weights, imports)]
    else:
        m = len(history)
        end = n - m - 1
        candidates = []
        first = True
        while end < len(imports):
            if first:
                total = sum(history) + sum(imports[: end + 1])
                first = False
            else:
                total = sum(imports[end - n + 1 : end + 1])
            candidates.append(weights[end] * total / n)
            end += n
    return max([float(incumbent_w), *candidates])


assert billed_peak_w([0, 0, 0, 0, 0, 6000], n=6) == 1000
assert round(billed_peak_w([3000, 1000], n=6, history=[2000, 4000, 0, 6000]), 1) == 2666.7
```

Expected: comparisons between capacity-charge runs use the billed metric above, not the raw native-timestep maximum.

## Caveats

- `current_period_peak`, `capacity_charge_window`, `capacity_charge_consideration` and `capacity_charge_current_interval_history` are MPC runtime inputs. The structural `capacity_charge_interval_timesteps` applies to the shared capacity-charge model.
- `capacity_charge_consideration` is separate from `capacity_charge_window`: it narrows which otherwise tariff-eligible occurrences count toward THIS solve's peak, and it never widens eligibility.
- Two different "history" quantities behave differently under consideration, and the distinction matters at `N>1`. `capacity_charge_current_interval_history` holds realised samples inside a **still-open** tariff interval; that interval's billed average does not exist until it closes, so the whole interval candidate is prospective and the endpoint consideration weight **does** scale it - the planned portion and the realised-history portion alike, all-or-nothing. `current_period_peak` is the already-completed, irreversible billed-history floor; it is an independent constraint that consideration **never** scales or erases, at any `N`.
- For well-aligned `N>1` consideration input, hold the chosen `capacity_charge_consideration` value constant across every planned native timestep of a given completed tariff interval - including, for the first, still-open interval, every decision index from `0` through its own endpoint - rather than only at the endpoint. The endpoint alone sets the applied weight, but a differing planned span still trips EMHASS's own misalignment warning. Never rewrite `capacity_charge_current_interval_history` to compensate; it holds only already-realised samples.
- `dayahead-optim` and `perfect-optim` have no elapsed-interval history. With `N>1`, start their horizon on a tariff measurement-interval boundary.
- A tariff interval incomplete at the far end of the horizon is not priced until a later receding-horizon solve can see its completion. No terminal continuation model is added here.
- Exact billing-period rollover with `N>1` assumes the billing-period boundary aligns with a tariff measurement-interval boundary. EMHASS does not split one aggregated interval across two billing periods.
- A bare `capacity_cost_per_kw` represents one capacity-charge component. For K independent components (different rates, windows, measurement intervals or incumbent peaks) in the same solve, pass `capacity_cost_per_kw` as a list of K rates and every capacity runtime input as a list of K entries — see Step 3e (issue #540 Part B).
- `current_period_peak` and interval history are in Watts; `capacity_cost_per_kw` is currency/kW.
- The caller owns tariff calendar/state. EMHASS does not persist billing-period peaks or compute tariff seasons/timezones.

## Credits

- Base capacity-charge feature — **#623**, implemented by @LesIT1, requested by @matti-oss.
- Weighted-sum / LP peak-cost formulation from the #623 discussion — @Whatsonyourmind.
- Demand-window feature — **#1066**, @hossamnagy.
- Tariff measurement-interval aggregation — **#540** discussion.
