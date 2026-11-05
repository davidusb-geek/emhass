# MPC orchestration via Node-RED

## Goal

Drive EMHASS naive-MPC optimization from a Node-RED flow on any cadence, recomputing runtime parameters per call. Transport-agnostic on the EMHASS side; you pick any sensor source (Modbus, MQTT, HA bridge, manufacturer API) for the inputs. After completing this recipe, your Node-RED instance posts a fresh MPC request every tick and your downstream consumers (smart-home controllers, dashboards, automations) receive the optimized plan.

## Prerequisites

- EMHASS reachable via HTTP on a known host:port (default `:5000`)
- Node-RED 3+ with the default `inject`, `function`, and `http request` nodes
- A static EMHASS config that already declares your deferrables / battery / thermal loads. This recipe only covers the *runtime params* (the values you change per MPC call). Static config lives in `config.yaml` / config-GUI.

## Step 1: Verify your static EMHASS config

<!-- source: src/emhass/data/config_defaults.json:41-46,109-117 -->

Before runtime overrides will work, the matching static keys must exist. Open your EMHASS `config.yaml` (or use the Web Config GUI) and check it has at least the keys this recipe will drive:

```yaml
optim_conf:
  set_use_battery: true                              # if you have a battery
  number_of_deferrable_loads: 2                      # adjust to your setup
  nominal_power_of_deferrable_loads: [3000, 11000]   # second load: the EV charger (Step 3 uses charger_kw = 11)
  operating_hours_of_each_deferrable_load: [4, 0]    # overridden per MPC call
  start_timesteps_of_each_deferrable_load: [0, 0]    # overridden per MPC call
  end_timesteps_of_each_deferrable_load: [0, 0]      # overridden per MPC call
```

The default `maximum_power_from_grid` (9000 W) is a hard limit. Below an 11 kW EV charger it makes a night charge without PV or battery infeasible for every load, so set it to your connection limit.

Reference table of the runtime params accepted by `treat_runtimeparams` that this recipe sends per call:

<!-- source: src/emhass/utils.py:933-1015 (treat_runtimeparams) -->

| Field | Type | Purpose |
|---|---|---|
| `operating_timesteps_of_each_deferrable_load` | `list[int]` | steps each deferrable should run. Once it is passed, `operating_hours_of_each_deferrable_load` is ignored for every load, so give all loads in steps (see the [EV study case](../study_cases/ev.md#runtime-payload)) |
| `start_timesteps_of_each_deferrable_load` | `list[int]` | earliest allowed step per deferrable |
| `end_timesteps_of_each_deferrable_load` | `list[int]` | the load may not run at this step or later, counted from the first optimization step; `0` means no limit |
| `load_cost_forecast` | `list[float]` | per-timestep tariff for load |
| `prod_price_forecast` | `list[float]` | per-timestep sell price for production |
| `soc_init` | `float` (0..1) | current battery state of charge as fraction |
| `prediction_horizon` | `int` | number of optimization steps to plan. **Always send it:** without it `naive-mpc-optim` plans only 10 steps |

Expected: EMHASS restarts cleanly with the static keys; `GET /api/get-config` returns them.

## Step 2: Add the cron trigger

Drag an `inject` node into your Node-RED tab. Configure:

- Payload: `{}` (empty object, type `JSON`)
- Repeat: `interval`, every 5 min (or whatever cadence you want; 5-15 min is typical in production). Plain price lists are positional from the first optimization step, which `method_ts_round` derives from the request time (see [Step indices and other positional inputs](../passing_data.md#step-indices-and-other-positional-inputs)). Set `method_ts_round` to `first` so that the first step is the step containing the tick, and build the lists from that step. This does not synchronise the two clocks: this flow builds the payload before EMHASS reads the request time, so a tick close to a step boundary can give EMHASS a first step one step later than the one the lists assume. Prefer sending the prices as a timestamped mapping when your source has timestamps, and check the first timestamp of the returned plan against the assumed first step before acting on it.
- Inject once after deploy: optional, useful for testing

This is the heartbeat that will drive every MPC call. Wire its output to the next node (Step 3).

Expected: when you click the inject node's input button, a `{}` message appears on the downstream debug node.

## Step 3: Build the `runtime_params` function

Add a `function` node downstream of the inject. The body computes runtime params from your sensor sources and assembles the JSON payload EMHASS expects:

```javascript
// Read whatever sensor values your stack exposes via context / flow / msg.
// The example values below are placeholders — wire them to your actual sources.

const charger_kw = 11.0;
const timestep_min = 30;          // must match the EMHASS optimization_time_step
const horizon_steps = 48;         // 48 × 30min = 24 elapsed hours (a fixed step count, also on a 23- or 25-hour DST day)

// Example: deferrable #2 is an EV, recompute its window each call
const ev_remaining_kwh = Number(flow.get("ev_remaining_kwh"));   // missing or non-numeric -> no EV, never a made-up amount
// Whole steps, rounded up (round() removes float noise). A whole number of steps is solved as a normal
// mixed-integer problem at the nominal power; fractional hours are solved as "Optimal (Relaxed)".
let ev_steps = Number.isFinite(ev_remaining_kwh)
    ? Math.max(0, Math.ceil(Math.round(ev_remaining_kwh / (charger_kw * timestep_min / 60) * 1e6) / 1e6)) : 0;
// The deadline as an epoch instant in ms, stored by your flow when the departure is known (a stored "minutes
// remaining" value would go stale). The difference of two instants is elapsed time; two local clock readings
// would be an hour off across a daylight saving change. No fabricated default: without a deadline there is no EV window.
const now_ms = Date.now();   // one clock reading for the deadline and the first step
const deadline_ms = Number(flow.get("ev_deadline_ms"));
const minutes_until_deadline = Number.isFinite(deadline_ms) ? (deadline_ms - now_ms) / 60000 : NaN;
// Step indices count from the first optimization step, which with method_ts_round "first" started
// minutes_into_step minutes ago (epoch based: assumes a UTC offset that is a whole number of steps).
const minutes_into_step = (now_ms % (timestep_min * 60000)) / 60000;
// EMHASS reads the clock after this node: in the last minute before a step boundary its first step may be the
// next one, so count one step fewer (the EV can then only stop earlier, never run in the deadline step).
const margin = minutes_into_step > timestep_min - 1 ? 1 : 0;
let end_step = Number.isFinite(minutes_until_deadline)
    ? Math.max(0, Math.min(Math.floor((minutes_until_deadline + minutes_into_step) / timestep_min), horizon_steps) - margin) : 0;
// An end of 0 means "no limit", not "stop now", so it is never sent for the EV. Plan no more steps than
// fit before the deadline (the deadline stays strict), and no EV at all when not one full step is left.
if (ev_steps > end_step) {
    if (ev_steps > 0) node.warn(`EV needs ${ev_steps} steps but ${end_step > 0 ? "only " + end_step : "none"} remain before the deadline (or the deadline is missing or past); the target will be missed`);
    ev_steps = end_step;
}
if (ev_steps === 0) end_step = horizon_steps;

// Battery SOC from your battery monitor, normalized to fraction 0..1
const soc_raw = flow.get("battery_soc_percent");
const soc_percent = (typeof soc_raw === "number" ||
    (typeof soc_raw === "string" && soc_raw.trim() !== ""))
    ? Number(soc_raw) : NaN;
if (!Number.isFinite(soc_percent) || soc_percent < 0 || soc_percent > 100) {
    node.warn("Skipping MPC tick: battery SOC missing, non-numeric or outside 0-100%");
    return null;
}
const soc_init = soc_percent / 100;

// Per-timestep price arrays from your tariff source, sent as plain lists: one
// value per optimization timestep, starting with the first step of the grid (the step
// containing now when method_ts_round is "first"). EMHASS
// rejects a list shorter than horizon_steps (never pads it) and ignores extra
// values. Prices may be negative.
const load_cost = flow.get("load_cost_per_step");
const prod_price = flow.get("prod_price_per_step");

// Fail closed: skip this tick instead of sending made-up prices.
const isValid = (a) => Array.isArray(a) && a.length >= horizon_steps
    && a.every((v) => typeof v === "number" && Number.isFinite(v));
if (!isValid(load_cost) || !isValid(prod_price)) {
    node.warn(`Skipping MPC tick: price forecast missing, shorter than ${horizon_steps} or not finite`);
    return null;
}

msg.payload = {
    prediction_horizon: horizon_steps,
    operating_timesteps_of_each_deferrable_load: [Math.ceil(4 * 60 / timestep_min), ev_steps],   // all loads in steps: hours are ignored once timesteps are passed
    start_timesteps_of_each_deferrable_load: [0, 0],
    end_timesteps_of_each_deferrable_load: [horizon_steps, end_step],
    load_cost_forecast: load_cost,
    prod_price_forecast: prod_price,
    soc_init: soc_init
};
return msg;
```

Expected: `msg.payload` now contains the full runtime-params object as JSON. Wire a `debug` node temporarily to verify before continuing.

## Step 4: Configure the `http request` node

Add an `http request` node downstream of Step 3. Settings:

- Method: `POST`
- URL: `http://<EMHASS_HOST>:5000/action/naive-mpc-optim`
- Headers: `Content-Type: application/json`
- Send: `as JSON`
- Return: `parsed JSON`
- Timeout: at least 120 000 ms (long MPC runs); in production a setup with deferrables + thermal regularly takes 90-120 s, day-ahead longer

Expected: the node returns `msg.payload` containing EMHASS's optimization result (CSV or JSON depending on EMHASS version), and `msg.statusCode === 200`. On `500`, EMHASS returns a JSON error body — read `msg.payload` for details.

## Step 5: Wire downstream consumers and the audit triplet

Two output channels you almost certainly want:

**(a) MQTT publish** to make the plan reachable for non-Node-RED consumers (smart-home controllers, dashboards, HA bridges). For each plan field you care about, add an `mqtt out` node with `retain: true` and a topic like `emhass/<field>` so new subscribers get last-known-state on reconnect.

**(b) Audit triplet** — three nodes wired alongside the flow to catch problems before they accumulate:

- `catch` node on the http-request: routes errors to the audit writer
- `status` node on the http-request: routes state changes (yellow=in-progress, red=error) to the audit writer
- `function` node formatting both into a single JSONL line written to a rotating audit file. Recommended per-tick fields: `{ts, status, mode, soc_target, price_cents, pv, load, next_action}`.

Expected: every successful tick appends one JSONL line to your audit file; every error appends a line with `status: "error"` and the EMHASS response body. Multi-hour silent outages become visible after the fact.

## Caveats

The following are observed-in-production patterns from running this flow shape for months. Specific thresholds shown are illustrative — tune to your inverter, sensors, and tariff.

- **Field-name versioning.** Runtime-param names are EMHASS-version-sensitive. If you upgrade EMHASS, re-grep `src/emhass/utils.py` for the names you use; key renames are not always called out in release notes.
- **Watchdog with separated signals.** Publish two retained MQTT topics from this flow: one heartbeat from the orchestrator itself (every tick), one `cycle-ok` signal flipped when the EMHASS POST returns 200. A downstream consumer can then distinguish "orchestrator down" from "EMHASS down". Threshold pattern: WARN at ~2× MPC cadence with no tick, CRITICAL at ~4×. For a 15-min cadence that is 30 / 60 min. Without this, audit logs can have multi-hour gaps that go unnoticed.
- **Override layer.** EMHASS plan is a *recommendation*, not a binding command. In the function node downstream of the response, add an override layer for edge cases the optimizer misses — e.g. divert PV-surplus to battery even when EMHASS said "hold", if battery has headroom (`SOC ≥ 90%`) and `pv_surplus > ~500 W`. Log the override reason as a separate audit field.
- **Hysteresis dead-band.** Add ~50 W (or your inverter's noise floor) of dead-band around charge/discharge mode-transition boundaries so the orchestrator doesn't flap between modes as PV and load wiggle around equilibrium.
- **Forecast resilience.** Single PV-forecast source is a single point of failure. Production-grade orchestrators run primary (commercial: Solcast, Forecast.Solar, etc.) + physics-fallback (Open-Meteo `global_tilted_irradiance` × DC_kWp × eta, clipped to inverter AC max) + a daily auto-calibration step (EMA-update of a correction factor from real-vs-modeled PV).
- **State between ticks.** Use `flow.set(...)` / `flow.get(...)` (not `context.set/get`) so the values survive Node-RED redeploys of unrelated tabs.
- **Overlapping requests.** EMHASS does not order `naive-mpc-optim` requests or guard the result file: every run writes the same `opt_res_latest.csv` that a later `publish-data` reads, and the request that *finishes* last wins. If a request takes longer than the tick interval (a solve with deferrables and thermal loads can take 90-120 s on small hardware, and the `http request` node timing out does not stop the solve on the server), the second request can finish first and the older plan then replaces the newer one. In a deterministic test with two overlapping requests, the request sent first finished second and replaced the newer plan. Skip a tick while a request is in flight. At the top of the Step 3 function: `const t = flow.get("mpc_in_flight_since"); if (t && Date.now() - t < 10 * 60000) { node.warn("Skipping MPC tick: previous request still in flight"); return null; }`. Set the flag only when the request is really sent, with `flow.set("mpc_in_flight_since", Date.now());` just before `return msg;` (the earlier `return null` paths send nothing, so they must not set it). Clear it with `flow.set("mpc_in_flight_since", null)` in a function node after the `http request` node, only when `typeof msg.statusCode === "number"` (a response arrived, so the solve has finished). Do not clear it on a timeout or connection error: the server may still be solving, so let the limit expire instead. Choose the 10-minute limit above your slowest solve; it is what releases the flag after a timeout or a lost response, so it can never stop the flow for good (a Node-RED restart also clears the in-memory flag). This protects one Node-RED flow only; several callers need a server-side guard, which EMHASS does not provide.
- **Price arrays.** As plain lists, `load_cost_forecast` and `prod_price_forecast` are positional from the first optimization step and must have at least `horizon_steps` entries of finite numbers (a list that starts one step early or late places every price on the wrong interval and still gives a valid-looking plan). EMHASS rejects a shorter list with an error instead of padding it, and fails the cycle on `NaN`, `null`, booleans or strings. Step 3 above checks both and skips the tick rather than sending a fabricated fallback. If your tariff source publishes timestamped prices, you can send them as a timestamped mapping instead and let EMHASS align them. See the [Forecast input contract](../passing_data.md#forecast-input-contract).
- **Battery SOC unit.** EMHASS expects `soc_init` as a fraction in [0, 1]. Most sensor sources publish percent; divide by 100. Step 3 preserves a valid 0% SOC and skips the tick if the source is missing, non-numeric, or outside 0-100%, rather than substituting a fabricated 50%. See [Battery-aware runtime params](battery_aware_runtime_params.md) for the full story.

## Credits

- Prior art: long-form MPC walkthrough at `docs/study_cases/mpc.md`.
- Patterns extracted from author's production Node-RED setup (months in service; multiple EMHASS-version iterations). Generic patterns only — no private flow JSON, sensor names, IPs, secrets, or location-specific data copied.
- Field names verified against `src/emhass/utils.py:treat_runtimeparams` and `src/emhass/data/config_defaults.json` on 2026-05-11.
