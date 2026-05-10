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
  nominal_power_of_deferrable_loads: [3000, 750]
  operating_hours_of_each_deferrable_load: [4, 0]    # overridden per MPC call
  start_timesteps_of_each_deferrable_load: [0, 0]    # overridden per MPC call
  end_timesteps_of_each_deferrable_load: [0, 0]      # overridden per MPC call
```

Reference table of the runtime params accepted by `treat_runtimeparams` that this recipe sends per call:

<!-- source: src/emhass/utils.py:933-1015 (treat_runtimeparams) -->

| Field | Type | Purpose |
|---|---|---|
| `operating_hours_of_each_deferrable_load` | `list[int]` | hours each deferrable should run |
| `start_timesteps_of_each_deferrable_load` | `list[int]` | earliest allowed step per deferrable |
| `end_timesteps_of_each_deferrable_load` | `list[int]` | latest allowed step per deferrable |
| `load_cost_forecast` | `list[float]` | per-timestep tariff for load |
| `prod_price_forecast` | `list[float]` | per-timestep sell price for production |
| `soc_init` | `float` (0..1) | current battery state of charge as fraction |

Expected: EMHASS restarts cleanly with the static keys; `GET /api/get-config` returns them.

## Step 2: Add the cron trigger

Drag an `inject` node into your Node-RED tab. Configure:

- Payload: `{}` (empty object, type `JSON`)
- Repeat: `interval`, every 5 min (or whatever cadence you want; 5-15 min is typical in production)
- Inject once after deploy: optional, useful for testing

This is the heartbeat that will drive every MPC call. Wire its output to the next node (Step 3).

Expected: when you click the inject node's input button, a `{}` message appears on the downstream debug node.

## Step 3: Build the `runtime_params` function

Add a `function` node downstream of the inject. The body computes runtime params from your sensor sources and assembles the JSON payload EMHASS expects:

```javascript
// Read whatever sensor values your stack exposes via context / flow / msg.
// The example values below are placeholders — wire them to your actual sources.

const charger_kw = 11.0;
const timestep_min = 30;          // must match EMHASS optimization_time_step_minutes
const horizon_steps = 48;         // 48 × 30min = 24h

// Example: deferrable #2 is an EV, recompute its window each call
const ev_remaining_kwh = flow.get("ev_remaining_kwh") || 0;
const ev_hours = Math.ceil(ev_remaining_kwh / charger_kw);
const minutes_until_deadline = flow.get("minutes_until_deadline") || 480;
const end_step = Math.floor(minutes_until_deadline / timestep_min);

// Battery SOC from your battery monitor, normalized to fraction 0..1
const soc_percent = flow.get("battery_soc_percent") || 50;
const soc_init = soc_percent / 100;

// Per-timestep price arrays from your tariff source.
// Length MUST equal horizon_steps — EMHASS pads / truncates silently otherwise.
const load_cost = flow.get("load_cost_per_step") || new Array(horizon_steps).fill(0.30);
const prod_price = flow.get("prod_price_per_step") || new Array(horizon_steps).fill(0.08);

if (load_cost.length !== horizon_steps || prod_price.length !== horizon_steps) {
    node.warn(`Forecast length mismatch: load=${load_cost.length}, prod=${prod_price.length}, expected=${horizon_steps}`);
}

msg.payload = {
    operating_hours_of_each_deferrable_load: [4, ev_hours],
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
- **Length of price arrays.** `load_cost_forecast` and `prod_price_forecast` must have at least `horizon_steps` entries, otherwise EMHASS pads / truncates and you may not notice silent misalignment. Step 3 above includes a runtime length-check.
- **Battery SOC unit.** EMHASS expects `soc_init` as a fraction in [0, 1]. Most sensor sources publish percent; divide by 100. See [Battery-aware runtime params](battery_aware_runtime_params.md) for the full story.

## Credits

- Prior art: long-form MPC walkthrough at `docs/study_cases/mpc.md`.
- Patterns extracted from author's production Node-RED setup (months in service; multiple EMHASS-version iterations). Generic patterns only — no private flow JSON, sensor names, IPs, secrets, or location-specific data copied.
- Field names verified against `src/emhass/utils.py:treat_runtimeparams` and `src/emhass/data/config_defaults.json` on 2026-05-11.
