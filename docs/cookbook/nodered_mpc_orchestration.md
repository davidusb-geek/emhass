# MPC orchestration via Node-RED

## Goal

Drive EMHASS naive-MPC optimization from a Node-RED flow on any cadence, recomputing runtime parameters per call. Transport-agnostic on the EMHASS side; you pick any sensor source (Modbus, MQTT, HA bridge, manufacturer API) for the inputs.

## Prerequisites

- EMHASS reachable via HTTP on a known host:port (default `:5000`)
- Node-RED 3+ with the default `inject`, `function`, and `http request` nodes
- A static EMHASS config that already declares your deferrables / battery / thermal loads. This recipe only covers the *runtime params* (the values you change per MPC call). Static config lives in `config.yaml` / config-GUI.

## Config

Static EMHASS config skeleton (only the parts relevant to MPC orchestration are shown — your real config will have more). Names per `src/emhass/data/config_defaults.json`:

<!-- source: src/emhass/data/config_defaults.json:41-56,109-117 -->

```yaml
optim_conf:
  set_use_battery: true                       # if you have a battery
  number_of_deferrable_loads: 2               # adjust to your setup
  nominal_power_of_deferrable_loads: [3000, 750]
  operating_hours_of_each_deferrable_load: [4, 0]      # overridden per MPC call
  start_timesteps_of_each_deferrable_load: [0, 0]      # overridden per MPC call
  end_timesteps_of_each_deferrable_load: [0, 0]        # overridden per MPC call
```

Runtime params accepted by `treat_runtimeparams` and overridden per MPC call:

<!-- source: src/emhass/utils.py:933-1015 (treat_runtimeparams) -->

| Field | Type | Purpose |
|---|---|---|
| `operating_hours_of_each_deferrable_load` | `list[int]` | hours each deferrable should run |
| `start_timesteps_of_each_deferrable_load` | `list[int]` | earliest allowed step per deferrable |
| `end_timesteps_of_each_deferrable_load` | `list[int]` | latest allowed step per deferrable |
| `load_cost_forecast` | `list[float]` | per-timestep tariff for load |
| `prod_price_forecast` | `list[float]` | per-timestep sell price for production |
| `soc_init` | `float` (0..1) | current battery state of charge as fraction |

## Snippet

<!-- transport: Node-RED 3.1 (tested against production setup); other languages welcome as separate recipes -->

A generic Node-RED flow shape — adapt to your sensor sources. Replace `<EMHASS_HOST>` with the EMHASS host:port. Do NOT copy verbatim, the field values are illustrative:

```text
[inject every 5 min] → [function: build runtime_params] → [http request: POST /action/naive-mpc-optim] → [debug]
```

**`function` node body** (JavaScript, what to put inside the Node-RED `function` node):

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

// Per-timestep price arrays from your tariff source (length = horizon_steps)
const load_cost = flow.get("load_cost_per_step") || new Array(horizon_steps).fill(0.30);
const prod_price = flow.get("prod_price_per_step") || new Array(horizon_steps).fill(0.08);

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

**`http request` node configuration:**

- Method: `POST`
- URL: `http://<EMHASS_HOST>:5000/action/naive-mpc-optim`
- Headers: `Content-Type: application/json`
- Send: `as JSON`
- Timeout: at least 120 000 ms (long MPC runs)

## Caveats

The following are observed-in-production patterns from running this flow shape for months across multiple EMHASS versions (V4.2 thermal/deferrable, V4.5 audit+API). Specific thresholds shown are illustrative — tune to your inverter, sensors, and tariff.

- **Field-name versioning.** Runtime-param names are EMHASS-version-sensitive. If you upgrade EMHASS, regrep `src/emhass/utils.py` for the names you use; key renames are not always called out in release notes.
- **MPC timeout.** Default Node-RED `http request` timeout is too short. Raise to ≥ 120 s. In production, naive-MPC with several deferrables + thermal regularly takes 90-120 s; day-ahead longer.
- **Watchdog with separated signals.** Publish two retained MQTT topics from this flow: one heartbeat from the orchestrator itself (every tick), one `cycle-ok` signal flipped when the EMHASS POST returns 200. A downstream consumer can then distinguish "orchestrator down" from "EMHASS down". Threshold pattern: WARN at ~2× MPC cadence with no tick, CRITICAL at ~4×. For a 15-min cadence that is 30 / 60 min. Without this, audit logs can have multi-hour gaps that go unnoticed.
- **Override layer.** EMHASS plan is a *recommendation*, not a binding command. In the function node downstream of the response, add an override layer for edge cases the optimizer misses — e.g. divert PV-surplus to battery even when EMHASS said "hold", if battery has headroom (`SOC ≥ 90%`) and `pv_surplus > ~500 W`. Log the override reason as a separate audit field.
- **Audit triplet.** Wire every long-running flow with three nodes: `catch` (errors), `status` (state changes), and a `function` that formats both into a JSONL line appended to a rotating audit log. Per-tick fields that pay off in post-mortems: `{ts, status, mode, soc_target, price_cents, pv, load, next_action}`. Without this, debugging a misbehaving night is hard.
- **Hysteresis dead-band.** Add ~50 W (or your inverter's noise floor) of dead-band around charge/discharge mode-transition boundaries so the orchestrator doesn't flap between modes as PV and load wiggle around equilibrium.
- **Forecast resilience.** Single PV-forecast source is a single point of failure. Production-grade orchestrators run primary (commercial: Solcast, Forecast.Solar, etc.) + physics-fallback (Open-Meteo `global_tilted_irradiance` × DC_kWp × eta, clipped to inverter AC max) + a daily auto-calibration step (EMA-update of a correction factor from real-vs-modeled PV).
- **State between ticks.** Use `flow.set(...)` / `flow.get(...)` (not `context.set/get`) so the values survive Node-RED redeploys of unrelated tabs.
- **Length of price arrays.** `load_cost_forecast` and `prod_price_forecast` must have at least `horizon_steps` entries, otherwise EMHASS pads / truncates and you may not notice silent misalignment.
- **Battery SOC unit.** EMHASS expects `soc_init` as a fraction in [0, 1]. Most sensor sources publish percent; divide by 100. See [Battery-aware runtime params](battery_aware_runtime_params.md) for the full story.

## Credits

- Prior art: long-form MPC walkthrough at `docs/study_cases/mpc.md`.
- Patterns extracted from author's production Node-RED setup (months in service; multiple EMHASS-version iterations). Generic patterns only — no private flow JSON, sensor names, IPs, secrets, or location-specific data copied.
- Field names verified against `src/emhass/utils.py:treat_runtimeparams` and `src/emhass/data/config_defaults.json` on 2026-05-11.
