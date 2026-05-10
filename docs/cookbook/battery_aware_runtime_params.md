# Battery-aware runtime params

## Goal

Feed live battery state of charge (SOC) back into EMHASS naive-MPC on every call so the optimizer plans against the real current state, not a stale assumption. Avoid the most common consumer bug: SOC unit mismatch (fraction vs percent).

## Prerequisites

- Battery is enabled in your static EMHASS config:

  ```yaml
  optim_conf:
    set_use_battery: true
  plant_conf:
    battery_target_state_of_charge: 0.6
    battery_minimum_state_of_charge: 0.3
    battery_maximum_state_of_charge: 0.9
    battery_discharge_power_max: 5000
    battery_charge_power_max: 5000
    battery_discharge_efficiency: 0.95
    battery_charge_efficiency: 0.95
  ```

  (Names per `src/emhass/data/config_defaults.json`.)
- A battery SOC sensor reachable from your orchestrator (Node-RED, AppDaemon, etc.). Common sources: inverter Modbus register, manufacturer cloud API, HA `sensor.battery_state_of_charge`. Any source works as long as you can read a number.
- An MPC orchestrator that already POSTs to `/action/naive-mpc-optim`. If you don't have one yet, see [MPC orchestration via Node-RED](nodered_mpc_orchestration.md).

## Config

<!-- source: src/emhass/data/config_defaults.json:109-117 -->

The static battery block above is the input contract. The single runtime param this recipe drives is `soc_init`:

<!-- source: src/emhass/utils.py:933 (treat_runtimeparams reads soc_init) -->

| Field | Type | Range | Notes |
|---|---|---|---|
| `soc_init` | `float` | `[battery_minimum_state_of_charge, battery_maximum_state_of_charge]` | Fraction 0..1, NOT percent. EMHASS logs a warning and refuses values outside [SOCmin, SOCmax]. |

EMHASS will reject `soc_init` outside the configured min / max bounds:

<!-- source: src/emhass/utils.py:937-944 (soc_init bound-check) -->

- If `soc_init < battery_minimum_state_of_charge`: EMHASS warns "Passed soc_init=... is lower than soc_min=..., keeping real initial SOC for optimization recovery" and falls back.
- If `soc_init > battery_maximum_state_of_charge`: equivalent warning, same fallback.

## Snippet

<!-- transport: Node-RED 3.1 (tested) — patterns translate directly to AppDaemon, Python, etc. -->

Append to the runtime_params your MPC orchestrator already sends. Generic Node-RED `function` node fragment:

```javascript
// Read whatever sensor exposes battery SOC. Examples:
//   const soc_percent = flow.get("battery_soc_percent");      // your-stack-specific
//   const soc_percent = msg.payload;                          // if previous node was a sensor read
//   const soc_percent = global.get("home_battery").soc;       // global context store

const soc_percent = flow.get("battery_soc_percent") || 50;

// EMHASS expects fraction. Divide percent by 100.
const soc_init = soc_percent / 100;

// Validate against configured min/max BEFORE sending — saves a round-trip warning log.
const SOC_MIN = 0.3;   // your battery_minimum_state_of_charge
const SOC_MAX = 0.9;   // your battery_maximum_state_of_charge
const soc_init_clamped = Math.max(SOC_MIN, Math.min(SOC_MAX, soc_init));

if (soc_init_clamped !== soc_init) {
  node.warn(`SOC ${soc_init} clamped to [${SOC_MIN}, ${SOC_MAX}]`);
}

msg.payload = msg.payload || {};
msg.payload.soc_init = soc_init_clamped;
return msg;
```

Wire this `function` node in series before the `http request` node from the [Node-RED MPC orchestration recipe](nodered_mpc_orchestration.md).

## Caveats

The following are observed-in-production patterns from months running this normalization across multiple sensor sources.

- **#1 bug: percent vs fraction.** EMHASS works in fraction (0..1). Most sensor sources expose percent (0..100). Always check the unit. Symptoms of getting it wrong: optimizer plans aggressive discharge (thinks battery is "full" at 80 because it sees 0.8 as 80% margin headroom), or plans aggressive charge (thinks battery is empty). See the [Plan-output schema](../plan_output_schema.md) for the symmetric output-side scaling trap on `SOC_opt`.
- **Dual-format robustness.** The same sensor in different transport stacks may publish fraction *or* percent. Pattern that handles both without branching:

  ```javascript
  const socPct = soc <= 1 ? soc * 100 : soc;
  ```

  Treat any value ≤ 1 as fraction-form and scale; anything else as already-percent. Production has seen the same battery exposed as fraction over Modbus and as percent over the inverter's cloud API on the same day.
- **Defensive fallback.** Battery sensors regularly return garbage during startup, after BMS calibration, or during firmware updates (`NaN`, negative values, or `> 100`). Prefer a known-safe default over propagating bad data:

  ```javascript
  let soc = parseFloat(msg.payload) || 0;
  if (soc < 0 || soc > 100 || isNaN(soc)) soc = 50;
  ```

  A `NaN` propagated into `runtime_params.soc_init` can crash the parsing layer between your orchestrator and EMHASS *before* EMHASS gets the chance to reject the bound-violation cleanly.
- **Stale sensor.** If your battery sensor publishes only on change and your MPC ticks regularly (e.g. every 5-15 min), a long idle period can serve a stale SOC. Wire a `delay` node with `last value` semantics, or read a "last_updated" timestamp and reject readings older than ~2× MPC period.
- **Bound rejection is silent in the optimizer.** EMHASS only logs a warning when `soc_init` is out of range; the solve continues with the fallback (`battery_target_state_of_charge`). If you depend on the value being honored, validate before sending (the snippet above does this).
- **Hardware BMS still owns safety.** EMHASS does not enforce battery safety limits — it computes a *plan*. Your battery's BMS / inverter must still enforce its own thermal, voltage, and current limits. EMHASS plans things the hardware can refuse.
- **SOC at horizon end.** EMHASS plans to land at `battery_target_state_of_charge` by horizon end by default. If you want a different terminal SOC for a specific call, pass `soc_final` in runtime params. (Out of scope for this recipe — see EMHASS naive-MPC docs for the full runtime-param list.)

## Credits

- SOC fraction-vs-percent gotcha discovered while building [PR #835 plan-output schema doc](https://github.com/davidusb-geek/emhass/pull/835). See `docs/plan_output_schema.md` (once #835 merges) for the symmetric output-side story.
- Dual-format robustness + defensive fallback patterns extracted from author's production Node-RED setup (months in service). Generic only — no private sensor names, IPs, or stack-specific identifiers copied.
- Field names verified against `src/emhass/utils.py:treat_runtimeparams` and `src/emhass/optimization.py` battery constraints on 2026-05-11.
