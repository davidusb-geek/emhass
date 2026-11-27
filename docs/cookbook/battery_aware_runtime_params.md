# Battery-aware runtime params

## Goal

Send the measured battery state of charge (SOC) to EMHASS naive-MPC on every call, as a fraction, so the plan starts from the real battery state. A tick with a missing or unusable reading sends nothing.

## Prerequisites

- EMHASS 0.19.0. Older versions handle an out-of-range SOC differently (see the changelog for 0.17.2 and 0.18.6).
- Battery is enabled in your EMHASS configuration (see Step 1 below)
- A battery SOC sensor reachable from your orchestrator (Node-RED, AppDaemon, etc.), for example an inverter Modbus register, a manufacturer cloud API or a Home Assistant sensor. You need to know whether it reports percent or a fraction.
- An MPC orchestrator that already POSTs to `/action/naive-mpc-optim`. If you don't have one yet, see [MPC orchestration via Node-RED](transport_nodered_mpc_orchestration.md). The JavaScript below goes in its Step 3 `function` node and replaces the battery SOC lines there.

## Step 1: Check your static battery config

<!-- source: src/emhass/data/config_defaults.json:145-155 -->

Check these keys on the configuration page of the EMHASS web UI, which saves them to `config.json`. With the Home Assistant app, they are on that page, not on the app's Configuration tab.

```json
{
  "set_use_battery": true,
  "battery_nominal_energy_capacity": 5000,
  "battery_minimum_state_of_charge": 0.3,
  "battery_maximum_state_of_charge": 0.9,
  "battery_target_state_of_charge": 0.6,
  "battery_discharge_power_max": 5000,
  "battery_charge_power_max": 5000,
  "battery_discharge_efficiency": 0.95,
  "battery_charge_efficiency": 0.95
}
```

The single runtime param this recipe drives is `soc_init`:

<!-- source: src/emhass/utils.py:3229-3257 (naive-mpc-optim reads soc_init for one battery) -->

| Field | Type | Range | Notes |
|---|---|---|---|
| `soc_init` | `float` | 0 to 1 | Fraction of nominal capacity, NOT percent. A value above 1 is rejected with an error. A value outside your configured minimum and maximum is kept as the starting SOC (see [Caveats](#caveats)). If omitted, EMHASS uses `battery_target_state_of_charge`. |

The code in this recipe handles one battery. With `number_of_batteries` above 1, a single number is applied to every battery: send a list with one fraction per battery, in order, apply Steps 3 and 4 to each reading, and skip the tick if any one of them fails. The results are then `SOC_opt_{k}` and `sensor.soc_batt_forecast_battery{k}`, with `k` starting at 0.

Expected: `GET /get-config` (with your [API token](../api_authentication.md)) returns these keys.

## Step 2: Read battery SOC from your source

<!-- transport: Node-RED function node. Steps 2 to 5 were run as one function body in Node.js 24; untested inside Node-RED -->

Read the SOC from wherever your stack exposes it, and state the unit of that source. Do not work the unit out from the value: `1` means 1% from a percent sensor and a full battery from a fraction sensor.

```javascript
// Unit of this one source: 100 if it reports percent (0 to 100), 1 if it reports a fraction (0 to 1).
const soc_full_scale = 100;
// A source that gives only the value (adapt the key to your stack):
const soc_reading = { value: flow.get("battery_soc_percent") };   // typical HA / Modbus poll
// A source that also gives the time it measured: have the node that reads it store both in ONE write,
// flow.set("battery_soc_reading", { value: ..., measured_ms: ... }), and use this line instead:
// const soc_reading = flow.get("battery_soc_reading") || {};
const soc_raw = soc_reading.value;
```

Expected: `soc_raw` holds the reading exactly as the source reports it, and `soc_full_scale` matches its unit.

(step-3-normalize-to-fraction-the-dual-format-trick)=
## Step 3: Validate and convert to a fraction

Sensors can return `unavailable`, empty or impossible values during startup, calibration or firmware updates. Skip the tick instead of sending a substitute.

```javascript
// A number, or a string holding one plain decimal number ("74", "74.5", "1e1"; not "74%", "0x4A" or "7_4").
const decimal = /^\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\s*$/;
const soc_value = typeof soc_raw === "number" || (typeof soc_raw === "string" && decimal.test(soc_raw))
    ? Number(soc_raw) : NaN;
if ((soc_full_scale !== 100 && soc_full_scale !== 1)
    || !Number.isFinite(soc_value) || soc_value < 0 || soc_value > soc_full_scale) {
    node.warn(`Skipping MPC tick: unusable battery SOC "${soc_raw}" for soc_full_scale ${soc_full_scale} (must be 100 or 1)`);
    return null;                                   // nothing is sent
}
const soc_init = soc_value / soc_full_scale;
```

Expected: with `soc_full_scale = 100`, readings of `0`, `1`, `74` and `100` give `0`, `0.01`, `0.74` and `1`. A missing, empty, boolean, non-numeric, negative or above-full-scale reading stops the flow with a warning. The limits are the physical ones, not your configured minimum and maximum.

(step-4-defensive-validation)=
## Step 4: Reject old readings

Step 3 shows that a value is a plausible number, not that it is current: a reading can look valid long after the source stopped measuring. If your source provides the time of the measurement itself, check it. Use a time stamped by the device or its API when it measured, in epoch milliseconds, taken from the same message as the value. The time the value last changed is not enough (SOC can stay constant for hours), and neither is the time your flow or Home Assistant received it (a cached or retained value looks new). Choose the limit from how often that source really measures and how fast your battery's SOC can move.

`measured_ms` must be a JavaScript number: digits sent as text are rejected. If your source sends text and you have confirmed that it is epoch milliseconds, store `Number(...)` of it.

```javascript
const soc_max_age_ms = null;      // null: age is NOT checked. Set it only if measured_ms is the source's measurement time
const soc_max_ahead_ms = 60000;   // how far the source clock may run ahead of this one
if (soc_max_age_ms !== null) {
    const measured_ms = soc_reading.measured_ms;
    const age_ms = Number.isFinite(measured_ms) ? Date.now() - measured_ms : NaN;
    if (!(age_ms >= -soc_max_ahead_ms && age_ms <= soc_max_age_ms)) {
        node.warn("Skipping MPC tick: battery SOC measurement time is missing, too old or ahead of this clock");
        return null;
    }
}
```

Expected: with a limit set (for example `15 * 60000`), an old, missing, seconds-based or non-number measurement time (including digits as text) stops the flow with a warning. With `null`, an old but plausible value is sent; where your source reports its own availability, check that as well.

## Step 5: Attach to runtime_params

What you need here depends on where Steps 2 to 4 went:

- **In the orchestrator's Step 3 function:** use Steps 2 to 4 only, in place of its battery SOC lines. Keep its `msg.payload` object, which already contains `soc_init: soc_init`, and its `return msg`. Do not add the code below.
- **In a standalone function:** end with the code below. It starts a new object, so that a sensor reading left in `msg.payload` is not sent as runtime params. Add your other runtime params to it, including `prediction_horizon`.

```javascript
msg.payload = { soc_init: soc_init };
return msg;
```

Expected: the `http request` node receives `soc_init` as a fraction from 0 to 1. On a skipped tick it receives nothing, so this flow creates no new plan. The earlier plan is not removed and another caller can still replace it: whatever acts on the plan should check when it was made (see [publish-data](../publish_data.md)).

## Reading the result

<!-- source: src/emhass/command_line.py:4650-4660 (SOC_opt is published times 100) -->

The action response is not the plan. After [`publish-data`](../publish_data.md), `sensor.soc_batt_forecast` shows SOC in percent; the `SOC_opt` column of the plan is a fraction ([Plan-output schema](../plan_output_schema.md#soc_opt-scaling-callout)). Each value is the SOC at the **end** of its time step, so the first row is not your `soc_init` unless the battery is idle in that step.

## Caveats

<!-- source: src/emhass/optimization.py:5013-5055 and 9904-9951 (recovery into the SOC band) -->

- **Unit per source.** The same battery can be published as percent by one integration and as a fraction by another. Set `soc_full_scale` for the source you read and check it again if you change source or firmware. The range check catches percent values above 1 arriving on a fraction source, but not fractions arriving on a percent source: `0.74` would be taken as 0.74%.
- **Outside the configured band.** Send the measured value; do not clamp it to your minimum or maximum, or the plan starts from a battery state that does not exist. EMHASS logs a warning and plans a return into the band, at the time it finds economic, not necessarily at once. The plan has to be back inside the band by its last step. If it cannot get there, for example because the horizon is too short for the available power, the run is infeasible. Batteries with availability or discharge-energy limits are treated differently (see [Good Practices, section 5](../study_cases/good_practices.md#5-optim_status-infeasible-triage-order)).
- **`soc_final` is separate.** This recipe does not send it, so EMHASS uses `battery_target_state_of_charge` for the end of the horizon. Send `soc_final` as well if you need a different end target.
- **Hardware BMS still owns safety.** EMHASS does not enforce battery safety limits: it computes a *plan*. Your battery's BMS / inverter must still enforce its own thermal, voltage, and current limits. EMHASS plans things the hardware can refuse.

## Credits

- SOC fraction-vs-percent gotcha discovered while building the [plan-output schema doc](https://github.com/davidusb-geek/emhass/pull/835). See [Plan-output schema](../plan_output_schema.md) for the output side.
- Recipe originally drawn from the author's production Node-RED setup. Generic only: no private sensor names, IPs, or stack-specific identifiers copied.
- Field names verified against `src/emhass/utils.py:treat_runtimeparams` and `src/emhass/optimization.py` battery constraints on 2026-11-26.
