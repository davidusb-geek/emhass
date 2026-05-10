# Battery-aware runtime params

## Goal

Feed live battery state of charge (SOC) back into EMHASS naive-MPC on every call so the optimizer plans against the real current state, not a stale assumption. Avoid the most common consumer bug: SOC unit mismatch (fraction vs percent). After this recipe, every MPC POST your orchestrator sends carries the current `soc_init`, validated and clamped to your configured bounds.

## Prerequisites

- Battery is enabled in your static EMHASS config (see Step 1 below)
- A battery SOC sensor reachable from your orchestrator (Node-RED, AppDaemon, etc.). Common sources: inverter Modbus register, manufacturer cloud API, HA `sensor.battery_state_of_charge`. Any source works as long as you can read a number.
- An MPC orchestrator that already POSTs to `/action/naive-mpc-optim`. If you don't have one yet, see [MPC orchestration via Node-RED](transport_nodered_mpc_orchestration.md). This recipe adds the battery branch to that orchestrator.

## Step 1: Check your static battery config

<!-- source: src/emhass/data/config_defaults.json:109-117 -->

Verify your EMHASS `config.yaml` (or Web Config) has the battery block with the canonical keys:

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

The single runtime param this recipe drives is `soc_init`:

<!-- source: src/emhass/utils.py:933-944 (treat_runtimeparams reads + bound-checks soc_init) -->

| Field | Type | Range | Notes |
|---|---|---|---|
| `soc_init` | `float` | `[battery_minimum_state_of_charge, battery_maximum_state_of_charge]` | Fraction 0..1, NOT percent. EMHASS logs a warning and silently falls back to `battery_target_state_of_charge` for values outside the configured min / max bounds. |

Expected: EMHASS starts cleanly, `GET /api/get-config` returns the battery block, and a baseline MPC call without `soc_init` lands at `battery_target_state_of_charge` in the result.

## Step 2: Read battery SOC from your source

In the `function` node where your orchestrator builds `runtime_params` (Step 3 of the [MPC orchestration recipe](transport_nodered_mpc_orchestration.md)), read the SOC from wherever your stack exposes it. Examples:

```javascript
// Pick ONE of these — adapt to your stack:
const soc_raw = flow.get("battery_soc_percent");          // typical HA / Modbus poll
// const soc_raw = msg.payload;                            // if previous node was a sensor read
// const soc_raw = global.get("home_battery").soc;         // global context store
```

Expected: `soc_raw` is a number (possibly in either percent 0-100 OR fraction 0-1 depending on source).

## Step 3: Normalize to fraction (the dual-format trick)

Different transport stacks publish SOC in different units. Use a single guard that handles both without branching:

```javascript
// Robust against both fraction-form (Modbus, EMHASS-internal) AND percent-form (HA, cloud API).
// Anything ≤ 1 is treated as fraction and scaled; anything else is already-percent.
const soc_percent = soc_raw <= 1 ? soc_raw * 100 : soc_raw;

// EMHASS wants fraction:
const soc_init = soc_percent / 100;
```

Expected: regardless of whether the upstream sensor publishes `0.74` or `74`, `soc_init` ends up as `0.74`.

## Step 4: Defensive validation

Battery sensors regularly return garbage during startup, after BMS calibration, or during firmware updates (`NaN`, negative values, or `> 100`). Validate before sending:

```javascript
let soc = parseFloat(soc_init) || 0;
if (soc < 0 || soc > 1 || isNaN(soc)) {
    node.warn(`Bad SOC reading: ${soc_raw}, defaulting to 0.5`);
    soc = 0.5;
}

// Optional: clamp to configured bounds. Saves a round-trip EMHASS warning log.
const SOC_MIN = 0.3;   // your battery_minimum_state_of_charge
const SOC_MAX = 0.9;   // your battery_maximum_state_of_charge
const soc_init_clamped = Math.max(SOC_MIN, Math.min(SOC_MAX, soc));

if (soc_init_clamped !== soc) {
    node.warn(`SOC ${soc} clamped to [${SOC_MIN}, ${SOC_MAX}]`);
}
```

Expected: any of `NaN`, `-3`, `123` get caught and either rejected (set to 0.5) or clamped to bounds; the `node.warn` lines appear in the Node-RED debug pane so you notice when a sensor goes bad.

## Step 5: Attach to runtime_params

Add `soc_init` to the runtime-params object the http-request node will send:

```javascript
msg.payload = msg.payload || {};
msg.payload.soc_init = soc_init_clamped;
return msg;
```

Expected: when this function-node fires, `msg.payload.soc_init` is a fraction in `[SOC_MIN, SOC_MAX]`, and the downstream `http request` node POSTs it to EMHASS as part of the MPC call. EMHASS's response `SOC_opt` series now starts from this real-current value.

## Caveats

The following are observed-in-production patterns from months running this normalization across multiple sensor sources.

- **#1 bug: percent vs fraction.** EMHASS works in fraction (0..1). Most sensor sources expose percent (0..100). Symptoms of getting it wrong: optimizer plans aggressive discharge (thinks battery is "full" at 80 because it sees 0.8 as 80% margin headroom), or plans aggressive charge (thinks battery is empty). See the [Plan-output schema](../plan_output_schema.md) for the symmetric output-side scaling trap on `SOC_opt`.
- **Dual-format-aware code.** The same battery in different transport stacks can publish fraction *or* percent. Production has seen the same inverter exposed as fraction over Modbus and as percent over the manufacturer's cloud API on the same day. The Step-3 guard handles both — keep it even if your current source is one format only; it costs nothing and saves you a debugging session after a vendor firmware update flips the unit.
- **Defensive fallback matters.** A `NaN` propagated into `runtime_params.soc_init` can crash the parsing layer between your orchestrator and EMHASS *before* EMHASS gets the chance to reject the bound-violation cleanly. Always validate upstream.
- **Stale sensor.** If your battery sensor publishes only on change and your MPC ticks regularly (e.g. every 5-15 min), a long idle period can serve a stale SOC. Wire a `delay` node with `last value` semantics, or read a "last_updated" timestamp and reject readings older than ~2× MPC period.
- **Bound rejection is silent in the optimizer.** EMHASS only logs a warning when `soc_init` is out of range; the solve continues with the fallback (`battery_target_state_of_charge`). Step 4 catches this client-side so the rejection is visible.
- **Hardware BMS still owns safety.** EMHASS does not enforce battery safety limits — it computes a *plan*. Your battery's BMS / inverter must still enforce its own thermal, voltage, and current limits. EMHASS plans things the hardware can refuse.

## Credits

- SOC fraction-vs-percent gotcha discovered while building [PR #835 plan-output schema doc](https://github.com/davidusb-geek/emhass/pull/835). See `docs/plan_output_schema.md` (once #835 merges) for the symmetric output-side story.
- Dual-format robustness + defensive fallback patterns extracted from author's production Node-RED setup (months in service). Generic only — no private sensor names, IPs, or stack-specific identifiers copied.
- Field names verified against `src/emhass/utils.py:treat_runtimeparams` and `src/emhass/optimization.py` battery constraints on 2026-05-11.
