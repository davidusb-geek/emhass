# EV charging with Node-RED, MQTT and evcc, without Home Assistant

## Goal

Plan the EV charge in EMHASS as one deferrable load and carry the plan out with **evcc**,
using **Node-RED** as the only glue: a flow reads live evcc state over MQTT, builds the
`naive-mpc-optim` runtime payload, reads the plan back, publishes the EV setpoints as
retained MQTT topics and switches the evcc loadpoint mode over REST. Home Assistant is not
in the data path. This is the Node-RED/MQTT variant of
[EMHASS as planner, evcc as executor](ev_evcc_executor.md); the EMHASS-side model, the
coupling direction and the deadline reasoning are the same, so that recipe is linked
rather than repeated.

## Prerequisites

- EMHASS version: ≥ 0.17.8 (`def_current_power`). Everything except that one key in Step 3
  also works on earlier releases.
- Optimization mode: `naive-mpc-optim`, `method_ts_round: first` (Step 1).
- evcc with its MQTT publisher enabled and the REST API reachable from Node-RED. evcc
  loadpoint ids are 1-based in both the MQTT topics and the REST paths.
- **evcc must NOT control the house battery.** Keep the battery out of evcc's meters; EMHASS
  owns battery dispatch.
- Node-RED 3.x, core nodes only (`inject`, `mqtt in/out`, `function`, `http request`, `rbe`,
  `catch`, `trigger`).
- Transport tested against: Node-RED 3.x, evcc 0.2xx/0.3xx (MQTT state topics, REST
  `/api/loadpoints/{id}/mode` and `/mincurrent`), EMHASS 0.18.x.

## Step 1: Verify your static EMHASS config

<!-- source: src/emhass/data/config_defaults.json:4-6,54-60,75-76,95 -->

The EV is the last deferrable load. Its hours and end step are placeholders here and are
overridden on every MPC call. `method_ts_round: first` makes the first optimization step
the step that contains the request, which the flow's step maths assumes (see
[Step indices and other positional inputs](../passing_data.md#step-indices-and-other-positional-inputs)).

```yaml
optimization_time_step: 30
method_ts_round: first
number_of_deferrable_loads: 3
nominal_power_of_deferrable_loads: [3000, 750, 11000]   # 11 kW = 3 x 16 A
operating_hours_of_each_deferrable_load: [4, 0, 0]       # EV hours set at runtime
start_timesteps_of_each_deferrable_load: [0, 0, 0]
end_timesteps_of_each_deferrable_load: [0, 0, 0]         # EV end set at runtime
treat_deferrable_load_as_semi_cont: [true, true, true]   # EV: on at nominal or off
```

Expected: EMHASS restarts cleanly and `naive-mpc-optim` returns a plan with a
`P_deferrable2` column.

## Step 2: Collect live evcc state over MQTT

<!-- transport: Node-RED 3.x (tested) -->

One `mqtt in` node subscribed to `evcc/loadpoints/1/+` feeds a `function` node that keeps
four values in flow context (plugged in, charging, charge power in W, vehicle SOC in %) plus
the arrival time of the SOC. The payload builder (Step 3) reads flow context, so the MPC
tick never waits on evcc.

```javascript
const map = { connected: "ev_connected", charging: "ev_charging",
              chargePower: "ev_charge_power_w", vehicleSoc: "ev_soc_percent" };
const key = map[msg.topic.split("/").pop()];
if (!key) return null;
const raw = String(msg.payload);
flow.set(key, raw === "true" ? true : raw === "false" ? false : Number(raw));
if (key === "ev_soc_percent") flow.set("ev_soc_ts", Date.now());
return null;
```

The house battery SOC and the price lists come from whatever source your stack already has;
store them the same way as `battery_soc_percent`, `load_cost_per_step` and
`prod_price_per_step`. The [Node-RED MPC orchestration recipe](transport_nodered_mpc_orchestration.md)
covers that part.

Expected: the Node-RED context sidebar shows the `ev_*` keys updating while the car is
plugged in.

## Step 3: Build the runtime payload

<!-- source: src/emhass/utils.py:2259 (treat_runtimeparams), :3286 (def_current_state), :3338 (def_current_power); src/emhass/optimization.py:761 (t=0 power pin) -->
<!-- transport: Node-RED 3.x (tested) -->

An `inject` node fires every MPC tick (15 min for a 30 min step) into this `function` node.
The EV is modelled as in the [EV study case](../study_cases/ev.md): energy to deliver =
(target SOC - live SOC) x usable capacity, deadline = next departure, both recomputed on
every call. Everything positional counts steps in elapsed time from the first optimization
step, which with `method_ts_round: first` is the step containing the tick.

```javascript
const step_min = 30, horizon = 48;                 // = optimization_time_step, prediction_horizon
const charger_kw = 11.0, capacity_kwh = 77.0;      // charger nominal, usable battery of the car
const target_soc = 80, departure_hhmm = "07:00";   // "be at 80 % by 07:00 local"
const soc_max_age_ms = 6 * 3600 * 1000;            // see Caveats: SOC staleness
const step_s = step_min * 60;
const now = Date.now();

// First optimization step as an epoch instant (UTC offset must be a whole number of steps).
const first_step = Math.floor(now / 1000 / step_s) * step_s;

// Next local departure, as an instant. Date handles the local clock, including DST.
const [hh, mm] = departure_hhmm.split(":").map(Number);
const dep = new Date(now); dep.setHours(hh, mm, 0, 0);
if (dep.getTime() <= now) dep.setDate(dep.getDate() + 1);
let end_step = Math.min(Math.floor((dep.getTime() / 1000 - first_step) / step_s), horizon);

// Remaining EV energy from evcc's live SOC. Fail closed: no SOC, stale SOC or unplugged -> 0 h.
const connected = flow.get("ev_connected") === true;
const soc = flow.get("ev_soc_percent");
const soc_fresh = Number.isFinite(soc) && (now - (flow.get("ev_soc_ts") || 0)) < soc_max_age_ms;
let ev_hours = 0;
if (connected && soc_fresh && end_step > 0) {
    const kwh = Math.max(0, (target_soc - soc) / 100 * capacity_kwh);
    ev_hours = Math.round(kwh / charger_kw * 100) / 100;       // fractional hours are honoured
    const max_hours = end_step * step_min / 60;
    if (ev_hours > max_hours) {                                // would be Infeasible: clamp and say so
        node.warn(`EV needs ${ev_hours} h but only ${max_hours} h remain before departure; clamping`);
        ev_hours = max_hours;
    }
}
if (ev_hours === 0) end_step = horizon;            // 0 would mean "no end limit", not "no EV"

const charging = flow.get("ev_charging") === true;
const charge_w = Number(flow.get("ev_charge_power_w")) || 0;
const soc_batt = Number(flow.get("battery_soc_percent"));
const load_cost = flow.get("load_cost_per_step"), prod_price = flow.get("prod_price_per_step");
const ok = (a) => Array.isArray(a) && a.length >= horizon && a.every((v) => Number.isFinite(v));
if (!ok(load_cost) || !ok(prod_price) || !Number.isFinite(soc_batt)) {
    node.warn("Skipping MPC tick: price lists or battery SOC missing or not finite");
    return null;                                   // fail closed, never send made-up inputs
}

flow.set("assumed_first_step", first_step);        // checked against the plan in Step 4
msg.payload = {
    prediction_horizon: horizon,
    soc_init: soc_batt / 100, soc_final: soc_batt / 100,
    load_cost_forecast: load_cost, prod_price_forecast: prod_price,
    operating_hours_of_each_deferrable_load: [4, 0, ev_hours],
    start_timesteps_of_each_deferrable_load: [0, 0, 0],
    end_timesteps_of_each_deferrable_load: [horizon, horizon, end_step],
    def_current_state: [false, false, charging],
    def_current_power: [0, 0, charging ? charge_w : 0],
};
return msg;
```

Wire the output to an `http request` node: `POST http://<emhass-host>:5000/action/naive-mpc-optim`,
send as JSON, return parsed JSON, timeout 120 s. `def_current_state` keeps EMHASS from
charging a startup penalty for a car that is already charging; `def_current_power` forces the
EV on at the first step (at nominal power for a semi-continuous load, at the measured power
for a modulated one) so a charge in progress is not planned off (the
[HA recipe, Step 5](ev_evcc_executor.md#step-5-handle-a-manually-started-or-mid-window-charge)
explains the pin). Both lists are per load in the Step 1 order.

Expected: HTTP 200, and `P_deferrable2` in the returned plan is non-zero only between now
and the departure step.

## Step 4: Read the plan back and publish the EV setpoints over MQTT

<!-- source: src/emhass/web_server.py:769 (/api/v1/plan) -->
<!-- transport: Node-RED 3.x (tested) -->

After a 200, a second `http request` node does `GET http://<emhass-host>:5000/api/v1/plan`.
The response is `{status, generated_at, plan: [...]}` with one record per step and a UTC
`timestamp` per record (see the [plan output schema](../plan_output_schema.md)). A `function`
node with four outputs turns the first record into retained topics, so any consumer gets
the last known state on reconnect:

```javascript
const p = msg.payload;
if (p.status !== "ok" || !Array.isArray(p.plan) || p.plan.length === 0) return null;
const first = p.plan[0];
if (Date.parse(first.timestamp) / 1000 !== flow.get("assumed_first_step")) {
    node.warn("Plan grid differs from the assumed first step; not acting on this plan");
    return null;                       // one grid per request, see passing_data.md
}
const ev_w = Number(first.P_deferrable2) || 0;
return [
    { topic: "emhass/ev/setpoint_w", payload: ev_w, retain: true },
    { topic: "emhass/ev/mode", payload: ev_w > 100 ? "now" : "off", retain: true },
    { topic: "emhass/ev/plan_generated_at", payload: p.generated_at, retain: true },
    { topic: "emhass/cycle_ok", payload: Date.now(), retain: true },
];
```

A separate `inject` node publishes `emhass/heartbeat` every tick regardless of the EMHASS
result, so a consumer can tell "Node-RED down" from "EMHASS down".

Expected: `mosquitto_sub -t 'emhass/#' -v` shows the four retained topics and a heartbeat.

## Step 5: Drive the evcc loadpoint over REST

<!-- transport: evcc 0.2xx/0.3xx (tested) -->

An `mqtt in` on `emhass/ev/mode`, an `rbe` node (send on change only) and an `http request`
node that posts the mode to evcc:

| `emhass/ev/mode` | REST call | Meaning |
|---|---|---|
| `now` | `POST http://<evcc-host>:7070/api/loadpoints/1/mode/now` | charge at the charger's configured rate during the planned slot |
| `off` | `POST http://<evcc-host>:7070/api/loadpoints/1/mode/off` | hold, the plan has no EV power this step |
| `pv` | `POST http://<evcc-host>:7070/api/loadpoints/1/mode/pv` | fallback: surplus only, used by the fail-safes in Step 6 |

evcc 0.316 renamed `pv` to `smart` on its read side; writing `pv` is still accepted and
selects `smart`. `now` and `off` are unchanged.

Once at flow start, also post the charger's minimum current so planner and executor agree
on the lowest non-zero power the car will draw:
`POST http://<evcc-host>:7070/api/loadpoints/1/mincurrent/6` (6 A x 3 x 230 V = 4.1 kW). If
you let EMHASS modulate the EV (`treat_deferrable_load_as_semi_cont: false` with
`minimum_power_of_deferrable_loads`), use that same watt value there, otherwise EMHASS plans
a floor evcc cannot deliver.

Expected: `GET http://<evcc-host>:7070/api/state` shows `loadpoints[0].mode` following the
retained topic within one tick, and the car charges only in the planned slots. evcc's live
SOC flows back through Step 2 into the next MPC call.

## Step 6: Fail-safes

<!-- transport: Node-RED 3.x (tested) -->

Three guards, one safe state: evcc in `pv` mode, surplus only, no grid power on a stale or
missing plan.

- **Watchdog on the plan.** A `trigger` node is reset by every `emhass/cycle_ok` message
  and fires after 2 x the MPC cadence with no reset. Its output posts `mode/pv` and
  publishes `emhass/ev/mode` = `pv` retained, so a stale `now` is not replayed on reconnect.
- **HTTP failure.** A `catch` node on both `http request` nodes routes to the same `pv`
  branch. EMHASS returns a JSON error body on 400 or 500; log it before switching.
- **Stale plan on restart.** The Step 5 consumer ignores a `now` whose retained
  `emhass/ev/plan_generated_at` is older than 2 x the cadence and posts `pv` instead. This
  covers a Node-RED restart replaying retained topics from before an outage.

Expected: stop the EMHASS container for 40 minutes; evcc switches to `pv` on its own and
follows the plan again after the next successful tick.

## Caveats

- **Coupling direction.** EMHASS plans, evcc executes. Do not also enable evcc's own
  charging plan or tariff logic on this loadpoint; two planners fight.
- **DST and the deadline.** The departure instant is computed by Node-RED on the local
  clock and converted into elapsed steps from the first optimization step; EMHASS
  `delta_forecast_daily` uses local calendar-day semantics and `prediction_horizon` is a
  fixed step count. If the departure must remain the same local wall-clock time across a
  daylight-saving transition, validate the computed `end_step` against the plan's timestamps
  that night; do not assume EMHASS forecast-window handling changes the flow's arithmetic.
  The epoch floor for the first step assumes a UTC offset that is a whole number of steps
  (see [Step indices and other positional inputs](../passing_data.md#step-indices-and-other-positional-inputs)).
- **SOC staleness.** evcc refreshes `vehicleSoc` only while the car answers; a sleeping car
  can report an hours-old value. Step 3 treats an SOC older than `soc_max_age_ms` as missing
  and plans 0 h. Pick the age from how often your vehicle API actually answers; a car that
  never refreshes is never charged, which is the safe failure but not the wanted one.
- **A missed deadline rolls into the next day.** Leftover energy is then planned into the
  cheapest slots before tomorrow's departure, often midday PV. If the charge must stay
  overnight, bound `start_timesteps`/`end_timesteps` to the overnight window.
- **No within-slot modulation by default.** With `treat_deferrable_load_as_semi_cont: true`
  the plan is on-at-nominal or off per step and `now` enforces exactly that. Use `pv` as the
  fallback, not as the normal mode, or the deadline is no longer guaranteed (the HA recipe's
  Step 4 discusses the trade-off).
- **The plan grid check is strict.** Step 4 refuses a plan whose first timestamp is not the
  step the flow assumed, which happens when the tick lands on a step boundary. Offset the
  `inject` node by a minute from the boundary rather than relaxing the check.
- **Secure the control endpoints.** EMHASS `/action/*` and the evcc loadpoint API both
  command real loads. Keep them on the local network or behind a reverse proxy with TLS and
  authentication; recent evcc versions can require an API key on the REST side.
- **Battery SOC unit.** `soc_init` and `soc_final` are fractions in [0, 1]; both are sent so
  the end-of-horizon constraint stays neutral (see
  [Good Practices, SOC semantics](../study_cases/good_practices.md#4-soc_init-and-soc_final-runtime-semantics)).

## Credits

- Pattern seed: Discussions [#789](https://github.com/davidusb-geek/emhass/discussions/789)
  (maintainer invitation for a Node-RED/evcc recipe) and
  [#824](https://github.com/davidusb-geek/emhass/discussions/824) (cookbook).
- HA/Jinja variant and deadline reasoning: [ev_evcc_executor.md](ev_evcc_executor.md).
  Payload scaffold: [transport_nodered_mpc_orchestration.md](transport_nodered_mpc_orchestration.md).
- Generic patterns only, extracted from a production Node-RED flow. No private flow JSON,
  topic prefixes, host names or vehicle data copied.
- Field names verified against `src/emhass/static/data/runtime_params.json` and
  `src/emhass/utils.py:treat_runtimeparams` on 2026-10-17.
