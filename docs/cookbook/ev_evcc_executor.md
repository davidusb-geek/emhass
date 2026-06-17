# EV charging with EMHASS as planner and evcc as executor

## Goal

Let EMHASS decide when and how much to charge an EV, as one deferrable load in the whole-home
optimization, and let **evcc** carry that plan out on the vehicle. EMHASS does the time-shifting:
which slots to charge in, against the tariff and the PV forecast. evcc is the vehicle-aware executor:
it talks to the car properly, reads live SOC, holds the charge current inside the charger's safe
limits, speaks the vehicle API or OCPP, and keeps the car awake. EMHASS stays the planning brain and
evcc is the actuator. The two never fight over the plan.

> This is the **opposite** coupling to
> [evcc-io/evcc#29815](https://github.com/evcc-io/evcc/discussions/29815), which proposes evcc
> calling EMHASS as its optimizer backend. Here evcc never optimizes the EV. It only carries out
> EMHASS's per-timestep decision. The two recipes can coexist later. This one works today with
> released evcc and EMHASS.

## Prerequisites

- EMHASS version: any current release (Steps 1-4). Step 5 (`def_current_power`) needs a release
  **after v0.17.7**: it is merged on master via
  [PR #982](https://github.com/davidusb-geek/emhass/pull/982) /
  [#605](https://github.com/davidusb-geek/emhass/issues/605) but not yet in a tagged release.
- Optimization mode: `naive-mpc-optim`.
- An evcc instance bound to the vehicle (a vehicle-API or OCPP charger). evcc holds its own vehicle
  credentials and reads SOC directly.
- **evcc must NOT control the house battery.** Exclude the battery from evcc's meters. EMHASS owns
  battery dispatch.
- Transport tested against: Home Assistant Core 2024.x, evcc 0.309 (default API port 7070). EMHASS
  REST is called via a HA `rest_command`.

## Step 1: Configure the EV as a deferrable load

<!-- source: src/emhass/data/config_defaults.json:42 -->

The EV is a windowed deferrable load: a fixed energy amount to deliver before a deadline, at up to
the charger's nominal power. This is the same base setup as the
[EV study case](../study_cases/ev.md), so start there for the full config. The slot looks like:

```yaml
number_of_deferrable_loads: 3
nominal_power_of_deferrable_loads:
  - 3000          # water heater
  - 750           # pool pump
  - 11000         # EV charger nominal power (e.g. 11 kW = 3-phase 16 A)
operating_hours_of_each_deferrable_load: [5, 8, 0]   # EV hours overridden at runtime
start_timesteps_of_each_deferrable_load: [0, 0, 0]
end_timesteps_of_each_deferrable_load:   [48, 48, 0] # EV end overridden at runtime
```

Expected: EMHASS restarts cleanly and `naive-mpc-optim` returns a plan with a third deferrable.

## Step 2: Feed live EV state into each MPC call

<!-- source: src/emhass/utils.py:1183 (treat_runtimeparams) -->
<!-- transport: Home Assistant Core 2024.x (tested) -->

Each MPC call computes the EV's remaining energy and deadline fresh. evcc reports live SOC, which the
evcc Home Assistant integration exposes as a sensor, so a HA `rest_command` can read it and translate
to EMHASS's `def_total_hours` (fractional hours are honoured, see Caveats) and `end_timesteps`. The
deadline is computed in the template, not read from evcc:

```yaml
rest_command:
  emhass_mpc:
    url: http://localhost:5000/action/naive-mpc-optim
    method: POST
    timeout: 120
    headers: {content-type: application/json}
    payload: >
      {%- set charger_kw = 11.0 -%}
      {%- set timestep_min = 30 -%}
      {%- set horizon = 48 -%}                         {# horizon_steps; the per-load arrays below are length = your deferrable count (3 here) #}
      {%- set ev_remaining_kwh = states('sensor.evcc_ev_remaining_kwh') | float(0) -%}
      {%- set ev_hours = (ev_remaining_kwh / charger_kw) | round(2) -%}
      {%- set deadline = today_at("05:00") if now() < today_at("05:00") else today_at("05:00") + timedelta(days=1) -%}
      {%- set end_step = ((deadline - now()).total_seconds() / 60 / timestep_min) | int -%}
      {
        "prediction_horizon": {{ horizon }},
        "soc_init": {{ states('sensor.battery_soc') | float(0) / 100 }},
        "def_total_hours": [ {{ states('sensor.wh_remaining_hours') | float(0) }}, {{ states('sensor.pool_remaining_hours') | float(0) }}, {{ ev_hours }} ],
        "end_timesteps_of_each_deferrable_load": [ {{ horizon }}, {{ horizon }}, {{ end_step }} ]
      }
```

The `sensor.*` names here are placeholders. Replace them with the entities your own setup publishes;
the `evcc_*` names are not guaranteed entity ids from the evcc integration. Every read uses
`| float(0)` on purpose, so an unavailable sensor degrades to 0 rather than injecting the literal
string `unknown` and producing invalid JSON.

Expected: the MPC response's `P_deferrable2` (the EV) is non-zero only inside the charging window.

## Step 3: Execute the plan with evcc, not a bare on/off switch

<!-- transport: evcc 0.309 (tested), Home Assistant Core 2024.x (tested) -->

EMHASS schedules the deferrable as on-at-nominal-or-off per timestep, and it has already done the
optimization: which slots to charge in, given the tariff and the PV forecast. Step 3 just enforces
that decision on the car. The reason to use evcc instead of a bare on-off switch is not extra
optimization, it is that evcc talks to the vehicle properly: it reads live SOC, holds the current
inside the charger's safe range, speaks the vehicle API or OCPP, and knows when the car is asleep. A
relay does none of that.

Define two `rest_command`s that set the evcc loadpoint mode. evcc listens on port 7070 by default,
and its loadpoint paths are 1-based, so the first loadpoint is `/loadpoints/1` even though
`/api/state` reports it as index 0:

```yaml
rest_command:
  evcc_mode_now:
    url: http://EVCC_HOST:7070/api/loadpoints/1/mode/now
    method: post
  evcc_mode_off:
    url: http://EVCC_HOST:7070/api/loadpoints/1/mode/off
    method: post
```

Then map EMHASS's per-timestep decision to a mode rather than a switch:

```yaml
automation:
  - alias: EV follow EMHASS plan (via evcc)
    trigger: [{platform: state, entity_id: sensor.p_deferrable2}]
    action:
      - choose:
          - conditions: [{condition: numeric_state, entity_id: sensor.p_deferrable2, above: 100}]
            sequence:
              - service: rest_command.evcc_mode_now
          - default:
              - service: rest_command.evcc_mode_off
```

`now` charges at the charger's configured rate during the slots EMHASS picked. EMHASS, not evcc,
decides when those slots are, so the tariff and PV optimization still happens, it just happens in the
planner. If you would rather evcc follow PV surplus within a slot, use the `pv` or `minpv` variant in
Step 4 and accept the trade-off it carries.

Expected: when the plan shows EV power, evcc starts or holds the charge; when the plan drops to zero,
evcc stops. evcc's live SOC feeds the next MPC call (Step 2), closing the loop. A sleeping car will
not respond to `now` until it is woken (see Caveats).

## Step 4: Overnight minimum-SOC-by-deadline (the "floor")

The common real requirement is "have the car at X% by 05:00, cheapest way possible, never during the
evening peak." Express it entirely through Steps 1-2: set the EV's required energy to
`(target_soc - current_soc) * battery_kWh` and `end_timesteps` to the 05:00 deadline. EMHASS then
places the charge in the cheapest in-window slots and leaves the peak alone, and Step 3 enforces it
with `now` during those slots.

If instead you want "PV surplus first, grid only as a last resort," that is an evcc-side choice and a
different trade-off. Run evcc in `pv` or `minpv` mode for the window in place of the Step 3 `now`
mapping, and keep a house-battery reserve so EMHASS doesn't plan the battery flat. The cost is that
`pv`/`minpv` throttles to available surplus, so evcc can deliver less than EMHASS planned and the
deadline target is no longer guaranteed. Use `now` (Step 3) when hitting the target matters more than
maximising self-supply, and `pv`/`minpv` when it is the other way round. Don't run both mappings at
once, they fight.

Expected: with the Step 3 `now` mapping the car reaches the target by the deadline using the cheapest
in-window slots; with the `pv`/`minpv` variant it charges from surplus and may stop short on a low-sun
day.

## Step 5: Handle a manually-started or mid-window charge

<!-- source: src/emhass/utils.py:1966 (def_current_power parse), src/emhass/optimization.py:399 (the t=0 pin) -->

If the driver plugs in and starts charging by hand mid-window, the house load sensor
(`sensor_power_load_no_var_loads`) excludes the charger, so `P_load[0]` handed to EMHASS is too low
and the optimizer may plan to switch off the load the driver just switched on. This is the
study-case "[Mid-session state is not forced](../study_cases/ev.md)" limit.

Pass the charger's **actual current power in watts** as `def_current_power` (a runtime-only per-load
list, in the same order as `nominal_power_of_deferrable_loads`). EMHASS pins the first timestep to
that power, forces the load ON at t=0, and suppresses the phantom startup penalty:

```jinja
{# add to the Step-2 payload, from evcc's live charge power #}
"def_current_power": [ 0, 0, {{ states('sensor.evcc_ev_charge_power_w') | float(0) }} ]
```

Expected: when the car is charging at, say, 6.8 kW, the plan's first EV timestep is pinned to 6.8 kW
and the optimizer schedules around it instead of stopping it.

## Caveats

- **Coupling direction.** This is evcc-as-executor under EMHASS, not the evcc-as-frontend proposal
  in [#29815](https://github.com/evcc-io/evcc/discussions/29815). Don't wire both.
- **No within-slot PV modulation by default.** EMHASS plans on-or-off-at-nominal per timestep (the
  [study-case power-modulation limit](../study_cases/ev.md)) and does the time-shifting; the Step 3
  `now` mapping enforces it at the charger's configured rate. If you want evcc to follow PV surplus
  inside a slot, use the `pv`/`minpv` variant in Step 4 and accept it may under-deliver versus the
  plan.
- **Deadline maths assumes no DST.** The Step 2 `timedelta(days=1)` is exactly 24 hours, so on a
  timezone that observes daylight saving the 05:00 deadline can land an hour off on the switch days.
  Adjust the template if that applies to you.
- **`def_current_power` needs a release after v0.17.7.** Before that, Step 5 doesn't apply and a
  manual mid-window charge may be re-planned off.
- **Keep evcc off the house battery.** Exclude the battery from evcc's meters. EMHASS owns it.
- **Fail safe.** An executor or transport failure should default to a safe state (a kill-switch plus
  a watchdog that stops commanding on stale data). See
  [Good Practices](../study_cases/good_practices.md).
- **Vehicle-API charging needs the car awake.** evcc does not auto-wake
  ([evcc#28652](https://github.com/evcc-io/evcc/issues/28652)); wake the car before commanding.

## Credits

- EV pattern seed: Discussions [#789](https://github.com/davidusb-geek/emhass/discussions/789) and
  [#824](https://github.com/davidusb-geek/emhass/discussions/824); coupling context
  [evcc#29815](https://github.com/evcc-io/evcc/discussions/29815).
- Mid-charge pin: issue [#605](https://github.com/davidusb-geek/emhass/issues/605) /
  PR [#982](https://github.com/davidusb-geek/emhass/pull/982).
- Prior art: [`docs/study_cases/ev.md`](../study_cases/ev.md).
- Field names verified against `src/emhass/utils.py` (`treat_runtimeparams`) on 2026-06-18.
