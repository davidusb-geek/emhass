# EV charging as a deferrable load

> **Type:** How-To Guide — task-oriented.
>
> ```{warning}
> **Early-draft page.** EV charging in EMHASS is an active community topic
> ([discussion #789](https://github.com/davidusb-geek/emhass/discussions/789)).
> The pattern below — treating the charger as a windowed deferrable load —
> covers the common case but has known limits. See *Known limits* at the
> end of this page before relying on it. Contributions and corrections
> welcome.
> ```

EMHASS treats an EV charger as a windowed deferrable load: a fixed energy amount that must be delivered before some end time, with charging power up to the charger's rated nominal power. The optimizer then chooses *when* in the available window to charge.

This page covers the **EMHASS-side configuration**. The transport — how the EV's required-energy and target-SOC reach EMHASS at runtime — depends on your charger and home-automation setup. Any source that can publish the right signals into Home Assistant (or directly into your `rest_command` payload) will work: OCPP-managed chargers, vehicle-API integrations (Tesla, Hyundai BlueLink, etc.), Modbus chargers, EVCC, or pure HA scripts ([one community example](https://sigenergy.annable.me/emhass/)).

## Scenario

| Component | Value |
|-----------|-------|
| EV charger | AC, single fixed charging power for the whole session |
| Vehicle | a target energy amount in kWh that must be delivered before a deadline |
| Existing system | PV + battery (per [Basic — PV + Battery](basic_pv_battery.md)) |
| Optimization mode | naive-mpc-optim |

## Configuration

The EV slot goes into `nominal_power_of_deferrable_loads`, with corresponding entries in `operating_hours_of_each_deferrable_load`, `start_timesteps_of_each_deferrable_load`, and `end_timesteps_of_each_deferrable_load`. Most of these are placeholders in the static config and get overridden per MPC call.

Example (assuming the EV is the third deferrable, after two existing loads):

```yaml
method_ts_round: first   # the step indices below count from the first optimization step
nominal_power_of_deferrable_loads:
  - 3000        # water heater
  - 750         # pool pump
  - <CHARGER_W> # EV charger nominal power, e.g. 3700 (1ph 16A), 7400 (1ph 32A), 11000 (3ph 16A); charger_kw in the script below = this / 1000
operating_hours_of_each_deferrable_load:
  - 5
  - 8
  - 0           # placeholder; overridden at runtime
start_timesteps_of_each_deferrable_load:
  - 0
  - 0
  - 0           # placeholder; overridden at runtime
end_timesteps_of_each_deferrable_load:
  - 48
  - 48
  - 0           # placeholder; overridden at runtime
```

## Runtime payload

Two values you compute fresh per MPC call, both counted on the EMHASS time grid (see [Step indices and other positional inputs](../passing_data.md#step-indices-and-other-positional-inputs)):

| Variable | Source | Calculation |
|----------|--------|-------------|
| `operating_timesteps_of_each_deferrable_load[2]` | a sensor that reports remaining kWh to deliver | `ceil(kWh_remaining / (charger_kW x step_hours))`, a whole number of steps |
| `end_timesteps_of_each_deferrable_load[2]` | departure deadline | whole steps that end at or before the deadline, counted from the **first optimization step**: `floor((deadline - first_step) / step)` in elapsed seconds, one fewer in the last 60 s before a step boundary (see the script) |

Three rules keep the deadline strict:

- **Count elapsed time from the first step, not the clock reading.** Subtracting two local datetimes (`deadline - now()`) is wall-clock arithmetic: across a daylight saving change it is an hour off, and it counts from `now()` while the step indices count from the first step. Use `as_timestamp()` and the first step of the grid (set `method_ts_round` to `first`; with the default `nearest`, a call in the second half of a step starts one step later than `first_step` and the EV could run one step after the deadline). The script counts one step fewer when it runs in the last 60 s before a step boundary, so, provided EMHASS receives the request within 60 s of the script's clock reading, a reading on the other side of the boundary can only make the EV stop earlier, never later. Check the first timestamp of the returned plan against `first_step` before acting on it.
- **`end_timesteps_of_each_deferrable_load = 0` means no limit, not "stop now".** A deadline that is less than one full step after the first step (or exactly one step, in the last 60 s before a boundary) gives `0`, and EMHASS then plans the charge at the cheapest slot of the whole horizon, after the deadline. The script below does not send an EV window in that case, and it does not turn an invalid deadline time into no limit: `today_at()` fails on an invalid time and the script stops.
- **If the energy does not fit before the deadline, do not stretch the deadline.** `ev_steps > end_step` makes the problem infeasible (EMHASS returns `Infeasible` and publishes no plan, which also stops every other load). The script plans at most `end_step` steps, which charges in every step before the deadline, and raises a notification that the target will be missed. When less than one full step remains (`end_step = 0`) the EV is not planned. The charge is never silently planned after the deadline.

Why whole steps: with the default `treat_deferrable_load_as_semi_cont: true` a load runs at its nominal power or not at all. A requirement that is a whole number of steps is solved as a normal mixed-integer problem; a fractional requirement such as `2.2` hours at 30-minute steps is solved as `Optimal (Relaxed)`: the energy is exact, but the integer constraints are dropped and at least one step runs at a partial power (4.4 kW in a test with an 11 kW charger). Rounding up to whole steps delivers at most one step of extra energy (5.5 kWh for 11 kW and 30 minutes). Pass hours that are not a multiple of the step only if your charger can follow a partial-power step.

A Home Assistant script, source-agnostic. The `variables` are evaluated in order, and the other loads stay planned when the EV cannot be scheduled:

```yaml
script:
  emhass_mpc:
    sequence:
      - variables:
          charger_kw: 11.0
          timestep_min: 30                  # = optimization_time_step
          horizon: 48
          deadline_time: "07:00"
          ev_remaining_kwh: "{{ states('sensor.YOUR_EV_REMAINING_KWH') | float }}"
          step_s: "{{ timestep_min * 60 }}"
          # First step with method_ts_round "first": the latest step boundary not after now, on the local clock.
          now_ts: "{{ as_timestamp(now()) }}"          # one clock reading for first_step and margin
          utc_offset_s: "{{ now().utcoffset().total_seconds() }}"
          first_step: "{{ ((now_ts + utc_offset_s) // step_s) * step_s - utc_offset_s }}"
          # EMHASS reads the clock after this script. Within the last 60 s before a step boundary its first step may be the next one, which would put every index one step too late and let the EV run in the deadline step, so count one step fewer.
          margin: "{{ 1 if (now_ts + utc_offset_s) % step_s > step_s - 60 else 0 }}"
          # Next departure as an instant; elapsed seconds, not wall-clock arithmetic.
          deadline_ts: >
            {{ as_timestamp(today_at(deadline_time) if now() < today_at(deadline_time)
               else today_at(deadline_time) + timedelta(days=1)) }}
          end_step: "{{ [([((deadline_ts - first_step) // step_s) | int, horizon] | min) - margin, 0] | max }}"
          ev_steps: "{{ [(ev_remaining_kwh / (charger_kw * timestep_min / 60)) | round(6) | round(0, 'ceil') | int, 0] | max }}"
          ev_plan_steps: "{{ [ev_steps, end_step] | min }}"
          # The other loads, in steps too (see the note below on mixing hours and timesteps).
          wh_steps: "{{ (states('sensor.water_heater_remaining_hours') | float * 60 / timestep_min) | round(6) | round(0, 'ceil') | int }}"
          pool_steps: "{{ (states('sensor.pool_pump_remaining_hours') | float * 60 / timestep_min) | round(6) | round(0, 'ceil') | int }}"
      - if: "{{ ev_steps > end_step }}"
        then:
          - action: persistent_notification.create
            data:
              notification_id: emhass_ev_deadline
              title: EV cannot reach its target by the deadline
              message: "{{ ev_steps }} steps needed, {{ end_step }} available before the deadline. The EV is planned for {{ ev_plan_steps }} steps; about {{ ([ev_remaining_kwh - ev_plan_steps * charger_kw * timestep_min / 60, 0] | max) | round(1) }} kWh will not be delivered."
      - action: rest_command.emhass_mpc
        data:
          horizon: "{{ horizon }}"
          wh_steps: "{{ wh_steps }}"
          pool_steps: "{{ pool_steps }}"
          ev_steps: "{{ ev_plan_steps }}"
          ev_end: "{{ end_step if ev_plan_steps > 0 else horizon }}"

rest_command:
  emhass_mpc:
    url: http://localhost:5000/action/naive-mpc-optim
    method: POST
    timeout: 120
    headers:
      content-type: application/json
    payload: >
      {
        "prediction_horizon": {{ horizon }},
        "soc_init": {{ states('sensor.battery_soc') | float / 100 }},
        "operating_timesteps_of_each_deferrable_load": [{{ wh_steps }}, {{ pool_steps }}, {{ ev_steps }}],
        "end_timesteps_of_each_deferrable_load": [{{ horizon }}, {{ horizon }}, {{ ev_end }}]
      }
```

Replace `sensor.YOUR_EV_REMAINING_KWH` with the actual sensor your integration publishes. The contract is: a sensor that reports kWh remaining to deliver before the next deadline. `| float` has no default on purpose: an unavailable sensor makes the script fail instead of planning with an invented energy value.

```{note}
Do not mix the two ways of giving the operating requirement in one call. As soon as `operating_timesteps_of_each_deferrable_load` is passed, EMHASS ignores `operating_hours_of_each_deferrable_load` for **every** load: in a test, a heater given `1.0` hours next to an EV given timesteps ran for no step at all. Give **every** configured deferrable load as timesteps (`ceil(hours / step_hours)`), as above, or all as hours; the list needs one entry per deferrable load (use `0` for a load without a requirement): a shorter list makes the call fail with an `IndexError`. The setting is per request: in a test of four consecutive requests, a later request that passed only hours was not affected by an earlier request's timesteps.
```

```{note}
The deadline is evaluated by Home Assistant/Jinja as an instant, in your configured time zone. A departure time inside the repeated hour of a daylight saving change resolves to its first occurrence, and a time inside the skipped hour resolves to the instant that wall-clock time would have with the pre-change offset (03:30 local in the Sydney spring-forward example); both were checked in Home Assistant for Australia/Sydney. Neither changes how EMHASS builds its own forecast window (`delta_forecast_daily`, local calendar days).
```

## Output

`sensor.p_deferrable2` carries the optimized EV charging power per timestep. An HA automation drives the charger:

```yaml
automation:
  - alias: EV charging follow EMHASS plan
    trigger:
      - platform: state
        entity_id: sensor.p_deferrable2
    action:
      - choose:
          - conditions:
              - condition: numeric_state
                entity_id: sensor.p_deferrable2
                above: 100
            sequence:
              - service: switch.turn_on
                target:
                  entity_id: switch.YOUR_EV_CHARGER
          - default:
              - service: switch.turn_off
                target:
                  entity_id: switch.YOUR_EV_CHARGER
```

Replace `switch.YOUR_EV_CHARGER` with the actual control entity for your charger.

## Known limits

Read these before designing around the pattern above.

- **Fractional hours are honoured, as energy.** `operating_hours_of_each_deferrable_load` accepts fractional values (e.g. `1.25`); EMHASS targets exactly `nominal_power × hours` of energy ([issue #373](https://github.com/davidusb-geek/emhass/issues/373)). A value that isn't a whole number of timesteps is met exactly, but with the default semi-continuous loads the solve is reported as `Optimal (Relaxed)` and at least one step runs below the nominal power (see the Runtime payload section). This page rounds up to whole timesteps and passes `operating_timesteps_of_each_deferrable_load` instead, which keeps every step at the nominal power at the cost of up to one step of extra energy.
- **Power modulation is not supported.** EMHASS schedules the deferrable as either-on-at-nominal-power-or-off per timestep. Modulating chargers (e.g. Tesla 3.6–11 kW continuous) cannot be cleanly expressed as a single deferrable; you would need multiple parallel deferrable slots at different powers, or model only the upper bound.
- **Mid-session state is not forced.** `def_current_state` flags the load as currently-on for startup-penalty purposes but does not force the optimizer to plan a non-zero power for the first timestep ([issue #605](https://github.com/davidusb-geek/emhass/issues/605)). If you start charging manually mid-window, the optimizer may plan to stop and re-start.
- **No native vehicle-API integration.** EMHASS does not query the vehicle SOC directly; you provide kWh-remaining via a sensor. The community has seen this gap and discussed approaches in [#789](https://github.com/davidusb-geek/emhass/discussions/789), including a fork by @tomvanacker85 with calendar-driven SOC targets. None of those have been upstreamed.
- **Opportunistic / surplus-only charging is awkward.** If you want "charge only from PV surplus, no grid", the deferrable-load model is not the right fit — the optimizer treats the kWh-target as mandatory. Surplus-only charging is typically handled outside EMHASS (in EVCC's PV mode, or HA automations) with EMHASS optimizing the rest of the system around it.

## Infeasibility

If `(end_step - start_step) × charger_kw × step_hours < required_kwh`, the optimization is infeasible. EMHASS returns `optim_status: "Infeasible"` and publishes nothing, for every load, not only the EV. The Runtime payload script checks the window before the call; it does not check `maximum_power_from_grid` (default 9000 W, below an 11 kW charger), which can also make a night charge without PV or battery infeasible. Remember that `end_timesteps_of_each_deferrable_load = 0` never makes a load infeasible: it removes the limit. See [Good Practices — infeasibility triage](good_practices.md).

## See also

- How-to: [MPC walkthrough](mpc.md) for the underlying MPC pattern
- How-to: [Heat-pump walkthrough](heat_pump_walkthrough.md) — combine EV + heat-pump in one system
- Reference: [Passing data](../passing_data.md) for all runtime params
- Reference: [Automations](../automations.md) — `shell_command` / `rest_command` patterns
- Discussion: [#789 — EV charging addon capability](https://github.com/davidusb-geek/emhass/discussions/789)
- Explanation: [Good Practices](good_practices.md) — infeasibility triage
