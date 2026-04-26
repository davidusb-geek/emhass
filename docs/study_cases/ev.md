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
nominal_power_of_deferrable_loads:
  - 3000        # water heater
  - 750         # pool pump
  - <CHARGER_W> # EV charger nominal power, e.g. 3700 (1ph 16A), 7400 (1ph 32A), 11000 (3ph 16A)
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

Two values you compute fresh per MPC call:

| Variable | Source | Calculation |
|----------|--------|-------------|
| `def_total_hours[2]` | a sensor that reports remaining kWh to deliver | `kWh_remaining / charger_kW`, rounded to whole hours (see *Known limits*) |
| `end_timesteps_of_each_deferrable_load[2]` | departure deadline | `(deadline - now) / optimization_time_step_minutes`, integer |

A Home Assistant `rest_command` template, source-agnostic:

```yaml
rest_command:
  emhass_mpc:
    url: http://localhost:5000/action/naive-mpc-optim
    method: POST
    timeout: 120
    headers:
      content-type: application/json
    payload: >
      {%- set charger_kw = 11.0 -%}
      {%- set timestep_min = 30 -%}
      {%- set horizon = 48 -%}
      {%- set ev_remaining_kwh = states('sensor.YOUR_EV_REMAINING_KWH') | float -%}
      {%- set ev_hours = (ev_remaining_kwh / charger_kw) | round(0, 'ceil') | int -%}
      {%- set deadline = today_at("07:00") if now() < today_at("07:00") else today_at("07:00") + timedelta(days=1) -%}
      {%- set departure_minutes = (deadline - now()).total_seconds() / 60 -%}
      {%- set end_step = (departure_minutes / timestep_min) | int -%}
      {
        "prediction_horizon": {{ horizon }},
        "soc_init": {{ states('sensor.battery_soc') | float / 100 }},
        "def_total_hours": [
          {{ states('sensor.water_heater_remaining_hours') }},
          {{ states('sensor.pool_pump_remaining_hours') }},
          {{ ev_hours }}
        ],
        "end_timesteps_of_each_deferrable_load": [{{ horizon }}, {{ horizon }}, {{ end_step }}]
      }
```

Replace `sensor.YOUR_EV_REMAINING_KWH` with the actual sensor your integration publishes. The contract is: a sensor that reports kWh remaining to deliver before the next deadline.

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

- **Integer hours only.** `operating_hours_of_each_deferrable_load` accepts whole hours, not fractions ([issue #373](https://github.com/davidusb-geek/emhass/issues/373)). With a 30-minute timestep and an 11 kW charger, that means kWh-required is rounded up to the next 11 kWh increment. Reduce the rounding error by using a smaller `optimization_time_step` (e.g. 15 minutes) — but then 48 timesteps becomes 96.
- **Power modulation is not supported.** EMHASS schedules the deferrable as either-on-at-nominal-power-or-off per timestep. Modulating chargers (e.g. Tesla 3.6–11 kW continuous) cannot be cleanly expressed as a single deferrable; you would need multiple parallel deferrable slots at different powers, or model only the upper bound.
- **Mid-session state is not forced.** `def_current_state` flags the load as currently-on for startup-penalty purposes but does not force the optimizer to plan a non-zero power for the first timestep ([issue #605](https://github.com/davidusb-geek/emhass/issues/605)). If you start charging manually mid-window, the optimizer may plan to stop and re-start.
- **No native vehicle-API integration.** EMHASS does not query the vehicle SOC directly; you provide kWh-remaining via a sensor. The community has seen this gap and discussed approaches in [#789](https://github.com/davidusb-geek/emhass/discussions/789), including a fork by @tomvanacker85 with calendar-driven SOC targets. None of those have been upstreamed.
- **Opportunistic / surplus-only charging is awkward.** If you want "charge only from PV surplus, no grid", the deferrable-load model is not the right fit — the optimizer treats the kWh-target as mandatory. Surplus-only charging is typically handled outside EMHASS (in EVCC's PV mode, or HA automations) with EMHASS optimizing the rest of the system around it.

## Infeasibility

If `(end_step - start_step) × charger_kw < required_kwh`, the optimization is infeasible. EMHASS returns `optim_status: "infeasible"` and publishes nothing. See [Good Practices — infeasibility triage](good_practices.md).

## See also

- How-to: [MPC walkthrough](mpc.md) for the underlying MPC pattern
- How-to: [Heat-pump walkthrough](heat_pump_walkthrough.md) — combine EV + heat-pump in one system
- Reference: [Passing data](../passing_data.md) for all runtime params
- Reference: [Automations](../automations.md) — `shell_command` / `rest_command` patterns
- Discussion: [#789 — EV charging addon capability](https://github.com/davidusb-geek/emhass/discussions/789)
- Explanation: [Good Practices](good_practices.md) — infeasibility triage
