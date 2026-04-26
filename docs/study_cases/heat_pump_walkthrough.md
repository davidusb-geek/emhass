# Heat-pump end-to-end walkthrough

> **Type:** How-To Guide — task-oriented, follow when integrating a heat pump as a thermal-battery deferrable in a real home with PV and battery.

This page walks through a complete real-house scenario: PV + electrochemical battery + heat-pump-driven thermal battery + (optionally) a regular EV charger. It is an *orchestration* page — for the parameter reference of the thermal battery model itself, see [Reference: thermal_battery](../thermal_battery.md).

## Scenario

A 130 m² single-family home in central Europe with:

| Component | Value |
|-----------|-------|
| PV | 8 kWp |
| Electrochemical battery | 10 kWh |
| Heat pump | 3 kW electric, supply 35 °C, underfloor heating slab 20 m³ |
| Other deferrable loads | 1 EV charger (treated separately, see [EV walkthrough](ev.md)) |
| Tariff | dynamic (e.g. Tibber, aWATTar, Octopus Agile) |
| Optimization mode | naive-mpc-optim, every 30 min |

## How the pieces connect

EMHASS treats the heat pump as a **thermal_battery** deferrable load. The optimizer simultaneously:

1. Schedules the heat pump to keep the underfloor slab within a min/max temperature comfort range (per `thermal_battery` config — see [thermal_battery.md](../thermal_battery.md) for all parameters).
2. Schedules the electrochemical battery to charge from PV / cheap grid hours and discharge into expensive hours.
3. Schedules other deferrable loads (washing machine, EV) inside their own windows.

All four schedules share the same horizon, the same forecasted PV, the same forecasted prices — the optimizer finds a globally cost-minimal joint plan.

## Configuration

Add `set_use_pv: true`, `set_use_battery: true` plus the standard battery parameters from [Basic — PV + Battery](basic_pv_battery.md). Then add the thermal_battery config under `def_load_config`. A condensed example for a modern home with underfloor heating:

```python
{
  "def_load_config": [
    {},
    {
      "thermal_battery": {
        "supply_temperature": 35.0,
        "volume": 20.0,
        "start_temperature": 22.0,
        "min_temperatures": [20.0] * 48,
        "max_temperatures": [26.0] * 48,
        "carnot_efficiency": 0.45,
        "u_value": 0.35,
        "envelope_area": 280.0,
        "ventilation_rate": 0.4,
        "heated_volume": 320.0,
        "thermal_inertia_time_constant": 2.0
      }
    }
  ]
}
```

For the meaning of each parameter (and the alternative HDD-based heating-demand method), see [thermal_battery.md](../thermal_battery.md). For solar gains and internal-gains parameters, see the same page.

## Run (MPC, every 30 min)

The MPC payload combines runtime SOC, runtime room temperature, and the `def_load_config`:

```yaml
rest_command:
  emhass_mpc:
    url: http://localhost:5000/action/naive-mpc-optim
    method: POST
    timeout: 300
    headers:
      content-type: application/json
    payload: >
      {%- set horizon = 48 -%}
      {
        "prediction_horizon": {{ horizon }},
        "soc_init": {{ states('sensor.battery_soc') | float / 100 }},
        "def_total_hours": [
          {{ states('sensor.washing_machine_remaining_hours') }}
        ],
        "def_load_config": [
          {},
          {
            "thermal_battery": {
              "supply_temperature": 35.0,
              "volume": 20.0,
              "start_temperature": {{ states('sensor.floor_temperature') | float }},
              "min_temperatures": {{ ([20.0] * horizon) | tojson }},
              "max_temperatures": {{ ([26.0] * horizon) | tojson }},
              "carnot_efficiency": 0.45,
              "u_value": 0.35,
              "envelope_area": 280.0,
              "ventilation_rate": 0.4,
              "heated_volume": 320.0,
              "thermal_inertia_time_constant": 2.0
            }
          }
        ],
        "outdoor_temperature_forecast": {{ ((state_attr("weather.home", "forecast") | map(attribute="temperature") | list)[:horizon] | tojson) }}
      }
```

Adjust `horizon` if you use a non-default `optimization_time_step`.

For the publish-data follow-up call (which converts the predicted thermal-battery temperature into a `sensor.temp_predicted0`-equivalent that drives your real heat pump's setpoint), see [thermal_battery.md — Published sensors](../thermal_battery.md#published-sensors).

## Interpretation

- The optimizer pre-heats the slab during low-price or PV-surplus hours, then lets it coast through expensive hours by drawing from thermal mass instead of running the heat pump.
- The electrochemical battery handles short-time-scale shifting (within hours), the thermal battery handles longer-scale shifting (across cheap-night → expensive-evening). They are complementary, not redundant.
- The `thermal_inertia_time_constant` of 2 h tells the optimizer that heat applied now reaches the slab gradually — without it, MPC tends to schedule pre-heating *too late* for short prediction horizons.

## See also

- Reference: [thermal_battery.md](../thermal_battery.md) — every parameter, calibration steps, troubleshooting
- Reference: [thermal_model.md](../thermal_model.md) — simpler thermal model without thermal-mass storage
- How-to: [MPC walkthrough](mpc.md) — generic rolling-horizon pattern
- How-to: [EV walkthrough](ev.md) — adding an EV deferrable to this scenario
- Explanation: [Good Practices](good_practices.md) — pre-heating windows, infeasibility triage
