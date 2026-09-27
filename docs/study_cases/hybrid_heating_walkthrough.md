# Hybrid heating walkthrough

> **Type:** How-To Guide — task-oriented, follow when a heat pump shares the work with a second heat source or more than one store.

This page takes a hybrid heating system from a description to a working
`heat_topology`, published sensors and a rolling MPC loop. For the field
reference see [heat_topology](../heat_topology.md); for a single heat pump on a
single tank, the [heat-pump walkthrough](heat_pump_walkthrough.md) with
`thermal_battery` is simpler.

## Scenario

| Component | Value |
|-----------|-------|
| Heat pump | 3 kW electric, weather-compensated supply 30-55 °C, condenser limit 55 °C |
| Gas boiler | 20 kW, 90 % efficient, gas at 0.09 currency/kWh |
| DHW tank | 200 L, kept between 45 and 60 °C, two shower peaks a day |
| Space-heating buffer | 500 L, 30-55 °C, feeds the underfloor heating |
| House | about 8 kWh/K of thermal mass, heat loss 0.25 kW/K, 15 m² of glazing, comfort band 19.5-22 °C |
| Constraint | the heat pump serves either the DHW tank or the buffer, never both at once |

## 1. Describe the system as a graph

Each physical part becomes one entry:

- **sources** make heat: the heat pump and the boiler;
- **storage** holds heat: the DHW tank, the buffer, and the house as a
  building zone (`thermal_mass` and `loss_coefficient` instead of a water
  `volume`);
- **flows** connect them: a source-to-storage flow is something the optimizer
  schedules (it becomes a deferrable load), a storage-to-storage flow is a heat
  transfer (the buffer feeding the house through the floor);
- **consumers** draw heat: the showers;
- **actuator_groups** express shared equipment: one heat pump, two targets.

```python
HORIZON = 48  # 24 h at 30-minute steps

heat_topology = {
    "sources": [
        {"id": "hp", "type": "heatpump", "nominal_power": 3000,
         "heating_curve": {"slope": 0.6, "offset": 40, "min_supply": 30, "max_supply": 55},
         "carnot_efficiency": 0.45, "max_supply_temperature": 55,
         "treat_as_semi_cont": False},
        {"id": "boiler", "type": "gas", "nominal_power": 20000, "efficiency": 0.9,
         "cost_track": "gas", "treat_as_semi_cont": False},
    ],
    "storage": [
        {"id": "dhw", "volume": 0.2, "start_temperature": 50,
         "min_temperature": [45], "max_temperature": [60], "thermal_loss": 0.05},
        {"id": "buffer", "volume": 0.5, "start_temperature": 40,
         "min_temperature": [30], "max_temperature": [55], "thermal_loss": 0.05},
        {"id": "house", "thermal_mass": 8, "loss_coefficient": 0.25,
         "start_temperature": 20.5, "min_temperature": [19.5], "max_temperature": [22],
         "desired_temperature": 20.5, "window_area": 15},
    ],
    "flows": [
        {"from": "hp", "to": "dhw"},
        {"from": "hp", "to": "buffer"},
        {"from": "boiler", "to": "dhw"},
        {"from": "buffer", "to": "house", "transfer_coefficient": 0.8,
         "max_transfer_power": 8000},
    ],
    "consumers": [
        {"id": "showers", "type": "profile", "target": "dhw",
         "profile": [0.0] * 14 + [1.5, 1.0] + [0.0] * 24 + [1.0, 1.5] + [0.0] * 6},
    ],
    "actuator_groups": [
        {"flows": [["hp", "dhw"], ["hp", "buffer"]], "mutual_exclusion": True},
    ],
    "cost_tracks": {"gas": [0.09] * HORIZON},
}
```

Notes on the choices:

- `max_supply_temperature: 55` on the heat pump: its condenser cannot heat past
  55 °C. The DHW band above that is left to the boiler.
- The boiler has its own `cost_track`, so it is priced at the gas tariff and
  kept out of the electric power balance (`gas` sources are non-electric).
- The house has no source of its own. It only receives heat from the buffer,
  at most `0.8 kW/K × (T_buffer − T_house)` and never more than 8 kW.
- The heat pump is continuous (`treat_as_semi_cont: false`); set it to `true` if
  your unit only runs on/off at its nominal power.

## 2. Configure it

Put the dictionary above under `heat_topology` in the configuration (in the
add-on configuration page, paste it as JSON into the `heat_topology` text box).
When you save, EMHASS compiles the topology; if something is wrong, the alert
names the field, for example `flows[2].from=ghost_source does not match any
source.id`, and nothing is saved.

The compiler turns the three source-to-storage flows into three deferrable loads,
in the order of `flows`:

| Load | Flow | Result columns |
|------|------|----------------|
| 0 | heat pump → DHW | `P_deferrable0`, `predicted_temp_heater0` (DHW temperature) |
| 1 | heat pump → buffer | `P_deferrable1`, `predicted_temp_heater1` (buffer temperature) |
| 2 | boiler → DHW | `P_deferrable2`, `predicted_temp_heater2` (DHW temperature again) |

The house has no load of its own, so its temperature is in
`predicted_temp_heater5`: 3 deferrable loads plus its position (2) in `storage`.
The buffer-to-house transfer is in `P_transfer_buffer_house` (delivered heat, W).

If you also have ordinary deferrable loads (washing machine, EV charger), set
`"extend_deferrable_loads": true` in the topology: your loads keep indices 0, 1,
... and the topology's loads are appended after them.

## 3. Run it as rolling MPC

Send the measured temperatures every run, so each plan starts from reality:

```json
{
  "prediction_horizon": 48,
  "shared_tank_start_temperatures": {"dhw": 48.0, "buffer": 41.5, "house": 20.3}
}
```

`shared_tank_start_temperatures` overrides the configured `start_temperature`
per storage id without resending the whole topology. A topology is rebuilt on
every run (no warm start), so this is always used.

If a storage has `thermal_inertia`, also send the heat that is still on its way
with `shared_tank_prior_heat`; see
[heat_topology](../heat_topology.md#rolling-mpc).

## 4. Publish and check the results

To get sensors for the temperatures, pass one entry per load index with the
publish call:

```json
{
  "custom_predicted_temperature_id": [
    {"entity_id": "sensor.dhw_temperature_predicted", "unit_of_measurement": "°C", "friendly_name": "DHW temperature (predicted)"},
    {"entity_id": "sensor.buffer_temperature_predicted", "unit_of_measurement": "°C", "friendly_name": "Buffer temperature (predicted)"}
  ]
}
```

The house has no load index, so its temperature (`predicted_temp_heater5`) and
the buffer-to-house transfer (`P_transfer_buffer_house`) are not published as
sensors. They are in the result CSV and in `/api/v1/plan`; read them there, for
example from a REST sensor or an automation.

What to expect with this configuration on a cold day: the heat pump does most of
the work, switching between the DHW tank and the buffer and never serving both
in the same step; the boiler only runs when the heat pump cannot deliver in time
or gas is cheaper than the heat pump's electricity; the buffer temperature rises
before the price peaks and the house stays inside 19.5-22 °C.

Check `optim_status` first:

| Status | Meaning |
|--------|---------|
| `Optimal` | a plan that satisfies every constraint |
| `Optimal (Relaxed)` | the full problem could not be solved; the plan ignores the on/off constraints |
| `Infeasible` | no plan satisfies the bounds; see below |

## Troubleshooting

- **`Infeasible`.** Usually a band the sources cannot hold: a minimum
  temperature above what the heat pump can reach without a second source, a DHW
  draw larger than the tank plus the sources can deliver in time, or a
  `mutual_exclusion` group that leaves too few steps for both targets. Widen the
  band or add capacity, one change at a time.
- **The heat pump never heats the DHW tank above 55 °C.** That is
  `max_supply_temperature` at work; the boiler covers that band.
- **COP looks too good on a mild day.** A heating-curve COP can model more heat
  than the unit can deliver; set `max_thermal_power` to the unit's rated thermal
  output.
- **The plan super-heats the buffer at an optimistic COP.** Try
  `cop_solver: auto`, which checks the COP against the temperature the plan
  reaches.
- **Every COP looks like a mild day.** The outdoor temperature is missing and
  EMHASS falls back to 15 °C; see
  [heat_topology](../heat_topology.md#validation-and-troubleshooting).
