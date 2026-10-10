# DHW tank with a heat pump and an electric booster

## Goal

Plan a domestic-hot-water tank that a heat pump heats up to its condenser limit and an electric element heats above it, including a scheduled legionella cycle, with the live tank temperature fed in on every MPC run.

## Prerequisites

- EMHASS ≥ 0.18.5 (per-source `max_supply_temperature`, `shared_tank_start_temperatures`, and a tank that starts below its minimum no longer makes the run infeasible).
- An outdoor temperature forecast: automatic with `weather_forecast_method: open-meteo`, otherwise pass `outdoor_temperature_forecast` at runtime (see [Passing data](../passing_data.md)).
- Half-hour optimization time step and a 48-step horizon in the examples. For another horizon, give every per-step list that many values.
- Transport: the examples are direct EMHASS configuration and runtime JSON. Adapter-specific transport (Node-RED, Home Assistant `rest_command`, AppDaemon) is untested here: contribution welcome.

## Step 1: Describe the tank as a heat topology

<!-- source: src/emhass/data/config_defaults.json:55 -->
<!-- source: src/emhass/utils.py:626 -->
<!-- source: src/emhass/utils.py:862 -->
<!-- transport: direct EMHASS configuration; adapter-specific transport untested -->

Two sources feed one storage. The heat pump is cheap per kWh of heat but cannot heat water above its condenser limit; the element can, at a COP of 1. `max_supply_temperature` makes the heat pump's limit hard, so the element is planned only for the band above it. Both sources are continuous (`treat_as_semi_cont: false`): a capped on/off source that would overshoot its ceiling in one full-power step never runs at all.

Set this as `heat_topology` in the configuration (in the add-on page, paste it into the `heat_topology` text box):

```json
{
  "sources": [
    {"id": "hp", "type": "heatpump", "nominal_power": 2500,
     "supply_temperature": 55, "carnot_efficiency": 0.4,
     "max_supply_temperature": 53, "treat_as_semi_cont": false},
    {"id": "booster", "type": "electric", "nominal_power": 3000,
     "efficiency": 1.0, "treat_as_semi_cont": false}
  ],
  "storage": [
    {"id": "dhw", "volume": 0.2, "start_temperature": 48.0, "thermal_loss": 0.035,
     "min_temperature": [45.0],
     "max_temperature": [65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                         65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                         65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                         65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65]}
  ],
  "consumers": [
    {"id": "showers", "type": "profile", "target": "dhw",
     "profile": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.6,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1.2, 0.6, 0, 0, 0, 0, 0, 0, 0, 0]}
  ],
  "flows": [{"from": "hp", "to": "dhw"}, {"from": "booster", "to": "dhw"}]
}
```

`volume` is in m3 and `profile` is the hot water drawn in kWh per timestep, repeated daily. A short `min_temperature` list is extended with its last value; a `max_temperature` list is not, so give it one value per step. The topology replaces the deferrable-load set: if you also have ordinary deferrable loads, add `"extend_deferrable_loads": true` (see [Heat topology](../heat_topology.md)).

Expected: saving succeeds, and the log of the next run shows `heat_topology compiled: 2 sources, 1 storage, 2 flows, 0 groups`. Load 0 is the heat pump and load 1 the element, in the order of `flows`.

## Step 2: Feed the measured tank temperature on every MPC run

<!-- source: src/emhass/utils.py:3295 -->
<!-- transport: direct EMHASS naive-mpc-optim runtime JSON; adapter-specific transport untested -->

The `start_temperature` in the configuration is only a placeholder. On every call, send the temperature your tank sensor reports, keyed by storage id:

```json
{
  "prediction_horizon": 48,
  "shared_tank_start_temperatures": {"dhw": 47.3}
}
```

A value that is not a finite number is ignored with a warning and the configured start temperature is used, so check the log if a plan looks off. If the tank reports a temperature below its minimum (after a long shower), the run still solves: the minimums of that run are priced instead of hard, and the tank recovers as fast as the sources allow.

Expected: `predicted_temp_heater0` in the result starts at the value you sent.

## Step 3: Schedule a legionella cycle

<!-- source: src/emhass/optimization.py:111 -->
<!-- transport: direct EMHASS naive-mpc-optim runtime JSON; adapter-specific transport untested -->

Raise the minimum to 60 degrees Celsius for the steps where the tank must be disinfected. The window moves with the clock, so send the topology of Step 1 at runtime with this `min_temperature` list instead of the static one (a runtime `heat_topology` replaces the configured one):

```json
"min_temperature": [45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                    45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                    45, 45, 45, 45, 45, 45, 60, 60, 60, 60, 45, 45,
                    45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45]
```

A minimum that rises later in the horizon is a hard constraint, and the plan heats ahead for it. By the time the window comes within 6 steps of the start of the horizon the tank is normally already hot. If it is not (the cycle was added late, or a large draw came in between), EMHASS treats it as a tank that starts too cold: the minimums of that run are priced rather than hard, and the tank reaches 60 degrees as soon as the sources allow.

Expected: with the example above, a tariff that is lower in steps 20 to 29, an 8 degree outdoor temperature and no PV, the heat pump (`P_deferrable0`) lifts the tank to 53.0 degrees just before the window, and the element (`P_deferrable1`) then runs for two steps to reach 60.2 degrees at step 30. The heat pump is at 0 W whenever the tank is above 53 degrees.

## Step 4: Publish and act on the plan

<!-- source: src/emhass/optimization.py:5793 -->
<!-- source: src/emhass/utils.py:2023 -->

`publish-data` posts `sensor.p_deferrable0` (heat pump) and `sensor.p_deferrable1` (element), and the planned tank temperature as `sensor.temp_predicted0`. Both loads report the same tank, so `sensor.temp_predicted1` is a duplicate. Rename them with one entry per load:

```json
{
  "custom_deferrable_forecast_id": [
    {"entity_id": "sensor.dhw_heat_pump_plan", "unit_of_measurement": "W", "friendly_name": "DHW heat pump plan"},
    {"entity_id": "sensor.dhw_booster_plan", "unit_of_measurement": "W", "friendly_name": "DHW booster plan"}
  ]
}
```

Drive the heat pump's DHW mode from load 0 and the element's relay from load 1. Both are continuous here, so treat a planned power above a small threshold (for example 100 W) as "on" if your device only switches.

Expected: the two plan sensors and the predicted temperature update on every MPC run.

## Caveats

- `supply_temperature` only sets the heat pump's COP. Without `max_supply_temperature` the plan may heat the tank above it with the heat pump alone.
- The heat pump's COP is computed from `supply_temperature` and the outdoor forecast. An exhaust-air or ground-source unit needs its own source temperature in `outdoor_temperature_forecast`.
- A minimum above every source's reach makes the run infeasible: with only the capped heat pump, a 60 degree minimum cannot be met.
- The temperature after the last step of the horizon is not constrained, so the last steps of a plan are not representative. Act on the first step and re-plan (rolling MPC).
- Results for a tank are in the result table and `/api/v1/plan` as `predicted_temp_heater{k}`, `min_temp_heater{k}` and `max_temp_heater{k}`.

## Credits

- Shared tanks and `heat_topology` for hybrid heating: issue **#539** and its discussion.
- Reference: [Heat topology](../heat_topology.md); long-form background in the [DHW walkthrough](../study_cases/dhw_walkthrough.md).
- Field names verified against `src/emhass/utils.py:compile_heat_topology` and `treat_runtimeparams` on 2026-10-10; the example was solved end to end on EMHASS 0.18.5.
