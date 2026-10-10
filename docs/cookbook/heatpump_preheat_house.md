# Pre-heat the house with a heat pump on cheap power

## Goal

Let EMHASS use the house itself as a heat store: heat a little above the target while electricity is cheap or PV is available, then let the heat pump idle through the price peak while the room temperature stays inside its comfort band.

## Prerequisites

- EMHASS ≥ 0.18.5 (building-zone storage in `heat_topology`: `thermal_mass`, `loss_coefficient`).
- An outdoor temperature forecast: automatic with `weather_forecast_method: open-meteo`, otherwise pass `outdoor_temperature_forecast` at runtime (see [Passing data](../passing_data.md)).
- A room temperature sensor, and a way to make the heat pump follow a planned power or on/off signal.
- Half-hour optimization time step and a 48-step horizon in the examples.
- Transport: the examples are direct EMHASS configuration and runtime JSON. Adapter-specific transport (Node-RED, Home Assistant `rest_command`, AppDaemon) is untested here: contribution welcome.

## Step 1: Estimate the two numbers that describe the house

<!-- source: src/emhass/optimization.py:4522 -->

The zone model needs a heat-loss coefficient and a heat capacity. The loss per step is `loss_coefficient * (indoor - outdoor)`, so a warmer house loses more: that is the price the optimizer weighs against cheaper power.

- `loss_coefficient` (kW/K): the heat the house needs on a cold design day, divided by the indoor-outdoor difference on that day. A house that needs 5 kW at 20 degrees inside and 0 degrees outside has `5 / 20 = 0.25`.
- `thermal_mass` (kWh/K): `loss_coefficient` times the cool-down time constant in hours. Switch the heating off on a cold evening without sun and note how long the indoor-outdoor difference takes to fall to 37 % of its starting value. A time constant of 72 hours gives `0.25 * 72 = 18`.

Expected: two numbers. Start with estimates, then compare `predicted_temp_heater0` with the measured room temperature over a few days and adjust: a house that cools faster than planned needs a smaller `thermal_mass` or a larger `loss_coefficient`.

## Step 2: Describe the heat pump and the house as a heat topology

<!-- source: src/emhass/data/config_defaults.json:55 -->
<!-- source: src/emhass/utils.py:812 -->
<!-- source: src/emhass/utils.py:1163 -->
<!-- transport: direct EMHASS configuration; adapter-specific transport untested -->

One source feeds one storage, and the storage is the house. The `heating_curve` gives the supply temperature per step as `offset - slope * outdoor`, limited to `min_supply`..`max_supply`; the COP follows from it and from the outdoor forecast. `min_power` is the lowest power the compressor can modulate to: the heat pump is either off or between `min_power` and `nominal_power`.

Set this as `heat_topology` in the configuration:

```json
{
  "sources": [
    {"id": "hp", "type": "heatpump", "nominal_power": 3000, "min_power": 900,
     "treat_as_semi_cont": false, "carnot_efficiency": 0.45,
     "heating_curve": {"slope": 0.6, "offset": 38, "min_supply": 28, "max_supply": 45}}
  ],
  "storage": [
    {"id": "house", "thermal_mass": 18, "loss_coefficient": 0.25,
     "start_temperature": 20.5,
     "min_temperatures": [19.5],
     "max_temperatures": [21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5,
                          21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5,
                          21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5,
                          21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5,
                          21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5,
                          21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5],
     "desired_temperatures": 20.5}
  ],
  "flows": [{"from": "hp", "to": "house"}]
}
```

`min_temperatures` and `max_temperatures` are the hard comfort band; the gap between them is the room the optimizer has to shift heat in time. `desired_temperatures` is a soft target: the plan returns to it when deviating does not pay. A short `min_temperatures` list is extended with its last value; `max_temperatures` needs one value per step. `nominal_power` and `min_power` are electrical watts. The types `heatpump` and `heat_pump` are the same.

Expected: saving succeeds, and the log of the next run shows `heat_topology compiled: 1 sources, 1 storage, 1 flows, 0 groups`.

## Step 3: Feed the measured room temperature on every MPC run

<!-- source: src/emhass/utils.py:3295 -->
<!-- transport: direct EMHASS naive-mpc-optim runtime JSON; adapter-specific transport untested -->

Send the room sensor's value on every call, keyed by storage id:

```json
{
  "prediction_horizon": 48,
  "shared_tank_start_temperatures": {"house": 20.2}
}
```

If the house is below `min_temperatures` (windows open, a cold morning after a setback), the run still solves: the minimums of that run are priced instead of hard, and the plan recovers as fast as the heat pump allows.

Expected: `predicted_temp_heater0` starts at the value you sent.

## Step 4: Read the plan

<!-- source: src/emhass/optimization.py:5793 -->

With the example above, a 2 degree outdoor temperature, no PV, and a tariff of 0.25 per kWh that drops to 0.10 in steps 8 to 15 and rises to 0.45 in steps 34 to 41, the plan is:

```text
steps  0- 1   heat pump lifts the house from 20.2 to the 20.5 target
steps  2-10   about 1150 W holds 20.5
steps 11-15   3000 W on the cheap tariff: the house reaches 21.5, the top of the band
steps 16-22   heat pump off, the house coasts back to 20.6
steps 23-28   900 to 1150 W holds 20.5
steps 29-33   3000 W again before the price peak, up to 21.45
steps 34-40   heat pump off through the peak, the house coasts back to 20.5
```

`sensor.p_deferrable0` carries the planned electrical power and `sensor.temp_predicted0` the planned room temperature after `publish-data`. Give the heat pump the planned power if it accepts one; otherwise use the planned temperature as its room setpoint, which makes it heat when the plan heats.

Expected: the heat pump runs hardest in the cheapest steps before an expensive period and is off during the expensive period, while the planned temperature stays between 19.5 and 21.5 degrees.

## Caveats

- A wider comfort band shifts more energy and costs more losses. With equal minimum and maximum there is nothing to shift.
- Underfloor heating responds slowly: add `thermal_inertia` (hours) on the storage so the plan knows the heat arrives later.
- Sun through the windows: add `window_area` (m2) and optionally `shgc` on the storage; this needs a GHI forecast (open-meteo).
- The COP comes from the heating curve, not from the temperature the plan actually reaches; on a house that stays within a degree or two this difference is small.
- `startup_penalty` discourages short cycling but can lengthen the solve considerably; check your solve times before relying on it.
- The temperature after the last step of the horizon is not constrained, so the last steps of a plan are not representative. Act on the first step and re-plan (rolling MPC).
- A gas boiler or a buffer between the heat pump and the house can be added to the same topology; see [Heat topology](../heat_topology.md).

## Credits

- Building-zone storage in `heat_topology`: issue **#539** and its discussion.
- Reference: [Heat topology](../heat_topology.md) and the standalone [thermal model](../thermal_model.md); long-form background in the [heat-pump walkthrough](../study_cases/heat_pump_walkthrough.md).
- Field names verified against `src/emhass/utils.py:compile_heat_topology` and `treat_runtimeparams` on 2026-10-10; the example was solved end to end on EMHASS 0.18.5.
