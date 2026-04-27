# Domestic hot water with a deadline-driven temperature profile

> **Type:** How-To Guide — task-oriented, follow when scheduling a heat-pump-fed DHW tank against dynamic prices.

A DHW tank is a thermal store, but unlike an underfloor slab it has natural usage spikes (morning shower, evening dishes) that anchor when the water has to be hot, not a continuous comfort range. The naive approach is a fixed daily profile (e.g. "always at least 55 °C between 09:00 and 16:00"), which forces the heat pump to run at a fixed time of day regardless of price.

A better pattern is a deadline-driven profile: a low base temperature most of the time, with short windowed spikes just before each usage deadline. The optimizer is then free to pick the cheapest slot in the 24 h leading up to each spike: overnight when prices are low, midday during PV surplus, or whenever a dynamic-tariff dip happens.

## Scenario

| Component | Value |
|-----------|-------|
| Heat pump | one combined HP that does space heating *and* DHW (e.g. Tecalor THZ, integrated air-source HP) |
| DHW tank | a hot-water tank with temperature sensor at the top |
| Usage pattern | morning shower around 07:00, evening dishes/shower around 18:00 |
| Tariff | dynamic (Tibber, aWATTar, Octopus Agile) |
| Optimization mode | naive-mpc-optim |

## Deadline profile

Build the `desired_temperatures` array as three layers stacked over the horizon:

| Layer | Value | Where |
|-------|-------|-------|
| Base (always) | 45 °C | every timestep, the comfort floor; tank may float down to here |
| Morning spike | 48 °C | last 30 min before 07:00 (1 timestep at 30-min step) |
| Evening spike | 50 °C | last 1 h before 18:00 (2 timesteps at 30-min step) |

In addition, set `overshoot_temperature: 59 °C` as a hard upper bound (separate constraint, not part of the per-timestep target array). It prevents Legionella-cycle conflicts and excessive tank stress.

These values are starting points. Adjust to your tank size, usage pattern, and water-comfort preference.

## Configuration

DHW runs through the same deferrable-load slot machinery as any other thermal load. Pick a slot for the DHW (e.g. the second deferrable) and configure its `thermal_config`:

```yaml
nominal_power_of_deferrable_loads:
  - 0           # space heating (handled separately, see heat_pump_walkthrough.md)
  - 2500        # DHW heat pump electric input — adjust to your unit
operating_hours_of_each_deferrable_load:
  - 0
  - 0           # DHW is fully temperature-driven, hour budget unused
```

The `thermal_config` for DHW comes from the runtime payload, not the static config; the deadline profile changes day to day (or hour to hour, if you adapt to forecast).

## Runtime payload

The cleanest way to assemble the `desired_temperatures` array is to compute it outside Home Assistant (Node-RED Function node, AppDaemon, a small Python script) and inject the finished list into the EMHASS payload. The contract EMHASS sees is just a list of length `prediction_horizon` floats.

Reference Python sketch for the deadline profile (30-minute timesteps, 24 h horizon):

```python
from datetime import datetime, timedelta

def build_dhw_profile(now, horizon=48, timestep_min=30,
                      base=45.0,
                      morning_temp=48.0, morning_hour=7, morning_window_slots=1,
                      evening_temp=50.0, evening_hour=18, evening_window_slots=2):
    """Return a list of length `horizon` with the deadline-driven target temps.

    Each spike fires in the last `*_window_slots` timesteps that end at or
    before the deadline hour. After a deadline passes today, the next spike
    rolls forward to tomorrow.
    """
    profile = [base] * horizon
    for hour, temp, slots in (
        (morning_hour, morning_temp, morning_window_slots),
        (evening_hour, evening_temp, evening_window_slots),
    ):
        deadline = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if deadline <= now:
            deadline += timedelta(days=1)
        deadline_step = int((deadline - now).total_seconds() // (timestep_min * 60))
        for k in range(max(0, deadline_step - slots), min(horizon, deadline_step)):
            profile[k] = temp
    return profile
```

Once your runtime layer (Node-RED, AppDaemon, etc.) holds the array, the HA `rest_command` payload simply forwards it:

```yaml
rest_command:
  emhass_dhw_mpc:
    url: http://localhost:5000/action/naive-mpc-optim
    method: POST
    timeout: 120
    headers:
      content-type: application/json
    payload: >
      {
        "prediction_horizon": 48,
        "soc_init": {{ states('sensor.battery_soc') | float / 100 }},
        "def_load_config": [
          {},
          {
            "thermal_config": {
              "heating_rate": 4.0,
              "cooling_constant": 0.02,
              "start_temperature": {{ states('sensor.dhw_tank_temperature') | float }},
              "sense": "heat",
              "overshoot_temperature": 59.0,
              "desired_temperatures": {{ states('sensor.dhw_desired_temperatures') }}
            }
          }
        ]
      }
```

The HA template assumes `sensor.dhw_desired_temperatures` is a JSON-encoded list maintained by your runtime layer. Adjust the literal `48` if your `optimization_time_step` is not 30 minutes; recompute the array length to match.

## How EMHASS uses this

EMHASS treats `desired_temperatures[k]` as the target for timestep `k`. With `cooling_constant` modelling the natural tank cool-down, the optimizer schedules heat-pump runtime anywhere in the horizon that minimizes cost while ensuring the tank reaches each target by its deadline. If the cheapest slot is overnight, it heats overnight and lets the tank coast. If a midday Tibber dip appears, it shifts heating into the dip.

The `overshoot_temperature` is a hard ceiling: the optimizer will not push the tank past it even if free PV is available.

## Real-world example

With a 26 April 2026 Tibber-day where the lowest spot price was −0.42 EUR/kWh between 13:00 and 14:30, the deadline profile produced:

- Old fixed profile (55 °C between 09:00–16:00): heat pump ran at 08:30–09:30 to hit the 09:00 deadline. Cost ≈ 0.20 EUR.
- New deadline profile (base 45, evening spike 50 at 18:00): heat pump ran at 13:00–14:30, exactly during the negative-price slot. Earned ≈ 1.50 EUR. Tank reached 50 °C by 18:00 deadline as required.

Net spread: ~1.70 EUR for the day. The same tank, the same demand, only the temperature profile changed.

## Caveats

- One heat pump can't do heating and DHW simultaneously. If your unit (e.g. Tecalor THZ, Daikin Altherma, Vaillant aroTHERM) shares a single compressor, EMHASS plans both deferrable loads but the unit's own controller picks priority each minute. Pass `def_current_state[<dhw_slot>] = wpDhwOn` (boolean from your real-world DHW-mode sensor) so the optimizer knows whether DHW is currently active in the first timestep.
- `cooling_constant` calibration matters. A too-optimistic value (tank cools slowly in the model but fast in reality) leads to "tank not hot enough at deadline" complaints. Start from a measured value: heat tank to 50 °C, log temperature over 24 h with no demand, fit the exponential decay. A 200 L tank in a typical utility room sits around `cooling_constant = 0.02` per hour. If you cannot calibrate, raise `desired_temperature_base` from 45 to 47–48 to give the optimizer more thermal margin.
- Legionella cycles are NOT handled. EMHASS does not know your tank needs a weekly 60 °C cycle for Legionella prevention. Run that as a separate scheduled override outside EMHASS, or raise the spike temperature to 60 °C on one day per week.
- `thermal_config` vs `thermal_battery`. This page uses the legacy `thermal_config` model with `desired_temperatures` because deadline profiles map naturally to per-timestep targets. The newer `thermal_battery` model uses `min_temperatures` / `max_temperatures` and is better suited to underfloor-slab scenarios; see [heat_pump_walkthrough.md](heat_pump_walkthrough.md).

## See also

- How-to: [Heat-pump walkthrough](heat_pump_walkthrough.md) (slab-heating with the `thermal_battery` model, complementary to this DHW pattern)
- How-to: [MPC walkthrough](mpc.md) (generic rolling-horizon pattern)
- Reference: [thermal_model.md](../thermal_model.md) (full parameter list for `thermal_config`)
- Reference: [Passing data](../passing_data.md) (runtime payload schema)
- Explanation: [Good Practices](good_practices.md) (forecast quality, infeasibility triage)
