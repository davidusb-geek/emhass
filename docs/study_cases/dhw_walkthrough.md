# Domestic hot water (DHW) — deadline-driven temperature profile

> **Type:** How-To Guide — task-oriented, follow when scheduling a heat-pump-fed DHW tank against dynamic prices.

A DHW tank is a thermal store, but unlike an underfloor slab it has natural usage spikes (morning shower, evening dishes) that anchor when the water has to be hot — not a continuous comfort range. The naive approach is a fixed daily profile (e.g. "always at least 55 °C between 09:00 and 16:00"), which forces the heat pump to run at a fixed time of day regardless of price.

A better pattern is a **deadline-driven profile**: a low base temperature most of the time, with short windowed spikes just before each usage deadline. The optimizer is then free to pick the cheapest slot in the 24 h leading up to each spike — overnight when prices are low, midday during PV surplus, or whenever a dynamic-tariff dip happens.

## Scenario

| Component | Value |
|-----------|-------|
| Heat pump | one combined HP that does space heating *and* DHW (e.g. Tecalor THZ, integrated air-source HP) |
| DHW tank | a hot-water tank with temperature sensor at the top |
| Usage pattern | morning shower around 07:00, evening dishes/shower around 18:00 |
| Tariff | dynamic (Tibber, aWATTar, Octopus Agile) |
| Optimization mode | naive-mpc-optim |

## Deadline profile

Express the DHW comfort target as three layers stacked in the runtime payload:

| Layer | Value | Where |
|-------|-------|-------|
| Base (always) | 45 °C | every timestep — comfort floor; tank may float down to here |
| Morning spike | 48 °C | last 30 min before 07:00 (1 timestep at 30-min step) |
| Evening spike | 50 °C | last 1 h before 18:00 (2 timesteps at 30-min step) |
| Anti-cap | 59 °C `overshoot_temperature` | upper bound to prevent Legionella-cycle conflict and tank stress |

These values are starting points — adjust to your tank size, usage pattern, and water-comfort preference.

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

The `thermal_config` for DHW comes from the runtime payload, not the static config — the deadline profile changes day to day (or hour to hour, if you adapt to forecast).

## Runtime payload

Build the deadline-profile arrays in your Home Assistant `rest_command` template (or Node-RED Function node). The example below assumes 30-minute timesteps and a 24 h horizon:

```yaml
rest_command:
  emhass_dhw_mpc:
    url: http://localhost:5000/action/naive-mpc-optim
    method: POST
    timeout: 120
    headers:
      content-type: application/json
    payload: >
      {%- set horizon = 48 -%}
      {%- set timestep_min = 30 -%}
      {%- set base = 45.0 -%}
      {%- set morning_temp = 48.0 -%}
      {%- set morning_hour = 7 -%}
      {%- set morning_window_slots = 1 -%}
      {%- set evening_temp = 50.0 -%}
      {%- set evening_hour = 18 -%}
      {%- set evening_window_slots = 2 -%}
      {%- set ns = namespace(profile=[]) -%}
      {%- for step in range(horizon) -%}
        {%- set hour = (now().hour + (step * timestep_min) // 60) % 24 -%}
        {%- set in_morning = (hour == morning_hour) and (step >= horizon - morning_window_slots) -%}
        {%- set in_evening = (hour == evening_hour) -%}
        {%- if in_evening -%}
          {%- set ns.profile = ns.profile + [evening_temp] -%}
        {%- elif in_morning -%}
          {%- set ns.profile = ns.profile + [morning_temp] -%}
        {%- else -%}
          {%- set ns.profile = ns.profile + [base] -%}
        {%- endif -%}
      {%- endfor -%}
      {
        "prediction_horizon": {{ horizon }},
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
              "desired_temperatures": {{ ns.profile | tojson }}
            }
          }
        ]
      }
```

The Jinja loop is illustrative — production systems usually build the array outside HA (Node-RED Function node, AppDaemon, etc.) where set logic is easier. The contract EMHASS sees is just a list of length `prediction_horizon` floats in `desired_temperatures`.

## How EMHASS uses this

EMHASS treats `desired_temperatures[k]` as the target for timestep `k`. With `cooling_constant` modelling the natural tank cool-down, the optimizer schedules heat-pump runtime **anywhere in the horizon** that minimizes cost while ensuring the tank reaches each target by its deadline. If the cheapest slot is overnight, it heats overnight and lets the tank coast. If a midday Tibber dip appears, it shifts heating into the dip.

The `overshoot_temperature` is a hard ceiling — the optimizer will not push the tank past it even if free PV is available.

## Real-world example

With a 26 April 2026 Tibber-day where the lowest spot price was −0.42 EUR/kWh between 13:00 and 14:30, the deadline profile produced:

- **Old fixed profile** (55 °C between 09:00–16:00): heat pump ran at 08:30–09:30 to hit the 09:00 deadline. Cost ≈ 0.20 EUR.
- **New deadline profile** (base 45, evening spike 50 at 18:00): heat pump ran at 13:00–14:30 — exactly during the negative-price slot. Earned ≈ 1.50 EUR. Tank reached 50 °C by 18:00 deadline as required.

Net spread: ~1.70 EUR for the day. The same tank, the same demand — only the temperature profile changed.

## Caveats

- **One heat pump can't do heating and DHW simultaneously.** If your unit (e.g. Tecalor THZ, Daikin Altherma, Vaillant aroTHERM) shares a single compressor, EMHASS plans both deferrable loads but the unit's own controller picks priority each minute. Pass `def_current_state[<dhw_slot>] = wpDhwOn` (boolean from your real-world DHW-mode sensor) so the optimizer knows whether DHW is currently active in the first timestep.
- **`cooling_constant` calibration matters.** A too-optimistic value (tank cools slowly in the model but fast in reality) leads to "tank not hot enough at deadline" complaints. Start from a measured value: heat tank to 50 °C, log temperature over 24 h with no demand, fit the exponential decay. A 200 L tank in a typical utility room sits around `cooling_constant = 0.02` per hour. If you cannot calibrate, raise `desired_temperature_base` from 45 to 47–48 to give the optimizer more thermal margin.
- **Legionella cycles are NOT handled.** EMHASS does not know your tank needs a weekly 60 °C cycle for Legionella prevention. Run that as a separate scheduled override outside EMHASS, or raise the spike temperature to 60 °C on one day per week.
- **`thermal_config` vs `thermal_battery`.** This page uses the legacy `thermal_config` model with `desired_temperatures` because deadline profiles map naturally to per-timestep targets. The newer `thermal_battery` model uses `min_temperatures` / `max_temperatures` and is better suited to underfloor-slab scenarios — see [heat_pump_walkthrough.md](heat_pump_walkthrough.md).

## See also

- How-to: [Heat-pump walkthrough](heat_pump_walkthrough.md) — slab-heating with the `thermal_battery` model, complementary to this DHW pattern
- How-to: [MPC walkthrough](mpc.md) — generic rolling-horizon pattern
- Reference: [thermal_model.md](../thermal_model.md) — full parameter list for `thermal_config`
- Reference: [Passing data](../passing_data.md) — runtime payload schema
- Explanation: [Good Practices](good_practices.md) — forecast-quality, infeasibility triage
