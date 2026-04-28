# Reference Configurations

> **Type:** Reference — lookup of config blueprints for common system archetypes. Pick the closest match and adapt.

These are anonymized, abstract templates. They are not real users' systems. Each block is a starting point — the parameter values reflect typical hardware in the archetype, but you must verify against your own equipment. See [Configuration](../config.md) for the full parameter reference.

```{note}
**Integer hours only.** `operating_hours_of_each_deferrable_load` accepts whole integers (see [issue #373](https://github.com/davidusb-geek/emhass/issues/373)). The blueprints below round all hour values up to the nearest integer. If your loads need finer control, reduce `optimization_time_step` (e.g. to 15 minutes) so the same wall-clock duration spans more timesteps.
```

## 1. Single-family home — PV + battery + heat pump + EV + dynamic tariff

The most common modern setup: ~150 m² home, 8–12 kWp PV, 10–15 kWh battery, air-source heat pump with underfloor heating, one EV, Tibber/aWATTar/Octopus dynamic pricing.

```yaml
# Static config (Add-on options or config_emhass.yaml)
set_use_pv: true
solar_forecast_kwp: 10
optimization_time_step: 30
# Note: prediction_horizon is a runtime parameter (passed in the MPC payload),
# not a static config field. With this 30-min step a 24 h horizon is 48 timesteps.

set_use_battery: true
battery_nominal_energy_capacity: 12000     # Wh (= 12 kWh)
battery_charge_power_max: 5000             # W
battery_discharge_power_max: 5000          # W
battery_minimum_state_of_charge: 0.2
battery_maximum_state_of_charge: 0.9
battery_target_state_of_charge: 0.5

nominal_power_of_deferrable_loads:
  - 3000        # water heater (regular deferrable)
  - 11000       # EV charger
  - 0           # heat pump (controlled via thermal_battery, set 0 here)

operating_hours_of_each_deferrable_load:
  - 4
  - 0           # EV: overridden at runtime
  - 0           # heat pump: thermal_battery handles it

# def_load_config — passed at runtime (template in heat_pump_walkthrough.md)
```

Cross-link: [Heat-pump walkthrough](heat_pump_walkthrough.md) for the full MPC payload assembly. Daily cost depends strongly on local tariff spread, PV size, weather, and house thermal characteristics — measure your own baseline before claiming savings.

## 2. PV-only — no battery, heat pump, static tariff

A simpler setup: 6–10 kWp PV, no battery, an air-source heat pump for water heating, fixed-rate import (no dynamic prices). EMHASS still helps schedule the heat pump and other deferrable loads against PV surplus.

```yaml
set_use_pv: true
solar_forecast_kwp: 8
optimization_time_step: 30

set_use_battery: false

nominal_power_of_deferrable_loads:
  - 3000        # water heater (heat-pump-driven, treat as regular deferrable)
  - 1500        # dishwasher

operating_hours_of_each_deferrable_load:
  - 4
  - 2           # rounded up from typical 1.5 h cycle

# Static prices (no dynamic tariff)
load_cost_forecast_method: list
prod_price_forecast_method: list
# Pass [0.30] * 48 and [0.08] * 48 at runtime, or set in config if truly static.
```

Most savings come from shifting deferrable loads into PV-surplus hours. Cost reduction is bounded by the share of the daily load that is actually shiftable.

## 3. Off-grid heavy — large PV + battery, no grid import

Cabin / off-grid setup: 15+ kWp PV, 30+ kWh battery, no grid connection (or grid connection with hard limits). Cost minimization is irrelevant — the goal is *don't run out of charge*.

```yaml
set_use_pv: true
solar_forecast_kwp: 20

set_use_battery: true
battery_nominal_energy_capacity: 30000     # Wh (= 30 kWh)
battery_charge_power_max: 10000            # W
battery_discharge_power_max: 10000         # W
battery_minimum_state_of_charge: 0.2       # lower floor — off-grid prefers more usable range
battery_maximum_state_of_charge: 0.95
battery_target_state_of_charge: 0.7        # always end day with reserve

# No grid: cost function reflects PV-curtailment penalty + load-shed penalty
costfun: self-consumption
```

For true off-grid use, modeling load shedding (turning loads off when SOC drops) is currently outside EMHASS's optimization model and must be handled in HA automations. The `costfun: self-consumption` mode minimises grid interaction without explicit load-shedding constraints.

## 4. Apartment with dynamic tariff — no PV, no battery

Renter setup: no rooftop access, no battery space. Just deferrable-load shifting against Tibber/aWATTar prices.

```yaml
set_use_pv: false
set_use_battery: false

nominal_power_of_deferrable_loads:
  - 2200        # washing machine
  - 2000        # dishwasher
  - 1800        # heat-pump tumble dryer

operating_hours_of_each_deferrable_load:
  - 2           # washing machine (rounded up from typical 1.5 h cycle)
  - 2           # dishwasher (rounded up from typical 1.5 h cycle)
  - 1           # tumble dryer
```

Savings depend on the daily price spread of your dynamic tariff and the share of total energy that runs through shiftable loads. For typical 0.10–0.25 EUR/kWh spreads, deferrable-only setups produce visible but modest savings — the marginal value scales linearly with the spread.

## 5. Heat-pump-focused — PV + heat pump, no battery, no EV

Mid-renovation home: PV installed but no battery yet, heat pump replaced gas, no EV. Most flexibility lives in the thermal battery (slab) rather than an electrochemical battery.

```yaml
set_use_pv: true
solar_forecast_kwp: 10
optimization_time_step: 30

set_use_battery: false

nominal_power_of_deferrable_loads:
  - 0           # heat pump via thermal_battery

# def_load_config with thermal_battery — see heat_pump_walkthrough.md
```

The thermal slab serves as the storage. Expect strong dependence on `thermal_inertia_time_constant` and accurate `outdoor_temperature_forecast`.

Cross-link: [Heat-pump walkthrough](heat_pump_walkthrough.md), [Reference: thermal_battery](../thermal_battery.md).

## 6. Multi-EV — PV + battery + 2 EVs

Two-EV household: each EV has independent departure time and required-energy. Both modeled as separate deferrable slots.

```yaml
set_use_pv: true
solar_forecast_kwp: 12

set_use_battery: true
battery_nominal_energy_capacity: 10000     # Wh (= 10 kWh)

nominal_power_of_deferrable_loads:
  - 3000        # water heater
  - 11000       # EV 1 (3-phase 16 A)
  - 7400        # EV 2 (1-phase 32 A)

operating_hours_of_each_deferrable_load:
  - 4
  - 0           # runtime override
  - 0           # runtime override

end_timesteps_of_each_deferrable_load:
  - 48
  - 0           # runtime override (per EV departure)
  - 0           # runtime override (per EV departure)
```

Pass per-EV `def_total_hours[1]`, `def_total_hours[2]`, `end_timesteps[1]`, `end_timesteps[2]` from each EV's source at runtime. See [EV walkthrough](ev.md) for the template pattern; replicate per EV.

## See also

- Reference: [Configuration](../config.md) — complete parameter reference
- Reference: [Passing data](../passing_data.md) — runtime payload schema
- Tutorials: [Basic — PV](basic_pv.md), [Basic — PV + Battery](basic_pv_battery.md)
- How-tos: [MPC](mpc.md), [Heat-pump walkthrough](heat_pump_walkthrough.md), [EV](ev.md)
- Explanation: [Good Practices](good_practices.md)
