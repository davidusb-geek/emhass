# 🔥 Thermal Integration

EMHASS can plan heat together with electricity. A domestic-hot-water (DHW) tank,
a space-heating buffer, a pool or the thermal mass of the house is modelled as a
temperature that the optimizer steers through the horizon, inside a comfort band
you set. It then decides when to make heat: on cheap power or surplus PV, and not
during a price peak, the same way it plans a battery's state of charge.

## Do I need this?

Only if you want EMHASS to schedule something that makes heat or cold: a heat
pump, an electric water heater, a gas boiler, an air conditioner. If you only
have PV, a battery and ordinary deferrable loads, skip this section. Nothing in
it is active by default: `heat_topology` is `null` and no deferrable load has a
thermal configuration, so the optimization is exactly the electrical one.

## Which model to use

| Your system | Use | Start with |
| --- | --- | --- |
| One device heating or cooling one room, no tank | [Deferrable load thermal model](thermal_model.md) (`thermal_config`) | the example on that page |
| One heat pump charging one tank or a floor slab, with a temperature-dependent COP | [Thermal battery](thermal_battery.md) (`thermal_battery`) | [Heat-pump walkthrough](study_cases/heat_pump_walkthrough.md) |
| Several sources or several stores: a heat pump plus a gas boiler or electric booster, a DHW tank plus a buffer, a buffer feeding the house, sources on different tariffs | [Heat topology graph model](heat_topology.md) (`heat_topology`) | [Hybrid heating walkthrough](study_cases/hybrid_heating_walkthrough.md) |

The first two are single-load models configured under `def_load_config`. The
heat topology describes the system as a small graph of sources, stores and
flows, which EMHASS compiles into deferrable loads and shared thermal tanks. It
can express both single-load models as one-source stores.

## Moving from `thermal_battery` to `heat_topology`

Systems tend to grow: a second tank, a booster, a boiler. The fields map
directly:

| `thermal_battery` field | In `heat_topology` |
| --- | --- |
| `supply_temperature`, `heating_curve`, `carnot_efficiency` | a `heatpump` source |
| `efficiency` (constant-efficiency mode) | a `gas`, `oil`, `district`, `electric` or `constant_efficiency` source |
| `volume`, `density`, `heat_capacity`, `thermal_loss` | a storage entry |
| `start_temperature`, `min_temperatures`, `max_temperatures`, `min_temperature_curve`, `desired_temperatures`, `overshoot_temperature`, `penalty_factor` | the same storage entry |
| `sense` (`heat` or `cool`) | the storage's `comfort_sense` |
| `draw_off_demand` | a `profile` consumer on that storage |
| `u_value`, `envelope_area`, `ventilation_rate`, `heated_volume` (or `specific_heating_demand`, `area`), `window_area`, `shgc`, `internal_gains_factor` | a `building_demand` consumer on that storage |
| the nominal power of the deferrable load | the source's `nominal_power` |

Then remove the `thermal_battery` entry from `def_load_config`. The compiler
creates one deferrable load per source-to-storage flow, numbered in the order of
`flows`, so the load index (and `sensor.p_deferrable{k}`) of the heat pump may
change: update `custom_deferrable_forecast_id` and
`custom_predicted_temperature_id` accordingly. If you also have ordinary
deferrable loads, set `extend_deferrable_loads` (see
[Combining with other deferrable loads](heat_topology.md#combining-with-other-deferrable-loads)).

## Temperature-dependent COP

A heat pump's COP falls as it heats the store hotter. The optimizer plans against
a COP at an assumed temperature; with `cop_solver: auto` it checks the plan
against the true temperature-dependent COP afterwards and refines it only when
the two disagree (default `static`: no refinement). See
[heat_topology](heat_topology.md) for when to turn it on and
[the mathematical model](advanced_math_model.md) for how it works.

```{toctree}
:maxdepth: 2
thermal_model
thermal_battery
heat_topology
```
