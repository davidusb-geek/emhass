# Heat topology graph model

`heat_topology` describes hybrid heating systems as a directed graph. EMHASS
compiles the graph into the existing deferrable-load and thermal-storage
configuration fields before building the optimization problem.

Use it when several heat sources feed the same storage, for example:

- a heat pump and gas boiler feeding one domestic-hot-water (DHW) tank;
- one boiler feeding a DHW tank and a space-heating buffer;
- heat sources with different energy tariffs; or
- one physical actuator that must not serve two thermal targets at the same
  time.

The graph model was introduced in EMHASS 0.17.4. Use `null` (the JSON value),
not the string `"null"`, to disable it.

## Where to configure it

Set `heat_topology` in `optim_conf` for a static configuration, or pass it as a
top-level runtime parameter to an optimization endpoint. Runtime parameters
take precedence over the static configuration.

In the add-on configuration page, the field is a multi-line text box. Paste
valid JSON, not a quoted JSON string. When you save, EMHASS compiles the
topology first: an invalid topology is not saved, and the alert names the
offending field, for example:
`heat_topology is invalid: heat_topology.flows[2].from='ghost_source' does not match any source.id or storage.id`.
For example, the Python configuration below
can be converted to JSON with:

```python
import json

print(json.dumps(heat_topology))
```

The compiler replaces the structural deferrable-load fields generated from the
graph, including `number_of_deferrable_loads`, `def_load_config`,
`shared_thermal_tanks`, `deferrable_load_groups`, and their per-load arrays.
Do not maintain a second, conflicting set of those fields by hand.

## Graph structure

A topology may contain these top-level keys:

| Key | Purpose |
| --- | --- |
| `sources` | Heat pumps, boilers, electric heaters, or other heat sources. |
| `storage` | Thermal stores or effective thermal masses. |
| `consumers` | DHW draw profiles, building demand, or pool comfort demand. |
| `flows` | Directed connections. A source-to-storage flow becomes one deferrable load; a storage-to-storage flow is a heat transfer (see [Tank-to-tank transfers](#tank-to-tank-transfers)). |
| `actuator_groups` | Optional constraints across flows that share physical equipment. |
| `cost_tracks` | Optional per-timestep energy prices referenced by sources. |

IDs must be unique within `sources` and `storage`, and a source and a storage
cannot share an ID. Every `flow.from` must match a source or storage ID, every
`flow.to` must match a storage ID, and every `consumer.target` must match a
storage ID.

### Sources

Every source requires `id`, `type`, and `nominal_power`. Power values are in
watts of source input:

| Field | Description |
| --- | --- |
| `id` | Unique source ID. |
| `type` | `heatpump`, `heat_pump`, `gas`, `oil`, `district`, `electric`, or `constant_efficiency`. |
| `nominal_power` | Maximum source input power in W. |
| `min_power` | Optional minimum input power in W; default `0`. Must not exceed `nominal_power`. |
| `treat_as_semi_cont` | Optional on/off-at-nominal behavior; default `true`. |
| `supply_temperature` | Fixed heat-pump supply temperature in degrees Celsius. |
| `heating_curve` | Alternative heat-pump supply-temperature curve. |
| `cooling_curve` | For a `cool` storage: the same shape as `heating_curve` (defaults `min_supply` 5, `max_supply` 18), giving a weather-compensated chilled supply temperature. The cooling Carnot lift (outdoor minus supply) is applied automatically. |
| `carnot_efficiency` | Heat-pump Carnot efficiency; default `0.4`. |
| `efficiency` | Required constant conversion efficiency for gas, oil, district, electric, and constant-efficiency sources. |
| `cost_track` | Optional key in `cost_tracks`. Without it, the shared electricity tariff is used. |
| `electric` | Optional override for electric-balance membership. |
| `max_supply_temperature` | Optional hard ceiling (degrees Celsius) on the storage temperature this source can heat into. A number, or a per-timestep list (a short list is extended with its last value). |
| `overshoot_temperature` | Optional threshold (degrees Celsius): with the storage's desired temperature set, this source stops beyond it (see below). Overrides the storage-level `overshoot_temperature`. |
| `startup_penalty` | Optional penalty per off-to-on switch, to discourage short cycling; default `0`. Each start costs `startup_penalty × nominal_power (kW) × electricity price × step length (h)`, priced at the electricity tariff even for a fuel source on its own `cost_track`. |
| `max_startups` | Optional hard limit on the number of starts over the horizon; default `0` (no limit). |

`startup_penalty` and `max_startups` act on the source's on/off state, which is
tied to its power only when the source is semi-continuous or has a `min_power`.
A continuous source without `min_power` can stay "on" at 0 W, so both have
little or no effect on it; give it a `min_power` if short cycling matters.
Both apply per flow: a source that feeds two storages compiles into one load
per flow, and each load counts its own starts. A source that switches from one
storage to the other therefore starts the second flow, and its `max_startups`
limits each flow separately, not the unit as a whole.

A heat pump requires `supply_temperature`, a `heating_curve`, or a
`cooling_curve`. A
constant-efficiency source requires `efficiency`.

Source type controls electric-balance membership by default:

- `heatpump`, `heat_pump`, and `electric` are electric loads;
- `gas`, `oil`, and `district` are non-electric loads; and
- `constant_efficiency` defaults to electric because its fuel is ambiguous.

An explicit `electric: true` or `electric: false` overrides the default. This
keeps a gas boiler's fuel input out of the household electric power balance.

#### Temperature-dependent COP refinement (`cop_solver`)

A heat pump's COP falls as it heats the storage hotter, but the optimizer
plans against a COP evaluated at an assumed temperature. When banking heat is
profitable, for example super-heating a buffer into surplus PV, the plan can
rely on a COP the unit cannot reach at that temperature. With
`cop_solver: auto`, EMHASS checks each curve-driven heat pump (`heating_curve`
or `cooling_curve`) after the solve and, only when the COP it used disagrees
with the temperature it planned, refines the storage's trajectory with an exact
dynamic program and solves once more. A constant `supply_temperature` source
has a fixed COP and is not refined.

The default is `static`: no refinement, exactly the previous behaviour. Turn it
on for a buffer or tank that a heat pump regularly charges well above its curve
supply temperature, and compare it with `static` on your own system: with
several coupled stores the refined plan is not always cheaper. A run where it
engages takes longer, because of the second solve. See
[the mathematical model](advanced_math_model.md) for the details.

#### Per-source temperature ceiling

When two sources feed the same storage but reach different maximum
temperatures, for example a DHW tank fed by a heat pump (condenser limit about
53 degrees Celsius) and an electric booster (up to 65 degrees Celsius), the
optimizer would otherwise let the cheaper heat pump cover the band above its
limit too, which it physically cannot deliver. `max_supply_temperature` makes
that limit hard: the source only injects heat while the storage is at or below
its ceiling, so the booster is scheduled exactly for the band above it. On a
storage with `thermal_inertia` the ceiling applies when the heat arrives, so heat
already on its way cannot push the storage past it.

```json
"sources": [
  {"id": "hp", "type": "heatpump", "nominal_power": 3500,
   "supply_temperature": 55, "max_supply_temperature": 53,
   "treat_as_semi_cont": false},
  {"id": "booster", "type": "electric", "nominal_power": 3000, "efficiency": 1.0}
]
```

A source may not push the storage past its ceiling within a step. A
semi-continuous source (the default) runs at its full nominal power, so if one
full-power step heats the storage by more than the gap between its temperature
and the ceiling, that source never runs and the other source does all the work,
with no warning. Make a capped source continuous (`"treat_as_semi_cont": false`),
as above, or use a shorter optimization time step.

This is a physical limit, not a preference: it also holds in the relaxed
fallback. To stop a source at a lower temperature while it can physically go
higher, use `overshoot_temperature` with a desired temperature on the storage
(see below); the two compose.

A per-step list applies step by step: the heat a source delivers during a step
stays within that step's ceiling, even when the next step's ceiling is higher.
When every source feeding a storage has a ceiling and the storage's
`min_temperature` lies above all of them at some step, the compiler logs a
warning: no source can heat the storage that far, so only a storage that is
already hot enough can hold that minimum. The ceiling is a heating limit: on a
cooling storage it does not apply, and the optimizer logs a warning when one
is set.

`supply_temperature` only sets the heat pump's COP; it is not a ceiling. Without
`max_supply_temperature` the optimizer may plan to heat the storage above the
supply temperature, and the compiler logs a warning for such a heat pump. To
cap the storage at the supply temperature, set `max_supply_temperature` equal
to `supply_temperature`. The cap is opt-in because it makes a storage whose
minimum temperature sits at the supply temperature infeasible.

### Storage

Each storage object requires:

| Field | Units | Description |
| --- | --- | --- |
| `id` | - | Unique storage ID. |
| `volume` | m3 | Active thermal-storage volume. |
| `start_temperature` | degrees Celsius | Temperature at the start of the optimization horizon. |
| `min_temperature` or `min_temperatures` | degrees Celsius | Per-timestep lower bounds. |
| `max_temperature` or `max_temperatures` | degrees Celsius | Per-timestep upper bounds. |

The graph compiler currently expects a list for either spelling. The singular
key name does not make a scalar valid: use `"min_temperature": [48.0]`, not
`"min_temperature": 48.0`.

Optional physical fields are `density` in kg/m3, `heat_capacity` in
kJ/(kg degree Celsius), and `thermal_loss`. Water defaults are approximately
`density: 1000` and `heat_capacity: 4.186`.

`min_temperature_curve` can provide a weather-compensated lower bound. Soft
comfort control is available through `desired_temperature` or
`desired_temperatures`, `overshoot_temperature`, `penalty_factor`, and
`comfort_sense` (`heat` or `cool`).

The storage's `overshoot_temperature` applies to every source that feeds it,
unless a source sets its own. That makes a two-stage setup: with
`desired_temperatures: 60` on the tank, `overshoot_temperature: 55` on the heat
pump and `75` on the electric element, the heat pump stops at 55 degrees
Celsius and the element lifts the tank above it only when the comfort penalty
justifies it. The desired temperature is soft, but the threshold is a hard stop
for its source. A continuous source does not heat (or cool) in a step that
would end beyond it, as for `thermal_config`. A semi-continuous source, the
default, is off while the storage starts a step beyond it, so one full-power
step can still carry the storage across. When every source feeding a storage
is continuous and has a threshold, a `min_temperature` above all of them (a
`max_temperature` below them when cooling) can only be held by a storage that
already starts there, and the compiler logs a warning for it.

If the storage starts below a minimum temperature it must meet soon (after a
cold night, a momentary sensor reading, or a setback floor that rises a few
steps later), its minimums are priced instead of hard for that run: every
degree below a configured minimum carries a high penalty. How long recovery
takes depends on the sources, the losses and any heat arriving through
transfers, so no fixed window is assumed. A storage that can recover quickly
still does so right away, because of the penalty, and then holds the minimum;
one that cannot recovers as fast as it can instead of making the problem
infeasible. Only the minimums of the first 6 timesteps trigger this; a minimum
that rises later in the horizon (for example a scheduled legionella cycle) stays
hard, because the plan can heat ahead for it.

#### Building-zone storage

A storage can also model a building zone (a house or a room) as a thermal mass
whose temperature drifts inside its comfort band. This is the load-shifting
model of the standalone [thermal model](thermal_model.md), available inside a
topology so a zone can be fed by the same hybrid sources as a tank. All fields
are optional; without them a storage is a water tank as before.

| Field | Units | Description |
| --- | --- | --- |
| `thermal_mass` | kWh/K | Heat capacity, instead of `volume`. |
| `loss_coefficient` | kW/K | Heat-loss coefficient UA. The loss becomes `UA * (T - outdoor)`, so a warmer zone loses more and the optimizer can pre-heat on cheap power and coast through a price peak. Cannot be combined with a `building_demand` consumer, which also models the loss to outdoor. |
| `thermal_inertia` | hours | Delay between the storage's own source heat and the temperature response, as in the thermal model. Applied in whole timesteps (rounded down) and capped at the horizon. Transfers are not lagged, so on a storage fed only by transfers it is ignored (with a warning). Over the first lagged steps no source heat arrives, so the minimum temperature there is priced rather than hard. |
| `window_area`, `shgc` | m2, fraction | Solar gain through glazing from the GHI forecast (`window_area * shgc * GHI`), which offsets the zone's heating need. `shgc` defaults to `0.6`. Applied only to a zone with `loss_coefficient`, and only when the weather data has GHI (open-meteo); otherwise it is zero. |

For example, a house held between 19.5 and 21.5 degrees Celsius, with a
soft target of 20.5, on a 48-step horizon (Python notation; convert it to JSON
as shown above):

```python
{
    "id": "house",
    "thermal_mass": 18,
    "loss_coefficient": 0.79,
    "start_temperature": 20.5,
    "min_temperatures": [19.5],  # a short minimum list is extended
    "max_temperatures": [21.5] * 48,  # a maximum list is not: one value per step
    "desired_temperatures": 20.5,
}
```

With these fields the older single-load models can be written as one-source
storage (their own configuration paths stay as they are):

| Model | As topology storage |
| --- | --- |
| [`thermal_battery`](thermal_battery.md) | one source, `volume`, and a `profile` consumer. |
| [`thermal_config`](thermal_model.md) | one source, `thermal_mass`, `loss_coefficient`, `thermal_inertia` and a comfort band. `cooling_constant` corresponds to `loss_coefficient / thermal_mass`. |

For predictable constraints, make the maximum-temperature array cover the
optimization horizon. A shorter minimum-temperature array is extended using
its final value, but a maximum-temperature array is not currently extended.

### Consumers

Consumers are folded into their target storage:

- `type: "profile"` supplies `profile`, a daily draw-off profile in kWh per
  timestep. EMHASS repeats it to fill a longer horizon.
- `type: "building_demand"` supplies the building-physics or degree-day fields
  described in [Thermal battery](thermal_battery.md). With the physics model
  (`u_value`, `envelope_area`, `ventilation_rate`, `heated_volume`), the
  optional `window_area`, `shgc`, and `internal_gains_factor` fields reduce the
  heating demand by window solar gain and internal gains, exactly as they do
  for a `thermal_battery`. Window solar gain needs a GHI forecast: the
  open-meteo weather method provides it (also on the `list` method, see below);
  other PV forecast methods do not, and the gain is then zero.
- `type: "pool_comfort"` supplies `solar_absorption_area` and optionally
  `solar_absorption_factor`.

Only one `building_demand` consumer is allowed per storage. Multiple profile
consumers targeting one storage are added element by element.

A `profile` and a `building_demand` consumer on the same storage add up, so one
storage can serve hot water and space heating at once (a combi tank). The
standing loss is counted once: the flat `thermal_loss` when a draw-off profile
is present, otherwise the indoor/outdoor loss. Set `indoor_target_temperature`
on the `building_demand` consumer; without it, a combi tank computes the
building demand against 20 degrees Celsius.

### Cost tracks

`cost_tracks` maps an ID to a per-timestep price series in currency/kWh of
source input. A source selects a series with `cost_track`.

For a hybrid heat-pump and gas-boiler system, omit `cost_track` from the heat
pump to retain the shared electricity tariff, and assign a separate gas-price
track to the boiler. Price arrays should cover the optimization horizon.

## Hybrid DHW example

This example models one 200 L DHW tank supplied by a 3.5 kW heat pump and a
25 kW gas boiler. It assumes 48 half-hour timesteps.

```python
HORIZON = 48

draw_off = [0.0] * 12 + [0.5, 0.3] + [0.0] * 22 + [0.8, 0.5, 0.3] + [0.0] * 9

heat_topology = {
    "sources": [
        {
            "id": "hp",
            "type": "heatpump",
            "supply_temperature": 55.0,
            "carnot_efficiency": 0.40,
            "nominal_power": 3500,
            "min_power": 800,
        },
        {
            "id": "gas",
            "type": "gas",
            "efficiency": 0.92,
            "nominal_power": 25000,
            "min_power": 8000,
            "cost_track": "gas_flat",
        },
    ],
    "storage": [
        {
            "id": "dhw",
            "volume": 0.20,
            "density": 997,
            "heat_capacity": 4.184,
            "start_temperature": 51.0,
            "min_temperature": [48.0] * HORIZON,
            "max_temperature": [62.0] * HORIZON,
            "thermal_loss": 0.035,
        }
    ],
    "consumers": [
        {
            "id": "dhw_draw",
            "type": "profile",
            "target": "dhw",
            "profile": draw_off,
        }
    ],
    "flows": [
        {"from": "hp", "to": "dhw"},
        {"from": "gas", "to": "dhw"},
    ],
    "cost_tracks": {
        "gas_flat": [0.085] * HORIZON,
    },
}
```

The compiler creates two deferrable loads and one shared thermal tank:

- load 0 is the electric heat pump;
- load 1 is the non-electric gas boiler;
- both loads contribute heat to the same tank state; and
- the gas load uses `gas_flat`, while the heat pump uses the shared electricity
  price.

The source input-to-heat relationship is:

```text
Q_thermal[t] = conversion_factor[t] * P_source[t] * timestep
```

For the heat pump, `conversion_factor` is its calculated COP. For the gas
boiler, it is the configured constant `efficiency`.

## Shared actuators

Use an actuator group when several graph flows represent one physical device
that cannot serve all targets simultaneously:

```python
"actuator_groups": [
    {
        "flows": [
            ["gas", "dhw"],
            ["gas", "buffer"],
        ],
        "mutual_exclusion": True,
        "max_combined_power": 25000,
    }
]
```

Each flow pair must exactly match an entry in `flows`.

`max_combined_power` adds a per-timestep cap on the sum of the member flows. It
does not replace their individual `min_power` and `nominal_power` limits.
`mutual_exclusion: true` additionally allows at most one member to be active.
For semi-continuous sources, an active flow runs at its nominal power, so the
group cap must be at least as large as every member that may run. For
continuous sources, the optimizer may modulate each active flow between its
individual minimum and nominal limits while respecting the group cap. A group
cap below a required member's feasible power can make the thermal problem
infeasible.

### Tank-to-tank transfers

A flow from one storage to another moves heat between them, for example a
buffer feeding a room through its emitters:

```json
{"from": "buffer", "to": "house", "transfer_coefficient": 0.8, "max_transfer_power": 6000}
```

| Field | Units | Description |
| --- | --- | --- |
| `transfer_coefficient` | kW/K | Emitter conductance, positive; default `1.0`. The transfer is at most `transfer_coefficient * (T_from - T_to)`, both at the start and at the end of the step. |
| `max_transfer_power` | W | Maximum transferred heat power, positive; default unlimited. |

Heat only flows from the hotter storage to the cooler one: when the receiver is
as warm as the feeder or warmer, the transfer is zero, and a step never ends
with the receiver warmer than its feeder. A storage that is only fed by a
transfer (no source flow) still gets a temperature state, but no
`min_temp_heater` / `max_temp_heater` / `target_temp_heater` columns.

## Publishing results

Each flow compiles to one deferrable load, numbered in the order of `flows`
(the first flow is load 0). The optimization results carry
`predicted_temp_heater{k}` (the temperature of the storage that load `k`
feeds) and `heating_demand_heater{k}` for every such load, just as for a
`thermal_battery`. When the storage sets `desired_temperature(s)`,
`min_temperature(s)` or `max_temperature(s)`, each of its loads also carries
`target_temp_heater{k}`, `min_temp_heater{k}` and `max_temp_heater{k}`, so the
comfort band can be plotted next to the predicted temperature. `min_temp_heater{k}`
is the floor the optimizer enforced, including a `min_temperature_curve`. To publish them
to Home Assistant, pass
`custom_predicted_temperature_id` and `custom_heating_demand_id` with one entry
per load index; see [Thermal battery](thermal_battery.md) for an example.
When several flows feed the same storage, each of their
`heating_demand_heater{k}` columns carries the storage's whole demand; do not
add them up.
When several flows feed the same storage, each of those loads reports that
storage's temperature. The ids are matched by position (entry `k` is load `k`);
loads without an entry publish under the default names, such as
`sensor.p_deferrable{k}` and `sensor.temp_predicted{k}`.

A storage fed only by a transfer has no load of its own; its temperature is in
`predicted_temp_heater{k}` with `k` equal to the number of deferrable loads plus
its position in the tank list (its position in `storage`, after any manual
`shared_thermal_tanks` in extend mode). Each transfer adds a `P_transfer_{from}_{to}` column
to the result (delivered heat, W): the schedule to drive the circulation pump
with. These columns are in the result CSV and the `/api/v1/plan` output; they
are not published as Home Assistant sensors.

### Rolling MPC

A topology is rebuilt on every run instead of reusing the cached (warm-start)
problem, so each run uses the live `start_temperature` and the current
forecast. This also holds for every day of a perfect-forecast run. Pass the
measured storage temperature as `start_temperature` in the `heat_topology` you
send at runtime, or send only `shared_tank_start_temperatures` (for example
`{"dhw": 48.5}`) to override the start temperature per storage id without
resending the topology. The solve takes a few seconds longer than a warm start.

## Combining with other deferrable loads

By default the compiler replaces the whole deferrable-load set with the
topology's flows, because it cannot tell configured loads apart from the
shipped defaults. If you also have ordinary deferrable loads (washing machine,
EV charger, ...) in the per-load arrays, set `extend_deferrable_loads` at the
top level of the topology:

```json
{
  "heat_topology": {
    "extend_deferrable_loads": true,
    "sources": ["..."],
    "storage": ["..."],
    "flows": ["..."]
  }
}
```

Your configured loads then keep indices `0..N-1`, and the topology's loads are
appended at `N..N+M-1`. `N` is your `number_of_deferrable_loads`: the shipped
defaults configure two example loads (3000 W for 4 h, and 750 W), so set
it to the number of ordinary loads you really have, or to 0. Shared-tank `load_ids` and actuator-group references are
shifted accordingly, and manually declared `shared_thermal_tanks` or
`deferrable_load_groups` entries are kept, with the compiled ones appended.
A per-load runtime parameter given as a single value (for example
`"nominal_power_of_deferrable_loads": 3000`) applies to your configured loads
only; the topology loads keep their compiled values.

```{warning}
The appended topology loads are numbered after your configured loads, so adding
or removing a manual load later renumbers them (`sensor.p_deferrable2` silently
changes meaning). Keep the manual load count stable once a topology is in use, or
remap the published entities with `custom_deferrable_forecast_id`.
```

## Validation and troubleshooting

EMHASS validates the graph before optimization and reports the offending field
path for:

- duplicate source or storage IDs;
- flows that reference unknown sources or storage;
- consumers that target unknown storage;
- unsupported source or consumer types;
- source or storage entries without an `id`, or an ID used by both a source and
  a storage;
- missing heat-pump supply-temperature data;
- a `profile` consumer without `profile`;
- a source `min_power` greater than its `nominal_power`;
- missing constant source efficiency; and
- missing cost-track references.

After successful compilation, the log contains:

```text
heat_topology compiled: <sources> sources, <storage> storage, <flows> flows, <groups> groups
```

If the topology is ignored, confirm that:

1. EMHASS is version 0.17.4 or newer;
2. the value is an object, not a quoted JSON string;
3. disabled configuration uses JSON `null`, not `"null"`; and
4. all temperature, demand, and cost arrays use the same timestep convention
   as the optimization horizon.

Heat-pump COP, heating curves, and building demand depend on the outdoor
temperature forecast:

- with `weather_forecast_method: "open-meteo"`, EMHASS retrieves it
  automatically;
- with the `list` method (PV passed as `pv_power_forecast`), EMHASS still
  fetches the open-meteo outdoor temperature and GHI for a heat topology; and
- an `outdoor_temperature_forecast` passed at runtime takes precedence over
  both.

If none of these is available, EMHASS uses a constant 15 degrees Celsius
without a separate warning. This happens, for example, on an offline install on
the `list` method, or with a PV forecast method that does not return weather data
(such as Solcast or solar.forecast); window solar gain is then zero as well. The
plan is still produced, but COP and heating demand are too optimistic in cold
weather, so pass `outdoor_temperature_forecast` in that case.
