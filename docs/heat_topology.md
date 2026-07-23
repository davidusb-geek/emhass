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

In the add-on configuration page, the field is an object input. Paste valid
JSON, not a quoted JSON string. For example, the Python configuration below
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
| `flows` | Directed source-to-storage connections. Each flow becomes one deferrable load. |
| `actuator_groups` | Optional constraints across flows that share physical equipment. |
| `cost_tracks` | Optional per-timestep energy prices referenced by sources. |

IDs must be unique within `sources` and `storage`. Every `flow.from` must match
a source ID, every `flow.to` must match a storage ID, and every
`consumer.target` must match a storage ID.

### Sources

Every source requires `id`, `type`, and `nominal_power`. Power values are in
watts of source input:

| Field | Description |
| --- | --- |
| `id` | Unique source ID. |
| `type` | `heatpump`, `heat_pump`, `gas`, `oil`, `district`, `electric`, or `constant_efficiency`. |
| `nominal_power` | Maximum source input power in W. |
| `min_power` | Optional minimum input power in W; default `0`. |
| `treat_as_semi_cont` | Optional on/off-at-nominal behavior; default `true`. |
| `supply_temperature` | Fixed heat-pump supply temperature in degrees Celsius. |
| `heating_curve` | Alternative heat-pump supply-temperature curve. |
| `carnot_efficiency` | Heat-pump Carnot efficiency; default `0.4`. |
| `efficiency` | Required constant conversion efficiency for gas, oil, district, electric, and constant-efficiency sources. |
| `cost_track` | Optional key in `cost_tracks`. Without it, the shared electricity tariff is used. |
| `electric` | Optional override for electric-balance membership. |

A heat pump requires either `supply_temperature` or a `heating_curve`. A
constant-efficiency source requires `efficiency`.

Source type controls electric-balance membership by default:

- `heatpump`, `heat_pump`, and `electric` are electric loads;
- `gas`, `oil`, and `district` are non-electric loads; and
- `constant_efficiency` defaults to electric because its fuel is ambiguous.

An explicit `electric: true` or `electric: false` overrides the default. This
keeps a gas boiler's fuel input out of the household electric power balance.

### Storage

Each storage object requires:

| Field | Units | Description |
| --- | --- | --- |
| `id` | - | Unique storage ID. |
| `volume` | m3 | Active thermal-storage volume. |
| `start_temperature` | degrees Celsius | Temperature at the start of the optimization horizon. |
| `min_temperature` or `min_temperatures` | degrees Celsius | Per-timestep lower bounds. |
| `max_temperature` or `max_temperatures` | degrees Celsius | Per-timestep upper bounds. |

Optional physical fields are `density` in kg/m3, `heat_capacity` in
kJ/(kg degree Celsius), and `thermal_loss`. Water defaults are approximately
`density: 1000` and `heat_capacity: 4.186`.

`min_temperature_curve` can provide a weather-compensated lower bound. Soft
comfort control is available through `desired_temperature` or
`desired_temperatures`, `overshoot_temperature`, `penalty_factor`, and
`comfort_sense` (`heat` or `cool`).

For predictable constraints, make the maximum-temperature array cover the
optimization horizon. A shorter minimum-temperature array is extended using
its final value, but a maximum-temperature array is not currently extended.

### Consumers

Consumers are folded into their target storage:

- `type: "profile"` supplies `profile`, a daily draw-off profile in kWh per
  timestep. EMHASS repeats it to fill a longer horizon.
- `type: "building_demand"` supplies the building-physics or degree-day fields
  described in [Thermal battery](thermal_battery.md).
- `type: "pool_comfort"` supplies `solar_absorption_area` and optionally
  `solar_absorption_factor`.

Only one `building_demand` consumer is allowed per storage. Multiple profile
consumers targeting one storage are added element by element.

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

draw_off = (
    [0.0] * 12
    + [0.5, 0.3]
    + [0.0] * 22
    + [0.8, 0.5, 0.3]
    + [0.0] * 9
)

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

## Validation and troubleshooting

EMHASS validates the graph before optimization and reports the offending field
path for:

- duplicate source or storage IDs;
- flows that reference unknown sources or storage;
- consumers that target unknown storage;
- unsupported source or consumer types;
- missing heat-pump supply-temperature data;
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
