# Battery self-identification

EMHASS can learn two of the battery constants the optimizer otherwise takes on trust, the usable capacity and the round-trip efficiency, from your own Home Assistant history. It follows the same learn-from-history pattern as the adjusted PV forecast (`set_use_adjusted_pv`).

This is opt-in and default off. In this first version it is advisory only: it reports what it finds but never changes the battery values the optimizer uses. You decide whether to act on the estimate.

## What it needs

- `set_use_battery` must be True.
- A signed AC-side battery power sensor, set as `sensor_power_battery` (in Watts). Either sign convention works, the direction is auto-detected.
- The measured battery state of charge, set as `sensor_battery_state_of_charge` (in percent).
- Enough history. The fit needs weeks of signed power and state of charge with enough reasonably deep charge and discharge cycles. If the data is too shallow, or the fit fails an internal sanity check, it publishes nothing and keeps your configured values.

The two sensors are only retrieved when battery self-identification is enabled, so they cost nothing on a normal run.

## Enabling it

Set these in your configuration:

- `set_use_battery_identification`: True to turn the feature on (default False).
- `sensor_power_battery` and `sensor_battery_state_of_charge`: the two sensors above.
- `battery_identification_trust_tier`: `observe` (default) or `suggest`, see below.
- `battery_identification_model_max_age`: how many hours before the estimate is re-fitted from fresh history. Default 24. Set to 0 to re-fit on every call. Like the adjusted-PV cache, this avoids re-pulling history on every run.

See the [configuration reference](config.md) for the full parameter descriptions.

## Trust tiers

`battery_identification_trust_tier` controls what happens with the estimate. Neither tier ever changes the configured battery values the optimizer uses.

- `observe` (default): the estimate is written to a JSON file (`battery_identification.json`) under the data path and to the log. Nothing is published to Home Assistant.
- `suggest`: in addition to the file and log, two read-only sensors are published to Home Assistant and a recommendation is logged.

## Reading the result

Under `suggest`, two sensors appear:

- `sensor.battery_identified_capacity`, the usable capacity in kWh.
- `sensor.battery_identified_round_trip_efficiency`, the lumped round-trip efficiency.

Each sensor carries its confidence interval (`ci_low` / `ci_high`), an internal cross-check, the time of the last successful fit (`fitted_at`), the number of charge and discharge segments used, and the assumptions, so you can judge how much to trust it from the sensor attributes alone.

To compare the capacity against your configuration, note the units: the sensor is in kWh while `battery_nominal_energy_capacity` is in Wh, so multiply the sensor value by 1000 before comparing. If you trust the estimate, update `battery_nominal_energy_capacity`, `battery_charge_efficiency` and `battery_discharge_efficiency` by hand.

## Limitations

- Advisory only in this version. It never overwrites the values the optimizer uses.
- It identifies a single lumped round-trip efficiency and cannot split it into separate charge and discharge figures, so it reports both as the square root of the round-trip efficiency and says so.
- Power-dependent efficiency, standby draw, and a DC-side charge/discharge split are not modelled. These are known limitations left for a later version.
