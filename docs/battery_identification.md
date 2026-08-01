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
- `sensor_power_battery` and `sensor_battery_state_of_charge`: the two sensors above. With more than one battery these must be lists, one sensor per battery, see [Multiple batteries](#multiple-batteries) below.
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

## Multiple batteries

With `number_of_batteries` greater than 1, identification runs once per battery, so each pack gets its own capacity and round-trip efficiency estimate. For that to work it needs to know which sensors belong to which battery:

- Set `sensor_power_battery` and `sensor_battery_state_of_charge` each to a list of exactly `number_of_batteries` entries, index-matched to the battery config lists (battery 0 first, same order everywhere).
- There is no scalar broadcast for these two, unlike the numeric battery parameters. One meter cannot tell two batteries apart, so a single sensor name at `number_of_batteries > 1` is not accepted. If either list is missing or the wrong length, identification skips with a warning that says which key is wrong and what it expected, and the rest of the optimization runs as normal.

Each battery is fitted independently. One pack can pass while another does not have enough usable cycles yet; the pack that passed is reported, the other keeps your configured values and is retried on later runs. The re-fit age (`battery_identification_model_max_age`) is also tracked per battery, from each battery's own last successful fit, so one pack's fresh result never delays another pack's retry. Each battery's cached result is bound to the exact sensor pair it was fitted from, so editing or reordering the sensor lists is safe: the affected battery just re-fits on the next run instead of serving a result from the wrong pack.

History retrieval itself is one shared batch per cycle covering every battery that is currently due for a re-fit, not a separate call per battery. If one of those batteries' sensors is unreachable or has gaps, the whole batch can fail or lose days, and the currently-due batteries defer to a later run; batteries whose cached result is still fresh are unaffected either way, since they are never part of that batch.

Under `suggest`, the published sensors follow the same per-battery naming as the rest of the multi-battery support: `sensor.battery_identified_capacity_battery0`, `sensor.battery_identified_round_trip_efficiency_battery0`, and so on per battery index. With a single battery the sensor names stay exactly as above and nothing changes.

## Limitations

- Advisory only in this version. It never overwrites the values the optimizer uses.
- It identifies a single lumped round-trip efficiency and cannot split it into separate charge and discharge figures, so it reports both as the square root of the round-trip efficiency and says so.
- Power-dependent efficiency, standby draw, and a DC-side charge/discharge split are not modelled. These are known limitations left for a later version.
