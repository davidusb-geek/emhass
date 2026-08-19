# Battery self-identification

EMHASS can learn two of the battery constants the optimizer otherwise takes on trust, the usable capacity and the round-trip efficiency, from your own Home Assistant history. It follows the same learn-from-history pattern as the adjusted PV forecast (`set_use_adjusted_pv`).

This is opt-in and default off. In this first version it is advisory only: it reports what it finds but never changes the battery values the optimizer uses. You decide whether to act on the estimate.

## What it needs

- `set_use_battery` must be True.
- A signed AC-side battery power sensor, set as `sensor_power_battery` (in Watts). Either sign convention works, the direction is auto-detected.
- The measured battery state of charge, set as `sensor_battery_state_of_charge` (in percent).
- Enough history. The fit needs weeks of signed power and state of charge with enough reasonably deep charge and discharge cycles. If the data is too shallow, or the fit fails an internal sanity check, that run publishes nothing new; a previously identified estimate, if any, keeps being served from the next run on (see [Reading the result](#reading-the-result)). The failed attempt itself is recorded, so the next retry backs off for `battery_identification_model_max_age` instead of re-pulling history and re-fitting on every run.

The two sensors are only retrieved when battery self-identification is enabled, so they cost nothing on a normal run. Their signed values are kept intact on this retrieval regardless of the `set_zero_min` data cleaning setting, which continues to sanitize the load data as usual.

Stretches of history with no recorded data are excluded from the fit. Any step between two samples longer than three times the `optimization_time_step` counts as a recorder gap: the battery state across it is unobserved, so it contributes no energy throughput and no state of charge change, and any charge or discharge run in progress ends at the last sample before the gap. This means history riddled with long gaps (a recorder outage, or a power sensor that only reports on changes and stays silent for hours) can honestly come back as "not enough data" rather than producing an estimate from invented energy. Shorter dropouts, up to two consecutive missing steps, are still bridged.

## Enabling it

Set these in your configuration:

- `set_use_battery_identification`: True to turn the feature on (default False).
- `sensor_power_battery` and `sensor_battery_state_of_charge`: the two sensors above. With more than one battery these must be lists, one sensor per battery, see [Multiple batteries](#multiple-batteries) below.
- `battery_identification_trust_tier`: `observe` (default) or `suggest`, see below.
- `battery_identification_model_max_age`: how many hours before a fit is attempted again from fresh history. This applies whether the last attempt succeeded or failed - a failed attempt is recorded too, so a setup whose fits keep failing retries at this cadence instead of re-pulling history every run. Default 24. Set to 0 to attempt a fit on every call. Like the adjusted-PV cache, this avoids re-pulling history when nothing needs to change. To force an immediate retry right after fixing a sensor or config issue, set this to 0 for one run, or delete `battery_identification.json`.

See the [configuration reference](config.md) for the full parameter descriptions.

## Trust tiers

`battery_identification_trust_tier` controls what happens with the estimate. Neither tier ever changes the configured battery values the optimizer uses.

- `observe` (default): the estimate is written to a JSON file (`battery_identification.json`) under the data path and to the log. Nothing is published to Home Assistant.
- `suggest`: in addition to the file and log, two read-only sensors are published to Home Assistant and a recommendation is logged.

## Reading the result

Under `suggest`, two sensors appear:

- `sensor.battery_identified_capacity`, the usable capacity in kWh.
- `sensor.battery_identified_round_trip_efficiency`, the lumped round-trip efficiency.

Each sensor carries its confidence interval (`ci_low` / `ci_high`), an internal cross-check, the time the reported fit was actually made (`fitted_at`), the number of charge and discharge segments used, and the assumptions, so you can judge how much to trust it from the sensor attributes alone. If later fit attempts fail, EMHASS keeps serving this same estimate rather than falling silent (each failed attempt is still logged, so check the log if you want to know it's happening) - `fitted_at` is how you can tell the estimate itself has stopped getting fresher even though nothing looks wrong on the sensor itself.

To compare the capacity against your configuration, note the units: the sensor is in kWh while `battery_nominal_energy_capacity` is in Wh, so multiply the sensor value by 1000 before comparing. If you trust the estimate, update `battery_nominal_energy_capacity`, `battery_charge_efficiency` and `battery_discharge_efficiency` by hand.

## Multiple batteries

With `number_of_batteries` greater than 1, identification runs once per battery, so each pack gets its own capacity and round-trip efficiency estimate. For that to work it needs to know which sensors belong to which battery:

- Set `sensor_power_battery` and `sensor_battery_state_of_charge` each to a list of exactly `number_of_batteries` entries, index-matched to the battery config lists (battery 0 first, same order everywhere).
- There is no scalar broadcast for these two, unlike the numeric battery parameters. One meter cannot tell two batteries apart, so a single sensor name at `number_of_batteries > 1` is not accepted. If either list is missing or the wrong length, identification skips with a warning that says which key is wrong and what it expected, and the rest of the optimization runs as normal.

Each battery is fitted independently. One pack can pass while another does not have enough usable cycles yet; the pack that passed is reported, the other keeps your configured values, and the failed attempt is recorded so that the pack retries at the same `battery_identification_model_max_age` cadence instead of re-fitting every run. The re-fit age is also tracked per battery, from each battery's own last attempt (successful or not), so one pack's fresh result - or fresh backoff - never delays another pack's retry. Each battery's cached result, and each recorded failed attempt, is bound to the exact sensor pair it came from, so editing or reordering the sensor lists is safe: the affected battery just re-fits on the next run instead of serving a result, or a backoff, from the wrong pack.

History retrieval itself is one shared batch per cycle covering every battery that is currently due for a re-fit, not a separate call per battery. If retrieval hits a hard error (an unreachable Home Assistant instance, an auth failure), the whole batch still fails and the currently-due batteries defer to a later run; batteries whose cached result is still fresh are unaffected either way, since they are never part of that batch.

Within that shared batch, one sensor having no data for a given day no longer drops the whole day for every other sensor in the batch. A day is only skipped outright if every requested sensor is missing for it; otherwise the day is kept and the sensor that was missing simply shows up as NaN for that stretch, same as a recorder gap. This means a stale or newly added battery no longer costs a healthy battery days of history it did have.

Under `suggest`, the published sensors follow the same per-battery naming as the rest of the multi-battery support: `sensor.battery_identified_capacity_battery0`, `sensor.battery_identified_round_trip_efficiency_battery0`, and so on per battery index. With a single battery the sensor names stay exactly as above and nothing changes.

## Limitations

- Advisory only in this version. It never overwrites the values the optimizer uses.
- It identifies a single lumped round-trip efficiency and cannot split it into separate charge and discharge figures, so it reports both as the square root of the round-trip efficiency and says so.
- Power-dependent efficiency, standby draw, and a DC-side charge/discharge split are not modelled. These are known limitations left for a later version.
