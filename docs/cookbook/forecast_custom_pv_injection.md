# Custom PV forecast injection via runtime params

## Goal

Send your own PV forecast to `naive-mpc-optim` on every MPC call, as a timestamp to W mapping, with an optional P10 curve as `pv_power_forecast_p10` and the live battery state of charge as `soc_init`. The source can be any sensor in the Solcast format: the Solcast integration, [RoofCast](https://github.com/JustAnotherPU/roofcast), or Forecast.Solar or another service reshaped by a template sensor.

## Prerequisites

- EMHASS >= 0.18.4 for `pv_power_forecast_p10` (CHANGELOG 0.18.4: "support caller-supplied PV P10"). The finite-number check that fails the cycle on bad values came in 0.18.5, and the rejection of `soc_init` above 1 in 0.18.6. From 0.18.6 the alignment and pair checks are the same code as on master. This recipe was run on 0.18.8 only. Master was not run: the source comments were read on master at commit 561c00e.
- One inverter. Releases up to 0.18.8 have no `number_of_inverters` key. On later versions it must stay at 1 (the default): above one inverter EMHASS wants one mapping per inverter, see [Two hybrid inverters, each with its own battery](battery_two_hybrid_inverters.md).
- A PV sensor with a `detailedForecast` attribute: a list of rows with `period_start`, `pv_estimate` in kW and, optionally, `pv_estimate10` in kW. The snippet assumes 30-minute rows, like Solcast.
- A battery SOC sensor in percent, if you have a battery.
- Transport tested against:
  - Home Assistant 2026.11 and the EMHASS 0.18.8 web server at a 15-min step (tested with the RoofCast example package: payload rendered in Home Assistant from RoofCast rows, which carry ISO string timestamps, and POSTed). That package selects its source with the logic of Step 4, written inline in the `for` line. From the `t0` line on, its payload is identical to Step 3.
  - Step 3 as printed, reading the two Solcast sensors with datetime `period_start` values: rendered offline only, with the template engine of the `homeassistant` 2026.1.2 package and stub states. Untested in a running Home Assistant, contribution welcome.
  - 15 and 30-min steps and a bias of 0.5: run offline with the EMHASS 0.18.8 CLI.
  - 60-min step: untested, contribution welcome.
  - Step 4: source sensor and Solcast payload rendered offline the same way, payload run with the EMHASS 0.18.8 CLI. Untested in a running Home Assistant, contribution welcome.

The values you send must follow the [Forecast input contract](../passing_data.md#forecast-input-contract). This recipe does not repeat it.

## Step 1: Check your static EMHASS config

<!-- source: src/emhass/data/config_defaults.json:4 -->
<!-- source: src/emhass/data/config_defaults.json:63 -->
<!-- source: src/emhass/data/config_defaults.json:142 -->
<!-- source: src/emhass/web_server.py:444 (GET /get-config) -->
<!-- transport: EMHASS config.json or add-on options (transport-independent) -->

The snippet needs to know your `optimization_time_step`, so note it. `weather_forecast_pv_quantile_bias` decides whether P10 does anything. Its default is 0.0, which means the plan uses P50 only.

```yaml
optimization_time_step: 30              # 15 and 30 verified; set step_min in Step 3 to the same value
weather_forecast_pv_quantile_bias: 0.0  # 0 = P50 only; 0.3 = 0.3 * P10 + 0.7 * P50
set_use_battery: true                   # only if you send soc_init
```

You do not need to change `weather_forecast_method`. When the runtime payload carries `pv_power_forecast`, EMHASS switches to the `list` method for that call.

Expected: EMHASS restarts cleanly and `GET /get-config` shows your step and bias.

## Step 2: Understand why the snippet sends a mapping

<!-- source: src/emhass/utils.py:3268 (the MPC grid is cut to prediction_horizon) -->
<!-- source: src/emhass/utils.py:3465-3475 (a list is used as is when it has enough values) -->
<!-- source: src/emhass/utils.py:2131-2170 (_align_runtime_forecast_mapping) -->

A plain list is positional: one value per optimization step. Solcast rows are 30 minutes long. On a 15-minute step, a list of 30-minute values is read as 15-minute values. The day is squeezed into half the time and EMHASS has no way to notice: the list only needs at least `prediction_horizon` values.

A mapping of timestamp to W is placed by time instead. `_align_runtime_forecast_mapping` does three things:

<!-- source: src/emhass/utils.py:2159 (mean per bucket) -->
<!-- source: src/emhass/utils.py:2168 (ffill then bfill) -->

1. It averages the points that fall in one optimization step.
2. It holds each value until the next point (hold-last). On a 15-minute step, a 30-minute row covers two steps. On a 60-minute step, two 30-minute rows are averaged into one step.
3. It fills steps before the first point with the first value.

There is no interpolation. If the horizon runs past your last point, the last value is held. For Solcast that value is 0 W late at night, but it is not a forecast. That is why Step 3 caps the horizon to what the forecast covers.

<!-- source: src/emhass/utils.py:2173 (_validate_and_align_external_pv_pair) -->

P50 and P10 must have the same timestamps. The pair is checked together and a mismatch fails the cycle.

Expected: you know which form you send. This recipe sends a timestamped mapping, so no length check on the PV values is needed.

## Step 3: Build the payload in a REST command

<!-- transport: Home Assistant 2026.11 (tested with RoofCast rows and the Step 4 source logic written inline, POSTed to the EMHASS 0.18.8 web server at a 15-min step; the Solcast sources printed here were rendered offline only with homeassistant 2026.1.2, untested in a running Home Assistant, contribution welcome) -->
<!-- source: src/emhass/utils.py:3100 and src/emhass/utils.py:5994 (soc_init above 1 is rejected) -->
<!-- source: src/emhass/optimization.py:10013 (MPC horizon below 5 steps is refused) -->

Set the four values at the top. Notes on each part:

- `step_min` must equal `optimization_time_step`. It only sets `prediction_horizon`. If it is smaller than the real step, the horizon runs past the forecast and EMHASS holds the last value.
- `soc_init` is a fraction from 0 to 1, not a percent, and EMHASS raises an error for a value above 1. The snippet caps the sensor at 100 before dividing by 100. If the SOC sensor is `unavailable`, `float` without a default makes the template fail, so Home Assistant sends nothing. The cycle is skipped instead of planning from a made-up SOC. No battery: remove the `soc_init` entry.
- Rows that started more than 30 minutes ago are dropped. The row that contains "now" is kept so it anchors the first step.
- A missing `pv_estimate10` falls back to the P50 value of that row, so both mappings keep the same timestamps. A row whose `pv_estimate` is missing or not a number makes the template fail, so nothing is sent. The one exception is a boolean: `true` passes as 1000 W.
- `covered` is the number of steps the forecast covers. `prediction_horizon` is the smaller of `covered` and `max_steps` (`horizon_h` counted in steps). EMHASS refuses an MPC horizon under 5 steps, so with fewer steps the PV keys are left out (see Caveats).

```yaml
rest_command:
  emhass_naive_mpc_optim:
    url: http://localhost:5000/action/naive-mpc-optim  # your EMHASS URL
    method: POST
    content_type: application/json
    payload: >-
      {%- set step_min = 30 -%}
      {%- set horizon_h = 24 -%}
      {%- set soc_entity = 'sensor.battery_soc' -%}
      {%- set sources = ['sensor.solcast_pv_forecast_forecast_today', 'sensor.solcast_pv_forecast_forecast_tomorrow'] -%}
      {%- set ns = namespace(rows=[]) -%}
      {%- for e in sources -%}
        {%- set ns.rows = ns.rows + (state_attr(e, 'detailedForecast') or []) -%}
      {%- endfor -%}
      {%- set t0 = now().timestamp() - 1800 -%}
      {%- set out = namespace(k=[], p50=[], p10=[], last=0) -%}
      {%- for r in ns.rows if as_timestamp(r.period_start, 0) > t0 -%}
        {%- set ts = as_timestamp(r.period_start) -%}
        {%- set out.k = out.k + [ts | timestamp_utc] -%}
        {%- set out.p50 = out.p50 + [(r.pv_estimate * 1000) | round(0)] -%}
        {%- set out.p10 = out.p10 + [((r.pv_estimate10 if r.pv_estimate10 is number else r.pv_estimate) * 1000) | round(0)] -%}
        {%- set out.last = [out.last, ts] | max -%}
      {%- endfor -%}
      {%- set max_steps = (horizon_h * 60 / step_min) | int -%}
      {%- set p = {'soc_init': ([states(soc_entity) | float, 100] | min / 100) | round(3), 'prediction_horizon': max_steps} -%}
      {%- set covered = ((out.last + 1800 - now().timestamp()) / (step_min * 60)) | int -%}
      {%- if out.k | count > 0 and covered >= 5 -%}
        {%- set p = dict(p, prediction_horizon=[covered, max_steps] | min,
                         pv_power_forecast=dict(zip(out.k, out.p50)),
                         pv_power_forecast_p10=dict(zip(out.k, out.p10))) -%}
      {%- endif -%}
      {{ p | tojson }}
```

For a single sensor whose `detailedForecast` spans several days (RoofCast does), set `sources = ['sensor.pv_forecast_today']`. The Solcast integration splits today and tomorrow over two sensors, so the example reads both. The two `1800` are the 30-minute row length.

Expected: rendered in Developer tools, Template, the payload looks like this (values shortened):

```json
{"prediction_horizon": 48,
 "pv_power_forecast": {"2026-06-01T05:00:00+00:00": 0, "2026-06-01T05:30:00+00:00": 410, "2026-06-01T06:00:00+00:00": 980},
 "pv_power_forecast_p10": {"2026-06-01T05:00:00+00:00": 0, "2026-06-01T05:30:00+00:00": 220, "2026-06-01T06:00:00+00:00": 560},
 "soc_init": 0.62}
```

## Step 4 (optional): Fall back to Solcast when your own source is stale

<!-- transport: Home Assistant template (rendered offline with the homeassistant 2026.1.2 template engine and stub states, payload run with the EMHASS 0.18.8 CLI; untested in a running Home Assistant, contribution welcome) -->

If your own forecast can go stale, choose the source with a template sensor and let the payload follow it. The freshness test below uses the `generated_at` and `models_ok` attributes that RoofCast publishes. Replace it with whatever freshness signal your source has.

```yaml
template:
  - sensor:
      - name: PV forecast source
        unique_id: pv_forecast_source
        state: >-
          {% set g = state_attr('sensor.pv_forecast_today', 'generated_at') %}
          {{ 'sensor.pv_forecast_today'
             if g is not none and (now() - as_datetime(g)).total_seconds() < 7200
                and state_attr('sensor.pv_forecast_today', 'models_ok') | int(0) >= 3
             else 'sensor.solcast_pv_forecast_forecast_today' }}
```

Then replace the `sources` line of Step 3 with these three lines:

```jinja
{%- set fallback = ['sensor.solcast_pv_forecast_forecast_today', 'sensor.solcast_pv_forecast_forecast_tomorrow'] -%}
{%- set src = states('sensor.pv_forecast_source') -%}
{%- set sources = fallback if src in fallback or src in ['unknown', 'unavailable'] else [src] -%}
```

Expected: when your sensor is older than two hours, the rendered payload carries the Solcast curve, read from both Solcast sensors.

## Step 5: Run it and read the EMHASS log

<!-- source: src/emhass/utils.py:3534 -->

Call `rest_command.emhass_naive_mpc_optim` from the automation that already runs your MPC, or after each forecast update.

Expected: the EMHASS log shows `Aligned paired external PV P50/P10 forecast from N timestamped point(s) onto M optimization timestep(s).`, with M equal to the `prediction_horizon` you sent, and `P_PV` in the result follows your P50 curve step by step (with a bias above 0, the blend of P50 and P10).

## Caveats

<!-- source: src/emhass/utils.py:3492-3493 and src/emhass/utils.py:3539-3540 (omitted PV keys) -->
<!-- source: src/emhass/utils.py:3501-3502 (supplying P10 selects the list method) -->

- **PV keys omitted.** With no forecast rows, or fewer than 5 steps of them, the snippet leaves out both PV keys. EMHASS then uses the `weather_forecast_method` from your config for that call. This is different from keys that are sent but invalid: those fail the cycle and EMHASS does not fall back.

<!-- source: src/emhass/forecast.py:1328 (_blend_external_pv_p10) -->
<!-- source: src/emhass/forecast.py:1364-1365 (bias * P10 + (1 - bias) * P50) -->
<!-- source: src/emhass/forecast.py:1539-1554 (warning when the bias cannot apply) -->

- **P10 only matters when the bias is above 0.** The blend is `bias * P10 + (1 - bias) * P50`. With the default 0.0 the P10 values are still validated, then ignored. On this `list` path, a bias set without a P10 companion logs a warning and is ignored.

<!-- source: src/emhass/utils.py:2807-2810 (runtime keys from associations.csv override the config) -->
<!-- source: src/emhass/data/associations.csv:94 -->

- **Bias per call.** You can add `"weather_forecast_pv_quantile_bias": 0.3` to the payload instead of the config. Runtime keys listed in `associations.csv` override the config for that call.

<!-- source: src/emhass/utils.py:3101-3109 -->

- **`soc_init` outside your min/max bounds** only logs a warning. EMHASS keeps the value you sent.
- **Stale SOC.** A sensor that only updates on change can be hours old. Check `last_updated` if your battery reports rarely.
- **Timestamps.** `timestamp_utc` renders ISO 8601 with a `+00:00` offset, as the contract asks.
- **Row length.** The `1800` in the snippet assumes 30-minute rows. Rows of another length are untested.
- **Forecast.Solar and other sources.** Build a template sensor whose `detailedForecast` attribute has the same row shape. This path is untested.

## Credits

- Pattern from Discussion #1341 (@JustAnotherPU), REST command reworked by @LesIT1 in JustAnotherPU/roofcast#1.
- Field names verified against `src/emhass/utils.py:treat_runtimeparams` on 2026-11-13.
