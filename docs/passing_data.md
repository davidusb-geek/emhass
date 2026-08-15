# Passing data to EMHASS

## Passing your own data

In EMHASS we have 4 forecasts to deal with:

- PV power production forecast (internally based on the weather forecast and the characteristics of your PV plant). This is given in Watts.

- Load power forecast: how much power your house will demand in the next 24 hours. This is given in Watts.

- Load cost forecast: the price of the energy from the grid in the next 24 hours. This is given in EUR/kWh.

- PV production selling price forecast: at what price are you selling your excess PV production in the next 24 hours. This is given in EUR/kWh.

The sensor containing the load data should be specified in the parameter `sensor_power_load_no_var_loads` in the configuration file. As we want to optimize household energy, we need to forecast the load power consumption. The default method for this is a naive approach using 1-day persistence. The load data variable should not contain the data from the deferrable loads themselves. For example, let's say that you set your deferrable load to be the washing machine. The variables that you should enter in EMHASS will be: `sensor_power_load_no_var_loads: 'sensor.power_load_no_var_loads'` and `sensor.power_load_no_var_loads = sensor.power_load - sensor.power_washing_machine`. This is supposing that the overall load of your house is contained in the variable: `sensor.power_load`. The sensor `sensor.power_load_no_var_loads` can be easily created with a new template sensor in Home Assistant.

If you are implementing an MPC controller, then you should also need to provide some data at the optimization runtime using the key `runtimeparams`.

The valid values to pass for both forecast data and MPC-related data are explained below.

### Forecast data at runtime

It is possible to provide EMHASS with your own forecast data. For this just add the data as a list of values to a data dictionary during the call to `emhass` using the `runtimeparams` option. 

For example, if using the add-on or the standalone docker installation you can pass this data as a list of values to the data dictionary during the `curl` POST:
```bash
curl -i -H 'Content-Type:application/json' -X POST -d '{"pv_power_forecast":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75, 804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}' http://localhost:5000/action/dayahead-optim
```

Or the equivalent `rest_command` implementation:
```yaml
rest_command:
  dayahead_optim:
    url: http://localhost:5000/action/dayahead-optim
    method: post
    content_type: application/json
    payload: >
      {
        "pv_power_forecast": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75, 804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      }
```

Or if using the legacy method using a Python virtual environment:
```bash
emhass --action 'dayahead-optim' --config ~/emhass/config.json --runtimeparams '{"pv_power_forecast":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75, 804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}'
```

The possible dictionary keys to pass data are:

- `pv_power_forecast` for the PV power production forecast.

- `load_power_forecast` for the Load power forecast.

- `load_cost_forecast` for the Load cost forecast.

- `prod_price_forecast` for the PV production selling price forecast.

#### Passing a forecast as timestamped values

Instead of a plain list, any of these forecast keys can be passed as an object that maps ISO 8601 timestamps to values. EMHASS aggregates the points to the optimization time step and then holds each value until the next provided point (step interpolation), so you only need to supply a point where the value changes. A point whose timestamp falls before the start of the optimization window is used to anchor the first values of the horizon.

```json
{
  "load_cost_forecast": {
    "2024-01-01T00:00:00+01:00": 0.20,
    "2024-01-01T06:00:00+01:00": 0.15,
    "2024-01-01T18:00:00+01:00": 0.30
  }
}
```

### Passing other data at runtime

It is possible to also pass other data during runtime to automate energy management. For example, it could be useful to dynamically update the total number of hours for each deferrable load (`operating_hours_of_each_deferrable_load`) using for instance a correlation with the outdoor temperature (useful for water heater for example). 

A machine-readable schema of the runtime-only optimization parameters is committed at
`src/emhass/static/data/runtime_params.json` (consumed by the generated OpenAPI spec and tooling).

Here is the list of the other additional dictionary keys that can be passed at runtime:

- `number_of_deferrable_loads` for the number of deferrable loads to consider.

- `nominal_power_of_deferrable_loads` for the nominal power for each deferrable load in Watts.

- `operating_hours_of_each_deferrable_load` for the total number of hours that each deferrable load should operate.
  - Alteratively, you can pass `operating_timesteps_of_each_deferrable_load` to set the total number of timesteps for each deferrable load. *(better parameter to use for setting under 1 hr)* 

- `start_timesteps_of_each_deferrable_load` for the timestep from which each deferrable load is allowed to operate (if you don't want the deferrable load to use the whole optimization timewindow).

- `end_timesteps_of_each_deferrable_load` for the timestep before which each deferrable load should operate (if you don't want the deferrable load to use the whole optimization timewindow).

- `def_current_state` Pass this as a list of booleans (True/False) to indicate the current deferrable load state. This is used internally to avoid incorrectly penalizing a deferrable load start if a forecast is run when that load is already running.

- `treat_deferrable_load_as_semi_cont` to define if we should treat each deferrable load as a semi-continuous variable.

- `set_deferrable_load_single_constant` to define if we should set each deferrable load as a constant fixed value variable with just one startup for each optimization task.

- Note: for the per-load array parameters above (and also `deferrable_load_max_cost`), a value shorter than `number_of_deferrable_loads` is padded with that parameter's default, whether it came from your configuration or from this runtime payload. A runtime-supplied short array also logs a warning naming the parameter and how many entries were added; a config-sourced one stays silent. Pass a full-length array to avoid the warning.

- `solcast_api_key` for the SolCast API key if you want to use this service for PV power production forecast.

- `solcast_rooftop_id` for the ID of your rooftop for the SolCast service implementation.

- `solar_forecast_kwp` for the PV peak installed power in kW used for the solar.forecast API call. 

- `battery_minimum_state_of_charge` the minimum possible SOC.

- `battery_maximum_state_of_charge` the maximum possible SOC.

- `battery_target_state_of_charge` for the desired target value of the initial and final SOC.

- `battery_discharge_power_max` for the maximum battery discharge power.

- `battery_charge_power_max` for the maximum battery charge power.

- `publish_prefix` use this key to pass a common prefix to all published data. This will add a prefix to the sensor name but also the forecast attribute keys within the sensor.

- `current_period_peak` the peak grid import (in Watts) already incurred during the current billing period. Only used by `naive-mpc-optim`, and only when a capacity/demand charge (`capacity_cost_per_kw`) is configured. See the dedicated section below.

### Requiring an intermediate battery SOC target (naive-mpc-optim)

By default the battery SOC is only pinned at the start (`soc_init`) and end (`soc_final`) of the horizon, leaving the optimizer free to choose the trajectory in between. Sometimes you want to *guarantee* the battery reaches a given charge level **partway through** the horizon — for example, "make sure the battery is full by the time the cheap window ends, even though it's fine to discharge it afterwards". This is the request in [issue #553](https://github.com/davidusb-geek/emhass/issues/553).

Two optional runtime keys (only used by `naive-mpc-optim`) make this possible:

- `soc_target`: the desired *minimum* SOC as a fraction in `[0, 1]` (NOT percent). It is clamped to `[battery_minimum_state_of_charge, battery_maximum_state_of_charge]`.
- `soc_target_timestep`: the 0-based horizon timestep by which `soc_target` must be reached. After this step the battery is free to discharge again. Defaults to the last timestep if `soc_target` is given without it.

Both default to *unset* (`None`), in which case nothing changes — the constraint is a no-op and existing behaviour is preserved. The target is a one-sided constraint (`SOC >= soc_target` at that step), so it never forces the battery to discharge.

For example, the following runtime payload tells EMHASS that the battery must reach 100% SOC by horizon step index 14 (the SoC after that step, ~7.5 h into a 30-min-step horizon), while still letting it discharge in the steps after that:

```json
{
  "prediction_horizon": 48,
  "soc_init": 0.30,
  "soc_target": 1.0,
  "soc_target_timestep": 14
}
```

As an HA `rest_command`:

```yaml
rest_command:
  naive_mpc_optim:
    url: http://localhost:5000/action/naive-mpc-optim
    method: post
    content_type: application/json
    payload: >
      {
        "prediction_horizon": 48,
        "soc_init": {{ states('sensor.battery_soc') | float / 100 }},
        "soc_target": 1.0,
        "soc_target_timestep": 14
      }
```

```{warning}
The target must be *reachable* by its timestep. If the requested SOC cannot be achieved in time given `soc_init`, `battery_charge_power_max` and the optimization time step, EMHASS logs a warning and the LP is likely to be **infeasible** (the target is a hard constraint, so it is not silently relaxed). Size `soc_target` / `soc_target_timestep` to what your charge power actually allows (e.g. don't ask for +0.7 SOC in 3 steps if a full charge takes 10). Note too that with the default `set_nodischarge_to_grid: true` a high target can be infeasible if local load is too low to shed the stored energy back down to `soc_final`. `soc_target_timestep` is clamped to `[0, prediction_horizon - 1]`.
```

### Capping the demand charge at the peak already incurred (naive-mpc-optim)

EMHASS can model a capacity / demand charge via the configuration parameter `capacity_cost_per_kw` (issue [#623](https://github.com/davidusb-geek/emhass/issues/623)): a cost on the single highest grid-import power over the optimization horizon, which pushes the optimizer to flatten import peaks.

In a real tariff the demand charge is assessed over the whole **billing period** (e.g. a month), not just the optimization horizon. Once you have already hit, say, 7 kW earlier in the month, there is no point spending battery energy now to keep this horizon's peak below 7 kW: that peak is already "locked in" and will be billed regardless. The optional runtime key `current_period_peak` tells EMHASS what that already-incurred peak is, so it only shaves the part of a new peak that is actually above it.

- `current_period_peak`: the peak grid import already reached this billing period, in **Watts** (not kW). Only effective when `capacity_cost_per_kw > 0`; ignored otherwise. The planned import peak is floored at this value, so the optimizer will not waste battery or deferrable flexibility trying to push the peak below a level already incurred.

It defaults to *unset* (`None`, equivalent to 0), in which case the full horizon peak is priced, identical to behaviour without this key. It is a one-sided floor (`peak_import >= current_period_peak`), so it never forces additional import and never makes the problem infeasible.

For example, with a capacity charge configured and 7 kW already incurred this month:

```json
{
  "prediction_horizon": 48,
  "soc_init": 0.30,
  "current_period_peak": 7000
}
```

As an HA `rest_command`, reading a kW sensor that tracks the monthly peak and converting to Watts:

```yaml
rest_command:
  naive_mpc_optim:
    url: http://localhost:5000/action/naive-mpc-optim
    method: post
    content_type: application/json
    payload: >
      {
        "prediction_horizon": 48,
        "soc_init": {{ states('sensor.battery_soc') | float / 100 }},
        "current_period_peak": {{ states('sensor.monthly_peak_grid_import_kw') | float(0) * 1000 }}
      }
```

```{warning}
`current_period_peak` is in **Watts**, not kilowatts: pass `7000`, not `7`, for a 7 kW peak (if your sensor is in kW, multiply by 1000 as shown above). It only has an effect when `capacity_cost_per_kw` is set above 0; on its own it does nothing. It is a floor, not a target: it never raises import or forces a new peak, it only stops the optimizer chasing a peak it cannot reduce. A negative, non-finite (NaN or infinity) or non-numeric value is ignored (treated as 0). Reset it at the start of each billing period (e.g. via a utility-meter cycle) so a stale high value does not suppress legitimate peak shaving in the new period.
```

### Restricting the demand charge to a tariff demand window (naive-mpc-optim)

Many demand tariffs do not charge the highest import of the whole day: they charge the highest import inside a **demand window** (e.g. 16:00-20:00 on business days only). Without a window, EMHASS prices the peak over every timestep of the horizon, so a deliberate off-window import (say, midday EV charging) pins `peak_import` and gets taxed for a cost that will never be billed - while shaving inside the actual window has zero marginal value (issues [#540](https://github.com/davidusb-geek/emhass/issues/540) / [#623](https://github.com/davidusb-geek/emhass/issues/623)).

The optional runtime key `capacity_charge_window` masks the peak constraint to the window:

- `capacity_charge_window`: a list of weights in `[0, 1]` of length `prediction_horizon`, aligned to the horizon timesteps like `load_cost_forecast`. The priced peak only "sees" timesteps where the mask is non-zero (`peak_import >= mask[t] * grid_import[t]`), so off-window import cannot inflate the demand charge. Only effective when `capacity_cost_per_kw > 0`; ignored otherwise.

EMHASS stays tariff-agnostic: the **caller owns the calendar** (business days, public holidays, seasons) and simply sends the right mask each cycle. Combined with `current_period_peak` this models a windowed monthly demand tariff faithfully: the window mask says *where* a peak can be set, the floor says *how high* the bar already is.

As an HA `rest_command` template fragment, marking 16:00-20:00 on weekdays for a 48-step / 30-minute horizon:

```yaml
      {% set nf = now().replace(minute=(now().minute // 30) * 30, second=0, microsecond=0) %}
      {% set ns = namespace(mask=[]) %}
      {% for i in range(48) %}
        {% set dt = nf + timedelta(minutes=30 * i) %}
        {% set ns.mask = ns.mask + [1 if (dt.weekday() < 5 and 16 <= dt.hour < 20) else 0] %}
      {% endfor %}
      "capacity_charge_window": {{ ns.mask | tojson }}
```

```{note}
The mask defaults to *unset* (`None`) = every timestep priced, identical to behaviour without this key. An all-zero mask (no window in this horizon) makes the demand term a constant - the plan is then identical to running without a capacity charge. An invalid mask (non-numeric entries, NaN/infinity, or shorter than the horizon) is ignored with a warning and the full horizon is priced; a longer mask is truncated; weights outside `[0, 1]` are clipped. Fractional weights are allowed and scale how much of that timestep's import the priced peak sees. If your billing period resets monthly, also zero the mask entries that fall in the *next* month and reset `current_period_peak` on the boundary, so the new period starts from a clean floor.
```

### Pricing the tariff's measurement interval, not the raw timestep (naive-mpc-optim)

Most demand tariffs bill on the **average** import power over a fixed clocked interval (e.g. 30 minutes), not a single optimisation timestep's instantaneous power (issue [#540](https://github.com/davidusb-geek/emhass/issues/540)). The config key `capacity_charge_interval_timesteps` (`optim_conf`, default `1`) sets how many native timesteps make up one tariff interval - at `1` the aggregation matrix is never constructed at all and the plain per-timestep peak epigraph from the sections above applies byte-for-byte unchanged; `N > 1` prices the average import power over each completed N-timestep interval instead, so a brief spike inside an otherwise-idle interval is weighted down instead of priced as if it lasted the whole interval. Like `capacity_cost_per_kw`, it is a normal structural `optim_conf` key (`associations.csv`) and follows the usual EMHASS config-override behaviour, including runtime override - there is no special restriction on changing it. Because it changes the shape of the capacity-charge constraint, changing its value changes the optimisation cache key and rebuilds the problem rather than reusing a warm-started one.

The runtime history mechanism below is scoped to `naive-mpc-optim` only, exactly like `current_period_peak` and `capacity_charge_window` above - no new plumbing was added for `dayahead-optim`/`perfect-optim` in this PR, and their existing capacity-charge behaviour is unaffected.

Because the MPC horizon rarely starts exactly on an interval boundary, pass the optional runtime key `capacity_charge_current_interval_history` so the first completed interval prices correctly:

- `capacity_charge_current_interval_history`: a list of the **average** positive grid-import power (Watts, oldest -> newest) for each native optimisation timestep already elapsed in the currently open tariff interval - the same per-timestep quantity `P_grid_pos` itself represents, not an arbitrary instantaneous sensor reading. If your source is an energy meter, derive it as `elapsed_energy_Wh / (optimization_time_step_minutes / 60)` per elapsed timestep. Its length (`0` to `capacity_charge_interval_timesteps - 1`) tells EMHASS how far the horizon start sits into the open interval. Only effective when `capacity_cost_per_kw > 0` and `capacity_charge_interval_timesteps > 1`; ignored otherwise.

```json
{
  "prediction_horizon": 24,
  "capacity_cost_per_kw": 8.0,
  "capacity_charge_interval_timesteps": 6,
  "capacity_charge_current_interval_history": [0, 0, 0, 6000]
}
```

EMHASS stays tariff-agnostic here too: the **caller owns clock/calendar alignment** (which native timestep the horizon starts on, within which tariff interval) - EMHASS does not compute timezone, season or wall-clock rules. `capacity_charge_window` still works alongside this, but note its semantics shift: for `N > 1` the mask is evaluated only at each completed interval's *endpoint*, which assumes the tariff's demand-window boundaries align with the tariff measurement-interval boundaries (i.e. the mask is effectively constant across any one N-timestep interval) - a window that is `1` throughout the tariff's demand hours still selects exactly the completed intervals that fall inside it.

A tariff interval still incomplete at the far end of the current MPC horizon is simply not priced by this solve - it has no completed-interval row yet. A later receding-horizon solve prices it once its completion becomes visible; this is inherent to only pricing *completed* intervals, not something to work around.

When `capacity_charge_interval_timesteps > 1`, `current_period_peak` (above) must be expressed in the same metric this epigraph now prices: the highest already-*completed* clocked tariff-interval **average** import power (Watts), not the maximum instantaneous or per-native-timestep import. The currently *open* interval belongs in `capacity_charge_current_interval_history`, not in `current_period_peak` - do not fold it into the incumbent floor until its tariff interval actually completes. `current_period_peak` itself is unchanged (same scalar Watts key); only which value is correct to pass changes when aggregation is on.

```{note}
The history defaults to *unset* (`None`) = an empty history, i.e. the horizon start is assumed to sit exactly on an interval boundary. An invalid history (non-numeric, NaN/infinity, negative values, or longer than `capacity_charge_interval_timesteps - 1`) is ignored with a warning, falling back to an empty history. At `capacity_charge_interval_timesteps = 1` the aggregation machinery is never constructed and the history is never inspected or validated.
```

### Passing forecast data

There is a complete dedicated section in the [Forecast](forecasts) section.

Specifically the [Passing your own forecast data](https://emhass.readthedocs.io/en/latest/forecasts.html#passing-your-own-forecast-data) section.


## WebSocket as a data source
EMHASS can retrieve historical sensor data directly from your Home Assistant instance using a WebSocket connection. By default, this connection fetches short-term statistics at a high resolution (5-minute intervals), which is ideal for real-time optimization and Model Predictive Control (MPC). To activate this option, simply set `use_websocket: true`.

If you are using the EMHASS-Add-on, this connection is handled automatically via the Home Assistant Supervisor API. If you are running EMHASS via Docker standalone or Python, ensure your Home Assistant URL and a Long-Lived Access Token are correctly provided in your configuration.

```{note} 

⚠️ **Important limitation regarding long-term historical data:**
When retrieving data via WebSocket, the availability of high-resolution (5-minute) data is strictly tied to your Home Assistant recorder's `purge_keep_days` setting (which defaults to 10 days).

If you configure EMHASS to retrieve a history window (`historic_days_to_retrieve`) that is *longer* than your `purge_keep_days`, Home Assistant will silently fall back to providing long-term statistics. Because these long-term statistics are aggregated to **one value per hour**, they lack the necessary granularity to use them for EMHASS, for example for accurate machine learning model fitting.

**Best Practices:**
- Use the **WebSocket** connection primarily for short-term history windows, MPC, and real-time optimization.
- For training machine learning models over longer historical horizons (e.g., months of data), use **InfluxDB** as your data source, as it retains high-resolution data without aggressive purging.

```


## InfluxDB as a data source
A new feature allows using **InfluxDB** as an alternative data source to the Home Assistant recorder database. This is beneficial for users who want to treat longer data retention periods for training machine learning models or to reduce the query load on their main Home Assistant instance.

When `use_influxdb: true` is set, EMHASS will fetch sensor data directly from your InfluxDB instance using the provided connection parameters. The `influxdb_username` and `influxdb_password` are treated as secrets.

If you are using the Influxdb official Home Assistant Add-on, then if you have set the integration via the `configuration.yaml` it will look like this: 

```yaml
  influxdb:
  host: xxxxxxxx-influxdb
  port: 8086
  database: homeassistant
  username: !secret influxdb_user
  password: !secret influxdb_password
  max_retries: 3
  default_measurement: state
  include:
    domains:
      - sensor
```

```{note} 

Here yo need to set your own values for: `xxxxxxxx-influxdb`, `influxdb_user` and `influxdb_password`
```

Then on the EMHASS configuration you need to set:

```json
{
  "influxdb_database": "homeassistant",
  "influxdb_host": "xxxxxxxx-influxdb",
  "influxdb_port": 8086,
  "influxdb_measurement": "W",
  "influxdb_retention_policy": "autogen",
  "influxdb_use_ssl": false,
  "influxdb_verify_ssl": false,
}
```

Finally, if using the Add-on, you need to fill both "influxdb_password" and "influxdb_username" in the Add-on **Configuration** pane.
If using the Docker standalone or legacy installation method, then you need to set these in the `secrets_emhass.yaml` file.

### Arithmetic expressions in the sensor list

When using InfluxDB, the entity id you would normally give to a sensor list parameter (such as `sensor_power_photovoltaics` or `sensor_power_load_no_var_loads`) can instead be an arithmetic expression that combines several InfluxDB time series. Wrap the expression in `{{ ... }}` and quote each entity id:

```text
{{'sensor.power_a' - 'sensor.power_b' * 1000}}
```

Each quoted entity is queried separately and the operation is applied element-wise on the values returned for the matching timestamps. The supported operators are `+`, `-`, `*`, `/`, `**`, `%`, unary `+`/`-` and numeric constants. Anything else (function calls, attribute access, comparisons, names that are not quoted entities, ...) is rejected, so an expression cannot run arbitrary code. An entity referenced more than once within expressions is only queried once. This is useful when the quantity EMHASS needs is not stored as a single sensor, for example a net power that is the difference of two meters, or a unit conversion.

```{note}

A few caveats apply because of how InfluxDB 1.x aggregates data:

- Each series is bucketed with `GROUP BY time() FILL(previous)`, so every entity in an expression lands on the same time grid and the arithmetic stays aligned. For the leading buckets of the requested window, EMHASS queries each entity slightly earlier (by one day) so the first values can be forward-filled from the most recent prior reading; a sensor whose previous reading is older than that still starts with a short gap, which the downstream regression absorbs.
- If a referenced sensor updates less often than once per day, that one-day lookback may not reach its previous reading and the expression starts with leading `NaN`. If that affects your setup, add the expression to `sensor_replace_zero` or `sensor_linear_interp` so the gap is filled before the optimization.
- An entity used both as a plain sensor and inside an expression is queried twice (the plain entry over the requested window, the expression entity over the padded window), so its plain column and its value inside an expression can differ at the leading buckets.
- InfluxDB 1.x cannot time-weight inside `GROUP BY`, so a source that updates more slowly than the optimization time step simply repeats its last value within each bucket. Keep this in mind when combining series that report at very different rates.
- If any entity referenced by an expression returns no data, the whole InfluxDB retrieval fails (a plain sensor that returns no data is still skipped, as before).
```


## Passing in secret parameters
Secret parameters are passed differently, depending on which method you choose. Alternative options are also present for passing secrets, if you are running EMHASS separately from Home Assistant. _(I.e. not via EMHASS-Add-on)_ 

### EMHASS with Docker or Python
Running EMHASS in Docker or Python by default retrieves all secret parameters via a passed `secrets_emhass.yaml` file. An example template has been provided under the name `secrets_emhass(example).yaml` on the EMHASS repo.

To pass the the secrets file:
- On Docker: *(via volume mount)*
```bash
Docker run ... -v ./secrets_emhass.yaml:/app/secrets_emhass.yaml ...
```
- On Python: *(optional: specify path as a argument)*
```bash
emhass ... --secrets=./secrets_emhass.yaml ...
```

#### Alternative Options
For users who are running EMHASS with methods other than EMHASS-Add-on, secret parameters can be passed with the use of environment variables. _(instead of `secrets_emhass.yaml`)_

Some environment variables include: `TIME_ZONE`, `LAT`, `LON`, `ALT`, `EMHASS_URL`, `EMHASS_KEY`

_Note: As of writing, EMHASS will override ENV secret parameters if the file is present._

For more information on passing arguments and environment variables using docker, have a look at some examples from [EMHASS Development](develop) section. 

### EMHASS-Add-on *(Emhass Add-on)*
By default, the `URL` and `KEY` parameters have been set to `empty`/blank in the Home Assistant configuration page for EMHASS addon. This results in EMHASS calling its Local `Supervisor API` to gain access. This is the easiest method, as there is no user input necessary.  

However, if you wish to receive/send sensor data to a different Home Assistant environment, set url and key values in the `hass_url` & `long_lived_token` hidden parameters on the Home Assistant EMHASS addon configuration page. *(E.g. http://localhost:8123/hassio/addon/emhass/config)*
-  `hass_url` example: `https://192.168.1.2:8123/`  
-  `long_lived_token` generated from the `Long-lived access tokens` section in your user profile settings
</br></br>

Secret Parameters such as: `solcast_api_key`, `solcast_rooftop_id` and `solar_forecast_kwp` _(used by their respective `weather_forecast_method` parameter values)_, can also be set via hidden parameters in the addon configuration page.

Secret Parameters such as: `time_zone`, `lon`, `lat` and `alt` are also automatically passed in via the Home Assistants `Supervisor API`. _(Values set in the Home Assistants config/general page)_  
_Note: Local currency could also be obtained via the Home Assistant environment, however as of writing, this functionality has not yet been developed._
