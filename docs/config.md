# Configuration file

In this section, we will explain all parameters used by EMHASS.

Note: For some context, the parameters bellow are grouped in configuration catagories. EMHASS will receive the and secrets parameters from `config.json` file and secret locations, then sort and format the parameters into their retrospective categories when migrating from the `config` dictionary to the `params` dictionary.
- The parameters needed to retrieve data from Home Assistant (`retrieve_hass_conf`)
- The parameters to define the optimization problem (`optim_conf`)
- The parameters used to model the system (`plant_conf`)

## Retrieve HASS data configuration

We will need to define these parameters to retrieve data from Home Assistant. There are no optional parameters. In the case of a list, an empty list is a valid entry.

- `optimization_time_step`: The time step to resample retrieved data from hass. This parameter is given in minutes. It should not be defined too low or you will run into memory problems when defining the Linear Programming optimization. Defaults to 30. 
- `historic_days_to_retrieve`: We will retrieve data from now to historic_days_to_retrieve days. Defaults to 2.
- `sensor_power_photovoltaics`: This is the name of the photovoltaic power-produced sensor in Watts from Home Assistant. For example: 'sensor.power_photovoltaics'. When using InfluxDB as the data source (`use_influxdb`), this can also be an arithmetic expression combining several InfluxDB time series (see the note below). This is especially useful for rescaling a series stored in the wrong unit of measure, for example converting a sensor logged in kW to the W that EMHASS expects.
- `sensor_power_load_no_var_loads`: The name of the household power consumption sensor in Watts from Home Assistant. The deferrable loads that we will want to include in the optimization problem should be subtracted from this sensor in HASS. For example: 'sensor.power_load_no_var_loads'. As with `sensor_power_photovoltaics`, when using InfluxDB this can be an arithmetic expression over several time series (see the note below). The subtraction can then be performed on the fly over the whole historical series, avoiding the need to create dedicated helper sensors in Home Assistant.
- `sensor_power_battery`: The name of the signed AC-side battery power sensor in Watts from Home Assistant, for example 'sensor.power_battery'. Only used by battery self-identification (`set_use_battery_identification`); either sign convention works as it is auto-detected. This sensor is only retrieved when battery self-identification is enabled.
- `sensor_battery_state_of_charge`: The name of the measured battery state of charge sensor in percent from Home Assistant, for example 'sensor.battery_state_of_charge'. Only used by battery self-identification (`set_use_battery_identification`), and only retrieved when it is enabled.
- `load_negative`: Set this parameter to True if the retrieved load variable is negative by convention. Defaults to False.
- `set_zero_min`: Set this parameter to True to give a special treatment for a minimum value saturation to zero for power consumption data. Values below zero are replaced by nans. Defaults to True.
- `var_replace_zero`: The list of retrieved variables that we would want to replace nans (if they exist) with zeros. For example:
	- 'sensor.power_photovoltaics'
- `sensor_linear_interp`: The list of retrieved variables that we would want to interpolate nans values using linear interpolation. For example:
	- 'sensor.power_photovoltaics'
	- 'sensor.power_load_no_var_loads'
- `method_ts_round`: Set the method for timestamp rounding, options are: first, last and nearest.
- `continual_publish`: set to True to save entities to .json after an optimization run. Then automatically republish the saved entities *(with updated current state value)* every `optimization_time_step` minutes. *entity data saved to data_path/entities.*
- `use_websocket`: Enable WebSocket as a data source instead of the Home Assistant API. This allows for longer historical data retention and better performance for machine learning models.
- `use_influxdb`: Enable InfluxDB (version 1.x) as a data source instead of the Home Assistant API. This allows for longer historical data retention and better performance for machine learning models. InfluxDB v2 is not currently supported.
- `influxdb_host`: The IP address or hostname of your InfluxDB instance. Defaults to `localhost`.
- `influxdb_port`: The port number for your InfluxDB instance. Defaults to 8086.
- `influxdb_database`: The name of the InfluxDB database containing your Home Assistant data. Defaults to `homeassistant`.
- `influxdb_measurement`: The measurement name where your sensor data is stored. Defaults to `W` for the Home Assistant integration.
- `influxdb_retention_policy`: The retention policy to use for InfluxDB queries. Defaults to `autogen`.

```{note}
When InfluxDB is enabled, any sensor list parameter (such as `sensor_power_photovoltaics` or `sensor_power_load_no_var_loads`) accepts an arithmetic expression over several InfluxDB time series instead of a single entity id, using the `{{ ... }}` syntax, for example `{{'sensor.power_a' - 'sensor.power_b' * 1000}}`. This is handy whenever the quantity EMHASS needs is not stored as a single sensor, for example:

- a net power that is the difference of two meters (grid import minus export, total consumption minus a sub-metered load, ...);
- a unit conversion, such as rescaling a series logged in kW to the W that EMHASS expects;
- combining several partial sources into one total (e.g. summing two PV strings or several circuits).

Because the operation is applied on the fly over the whole retrieved history, you no longer need to create dedicated template/helper sensors in Home Assistant just to feed EMHASS. See [the InfluxDB section in the passing data documentation](passing_data.md#arithmetic-expressions-in-the-sensor-list) for the full syntax, the supported operators and the caveats.
```

A second part of this section is given by some privacy-sensitive parameters that should be included as:
- The list of secret parameters filled in the form on the Add-on **Configuration** pane if using the Add-on installation method.
- A `secrets_emhass.yaml` file alongside the `config_emhass.yaml` file if using the Docker standalone or legacy installation method.

The **secrets** parameters are:

- `hass_url`: The URL to your Home Assistant instance. For example: https://myhass.duckdns.org/
- `long_lived_token`: A Long-Lived Access Token from the Lovelace profile page.
- `time_zone`: The time zone of your system. For example: Europe/Paris.
- `Latitude`: The latitude. For example: 45.0.
- `Longitude`: The longitude. For example: 6.0
- `alt`: The altitude in meters. For example: 100.0
- `solcast_api_key`: The Solcast API key (weather_forecast_method=solcast)
- `solcast_rooftop_id`: The Solcast rooftop ID (weather_forecast_method=solcast)
- `solar_forecast_kwp`: The PV peak installed power in kW used for the 'solar.forecast' API call (weather_forecast_method=solar.forecast)
- `influxdb_username`: Username for authenticating with InfluxDB. Leave empty if no authentication is required.
- `influxdb_password`: Password for authenticating with InfluxDB. Leave empty if no authentication is required.

## Optimization configuration parameters

These are the parameters needed to properly define the optimization problem.

- `set_use_pv`: Set to True if we should consider an solar PV system. Defaults to False.
- `set_use_battery`: Set to True if we should consider an energy storage device such as a Li-Ion battery. Defaults to False.
- `delta_forecast_daily`: The number of days for forecasted data. Defaults to 1.
- `number_of_deferrable_loads`: Define the number of deferrable loads to consider. Defaults to 2.
- `nominal_power_of_deferrable_loads`: The nominal power for each deferrable load in Watts. This is a list with a number of elements consistent with the number of deferrable loads defined before. A load can also be given a list of powers in place of a single value to define a **sequence load**: a fixed power profile that the optimizer runs exactly once, contiguously, choosing only the start time (for example a washing-machine programme). Mind the nesting: with `number_of_deferrable_loads: 1`, `[[1000, 2000, 1500]]` is one sequence load with the profile 1000 W, 2000 W, 1500 W, whereas `[1000, 2000, 1500]` with `number_of_deferrable_loads: 3` is three separate single-power loads. A sequence load's runtime is fixed by the length of its sequence, so `operating_hours_of_each_deferrable_load` does not apply to it and that load's value (including 0) is ignored. For example:
	- 3000
	- 750
- `operating_hours_of_each_deferrable_load`: The total number of hours that each deferrable load should operate. Fractional values are accepted (e.g. `4.5`); the optimizer schedules the exact `nominal_power × hours` of energy. For control finer than the optimization timestep you can also pass `operating_timesteps_of_each_deferrable_load`. For example:
	- 5
	- 8
- `start_timesteps_of_each_deferrable_load`: A list of integers defining the **earliest time step index** from which each deferrable load is allowed to start consuming power.
	- **Value type:** Integer (Index of the time step, *not* the hour).
	- **Default/Disable:** If a value of **0** (or negative) is provided, the constraint is disabled, and the load is allowed to start immediately from the beginning of the optimization window (Index 0).
	- Example: With a `30 min` (0.5h) time step:
		- `0`: Can start immediately (00:00).
		- `4`: Can start after 2 hours (02:00).

```{note} 

Since `start_timesteps` are relative indexes starting from 0 (the moment the optimization begins), they are heavily dependent on your optimization launch time. So the index is relative to the start of the optimization window (Launch Time = Index 0).
Example: 
- Launch at 7:00 AM, allowed to start at 9:00 AM (2h delay).
- Time step 30 min.
- Value = 2 hours / 0.5 = 4.
```

- `end_timesteps_of_each_deferrable_load`: A list of integers defining the **deadline time step index** by which each deferrable load must stop consuming power. The load is strictly forbidden from operating at or after this time step.
	- **Value type:** Integer (Index of the time step).
	- **Default/Disable:** If a value of **0** (or negative) is provided, the constraint is disabled, and the load is allowed to operate up until the very end of the prediction horizon (e.g., the full 24h window).
	- Example: With a `30 min` (0.5h) time step:
		- `0`: Can run anytime until the end of the horizon.
		- `21`: Must finish strictly before timestep 21 (i.e., must stop by 10.5 hours / 10:30 AM).

```{note} 

Since `end_timesteps` are relative indexes starting from 0 (the moment the optimization begins), they are heavily dependent on your optimization launch time. So the index is relative to the start of the optimization window.
Example: 
- Launch at 7:00 AM, must finish by 6:00 PM (18:00).
- Duration = 11 hours.
- Time step 30 min.
- Value = 11 hours / 0.5 = 22.
```

- `treat_deferrable_load_as_semi_cont`: Define if we should treat each deferrable load as a semi-continuous variable. Semi-continuous variables (`True`) are variables that must take a value that can be either their maximum or minimum/zero (for example On = Maximum load, Off = 0 W). Non semi-continuous (which means continuous) variables (`False`) can take any values between their maximum and minimum. For example:
	- True
	- True
- `set_deferrable_load_single_constant`: Define if we should set each deferrable load as a constant fixed value variable with just one startup for each optimization task. For example:
	- False
	- False
- `set_deferrable_startup_penalty`: Set to a list of floats. For each deferrable load with a penalty `P`, each time the deferrable load turns on will incur an additional cost of `P * nominal_power_of_deferrable_loads * cost_of_electricity` at that time.
- `def_minimum_on_time`: Per-load minimum number of consecutive optimization timesteps a load must stay ON once started (short-cycle / min-up-time protection). One integer per deferrable load. Set to `0` (default) to disable for that load -- **default-off, exact no-op**. Example: `[3, 0]` requires load 0 to stay on for at least 3 consecutive timesteps once started; load 1 has no constraint.

  Unit: timesteps. Convert to minutes: `N x optimization_time_step`. With the default 30-minute step, `3 timesteps = 90 minutes`.

  **Primary use case:** `treat_deferrable_load_as_semi_cont: true` loads (heat pump, AC, pump) where `bin2 = 1` forces the full nominal power -- min-on-time prevents short-cycling that wears the compressor. For loads without `treat_deferrable_load_as_semi_cont`, the constraint holds the ON/OFF binary but does not force a specific power level.

  **Not applied to `set_deferrable_load_single_constant` loads:** those already run as one continuous block, so a separate minimum on-time is redundant and is ignored for them.

  **Self-protecting against window end:** if a start cannot complete the minimum ON window before the load's `end_timesteps_of_each_deferrable_load`, the solver simply does not take that start -- the problem stays Optimal. No forced infeasibility.

  **Composes with `set_deferrable_max_startups`:** the two constraints are independent (count vs window) and coexist cleanly. A pathological combination of low max-startups, long min-on, and a short operating window can over-constrain a load; if you use both, ensure `max_startups x min_on_time <= available_timesteps`.

  **Initial-condition remainder:** to carry the remaining on-time across MPC ticks when the load is already running, also pass `def_current_on_timesteps` at runtime (see Passing data at runtime).
- `def_minimum_off_time`: Per-load minimum number of consecutive optimization timesteps a load must stay OFF once it stops (short-cycle / min-down-time protection). One integer per deferrable load. Set to `0` (default) to disable for that load -- **default-off, exact no-op**. Example: `[3, 0]` requires load 0 to remain off for at least 3 consecutive timesteps after stopping; load 1 has no constraint.

  Unit: timesteps. Convert to minutes: `N x optimization_time_step`. With the default 30-minute step, `3 timesteps = 90 minutes`.

  **Primary use case:** `treat_deferrable_load_as_semi_cont: true` loads (heat pump, AC, compressor) where rapid ON-OFF-ON cycling causes compressor wear or violates manufacturer limits. Prevents the optimizer from restarting a load immediately after stopping it.

  **Not applied to `set_deferrable_load_single_constant` loads:** those already run as one continuous block, so a separate minimum off-time constraint is not applicable and is ignored for them.

  **Self-protecting against horizon end:** if the load turns off near the end of the horizon and the minimum OFF window extends beyond the horizon, the constraint applies only within the remaining timesteps -- the problem stays Optimal.

  **Composes with `def_minimum_on_time` and `set_deferrable_max_startups`:** all three constraints are independent and coexist cleanly. A long minimum off-time combined with a tight operating window can leave too few timesteps for a load to deliver its required energy; if you rely on both, make sure the minimum off-time still leaves enough operating timesteps for the load to run.

  **Initial-condition remainder:** to carry the remaining off-time across MPC ticks when the load is already stopped, also pass `def_current_off_timesteps` at runtime (see Passing data at runtime).
- `def_current_power`: Per-load actual power (watts) each deferrable load is drawing right now at the start of the optimization horizon. **Default `[0, 0]` (or absent key) is an exact no-op** -- today's behaviour is preserved. One non-negative float per deferrable load, in the same order as `nominal_power_of_deferrable_loads`. Supply the real sensor reading in watts.

  Unit: W (watts). Runtime-overridable on every MPC tick without rebuilding the solver cache.

  When `def_current_power[k] > 0`, the optimizer applies three coordinated effects:

  1. **Pin:** fixes `P_deferrable[k][t=0]` to the supplied power, correcting the t=0 power balance when the load is excluded from the main load sensor.
  2. **Force ON:** forces the load's binary `bin2[0] = 1` at t=0, preventing the optimizer from immediately scheduling a stop.
  3. **Suppress phantom startup:** sets the internal initial-state to ON so no startup penalty fires at t=0 (equivalent to also passing `def_current_state[k]=true`).

  This is a separate runtime input rather than an extension of `def_current_state`, so the binary on/off decision stays strict 0/1 and a fractional watt value can never weaken it.

  **Semi-continuous loads (`treat_deferrable_load_as_semi_cont: true`):** because semi-continuous power is strictly `nominal * bin` (cannot be an arbitrary fraction of nominal), the power pin is omitted for these loads. Supplying any non-zero value still triggers the force-ON and phantom-startup suppression, so the load is held ON at its full nominal power at t=0. The supplied wattage acts as an on/off signal only.

  **Loads with minimum power (`minimum_power_of_deferrable_loads > 0`):** the supplied value should be at or above the configured minimum to avoid an infeasible t=0 power balance. Supplying a value below the minimum and above zero will produce an infeasible solve (the pin forces `p = supplied` while the min-power constraint forces `p >= min_power`).

  **Supplied value exceeds nominal:** the existing window-mask upper bound (`p <= nominal * mask`) prevents values above nominal from being pinned; the solve will be infeasible. Supply values within `[min_power, nominal]`.

  **Single-constant, sequence and thermal loads** ignore `def_current_power` (it is a no-op for those load types). A single-constant load runs as one fixed block, so its currently-running state is handled by `def_current_state` (which pins the remaining required timesteps); pinning a below-nominal power there would fight the required-energy target. Use `def_current_state` for a currently-running single-constant load.

  **Typical MPC use:** read the load's power sensor at the start of each optimization tick and pass the reading in watts via the runtime API so the optimizer knows the load is running and at what level.
- `def_current_operating_timesteps`: Per-load integer list: how many operating timesteps each must-run deferrable load has already completed today at the start of this optimization horizon. The optimizer schedules only the REMAINDER: `required_timesteps` and `target_energy` are each decremented by the elapsed amount, clamped at 0. **Default `null` (absent key) or `[0, 0]` is an exact no-op** -- today's plan is unchanged. One non-negative integer per deferrable load, in the same order as `nominal_power_of_deferrable_loads`.

  Unit: timesteps. Convert from time: `elapsed_hours / optimization_time_step`. With the default 30-minute step, 2 elapsed hours = 4 timesteps.

  Applies to **both standard and single_constant must-run loads** (unlike `def_current_on_timesteps` which is gated on `def_minimum_on_time > 0` and excluded for single_constant loads). If a load has no required operating hours configured (`operating_hours_of_each_deferrable_load[k] == 0`), the elapsed value is ignored.

  When the elapsed count equals or exceeds the total required timesteps, the remainder clamps to 0 and the constraint is fully relaxed -- the load is no longer forced to run in this tick, and the solve stays Optimal.

  **Interaction with `def_current_state`:** if you also pass `def_current_state[k]=true` for a single_constant load, the optimizer uses both signals: the currently-running pin (block A) uses the *decremented* `required_timesteps`, so the running-block anchor also shrinks automatically.

  **Daily reset boundary:** EMHASS does not track wall-clock day rollover. The caller (supervisor / automation) is responsible for resetting the elapsed count to 0 at the start of each day, exactly as with `def_current_on_timesteps` and `def_current_power`.

  **Typical MPC use:** at each optimization tick, read the load's run-time counter (e.g. from a Home Assistant energy meter or runtime helper) and pass the elapsed timesteps so the optimizer schedules only the outstanding run.
- `deferrable_load_max_cost`: Make a deferrable load *optional* by capping how much it may cost to run. This is a list of floats, one value per deferrable load, in the same currency units as your load cost forecast. The default `0` keeps a load **mandatory**: the optimizer must deliver its full `operating_hours_of_each_deferrable_load` of energy within the horizon, as it always has. A value above `0` makes the load **optional**: the optimizer schedules it only when its complete run can be done for less than that amount (the cost of the energy it consumes plus any `set_deferrable_startup_penalty`), otherwise the load is left off for the whole horizon. This suits a "nice to have" load that is only worth running when energy is cheap enough, for example a heat pump boosting hot water above its normal setpoint, or an immersion heater that should only soak up surplus when prices are very low. The cap is an all-or-nothing budget for the load's entire run, not a per-timestep price limit. It applies to every deferrable load type (standard, semi-continuous, single-constant and sequence loads) except thermal loads configured with a `thermal_config` or `thermal_battery`, which follow their own temperature targets instead. For example:
	```json
	"deferrable_load_max_cost": [0, 0.65]
	```
	Here `deferrable0` stays mandatory and `deferrable1` is scheduled only if its full run costs less than 0.65. Defaults to `0` for every load (all loads mandatory).
- `deferrable_load_groups`: Define groups of deferrable loads that share a physical actuator (e.g. a heat pump serving both hot water and underfloor heating). Each group can enforce a shared power budget, mutual exclusion, or both. This is a list of group objects, each with the following fields:
	- `names`: List of deferrable load names in the group (e.g. `["deferrable0", "deferrable1"]`).
	- `max_power` *(optional when `mutual_exclusion` is `true`)*: Maximum combined power in Watts for all loads in the group at any timestep. Required when `mutual_exclusion` is `false`.
	- `mutual_exclusion` *(optional, defaults to `false`)*: When `true`, only one load in the group may be active at any timestep. Members may be a mix of semi-continuous and non-semi-continuous loads — the optimizer reuses `p_def_bin2` for semi-continuous members and creates an anonymous activity binary linked to `p_deferrable` for non-semi-continuous ones.

	A load cannot belong to multiple groups. Examples:
	```json
	"deferrable_load_groups": [
	  {"names": ["deferrable0", "deferrable1"], "max_power": 2500}
	]
	```
	```json
	"deferrable_load_groups": [
	  {"names": ["deferrable0", "deferrable1"], "mutual_exclusion": true}
	]
	```
	```json
	"deferrable_load_groups": [
	  {"names": ["deferrable0", "deferrable1"], "max_power": 2500, "mutual_exclusion": true}
	]
	```
	Defaults to an empty list (no groups).
- `weather_forecast_method`: This will define the weather forecast method that will be used. The options are `open-meteo` to use the weather forecast API proposed by [Open-Meteo](https://open-meteo.com/), `solcast` to use the [Solcast](https://solcast.com/) solar forecast service, `solar.forecast` to use the free public [Solar.Forecast](https://forecast.solar/) account and finally the `csv` to load a CSV file. When loading a CSV file this will be directly considered as the PV power forecast in Watts. The default CSV file path that will be used is `/data/data_weather_forecast.csv`. This method is useful to load and use other external forecasting service data in EMHASS. Defaults to `open-meteo` method.
- `weather_forecast_pv_quantile_bias`: A blend factor that biases the Solcast PV forecast toward its conservative P10 (low) estimate. Type: float. Valid range: `[0, 1]`. Default: `0.0` (pure P50, unchanged behaviour). A value of `1.0` uses the P10 estimate exclusively. Intermediate values blend linearly: `estimate = bias * P10 + (1 - bias) * P50`. Higher values produce more defensive plans — the optimizer is told less solar will be available, so it tends to hold more battery reserve when solar is uncertain. Only effective when `weather_forecast_method` is set to `solcast`; has no effect for other forecast methods.
- `load_forecast_method`: The load forecast method that will be used. The options are `typical` which uses basic statistics and a year long load power data grouped by the current day-of-the-week of the current month, `naive` also called persistence that assumes that the forecast for a future period will be equal to the observed values in a past period, `mlforecaster` that uses regression models considering auto-regression lags as features and finally the `csv` to load a CSV file. When loading a CSV file this will be directly considered as the PV power forecast in Watts. The default CSV file path that will be used is `/data/data_weather_forecast.csv`. This method is useful to load and use other external forecasting service data in EMHASS. Defaults to `typical`.
```{note} 

For more information on these methods check the dedicated [Forecast section](forecasts)
```
- `load_cost_forecast_method`: Define the method that will be used for load cost forecast. The options are `hp_hc_periods` for peak and non-peak hours contracts and `csv` to load custom cost from CSV file. The default CSV file path that will be used is `/data/data_load_cost_forecast.csv`.
The following parameters and definitions are only needed if `load_cost_forecast_method=hp_hc_periods`:
	- `load_peak_hour_periods`: Define a list of peak hour periods for load consumption from the grid. This is useful if you have a contract with peak and non-peak hours. For example for two peak hour periods: 
		- period_hp_1:
			- start: '02:54'
			- end: '15:24'
		- period_hp_2:
			- start: '17:24'
			- end: '20:24'
	- `load_peak_hours_cost`: The cost of the electrical energy from the grid during peak hours in currency/kWh. Defaults to 0.1907.
	- `load_offpeak_hours_cost`: The cost of the electrical energy from the grid during non-peak hours in currency/kWh. Defaults to 0.1419.
- `production_price_forecast_method`: Define the method that will be used for PV power production price forecast. This is the price that is paid by the utility for energy injected into the grid. The options are `constant` for a constant fixed value or `csv` to load custom price forecasts from a CSV file. The default CSV file path that will be used is `/data/data_prod_price_forecast.csv`.
```{note} 

For all the forecast methods (`weather`, `load_power`, `load_cost` and `production_price`) it is also possible to pass the data from external services using a list of values or a dictionary with timestamps. For more information check the dedicated [Passing your own forecast data](https://emhass.readthedocs.io/en/latest/forecasts.html#passing-your-own-forecast-data) section.
```
- `photovoltaic_production_sell_price`: The paid price for energy injected to the grid from excess PV production in currency/kWh. Defaults to 0.065. This parameter is only needed if production_price_forecast_method='constant'.
- `set_total_pv_sell`: Set this parameter to true to consider that all the PV power produced is injected to the grid. No direct self-consumption. The default is false, for a system with direct self-consumption.
- `set_use_adjusted_pv`: Set to True to enable machine learning-based PV forecast adjustment. This uses historical data to train a regression model that corrects PV forecasts based on local conditions. Defaults to False. See the [Forecasts](https://emhass.readthedocs.io/en/latest/forecasts.html#adjusting-pv-forecasts-using-machine-learning) section for more details.
- `adjusted_pv_regression_model`: The regression model to use for PV forecast adjustment. See `REGRESSION_METHODS` in `machine_learning_regressor.py` for the authoritative list. Currently available: 'LinearRegression', 'RidgeRegression', 'LassoRegression' (default), 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'SVR', 'RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'MLPRegressor'. Only used when `set_use_adjusted_pv` is True.
- `adjusted_pv_solar_elevation_threshold`: The solar elevation threshold in degrees below which the adjusted PV forecast is set to zero. This prevents negative or unrealistic values during low sun angles. Defaults to 10.
- `adjusted_pv_model_max_age`: Maximum age in hours before the adjusted PV regression model is re-fitted. If the saved model is older than this value, a new model will be trained using fresh historical data. Set to 0 to force re-fitting on every call. Defaults to 24 hours (1 day). This caching mechanism significantly reduces API calls to Home Assistant and speeds up optimization runs.
- `set_use_battery_identification`: Set to True to learn the battery usable capacity and round-trip efficiency from Home Assistant history instead of relying only on the hand-entered `battery_nominal_energy_capacity`, `battery_charge_efficiency` and `battery_discharge_efficiency`. Opt-in and default False. Requires `set_use_battery` to be True and the `sensor_power_battery` and `sensor_battery_state_of_charge` sensors. In this first version the feature is advisory only: it never changes the battery values the optimizer uses. See the note below.
- `battery_identification_trust_tier`: What battery self-identification does with its estimate. `observe` (default) writes the estimate to a JSON file under the data path and to the log only. `suggest` additionally publishes two read-only Home Assistant sensors (`sensor.battery_identified_capacity` and `sensor.battery_identified_round_trip_efficiency`, each carrying the confidence interval and sample counts as attributes) and logs a recommendation. Neither tier changes the configured battery values used by the optimizer.
- `battery_identification_model_max_age`: Maximum age in hours before the battery identification estimate is re-fitted from fresh Home Assistant history. Set to 0 to force re-fitting on every call. Defaults to 24 hours (1 day). Like the adjusted-PV cache, this avoids re-pulling history on every run.

```{note}
Battery self-identification (`set_use_battery_identification`) is an opt-in, default-off feature that learns two of the battery constants the optimizer otherwise takes on trust: usable capacity and round-trip efficiency. It follows the same learn-from-history pattern as `set_use_adjusted_pv`. From a single AC-side power meter plus the reported state of charge it can identify the usable capacity (in the same reported-SoC units the optimizer already uses) and the lumped round-trip efficiency, but it cannot split that efficiency into separate charge and discharge figures, so it sets both to `sqrt(round_trip_efficiency)` and says so. It is data-hungry: it needs weeks of signed power and SoC with enough deep cycles, and if the data is too shallow or the fit fails a sanity check it publishes nothing and keeps your configured values. In this first version it is advisory only (`observe`/`suggest`); it never overwrites the values the optimizer uses. Power-dependent efficiency, standby draw, and the charge/discharge split are known limitations left for a later version.
```
- `num_threads`: Set the number of threads to pass to LP solvers that support specifying a number of threads. Defaults to 0 (auto-detect).
- `lp_solver_timeout`: Maximum time in seconds the solver is allowed to run before stopping. Defaults to 45.
- `lp_solver_mip_rel_gap`: MIP (Mixed-Integer Programming) relative gap tolerance. For problems with binary variables (semi-continuous loads, single-constant loads, etc.), the solver will stop when it finds a solution within this percentage of the optimal. A value of 0.05 (5%) means the solver stops when the solution is guaranteed to be within 5% of optimal. Higher values solve faster with minimal quality impact. Defaults to 0.01 (within 1% of optimal); this keeps deep-horizon problems from timing out before any plan is published. Set to 0 for exact optimal (this was the previous default).
- `set_nocharge_from_grid`: Set this to true if you want to forbid charging the battery from the grid. The battery will only be charged from excess PV.
- `set_nodischarge_to_grid`: Set to true to forbid discharging battery power to the grid. The constraint depends on inverter topology: hybrid inverters (`inverter_is_hybrid` true) block battery discharge whenever PV is exporting; AC-coupled systems instead bound grid export to the available PV (net of any curtailment when `compute_curtailment` is true), which prevents battery energy from reaching the grid while still allowing the battery to supply local load during export. The AC-coupled form fixes the infeasibility reported in issue #936.
- `set_battery_first_priority`: Set this to true to forbid importing from the grid while the battery is still above its minimum SoC, so stored energy is drained before any grid import. This is mainly useful on a flat (non time-of-use) tariff, where the solver would otherwise be free to interleave grid import with discharge while the battery is still full. Default is false. Note that it is a hard constraint: it can make the optimization infeasible in a timestep where the load minus PV exceeds the battery's maximum discharge power, so only enable it when your battery discharge power can cover your load.
- `set_battery_dynamic`: Set a power dynamic limiting condition to the battery power. This is an additional constraint on the battery dynamic in power per unit of time, which allows you to set a percentage of the battery's nominal full power as the maximum power allowed for (dis)charge.
- `battery_dynamic_max`: The maximum positive (for discharge) battery power dynamic. This is the allowed power variation (in percentage) of battery maximum power per unit of time.
- `battery_dynamic_min`: The maximum negative (for charge) battery power dynamic. This is the allowed power variation (in percentage) of battery maximum power per unit of time.
- `weight_battery_discharge`: An additional weight (currency/ kWh) applied in the cost function to battery usage for discharging. Defaults to 0.00
- `weight_battery_charge`: An additional weight (currency/ kWh) applied in the cost function to battery usage for charging. Defaults to 0.00
- `battery_soc_deficit_threshold`: The state of charge below which a deficit penalty applies, for example 0.40 for 40%. Used together with `battery_soc_deficit_cost`. Defaults to 0.40.
- `battery_soc_deficit_cost`: A virtual cost (currency/kWh/h) applied for each kWh the battery sits below `battery_soc_deficit_threshold`, per hour. This discourages draining the battery too low. Defaults to 0.00 (disabled).
- `battery_soc_surplus_threshold`: The state of charge above which a surplus penalty applies, for example 0.90 for 90%. Used together with `battery_soc_surplus_cost`. Defaults to 0.90.
- `battery_soc_surplus_cost`: A virtual cost (currency/kWh/h) applied for each kWh the battery sits above `battery_soc_surplus_threshold`, per hour. This discourages dwelling near full charge, so the battery fills more gradually into expected solar peaks and spends less time at a high state of charge. It is the mirror of `battery_soc_deficit_cost`. Defaults to 0.00 (disabled).
- `capacity_cost_per_kw`: A cost in currency per kW applied to the peak grid import power over the optimization horizon, for tariffs that include a capacity or demand charge. Defaults to 0, which disables the feature and leaves the plan unchanged. With a positive value the optimizer adds a single scalar term that prices the highest import power it plans, so it will flatten that peak where doing so is cheaper than the resulting change in energy cost. The peak is charged once because it is a power charge, not energy, so the term is not scaled by the timestep. Set it to the marginal value to you of shaving 1 kW off your peak. Note this is a power-based capacity charge and is separate from `load_peak_hours_cost`, which is a time-of-use energy price.

## System configuration parameters

These are the technical parameters of the energy system of the household.

- `maximum_power_from_grid`: The maximum power that can be supplied by the utility grid in Watts (consumption). Defaults to 9000.
- `maximum_power_to_grid`: The maximum power that can be supplied to the utility grid in Watts (injection). Defaults to 9000.

We will define the technical parameters of the PV installation. For the modeling task we rely on the PVLib Python package. For more information see: [https://pvlib-python.readthedocs.io/en/stable/](https://pvlib-python.readthedocs.io/en/stable/)
A dedicated web app will help you search for your correct PV module and inverter names: [https://emhass-pvlib-database.streamlit.app/](https://emhass-pvlib-database.streamlit.app/)
If your specific model is not found in these lists then solution (1) is to pick another model as close as possible as yours in terms of the nominal power.
Solution (2) would be to use SolCast and pass that data directly to emhass as a list of values from a template. Take a look at this example here: [https://emhass.readthedocs.io/en/latest/forecasts.html#example-using-solcast-forecast-amber-prices](https://emhass.readthedocs.io/en/latest/forecasts.html#example-using-solcast-forecast-amber-prices)

- `pv_module_model`: The PV module model. You can provide this value in two ways:
	- **By Name (Recommended for precision):** The exact model name from the CEC database (e.g., `'CSUN_Eurasia_Energy_Systems_Industry_and_Trade_CSUN295_60M'`). Remember to replace special characters with `_`.
	- **By Power (New):** An integer or float representing the nominal power of the panel in Watts (STC). EMHASS will automatically find the closest matching panel in the database (e.g., `300` or `'300'`).
	- *Note:* This parameter can be a list of items to enable the simulation of mixed orientation systems (e.g., one east-facing array and one west-facing array).

- **`pv_inverter_model`**: The PV inverter model. You can provide this value in two ways:
	- **By Name (Recommended for precision):** The exact model name from the CEC database (e.g., `'Fronius_International_GmbH__Fronius_Primo_5_0_1_208_240__240V_'`). Remember to replace special characters with `_`.
	- **By Power (New):** An integer or float representing the nominal AC power of the inverter in Watts. EMHASS will automatically find the closest matching inverter in the database (e.g., `5000` or `'5000'`).
	- *Note:* This parameter can be a list of items to enable the simulation of mixed orientation systems.

Then the additional technical parameters:

- `surface_tilt`: The tilt angle of your solar panels. Defaults to 30. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example, one east-facing array (azimuth=90) and one west-facing array (azimuth=270). 
- `surface_azimuth`: The azimuth of your PV installation. Defaults to 205. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example, one east-facing array (azimuth=90) and one west-facing array (azimuth=270). 
- `modules_per_string`: The number of modules per string. Defaults to 16. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example, one east-facing array (azimuth=90) and one west-facing array (azimuth=270). 
- `strings_per_inverter`: The number of used strings per inverter. Defaults to 1. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example one east-facing array (azimuth=90) and one west-facing array (azimuth=270).
- `inverter_is_hybrid`: Set to True to consider that the installation inverter is hybrid for PV and batteries (Default False).
- `compute_curtailment`: Set to True to compute a special PV curtailment variable (Default False). When enabled, curtailment that is cost-equivalent is scheduled as late as possible in the optimization horizon (issue #342).
- `inverter_stress_cost`: The virtual penalty cost (in currency/kWh) applied if the inverter runs at its maximum nominal power (Recommended: 0.05 - 0.20).
- `inverter_stress_segments`: The number of linear segments used to approximate the quadratic curve. Higher values are more accurate but increase computation slightly (Recommended: 10).

If your system has a battery (set_use_battery=True), then you should define the following parameters:

- `battery_discharge_power_max`: The maximum discharge power in Watts. Defaults to 1000.
- `battery_charge_power_max`: The maximum charge power in Watts. Defaults to 1000.
- `battery_discharge_efficiency`: The discharge efficiency. Defaults to 0.95.
- `battery_charge_efficiency`: The charge efficiency. Defaults to 0.95.
- `battery_nominal_energy_capacity`: The total capacity of the battery stack in Wh. Defaults to 5000.
- `battery_minimum_state_of_charge`: The minimum allowable battery state of charge. Defaults to 0.3.
- `battery_maximum_state_of_charge`: The maximum allowable battery state of charge. Defaults to 0.9.
- `battery_target_state_of_charge`: The desired battery state of charge at the end of each optimization cycle. Defaults to 0.6.