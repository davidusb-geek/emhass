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
- `sensor_power_photovoltaics`: This is the name of the photovoltaic power-produced sensor in Watts from Home Assistant. For example: 'sensor.power_photovoltaics'.
- `sensor_power_load_no_var_loads`: The name of the household power consumption sensor in Watts from Home Assistant. The deferrable loads that we will want to include in the optimization problem should be subtracted from this sensor in HASS. For example: 'sensor.power_load_no_var_loads'
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

A second part of this section is given by some privacy-sensitive parameters that should be included as:
- The list of secret parameters filled in the form on the Add-on **Configuration** pane if using the Add-on installation method.
- A `secrets_emhass.yaml` file alongside the `config_emhass.yaml` file if using the Docker standalone or legacy installation method.

The **secrets** parameters are:

- `hass_url`: The URL to your Home Assistant instance. For example: https://myhass.duckdns.org/
- `long_lived_token`: A Long-Lived Access Token from the Lovelace profile page.
- `time_zone`: The time zone of your system. For example: Europe/Paris.
- `lat`: The latitude. For example: 45.0.
- `lon`: The longitude. For example: 6.0
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
- `nominal_power_of_deferrable_loads`: The nominal power for each deferrable load in Watts. This is a list with a number of elements consistent with the number of deferrable loads defined before. For example:
	- 3000
	- 750
- `operating_hours_of_each_deferrable_load`: The total number of hours that each deferrable load should operate. For example:
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
- `weather_forecast_method`: This will define the weather forecast method that will be used. The options are `open-meteo` to use the weather forecast API proposed by [Open-Meteo](https://open-meteo.com/), `solcast` to use the [Solcast](https://solcast.com/) solar forecast service, `solar.forecast` to use the free public [Solar.Forecast](https://forecast.solar/) account and finally the `csv` to load a CSV file. When loading a CSV file this will be directly considered as the PV power forecast in Watts. The default CSV file path that will be used is `/data/data_weather_forecast.csv`. This method is useful to load and use other external forecasting service data in EMHASS. Defaults to `open-meteo` method.
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
	- `load_peak_hours_cost`: The cost of the electrical energy from the grid during peak hours in €/kWh. Defaults to 0.1907.
	- `load_offpeak_hours_cost`: The cost of the electrical energy from the grid during non-peak hours in €/kWh. Defaults to 0.1419.
- `production_price_forecast_method`: Define the method that will be used for PV power production price forecast. This is the price that is paid by the utility for energy injected into the grid. The options are `constant` for a constant fixed value or `csv` to load custom price forecasts from a CSV file. The default CSV file path that will be used is `/data/data_prod_price_forecast.csv`.
```{note} 

For all the forecast methods (`weather`, `load_power`, `load_cost` and `production_price`) it is also possible to pass the data from external services using list of values or a dictionary with timestamps. For more information check the dedicated [Passing your own forecast data](https://emhass.readthedocs.io/en/latest/forecasts.html#passing-your-own-forecast-data) section.
```
- `photovoltaic_production_sell_price`: The paid price for energy injected to the grid from excedent PV production in €/kWh. Defaults to 0.065. This parameter is only needed if production_price_forecast_method='constant'.
- `set_total_pv_sell`: Set this parameter to true to consider that all the PV power produced is injected to the grid. No direct self-consumption. The default is false, for a system with direct self-consumption.
- `set_use_adjusted_pv`: Set to True to enable machine learning-based PV forecast adjustment. This uses historical data to train a regression model that corrects PV forecasts based on local conditions. Defaults to False. See the [forecasts documentation](https://emhass.readthedocs.io/en/latest/forecasts.html#adjusting-pv-forecasts-using-machine-learning) for more details.
- `adjusted_pv_regression_model`: The regression model to use for PV forecast adjustment. See `REGRESSION_METHODS` in `machine_learning_regressor.py` for the authoritative list. Currently available: 'LinearRegression', 'RidgeRegression', 'LassoRegression' (default), 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'SVR', 'RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'MLPRegressor'. Only used when `set_use_adjusted_pv` is True.
- `adjusted_pv_solar_elevation_threshold`: The solar elevation threshold in degrees below which the adjusted PV forecast is set to zero. This prevents negative or unrealistic values during low sun angles. Defaults to 10.
- `adjusted_pv_model_max_age`: Maximum age in hours before the adjusted PV regression model is re-fitted. If the saved model is older than this value, a new model will be trained using fresh historical data. Set to 0 to force re-fitting on every call. Defaults to 24 hours (1 day). This caching mechanism significantly reduces API calls to Home Assistant and speeds up optimization runs.
- `lp_solver`: Set the name of the linear programming solver that will be used. Defaults to 'COIN_CMD'. The options are 'PULP_CBC_CMD', 'GLPK_CMD', 'HiGHS', and 'COIN_CMD'.
- `lp_solver_path`: Set the path to the LP solver. Defaults to '/usr/bin/cbc'. 
- `num_threads`: Set the number of threads to pass to LP solvers that support specifying a number of threads. Defaults to 0 (auto-detect).
- `set_nocharge_from_grid`: Set this to true if you want to forbid charging the battery from the grid. The battery will only be charged from excess PV.
- `set_nodischarge_to_grid`: Set this to true if you want to forbid discharging battery power to the grid.
- `set_battery_dynamic`: Set a power dynamic limiting condition to the battery power. This is an additional constraint on the battery dynamic in power per unit of time, which allows you to set a percentage of the battery's nominal full power as the maximum power allowed for (dis)charge.
- `battery_dynamic_max`: The maximum positive (for discharge) battery power dynamic. This is the allowed power variation (in percentage) of battery maximum power per unit of time.
- `battery_dynamic_min`: The maximum negative (for charge) battery power dynamic. This is the allowed power variation (in percentage) of battery maximum power per unit of time.
- `weight_battery_discharge`: An additional weight (currency/ kWh) applied in the cost function to battery usage for discharging. Defaults to 0.00
- `weight_battery_charge`: An additional weight (currency/ kWh) applied in the cost function to battery usage for charging. Defaults to 0.00

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
- `compute_curtailment`: Set to True to compute a special PV curtailment variable (Default False).
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