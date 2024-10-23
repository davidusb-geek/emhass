# Configuration file

In this section, we will explain all the parts of the `config_emhass.yaml` needed to properly run EMHASS.

We will find three main parts in the configuration file:

- The parameters needed to retrieve data from Home Assistant (retrieve_hass_conf)
- The parameters to define the optimization problem (optim_conf)
- The parameters used to model the system (plant_conf)

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

A second part of this section is given by some privacy-sensitive parameters that should be included in a `secrets_emhass.yaml` file alongside the `config_emhass.yaml` file.

The parameters in the `secrets_emhass.yaml` file are:

- `hass_url`: The URL to your Home Assistant instance. For example: https://myhass.duckdns.org/
- `long_lived_token`: A Long-Lived Access Token from the Lovelace profile page.
- `time_zone`: The time zone of your system. For example: Europe/Paris.
- `lat`: The latitude. For example: 45.0.
- `lon`: The longitude. For example: 6.0
- `alt`: The altitude in meters. For example: 100.0

## Optimization configuration parameters

These are the parameters needed to properly define the optimization problem.

- `set_use_battery`: Set to True if we should consider an energy storage device such as a Li-Ion battery. Defaults to False.
- `delta_forecast_daily`: The number of days for forecasted data. Defaults to 1.
- `number_of_deferrable_loads`: Define the number of deferrable loads to consider. Defaults to 2.
- `nominal_power_of_deferrable_loads`: The nominal power for each deferrable load in Watts. This is a list with a number of elements consistent with the number of deferrable loads defined before. For example:
	- 3000
	- 750
- `operating_hours_of_each_deferrable_load`: The total number of hours that each deferrable load should operate. For example:
	- 5
	- 8
- `start_timesteps_of_each_deferrable_load`: The timestep as from which each deferrable load is allowed to operate (if you don't want the deferrable load to use the whole optimization time window). If you specify a value of 0 (or negative), the deferrable load will be optimized as from the beginning of the complete prediction horizon window. For example:
    - 0
    - 1 
- `end_timesteps_of_each_deferrable_load`: The timestep before which each deferrable load should operate. The deferrable load is not allowed to operate after the specified time step. If a value of 0 (or negative) is provided, the deferrable load is allowed to operate in the complete optimization window). For example:
	- 0
	- 3
- `treat_deferrable_load_as_semi_cont`: Define if we should treat each deferrable load as a semi-continuous variable. Semi-continuous variables (`True`) are variables that must take a value that can be either their maximum or minimum/zero (for example On = Maximum load, Off = 0 W). Non semi-continuous (which means continuous) variables (`False`) can take any values between their maximum and minimum. For example:
	- True
	- True
- `set_deferrable_load_single_constant`: Define if we should set each deferrable load as a constant fixed value variable with just one startup for each optimization task. For example:
	- False
	- False
- `set_deferrable_startup_penalty`: Set to a list of floats. For each deferrable load with a penalty `P`, each time the deferrable load turns on will incur an additional cost of `P * nominal_power_of_deferrable_loads * cost_of_electricity` at that time. 
- `weather_forecast_method`: This will define the weather forecast method that will be used. The options are 'scrapper' for a scrapping method for weather forecast from clearoutside.com and 'csv' to load a CSV file. When loading a CSV file this will be directly considered as the PV power forecast in Watts. The default CSV file path that will be used is '/data/data_weather_forecast.csv'. Defaults to 'scrapper' method.
- `load_forecast_method`: The load forecast method that will be used. The options are 'csv' to load a CSV file or 'naive' for a simple 1-day persistence model. The default CSV file path that will be used is '/data/data_load_forecast.csv'. Defaults to 'naive'.
- `load_cost_forecast_method`: Define the method that will be used for load cost forecast. The options are 'hp_hc_periods' for peak and non-peak hours contracts and 'csv' to load custom cost from CSV file. The default CSV file path that will be used is '/data/data_load_cost_forecast.csv'.
The following parameters and definitions are only needed if load_cost_forecast_method='hp_hc_periods':
	- `load_peak_hour_periods`: Define a list of peak hour periods for load consumption from the grid. This is useful if you have a contract with peak and non-peak hours. For example for two peak hour periods: 
		- period_hp_1:
			- start: '02:54'
			- end: '15:24'
		- period_hp_2:
			- start: '17:24'
			- end: '20:24'
	- `load_peak_hours_cost`: The cost of the electrical energy from the grid during peak hours in €/kWh. Defaults to 0.1907.
	- `load_offpeak_hours_cost`: The cost of the electrical energy from the grid during non-peak hours in €/kWh. Defaults to 0.1419.

- `production_price_forecast_method`: Define the method that will be used for PV power production price forecast. This is the price that is paid by the utility for energy injected into the grid. The options are 'constant' for a constant fixed value or 'csv' to load custom price forecasts from a CSV file. The default CSV file path that will be used is '/data/data_prod_price_forecast.csv'.
- `photovoltaic_production_sell_price`: The paid price for energy injected to the grid from excedent PV production in €/kWh. Defaults to 0.065. This parameter is only needed if production_price_forecast_method='constant'.
- `set_total_pv_sell`: Set this parameter to true to consider that all the PV power produced is injected to the grid. No direct self-consumption. The default is false, for a system with direct self-consumption.
- `lp_solver`: Set the name of the linear programming solver that will be used. Defaults to 'COIN_CMD'. The options are 'PULP_CBC_CMD', 'GLPK_CMD' and 'COIN_CMD'. 
- `lp_solver_path`: Set the path to the LP solver. Defaults to '/usr/bin/cbc'. 
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

- `pv_module_model`: The PV module model. For example: 'CSUN_Eurasia_Energy_Systems_Industry_and_Trade_CSUN295_60M'. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example one east-facing array (azimuth=90) and one west-facing array (azimuth=270). When finding the correct model for your installation remember to replace all the special characters in the model name with '_'. The name of the table column for your device on the webapp will already have the correct naming convention.
- `pv_inverter_model`: The PV inverter model. For example: 'Fronius_International_GmbH__Fronius_Primo_5_0_1_208_240__240V_'. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example, one east-facing array (azimuth=90) and one west-facing array (azimuth=270). When finding the correct model for your installation remember to replace all the special characters in the model name with '_'. The name of the table column for your device on the web app will already have the correct naming convention.
- `surface_tilt`: The tilt angle of your solar panels. Defaults to 30. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example, one east-facing array (azimuth=90) and one west-facing array (azimuth=270). 
- `surface_azimuth`: The azimuth of your PV installation. Defaults to 205. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example, one east-facing array (azimuth=90) and one west-facing array (azimuth=270). 
- `modules_per_string`: The number of modules per string. Defaults to 16. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example, one east-facing array (azimuth=90) and one west-facing array (azimuth=270). 
- `strings_per_inverter`: The number of used strings per inverter. Defaults to 1. This parameter can be a list of items to enable the simulation of mixed orientation systems, for example one east-facing array (azimuth=90) and one west-facing array (azimuth=270).
- `inverter_is_hybrid`: Set to True to consider that the installation inverter is hybrid for PV and batteries (Default False).
- `compute_curtailment`: Set to True to compute a special PV curtailment variable (Default False).

If your system has a battery (set_use_battery=True), then you should define the following parameters:

- `battery_discharge_power_max`: The maximum discharge power in Watts. Defaults to 1000.
- `battery_charge_power_max`: The maximum charge power in Watts. Defaults to 1000.
- `battery_discharge_efficiency`: The discharge efficiency. Defaults to 0.95.
- `battery_charge_efficiency`: The charge efficiency. Defaults to 0.95.
- `battery_nominal_energy_capacity`: The total capacity of the battery stack in Wh. Defaults to 5000.
- `battery_minimum_state_of_charge`: The minimum allowable battery state of charge. Defaults to 0.3.
- `battery_maximum_state_of_charge`: The maximum allowable battery state of charge. Defaults to 0.9.
- `battery_target_state_of_charge`: The desired battery state of charge at the end of each optimization cycle. Defaults to 0.6.
