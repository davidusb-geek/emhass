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

### Passing other data at runtime

It is possible to also pass other data during runtime to automate energy management. For example, it could be useful to dynamically update the total number of hours for each deferrable load (`operating_hours_of_each_deferrable_load`) using for instance a correlation with the outdoor temperature (useful for water heater for example). 

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

- `solcast_api_key` for the SolCast API key if you want to use this service for PV power production forecast.

- `solcast_rooftop_id` for the ID of your rooftop for the SolCast service implementation.

- `solar_forecast_kwp` for the PV peak installed power in kW used for the solar.forecast API call. 

- `battery_minimum_state_of_charge` the minimum possible SOC.

- `battery_maximum_state_of_charge` the maximum possible SOC.

- `battery_target_state_of_charge` for the desired target value of the initial and final SOC.

- `battery_discharge_power_max` for the maximum battery discharge power.

- `battery_charge_power_max` for the maximum battery charge power.

- `publish_prefix` use this key to pass a common prefix to all published data. This will add a prefix to the sensor name but also the forecast attribute keys within the sensor.

### Passing forecast data

There is a complete dedicated section in the [Forecast section](https://emhass.readthedocs.io/en/latest/forecasts.html).

Specifically the [Passing your own forecast data](https://emhass.readthedocs.io/en/latest/forecasts.html#passing-your-own-forecast-data) section.


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
  "influxdb_measurement": "state",
  "influxdb_retention_policy": "autogen",
  "influxdb_use_ssl": false,
  "influxdb_verify_ssl": false,
}
```

Finally, if using the Add-on, you need to fill both "influxdb_password" and "influxdb_username" in the Add-on **Configuration** pane.
If using the Docker standalone or legacy installation method, then you need to set these in the `secrets_emhass.yaml` file.


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

For more information on passing arguments and environment variables using docker, have a look at some examples from [Configuration and Installation](https://emhass.readthedocs.io/en/latest/intro.html#configuration-and-installation) and [EMHASS Development](https://emhass.readthedocs.io/en/latest/develop.html) pages. 

### EMHASS-Add-on *(Emhass Add-on)*
By default, the `URL` and `KEY` parameters have been set to `empty`/blank in the Home Assistant configuration page for EMHASS addon. This results in EMHASS calling its Local `Supervisor API` to gain access. This is the easiest method, as there is no user input necessary.  

However, if you wish to receive/send sensor data to a different Home Assistant environment, set url and key values in the `hass_url` & `long_lived_token` hidden parameters on the Home Assistant EMHASS addon configuration page. *(E.g. http://localhost:8123/hassio/addon/emhass/config)*
-  `hass_url` example: `https://192.168.1.2:8123/`  
-  `long_lived_token` generated from the `Long-lived access tokens` section in your user profile settings
</br></br>

Secret Parameters such as: `solcast_api_key`, `solcast_rooftop_id` and `solar_forecast_kwp` _(used by their respective `weather_forecast_method` parameter values)_, can also be set via hidden parameters in the addon configuration page.

Secret Parameters such as: `time_zone`, `lon`, `lat` and `alt` are also automatically passed in via the Home Assistants `Supervisor API`. _(Values set in the Home Assistants config/general page)_  
_Note: Local currency could also be obtained via the Home Assistant environment, however as of writing, this functionality has not yet been developed._
