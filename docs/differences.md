# EMHASS & EMHASS-Add-on differences
Users will pass parameters and secrets into EMHASS differently, based on what method your running emhass. (Add-on, Docker, Python)
This page tries to help to resolve the common confusion between the different methods.

## Legacy Parameter definitions  
After EMHASS version:`0.10.6`, EMHASS has merged the parameter config from the legacy modes (`config_emhass.yaml` & `options.json`) to a central `config.json`.

The resulting change saw a migration of the parameter naming conventions.   
*Many of the new parameter definitions seen in `config.json` are copied from the Add-on, however not all.*

To simply convert from the legacy method (EMHASS>=0.10.6) to the new `config.json` method, see this video guide:  
- Standalone Mode: https://youtu.be/T85DAdXnGFY?feature=shared&t=938
- Addon Mode: https://youtu.be/T85DAdXnGFY?feature=shared&t=1341

See below for a list of associations between the parameters from `config_emhass.yaml` *(Legacy Standalone mode)*, `options.json` *(Legacy Add-on mode)* and the `config.json` parameter definitions:  

| config catagories | config_emhass.yaml *(Legacy)* | config.json | options.json list dictionary key *(Legacy)* |
| ------ | ------------------ | ------------ | -------------------------------- |
| retrieve_hass_conf |  freq |  optimization_time_step | |
| retrieve_hass_conf |  days_to_retrieve |  historic_days_to_retrieve | |
| retrieve_hass_conf |  var_PV |  sensor_power_photovoltaics | |
| retrieve_hass_conf |  var_load |  sensor_power_load_no_var_loads | |
| retrieve_hass_conf |  load_negative |  load_negative | |
| retrieve_hass_conf |  set_zero_min |  set_zero_min | |
| retrieve_hass_conf |  method_ts_round |  method_ts_round | |
| retrieve_hass_conf |  continual_publish |  continual_publish | |
| params_secrets |  solcast_api_key |  optional_solcast_api_key | |
| params_secrets |  solcast_rooftop_id |  optional_solcast_rooftop_id | |
| params_secrets |  solar_forecast_kwp |  optional_solar_forecast_kwp | |
| params_secrets |  time_zone |  time_zone | |
| params_secrets |  lat |  Latitude | |
| params_secrets |  lon |  Longitude | |
| params_secrets |  alt |  Altitude | |
| optim_conf |  set_use_battery |  set_use_battery | |
| optim_conf |  num_def_loads |  number_of_deferrable_loads | |
| optim_conf |  P_deferrable_nom |  list_nominal_power_of_deferrable_loads |  nominal_power_of_deferrable_loads | 
| optim_conf |  def_total_hours |  list_operating_hours_of_each_deferrable_load |  operating_hours_of_each_deferrable_load | 
| optim_conf |  treat_def_as_semi_cont |  list_treat_deferrable_load_as_semi_cont |  treat_deferrable_load_as_semi_cont | 
| optim_conf |  set_def_constant |  list_set_deferrable_load_single_constant |  set_deferrable_load_single_constant | 
| optim_conf |  def_start_penalty |  list_set_deferrable_startup_penalty |  set_deferrable_startup_penalty | 
| optim_conf |  weather_forecast_method |  weather_forecast_method | |
| optim_conf |  load_forecast_method |  load_forecast_method | |
| optim_conf |  delta_forecast |  delta_forecast_daily | |
| optim_conf |  load_cost_forecast_method |  load_cost_forecast_method | |
| optim_conf |  load_cost_hp |  load_peak_hours_cost | |
| optim_conf |  load_cost_hc |  load_offpeak_hours_cost | |
| optim_conf |  prod_price_forecast_method |  production_price_forecast_method | |
| optim_conf |  prod_sell_price |  photovoltaic_production_sell_price | |
| optim_conf |  set_total_pv_sell |  set_total_pv_sell | |
| optim_conf |  lp_solver |  lp_solver | |
| optim_conf |  lp_solver_path |  lp_solver_path | |
| optim_conf |  set_nocharge_from_grid |  set_nocharge_from_grid | |
| optim_conf |  set_nodischarge_to_grid |  set_nodischarge_to_grid | |
| optim_conf |  set_battery_dynamic |  set_battery_dynamic | |
| optim_conf |  battery_dynamic_max |  battery_dynamic_max | |
| optim_conf |  battery_dynamic_min |  battery_dynamic_min | |
| optim_conf |  weight_battery_discharge |  weight_battery_discharge | | 
| optim_conf |  weight_battery_charge |  weight_battery_charge | |
| optim_conf |  def_start_timestep |  list_start_timesteps_of_each_deferrable_load |  start_timesteps_of_each_deferrable_load | 
| optim_conf |  def_end_timestep |  list_end_timesteps_of_each_deferrable_load |  end_timesteps_of_each_deferrable_load | 
| plant_conf |  P_grid_max |  maximum_power_from_grid | |
| plant_conf |  module_model |  list_pv_module_model |  pv_module_model  | |
| plant_conf |  inverter_model |  list_pv_inverter_model |  pv_inverter_model  | |
| plant_conf |  surface_tilt |  list_surface_tilt |  surface_tilt  | |
| plant_conf |  surface_azimuth |  list_surface_azimuth |  surface_azimuth | |
| plant_conf |  modules_per_string,list_modules_per_string |  modules_per_string | |
| plant_conf |  strings_per_inverter |  list_strings_per_inverter |  strings_per_inverter | |
| plant_conf |  Pd_max |  battery_discharge_power_max || 
| plant_conf |  Pc_max |  battery_charge_power_max | |
| plant_conf |  eta_disch |  battery_discharge_efficiency | |
| plant_conf |  eta_ch |  battery_charge_efficiency | |
| plant_conf |  Enom |  battery_nominal_energy_capacity | |
| plant_conf |  SOCmin |  battery_minimum_state_of_charge | |
| plant_conf |  SOCmax |  battery_maximum_state_of_charge | |
| plant_conf |  SOCtarget |  battery_target_state_of_charge | |

Descriptions of each parameter can be found at:
-  [`Configuration Documentation`](https://emhass.readthedocs.io/en/latest/config.html) 
- Configuration page on EMHASS web server (E.g. http://localhost:5000/configuration)

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



