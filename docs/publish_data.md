# Publish data to Home Assistant

## The publish-data specificities

`publish-data` (which is either run manually or automatically via `continual_publish` or Home Assistant automation), will push the optimization results to Home Assistant for each deferrable load defined in the configuration. For example, if you have defined two deferrable loads, then the command will publish `sensor.p_deferrable0` and `sensor.p_deferrable1` to Home Assistant. When the `dayahead-optim` is launched, after the optimization, either entity json files or a csv file will be saved on disk. The `publish-data` command will load the latest csv/json files to look for the closest timestamp that matches the current time using the `datetime.now()` method in Python. This means that if EMHASS is configured for 30-minute time step optimizations, the csv/json will be saved with timestamps 00:00, 00:30, 01:00, 01:30, ... and so on. If the current time is 00:05, and parameter `method_ts_round` is set to `nearest` in the configuration, then the closest timestamp of the optimization results that will be published is 00:00. If the current time is 00:25, then the closest timestamp of the optimization results that will be published is 00:30.

The `publish-data` command will also publish PV and load forecast data on sensors `p_pv_forecast` and `p_load_forecast`. If using a battery, then the battery-optimized power and the SOC will be published on sensors `p_batt_forecast` and `soc_batt_forecast`. On these sensors, the future values are passed as nested attributes.

If you run publish manually *(or via a Home Assistant Automation)*, it is possible to provide custom sensor names for all the data exported by the `publish-data` command. For this, when using the `publish-data` endpoint we can just add some runtime parameters as dictionaries like this:
```yaml
shell_command:
  publish_data: "curl -i -H \"Content-Type:application/json\" -X POST -d '{\"custom_load_forecast_id\": {\"entity_id\": \"sensor.p_load_forecast\", \"unit_of_measurement\": \"W\", \"friendly_name\": \"Load Power Forecast\"}}' http://localhost:5000/action/publish-data"
```

These keys are available to modify: `custom_pv_forecast_id`, `custom_load_forecast_id`, `custom_batt_forecast_id`, `custom_batt_soc_forecast_id`,  `custom_grid_forecast_id`, `custom_cost_fun_id`, `custom_deferrable_forecast_id`, `custom_unit_load_cost_id` and `custom_unit_prod_price_id`.

If you provide the `custom_deferrable_forecast_id` then the passed data should be a list of dictionaries, like this:
```yaml
shell_command:
  publish_data: "curl -i -H \"Content-Type:application/json\" -X POST -d '{\"custom_deferrable_forecast_id\": [{\"entity_id\": \"sensor.p_deferrable0\",\"unit_of_measurement\": \"W\", \"friendly_name\": \"Deferrable Load 0\"},{\"entity_id\": \"sensor.p_deferrable1\",\"unit_of_measurement\": \"W\", \"friendly_name\": \"Deferrable Load 1\"}]}' http://localhost:5000/action/publish-data"
```
You should be careful that the list of dictionaries has the correct length, which is the number of defined deferrable loads.

## Computed variables and published data

Below you can find a list of the variables resulting from EMHASS computation, shown in the charts and published to Home Assistant through the ```publish_data``` command:

| EMHASS variable | Definition | Home Assistant published sensor |
| --------------- | ---------- | --------------------------------|
| P_PV | Forecasted power generation from your solar panels (Watts). This helps you predict how much solar energy you will produce during the forecast period. | sensor.p_pv_forecast |
| P_Load | Forecasted household power consumption (Watts). This gives you an idea of how much energy your appliances are expected to use. | sensor.p_load_forecast |
| P_deferrableX<br/>[X = 0, 1, 2, ...] | Forecasted power consumption of deferrable loads (Watts). Deferable loads are appliances that can be managed by EMHASS. EMHASS helps you optimize energy usage by prioritizing solar self-consumption and minimizing reliance on the grid or by taking advantage or supply and feed-in tariff volatility. You can have multiple deferable loads and you use this sensor in HA to control these loads via smart switch or other IoT means at your disposal. | sensor.p_deferrableX |
| P_grid_pos | Forecasted power imported from the grid (Watts). This indicates the amount of energy you are expected to draw from the grid when your solar production is insufficient to meet your needs or it is advantageous to consume from the grid. | - |
| P_grid_neg | Forecasted power exported to the grid (Watts). This indicates the amount of excess solar energy you are expected to send back to the grid during the forecast period. | - |
| P_batt | Forecasted (dis)charge power load (Watts) for the battery (if installed). If negative it indicates the battery is charging, if positive that the battery is discharging. | sensor.p_batt_forecast |
| P_grid | Forecasted net power flow between your home and the grid (Watts). This is calculated as P_grid_pos + P_grid_neg. A positive value indicates net import, while a negative value indicates net export. | sensor.p_grid_forecast |
| SOC_opt | Forecasted battery optimized Status Of Charge (SOC) percentage level | sensor.soc_batt_forecast |
| unit_load_cost | Forecasted cost per unit of energy you pay to the grid (typically "Currency"/kWh). This helps you understand the expected energy cost during the forecast period. | sensor.unit_load_cost |
| unit_prod_price | Forecasted price you receive for selling excess solar energy back to the grid (typically "Currency"/kWh). This helps you understand the potential income from your solar production. | sensor.unit_prod_price |
| cost_profit | Forecasted profit or loss from your energy usage for the forecast period. This is calculated as unit_load_cost * P_Load - unit_prod_price * P_grid_pos. A positive value indicates a profit, while a negative value indicates a loss. | sensor.total_cost_profit_value |
| cost_fun_cost | Forecasted cost associated with deferring loads to maximize solar self-consumption. This helps you evaluate the trade-off between managing the load and not managing and potential cost savings. | sensor.total_cost_fun_value |
| optim_status | This contains the status of the latest execution and is the same you can see in the Log following an optimization job. Its values can be Optimal or Infeasible. | sensor.optim_status |

## Alternative publish methods
Due to the flexibility of EMHASS, multiple different approaches to publishing the optimization results have been created. Select an option that best meets your use case:

### Publish last optimization *(manual)*
By default, running an optimization in EMHASS will output the results into the CSV file: `data_path/opt_res_latest.csv`  *(overriding the existing data on that file)*. We run the publish command to publish the last optimization saved in the `opt_res_latest.csv`:
```bash
# RUN dayahead
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/dayahead-optim
# Then publish the results of dayahead
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/publish-data
```
*Note, the published entities from the publish-data action will not automatically update the entities' current state (current state being used to check when to turn on and off appliances via Home Assistant automations). To update the EMHASS entities state, another publish would have to be re-run later when the current time matches the next value's timestamp (e.g. every 30 minutes). See examples below for methods to automate the publish-action.*

```{note}

If your are using the thermal model, then for this manual publish implementation you need to indicate which of your deferrable loads are thermal.
We do this with the `"def_load_config"` entry.

If you have just one deferrable load which is a thermal load then the payload of `curl` publish command should look like:
`{"def_load_config": [{"thermal_config": {}}]}`

Similarly if you have three deferrable loads where the second load is a thermal load the payload would have been:
`{"def_load_config": [{},{"thermal_config": {}},{}]}`

```

### Continual_publish *(EMHASS Automation)*
Setting `continual_publish` to `true` in the configuration saves the output of the optimization into the `data_path/entities` folder *(a .json file for each sensor/entity)*. A constant loop (in `optimization_time_step` minutes) will run, observe the .json files in that folder, and publish the saved files periodically (updating the current state of the entity by comparing date.now with the saved data value timestamps). 

For users that wish to run multiple different optimizations, you can set the runtime parameter: `publish_prefix` to something like: `"mpc_"` or `"dh_"`. This will generate unique entity_id names per optimization and save these unique entities as separate files in the folder. All the entity files will then be updated when the next loop iteration runs. If a different `optimization_time_step` integer was passed as a runtime parameter in an optimization, the `continual_publish` loop will be based on the lowest `optimization_time_step` saved. An example:

```bash
# RUN dayahead, with optimization_time_step=30 (default), prefix=dh_ 
curl -i -H 'Content-Type:application/json' -X POST -d '{"publish_prefix":"dh_"}' http://localhost:5000/action/dayahead-optim
# RUN MPC, with optimization_time_step=5, prefix=mpc_
curl -i -H 'Content-Type:application/json' -X POST -d '{'optimization_time_step':5,"publish_prefix":"mpc_"}' http://localhost:5000/action/naive-mpc-optim
```
This will tell continual_publish to loop every 5 minutes based on the optimization_time_step passed in MPC. All entities from the output of dayahead "dh_" and MPC "mpc_" will be published every 5 minutes.

</br>

*It is recommended to use the 2 other options below once you have a more advanced understanding of EMHASS and/or Home Assistant.*

### Mixture of continual_publish and manual *(Home Assistant Automation for Publish)*

You can choose to save one optimization for continual_publish and bypass another optimization by setting `'continual_publish':false` runtime parameter:
```bash
# RUN dayahead, with optimization_time_step=30 (default), prefix=dh_, included into continual_publish
curl -i -H 'Content-Type:application/json' -X POST -d '{"publish_prefix":"dh_"}' http://localhost:5000/action/dayahead-optim

# RUN MPC, with optimization_time_step=5, prefix=mpc_, Manually publish, excluded from continual_publish loop
curl -i -H 'Content-Type:application/json' -X POST -d '{'continual_publish':false,'optimization_time_step':5,"publish_prefix":"mpc_"}' http://localhost:5000/action/naive-mpc-optim
# Publish MPC output
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/publish-data
```
This example saves the dayahead optimization into `data_path/entities` as .json files, being included in the `continutal_publish` loop (publishing every 30 minutes). The MPC optimization will not be saved in `data_path/entities`, and therefore only into `data_path/opt_res_latest.csv`. Requiring a publish-data action to be run manually (or via a Home Assistant) Automation for the MPC results. 

### Manual *(Home Assistant Automation for Publish)*

For users who wish to have full control of exactly when they would like to run a publish and have the ability to save multiple different optimizations. The `entity_save` runtime parameter has been created to save the optimization output entities to .json files whilst  `continual_publish` is set to `false` in the configuration. Allowing the user to reference the saved .json files manually via a publish:

in configuration page/`config.json` :
```json
"continual_publish": false
```
POST action :
```bash
# RUN dayahead, with optimization_time_step=30 (default), prefix=dh_, save entity
curl -i -H 'Content-Type:application/json' -X POST -d '{"entity_save": true, "publish_prefix":"dh_"}' http://localhost:5000/action/dayahead-optim
# RUN MPC, with optimization_time_step=5, prefix=mpc_, save entity
curl -i -H 'Content-Type:application/json' -X POST -d '{"entity_save": true", 'optimization_time_step':5,"publish_prefix":"mpc_"}' http://localhost:5000/action/naive-mpc-optim
```
You can then reference these .json saved entities via their `publish_prefix`. Include the same `publish_prefix` in the `publish_data` action:
```bash
#Publish the MPC optimization ran above 
curl -i -H 'Content-Type:application/json' -X POST -d '{"publish_prefix":"mpc_"}'  http://localhost:5000/action/publish-data
```
This will publish all entities from the MPC (_mpc) optimization above.
</br>
Alternatively, you can choose to publish all the saved files .json files with `publish_prefix` = all:
```bash
#Publish all saved entities
curl -i -H 'Content-Type:application/json' -X POST -d '{"publish_prefix":"all"}'  http://localhost:5000/action/publish-data
```
This action will publish the dayahead (_dh) and MPC (_mpc) optimization results from the optimizations above.