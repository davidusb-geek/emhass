# The forecast module

EMHASS will need 4 forecasts to work properly:

- PV power production forecast (internally based on the weather forecast and the characteristics of your PV plant). This is given in Watts.

- Load power forecast: how much power your house will demand in the next 24 hours. This is given in Watts.

- Load cost forecast: the price of the energy from the grid in the next 24 hours. This is given in EUR/kWh.

- PV production selling price forecast: the price at which you will sell your excess PV production in the next 24 hours. This is given in currency/kWh.

Some methods are generalized to the 4 forecasts needed. For all the forecasts it is possible to pass the data either as a passed list of values or by reading from a CSV file. With these methods, it is then possible to use data from external forecast providers.
    
Then there are the methods that are specific to each type of forecast and that proposed forecast is treated and generated internally by this EMHASS forecast class. 

For the **weather forecast**, the first method (`open-meteo`) uses Open-Meteo weather forecast API, which proposes detailed forecasts based on Lat/Lon locations. Another method (`solcast`) is using the Solcast PV production forecast service. A final method (`solar.forecast`) is using another external service: Solar.Forecast, for which just the nominal PV peak installed power should be provided. Search the forecast section on the documentation for examples of how to implement these different methods.

The `get_power_from_weather` method is proposed here to convert irradiance data to electrical power. The PVLib module is used to model the PV plant. A dedicated web app will help you search for your correct PV module and inverter: [https://emhass-pvlib-database.streamlit.app/](https://emhass-pvlib-database.streamlit.app/)

The specific methods for the **load forecast** is a first method (`typical`) that uses historic values of a typical household power consumption. This method uses simple statistic methods and load power grouped by the same day-of-the-week of the current month. The load power is scaled using the parameter `maximum_power_from_grid`. A second method (`naive`) that uses a naive approach, also called persistence. It simply assumes that the forecast for 
a future period will be equal to the observed values in a past period. The past period is controlled using the parameter `delta_forecast_daily`. A third method (`mlforecaster`)
uses an internal custom forecasting model using machine learning. There is a section in the documentation explaining how to use this method.
    
```{note} 

This custom machine learning model is introduced from v0.4.0. EMHASS proposed this new `mlforecaster` class with `fit`, `predict` and `tune` methods. Only the `predict` method is used here to generate new forecasts, but it is necessary to previously fit a forecaster model and it is a good idea to optimize the model hyperparameters using the `tune` method. See the dedicated section in the documentation for more help.
```

For the PV production selling price and Load cost forecasts the privileged method is a direct read from a user-provided list of values. The list should be passed as a runtime parameter during the `curl` to the EMHASS API.

## PV power production forecast

#### open-meteo 

The default method for PV power forecast is the weather forecast API proposed by [Open-Meteo](https://open-meteo.com/). For more detail see the [Open-Meteo API documentation](https://open-meteo.com/en/docs). This is obtained using `method=open-meteo`. This site proposes detailed forecasts based on Lat/Lon locations. The weather forecast data is then converted into PV power production using the `list_pv_module_model` and `list_pv_inverter_model` parameters defined in the configuration.

#### solcast 

The second method uses the Solcast solar forecast service. Go to [https://solcast.com/](https://solcast.com/) and configure your system. You will need to set `method=solcast` and use two parameters `solcast_rooftop_id` and `solcast_api_key` that should be passed as parameters at runtime or provided in the configuration/secrets. The free hobbyist account will be limited to 10 API requests per day, the granularity will be 30 minutes and the forecast will be updated every 6 hours. If needed, better performances may be obtained with paid plans: [https://solcast.com/pricing/live-and-forecast](https://solcast.com/pricing/live-and-forecast).

For example:
```yaml
# Set weather_forecast_method parameter to solcast in your configuration (configuration page / config_emhass.yaml)
weather_forecast_method: 'solcast'
```
```bash
# Example of running day-ahead, passing Solcast secrets fyi runtime parameters (i.e. not set in configuration)
curl -i -H "Content-Type:application/json" -X POST -d '{"solcast_rooftop_id":"<your_system_id>","solcast_api_key":"<your_secret_api_key>"}' http://localhost:5000/action/dayahead-optim
```
</br>

#### solar.forecast 

A third method uses the Solar.Forecast service. You will need to set `method=solar.forecast` and use just one parameter `solar_forecast_kwp` (the PV peak installed power in kW) that should be passed at runtime. This will be using the free public Solar.Forecast account with 12 API requests per hour, per IP, and 1h data resolution. As with Solcast, there are paid account services that may result in better forecasts.

For example, for a 5 kW installation:
```bash
curl -i -H "Content-Type:application/json" -X POST -d '{"solar_forecast_kwp":5}' http://localhost:5000/action/dayahead-optim
```

```{note} 

If you use the Solar.Forecast or Solcast methods, or explicitly pass the PV power forecast values (see below), the list_pv_module_model and list_pv_inverter_model parameters defined in the configuration will be ignored.
```

#### Caching PV Forecast
For the MPC users, running optimizations regularly; You may wish to cache your PV forecast results, to reuse throughout the day.
Partially for those who use the free plan of Solcast, Caching can help reduce the amount of calls bellow 10 a day.
Caching Forecast data will also speed up the forecast process, bypassing the need to call to the external forecast API each MPC run. 

```bash
# Run weather forecast and cache results (Recommended to run this 1-10 times a day, throughout the day)
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/weather-forecast-cache

# Then run your regular MPC call (E.g. every 5 minutes)
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/naive-mpc-optim
```
EMHASS will see the saved cache file and use its data over pulling new data from the API.

`weather_forecast_cache` can also be provided as a runtime parameter, in an optimization, to save the forecast results to cache:
```bash
# Example of running day-ahead and optimization storing the retrieved Solcast data to cache
curl -i -H 'Content-Type:application/json' -X POST -d '{"weather_forecast_cache":true}' http://localhost:5000/action/dayahead-optim
```

By default, if EMHASS finds a problem with the cache file, the cache will be automatically deleted. Due to the missing cache, the next optimization will run and pull data from the External API.
For Solcast only, If you wish to make sure that a certain optimization will only use the cached data, (otherwise present an error) the runtime parameter `weather_forecast_cache_only` can be used:
```bash
# Run the weather forecast action 1-10 times a day 
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/weather-forecast-cache

# Then run your regular MPC call (E.g. every 5 minutes) and make sure it only uses the Solcast cache. (do not pull from Solcast)
curl -i -H 'Content-Type:application/json' -X POST -d '{"weather_forecast_cache_only":true}' http://localhost:5000/action/naive-mpc-optim
```

#### Caching Open-Meteo Weather Service Usage

When you have EMHASS configured to use the Open-Meteo weather service, to minimize API calls to the service, and to provide
resilience in case of transient connectivity issues, EMHASS caches successful calls to the Open-Meteo API in a
`cached-open-meteo-forecast.json` file in the data directory. The JSON file contains the default 3 days of weather forecast data.
This Open-Meteo cache is independent of the PV cache discussed above and will be used even when the PV cache is not enabled.
By default, when the JSON file is older than 30 minutes, attempts will be made to replace it with a more recent version
from the Open-Meteo weather service. It will only be replaced if this is successful. If any errors occur the older version
will continue to be used until a new version can been fetched. The maximum cache age, with a default value of 30 minutes, can be
configured using the `open_meteo_cache_max_age` setting in config.json or as a parameter in EMHASS REST API calls.
The value is specified in minutes. If you want to disable caching you can specify a value of 0.

#### Adjusting PV Forecasts using machine learning
EMHASS provides methods to adjust the PV power forecast using machine learning regression techniques. The adjustment process consists of two steps: training a regression model using historical PV data and then applying the trained model to correct new PV forecasts.

This functionality may help to **fine-tune** the PV prediction and model some local behavior of your PV production such as: tree shading, under-production due to dust/dirt, curtailment events, local micro-weather conditions, etc.

To activate this option all that is needed is to set `set_use_adjusted_pv` to `True` in the configuration.

The **Model Training** uses the `adjust_pv_forecast_fit` method in the `Forecast` class. This method fits a regression model to adjust the PV forecast. It uses historical forecasted and actual PV production data as training input, incorporating additional features such as time of day and solar angles. The model is trained using time-series cross-validation, with hyperparameter tuning performed via grid search. The best model is selected based on mean squared error scoring. The historical data retrieved depends on the `historic_days_to_retrieve` parameter in the configuration. By default, the method uses `LassoRegression`, but other models can also be specified using parameter `adjusted_pv_regression_model`. Once the model is trained, it computes root mean squared error (RMSE) and R² metrics to assess performance. These metrics are logged for reference. If debugging is disabled, the trained model is saved for future use.

The actual **Forecast Adjustment** is perfomed by the `adjust_pv_forecast_predict` method. This method applies the trained regression model to adjust PV forecast data. Before making predictions, the method enhances the data by adding date-based and solar-related features. It then uses the trained model to predict the adjusted forecast. A correction is applied based on solar elevation to prevent negative or unrealistic values, ensuring that the adjusted forecast remains physically meaningful. The correction based on solar elevation can be parametrized using a treshold value with parameter `adjusted_pv_solar_elevation_threshold` from the configuration. 

## Load power forecast

```{note} 

New in EMHASS v0.12.0: the default method for load power forecast is the `typical` statistics-based method!
```

The default method for load forecast is the `typical` method, which uses basic statistics and a year long load power data grouped by the current day-of-the-week of the current month. This provides a typical daily load power characteristic with a 30 minute resolution. The load power is scaled using the parameter `maximum_power_from_grid`. This method uses the default data with 1-year of load power consumption in file `data/data_train_load_clustering.pkl`. You can customize this data to your own household consumption by running erasing the previous file and running the script `scripts/load_clustering.py` (this will try to fetch 365 days of data from your load power sensor). However, if you have a working configuration without any problems with data retrieve from Home Assistant, then it is adviced to use the more advanced method `mlforecaster`.

A second method is a naive method, also called persistence. This is obtained using `method=naive`. This method simply assumes that the forecast for a future period will be equal to the observed values in a past period. The past period is controlled using the parameter `delta_forecast_daily` and the default value for this is 24h.

This is presented graphically here:

![](./images/naive_forecast.png)

Starting with v0.4.0, a new forecast framework is proposed within EMHASS. It provides a more efficient way to forecast the power load consumption. It is based on the `skforecast` module that uses `scikit-learn` regression models considering auto-regression lags as features. The hyperparameter optimization is proposed using Bayesian optimization from the `optuna` module. To use this change to `method=mlforecaster` in the configuration.

The API provides fit, predict and tune methods.

The following is an example of a trained model using a KNN regressor:

![](./images/load_forecast_knn_optimized.svg)

The naive persistence model performs very well on the 2-day test period, however, is well outperformed by the KNN regressor when back-testing on the complete training set (10 months of 30-minute time step data).

The hyperparameter tuning using Bayesian optimization improves the bare KNN regressor from $R^2=0.59$ to $R^2=0.75$. The optimized number of lags is $48$.

See the [machine learning forecaster](mlforecaster.md) section for more details.

## Load cost forecast

The default method for load cost forecast is defined for a peak and non-peak hours contract type. This is obtained using `method=hp_hc_periods`.

When using this method you can provide a list of peak-hour periods, so you can add as many peak-hour periods as possible.

As an example for a two peak-hour periods contract you will need to define the following list in the configuration file:

    - load_peak_hour_periods:
        - period_hp_1:
            - start: '02:54'
            - end: '15:24'
        - period_hp_2:
            - start: '17:24'
            - end: '20:24'
    - load_peak_hours_cost: 0.1907
    - load_offpeak_hours_cost: 0.1419

This example is presented graphically here:

![](./images/hp_hc_periods.png)

## PV production selling price forecast

The default method for this forecast is simply a constant value. This can be obtained using `method=constant`.

Then you will need to define the `photovoltaic_production_sell_price` variable to provide the correct price for energy injected to the grid from excedent PV production in €/kWh.

## Passing your own forecast data

For all the needed forecasts in EMHASS, two other methods allow the user to provide their own forecast value. This may be used to provide a forecast provided by a more powerful and accurate forecaster. The two methods are: `csv` and `list`.

For the `csv` method you should push a csv file to the `data` folder. The CSV file should contain no header and the timestamped data should have the following format:
    
    2021-04-29 00:00:00+00:00,287.07
    2021-04-29 00:30:00+00:00,274.27
    2021-04-29 01:00:00+00:00,243.38
    ...

For the list method, you just have to add the data as a list of values to a data dictionary during the call to `emhass` using the `runtimeparams` option.

The possible dictionary keys to pass data are:

- `pv_power_forecast` for the PV power production forecast.

- `load_power_forecast` for the Load power forecast.

- `load_cost_forecast` for the Load cost forecast.

- `prod_price_forecast` for the PV production selling price forecast.

For example, if using the add-on or the docker method, you can pass this data as a list of values to the data dictionary during the `curl` POST:
```bash
curl -i -H "Content-Type: application/json" -X POST -d '{"pv_power_forecast":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75, 804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}' http://localhost:5000/action/dayahead-optim
```

You need to be careful here to send the correct amount of data on this list, the correct length. For example, if the data time step is defined as 1 hour and you are performing a day-ahead optimization, then this list length should be 24 data points.

### Example using: Solcast forecast + Amber prices

If you're using Solcast then you can define the following sensors in your system:

```yaml
sensors:

  - platform: rest
    name: "Solcast Forecast Data"
    json_attributes:
      - forecasts
    resource: https://api.solcast.com.au/rooftop_sites/yyyy/forecasts?format=json&api_key=xxx&hours=24
    method: GET
    value_template: "{{ (value_json.forecasts[0].pv_estimate)|round(2) }}"
    unit_of_measurement: "kW"
    device_class: power
    scan_interval: 8000
    force_update: true

  - platform: template
    sensors:
      solcast_24hrs_forecast :
        value_template: >-
          {%- set power = state_attr('sensor.solcast_forecast_data', 'forecasts') | map(attribute='pv_estimate') | list %}
          {%- set values_all = namespace(all=[]) %}
          {% for i in range(power | length) %}
           {%- set v = (power[i] | float |multiply(1000) ) | int(0) %}
            {%- set values_all.all = values_all.all + [ v ] %}
          {%- endfor %} {{ (values_all.all)[:48] }}
```

With this, you can now feed this Solcast forecast to EMHASS along with the mapping of the Amber prices. 

An MPC call may look like this for 4 deferrable loads:

```yaml
    post_mpc_optim_solcast: "curl -i -H \"Content-Type: application/json\" -X POST -d '{\"load_cost_forecast\":{{(
          ([states('sensor.amber_general_price')|float(0)] +
          state_attr('sensor.amber_general_forecast', 'forecasts') |map(attribute='per_kwh')|list)[:48])
          }}, \"prod_price_forecast\":{{(
          ([states('sensor.amber_feed_in_price')|float(0)] +
          state_attr('sensor.amber_feed_in_forecast', 'forecasts')|map(attribute='per_kwh')|list)[:48]) 
          }}, \"pv_power_forecast\":{{states('sensor.solcast_24hrs_forecast')
          }}, \"prediction_horizon\":48,\"soc_init\":{{(states('sensor.powerwall_charge')|float(0))/100
          }},\"soc_final\":0.05,\"operating_hours_of_each_deferrable_load\":[2,0,0,0]}' http://localhost:5000/action/naive-mpc-optim"
```

Thanks to [@purcell_labs](https://github.com/purcell-lab) for this example configuration.

### Example combining multiple Solcast configurations

If you have multiple rooftops, for example for east-west facing solar panels, then you will need to fuze the sensors providing the different forecasts on a single one using templates in Home Assistant. Then feed that single sensor data passing the data as a list when calling the shell command.

Here is a sample configuration to achieve this, thanks to [@gieljnssns](https://github.com/gieljnssns) for sharing.

The two sensors using rest sensors:

```yaml
- platform: rest
  name: "Solcast Forecast huis"
  json_attributes:
    - forecasts
  resource: https://api.solcast.com.au/rooftop_sites/xxxxxxxxxxc/forecasts?format=json&api_key=xxxxxxxxx&hours=24
  method: GET
  value_template: "{{ (value_json.forecasts[0].pv_estimate)|round(2) }}"
  unit_of_measurement: "kW"
  device_class: power
  scan_interval: 86400
  force_update: true

- platform: rest
  name: "Solcast Forecast garage"
  json_attributes:
    - forecasts
  resource: https://api.solcast.com.au/rooftop_sites/xxxxxxxxxxc/forecasts?format=json&api_key=xxxxxxxxx&hours=24
  method: GET
  value_template: "{{ (value_json.forecasts[0].pv_estimate)|round(2) }}"
  unit_of_measurement: "kW"
  device_class: power
  scan_interval: 86400
  force_update: true
```

Then two templates, one for each sensor:

```yaml
    solcast_24hrs_forecast_garage:
      value_template: >-
        {%- set power = state_attr('sensor.solcast_forecast_garage', 'forecasts') | map(attribute='pv_estimate') | list %}
        {%- set values_all = namespace(all=[]) %}
        {% for i in range(power | length) %}
          {%- set v = (power[i] | float |multiply(1000) ) | int(0) %}
          {%- set values_all.all = values_all.all + [ v ] %}
        {%- endfor %} {{ (values_all.all)[:48] }}

    solcast_24hrs_forecast_huis:
      value_template: >-
        {%- set power = state_attr('sensor.solcast_forecast_huis', 'forecasts') | map(attribute='pv_estimate') | list %}
        {%- set values_all = namespace(all=[]) %}
        {% for i in range(power | length) %}
          {%- set v = (power[i] | float |multiply(1000) ) | int(0) %}
          {%- set values_all.all = values_all.all + [ v ] %}
        {%- endfor %} {{ (values_all.all)[:48] }}
```

And the fusion of the two sensors:

```yaml
    solcast_24hrs_forecast:
      value_template: >-
        {% set a = states("sensor.solcast_24hrs_forecast_garage")[1:-1].split(',') | map('int') | list %}
        {% set b = states("sensor.solcast_24hrs_forecast_huis")[1:-1].split(',') | map('int') | list %}
        {% set ns = namespace(items = []) %}
        {% for i in range(a | length) %}
          {% set ns.items = ns.items + [ a[i]  + b[i]  ] %}
        {% endfor %}
        {{ ns.items }}
```

And finally the shell command:

```yaml
dayahead_optim: "curl -i -H \"Content-Type:application/json\" -X POST -d '{\"pv_power_forecast\":{{states('sensor.solcast_24hrs_forecast')}}}' http://localhost:5001/action/dayahead-optim"
```

### Example using the Nordpool integration

The Nordpool integration provides spot market electricity prices (consumption and production) for the Nordic, Baltic and part of Western Europe.
An integration for Home Assistant can be found here: https://github.com/custom-components/nordpool

After setup the sensors should appear in Home Assistant for raw `today` and `tomorrow` values.

The subsequent shell command to concatenate `today` and `tomorrow` values can be for example:

```yaml
shell_command:
  trigger_nordpool_forecast: "curl -i -H \"Content-Type: application/json\" -X POST -d '{\"load_cost_forecast\":{{((state_attr('sensor.nordpool', 'raw_today') | map(attribute='value') | list  + state_attr('sensor.nordpool', 'raw_tomorrow') | map(attribute='value') | list))[now().hour:][:24] }},\"prod_price_forecast\":{{((state_attr('sensor.nordpool', 'raw_today') | map(attribute='value') | list  + state_attr('sensor.nordpool', 'raw_tomorrow') | map(attribute='value') | list))[now().hour:][:24]}}}' http://localhost:5000/action/dayahead-optim"
```

## Now/current values in forecasts

When implementing MPC applications with high optimization_time_step it can be interesting if, at each MPC iteration, the forecast values are updated with the real now/current values measured from live data. This is useful to improve the accuracy of the short-term forecasts. As shown in some of the references below, mixing with a persistence model makes sense since this type of model performs very well at low temporal resolutions (intra-hour).

A simple integration of current/now values for PV and load forecast is implemented using a mixed one-observation persistence model and the one-step-ahead forecasted values from the current passed method. 

This can be represented by the following equation at time $t=k$: 

$$
P^{mix}_{PV} = \alpha \hat{P}_{PV}(k) + \beta P_{PV}(k-1)
$$

Where $P^{mix}_{PV}$ is the mixed power forecast for PV production, $\hat{P}_{PV}(k)$ is the current first element of the original forecast data, $P_{PV}(k-1)$ is the now/current value of PV production and $\alpha$ and $\beta$ are coefficients that can be fixed to reflect desired dominance of now/current values over the original forecast data or vice-versa.

The `alpha` and `beta` values can be passed in the dictionary using the `runtimeparams` option during the call to `emhass`. If not passed they will both take the default 0.5 value. These values should be fixed following your own analysis of how much weight you want to put on measured values to be used as the persistence forecast. This will also depend on your fixed optimization time step. As a default, they will be at 0.5, but if you want to give more weight to measured persistence values, then you can try lower $\alpha$ and rising $\beta$, for example: `alpha=0.25`, `beta=0.75`. After this, you will need to check with the recorded history if these values fit your needs.

## References

- E. Lorenz, J. Kuhnert, A. Hammer, D. Heinemann, Photovoltaic (PV) power predictions with PV measurements, satellite data and numerical weather predictions. Presented at CM2E, Energy & Environment
Symposium, Martinique, 2014.

- Maimouna Diagne, Mathieu David, Philippe Lauret, John Boland, NicolasSchmutz, Review of solar irradiance forecasting methods and a proposition for small-scale insular grids. Renewable and Sustainable Energy Reviews 27 (2013) 65–76.

- Bryan Lima, Sercan O. Arik, Nicolas Loeff, Tomas Pfister, Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. arXiv:1912.09363v3 [stat.ML] 27 Sep 2020.
