# Example configurations

In this section example configurations are presented as study cases using real data.

The example configuration all start with the default configuration that is located at: [https://github.com/davidusb-geek/emhass/tree/master/src/emhass/data/config_defaults.json](https://github.com/davidusb-geek/emhass/tree/master/src/emhass/data/config_defaults.json)

We will also use a JSON format to store some secret parameters.
You can add your own secrets parameters to retrieve your own data and reproduce the results presented on this page.
A template for the secrets file is found here: [https://github.com/davidusb-geek/emhass/options.json](https://github.com/davidusb-geek/emhass/options.json)

## First test system: a simple system with no PV and two deferrable loads

In this example, we will consider a simple system with no PV installation and just two deferrable loads that we want to optimize their schedule.

For this, the following parameter can be set in the `options.json` file: `solar_forecast_kwp: 0`. Also, we will set the PV forecast method to `method='solar.forecast'`. For this we can modify the original default configuration file `config_defaults.json` and save for example as `config_emhass.json`.
This is a simple way to just set a vector with zero values on the PV forecast power, emulating the case where there is no PV installation. The other values on the configuration file are set to their default values.

### Day-ahead optimization

Let's perform a day-ahead optimization task on this simple system. We want to schedule our two deferrable loads.

For this, we use the following command (this example is using the legacy EMHASS Python module command line, check the documentation for the alternative REST commands):
```
emhass --action 'dayahead-optim' --config '/home/user/emhass/config_emhass.json' --costfun 'profit'
```

The retrieved input forecasted powers are shown below:

![](./images/inputs_dayahead.png)

Finally, the optimization results are:

![](./images/optim_results_defLoads_dayaheadOptim.png)

For this system, the total value of the obtained cost function is **-5.38 EUR**. 

## A second test system: a 5kW PV installation and two deferrable loads

Let's add a 5 kWp solar production with two deferrable loads. No battery is considered for now. In this case the configuration used is exactly the same as the default configuration proposed with EMHASS in the `config_defaults.json` file. 

We will first consider a perfect optimization task, to obtain the optimization results with perfectly known PV production and load power values for the last week.

### Perfect optimization

Let's perform a 7-day historical data optimization.

For this, we use the following command (using the legacy EMHASS Python module command line):
```
emhass --action 'perfect-optim' --config '/home/user/emhass/config_emhass.json' --costfun 'profit'
```

The retrieved input powers are shown below:

![](./images/inputs_power.png)

The input load cost and PV production selling prices are presented in the following figure:

![](./images/inputs_cost_price.png)

Finally, the optimization results are:

![](./images/optim_results_PV_defLoads_perfectOptim.png)

For this 7-day period, the total value of the cost function was **-26.23 EUR**. 

### Day-ahead optimization

As with the simple system, we will now perform a day-ahead optimization task. We use again the `dayahead-optim` action or endpoint.

The optimization results are:

![](./images/optim_results_PV_defLoads_dayaheadOptim.png)

For this system, the total value of the obtained cost function is **-1.56 EUR**. We can note the important improvement in the cost function value when adding a PV installation.

## A third test system: a 5kW PV installation, a 5kWh battery and two deferrable loads

Now we will consider a complete system with PV and added batteries. To add the battery we will set `set_use_battery: true` in the `config_emhass.json` file.

In this case, we want to schedule our deferrable loads but also the battery charge/discharge. We use again the `dayahead-optim` action or endpoint.

The optimization results are:

![](./images/optim_results_PV_Batt_defLoads_dayaheadOptim.png)

The battery state of charge plot is shown below:

![](./images/optim_results_PV_Batt_defLoads_dayaheadOptim_SOC.png)

For this system, the total value of the obtained cost function is **-1.23 EUR**, a substantial improvement when adding a battery.

## Configuration example to pass data at runtime

As we showed in the forecast module section, we can pass our own forecast data using lists of values passed at runtime using templates. However, it is possible to also pass other data during runtime to automate energy management.

For example, let's suppose that for the default configuration with two deferrable loads, we want to correlate and control them to the outside temperature. This will be used to build a list of the total number of hours for each deferrable load (`operating_hours_of_each_deferrable_load`). In this example, the first deferrable load is a water heater and the second is the pool pump.

We will begin by defining a temperature sensor on a 12 hours sliding window using the filter platform for the outside temperature:
```
  - platform: filter
    name: "Outdoor temperature mean over last 12 hours"
    entity_id: sensor.temp_air_out
    filters:
      - filter: time_simple_moving_average
        window_size: "12:00"
        precision: 0
```
Then we will use a template sensor to build our list of the total number of hours for each deferrable load:
```
  - platform: template
    sensors:
      list_operating_hours_of_each_deferrable_load:
        value_template: >-
          {% if states("sensor.outdoor_temperature_mean_over_last_12_hours") < "10" %}
            {{ [5, 0] | list }}
          {% elif states("sensor.outdoor_temperature_mean_over_last_12_hours") >= "10" and states("sensor.outdoor_temperature_mean_over_last_12_hours") < "15" %}
            {{ [4, 0] | list }}
          {% elif states("sensor.outdoor_temperature_mean_over_last_12_hours") >= "15" and states("sensor.outdoor_temperature_mean_over_last_12_hours") < "20" %}
            {{ [4, 6] | list }}
          {% elif states("sensor.outdoor_temperature_mean_over_last_12_hours") >= "20" and states("sensor.outdoor_temperature_mean_over_last_12_hours") < "25" %}
            {{ [3, 9] | list }}
          {% else %}
            {{ [3, 12] | list }}
          {% endif %}
```
The values for the total number of operating hours were tuned by trial and error throughout a whole year. These values work fine for a 3000W water heater (the first value in the list) and a 750W pool pump (the second value in the list).

Finally, my two shell commands for EMHASS will look like this:
```
shell_command:
  dayahead_optim: "curl -i -H \"Content-Type: application/json\" -X POST -d '{\"operating_hours_of_each_deferrable_load\":{{states('sensor.list_operating_hours_of_each_deferrable_load')}}}' http://localhost:5000/action/dayahead-optim"
  publish_data: "curl -i -H \"Content-Type: application/json\" -X POST -d '{}' http://localhost:5000/action/publish-data"
```
The dedicated automation for these shell commands can be for example:
```
- alias: EMHASS day-ahead optimization
  trigger:
    platform: time
    at: '05:30:00'
  action:
  - service: shell_command.dayahead_optim
- alias: EMHASS publish data
  trigger:
  - minutes: /5
    platform: time_pattern
  action:
  - service: shell_command.publish_data
```

## Some real forecast data

The real implementation of EMHASS and its efficiency depends on the quality of the forecasted PV power production and the house load consumption.

Here is an extract of the PV power production forecast with the default PV forecast method from EMHASS: a web scraping of the clearoutside page based on the defined lat/lon location of the system. These are the forecast results of the GFS model compared with the real PV-produced data for a 4-day period. 

![](./images/forecasted_PV_data.png)
