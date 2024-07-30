# Deferrable load thermal model

EMHASS supports defining a deferrable load as a thermal model.
This is useful to control thermal equipment such as heaters, heat pumps, air conditioners, etc.
The advantage of using this approach is that you will be able to define your desired room temperature just as you will do with your real equipment thermostat.
Then EMHASS will deliver the operating schedule to maintain that desired temperature while minimizing the energy bill and taking into account the forecasted outdoor temperature.

A big thanks to @werdnum for proposing this model and the initial code for implementing this.

## The thermal model

The thermal model implemented in EMHASS is a linear model represented by the following equation:

$$
    T_{in}^{pred}[k+1] = T_{in}^{pred}[k] + P_{def}[k]\frac{\alpha_h\Delta t}{P_{def}^{nom}}-(\gamma_c(T_{in}^{pred}[k] - T_{out}^{fcst}[k]))
$$

where $k$ is each time instant, $T_{in}^{pred}$ is the indoor predicted temperature, $T_{out}^{fcst}$ is the outdoor forecasted temperature and $P_{def}$ is the deferrable load power.

In this model we can see two main configuration parameters:
- The heating rate $\alpha_h$ in degrees per hour.
- The cooling constant $\gamma_c$ in degrees per hour per degree of cooling.

These parameters are defined according to the thermal characteristics of the building/house.
It was reported by @werdnum, that values of $\alpha_h=5.0$ and $\gamma_c=0.1$ were reasonable in his case. 
Of course, these parameters should be adapted to each use case. This can be done with historical values of the deferrable load operation and the different temperatures (indoor/outdoor).

The following diagram tries to represent an example behavior of this model:

![](./images/thermal_load_diagram.svg)

## Implementing the model

To implement this model we need to provide a configuration for the discussed parameters and the input temperatures. You need to pass in the start temperature, the desired room temperature per timestep, and the forecasted outdoor temperature per timestep.

We will control this by using data passed at runtime.
The first step will be to define a new entry `def_load_config`, this will be used as a dictionary to store any needed special configuration for each deferrable load.

For example, if we have just **two** deferrable loads and the **second** load is a **thermal load** then we will define `def_load_config` as:
```
'def_load_config': {
    {},
    {'thermal_config': {
        'heating_rate': 5.0,
        'cooling_constant': 0.1,
        'overshoot_temperature': 24.0,
        'start_temperature': 20,
        'desired_temperatures': [...]
    }}
}
```

Here the `desired_temperatures` is a list of float values for each time step.

Now we also need to define the other needed input, the `outdoor_temperature_forecast`, which is a list of float values. The list of floats for `desired_temperatures` and the list in `outdoor_temperature_forecast` should have proper lengths, if using MPC the length should be at least equal to the prediction horizon.

Here is an example modified from a working example provided by @werdnum to pass all the needed data at runtime.
This example is given for the following configuration: just one deferrable load (a thermal load), no PV, no battery, an MPC application, and pre-defined heating intervals times. 

```
rest_command:
  emhass_forecast:
    url: http://localhost:5000/action/naive-mpc-optim
    method: post
    timeout: 300
    payload: >
      {% macro time_to_timestep(time) -%}
        {{ (((today_at(time) - now()) / timedelta(minutes=30)) | round(0, 'ceiling')) % 48 }}
      {%- endmacro %}
      {%- set horizon = 24 -%}
      {%- set heated_intervals = [[time_to_timestep("06:30")|int, time_to_timestep("07:30")|int], [time_to_timestep("17:30")|int, time_to_timestep("23:00")|int]] -%}
      {%- set pv_power_forecast = namespace(all=[]) -%}
      {% for i in range(horizon) %}
        {%- set pv_power_forecast.all = pv_power_forecast.all + [ 0.0 ] -%}
      {% endfor %}
      {%- set load_power_forecast = namespace(all=[]) -%}
      {% for i in range(horizon) %}
        {%- set load_power_forecast.all = load_power_forecast.all + [ 0.0 ] -%}
      {% endfor %}
      {
        "prediction_horizon": {{ horizon }},
        "load_cost_forecast": {{ (state_attr('sensor.electricity_price_forecast', 'forecasts') | map(attribute='currency_per_kWh') | list)[:horizon] | tojson }},
        "pv_power_forecast": {{ (pv_power_forecast.all)[:horizon] | tojson }},
        "load_power_forecast": {{ (load_power_forecast.all)[:horizon] | tojson }},
        "def_load_config": [
          {"thermal_config": {
            "heating_rate": 5.0,
            "cooling_constant": 0.1,
            "overshoot_temperature": {{ (states('sensor.my_room_temperature') | float) + 3.0 }},
            "start_temperature": {{ states('sensor.my_room_temperature') }},
            "desired_temperatures": [
              {%- set comma = joiner(", ") -%}
              {%- for i in range(horizon) -%}
                {%- set timestep = i -%}
                {{ comma() }}
                {% for interval in heated_intervals if timestep >= interval[0] and timestep <= interval[1] %}
                21.0
                {%- else -%}
                15.0
                {%- endfor %}
              {%- endfor %}
            ]}
          }
        ],
        "outdoor_temperature_forecast": {{ ((state_attr("sensor.weather_hourly", "forecast") | map(attribute="temperature") | list)[:horizon] | tojson) }}
      }
```

For the data publish command we need to provide the information about which deferrable loads are thermal loads.
In the previous example with just one thermal load, the working example for a publish command will be:
```
shell_command:
  publish_data: 'curl -i -H "Content-Type: application/json" -X POST -d ''{"def_load_config": [{"thermal_config": {}}]}'' http://localhost:5000/action/publish-data'
```
As we can see the thermal configuration can be left empty as what is needed is the `thermal_config` key. This is needed if using the add-on, for user using a `config_emhass.yaml` configuration file this is not needed if the `def_load_config` dictionary is directly defined there. 
For a configuration with **three** deferrable loads where the **second** load is a thermal load the payload would have been:
```
{"def_load_config": [{},{"thermal_config": {}},{}]}
```

After the publish command is executed a sensor with each deferrable load power will be published to Home Assistant as usual.
But for each thermal load also the predicted temperature will be published. For the example of just one deferrable and one thermal load this sensor is created: `sensor.temp_predicted0`.
This temperature sensor can then be used to control your climate entity by setting the temperature setpoint to this predicted room temperature.
