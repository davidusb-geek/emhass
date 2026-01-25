# Home Assistant Automations

To automate EMHASS with Home Assistant, we will need to define some shell commands in the Home Assistant `configuration.yaml` file and some basic automations in the `automations.yaml` file.
In the next few paragraphs, we are going to consider the `dayahead-optim` optimization strategy, which is also the first that was implemented, and we will also cover how to publish the optimization  results.  
Additional optimization strategies were developed later, that can be used in combination with/replace the `dayahead-optim` strategy, such as MPC, or to expand the functionalities such as the Machine Learning method to predict your household consumption. Each of them has some specificities and features and will be considered in dedicated sections.

## Dayahead Optimization - Method 1) Add-on and docker standalone

We can use the `shell_command` integration in `configuration.yaml`:
```yaml
shell_command:
  dayahead_optim: "curl -i -H \"Content-Type:application/json\" -X POST -d '{}' http://localhost:5000/action/dayahead-optim"
  publish_data: "curl -i -H \"Content-Type:application/json\" -X POST -d '{}' http://localhost:5000/action/publish-data"
```
An alternative that will be useful when passing data at runtime (see dedicated section), we can use the the `rest_command` instead:
```yaml
rest_command:
  url: http://127.0.0.1:5000/action/dayahead-optim
  method: POST
  headers:
    content-type: application/json
  payload: >-
    {}
```

## Dayahead Optimization - Method 2) Legacy method using a Python virtual environment

In `configuration.yaml`:
```yaml
shell_command:
  dayahead_optim: ~/emhass/scripts/dayahead_optim.sh
  publish_data: ~/emhass/scripts/publish_data.sh
```
Create the file `dayahead_optim.sh` with the following content:
```bash
#!/bin/bash
. ~/emhassenv/bin/activate
emhass --action 'dayahead-optim' --config ~/emhass/config.json
```
And the file `publish_data.sh` with the following content:
```bash
#!/bin/bash
. ~/emhassenv/bin/activate
emhass --action 'publish-data' --config ~/emhass/config.json
```
Then specify user rights and make the files executables:
```bash
sudo chmod -R 755 ~/emhass/scripts/dayahead_optim.sh
sudo chmod -R 755 ~/emhass/scripts/publish_data.sh
sudo chmod +x ~/emhass/scripts/dayahead_optim.sh
sudo chmod +x ~/emhass/scripts/publish_data.sh
```

## Common for any installation method

### Options 1, Home Assistant automate publish

In `automations.yaml`:
```yaml
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
In these automations the day-ahead optimization is performed once a day, every day at 5:30am, and the data *(output of automation)* is published every 5 minutes.

### Option 2, EMHASS automated publish 

In `automations.yaml`:
```yaml
- alias: EMHASS day-ahead optimization
  trigger:
    platform: time
    at: '05:30:00'
  action:
  - service: shell_command.dayahead_optim
  - service: shell_command.publish_data
```
in configuration page/`config.json` 
```json
"method_ts_round": "first"
"continual_publish": true
```
In this automation, the day-ahead optimization is performed once a day, every day at 5:30am. 
If the `optimization_time_step` parameter is set to `30` *(default)* in the configuration, the results of the day-ahead optimization will generate 48 values *(for each entity)*, a value for every 30 minutes in a day *(i.e. 24 hrs x 2)*.

Setting the parameter `continual_publish` to `true` in the configuration page will allow EMHASS to store the optimization results as entities/sensors into separate json files. `continual_publish` will periodically (every `optimization_time_step` amount of minutes) run a publish, and publish the optimization results of each generated entities/sensors to Home Assistant. The current state of the sensor/entity being updated every time publish runs, selecting one of the 48 stored values, by comparing the stored values' timestamps, the current timestamp and [`'method_ts_round': "first"`](https://emhass.readthedocs.io/en/latest/publish_data.html#the-publish-data-specificities) to select the optimal stored value for the current state.

option 1 and 2 are very similar, however, option 2 (`continual_publish`) will require a CPU thread to constantly be run inside of EMHASS, lowering efficiency. The reason why you may pick one over the other is explained in more detail below in [continual_publish](https://emhass.readthedocs.io/en/latest/publish_data.html#continual-publish-emhass-automation).

Lastly, we can link an EMHASS published entity/sensor's current state to a Home Assistant entity on/off switch, controlling a desired controllable load. 
For example, imagine that I want to control my water heater. I can use a published `deferrable` EMHASS entity to control my water heater's desired behavior. In this case, we could use an automation like the below, to control the desired water heater on and off:
  
on:
```yaml
automation:
- alias: Water Heater Optimized ON
  trigger:
  - minutes: /5
    platform: time_pattern
  condition:
  - condition: numeric_state
    entity_id: sensor.p_deferrable0
    above: 0.1
  action:
    - service: homeassistant.turn_on
      entity_id: switch.water_heater_switch
```
off:
```yaml
automation:
- alias: Water Heater Optimized OFF
  trigger:
  - minutes: /5
    platform: time_pattern
  condition:
  - condition: numeric_state
    entity_id: sensor.p_deferrable0
    below: 0.1
  action:
    - service: homeassistant.turn_off
      entity_id: switch.water_heater_switch
```
These automations will turn on and off the Home Assistant entity `switch.water_heater_switch` using the current state from the EMHASS entity `sensor.p_deferrable0`. `sensor.p_deferrable0`  being the entity generated from the EMHASS day-ahead optimization and published by examples above. The `sensor.p_deferrable0` entity's current state is updated every 30 minutes (or `optimization_time_step` minutes) via an automated publish option 1 or 2. *(selecting one of the 48 stored data values)*