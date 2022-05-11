<div align="center">
  <br>
  <img alt="EMHASS" src="https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/emhass_logo.png" width="300px">
  <h1>Energy Management for Home Assistant</h1>
  <strong></strong>
</div>
<br>
<p align="center">
  <a href="https://github.com/davidusb-geek/emhass/releases">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/davidusb-geek/emhass">
  </a>
  <a href="https://github.com/davidusb-geek/emhass/actions">
    <img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/davidusb-geek/emhass/Python%20package">
  </a>
  <a href="https://github.com/davidusb-geek/emhass/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/davidusb-geek/emhass">
  </a>
  <a href="https://pypi.org/project/emhass/">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/emhass">
  </a>
  <a href="https://pypi.org/project/emhass/">
    <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/emhass">
  </a>
  <a href="https://emhass.readthedocs.io/en/latest/">
    <img alt="Read the Docs" src="https://img.shields.io/readthedocs/emhass">
  </a>
</p>

EHMASS is a Python module designed to optimize your home energy interfacing with Home Assistant.

## Context

This module was conceived as an energy management optimization tool for residential electric power consumption and production systems.
The goal is to optimize the energy use in order to maximize autoconsumption.
The main study case is a household where we have solar panels, a grid connection and one or more controllable (deferrable) electrical loads. Including an energy storage system using batteries is also possible in the code.
The package is highly configurable with an object oriented modular approach and a main configuration file defined by the user.
EMHASS was designed to be integrated with Home Assistant, hence it's name. Installation instructions and example Home Assistant automation configurations are given below.

The main dependencies of this project are PVLib to model power from a PV residential installation and the PuLP Python package to perform the actual optimizations using the Linear Programming approach.

The complete documentation for this package is [available here](https://emhass.readthedocs.io/en/latest/).

## Installation

It is recommended to install on a virtual environment.
For this you will need `virtualenv`, install it using:
```
sudo apt install python3-virtualenv
```
Then create and activate the virtual environment:
```
virtualenv -p /usr/bin/python3 emhassenv
cd emhassenv
source bin/activate
```
Install using the distribution files:
```
python3 -m pip install emhass
```
Clone this repository to obtain the example configuration files.
We will suppose that this repository is cloned to:
```
/home/user/emhass
```
This will be the root path containing the yaml configuration files (`config_emhass.yaml` and `secrets_emhass.yaml`) and the different needed folders (a `data` folder to store the optimizations results and a `scripts` folder containing the bash scripts described further below).

To upgrade the installation in the future just use:
```
python3 -m pip install --upgrade emhass
```

### Using Docker

To install using docker you will need to build your image locally. For this clone this repository, setup your `config_emhass.yaml` file and use the provided make file with this command:
```
make -f deploy_docker.mk clean_deploy
```
Then load the image in the .tar file:
```
docker load -i <TarFileName>.tar
```
Finally launch the docker itself:
```
docker run -it --restart always -p 5000:5000 -e "LOCAL_COSTFUN=profit" -v $(pwd)/config_emhass.yaml:/app/config_emhass.yaml -v $(pwd)/secrets_emhass.yaml:/app/secrets_emhass.yaml --name DockerEMHASS <REPOSITORY:TAG>
```

### The EMHASS add-on

For Home Assistant OS and HA Supervised users, I've developed an add-on that will help you use EMHASS. The add-on is more user friendly as the configuration can be modified directly in the add-on options pane and also it exposes a web ui that can be used to inspect the optimization results and manually trigger a new optimization.

You can find the add-on with the installation instructions here: [https://github.com/davidusb-geek/emhass-add-on](https://github.com/davidusb-geek/emhass-add-on)

The add-on usage instructions can be found on the documentation pane of the add-on once installed or directly here: [EMHASS Add-on documentation](https://github.com/davidusb-geek/emhass-add-on/blob/main/emhass/DOCS.md)

These architectures are supported: `amd64`, `armv7` and `aarch64`.

## Usage

To run a command simply use the `emhass` command followed by the needed arguments.
The available arguments are:
- `--action`: That is used to set the desired action, options are: `perfect-optim`, `dayahead-optim`, `naive-mpc-optim` and `publish-data`
- `--config`: Define path to the config.yaml file (including the yaml file itself)
- `--costfun`: Define the type of cost function, this is optional and the options are: `profit` (default), `cost`, `self-consumption`
- `--log2file`: Define if we should log to a file or not, this is optional and the options are: `True` or `False` (default)
- `--params`: Configuration as JSON. 
- `--runtimeparams`: Data passed at runtime. This can be used to pass you own forecast data to EMHASS.
- `--version`: Show the current version of EMHASS.

For example, the following line command can be used to perform a day-ahead optimization task:
```
emhass --action 'dayahead-optim' --config '/home/user/emhass/config_emhass.yaml' --costfun 'profit'
```
Before running any valuable command you need to modify the `config_emhass.yaml` and `secrets_emhass.yaml` files. These files should contain the information adapted to your own system. To do this take a look at the special section for this in the [documentation](https://emhass.readthedocs.io/en/latest/config.html).

If using the add-on or the standalone docker installation, it exposes a simple webserver on port 5000. You can access it directly using your brower, ex: http://localhost:5000.

With this web server you can perform RESTful POST commands on one ENDPOINT called `action` with two main options:

- A POST call to `action/perfect-optim` to perform a perfect optimization task on the historical data.
- A POST call to `action/dayahead-optim` to perform a day-ahead optimization task of your home energy.
- A POST call to `action/naive-mpc-optim` to perform a naive Model Predictive Controller optimization task. If using this option you will need to define the correct `runtimeparams` (see further below).
- A POST call to `action/publish-data` to publish the optimization results data for the current timestamp.

A `curl` command can then be used to launch an optimization task like this: `curl -i -H "Content-Type: application/json" -X POST -d '{}' http://localhost:5000/action/dayahead-optim`.

## Home Assistant integration

To integrate with home assistant we will need to define some shell commands in the `configuration.yaml` file and some basic automations in the `automations.yaml` file.

In `configuration.yaml`:
```
shell_command:
  dayahead_optim: /home/user/emhass/scripts/dayahead_optim.sh
  publish_data: /home/user/emhass/scripts/publish_data.sh
```
If using the add-on you can use this instead on the `configuration.yaml` file:
```
shell_command:
  dayahead_optim: curl -i -H "Content-Type: application/json" -X POST -d '{}' http://localhost:5000/action/dayahead-optim
  publish_data: curl -i -H "Content-Type: application/json" -X POST -d '{}' http://localhost:5000/action/publish-data 
```

And in `automations.yaml`:
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
In these automations the day-ahead optimization is performed everyday at 5:30am and the data is published every 5 minutes.

Create the file `dayahead_optim.sh` with the following content:
```
#!/bin/bash
. /home/user/emhassenv/bin/activate
emhass --action 'dayahead-optim' --config '/home/user/emhass/config_emhass.yaml'
```
And the file `publish_data.sh` with the following content:
```
#!/bin/bash
. /home/user/emhassenv/bin/activate
emhass --action 'publish-data' --config '/home/user/emhass/config_emhass.yaml'
```
Then specify user rights and make the files executables:
```
sudo chmod -R 755 /home/user/emhass/scripts/dayahead_optim.sh
sudo chmod -R 755 /home/user/emhass/scripts/publish_data.sh
sudo chmod +x /home/user/emhass/scripts/dayahead_optim.sh
sudo chmod +x /home/user/emhass/scripts/publish_data.sh
```
The final action will be to link a sensor value in Home Assistant to control the switch of a desired controllable load. For example imagine that I want to control my water heater and that the `publish-data` action is publishing the optimized value of a deferrable load that I have linked to my water heater desired behavior. In this case we could use an automation like this one below to control the desired real switch:
```
automation:
  trigger:
    - platform: numeric_state
      entity_id:
        - sensor.p_deferrable1
      above: 0.1
  action:
    - service: homeassistant.turn_on
      entity_id: switch.water_heater
```
A second automation should be used to turn off the switch:
```
automation:
  trigger:
    - platform: numeric_state
      entity_id:
        - sensor.p_deferrable1
      below: 0.1
  action:
    - service: homeassistant.turn_off
      entity_id: switch.water_heater
```
The `publish-data` command will push to Home Assistant the optimization results for each deferrable load defined in the configuration. For example if you have defined two deferrable loads, then the command will publish `sensor.p_deferrable1` and `sensor.p_deferrable2` to Home Assistant. When the `dayahead-optim` is launched, after the optimization, a csv file will be saved on disk. The `publish-data` command will load the latest csv file and look for the closest timestamp that match the current time using the `datetime.now()` method in Python. This means that if EMHASS is configured for 30min time step optimizations, the csv will be saved with timestamps 00:00, 00:30, 01:00, 01:30, ... and so on. If the current time is 00:05, then the closest timestamp of the optimization results that will be published is 00:00. If the current time is 00:25, then the closest timestamp of the optimization results that will be published is 00:30.


## Forecast data

In EMHASS we have basically 4 forecasts to deal with:

- PV power production forecast (internally based on the weather forecast and the characteristics of your PV plant). This is given in Watts.

- Load power forecast: how much power your house will demand on the next 24h. This is given in Watts.

- Load cost forecast: the price of the energy from the grid on the next 24h. This is given in EUR/kWh.

- PV production selling price forecast: at what price are you selling your excess PV production on the next 24h. This is given in EUR/kWh.

Maybe the hardest part is the load data: parameter `var_load` in the configuration file. As we want to optimize the household energies, when need to forecast the load power conumption. The default method for this is a naive approach using 1-day persistence, this mean that the load data variable should not contain the data from the deferrable loads themselves. For example, lets say that you set your deferrable load to be the washing machine. The variable that you should enter in EMHASS will be: `var_load: 'sensor.power_load_no_var_loads'` and `sensor_power_load_no_var_loads = sensor_power_load - sensor_power_washing_machine`. This is supposing that the overall load of your house is contained in variable: `sensor_power_load`. The sensor `sensor_power_load_no_var_loads` can be easily created with a new template sensor in Home Assistant.

### Passing your own forecast data

It is possible to provide EMHASS with your own forecast data. For this just add the data as list of values to a data dictionnary during the call to `emhass` using the `runtimeparams` option. 

For example:
```
emhass --action 'dayahead-optim' --config '/home/user/emhass/config_emhass.yaml' --runtimeparams '{"pv_power_forecast":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75, 804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}'
```
If using the add-on you can pass this data as list of values to the data dictionnary during the `curl` POST:
```
curl -i -H "Content-Type: application/json" -X POST -d '{"pv_power_forecast":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75, 804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}' http://localhost:5000/action/dayahead-optim
```

The possible dictionnary keys to pass data are:

- `pv_power_forecast` for the PV power production forecast.

- `load_power_forecast` for the Load power forecast.

- `load_cost_forecast` for the Load cost forecast.

- `prod_price_forecast` for the PV production selling price forecast.

### A naive Model Predictive Controller

A MPC controller was introduced in v0.3.0. This an informal/naive representation of a MPC controller. 

A MPC controller performs the following actions:

- Set the prediction horizon and receiding horizon parameters.
- Perform an optimization on the prediction horizon.
- Apply the first element of the obtained optimized control variables.
- Repeat at a relatively high frequency, ex: 5 min.

This is the receiding horizon principle.

When applyin this controller, the following `runtimeparams` should be defined:

- `prediction_horizon` for the MPC prediction horizon. Fix this at at least 5 times the optimization time step.

- `soc_init` for the initial value of the battery SOC for the current iteration of the MPC. 

- `soc_final` for the final value of the battery SOC for the current iteration of the MPC. 

- `def_total_hours` for the list of deferrable loads functioning hours. These values can decrease as the day advances to take into account receidding horizon daily energy objectives for each deferrable load.

A correct call for a MPC optimization should look like:

```
curl -i -H "Content-Type: application/json" -X POST -d '{"pv_power_forecast":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75, 804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "prediction_horizon":10, "soc_init":0.5,"soc_final":0.6,"def_total_hours":[1,3]}' http://localhost:5000/action/naive-mpc-optim
```

## Development

Create a developer environment:
```
virtualenv -p /usr/bin/python3 emhass-dev
```
To develop using Anaconda use (pick the correct Python and Pip versions):
```
conda create --name emhass-dev python=3.8 pip=21.0.1
```
Then activate environment and install the required packages using:
```
pip install -r requirements.txt
```
Add `emhass` to the Python path using the path to `src`, for example:
```
/home/user/emhass/src
```
If working on linux we can add these lines to the `~/.bashrc` file:
```
# Python modules
export PYTHONPATH="${PYTHONPATH}:/home/user/emhass/src"
```
Don't foget to source the `~/.bashrc` file:
```
source ~/.bashrc
```
Update the build package:
```
python3 -m pip install --upgrade build
```
And generate distribution archives with:
```
python3 -m build
```
Or with:
```
python3 setup.py build bdist_wheel
```
Create a new tag version:
```
git tag vX.X.X
git push origin --tags
```
Upload to pypi:
```
twine upload dist/*
```

## Troubleshooting

Some problems may arise from solver related issues in the Pulp package. It was found that for arm64 architectures (ie. Raspberry Pi4, 64 bits) the default solver is not avaliable. A workaround is to install a new solver. The `glpk` solver is an option and can be installed with:
```
sudo apt-get install glpk-utils
```
After this it should be available for use and EMHASS can use it as a fallback option.


## License

MIT License

Copyright (c) 2021-2022 David HERNANDEZ

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
