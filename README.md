<div align="center">
  <br>
  <img alt="EMHASS" src="./docs/images/emhass_logo.png" width="300px">
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
  <a href="https://app.codecov.io/gh/davidusb-geek/emhass">
    <img alt="Codecov" src="https://img.shields.io/codecov/c/github/davidusb-geek/emhass">
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

## Usage

To run a command simply use the `emhass` command followed by the needed arguments.
The available arguments are:
- `--action`: That is used to set the desired action, options are: `perfect-optim`, `dayahead-optim` and `publish-data`
- `--config`: Define path to the config.yaml file (including the yaml file itself)
- `--costfun`: Define the type of cost function, this is optional and the options are: `profit` (default), `cost`, `self-consumption`
- `--log2file`: Define if we should log to a file or not, this is optional and the options are: `True` or `False` (default)

For example, the following line command can be used to perform a day-ahead optimization task:
```
emhass --action 'dayahead-optim' --config '/home/user/emhass/config_emhass.yaml' --costfun 'profit'
```
Before running any valuable command you need to modify the `config_emhass.yaml` and `secrets_emhass.yaml` files. These files should contain the information adapted to your own system. To do this take a look at the special section for this in the [documentation](https://emhass.readthedocs.io/en/latest/config.html).

## Home Assistant integration

To integrate with home assistant we will need to define some shell commands in the `configuration.yaml` file and some basic automations in the `automations.yaml` file.

In `configuration.yaml`:
```
shell_command:
  dayahead_optim: /home/user/emhass/scripts/dayahead_optim.sh
  publish_data: /home/user/emhass/scripts/publish_data.sh
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
In these automations the optimization is performed everyday at 5:30am and the data is published every 5 minutes.
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

## Development

To develop using Anaconda:
```
conda create --name emhass-dev python=3.8 pip=21.0.1
```
Then activate environment and install the required packages using:
```
pip install -r docs/requirements.txt
```
Add `emhass` to the Python path using the path to `src`, for example:
```
/home/user/emhass/src
```
Update the build package:
```
python3 -m pip install --upgrade build
```
And generate distribution archives with:
```
python3 -m build
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
To generate de documentation we will use Sphynx, the following packages are needed:
```
pip install sphinx==3.5.4 sphinx-rtd-theme==0.5.2 myst-parser==0.14.0
```
The actual documentation is generated using:
```
sphinx-apidoc -o ./ ../src/emhass/
make clean
make html
```

## TODO
### New functionalities
- [x] Create a plotting script to visualize the optimization results.
- [x] Propose multiple types of cost functions: profit maximization, self-consumption maximization, etc.
- [x] Integrate the possibility of variable tariffs, for purshasing and selling energy to the grid.
- [ ] Implement an energy management with a Model Predictive Control approach. Consider implementing the receiding horizon approach.
- [ ] Introduce the modeling of constraints during optimization for a thermal energy storage.
- [ ] Add elasticity to LP formulation in case on infeasible solution.
### Related to forecasting improvement
- [x] Define the type of forecast that should be used from the configuration file.
- [x] Move get_load_unit_cost from optimization to forecast class: define forecast methods for load and PV production prices.
- [ ] Improve load forecasting using a time series forecast algorithm. Some tests were made with fbprophet but results are not completly satisfactory. The model needs some regressors for more accuracy.
- [ ] Test with LTSM with or without Autoencoders.
### Packaging, HA integration, testing
- [ ] EMHASS hass been tested in Home Assistan Core. It need to be tested on Home Assistant Operating System and Home Assistant Container. 
- [ ] Create an EMHASS add-on for even easier installation on Home Assistant Operating System and Home Assistant Supervised.
- [ ] Package everything in a docker container.
- [ ] Improve testing to be used with no running hass instance.
