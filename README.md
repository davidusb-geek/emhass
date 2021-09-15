# emhass

EMHASS: Energy Management for Home Assistant

![](./docs/images/emhass_logo.png)

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
... or the compiled whl file:
```
pip install emhass-X.X.X-py3-none-any.whl
```
Clone this repository to obtain the example configuration files.
We will suppose that this repository is cloned to:
```
/home/user/emhass
```
This will be the root path containing the yaml configuration files (`config.yaml` and `secrets.yaml`) and the different needed folders (a `data` folder to store the optimizations results and a `scripts` folder containing the bash scripts described further below).

## Usage

To run a command simply use the `emhass` command followed by the needed arguments.
The available arguments are:
- `--action`: That is used to set the desired action, options are: `perfect-optim`, `dayahead-optim` and `publish-data`
- `--config`: Define path to the config.yaml file

For example, the following line command can be used to perform a day-ahead optimization task:
```
emhass --action 'dayahead-optim' --config '/home/user/emhass'

```
Before running any valuable command you need to modify the config.yaml and secrets.yaml files. 

## Home Assistant integration

To integrate with home assistant we will need to define some shell commands in the configuration.yaml file and some basic automations in the automations.yaml file.

In configuration.yaml:
```
shell_command:
  dayahead_optim: /home/user/emhass/scripts/dayahead_optim.sh
  publish_data: /home/user/emhass/scripts/publish_data.sh
```
And in automations.yaml:
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
emhass --action 'dayahead-optim' --config '/home/user/emhass'
```
And the file `publish_data.sh` with the following content:
```
#!/bin/bash
. /home/user/emhassenv/bin/activate
emhass --action 'publish-data' --config '/home/user/emhass'
```
Then specify user rights and make the files executables:
```
sudo chmod -R 755 /home/user/emhass/scripts/dayahead_optim.sh
sudo chmod -R 755 /home/user/emhass/scripts/publish_data.sh
sudo chmod +x /home/user/emhass/scripts/dayahead_optim.sh
sudo chmod +x /home/user/emhass/scripts/publish_data.sh
```
The final action will be to link a sensor value in Home Assistant to control the switch of a desired controllable load. For example imagine that I want to control my water heater and that the `publish-data` action is publishing the optimized value of a deferrable load that I have linked to my water heater desider behavior. In this case we could use an automation like this one below to control the desired real switch:
```
automation:
  trigger:
    - platform: numeric_state
      entity_id:
        - sensor.p_deferrable1
      above: 0
  action:
    - service: homeassistant.turn_on
      entity_id: switch.water_heater
```

## Development

To develop using Anaconda:
```
conda create --name emhass-dev python=3.8 pip=21.0.1
```
Then activate environment and install emhass using the provided setup.py file:
```
conda activate emhass-dev
python setup.py install
```
Add more packages if needed, this is optional if using spyder:
```
conda install spyder-kernels
```
Update the build package:
```
python3 -m pip install --upgrade build
```
And generate distribution archives with::
```
python3 -m build
```
In Spyder you can use CTRL+F6 and add the needed arguments in the "Command line options". The runfile command in the ipython console may look like this:
```
runfile('/home/user/emhass/src/emhass/command_line.py', args='--action "dayahead-optim" --config "/home/user/emhass"', wdir='/home/user/emhass/src/emhass')
```
To generate de documentation we will use Sphynx, the following packages are needed:
```
pip install sphinx==3.5.4 sphinx-rtd-theme==0.5.2 myst-parser==0.14.0
```
The actual documentation is generated using:
```
make clean
make html
sphinx-apidoc -o ./ ../src/emhass/
```

## TODO

- [ ] Implement an energy management with a Model Predictive Control approach. Consider implementing the receiding horizon approach. 
- [ ] Improve load forecasting using a time series forecast algorithm. Some tests were made with fbprophet but results are not completly satisfactory. The model needs some regressors for more accuracy.
- [ ] Introduce the modeling of constraints during optimization for a thermal energy storage
- [ ] EMHASS hass been tested in Home Assistan Core. It need to be tested on Home Assistant Operating System and Home Assistant Container. 
- [ ] Create an EMHASS add-on for even easier installation on Home Assistant Operating System and Home Assistant Supervised
- [ ] Package everything in a docker container
- [ ] Define the type of forecast that should be used from the configuration file
- [ ] Create a plotting class to visualize the optimization results