# Changelog

## 0.11.0 - 2024-10-25

This version marks huge improvement works by @GeoDerp aiming to simplfy the intial and normal setup of EMHASS. The workflow for setting the EMHASS configuration regardless of the installation method has now been centralized on the single `config.json` file. The webserver has now a configuration tab that can be used to to modify and save the `config.json` file.

The complete discussion of the changes on this thread:
[https://github.com/davidusb-geek/emhass/pull/334](https://github.com/davidusb-geek/emhass/pull/334)

### Automatic version bot improvements
- Bump h5py from 3.11.0 to 3.12.1
- Bump markupsafe from 2.1.5 to 3.0.2

### Fix
- Revert to myst-parser==3.0.1 to solve documentation compilation failures

## 0.10.6 - 2024-07-14
### Fix
- Fixed bug on predicted room temeprature publish, wrong key on DataFrame

## 0.10.5 - 2024-07-12
### Improvement
- Added support for pubishing thermal load data, namely the predicted room temperature

## 0.10.4 - 2024-07-10
### Improvement
- Added a new thermal modeling, see the new section in the documentation for help to implement this of model for thermal deferrable loads
- Improved documentation

## 0.10.3 - 2024-07-06
### Improvement
- Added improved support for `def_start_penalty` option
- Improved documentation

## 0.10.2 - 2024-07-06
### Improvement
- Weather forecast caching and Solcast method fix by @GeoDerp
- Added a new configuration parameter to control wether we compute PV curtailment or not
- Added hybrid inverter to data publish
- It is now possible to pass these battery parameters at runtime: `SOCmin`, `SOCmax`, `Pd_max` and `Pc_max`
### Fix
- Fixed problem with negative PV forecast values in optimization.py, by @GeoDerp

## 0.10.1 - 2024-06-03
### Fix
- Fixed PV curtailment maximum possible value constraint
- Added PV curtailement to variable to publish to HA

## 0.10.0 - 2024-06-02
### BREAKING CHANGE
- In this new version we have added support for PV curtailment computation. While doing this the nominal PV peak power is needed. The easiest way find this information is by directly using the `inverter_model` defined in the configuration. As this is needed in the optimization to correctly compute PV curtailment, this parameter need to be properly defined for your installation. Before this chage this parameter was only needed if using the PV forecast method `scrapper`, but now it is not optional as it is directly used in the optimization. 
Use the dedicated webapp to find the correct model for your inverter, if you cannot find your exact brand/model then just pick an inverter with the same nominal power as yours: [https://emhass-pvlib-database.streamlit.app/](https://emhass-pvlib-database.streamlit.app/)
### Improvement
- Added support for hybrid inverters and PV curtailment computation
- Implemented a new `continual_publish` service that avoid the need of setting a special automation for data publish. Thanks to @GeoDerp
- Implement a deferrable load start penalty functionality. Thanks to @werdnum
  - This feature also implement a `def_current_state` that can be passed at runtime to let the optimization consider that a deferrable load is currently scheduled or under operation when launching the optimization task
### Fix
- Fixed forecast methods to treat delta_forecast higher than 1 day
- Fixed solar.forecast wrong interpolation of nan values

## 0.9.1 - 2024-05-13
### Fix
- Fix patch for issue with paths to modules and inverters database
- Fixed code formatting, or at least trying to keep a unique format

## 0.9.0 - 2024-05-10
### Improvement
- On this new version we now have a new method to train a regression model using Scikit-Learn methods. This is the contribution of @gieljnssns. Check the dedicated section the documentation to this new feature: [https://emhass.readthedocs.io/en/latest/mlregressor.html](https://emhass.readthedocs.io/en/latest/mlregressor.html)
- Again another bunch of nice improvements by @GeoDerp:
  - Added Dictionary var containing EMHASS paths
  - MLForcaster error suppression
  - Add `freq` as runtime parameter
  - Improved documentation added README buttons
- Bumping dependencies:
  - Bump h5py from 3.10.0 to 3.11.0
  - Bump myst-parser from 2.0.0 to 3.0.1
  - Bump skforecast from 0.11.0 to 0.12.0

## 0.8.6 - 2024-04-07
### Fix
- Fixed bug from forecast out method related to issue 240
- Fix patch for some issues with package file paths

## 0.8.5 - 2024-04-01
### Improvement
- Simplified fetch urls to relatives
- Improved code for passed forecast data error handling in utils.py
- Added new tests for forecast longer than 24h by changing parameter `delta_forecast`
- Added new files for updated PV modules and inverters database for use with PVLib
- Added a new webapp to help configuring modules and inverters: [https://emhass-pvlib-database.streamlit.app/](https://emhass-pvlib-database.streamlit.app/)
- Added a new `P_to_grid_max` variable, different from the current `P_from_grid_max` option
### Fix
- style.css auto format and adjusted table styling
- Changed pandas datetime rounding to nonexistent='shift_forward' to help survive DST change
- Dropped support for Python 3.9

## 0.8.4 - 2024-03-13
### Improvement
- Improved documentation
- Improved logging errors on missing day info
### Fix
- Missing round treatment for DST survival in utils.py 
- Webui large icons fix

## 0.8.3 - 2024-03-11
### Fix
- Fixed web_server options_json bug in standalone fix 

## 0.8.2 - 2024-03-10
### Improvement
- Proposed a new solution to survive DST using special option of Pandas `round` method
- Added option to `web_server` to init `data_path` as an options param
- Styling docs and html files on webui
- Advanced and basic pages improvements on webui
### Fix
- Fixed support for ARM achitectures

## 0.8.1 - 2024-02-28
### Improvement
- Improved documentation
### Fix
- Persistent data storage fix
- Docker Standalone Publish Workspace Fix

## 0.8.0 - 2024-02-25
### Improvement
- Thanks to the great work from @GeoDerp we now have a unified/centralized Dockerfile that allows for testing different installation configuration methods in one place. This greatly helps testing, notably emulating the add-on environment. This will improve overall testing for both teh core code and the add-on. Again many thanks!
- There were also a lot of nice improveements from @GeoDerp to the webui, namely: styling, dynamic table, optimization feedback after button press, logging, a new clear button, etc.
- From now on we will unify the semantic versioning for both the main core code and the add-on.

## 0.7.8 - 2024-02-18
### Improvement
Added some nice logging functionalities and responsiveness on the webui.
Thanks to @GeoDerp for this great work!
- new actionLogs.txt is generated in datapath storing sessions app.logger info
- on successful html button press, fetch is called to get html containing latest table data
- on html button press, If app.logger ERROR is present, send action log back and present on page.

## 0.7.7 - 2024-02-10
### Improvement
- Bumped the webui. Some great new features and styling. Now it is possible to pass data directly as lsit of values when using the buttons in the webui. Thanks to @GeoDerp
- Added two additional testing environment options. Thanks to @GeoDerp
### Fix
- Bump markupsafe from 2.1.4 to 2.1.5

## 0.7.6 - 2024-02-06
### Fix
- Fixed number of startups constraint for deferrable load at the begining of the optimization period
- Fixed list of bools from options.json
- Fixed some testing and debugging scripts

## 0.7.5 - 2024-02-04
### Fix
- Fixing again "perform_backtest": "false" has no effect

## 0.7.4 - 2024-02-04
### Fix
- Fixed broken build params method. Reverting back to alternate PR from @GeoDerp

## 0.7.3 - 2024-02-04
### Fix
- Fixed bug when booleans, solving "perform_backtest": "false" has no effect
- Refactored util.py method to handle optional parameters
- Updated web server, solving runtime issues
- Solved issue passing solcast and solar.forecast runtime params 
- Updated documentation requirements

## 0.7.2 - 2024-01-30
### Fix
- Patched new version wer server issues of missing list types

## 0.7.1 - 2024-01-29
### Fix
- Patched new version wer server issues accessing correct paths

## 0.7.0 - 2024-01-28
### Improvement
- Added a new feature to provide operating time windows for deferrable loads. Thanks to @michaelpiron
- Added lots of new options to be configured by the user. Thanks to @GeoDerp
- Updated stylesheet with mobile & dark theme by @GeoDerp
- Improved launch.json to fully test EMHASS on different configurations. Thanks to @GeoDerp
- Added new script to debug and develop new time series clustering feature
- Improved documentation. Thanks to @g1za
### Fix
- Updated github workflow actions/checkout to v4 and actions/setup-python to v5
- Changed default values for weight_battery_discharge and weight_battery_charge to zero
- Renamed classes to conform to PEP8
- Bump markupsafe from 2.1.3 to 2.1.4 

## 0.6.2 - 2024-01-04
### Improvement
- Added option to pass additional weight for battery usage
- Improved coverage
### Fix
- Updated optimization constraints to solve conflict for `set_def_constant` and `treat_def_as_semi_cont` cases

## 0.6.1 - 2023-12-18
### Fix
- Patching EMHASS for Python 3.11. New explicit dependecy h5py==3.10.0
- Updated Dockerfile to easily test add-on build

## 0.6.0 - 2023-12-16
### Improvement
- Now Python 3.11 is fully supported, thanks to @pail23
- We now publish the optimization status on sensor.optim_status
- Bumped setuptools, skforecast, numpy, scipy, pandas
- A good bunch of documentation improvements thanks to @g1za
- Improved code coverage (a little bit ;-)
### Fix
- Some fixes managing time zones, thanks to @pail23
- Bug fix on grid cost function equation, thanks to @michaelpiron
- Applying a first set of fixes proposed by @smurfix:
  - Don't ignore HTTP errors
  - Handle missing variable correctly
  - Slight error message improvement
  - Just use the default solver
  - Get locations from environment in non-app mode
  - Tolerate running directly from source

## 0.5.1 - 2023-10-19
### Improvement
- Improved documentation, thanks to @g1za
- Bumped skforecast to 0.10.1
- Added a new initial script for exploration of time series clustering. This will one day replace the need to configure the house load sensor with substracted deferrable load consumption
### Fix
- Updated automated tesing, dropped support for Python 3.8

## 0.5.0 - 2023-09-03
### Improvement
- Finally added support for ingress thanks to the work from @siku2
- Added a devcontainer and pave the way for ingress access
### Fix
- Added some code to fix some numerical syntax issues in tables

## 0.4.15 - 2023-08-11
### Improvement
- Bumped pvlib to 0.10.1
- Updated documentation for forecasts methods.
### Fix
- Fixed error message on utils.py

## 0.4.14 - 2023-07-17
### Improvement
- Bumped skforecast to latest 0.9.1.
- The standalone dockerfile was updated by @majorforg to include the CBC solver.
### Fix
- Fix rounding for price & cost forecasts by @purcell-lab

## 0.4.13 - 2023-06-29
### Improvement
- Added support for data reconstruction when missing values on last window for ML forecaster prediction.
- Added treatment of SOCtarget passed at runtime for day-ahead optimization.
- Added publish_prefix key to pass a common prefix to all published data.
### Fix
- Patched sensor rounding problem.
- Bump myst-parser from 1.0.0 to 2.0.0
- Fixed missing attributes is the sensors when using the custom IDs.

## 0.4.12 - 2023-06-03
### Improvement
- Added forecasts for unit_prod_price and unit_load_cost.
- Improved documentation.
### Fix
- Bump skforecast from 0.8.0 to 0.8.1

## 0.4.11 - 2023-05-27
### Improvement
- Adding new constraints to limit the dynamics (kW/sec) of deferrable loads and battery power. The LP formulation works correctly and a work should be done on integrating the user input parameters to control this functionality.
- Added new constraint to avoid battery discharging to the grid.
- Added possibility to set the logging level.
### Fix
- Bumped version of skforecast from 0.6.0 to 0.8.1. Doing this mainly implies changing how the exogenous data is passed to fit and predict methods.
- Fixed wrong path for csv files when using load cost and prod price forecasts.

## 0.4.10 - 2023-05-21
### Fix
- Fixed wrong name of new cost sensor.
- Fixed units of measurements of costs to €/kWh.
- Added color sequence to plot figures, now avery line should be plotted with a different color.

## 0.4.9 - 2023-05-20
### Fix
- Updated default value for total number of days for ML model training.
- Added publish of unit_load_cost and unit_prod_price sensors.
- Improved docs intro.
- Bump myst-parser from 0.18.1 to 1.0.0

## 0.4.8 - 2023-03-17
### Fix
- Fixed to correct index length for ML forecaster prediction series.

## 0.4.7 - 2023-03-16
### Fix
- Fixed wrong column name for var_load when using predict with ML forecaster.

## 0.4.6 - 2023-03-16
### Fix
- Fixed wrong path for saved ML forecaster model.

## 0.4.5 - 2023-03-10
### Fix
- Fixed default behavior for passed data.
- Added a new graph for tune results.

## 0.4.4 - 2023-03-09
### Fix
- Added missing possibility to set the method for load forecast to 'mlforecaster'.
- Fixed logging formatting.

## 0.4.3 - 2023-03-09
### Fix
- Fixed logging.
- Fixed missing module on docker standalone mode.

## 0.4.2 - 2023-03-07
### Fix
- Fixed handling of default passed params.

## 0.4.1 - 2023-03-06
### Improvement
- Improved the documentation and the in-code docstrings.
- Added the possibility to save the optimized model after a tuning routine.
- Added the possibility to publish predict results to a Home Assistant sensor.
- Added the possibility to provide custom entity_id, unit_of_measurement and friendly_name for each published data.

## 0.4.0 - 2023-03-06
### Improvement
- A brand new load forecast module and more... The new forecast module can actually be used to foreast any Home Assistant variable. The API provides fit, predict and tune methods. By the default it provides a more efficient way to forecast the power load consumption. It is based on the skforecast module that uses scikit-learn regression models considering auto-regression lags as features. The hyperparameter optimization is proposed using bayesian optimization from the optuna module.
- A new documentation section covering the new forecast module.
### Fix
- Fixed Solar.Forecast issues with lists of parameters.
- Fixed latex equations rendering on documentation, dropped Mathjax.
- Refactored images in documentation, now using only SVG for plotly figures.
- Bumped requirements to latest non-conflicting versions.

## 0.3.36 - 2023-01-31
### Fix
- Fixed message handling from request module.

## 0.3.35 - 2023-01-31
### Fix
- Fixed access to injection_dict for the first time that emhass is used.

## 0.3.34 - 2023-01-30
### Fix
- Fixed bugs with paths again, now using the official pathlib everywhere.

## 0.3.32 - 2023-01-30
### Fix
- Fixed bugs on handling data folder name.
- Improved warning messages when passing list of values with items detected as non digits.

## 0.3.29 - 2023-01-28
### Improvement
- Implemented data storage to survive add-on restarts.

## 0.3.25 - 2023-01-27
### Fix
- Fixed dependencies, uniform working versions of Numpy, Pandas and Tables.

## 0.3.24 - 2023-01-27
### Fix
- Fixed dependencies, rolled back to older fixed Numpy and Pandas versions.

## 0.3.23 - 2023-01-26
### Fix
- Fixed missing integration of additional `set_nocharge_from_grid` in the web server.
- Improved the documentation.

## 0.3.22 - 2023-01-26
### Improvement
- Improved unittest for mock get requests.
- Improved coverage.
### Fix
- Code works even if no battery data is configured.
- Added more explicit logging error message in the case of no data retrieved from Home Assistant.

## 0.3.21 - 2022-10-21
### Fix
- Fixed docstrings
- Added github worflows for coverage and automatic package compiling.
- Fixing interpolation for Forecast.Solar data.

## 0.3.20 - 2022-10-05
### Improvement
- Added more detailed examples to the forecast module documentation.
- Improved handling of datatime indexes in DataFrames on forecast module.
- Added warning messages if passed list values contains non numeric items.
- Added missing unittests for forecast module with request.get dependencies using MagicMock.
- Added the Solar.Forecast method.

## 0.3.19 - 2022-09-14
### Fix
- Updated default values for a working LP solver.
- Removed option to provide a custom web ui url.
- Added extra runtime parameters to use solcast PV forecast.

## 0.3.18 - 2022-08-27
### Improvement
- Improving documentation, added more information on forecast page.
- Added support for SolCast PV production forecasts. 
- Added possibility to pass some optimization parameters at runtime.
- Added some unittest for passing data as list testing.
### Fix
- Fixed small bug on webserver using pandas sum function for non numeric data. This was throwing the following message: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated.

## 0.3.17 - 2022-06-12
### Fix
- Fixed wrong variables names for mixed forecasts.
- Fixed handling of load sensor name in command line setup function.

## 0.3.16 - 2022-06-10
### Improvement
- Improving documentation, added "what is this" section and added some infographics.
- Added new forecasts methods chapter in documentation.
- Added publish of sensors for p_grid_forecast & total value of cost function.
- Implemented now/current value forecast correction when using MPC.

## 0.3.15 - 2022-06-06
### Fix
- Fixed small bug with wrong DF variable name in web server.

## 0.3.14 - 2022-06-05
### Improvement
- Added one more table to the webui showing the cost totals.
### Fix
- Fixed wrong type error when serializing numpy ints. Converted ints to Python type.

## 0.3.13 - 2022-05-20
### Fix
- Fix wrong default value implementation for solver params.

## 0.3.12 - 2022-05-20
### Improvement
- Added support to provide solver name and path as parameters in the configuration file.

## 0.3.11 - 2022-05-23
### Fix
- Fixed unittests not passing.

## 0.3.10 - 2022-05-23
### Improvement
- Added data as attributes for forecasts (PV and load), deferrable loads and battery: power+SOC.
- Improved the graph in the webserver, now using subplots.
- Rearranged vertical space in index.html template.
### Fix
- Added threads option on waitress to possibly improve queue problem.

## 0.3.9 - 2022-05-19
### Improvement
- Improved publish data logging.
- Round published data.
- Attributes to forecasts published data.
- Improved centering html on small devices.
- Improved handling of closest index when publishing data.
### Fix
- Fixed problem with csv filenames, was using only filename specific to dayahead optimization.
- Fixed data list handling for load cost and prod price forecasts.
- Fixed publish data dictionary to contain only data from correct entity_id.
- May have solved double log lines.

## 0.3.8 - 2022-05-17
### Fix
- Still fixing issues when passing csv files and lists.

## 0.3.7 - 2022-05-17
### Fix
- Fixed None weather df issue when passing lists.
- Improved command line unittests.

## 0.3.6 - 2022-05-17
### Fix
- Fixed wrong handling of list values when preparing data for MPC.

## 0.3.5 - 2022-05-16
### Fix
- Fixed wrong mpc pred horizon param.

## 0.3.4 - 2022-05-16
### Fix
- Fixed unloaded json object problem.
- Added static style.css as package_data.

## 0.3.3 - 2022-05-15
### Fix
- Fixed dealing with bool and argparse module.
- Added the templates as package_data so that they can be found by jinja2 PackageLoader.

## 0.3.2 - 2022-05-13
### Fix
- Fixed command_line and utils problem when params is None.

## 0.3.1 - 2022-05-13
### Fix
- Fixed template rendering problems.

## 0.3.0 - 2022-05-12
### Improvement
- Improved the command line setup function to perform the correct amount calls as needed by each action.
- Added a new naive model predictive control function.
- Added runtime parameter option for better code order.
- Moved treatment of runtime parameters from the add-on to the core emhass module. This adds more clarity to the code and also was needed when passing runtime parameters to emhass in standalone mode.
- Added add-on paramter to command line to define if launching emhass from add-on or in standalone mode.
- Added new testing file for command_line.
- Added a webserver. Moved the webserver from the add-on to the core emhass module.
- Added a WSGI production server for flask using waitress.
- Added a Dockerfile and procedure to run emhass in standalone mode.
- Updated the documentation.

## 0.2.14 - 2022-05-05
### Improvement
- Added more info on publish data errors when not key found. This error may mean that the optimization task may need to be relaunched or it did not converged to a solution.
- Added better info to the configuration documentation when integrating PV module and inverter models from PVLib database. An underscore _ character should be added inplace of each special character.
### Fix
- Fixed missing timeStep product for correct deferrable load total energies.

## 0.2.13 - 2022-05-01
### Improvement
- Added support to pass list of PV plant parameters. This will enable to simulate mixed orientation systems, for example one east-facing array (azimuth=90) and one west-facing array (azimuth=270).
### Fix
- Fixed issue computing correct final cost values.

## 0.2.12 - 2022-04-28
### Improvement
- Added config parameter to consider that all PV power is injected to the grid.

## 0.2.11 - 2022-04-28
### Fix
- Fixed wrong handling of DateTimeIndex when dealing with forecast method for list of values and csv read.

## 0.2.10 - 2022-04-26
### Fix
- Fixed faulty forecast method for list of values and csv read.

## 0.2.9 - 2022-04-21
### Fix
- Fixed get_loc deprecation warning using now get_indexer pandas method. Improved selection of closest index.

## 0.2.8 - 2022-04-18
### Fix
- Fixed if sentence to correctly use the supervisor API for publish data.
- Fixing a error computing the nearest index of DataFrame. Using a try/catch strategy to use nearest method as a backup.

## 0.2.7 - 2022-04-18
### Fix
- Fixing a fatal error where the publish data function will never reach the savec csv file as the default filename is not equal to the expected filename in publish_data.

## 0.2.6 - 2022-04-16
### Improvement
- Improved handling of errors concerning solver issues with Pulp. Added support for `glpk` solver. For now just using a try/catch strategy but should update to solver passed as a parameter to EMHASS.

## 0.2.5 - 2022-04-12
### Fix
- Fixed missing numpy import in utils.

## 0.2.4 - 2022-04-12
### Fix
- Fixed missing command to retrieve PV power forecast when using list of values.
- Updated handling of freq definition to a pandas index.

## 0.2.3 - 2022-03-29
### Improvement
- Improved support for the new add-on and direct communication via the supervisor.
- The CLI now can return the version using the --version argument.
- Improved comments in forecast class.
### Added
- Added unittest for csv method for weather forecast.
- Added support for passing lists of values to all forecasting methods.
### Fix
- Removed dependency from PVLib Forecast class, as it has been marked as deprecated.
- Fixed failing docs builds due to uncompatible jinja2 an markupsafe versions.

## 0.2.2 - 2022-03-05
### Added
- Added a new input data file using pickle for tests.
- Added support to select if the logger should save to a file or not.
- Added CI workflow using github actions.
### Breaking changes
- Changed package usage of configuration file path, now the user must provide the complete path including the yaml file itself.
- Changed the names of the configuration and secrets yaml files. These changes will avoid compatibility issues when using hass add-ons.

## 0.2.1 - 2021-12-22
### Fixed
- Cleaned unittest implementation
### Added
- Added support to choose the methods for weather and load forecast from configuration file.
- Added new load cost and power production price forecasts, mainly allowing the user to load a CSV file with their own forecast.

## 0.2.0 - 2021-10-14
### Fixed
- Fixed tests to pass with latest changes on passing path and logger arguments.
- Updated requirements for PVLib and Protobuf.
- Removed unnecessary use of abstract classes.
- Fixed test_optimization bad setup.
- Fixed logger integration in classes
- Updated documentation
### Added
- Implemented typing for compatibility with Python4
- Implemented different types of cost functions

## 0.1.5 - 2021-09-22
### Fixed
- Fix a recurrent previous bug when using get_loc. The correct default behavior when using get_loc is changed from backfill to ffill.

## 0.1.4 - 2021-09-18
### Fixed
- Fixed a bug when publish-data and reading the CSV file, the index was not correctly defined, so there was a bug when applying pandas get_loc.
### Added
- Added a global requirements.txt file for pip install.

## 0.1.3 - 2021-09-17
### Fixed
- Fixed packaging and configuration for readthedocs.

## 0.1.2 - 2021-09-17
### Fixed
- Modified the cost function equation signs for more clarity. Now the LP is fixed to maximize a profit given by the revenues from selling PV to the grind minus the energy cost of consumed energy.
- Fixed a deprecation warning from PVLib when retrieving results from the ModelChain object. Now using modelchain.results.ac.

## 0.1.1 - 2021-09-17
### Fixed
- Fixed sign error in cost function.
- Change publish_data get_loc behavior from nearest to backfill.
- Changed and updated behavior of the logger. It is constructed and integrated directly in the main function of the command_line.py file. It now writes to a log file by default.
- Fixed some typos and errors in the documentation.

## 0.1.0 - 2021-09-12
### Added
- Added the first public repository for this project.


# Notes
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
