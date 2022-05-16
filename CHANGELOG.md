# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.5] - 2022-05-16
### Fix
- Fixed unloaded json object problem.

## [0.3.4] - 2022-05-16
### Fix
- Fixed unloaded json object problem.
- Added static style.css as package_data.

## [0.3.3] - 2022-05-15
### Fix
- Fixed dealing with bool and argparse module.
- Added the templates as package_data so that they can be found by jinja2 PackageLoader.

## [0.3.2] - 2022-05-13
### Fix
- Fixed command_line and utils problem when params is None.

## [0.3.1] - 2022-05-13
### Fix
- Fixed template rendering problems.

## [0.3.0] - 2022-05-12
### Improvement
- Improved the command line setup function to perform the correct amount calls as needed by each action.
- Added a new naive model predictive control function.
- Added runtime parameter option for better code order.
- Moved treatment of runtime parameters from the add-on to the core emhass module. This adds more clarity to the code andd also was needed when passing runtime paramters to emhass in standalone mode.
- Added add-on paramter to command line to define if launching emhass from add-on or in standalone mode.
- Added new testing file for command_line.
- Added a webserver. Moved the webserver from the add-on to the core emhass module.
- Added a WSGI production server for flask using waitress.
- Added a Dockerfile and procedure to run emhass in standalone mode.
- Updated the documentation.

## [0.2.14] - 2022-05-05
### Improvement
- Added more info on publish data errors when not key found. This error may mean that the optimization task may need to be relaunched or it did not converged to a solution.
- Added better info to the configuration documentation when integrating PV module and inverter models from PVLib database. An underscore _ character should be added inplace of each special character.
### Fix
- Fixed missing timeStep product for correct deferrable load total energies.

## [0.2.13] - 2022-05-01
### Improvement
- Added support to pass list of PV plant parameters. This will enable to simulate mixed orientation systems, for example one east-facing array (azimuth=90) and one west-facing array (azimuth=270).
### Fix
- Fixed issue computing correct final cost values.

## [0.2.12] - 2022-04-28
### Improvement
- Added config parameter to consider that all PV power is injected to the grid.

## [0.2.11] - 2022-04-28
### Fix
- Fixed wrong handling of DateTimeIndex when dealing with forecast method for list of values and csv read.

## [0.2.10] - 2022-04-26
### Fix
- Fixed faulty forecast method for list of values and csv read.

## [0.2.9] - 2022-04-21
### Fix
- Fixed get_loc deprecation warning using now get_indexer pandas method. Improved selection of closest index.

## [0.2.8] - 2022-04-18
### Fix
- Fixed if sentence to correctly use the supervisor API for publish data.
- Fixing a error computing the nearest index of DataFrame. Using a try/catch strategy to use nearest method as a backup.

## [0.2.7] - 2022-04-18
### Fix
- Fixing a fatal error where the publish data function will never reach the savec csv file as the default filename is not equal to the expected filename in publish_data.

## [0.2.6] - 2022-04-16
### Improvement
- Improved handling of errors concerning solver issues with Pulp. Added support for `glpk` solver. For now just using a try/catch strategy but should update to solver passed as a parameter to EMHASS.

## [0.2.5] - 2022-04-12
### Fix
- Fixed missing numpy import in utils.

## [0.2.4] - 2022-04-12
### Fix
- Fixed missing command to retrieve PV power forecast when using list of values.
- Updated handling of freq definition to a pandas index.

## [0.2.3] - 2022-03-29
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

## [0.2.2] - 2022-03-05
### Added
- Added a new input data file using pickle for tests.
- Added support to select if the logger should save to a file or not.
- Added CI workflow using github actions.
### Breaking changes
- Changed package usage of configuration file path, now the user must provide the complete path including the yaml file itself.
- Changed the names of the configuration and secrets yaml files. These changes will avoid compatibility issues when using hass add-ons.

## [0.2.1] - 2021-12-22
### Fixed
- Cleaned unittest implementation
### Added
- Added support to choose the methods for weather and load forecast from configuration file.
- Added new load cost and power production price forecasts, mainly allowing the user to load a CSV file with their own forecast.

## [0.2.0] - 2021-10-14
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

## [0.1.5] - 2021-09-22
### Fixed
- Fix a recurrent previous bug when using get_loc. The correct default behavior when using get_loc is changed from backfill to ffill.

## [0.1.4] - 2021-09-18
### Fixed
- Fixed a bug when publish-data and reading the CSV file, the index was not correctly defined, so there was a bug when applying pandas get_loc.
### Added
- Added a global requirements.txt file for pip install.

## [0.1.3] - 2021-09-17
### Fixed
- Fixed packaging and configuration for readthedocs.

## [0.1.2] - 2021-09-17
### Fixed
- Modified the cost function equation signs for more clarity. Now the LP is fixed to maximize a profit given by the revenues from selling PV to the grind minus the energy cost of consumed energy.
- Fixed a deprecation warning from PVLib when retrieving results from the ModelChain object. Now using modelchain.results.ac.

## [0.1.1] - 2021-09-17
### Fixed
- Fixed sign error in cost function.
- Change publish_data get_loc behavior from nearest to backfill.
- Changed and updated behavior of the logger. It is constructed and integrated directly in the main function of the command_line.py file. It now writes to a log file by default.
- Fixed some typos and errors in the documentation.

## [0.1.0] - 2021-09-12
### Added
- Added the first public repository for this project.

[0.1.0]: https://github.com/davidusb-geek/emhass/releases/tag/v0.1.0
[0.1.1]: https://github.com/davidusb-geek/emhass/releases/tag/v0.1.1
[0.1.2]: https://github.com/davidusb-geek/emhass/releases/tag/v0.1.2
[0.1.3]: https://github.com/davidusb-geek/emhass/releases/tag/v0.1.3
[0.1.4]: https://github.com/davidusb-geek/emhass/releases/tag/v0.1.4
[0.1.5]: https://github.com/davidusb-geek/emhass/releases/tag/v0.1.5
[0.2.0]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.0
[0.2.1]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.1
[0.2.2]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.2
[0.2.3]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.3
[0.2.4]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.4
[0.2.5]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.5
[0.2.6]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.6
[0.2.7]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.7
[0.2.8]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.8
[0.2.9]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.9
[0.2.10]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.10
[0.2.11]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.11
[0.2.12]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.12
[0.2.13]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.13
[0.2.14]: https://github.com/davidusb-geek/emhass/releases/tag/v0.2.14
[0.3.0]: https://github.com/davidusb-geek/emhass/releases/tag/v0.3.0