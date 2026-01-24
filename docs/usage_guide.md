# Usage guide

## Method 1) Add-on and Docker

If using the add-on or the Docker installation, it exposes a simple webserver on port 5000. You can access it directly using your browser. (E.g.: http://localhost:5000)

With this web server, you can perform RESTful POST commands on multiple ENDPOINTS with the prefix `action/*`:

- A POST call to `action/perfect-optim` to perform a perfect optimization task on the historical data.
- A POST call to `action/dayahead-optim` to perform a day-ahead optimization task of your home energy.
- A POST call to `action/naive-mpc-optim` to perform a naive Model Predictive Controller optimization task. If using this option you will need to define the correct `runtimeparams` (see "Passing data to EMHASS" section).
- A POST call to `action/publish-data` to publish the optimization results data for the current timestamp.
- A POST call to `action/forecast-model-fit` to train a machine learning forecaster model with the passed data (see the [ML Forecaster](mlforecaster) section for more help).
- A POST call to `action/forecast-model-predict` to obtain a forecast from a pre-trained machine learning forecaster model (see the [ML Forecaster](mlforecaster) section for more help).
- A POST call to `action/forecast-model-tune` to optimize the machine learning forecaster models hyperparameters using Bayesian optimization (see the [ML Forecaster](mlforecaster) section for more help).

A `curl` command can then be used to launch an optimization task like this: `curl -i -H 'Content-Type:application/json' -X POST -d '{}' http://localhost:5000/action/dayahead-optim`.

## Method 2) Legacy method using a Python virtual environment

To run a command simply use the `emhass` CLI command followed by the needed arguments.
The available arguments are:
- `--action`: This is used to set the desired action, options are: `perfect-optim`, `dayahead-optim`, `naive-mpc-optim`, `publish-data`, `forecast-model-fit`, `forecast-model-predict` and `forecast-model-tune`.
- `--config`: Define the path to the config.json file (including the yaml file itself)
- `--secrets`: Define secret parameter file (secrets_emhass.yaml) path
- `--costfun`: Define the type of cost function, this is optional and the options are: `profit` (default), `cost`, `self-consumption`
- `--log2file`: Define if we should log to a file or not, this is optional and the options are: `True` or `False` (default)
- `--params`: Configuration as JSON. 
- `--runtimeparams`: Data passed at runtime. This can be used to pass your own forecast data to EMHASS.
- `--debug`: Use `True` for testing purposes.
- `--version`: Show the current version of EMHASS.
- `--root`: Define path emhass root (E.g. ~/emhass )
- `--data`: Define path to the Data files (.csv & .pkl) (E.g. ~/emhass/data/ )

For example, the following line command can be used to perform a day-ahead optimization task:
```bash
emhass --action 'dayahead-optim' --config ~/emhass/config.json --costfun 'profit'
```
Before running any valuable command you need to modify the `config.json` and `secrets_emhass.yaml` files. These files should contain the information adapted to your own system. To do this take a look at the special section for this in the [Configuration](config) section.
