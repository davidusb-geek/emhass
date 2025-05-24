# EMHASS Development

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

# Setup

There are multiple different approaches to developing for EMHASS.  
The choice depends on your and preference (Python venv/DevContainer/Docker).  
Below are some development workflow examples:  
_Note: It is preferred to run the actions and unittest once before submitting and pull request._

## Step 1 - Fork

_With your preferred Git tool of choice:_  
Fork the EMHASS github repository into your own account, then clone the forked repository into your local development platform. (ie. PC or Codespace)
Here you may also wish to add the add the original/upstream repository as a remote, allowing you to fetch and merge new updates from the original repository.

A command example may be:
```bash
# on GitHub, Fork url, then:
git clone git@github.com:<YOURUSERNAME>/emhass.git
cd emhass
# add remote, call it upstream
git remote add upstream https://github.com/davidusb-geek/emhass.git
```

## Step 2 - Develop

To develop and test code choose one of the following methods:

### Method 1 - Python Virtual Environment

We can use python virtual environments to build, develop and test/unittest the code.

_confirm terminal is in the root `emhass` directory before starting_

**Create a developer environment:**

Using the [`uv` package manager](https://docs.astral.sh/uv/):
```bash
# With the 'test' packages to run unit tests locally.
uv sync --extra test
# If on ARM, try adding piwheels as an index.
#uv sync --extra test --index=https://www.piwheels.org/simple
```

Using virtualenv and pip:
```bash
virtualenv .venv

# Then activate the virtualenv, see below...

# With the 'test' packages to run unit tests locally.
python3 -m pip install -e '.[test]'
```

To activate the virtualenv, created by either uv or pip:
- Linux:
  ```bash
  source .venv/bin/activate
  ```
- windows:
  ```cmd
  .venv\Scripts\activate.bat
  ```

This installs dependencies and creates a `.venv` virtualenv in the working directory.
An IDE like VSCode should automatically catch that a new virtual env was created.

**Set paths with environment variables:**

- Linux
  ```bash
  export OPTIONS_PATH="${PWD}/options.json" && export USE_OPTIONS="True" ##optional to test options.json
  export CONFIG_PATH="${PWD}/config.yaml"
  export SECRETS_PATH="${PWD}/secrets_emhass.yaml" ##optional to test secrets_emhass.yaml
  export DATA_PATH="${PWD}/data/"
  ```
  Optionally, use [direnv](https://direnv.net/) to have these variables handled for you.

- windows
  ```batch
  set "OPTIONS_PATH=%cd%/options.json"  & ::  optional to test options.json
  set "USE_OPTIONS=True"                & ::  optional to test options.json
  set "CONFIG_PATH=%cd%/config.json"
  set "SECRETS_PATH=%cd%/secrets_emhass.yaml" & ::  optional to test secrets_emhass.yam
  set "DATA_PATH=%cd%/data/"
  ```

_Make sure `secrets_emhass.yaml` has been created and set. Copy `secrets_emhass(example).yaml` for an example._

**Run EMHASS**
```bash
python3 ./src/emhass/web_server.py
```
or
``` bash
emhass --action 'dayahead-optim' --config ./config.json --root ./src/emhass --costfun 'profit' --data ./data
```

**Run unittests**
```bash
pytest
```

### Method 2: VS-Code Debug and Run via Dev Container

In VS-Code, you can run a Docker Dev Container to set up a virtual environment. The Dev Container's Container will be almost identical to the one build for EMHASS (Docker/Add-on). There you can edit and test EMHASS.

The recommended steps to run are:

- Open forked root (`emhass`) folder inside of VS-Code
- VS-Code will ask if you want to run in a dev-container, say yes _([Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) must be set up first)_. *(Shortcut: `F1` > `Dev Containers: Rebuild and Reopen in Container`)*
- Edit some code...
- Compile emhass by pressing `control+shift+p` > `Tasks: Run Task` > `EMHASS Install`.
  This has been set up in the [tasks.json](https://github.com/davidusb-geek/emhass/blob/master/.vscode/tasks.json) file. - Before _run & debug_, re-run `EMHASS Install` task every time a change has been made to emhass.
- Launch and debug the program via the [`Run and Debug`](https://code.visualstudio.com/docs/editor/debugging) tab /`Ctrl+Shift+D` > `EMHASS run` This has been set up in the [Launch.json](https://github.com/davidusb-geek/emhass/blob/master/.vscode/launch.json) .

#### Simulate Docker Method or Add-on method
Since the main difference between the two methods are how secrets are passed. You can switch between the two methods by:  
  
**Docker**:  
  - Create a `secrets_emhass.yaml` file and append your secret parameters  

**Add-on**:  
  - Modify the `options.json` file to contain you secret parameters  

#### Unittests
Lastly, you can run all the unittests by heading to the [`Testing`](https://code.visualstudio.com/docs/python/testing) tab on the left hand side.  This is recommended before creating a pull request.

### Method 3 - Docker Virtual Environment

With Docker, you can test the production EMHASS environment for both Docker and Add-on methods.

Depending on the method you wish to test, the `docker run` command will require different passed arguments to function. See following examples:

_Note: Make sure your terminal is in the root `emhass` repository directory before running the docker build._

#### Docker run Add-on Method:

```bash
docker build -t emhass/test .

# pass secrets via options.json (similar to what Home Assistant automatically creates from the addon configuration page)
docker run -it -p 5000:5000 --name emhass-test -v ./options.json:/data/options.json emhass/test
```

**Note:**
- to apply a file change in the local EMHASS repository, you will need to re-build and re-run the Docker image/container in order for the change to take effect. (excluding volume mounted (-v) files/folders)
- if you are planning to modify the configs: `options.json`, `secrets_emhass.yaml` or `config.json`, you can [volume mount](https://docs.docker.com/engine/storage/bind-mounts/) them with `-v`. This syncs the Host file to the file inside the container.
*If running inside of podman, add :z at the end of the volume mount E.g:`-v ./options.json:/data/options.json:z`*

#### Docker run for Docker Method:

```bash
docker build -t emhass/test .

# pass the secrets_emhass.yaml
docker run -it -p 5000:5000 --name emhass-test -v ./secrets_emhass.yaml:/app/secrets_emhass.yaml emhass/test
```

#### Sync with local data folder 
For those who wish to mount/sync the local `data` folder with the data folder from inside the docker container, volume mount the data folder with `-v` .
```bash
docker run ... -v ./data/:/data/ ...
```

You can also mount data files (ex .csv)  separately
```bash
docker run... -v ./data/heating_prediction.csv:/data/ ...
```

#### Issue with TARGETARCH
If your docker build fails with an error related to `TARGETARCH`. It may be best to add your device's architecture manually:

Example with `armhf` architecture 
```bash
docker build ... --build-arg TARGETARCH=armhf --build-arg os_version=raspbian ...
```
*For `armhf` only, also pass a build-arg for `os_version=raspbian`* 


#### Delete built Docker image

We can delete the Docker image and container via:

```bash
# force delete Docker container
docker rm -f emhass-test 

# delete Docker image
docker rmi emhass/test 
```

#### Other Docker Options

**Rapid Testing**  
As editing and testing EMHASS via docker may be repetitive (rebuilding image and deleting containers), you may want to simplify the removal, build and run process.

**For rapid Docker testing, try a command chain:**  
_Linux:_
```bash
docker build -t emhass/test . && docker run --rm -it -p 5000:5000 -v ./secrets_emhass.yaml:/app/secrets_emhass.yaml --name emhass-test emhass/test 
```
_The example command chain rebuilds the Docker image, and runs a new container with the newly built image. The `--rm` has been added to the `docker run` to delete the container once ended to avoid manual deletion every time._  
_This use case may not require any volume mounts (unless you use secrets_emhass.yaml) as the Docker build process will pull the latest configs as it builds._

**Environment Variables**  
 you can also pass location, key and url secret parameters via environment variables.

```bash
docker build -t emhass/test --build-arg build_version=addon-local .

docker run -it -p 5000:5000 --name emhass-test -e URL="YOURHAURLHERE" -e KEY="YOURHAKEYHERE" -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" emhass/test
```

This allows the user to set variables before the build
Linux:

```bash
export EMHASS_URL="YOURHAURLHERE"
export EMHASS_KEY="YOURHAKEYHERE"
export TIME_ZONE="Europe/Paris"
export LAT="45.83"
export LON="6.86"
export ALT="4807.8"

docker build -t emhass/test --build-arg build_version=addon-local .

docker run -it -p 5000:5000 --name emhass-test -e EMHASS_KEY -e EMHASS_URL -e TIME_ZONE -e LAT -e LON -e ALT emhass/test
```

### Example Docker testing pipeline 
The following pipeline will run unittest and most of the EMHASS actions. This may be a good option for those who wish to test their changes against the production EMHASS environment.

*Linux:*  
*Assuming docker and git installed*
```bash
#setup environment variables for test
export repo=https://github.com/davidusb-geek/emhass.git
export branch=master
#Ex. HAURL=https://localhost:8123/
export HAURL=HOMEASSISTANTURLHERE
export HAKEY=HOMEASSISTANTKEYHERE

git clone $repo
cd emhass 
git checkout $branch
```

```bash
# testing with option.json (replace -v options.json with secrets_emhass.yaml to test both secret files)
docker build -t emhass/test .
docker run --rm -it -p 5000:5000 --name emhass-test -v $(pwd)/data/heating_prediction.csv:/data/heating_prediction.csv -v $(pwd)/options.json:/app/options.json emhass/test
```
```bash
# run actions one-by-one, on a separate terminal
curl -i -H 'Content-Type:application/json' -X POST -d '{"pv_power_forecast":[0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93], "prediction_horizon":10, "soc_init":0.5,"soc_final":0.6}' http://localhost:5000/action/naive-mpc-optim
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/perfect-optim
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/dayahead-optim
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/forecast-model-fit
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/forecast-model-predict
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/forecast-model-tune
curl -i -H "Content-Type:application/json" -X POST -d  '{"csv_file": "heating_prediction.csv", "features": ["degreeday", "solar"], "target": "hour", "regression_model": "RandomForestRegression", "model_type": "heating_hours_degreeday", "timestamp": "timestamp", "date_features": ["month", "day_of_week"], "new_values": [12.79, 4.766, 1, 2] }' http://localhost:5000/action/regressor-model-fit
curl -i -H "Content-Type:application/json" -X POST -d  '{"mlr_predict_entity_id": "sensor.mlr_predict", "mlr_predict_unit_of_measurement": "h", "mlr_predict_friendly_name": "mlr predictor", "new_values": [8.2, 7.23, 2, 6], "model_type": "heating_hours_degreeday" }' http://localhost:5000/action/regressor-model-predict
curl -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/publish-data
```

```bash
# testing unittest (add extra necessary files via volume mount)
docker run --rm -it -p 5000:5000 --name emhass-test -v $(pwd)/tests/:/app/tests/ -v $(pwd)/data/:/data/ -v $(pwd)/"secrets_emhass(example).yaml":/app/"secrets_emhass(example).yaml" -v $(pwd)/options.json:/app/options.json -v $(pwd)/config_emhass.yaml:/app/config_emhass.yaml -v $(pwd)/secrets_emhass.yaml:/app/secrets_emhass.yaml emhass/test
```
```bash
# run unittest's on separate terminal after installing requests-mock
docker exec emhass-test apt-get update 
docker exec emhass-test apt-get install python3-requests-mock -y
docker exec emhass-test python3 -m unittest discover -s ./tests -p 'test_*.py' | grep error
```
*Note: may need to set `--build-arg TARGETARCH=YOUR-ARCH` in docker build*

User may wish to re-test with tweaked parameters such as `lp_solver`, `weather_forecast_method` and `load_forecast_method`, in `config.json` to broaden the testing scope. 
*See [Differences](https://emhass.readthedocs.io/en/latest/differences.html) for more information on how the different methods of running EMHASS differ.*

### Adding a parameter
When enhancing EMHASS, users may like to add or modify the EMHASS parameters. To add a new parameter see the following steps:

*Example parameter = `this_parameter_is_amazing`*

Append a line into `associations.csv` :
*So that build_params() knows what config catagorie to allocate the parameter*
```csv
...
retrieve_hass_conf,,this_parameter_is_amazing
```
 - Alternatively if you want to support this parameter with the yaml conversion *(Ie. allow the parameter to be converted from config_emhass.yaml)*
    ```csv
    ...
    retrieve_hass_conf,his_parameter_is_amazing,this_parameter_is_amazing
    ```

Append a line into the `config_defaults.json`
*To set a default value for the user if none is provided in `config.json`*
```json
"...": "...",
  "this_parameter_is_amazing": [
    0.1,
    0.1
  ]
```

Lastly, to support the configuration website to generate the parameter in the list view, append the `param_definitions.json` file:
```json
"this_parameter_is_amazing": {
      "friendly_name": "This parameter is amazing",
      "Description": "This parameter functions as you expect. It makes EMHASS AMAZING!",
      "input": "array.float",
      "default_value": 0.777
    }
```
*Note: The `default_value` in this case acts (or should act) as last resort fallback if default_config.json is not found. It also acts as the default value when you append (press plus) to an array.\* parameter*

![Screenshot from 2024-09-09 16-45-32](https://github.com/user-attachments/assets/01e7984f-3332-4e25-8076-160f51a2e0c4)

If you are only adding another option for a existing parameter, editing param_definitions.json file should be all you need. (allowing the user to select the option from the configuration page):
```json
"load_forecast_method": {
  "friendly_name": "Load forecast method",
  "Description": "The load forecast method that will be used. The options are ‘csv’ to load a CSV file or ‘naive’ for a simple 1-day persistence model.",
  "input": "select",
  "select_options": [
    "naive",
    "mlforecaster",
    "csv",
    "CALL_NEW_OPTION"
  ],
  "default_value": "naive"
},
```

## Step 3 - Pull request

Once developed, commit your code, and push the commit to your fork on Github.
Once ready, submit a pull request with your fork to the [davidusb-geek/emhass@master](https://github.com/davidusb-geek/emhass) repository.
