# EMHASS Development

There are multiple different approaches to developing EMHASS.  
The choice depends on EMHASS mode (standalone/add-on) and preference (Python venv/DevContainer/Docker).  
Below are some development workflow examples:  
_Note: It is preferred to run both addon mode, standalone mode and unitest once before submitting and pull request._

## Step 1 - Fork

_With your preferred Git tool of choice:_  
Fork the EMHASS github repository into your own account, then clone the forked repository into your local development platform. (ie. PC or Codespace)
Here you may also wish to add the add the origional/upstream repository as a remote, allowing you to fetch and merge new updates from the origional repository.

A command example may be:
```bash
# on GitHub, Fork url, then:
git clone https://github.com/<YOURUSERNAME>/emhass.git
cd emhass
# add remote, call it upstream
git remote add upstream https://github.com/OWNER/REPOSITORY.git
```

## Step 2 - Develop

To develop and test code choose one of the following methods:

### Method 1 - Python Virtual Environment

We can use python virtual environments to build, develop and test/unitest the code.
This method works well with standalone mode.

_confirm terminal is in the root `emhass` directory before starting_

**Install requirements**
```bash
python3 -m pip install -r requirements.txt #if arm try setting --extra-index-url=https://www.piwheels.org/simple
```

**Create a developer environment:**

```bash
python3 -m venv .venv
```

**Activate the environment:**

- linux:

  ```bash
  source .venv/bin/activate
  ```

- windows:

  ```cmd
  .venv\Scripts\activate.bat
  ```

An IDE like VSCode should automatically catch that a new virtual env was created.

**Install the _emhass_ package in editable mode:**

```bash
python3 -m pip install -e .
```

**Set paths with environment variables:**

- Linux
  ```bash
  export OPTIONS_PATH="${PWD}/options.json" && export USE_OPTIONS="True" ##optional to test options.json
  export CONFIG_PATH="${PWD}/config_emhass.yaml"
  export SECRETS_PATH="${PWD}/secrets_emhass.yaml"
  export DATA_PATH="${PWD}/data/"
  ```
- windows
  ```cmd
  set "OPTIONS_PATH=%cd%/options.json"  & ::  optional to test options.json
  set "USE_OPTIONS=True"                & ::  optional to test options.json
  set "CONFIG_PATH=%cd%/config_emhass.yaml"
  set "SECRETS_PATH=%cd%/secrets_emhass.yaml"
  set "DATA_PATH=%cd%/data/"
  ```

_Make sure `secrets_emhass.yaml` has been created and set. Copy `secrets_emhass(example).yaml` for an example._

**Run EMHASS**

```
python3 src/emhass/web_server.py
```

**Run unitests**

```
python3 -m unittest discover -s ./tests -p 'test_*.py'
```

_unitest will need to be installed prior_

### Method 2: VS-Code Debug and Run via DevContainer

In VS-Code, you can run a Docker DevContainer to set up a virtual environment. There you can edit and test EMHASS.

The recommended steps to run are:

- Open forked root (`emhass`) folder inside of VS-Code
- VS-Code will ask if you want to run in a dev-container, say yes _([Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) must be set up first)_. *(Shortcut: `F1` > `Dev Containers: Rebuild and Reopen in Container`)*
- Edit some code...
- Compile emhass by pressing `control+shift+p` > `Tasks: Run Task` > `EMHASS Install`.
  This has been set up in the [tasks.json](https://github.com/davidusb-geek/emhass/blob/master/.vscode/tasks.json) file. - Before _run & debug_, re-run `EMHASS Install` task every time a change has been made to emhass.
- Launch and debug the application via selecting the [`Run and Debug`](https://code.visualstudio.com/docs/editor/debugging) tab /`Ctrl+Shift+D` > `EMHASS run Addon`. This has been set up in the [Launch.json](https://github.com/davidusb-geek/emhass/blob/master/.vscode/launch.json) .

  - You will need to modify the `EMHASS_URL` _(http://HAIPHERE:8123/)_ and `EMHASS_KEY` _(PLACEKEYHERE)_ inside of Launch.json that matches your HA environment before running.
  - If you want to change your parameters, you can edit options.json file before launch.
  - you can also choose to run `EMHASS run` instead of `EMHASS run Addon`. This acts more like standalone mode an removes the use of options.json. _(user sets parameters in config_emhass.yaml instead)_

- You can run all the unitests by heading to the [`Testing`](https://code.visualstudio.com/docs/python/testing) tab on the left hand side.  
  This is recommended before creating a pull request.

### Method 3 - Docker Virtual Environment

With Docker, you can test EMHASS in both standalone and add-on mode via modifying the build argument: `build_version` with values: `standalone`, `addon-pip`, `addon-git`, `addon-local`.  
Since emhass-add-on is using the same docker base, this method is good to test the add-on functionality of your code. _(addon-local)_

Depending on your choice of running standalone or addon, `docker run` will require different passed variables/arguments to function. See following examples:

_Note: Make sure your terminal is in the root `emhass` directory before running the docker build._

#### Docker run add-on via with local files:

**addon-local** copies the local emhass files (from your device) to compile and run in addon mode.

```bash
docker build -t emhass/docker --build-arg build_version=addon-local .

docker run -it -p 5000:5000 --name emhass-container -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
```

**Note:**

- `addon` mode can have secret parameters passed in at run via variables `-e`, arguments (`--key`,`--url`) or via `secrets_emhass.yaml` with a volume mount
- on file change, you will need to re-build and re-run the Docker image/container in order for the change to take effect. (excluding volume mounted configs)
- if you are planning to modify the configs: options.json, secrets_emhass.yaml or config_emhass.yaml, you can volume mount them with `-v`:

  ```bash
  docker build -t emhass/docker --build-arg build_version=addon-local .
  
  docker run -it -p 5000:5000 --name emhass-container -v $(pwd)/options.json:/app/options.json -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
  ```

  This allows the editing of config files without re-building the Docker Image. On config change, restart the container to take effect:

  ```bash
  docker stop emhass-container
  
  docker start emhass-container
  ```

#### Docker run Standalone with local files:

**standalone** copies the local emhass files (from your device) to compile and run in standalone mode.

```bash
docker build -t emhass/docker --build-arg build_version=standalone .

docker run -it -p 5000:5000 --name emhass-container -v $(pwd)/config_emhass.yaml:/app/config_emhass.yaml -v $(pwd)/secrets_emhass.yaml:/app/secrets_emhass.yaml emhass/docker
```

_Standalone mode can use `secrets_emhass.yaml` to pass secret parameters (overriding secrets provided by ARG/ENV's). Copy `secrets_emhass(example).yaml` for an example._

#### Docker run add-on with Git or pip:

If you would like to test with the current production/master versions of emhass, you can do so via pip or Git. With Git, you can also specify other repos/branches outside of `davidusb-geek/emhass:master`. 

**addon-pip** will be the closest environment to the production emhass-add-on.  
However, both come with the disadvantage of not easily being able to edit the emhass package itself.

**Docker run add-on git**

```bash
docker build -t emhass/docker --build-arg build_version=addon-git .

docker run -it -p 5000:5000 --name emhass-container -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" -v $(pwd)/options.json:/app/options.json emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
```

To test a repo and branch outside of `davidusb-geek/emhass:master`:
_(Utilizing build args `build_repo` and `build_branch`)_  
_Linux:_
```bash
repo=https://github.com/davidusb-geek/emhass.git
branch=master

docker build -t emhass/docker --build-arg build_version=addon-git --build-arg build_repo=$repo --build-arg build_branch=$branch .

docker run -it -p 5000:5000 --name emhass-container -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" -v $(pwd)/options.json:/app/options.json emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
```

**Docker run add-on pip:**

```bash
docker build -t emhass/docker --build-arg build_version=addon-pip .

docker run -it -p 5000:5000 --name emhass-container -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" -v $(pwd)/options.json:/app/options.json emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
```
To build with specific pip version, set with build arg: `build_pip_version`: 
```bash
docker build -t emhass/docker --build-arg build_version=addon-pip --build-arg build_pip_version='==0.7.7' .

docker run -it -p 5000:5000 --name emhass-container -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" -v $(pwd)/options.json:/app/options.json emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
```
</br>

_You can add or remove file volume mounts with the `-v` tag, this should override the file in the container (ex. options.json)_
  
#### EMHASS older then **0.7.9** 
For older versions of EMHASS, you may wish to specify the _config_, _data_ and _options_ paths to avoid errors:
```bash
docker run ... -e OPTIONS_PATH='/app/options.json' -e CONFIG_PATH='/app/config_emhass.yaml' -e DATA_PATH='/app/data/'  ...
```
For example pip:
```bash
docker build -t emhass/docker --build-arg build_version=addon-pip .

docker run -it -p 5000:5000 --name emhass-container -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris"  -e CONFIG_PATH='/app/config_emhass.yaml' -e DATA_PATH='/app/data/' -e OPTIONS_PATH='/app/options.json' -v $(pwd)/options.json:/app/options.json emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
```

#### Sync with local data folder 
For those who wish to mount/sync the local `data` folder with the data folder from the docker container, volume mount the data folder with `-v` .
```bash
docker run ... -v $(pwd)/data/:/app/data ...
```

You can also mount data (ex .csv)  files separately
```bash
docker run... -v $(pwd)/data/heating_prediction.csv:/app/data/ ...
```

#### Issue with TARGETARCH
If your docker build fails with an error related to `TARGETARCH`. It may be best to add your devices architecture manually:

Example with armhf architecture 
```bash
docker build ... --build-arg TARGETARCH=armhf --build-arg os_version=raspbian ...
```
*For `armhf` only, create a build-arg for `os_version=raspbian`* 


#### Delete built Docker image

We can delete the Docker image and container via:

```bash
docker rm -f emhass-container #force delete Docker container

docker rmi emhass/docker #delete Docker image
```

#### Other Docker Options

**Rapid Testing**  
As editing and testing EMHASS via docker may be repetitive (rebuilding image and deleting containers), you may want to simplify the removal, build and run process.

**For rapid Docker testing, try a command chain:**  
_Linux:_
```bash
docker build -t emhass/docker --build-arg build_version=addon-local . && docker run --rm -it -p 5000:5000 -v $(pwd)/secrets_emhass.yaml:/app/secrets_emhass.yaml --name emhass-container emhass/docker 
```

_The example command chain rebuilds Docker image, and runs new container with newly built image. `--rm` has been added to the `docker run` to delete the container once ended to avoid manual deletion every time._  
_This use case may not require any volume mounts (unless you use secrets_emhass.yaml) as the Docker build process will pull the latest versions of the configs as it builds._


**Environment Variables**  
Running addon mode, you can also pass location, key and url secret parameters via environment variables.

```bash
docker build -t emhass/docker --build-arg build_version=addon-local .

docker run -it -p 5000:5000 --name emhass-container -e URL="YOURHAURLHERE" -e KEY="YOURHAKEYHERE" -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" emhass/docker
```

This allows the user to set variables prior to build
Linux:

```bash
export EMHASS_URL="YOURHAURLHERE"
export EMHASS_KEY="YOURHAKEYHERE"
export TIME_ZONE="Europe/Paris"
export LAT="45.83"
export LON="6.86"
export ALT="4807.8"

docker build -t emhass/docker --build-arg build_version=addon-local .

docker run -it -p 5000:5000 --name emhass-container -e EMHASS_KEY -e EMHASS_URL -e TIME_ZONE -e LAT -e LON -e ALT emhass/docker
```

### Example Docker testing pipeline 
If you are wishing to test your changes compatibility, check out this example as a template:

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
#testing addon (build and run)
docker build -t emhass/docker --build-arg build_version=addon-local .
docker run --rm -it -p 5000:5000 --name emhass-container -v $(pwd)/data/heating_prediction.csv:/app/data/heating_prediction.csv -v $(pwd)/options.json:/app/options.json -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" emhass/docker --url $HAURL --key $HAKEY
```
```bash
#run actions on a separate terminal
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
#testing standalone (build and run)
docker build -t emhass/docker --build-arg build_version=standalone .
#make secrets_emhass
cat <<EOT > secrets_emhass.yaml
hass_url: $HAURL
long_lived_token: $HAKEY
time_zone: Europe/Paris
lat: 45.83
lon: 6.86
alt: 4807.8
EOT
docker run --rm -it -p 5000:5000 --name emhass-container -v $(pwd)/data/heating_prediction.csv:/app/data/heating_prediction.csv -v $(pwd)/config_emhass.yaml:/app/config_emhass.yaml -v $(pwd)/secrets_emhass.yaml:/app/secrets_emhass.yaml emhass/docker 
```
```bash
#run actions on a separate terminal
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
#testing unittest (run standalone with extra files)
docker run --rm -it -p 5000:5000 --name emhass-container -v $(pwd)/tests/:/app/tests/ -v $(pwd)/data/:/app/data/ -v $(pwd)/"secrets_emhass(example).yaml":/app/"secrets_emhass(example).yaml" -v $(pwd)/options.json:/app/options.json -v $(pwd)/config_emhass.yaml:/app/config_emhass.yaml -v $(pwd)/secrets_emhass.yaml:/app/secrets_emhass.yaml emhass/docker
```
```bash
#run unittest's on separate terminal
docker exec emhass-container apt-get update 
docker exec emhass-container apt-get install python3-requests-mock -y
docker exec emhass-container python3 -m unittest discover -s ./tests -p 'test_*.py' | grep error
```

User may wish to re-test with tweaked parameters such as `lp_solver`, `weather_forecast_method` and `load_forecast_method`, in `config_emhass.yaml` *(standalone)* or `options.json` *(addon)*, to broaden the testing scope. 
*see [EMHASS & EMHASS-Add-on differences](https://emhass.readthedocs.io/en/latest/differences.html) for more information on how these config_emhass & options files differ*

*Note: may need to set `--build-arg TARGETARCH=YOUR-ARCH` in docker build*

## Step 3 - Pull request

Once developed, commit your code, and push to your fork.
Then submit a pull request with your fork to the [davidusb-geek/emhass@master](https://github.com/davidusb-geek/emhass) repository.
