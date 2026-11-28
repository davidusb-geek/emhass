# EMHASS Development

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Setup

There are multiple different approaches to developing for EMHASS.  
The choice depends on your and preference (Python venv/DevContainer/Docker).  
Below are some development workflow examples:  
_Note: It is preferred to run the actions and unittest once before submitting and pull request._

### Step 1 - Fork

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

### Step 2 - Develop

To develop and test code choose one of the following methods:

#### Method 1 - Python Virtual Environment

We can use python virtual environments to build, develop and test/unittest the code.

_confirm terminal is in the root `emhass` directory before starting_

**Create a developer environment:**

Use a Python interpreter satisfying `requires-python` in `pyproject.toml`; see `.github/workflows/python-test.yml` for the interpreter/OS combinations currently tested in CI.

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
  ```console
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

#### Method 2: VS-Code Debug and Run via Dev Container

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

Depending on the method you wish to test, the `docker run` command will require different passed arguments to function. See following examples.

The web server requires an API token (see [API authentication](api_authentication.md)). For a throw-away development container, make a token first in a POSIX shell (`export EMHASS_API_TOKEN="$(openssl rand -hex 32)"`), pass it with `-e EMHASS_API_TOKEN` and send it as `Authorization: Bearer $EMHASS_API_TOKEN`, or, on a network you fully control, `-e EMHASS_API_AUTH_DISABLED=true`. The test suite sets the opt-out for the older web tests in `tests/conftest.py`; `tests/test_api_auth.py`, `tests/test_api_ingress.py` and `tests/test_api_provisioning.py` test the real behaviour.

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
docker build -t emhass/test --build-arg BUILD_VERSION=addon-local .

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

docker build -t emhass/test --build-arg BUILD_VERSION=addon-local .

docker run -it -p 5000:5000 --name emhass-test -e EMHASS_KEY -e EMHASS_URL -e TIME_ZONE -e LAT -e LON -e ALT emhass/test
```

#### Example Docker testing pipeline 
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
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H 'Content-Type:application/json' -X POST -d '{"pv_power_forecast":[0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93], "prediction_horizon":10, "soc_init":0.5,"soc_final":0.6}' http://localhost:5000/action/naive-mpc-optim
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/perfect-optim
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/dayahead-optim
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/forecast-model-fit
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/forecast-model-predict
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/forecast-model-tune
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/forecast-calibration
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H 'Content-Type:application/json' -X POST -d '{"p10":[80,80,80],"p50":[100,100,100],"actual":[70,90,90],"target_shortfall_rate":0.10}' http://localhost:5000/action/pv-bias-calibration
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H "Content-Type:application/json" -X POST -d  '{"csv_file": "heating_prediction.csv", "features": ["degreeday", "solar"], "target": "hour", "regression_model": "RandomForestRegression", "model_type": "heating_hours_degreeday", "timestamp": "timestamp", "date_features": ["month", "day_of_week"], "new_values": [12.79, 4.766, 1, 2] }' http://localhost:5000/action/regressor-model-fit
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H "Content-Type:application/json" -X POST -d  '{"mlr_predict_entity_id": "sensor.mlr_predict", "mlr_predict_unit_of_measurement": "h", "mlr_predict_friendly_name": "mlr predictor", "new_values": [8.2, 7.23, 2, 6], "model_type": "heating_hours_degreeday" }' http://localhost:5000/action/regressor-model-predict
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -i -H 'Content-Type:application/json' -X POST -d {} http://localhost:5000/action/publish-data
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

#### Adding a parameter
When enhancing EMHASS, users may like to add or modify the EMHASS parameters. To add a new parameter see the following steps:

*Example parameter = `this_parameter_is_amazing`*

1. Append a line into `associations.csv` :
*So that build_params() knows what config catagorie to allocate the parameter*
```text
...
retrieve_hass_conf,,this_parameter_is_amazing
```
 - Alternatively if you want to support this parameter with the yaml conversion *(Ie. allow the parameter to be converted from config_emhass.yaml)*
    ```text
    ...
    retrieve_hass_conf,his_parameter_is_amazing,this_parameter_is_amazing
    ```

2. Append a line into the `config_defaults.json`
*To set a default value for the user if none is provided in `config.json`*
```json
"...": "...",
  "this_parameter_is_amazing": [
    0.1,
    0.1
  ]
```

3. Update the Web UI in `param_definitions.json`
To support the configuration website to generate the parameter in the list view, append the `param_definitions.json` file:
```json
"this_parameter_is_amazing": {
      "friendly_name": "This parameter is amazing",
      "Description": "This parameter functions as you expect. It makes EMHASS AMAZING!",
      "input": "array.float",
      "default_value": 0.777
    }
```
*Note: The `default_value` in this case acts (or should act) as last resort fallback if default_config.json is not found. It also acts as the default value when you append (press plus) to an array.\* parameter*

*Note: a static table-valued parameter (`input: "array.array.float"` with a list `default_value`, such as `battery_charge_power_derating` or the `inverter_power_curve_*` curves) is edited in the list view as one JSON text field and saved with `JSON.parse`; the UI only checks that it is nested numeric arrays, so validate the content in Python. A nullable per-load nested array (`default_value: null`) keeps the per-load rendering.*

![Screenshot from 2024-09-09 16-45-32](https://github.com/user-attachments/assets/01e7984f-3332-4e25-8076-160f51a2e0c4)

If you are only adding another option for a existing parameter, editing `param_definitions.json` file should be all you need (allowing the user to select the option from the configuration page):
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

4. Check the Optimization Cache Key (`command_line.py`)
If your new parameter affects the mathematical structure of the optimization problem (e.g., adding constraints, changing binary variables, or adding penalty weights), it must trigger a cache miss when changed.

Usually nothing needs to be added. `_compute_cache_key` already hashes every `plant_conf` key (`plant_conf_hash`) and every `optim_conf` key (`optim_conf_structural_hash`) except those in the `plant_runtime_keys` / `optim_conf_runtime_keys` sets, which are the values updated per call without a rebuild (CVXPY Parameters, solver options, forecast-method selectors). A new structural parameter in `plant_conf` or `optim_conf` is therefore covered automatically: do NOT add a redundant field. Instead, add a test that a change to its value changes the key (see `test_curves_invalidate_the_optimization_cache_through_plant_conf_hash` in `tests/test_inverter_power_curves.py`).

Add an explicit field to the `OptimizationCacheKey` dataclass and to `_compute_cache_key` only if the parameter is *not* covered by those hashes: it lives outside `plant_conf`/`optim_conf` (e.g. `retrieve_hass_conf`), is listed in a runtime-keys set but still changes the structure, or needs a canonical form first (e.g. scalar and singleton list must share one key):
```python
@dataclass(frozen=True)
class OptimizationCacheKey:
    # ... existing parameters ...
    this_parameter_is_amazing: tuple


# Inside _compute_cache_key:
return OptimizationCacheKey(
    # ... existing parameters ...
    this_parameter_is_amazing=to_tuple(optim_conf.get("this_parameter_is_amazing", [])),
)
```

#### Note on Deferrable Load Lists
If your parameter is a list that must match the `number_of_deferrable_loads` (like `set_deferrable_max_startups`), you must also ensure it gets padded correctly in `utils.py`. Add a call to `check_def_loads()` inside the `build_params()` method:
```json
optim_conf["this_parameter_is_amazing"] = check_def_loads(
    optim_conf["number_of_deferrable_loads"],
    optim_conf,
    0, # Your default pad value
    "this_parameter_is_amazing",
    logger,
)
```

#### Changing default values
When changing the default value of an *existing* parameter (as opposed to adding a new one), `src/emhass/static/data/param_definitions.json` is the source of truth. The GUI config form reads `param_definitions.json` and writes user-saved values into `config.json`, which then wins at runtime. `src/emhass/data/config_defaults.json` is the loader fallback for headless installs (Docker without GUI save, `options.json`-only).
Edit order:
1. Update `default_value` in `param_definitions.json` (and any matching description text such as "Defaults to N" mentions).
2. Update the same key in `config_defaults.json` to match. The two files must agree; drift produces silent default-divergence between GUI and headless modes.

### Step 3 - Pull request

Once developed, commit your code, and push the commit to your fork on GitHub.
Once ready, submit a pull request with your fork to the [davidusb-geek/emhass@master](https://github.com/davidusb-geek/emhass) repository.

#### SonarQube Cloud analysis on pull requests

The `CodeCov` workflow generates `coverage.xml`, uploads it to Codecov and runs the SonarQube Cloud scan. The scan waits up to 300 seconds for the quality gate, and the job fails if the gate fails.

GitHub does not give repository secrets such as `SONAR_TOKEN` to workflows triggered by pull requests from forks, so for those pull requests the upload and scan steps are skipped. **A green `CodeCov` check therefore does not mean that Sonar analysed the change.** Check the steps in the job log:

| Result | What you see |
| --- | --- |
| Skipped | The `SonarQube Scan` step is skipped on a fork pull request. |
| Passed | The step succeeds and its log reports `QUALITY GATE STATUS: PASSED`. |
| Failed | The step fails and its log reports `QUALITY GATE STATUS: FAILED`. |
| Unavailable | No verdict was reported: the step failed for another reason (token, network, timeout), or it never ran because an earlier step failed. |

Dependabot pull requests use branches in this repository, but runs that Dependabot triggers receive only Dependabot secrets. The Codecov upload can then fail before the scan starts, so the result is unavailable.

A pull request run checks out and scans a temporary merge of the head into `master`, not the head alone. The checkout step logs it as `HEAD is now at <merge-sha> Merge <head-sha> into <base-sha>`. The scanner nevertheless records the head commit as the analysed revision (`SCM revision ID` in its log), so that SHA alone does not show which `master` commit was included. If the head conflicts with `master`, the workflow does not run and the result is unavailable.

Only "Passed" counts as a pass. For a failed gate, follow the dashboard link in the log, fix the findings and run the analysis again on the new revision. On a timeout or other unavailable result, re-run the job (but never on a temporary verification pull request, see below); the analysis is not verified until a verdict is reported. On pushes to `master` the gate covers the project's whole new code period, so it can fail because of findings merged earlier.

The check is not required for every pull request, because fork pull requests cannot produce it.

#### Maintainer verification of a fork pull request

A maintainer can analyse a fork contribution by opening a temporary pull request from a branch in this repository. **This runs the contributor's code with this repository's credentials.** Use it only when a maintainer responsible for those credentials has explicitly decided to trust the contribution and authorised running it, after reviewing all code that will be executed and every dependency it adds or changes. A quick or partial review is not enough. **Review does not isolate credentials.** It can miss things, and it does not cover what dependencies and build tools download and execute, so credentials can still be exposed. If trust cannot be established, do not use this procedure: the Sonar result for that pull request stays unavailable.

Opening that pull request to `master` starts these workflows. `Check CEC databases` sets its permissions for the whole workflow; the others set them per job. The table lists repository secrets and write permissions, not read permissions:

| Workflow and job | Use of contributor files | Secrets | Write permissions |
| --- | --- | --- | --- |
| `CodeCov`: `run` | Installs the project and runs the tests | `SONAR_TOKEN`, `CODECOV_TOKEN` | None |
| `Code Quality Scan`: `build` | Runs a code review action and installs the project | `SOURCERY_TOKEN` | None |
| `Code Quality Scan`: `scan-pr` | Dependency scan | None | Security results |
| `Python test`: `build` | Installs the project and runs the tests on Linux, macOS and Windows | None | None |
| `Python test`: `scan-pr` | Dependency scan | None | Security results |
| `Test Building Docker Image`: `build` | Builds and runs the image | None | None |
| `Test Building Docker Image`: `osv-scan` | Scans the package lists from the build | None | Security results |
| `CodeQL`: `analyze` | Static analysis | None | Security results |
| `Check CEC databases`: `check` (only if the pull request changes a file in its path filter) | Runs the database build script, with package versions named in the pull request | None | None |

Treat every secret and token in a job as available to any code that runs in it, including code that runs before the step that uses it:

- **Checkout:** `actions/checkout` by default keeps the job's `GITHUB_TOKEN` available to later steps.
- **Dependencies:** installing, testing and building download and execute third-party code. The `uv.lock` you reviewed does not fix everything that runs. `Python test` installs with `--upgrade`, `uv sync` re-resolves a lock file that is out of date, and `CodeCov` and `Code Quality Scan` also install tools that are not in the lock file.
- **Caches:** jobs can restore caches saved from `master`. GitHub limits a cache saved by a pull request run to that pull request.

If `.github/` differs from `master`, the table no longer holds, and changed workflows can run as soon as the branch is pushed. Do not run any workflow manually on the temporary branch, because the publishing workflows could publish its code.

In the commands below, `upstream` is the remote pointing at `davidusb-geek/emhass`.

1. Pin the head and the base before you review. Record the full 40-character SHAs printed by these commands, the pull request head and current `master`:

   ```bash
   git fetch --no-tags upstream refs/pull/<number>/head && git rev-parse 'FETCH_HEAD^{commit}'
   git fetch --no-tags upstream refs/heads/master && git rev-parse 'FETCH_HEAD^{commit}'
   ```
2. Review the complete change between those two commits (`git diff --no-ext-diff --no-textconv <reviewed-base-sha>...<reviewed-head-sha>`, so that no local diff or text conversion program runs on the contributor's files), including build and dependency files (`pyproject.toml`, `uv.lock`, `Dockerfile`), tests and `conftest.py`, scripts, and `sonar-project.properties`. Do not run this verification if you cannot account for every change and accept the contribution as trusted. Then run the following in a `bash` shell, and continue only if it prints the `OK` line:

   ```bash
   unset verified verified_base
   git fetch --no-tags upstream refs/heads/master &&
   base=$(git rev-parse --verify 'FETCH_HEAD^{commit}') &&
   test "$base" = <reviewed-base-sha> &&
   git fetch --no-tags upstream refs/pull/<number>/head &&
   head=$(git rev-parse --verify 'FETCH_HEAD^{commit}') &&
   test "$head" = <reviewed-head-sha> &&
   git diff --no-ext-diff --no-textconv --exit-code "$base" "$head" -- ':(top).github/' &&
   verified=$head verified_base=$base &&
   echo "OK head=$head base=$base"
   ```

   It stops if a fetch fails, if the pull request head or `master` is not the commit you recorded in step 1, or if `.github/` differs from current `master` (which also happens when `master` changed `.github/` after the branch was created).
3. In the same shell, run the following. It fetches again and pushes the reviewed commit only if step 2 succeeded and neither `master` nor the pull request head has changed since. `<n>` is a number you have not used before for this pull request.

   ```bash
   test -n "$verified" &&
   git fetch --no-tags upstream refs/heads/master &&
   test "$(git rev-parse --verify 'FETCH_HEAD^{commit}')" = "$verified_base" &&
   git fetch --no-tags upstream refs/pull/<number>/head &&
   test "$(git rev-parse --verify 'FETCH_HEAD^{commit}')" = "$verified" &&
   git push upstream "$verified:refs/heads/verify/pr-<number>-<n>"
   ```

   If it does not push, start again from step 1. The check narrows the window but cannot close it: `master` or the pull request can still change after the check and before the workflows run.
4. Open a draft pull request from that branch to `master`, titled so that it is clearly not for merging.
5. Wait for the `CodeCov` job and note its run ID and attempt number. Check GitHub's own record of that run:

   ```bash
   gh run view <run-id> --repo davidusb-geek/emhass --attempt <attempt> --json event,headBranch,headSha,attempt
   ```

   `event` must be `pull_request`, `headBranch` the temporary branch and `headSha` the reviewed head SHA. Then, in the log of its checkout step, find the line `HEAD is now at <merge-sha> Merge <head-sha> into <base-sha>`. The merge commit is what was checked out and scanned. `<head-sha>` must be the head you reviewed and `<base-sha>` the base you recorded in step 1. Otherwise the result is unverified, although the code has already run.
6. In the log of the `SonarQube Scan` step of that same run and attempt, confirm all of the following:
   - the scanner configured itself for the temporary pull request's number;
   - `SCM revision ID` is the reviewed head SHA, not the merge SHA;
   - a `QUALITY GATE STATUS` line gives the verdict;
   - the `ANALYSIS SUCCESSFUL` link is for the temporary pull request's number.

   The `SonarCloud Code Analysis` check and the SonarQube Cloud page are supporting evidence only. The same head SHA also belongs to the original pull request, and they show a latest analysis that may not be this run's. If anything above is missing, ambiguous or contradicted by them, or the temporary pull request was reopened or updated, the result is unverified. Neither the job log nor SonarQube Cloud can prove the result for malicious code: it runs earlier in the same job and can alter the files, the log or the submitted analysis.
7. Stop the temporary pull request's runs before cleaning up. Waiting for `CodeCov` is not enough: other workflows can still be running the contributor's code, and you must not rely on closing the pull request or deleting the branch to stop them. Check every run on the temporary branch, reading all pages of the result:

   ```bash
   runs=$(gh api --paginate 'repos/davidusb-geek/emhass/actions/runs?branch=verify/pr-<number>-<n>&per_page=100' \
     --jq '.total_count as $total | .workflow_runs[] | "\($total) \(.id) \(.status) \(.name)"') &&
   printf '%s\n' "$runs" | awk '
     NF { total[$1]; seen[$2]; expected = $1; if ($3 != "completed") { print "ACTIVE", $0; active = 1 } }
     END {
       for (t in total) totals++
       for (i in seen) found++
       if (totals != 1 || found != expected || expected > 1000) { print "INCOMPLETE LISTING"; exit 1 }
       if (active) exit 1
       print "ALL COMPLETED: " found " runs"
     }'
   ```

   Only the `ALL COMPLETED` line counts as success. An error, `INCOMPLETE LISTING` or no output means the state is unknown, so treat runs as still active. GitHub returns at most 1,000 runs for this search; beyond that, use the repository's Actions page. For each `ACTIVE` line, cancel the run ID (the second number) with `gh run cancel <run-id> --repo davidusb-geek/emhass`, adding `--force` if it does not stop. New runs can take a moment to appear: repeat the check until it prints `ALL COMPLETED` with the same count twice, at least two minutes apart, and confirm on the Actions page that every workflow you expected has run.
8. Save the logs (`gh run view <run-id> --repo davidusb-geek/emhass --log > run-<run-id>.log`) and do not delete the runs. Close the temporary pull request without merging it, delete the branch, then run the check from step 7 again. Cleanup has succeeded only if it still prints `ALL COMPLETED`. **Never re-run any of them:** a re-run executes the same commit again with the job's credentials, and GitHub allows it for up to 30 days. If the result was unavailable, start again from step 1 with a new `<n>`.
9. Comment on the original pull request with the temporary pull request number, the head, base and merge SHAs, the result, the link to the workflow run and attempt, and the Sonar link from that log.

**If the code did something you did not expect, or you suspect a compromise,** act at once, without waiting for any job to finish: cancel all runs of the temporary branch, do not re-run them, save their logs, and promptly revoke or rotate `SONAR_TOKEN`, `CODECOV_TOKEN` and `SOURCERY_TOKEN`. Do not let a run that will not stop, or a listing you cannot complete, delay the rotation. Cancelling does not undo what has already been sent elsewhere, so rotate even if you cancelled quickly. Then check security alerts, code scanning results, workflow runs and branches.

The result applies only to the head and base SHAs from step 5. It is stale as soon as the original pull request's head or `master` differs from them: verify again, with a new review if the head changed.
