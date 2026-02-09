#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
import pickle
import re
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import aiofiles
import jinja2
import orjson
import uvicorn
import yaml
from markupsafe import Markup
from quart import Quart, make_response, request
from quart import logging as log

from emhass.command_line import (
    continual_publish,
    dayahead_forecast_optim,
    export_influxdb_to_csv,
    forecast_model_fit,
    forecast_model_predict,
    forecast_model_tune,
    naive_mpc_optim,
    perfect_forecast_optim,
    publish_data,
    regressor_model_fit,
    regressor_model_predict,
    set_input_data_dict,
    weather_forecast_cache,
)
from emhass.connection_manager import close_global_connection, get_websocket_client, is_connected
from emhass.utils import (
    build_config,
    build_legacy_config_params,
    build_params,
    build_secrets,
    get_injection_dict,
    get_injection_dict_forecast_model_fit,
    get_injection_dict_forecast_model_tune,
    get_keys_to_mask,
    param_to_config,
)

app = Quart(__name__)

emhass_conf: dict[str, Path] = {}
entity_path: Path = Path()
params_secrets: dict[str, str | float] = {}
continual_publish_thread: list = []
injection_dict: dict = {}

templates = jinja2.Environment(
    autoescape=True,
    loader=jinja2.PackageLoader("emhass", "templates"),
)

action_log_str = "action_logs.txt"
injection_dict_file = "injection_dict.pkl"
params_file = "params.pkl"
error_msg_associations_file = "Unable to obtain associations file"


# Add custom filter for trusted HTML content
def mark_safe(value):
    """Mark pre-rendered HTML plots as safe (use only for trusted content)"""
    if value is None:
        return ""
    return Markup(value)


templates.filters["mark_safe"] = mark_safe


# Register async startup and shutdown handlers
@app.before_serving
async def before_serving():
    """Initialize EMHASS before starting to serve requests."""
    # Initialize the application
    try:
        await initialize()
        app.logger.info("Full initialization completed")
    except Exception as e:
        app.logger.warning(f"Full initialization failed (this is normal in test environments): {e}")
        app.logger.info("Continuing without WebSocket connection...")
        # The initialize() function already sets up all necessary components except WebSocket
        # So we can continue serving requests even if WebSocket connection fails


@app.after_serving
async def after_serving():
    """Clean up resources after serving."""
    try:
        # Only close WebSocket connection if it was established
        if is_connected():
            await close_global_connection()
            app.logger.info("WebSocket connection closed")
        else:
            app.logger.info("No WebSocket connection to close")
    except Exception as e:
        app.logger.warning(f"WebSocket shutdown failed: {e}")
    app.logger.info("Quart shutdown complete")


async def check_file_log(ref_string: str | None = None) -> bool:
    """
    Check logfile for error, anything after string match if provided.

    :param ref_string: String to reduce log area to check for errors. Use to reduce log to check anything after string match (ie. an action).
    :type ref_string: str
    :return: Boolean return if error was found in logs
    :rtype: bool

    """
    log_array: list[str] = []

    if ref_string is not None:
        log_array = await grab_log(
            ref_string
        )  # grab reduced log array (everything after string match)
    else:
        if (emhass_conf["data_path"] / action_log_str).exists():
            async with aiofiles.open(str(emhass_conf["data_path"] / action_log_str)) as fp:
                content = await fp.read()
                log_array = content.splitlines()
        else:
            app.logger.debug("Unable to obtain {action_log_str}")
            return False

    for log_string in log_array:
        if log_string.split(" ", 1)[0] == "ERROR":
            return True
    return False


async def grab_log(ref_string: str | None = None) -> list[str]:
    """
    Find string in logs, append all lines after into list to return.

    :param ref_string: String used to string match log.
    :type ref_string: str
    :return: List of lines in log after string match.
    :rtype: list

    """
    is_found = []
    output = []
    if (emhass_conf["data_path"] / action_log_str).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / action_log_str)) as fp:
            content = await fp.read()
            log_array = content.splitlines()
        # Find all string matches, log key (line Number) in is_found
        for x in range(len(log_array) - 1):
            if re.search(ref_string, log_array[x]):
                is_found.append(x)
        if len(is_found) != 0:
            # Use last item in is_found to extract action logs
            for x in range(is_found[-1], len(log_array)):
                output.append(log_array[x])
    return output


# Clear the log file
async def clear_file_log():
    """
    Clear the contents of the log file

    """
    if (emhass_conf["data_path"] / action_log_str).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / action_log_str), "w") as fp:
            await fp.write("")


@app.route("/")
@app.route("/index")
async def index():
    """
    Render initial index page and serve to web server.
    Appends plot tables saved from previous optimization into index.html, then serves.
    """
    app.logger.info("EMHASS server online, serving index.html...")

    # Load cached dict (if exists), to present generated plot tables
    if (emhass_conf["data_path"] / injection_dict_file).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / injection_dict_file), "rb") as fid:
            content = await fid.read()
            injection_dict = pickle.loads(content)
    else:
        app.logger.info(
            "The data container dictionary is empty... Please launch an optimization task"
        )
        injection_dict = {}

    template = templates.get_template("index.html")
    return await make_response(template.render(injection_dict=injection_dict))


@app.route("/configuration", methods=["GET", "POST"])
async def configuration():
    """
    Configuration page actions:
    Render and serve configuration page html
    """
    # Define the list of secret parameters managed by the UI
    secret_params = get_keys_to_mask()

    if request.method == "POST":
        app.logger.info("Saving configuration/secrets...")
        form_data = await request.form

        # Load existing secrets
        secrets = {}
        # Ensure we have the path from config, fallback to default if missing
        secrets_path = emhass_conf.get("secrets_path", Path("/app/secrets_emhass.yaml"))

        # Try to load existing secrets to preserve others (Async)
        if secrets_path.exists():
            try:
                async with aiofiles.open(secrets_path) as file:
                    content = await file.read()
                    loaded = yaml.safe_load(content)
                    if loaded:
                        secrets = loaded
            except Exception as e:
                app.logger.error(f"Error reading secrets file: {e}")

        # Update secrets with form data
        updated = False
        for key in secret_params:
            if key in form_data:
                value = form_data[key]
                if value != "***":
                    secrets[key] = value
                    updated = True

        # Save to file if changes were made (Async)
        if updated:
            try:
                async with aiofiles.open(secrets_path, "w") as file:
                    # dump returns string if stream is None
                    content = yaml.dump(secrets, default_flow_style=False)
                    await file.write(content)

                app.logger.info("Secrets saved successfully.")

                # Update the global params_secrets
                global params_secrets
                params_secrets.update(secrets)

            except Exception as e:
                app.logger.error(f"Error saving secrets file: {e}")

    app.logger.info("serving configuration.html...")

    # get params
    if (emhass_conf["data_path"] / params_file).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / params_file), "rb") as fid:
            content = await fid.read()
            _, params = pickle.loads(content)  # Don't overwrite emhass_conf["config_path"]
    else:
        params = {}

    template = templates.get_template("configuration.html")
    return await make_response(template.render(config=params))


@app.route("/template", methods=["GET"])
async def template_action():
    """
    template page actions:
    Render and serve template html
    """
    app.logger.info(" >> Sending rendered template data")

    if (emhass_conf["data_path"] / injection_dict_file).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / injection_dict_file), "rb") as fid:
            content = await fid.read()
            injection_dict = pickle.loads(content)
    else:
        app.logger.warning("Unable to obtain plot data from {injection_dict_file}")
        app.logger.warning("Try running an launch an optimization task")
        injection_dict = {}

    template = templates.get_template("template.html")
    return await make_response(template.render(injection_dict=injection_dict))


@app.route("/get-config", methods=["GET"])
async def parameter_get():
    """
    Get request action that builds, formats and sends config as json (config.json format)

    """
    app.logger.debug("Obtaining current saved parameters as config")
    # Build config from all possible sources (inc. legacy yaml config)
    config = await build_config(
        emhass_conf,
        app.logger,
        str(emhass_conf["defaults_path"]),
        str(emhass_conf["config_path"]),
        str(emhass_conf["legacy_config_path"]),
    )
    if type(config) is bool and not config:
        return await make_response(["failed to retrieve default config file"], 500)
    # Format parameters in config with params (converting legacy json parameters from options.json if any)
    params = await build_params(emhass_conf, {}, config, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)
    # Covert formatted parameters from params back into config.json format
    return_config = param_to_config(params, app.logger)
    # Send config
    return await make_response(return_config, 201)


# Get default Config
@app.route("/get-config/defaults", methods=["GET"])
async def config_get():
    """
    Get request action, retrieves and sends default configuration

    """
    app.logger.debug("Obtaining default parameters")
    # Build config, passing only default file
    config = await build_config(emhass_conf, app.logger, str(emhass_conf["defaults_path"]))
    if type(config) is bool and not config:
        return await make_response(["failed to retrieve default config file"], 500)
    # Format parameters in config with params
    params = await build_params(emhass_conf, {}, config, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)
    # Covert formatted parameters from params back into config.json format
    return_config = param_to_config(params, app.logger)
    # Send params
    return await make_response(return_config, 201)


# Get YAML-to-JSON config
@app.route("/get-json", methods=["POST"])
async def json_convert():
    """
    Post request action, receives yaml config (config_emhass.yaml or EMHASS-Add-on config page) and converts to config json format.

    """
    app.logger.info("Attempting to convert YAML to JSON")
    data = await request.get_data()
    yaml_config = yaml.safe_load(data)

    # If filed to Parse YAML
    if yaml_config is None:
        return await make_response(["failed to Parse YAML from data"], 400)
    # Test YAML is legacy config format (from config_emhass.yaml)
    test_legacy_config = await build_legacy_config_params(emhass_conf, yaml_config, app.logger)
    if test_legacy_config:
        yaml_config = test_legacy_config
    # Format YAML to params (format params. check if params match legacy option.json format)
    params = await build_params(emhass_conf, {}, yaml_config, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)
    # Covert formatted parameters from params back into config.json format
    config = param_to_config(params, app.logger)
    # convert json to str
    config = orjson.dumps(config).decode()

    # Send params
    return await make_response(config, 201)


@app.route("/set-config", methods=["POST"])
async def parameter_set():
    """
    Receive JSON config, and save config to file (config.json and param.pkl)

    """
    config = {}
    if not emhass_conf["defaults_path"]:
        return await make_response(["Unable to Obtain defaults_path from emhass_conf"], 500)
    if not emhass_conf["config_path"]:
        return await make_response(["Unable to Obtain config_path from emhass_conf"], 500)

    # Load defaults as a reference point (for sorting) and a base to override
    if (
        os.path.exists(emhass_conf["defaults_path"])
        and Path(emhass_conf["defaults_path"]).is_file()
    ):
        async with aiofiles.open(str(emhass_conf["defaults_path"])) as data:
            content = await data.read()
            config = orjson.loads(content)
    else:
        app.logger.warning(
            "Unable to obtain default config. only parameters passed from request will be saved to config.json"
        )

    # Retrieve sent config json
    request_data = await request.get_json(force=True)

    # check if data is empty
    if len(request_data) == 0:
        return await make_response(["failed to retrieve config json"], 400)

    # Format config by converting to params (format params. check if params match legacy option.json format. If so format)
    params = await build_params(emhass_conf, params_secrets, request_data, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)

    # Covert formatted parameters from params back into config.json format.
    # Overwrite existing default parameters in config
    config.update(param_to_config(params, app.logger))

    # Save config to config.json
    if os.path.exists(emhass_conf["config_path"].parent):
        async with aiofiles.open(str(emhass_conf["config_path"]), "w") as f:
            await f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2).decode())
    else:
        return await make_response(["Unable to save config file"], 500)

    # Save params with updated config
    if os.path.exists(emhass_conf["data_path"]):
        async with aiofiles.open(str(emhass_conf["data_path"] / params_file), "wb") as fid:
            content = pickle.dumps(
                (
                    emhass_conf["config_path"],
                    await build_params(emhass_conf, params_secrets, config, app.logger),
                )
            )
            await fid.write(content)
    else:
        return await make_response(["Unable to save params file, missing data_path"], 500)

    app.logger.info("Saved parameters from webserver")
    return await make_response({}, 201)


async def _load_params_and_runtime(request, emhass_conf, logger):
    """
    Loads configuration parameters from pickle and runtime parameters from the request.
    Returns a tuple (params, costfun, runtimeparams) or raises an exception/returns None on failure.
    """
    action_str = " >> Obtaining params: "
    logger.info(action_str)

    # Load params.pkl
    params = None
    costfun = "profit"
    params_path = emhass_conf["data_path"] / params_file

    if params_path.exists():
        async with aiofiles.open(str(params_path), "rb") as fid:
            content = await fid.read()
            _, params = pickle.loads(content)  # Don't overwrite emhass_conf["config_path"]
            # Set local costfun variable
            if params.get("optim_conf") is not None:
                costfun = params["optim_conf"].get("costfun", "profit")
            params = orjson.dumps(params).decode()
    else:
        logger.error("Unable to find params.pkl file")
        return None, None, None

    # Load runtime params
    try:
        runtimeparams = await request.get_json(force=True)
        if runtimeparams:
            logger.info("Passed runtime parameters: " + str(runtimeparams))
        else:
            runtimeparams = {}
    except Exception as e:
        logger.error(f"Error parsing runtime params JSON: {e}")
        logger.error("Check your payload for syntax errors (e.g., use 'false' instead of 'False')")
        runtimeparams = {}

    runtimeparams = orjson.dumps(runtimeparams).decode()

    return params, costfun, runtimeparams


async def _handle_action_dispatch(
    action_name, input_data_dict, emhass_conf, params, runtimeparams, logger
):
    """
    Dispatches the specific logic based on the action_name.
    Returns (response_msg, status_code).
    """
    # Actions that don't require input_data_dict or have specific flows
    if action_name == "weather-forecast-cache":
        action_str = " >> Performing weather forecast, try to caching result"
        logger.info(action_str)
        await weather_forecast_cache(emhass_conf, params, runtimeparams, logger)
        return "EMHASS >> Weather Forecast has run and results possibly cached... \n", 201

    if action_name == "export-influxdb-to-csv":
        action_str = " >> Exporting InfluxDB data to CSV..."
        logger.info(action_str)
        success = await export_influxdb_to_csv(None, logger, emhass_conf, params, runtimeparams)
        if success:
            return "EMHASS >> Action export-influxdb-to-csv executed successfully... \n", 201
        return await grab_log(action_str), 400

    # Actions requiring input_data_dict
    if action_name == "publish-data":
        action_str = " >> Publishing data..."
        logger.info(action_str)
        _ = await publish_data(input_data_dict, logger)
        return "EMHASS >> Action publish-data executed... \n", 201

    # Mapping for optimization actions to their functions
    optim_actions = {
        "perfect-optim": perfect_forecast_optim,
        "dayahead-optim": dayahead_forecast_optim,
        "naive-mpc-optim": naive_mpc_optim,
    }

    if action_name in optim_actions:
        action_str = f" >> Performing {action_name}..."
        logger.info(action_str)
        opt_res = await optim_actions[action_name](input_data_dict, logger)
        injection_dict = get_injection_dict(opt_res)
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return f"EMHASS >> Action {action_name} executed... \n", 201

    # Delegate Machine Learning actions to helper
    ml_response = await _handle_ml_actions(action_name, input_data_dict, emhass_conf, logger)
    if ml_response:
        return ml_response

    # Fallback for invalid action
    logger.error("ERROR: passed action is not valid")
    return "EMHASS >> ERROR: Passed action is not valid... \n", 400


async def _handle_ml_actions(action_name, input_data_dict, emhass_conf, logger):
    """
    Helper function to handle Machine Learning specific actions.
    Returns (msg, status) if action is handled, otherwise None.
    """
    # forecast-model-fit
    if action_name == "forecast-model-fit":
        action_str = " >> Performing a machine learning forecast model fit..."
        logger.info(action_str)
        df_fit_pred, _, mlf = await forecast_model_fit(input_data_dict, logger)
        injection_dict = get_injection_dict_forecast_model_fit(df_fit_pred, mlf)
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action forecast-model-fit executed... \n", 201

    # forecast-model-predict
    if action_name == "forecast-model-predict":
        action_str = " >> Performing a machine learning forecast model predict..."
        logger.info(action_str)
        df_pred = await forecast_model_predict(input_data_dict, logger)
        if df_pred is None:
            return await grab_log(action_str), 400

        table1 = df_pred.reset_index().to_html(classes="mystyle", index=False)
        injection_dict = {
            "title": "<h2>Custom machine learning forecast model predict</h2>",
            "subsubtitle0": "<h4>Performed a prediction using a pre-trained model</h4>",
            "table1": table1,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action forecast-model-predict executed... \n", 201

    # forecast-model-tune
    if action_name == "forecast-model-tune":
        action_str = " >> Performing a machine learning forecast model tune..."
        logger.info(action_str)
        df_pred_optim, mlf = await forecast_model_tune(input_data_dict, logger)
        if df_pred_optim is None or mlf is None:
            return await grab_log(action_str), 400

        injection_dict = get_injection_dict_forecast_model_tune(df_pred_optim, mlf)
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action forecast-model-tune executed... \n", 201

    # regressor-model-fit
    if action_name == "regressor-model-fit":
        action_str = " >> Performing a machine learning regressor fit..."
        logger.info(action_str)
        await regressor_model_fit(input_data_dict, logger)
        return "EMHASS >> Action regressor-model-fit executed... \n", 201

    # regressor-model-predict
    if action_name == "regressor-model-predict":
        action_str = " >> Performing a machine learning regressor predict..."
        logger.info(action_str)
        await regressor_model_predict(input_data_dict, logger)
        return "EMHASS >> Action regressor-model-predict executed... \n", 201

    return None


async def _save_injection_dict(injection_dict, data_path):
    """Helper to save injection dict to pickle."""
    async with aiofiles.open(str(data_path / injection_dict_file), "wb") as fid:
        content = pickle.dumps(injection_dict)
        await fid.write(content)


@app.route("/action/<action_name>", methods=["POST"])
async def action_call(action_name: str):
    """
    Receive Post action, run action according to passed slug(action_name)
    """
    global continual_publish_thread
    global injection_dict

    # Load Parameters
    params, costfun, runtimeparams = await _load_params_and_runtime(
        request, emhass_conf, app.logger
    )
    if params is None:
        return await make_response(await grab_log(" >> Obtaining params: "), 400)

    # Check for actions that do not need input_data_dict
    if action_name in ["weather-forecast-cache", "export-influxdb-to-csv"]:
        msg, status = await _handle_action_dispatch(
            action_name, None, emhass_conf, params, runtimeparams, app.logger
        )
        if status == 400:
            return await make_response(msg, status)

        # Check logs for these specific actions
        action_str = f" >> Performing {action_name}..."
        if not await check_file_log(action_str):
            return await make_response(msg, status)
        return await make_response(await grab_log(action_str), 400)

    # Set Input Data Dict (Common for all other actions)
    action_str = " >> Setting input data dict"
    app.logger.info(action_str)
    input_data_dict = await set_input_data_dict(
        emhass_conf, costfun, params, runtimeparams, action_name, app.logger
    )

    if not input_data_dict:
        return await make_response(await grab_log(action_str), 400)

    # Handle Continual Publish Threading
    if len(continual_publish_thread) == 0 and input_data_dict["retrieve_hass_conf"].get(
        "continual_publish", False
    ):
        continual_loop = threading.Thread(
            name="continual_publish",
            target=lambda: asyncio.run(continual_publish(input_data_dict, entity_path, app.logger)),
        )
        continual_loop.start()
        continual_publish_thread.append(continual_loop)

    # Execute Action
    msg, status = await _handle_action_dispatch(
        action_name, input_data_dict, emhass_conf, params, runtimeparams, app.logger
    )

    # Final Log Check & Response
    if status == 201:
        if not await check_file_log(" >> "):
            return await make_response(msg, 201)
        return await make_response(await grab_log(" >> "), 400)

    return await make_response(msg, status)


async def _setup_paths() -> tuple[Path, Path, Path, Path, Path, Path]:
    """Helper to set up environment paths and update emhass_conf."""
    # Find env's, not not set defaults
    DATA_PATH = os.getenv("DATA_PATH", default="/data/")
    ROOT_PATH = os.getenv("ROOT_PATH", default=str(Path(__file__).parent))
    CONFIG_PATH = os.getenv("CONFIG_PATH", default="/share/config.json")
    OPTIONS_PATH = os.getenv("OPTIONS_PATH", default="/data/options.json")
    DEFAULTS_PATH = os.getenv("DEFAULTS_PATH", default=ROOT_PATH + "/data/config_defaults.json")
    ASSOCIATIONS_PATH = os.getenv("ASSOCIATIONS_PATH", default=ROOT_PATH + "/data/associations.csv")
    LEGACY_CONFIG_PATH = os.getenv("LEGACY_CONFIG_PATH", default="/app/config_emhass.yaml")
    # Define the paths
    config_path = Path(CONFIG_PATH)
    options_path = Path(OPTIONS_PATH)
    defaults_path = Path(DEFAULTS_PATH)
    associations_path = Path(ASSOCIATIONS_PATH)
    legacy_config_path = Path(LEGACY_CONFIG_PATH)
    data_path = Path(DATA_PATH)
    root_path = Path(ROOT_PATH)
    # Add paths to emhass_conf
    emhass_conf["config_path"] = config_path
    emhass_conf["options_path"] = options_path
    emhass_conf["defaults_path"] = defaults_path
    emhass_conf["associations_path"] = associations_path
    emhass_conf["legacy_config_path"] = legacy_config_path
    emhass_conf["data_path"] = data_path
    emhass_conf["root_path"] = root_path
    return (
        config_path,
        options_path,
        defaults_path,
        associations_path,
        legacy_config_path,
        root_path,
    )


async def _build_configuration(
    config_path: Path, legacy_config_path: Path, defaults_path: Path
) -> tuple[dict, str, str]:
    """Helper to build configuration and local variables."""
    config = {}
    # Combine parameters from configuration sources (if exists)
    config.update(
        await build_config(
            emhass_conf,
            app.logger,
            str(defaults_path),
            str(config_path) if config_path.exists() else None,
            str(legacy_config_path) if legacy_config_path.exists() else None,
        )
    )
    if type(config) is bool and not config:
        raise Exception("Failed to find default config")
    # Set local variables
    costfun = os.getenv("LOCAL_COSTFUN", config.get("costfun", "profit"))
    logging_level = os.getenv("LOGGING_LEVEL", config.get("logging_level", "INFO"))
    # Temporary set logging level if debug
    if logging_level == "DEBUG":
        app.logger.setLevel(logging.DEBUG)
    return config, costfun, logging_level


async def _setup_secrets(args: dict | None, options_path: Path) -> str:
    """Helper to parse arguments and build secrets."""
    ## Secrets
    # Argument
    argument = {}
    no_response = False
    if args is not None:
        if args.get("url", None):
            argument["url"] = args["url"]
        if args.get("key", None):
            argument["key"] = args["key"]
        if args.get("no_response", None):
            no_response = args["no_response"]

    # Define secrets_path and save to emhass_conf
    secrets_path = Path(os.getenv("SECRETS_PATH", default="/app/secrets_emhass.yaml"))

    # Store it in the global config so configuration() can use it later
    global emhass_conf
    emhass_conf["secrets_path"] = secrets_path

    # Combine secrets from ENV, Arguments/ARG, Secrets file (secrets_emhass.yaml), options (options.json from addon configuration file) and/or Home Assistant Standalone API (if exist)
    emhass_conf, secrets = await build_secrets(
        emhass_conf,
        app.logger,
        secrets_path=secrets_path,  # Use the variable we defined above
        options_path=str(options_path),
        argument=argument,
        no_response=bool(no_response),
    )
    params_secrets.update(secrets)
    return params_secrets.get("server_ip", "0.0.0.0")


def _validate_data_path(root_path: Path) -> None:
    """Helper to validate and create the data path if necessary."""
    # Check if data path exists
    if not os.path.isdir(emhass_conf["data_path"]):
        app.logger.warning("Unable to find data_path: " + str(emhass_conf["data_path"]))
        if os.path.isdir(Path("/data/")):
            emhass_conf["data_path"] = Path("/data/")
        else:
            Path(root_path / "data/").mkdir(parents=True, exist_ok=True)
            emhass_conf["data_path"] = root_path / "data/"
        app.logger.info("data_path has been set to " + str(emhass_conf["data_path"]))


async def _load_injection_dict() -> dict | None:
    """Helper to load the injection dictionary."""
    # Initialize this global dict
    if (emhass_conf["data_path"] / injection_dict_file).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / injection_dict_file), "rb") as fid:
            content = await fid.read()
            return pickle.loads(content)
    else:
        return None


async def _build_and_save_params(
    config: dict, costfun: str, logging_level: str, config_path: Path
) -> dict:
    """Helper to build parameters and save them to a pickle file."""
    # Build params from config and param_secrets (migrate params to correct config catagories), save result to params.pkl
    params = await build_params(emhass_conf, params_secrets, config, app.logger)
    if type(params) is bool:
        raise Exception("A error has occurred while building params")
    # Update params with local variables
    params["optim_conf"]["costfun"] = costfun
    params["optim_conf"]["logging_level"] = logging_level
    # Save params to file for later reference (use emhass_conf["config_path"] which may have been updated by build_secrets)
    if os.path.exists(str(emhass_conf["data_path"])):
        async with aiofiles.open(str(emhass_conf["data_path"] / params_file), "wb") as fid:
            content = pickle.dumps((emhass_conf["config_path"], params))
            await fid.write(content)
    else:
        raise Exception("missing: " + str(emhass_conf["data_path"]))
    return params


async def _configure_logging(logging_level: str) -> None:
    """Helper to configure logging handlers and levels."""
    # Define loggers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log.default_handler.setFormatter(formatter)
    # Action file logger
    file_logger = logging.FileHandler(str(emhass_conf["data_path"] / action_log_str))
    formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    file_logger.setFormatter(formatter)  # add format to Handler
    if logging_level == "DEBUG":
        app.logger.setLevel(logging.DEBUG)
        file_logger.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        app.logger.setLevel(logging.INFO)
        file_logger.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        app.logger.setLevel(logging.WARNING)
        file_logger.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        app.logger.setLevel(logging.ERROR)
        file_logger.setLevel(logging.ERROR)
    else:
        app.logger.setLevel(logging.DEBUG)
        file_logger.setLevel(logging.DEBUG)
    app.logger.propagate = False
    app.logger.addHandler(file_logger)
    # Clear Action File logger file, ready for new instance
    await clear_file_log()


def _cleanup_entities() -> Path:
    """Helper to remove entity/metadata files."""
    # If entity_path exists, remove any entity/metadata files
    ent_path = emhass_conf["data_path"] / "entities"
    if os.path.exists(ent_path):
        entity_path_contents = os.listdir(ent_path)
        if len(entity_path_contents) > 0:
            for entity in entity_path_contents:
                os.remove(ent_path / entity)
    return ent_path


async def _initialize_connections(params: dict) -> None:
    """Helper to initialize WebSocket or InfluxDB connections."""
    # Initialize persistent WebSocket connection only if use_websocket is enabled
    use_websocket = params.get("retrieve_hass_conf", {}).get("use_websocket", False)
    use_influxdb = params.get("retrieve_hass_conf", {}).get("use_influxdb", False)
    # Initialize persistent WebSocket connection if enabled
    if use_websocket:
        app.logger.info("WebSocket mode enabled - initializing connection...")
        try:
            await get_websocket_client(
                hass_url=params_secrets["hass_url"],
                token=params_secrets["long_lived_token"],
                logger=app.logger,
            )
            app.logger.info("WebSocket connection established")
            # WebSocket shutdown is already handled by @app.after_serving
        except Exception as ws_error:
            app.logger.warning(f"WebSocket connection failed: {ws_error}")
            app.logger.info("Continuing without WebSocket connection...")
            # Re-raise the exception so before_serving can handle it
            raise
    # Log InfluxDB mode if enabled (No persistent connection init required here)
    elif use_influxdb:
        app.logger.info("InfluxDB mode enabled - using InfluxDB for data retrieval")
    # Default to REST API if neither is enabled
    else:
        app.logger.info("WebSocket and InfluxDB modes disabled - using REST API for data retrieval")


async def initialize(args: dict | None = None):
    global emhass_conf, params_secrets, continual_publish_thread, injection_dict, entity_path
    # Setup paths
    (
        config_path,
        options_path,
        defaults_path,
        _,
        legacy_config_path,
        root_path,
    ) = await _setup_paths()
    # Setup Secrets (must run BEFORE build_configuration to allow options.json to override config_path)
    server_ip = await _setup_secrets(args, options_path)
    # Build configuration (now uses potentially updated emhass_conf["config_path"] from options.json)
    config, costfun, logging_level = await _build_configuration(
        emhass_conf["config_path"], emhass_conf.get("legacy_config_path", legacy_config_path), defaults_path
    )
    # Validate Data Path
    _validate_data_path(root_path)
    # Load Injection Dict
    injection_dict = await _load_injection_dict()
    # Build and Save Params
    params = await _build_and_save_params(config, costfun, logging_level, config_path)
    # Configure Logging
    await _configure_logging(logging_level)
    # Cleanup Entities
    entity_path = _cleanup_entities()
    # Initialize Continual Publish Thread
    # Initialise continual publish thread list
    continual_publish_thread = []
    # Log Startup Info
    # Logging
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Launching the emhass webserver at: http://" + server_ip + ":" + str(port))
    app.logger.info(
        "Home Assistant data fetch will be performed using url: " + params_secrets["hass_url"]
    )
    app.logger.info("The data path is: " + str(emhass_conf["data_path"]))
    app.logger.info("The config path is: " + str(emhass_conf["config_path"]))
    app.logger.info("The logging is: " + str(logging_level))
    try:
        app.logger.info("Using core emhass version: " + version("emhass"))
    except PackageNotFoundError:
        app.logger.info("Using development emhass version")
    # Initialize Connections (WebSocket/InfluxDB)
    await _initialize_connections(params)
    app.logger.info("Initialization complete")


async def main() -> None:
    """
    Main function to handle command line arguments.

    Note: In production, the app should be run via gunicorn with uvicorn workers:
    gunicorn emhass.web_server:app -c gunicorn.conf.py -k uvicorn.workers.UvicornWorker
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="HA URL")
    parser.add_argument("--key", type=str, help="HA long‑lived token")
    parser.add_argument("--no_response", action="store_true")
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    # Initialize the app before starting server
    await initialize(args_dict)
    # For direct execution (development/testing), use uvicorn programmatically
    host = params_secrets.get("server_ip", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    app.logger.info(f"Starting server directly on {host}:{port}")
    # Use uvicorn.Server to run within existing event loop
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
