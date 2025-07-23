#!/usr/bin/env python3

import argparse
import asyncio
import atexit
import logging
import os
import pickle
import re
import signal
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import aiofiles
import jinja2
import orjson
import yaml
from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart, make_response, request
from quart import logging as log

from emhass.command_line_async import (
    continual_publish,
    dayahead_forecast_optim,
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
from emhass.connection_manager import close_global_connection, get_websocket_client
from emhass.utils_async import (
    build_config,
    build_legacy_config_params,
    build_params,
    build_secrets,
    get_injection_dict,
    get_injection_dict_forecast_model_fit,
    get_injection_dict_forecast_model_tune,
    param_to_config,
)

app = Quart(__name__)

emhass_conf = {}
entity_path = None
params_secrets = {}
continual_publish_thread = []
injection_dict = {}

templates = jinja2.Environment(
    loader=jinja2.PackageLoader("emhass", "templates"),
)






# Register async startup and shutdown handlers
@app.before_serving
async def before_serving():
    """Initialize the app before serving requests"""
    app.logger.info("🚀 Quart app starting up")

    # Try to initialize the full application, but handle WebSocket failures gracefully
    try:
        await initialize()
        app.logger.info("✅ Full initialization completed including WebSocket connection")
    except Exception as e:
        app.logger.warning(f"⚠️ Full initialization failed (this is normal in test environments): {e}")
        app.logger.info("🔄 Continuing without WebSocket connection...")
        # The initialize() function already sets up all necessary components except WebSocket
        # So we can continue serving requests even if WebSocket connection fails

@app.after_serving
async def after_serving():
    """Clean shutdown of the app"""
    app.logger.info("🛑 Quart app shutting down...")
    try:
        await close_global_connection()
        app.logger.info("✅ WebSocket connection closed")
    except Exception as e:
        app.logger.warning(f"❌ WebSocket shutdown failed: {e}")
    app.logger.info("✅ Quart shutdown complete")
#         except Exception as e:
#             app.logger.warning(f"❌ WebSocket shutdown failed: {e}")
#         app.logger.info("✅ Quart app shutdown complete")

#     return app


async def checkFileLog(refString=None) -> bool:
    """
    Check logfile for error, anything after string match if provided.

    :param refString: String to reduce log area to check for errors. Use to reduce log to check anything after string match (ie. an action).
    :type refString: str
    :return: Boolean return if error was found in logs
    :rtype: bool

    """
    if refString is not None:
        logArray = await grabLog(
            refString
        )  # grab reduced log array (everything after string match)
    else:
        if (emhass_conf["data_path"] / "actionLogs.txt").exists():
            async with aiofiles.open(str(emhass_conf["data_path"] / "actionLogs.txt")) as fp:
                content = await fp.read()
                logArray = content.splitlines()
        else:
            app.logger.debug("Unable to obtain actionLogs.txt")
            return False
    for logString in logArray:
        if logString.split(" ", 1)[0] == "ERROR":
            return True
    return False


async def grabLog(refString) -> list:
    """
    Find string in logs, append all lines after into list to return.

    :param refString: String used to string match log.
    :type refString: str
    :return: List of lines in log after string match.
    :rtype: list

    """
    isFound = []
    output = []
    if (emhass_conf["data_path"] / "actionLogs.txt").exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / "actionLogs.txt")) as fp:
            content = await fp.read()
            logArray = content.splitlines()
        # Find all string matches, log key (line Number) in isFound
        for x in range(len(logArray) - 1):
            if re.search(refString, logArray[x]):
                isFound.append(x)
        if len(isFound) != 0:
            # Use last item in isFound to extract action logs
            for x in range(isFound[-1], len(logArray)):
                output.append(logArray[x])
    return output


# Clear the log file
async def clearFileLog():
    """
    Clear the contents of the log file (actionLogs.txt)

    """
    if (emhass_conf["data_path"] / "actionLogs.txt").exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / "actionLogs.txt"), "w") as fp:
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
    if (emhass_conf["data_path"] / "injection_dict.pkl").exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / "injection_dict.pkl"), "rb") as fid:
            content = await fid.read()
            injection_dict = pickle.loads(content)
    else:
        app.logger.info(
            "The data container dictionary is empty... Please launch an optimization task"
        )
        injection_dict = {}

    template = templates.get_template("index.html")
    return await make_response(template.render(injection_dict=injection_dict))


@app.route("/configuration")
async def configuration():
    """
    Configuration page actions:
    Render and serve configuration page html

    """
    app.logger.info("serving configuration.html...")
    # get params
    params = {}
    if (emhass_conf["data_path"] / "params.pkl").exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / "params.pkl"), "rb") as fid:
            content = await fid.read()
            emhass_conf["config_path"], params = pickle.loads(content)

    template = templates.get_template("configuration.html")
    return await make_response(template.render(config=params))


@app.route("/template", methods=["GET"])
async def template_action():
    """
    template page actions:
    Render and serve template html

    """
    app.logger.info(" >> Sending rendered template table data")
    if (emhass_conf["data_path"] / "injection_dict.pkl").exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / "injection_dict.pkl"), "rb") as fid:
            content = await fid.read()
            injection_dict = pickle.loads(content)
    else:
        app.logger.warning("Unable to obtain plot data from injection_dict.pkl")
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
        emhass_conf["defaults_path"],
        emhass_conf["config_path"],
        emhass_conf["legacy_config_path"],
    )
    if type(config) is bool and not config:
        return await make_response(["failed to retrieve default config file"], 500)
    # Format parameters in config with params (converting legacy json parameters from options.json if any)
    params = await build_params(emhass_conf, {}, config, app.logger)
    if type(params) is bool and not params:
        return await make_response(["Unable to obtain associations file"], 500)
    # Covert formatted parameters from params back into config.json format
    return_config = await param_to_config(params, app.logger)
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
    config = await build_config(emhass_conf, app.logger, emhass_conf["defaults_path"])
    if type(config) is bool and not config:
        return await make_response(["failed to retrieve default config file"], 500)
    # Format parameters in config with params
    params = await build_params(emhass_conf, {}, config, app.logger)
    if type(params) is bool and not params:
        return await make_response(["Unable to obtain associations file"], 500)
    # Covert formatted parameters from params back into config.json format
    return_config = await param_to_config(params, app.logger)
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
    test_legacy_config = await build_legacy_config_params(
        emhass_conf, yaml_config, app.logger
    )
    if test_legacy_config:
        yaml_config = test_legacy_config
    # Format YAML to params (format params. check if params match legacy option.json format)
    params = await build_params(emhass_conf, {}, yaml_config, app.logger)
    if type(params) is bool and not params:
        return await make_response(["Unable to obtain associations file"], 500)
    # Covert formatted parameters from params back into config.json format
    config = await param_to_config(params, app.logger)
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
        return await make_response(["Unable to obtain associations file"], 500)

    # Covert formatted parameters from params back into config.json format.
    # Overwrite existing default parameters in config
    config.update(await param_to_config(params, app.logger))

    # Save config to config.json
    if os.path.exists(emhass_conf["config_path"].parent):
        async with aiofiles.open(str(emhass_conf["config_path"]), "w") as f:
            await f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2).decode())
    else:
        return await make_response(["Unable to save config file"], 500)

    # Save params with updated config
    if os.path.exists(emhass_conf["data_path"]):
        async with aiofiles.open(str(emhass_conf["data_path"] / "params.pkl"), "wb") as fid:
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


@app.route("/action/<action_name>", methods=["POST"])
async def action_call(action_name):
    """
    Receive Post action, run action according to passed slug(action_name) (e.g. /action/publish-data)

    :param action_name: Slug/Action string corresponding to which action to take
    :type action_name: String

    """
    global continual_publish_thread
    global injection_dict

    # Setting up parameters
    # Params
    ActionStr = " >> Obtaining params: "
    app.logger.info(ActionStr)
    costfun = "profit"  # Default value
    if (emhass_conf["data_path"] / "params.pkl").exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / "params.pkl"), "rb") as fid:
            content = await fid.read()
            emhass_conf["config_path"], params = pickle.loads(content)
            # Set local costfun variable
            if params.get("optim_conf", None) is not None:
                costfun = params["optim_conf"].get("costfun", "profit")
            params = orjson.dumps(params).decode()
    else:
        app.logger.error("Unable to find params.pkl file")
        return await make_response(await grabLog(ActionStr), 400)
    # Runtime
    runtimeparams = await request.get_json(force=True, silent=True)
    if runtimeparams is not None:
        if runtimeparams != "{}":
            app.logger.debug("Passed runtime parameters: " + str(runtimeparams))
    else:
        app.logger.warning("Unable to parse runtime parameters")
        runtimeparams = {}
    runtimeparams = orjson.dumps(runtimeparams).decode()

    # weather-forecast-cache (check before set_input_data_dict)
    if action_name == "weather-forecast-cache":
        ActionStr = " >> Performing weather forecast, try to caching result"
        app.logger.info(ActionStr)
        await weather_forecast_cache(emhass_conf, params, runtimeparams, app.logger)
        msg = "EMHASS >> Weather Forecast has run and results possibly cached... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)

    ActionStr = " >> Setting input data dict"
    app.logger.info(ActionStr)
    input_data_dict = await set_input_data_dict(
        emhass_conf, costfun, params, runtimeparams, action_name, app.logger
    )
    if not input_data_dict:
        return await make_response(await grabLog(ActionStr), 400)

    # If continual_publish is True, start thread with loop function
    if len(continual_publish_thread) == 0 and input_data_dict["retrieve_hass_conf"].get(
        "continual_publish", False
    ):
        # Start Thread
        continualLoop = threading.Thread(
            name="continual_publish",
            target=lambda: asyncio.run(continual_publish(input_data_dict, entity_path or Path(), app.logger)),
        )
        continualLoop.start()
        continual_publish_thread.append(continualLoop)

    # Run action based on POST request
    # If error in log when running action, return actions log (list) as response. (Using ActionStr as a reference of the action start in the log)
    if action_name == "publish-data":
        ActionStr = " >> Publishing data..."
        app.logger.info(ActionStr)
        _ = await publish_data(input_data_dict, app.logger)
        msg = "EMHASS >> Action publish-data executed... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)
    # perfect-optim
    elif action_name == "perfect-optim":
        ActionStr = " >> Performing perfect optimization..."
        app.logger.info(ActionStr)
        opt_res = await perfect_forecast_optim(input_data_dict, app.logger)
        injection_dict = await get_injection_dict(opt_res)
        async with aiofiles.open(str(emhass_conf["data_path"] / "injection_dict.pkl"), "wb") as fid:
            content = pickle.dumps(injection_dict)
            await fid.write(content)
        msg = "EMHASS >> Action perfect-optim executed... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)
    # dayahead-optim
    elif action_name == "dayahead-optim":
        ActionStr = " >> Performing dayahead optimization..."
        app.logger.info(ActionStr)
        opt_res = await dayahead_forecast_optim(input_data_dict, app.logger)
        injection_dict = await get_injection_dict(opt_res)
        async with aiofiles.open(str(emhass_conf["data_path"] / "injection_dict.pkl"), "wb") as fid:
            content = pickle.dumps(injection_dict)
            await fid.write(content)
        msg = "EMHASS >> Action dayahead-optim executed... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)
    # naive-mpc-optim
    elif action_name == "naive-mpc-optim":
        ActionStr = " >> Performing naive MPC optimization..."
        app.logger.info(ActionStr)

        # Validate input_data_dict
        if not input_data_dict or len(input_data_dict) == 0:
            app.logger.error("Input data dictionary is empty - cannot perform optimization")
            return await make_response("EMHASS >> Error: No input data available for optimization", 400)

        opt_res = await naive_mpc_optim(input_data_dict, app.logger)

        # Check if optimization returned valid results
        if opt_res is None or (isinstance(opt_res, bool) and not opt_res):
            app.logger.error("Naive MPC optimization failed")
            return await make_response("EMHASS >> Error: Naive MPC optimization failed", 400)

        injection_dict = await get_injection_dict(opt_res)
        async with aiofiles.open(str(emhass_conf["data_path"] / "injection_dict.pkl"), "wb") as fid:
            content = pickle.dumps(injection_dict)
            await fid.write(content)
        msg = "EMHASS >> Action naive-mpc-optim executed... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)
    # forecast-model-fit
    elif action_name == "forecast-model-fit":
        ActionStr = " >> Performing a machine learning forecast model fit..."
        app.logger.info(ActionStr)
        df_fit_pred, _, mlf = await forecast_model_fit(input_data_dict, app.logger)
        injection_dict = await get_injection_dict_forecast_model_fit(df_fit_pred, mlf)
        async with aiofiles.open(str(emhass_conf["data_path"] / "injection_dict.pkl"), "wb") as fid:
            content = pickle.dumps(injection_dict)
            await fid.write(content)
        msg = "EMHASS >> Action forecast-model-fit executed... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)
    # forecast-model-predict
    elif action_name == "forecast-model-predict":
        ActionStr = " >> Performing a machine learning forecast model predict..."
        app.logger.info(ActionStr)
        df_pred = await forecast_model_predict(input_data_dict, app.logger)
        if df_pred is None:
            return await make_response(await grabLog(ActionStr), 400)
        table1 = df_pred.reset_index().to_html(classes="mystyle", index=False)
        injection_dict = {}
        injection_dict["title"] = (
            "<h2>Custom machine learning forecast model predict</h2>"
        )
        injection_dict["subsubtitle0"] = (
            "<h4>Performed a prediction using a pre-trained model</h4>"
        )
        injection_dict["table1"] = table1
        async with aiofiles.open(str(emhass_conf["data_path"] / "injection_dict.pkl"), "wb") as fid:
            content = pickle.dumps(injection_dict)
            await fid.write(content)
        msg = "EMHASS >> Action forecast-model-predict executed... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)
    # forecast-model-tune
    elif action_name == "forecast-model-tune":
        ActionStr = " >> Performing a machine learning forecast model tune..."
        app.logger.info(ActionStr)
        df_pred_optim, mlf = await forecast_model_tune(input_data_dict, app.logger)
        if df_pred_optim is None or mlf is None:
            return await make_response(await grabLog(ActionStr), 400)
        injection_dict = await get_injection_dict_forecast_model_tune(df_pred_optim, mlf)
        async with aiofiles.open(str(emhass_conf["data_path"] / "injection_dict.pkl"), "wb") as fid:
            content = pickle.dumps(injection_dict)
            await fid.write(content)
        msg = "EMHASS >> Action forecast-model-tune executed... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)
    # regressor-model-fit
    elif action_name == "regressor-model-fit":
        ActionStr = " >> Performing a machine learning regressor fit..."
        app.logger.info(ActionStr)
        await regressor_model_fit(input_data_dict, app.logger)
        msg = "EMHASS >> Action regressor-model-fit executed... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)
    # regressor-model-predict
    elif action_name == "regressor-model-predict":
        ActionStr = " >> Performing a machine learning regressor predict..."
        app.logger.info(ActionStr)
        await regressor_model_predict(input_data_dict, app.logger)
        msg = "EMHASS >> Action regressor-model-predict executed... \n"
        if not await checkFileLog(ActionStr):
            return await make_response(msg, 201)
        return await make_response(await grabLog(ActionStr), 400)
    # Else return error
    else:
        app.logger.error("ERROR: passed action is not valid")
        msg = "EMHASS >> ERROR: Passed action is not valid... \n"
        return await make_response(msg, 400)



async def initialize():
    global emhass_conf, params_secrets, continual_publish_thread, injection_dict

    config = {}
    params = None

    # Find env's, not not set defaults
    DATA_PATH = os.getenv("DATA_PATH", default="/data/")
    ROOT_PATH = os.getenv("ROOT_PATH", default=str(Path(__file__).parent))
    CONFIG_PATH = os.getenv("CONFIG_PATH", default="/share/config.json")
    OPTIONS_PATH = os.getenv("OPTIONS_PATH", default="/data/options.json")
    DEFAULTS_PATH = os.getenv(
        "DEFAULTS_PATH", default=ROOT_PATH + "/data/config_defaults.json"
    )
    ASSOCIATIONS_PATH = os.getenv(
        "ASSOCIATIONS_PATH", default=ROOT_PATH + "/data/associations.csv"
    )
    LEGACY_CONFIG_PATH = os.getenv(
        "LEGACY_CONFIG_PATH", default="/app/config_emhass.yaml"
    )

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

    # Combine parameters from configuration sources (if exists)
    config.update(
        await build_config(
            emhass_conf, app.logger, str(defaults_path), str(config_path), str(legacy_config_path)
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

    ## Secrets
    # Argument
    argument = {}
    no_response = False
    # Combine secrets from ENV, Arguments/ARG, Secrets file (secrets_emhass.yaml), options (options.json from addon configuration file) and/or Home Assistant Standalone API (if exist)
    emhass_conf, secrets = await build_secrets(
        emhass_conf,
        app.logger,
        argument,
        str(options_path),
        os.getenv("SECRETS_PATH", default="/app/secrets_emhass.yaml"),
        bool(no_response),
    )
    params_secrets.update(secrets)

    server_ip = params_secrets.get("server_ip", "0.0.0.0")

    # Check if data path exists
    if not os.path.isdir(emhass_conf["data_path"]):
        app.logger.warning("Unable to find data_path: " + str(emhass_conf["data_path"]))
        if os.path.isdir(Path("/data/")):
            emhass_conf["data_path"] = Path("/data/")
        else:
            Path(root_path / "data/").mkdir(parents=True, exist_ok=True)
            emhass_conf["data_path"] = root_path / "data/"
        app.logger.info("data_path has been set to " + str(emhass_conf["data_path"]))

    # Initialize this global dict
    if (emhass_conf["data_path"] / "injection_dict.pkl").exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / "injection_dict.pkl"), "rb") as fid:
            content = await fid.read()
            injection_dict = pickle.loads(content)
    else:
        injection_dict = None

    # Build params from config and param_secrets (migrate params to correct config catagories), save result to params.pkl
    params = await build_params(emhass_conf, params_secrets, config, app.logger)
    if type(params) is bool:
        raise Exception("A error has occurred while building params")
    # Update params with local variables
    params["optim_conf"]["costfun"] = costfun
    params["optim_conf"]["logging_level"] = logging_level

    # Save params to file for later reference
    if os.path.exists(str(emhass_conf["data_path"])):
        async with aiofiles.open(str(emhass_conf["data_path"] / "params.pkl"), "wb") as fid:
            content = pickle.dumps((config_path, params))
            await fid.write(content)
    else:
        raise Exception("missing: " + str(emhass_conf["data_path"]))

    # Define loggers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log.default_handler.setFormatter(formatter)
    # Action file logger
    fileLogger = logging.FileHandler(str(emhass_conf["data_path"] / "actionLogs.txt"))
    formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    fileLogger.setFormatter(formatter)  # add format to Handler
    if logging_level == "DEBUG":
        app.logger.setLevel(logging.DEBUG)
        fileLogger.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        app.logger.setLevel(logging.INFO)
        fileLogger.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        app.logger.setLevel(logging.WARNING)
        fileLogger.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        app.logger.setLevel(logging.ERROR)
        fileLogger.setLevel(logging.ERROR)
    else:
        app.logger.setLevel(logging.DEBUG)
        fileLogger.setLevel(logging.DEBUG)
    app.logger.propagate = False
    app.logger.addHandler(fileLogger)
    # Clear Action File logger file, ready for new instance
    await clearFileLog()

    # If entity_path exists, remove any entity/metadata files
    entity_path = emhass_conf["data_path"] / "entities"
    if os.path.exists(entity_path):
        entity_pathContents = os.listdir(entity_path)
        if len(entity_pathContents) > 0:
            for entity in entity_pathContents:
                os.remove(entity_path / entity)

    # Initialise continual publish thread list
    continual_publish_thread = []

    # Logging
    port = int(os.environ.get("PORT", 5000))
    app.logger.info(
        "Launching the emhass webserver at: http://" + server_ip + ":" + str(port)
    )
    app.logger.info(
        "Home Assistant data fetch will be performed using url: "
        + params_secrets["hass_url"]
    )
    app.logger.info("The data path is: " + str(emhass_conf["data_path"]))
    app.logger.info("The logging is: " + str(logging_level))
    try:
        app.logger.info("Using core emhass version: " + version("emhass"))
    except PackageNotFoundError:
        app.logger.info("Using development emhass version")

    # Initialize persistent WebSocket connection (this may fail in test environments)
    try:
        await get_websocket_client(
                hass_url=params_secrets["hass_url"],
                token=params_secrets["long_lived_token"],
                logger=app.logger
            )
        app.logger.info("✅ WebSocket connection established")

        # WebSocket shutdown is already handled by @app.after_serving
        # No need for atexit handler
    except Exception as ws_error:
        app.logger.warning(f"WebSocket connection failed: {ws_error}")
        app.logger.info("Continuing without WebSocket connection...")
        # Re-raise the exception so before_serving can handle it
        raise

    # atexit wordt niet meer nodig aangezien we via Quart shutdown werken.

    # eventuele extra opstart-logica...
    app.logger.info("✅ Initialization complete")

async def setup_config_and_paths(args_dict) -> (str, int):
    """Rebuild minimal config to pass into hypercorn."""
    # Kan uitgebreid worden met build_config/build_secrets etc. als header
    host = params_secrets.get("server_ip", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    return host, port

async def main_with_server(args_dict):
    host, port = await setup_config_and_paths(args_dict)
    config = Config()
    config.bind = [f"{host}:{port}"]
    config.use_reloader = False

    stop_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    serve_task = asyncio.create_task(serve(app, config))
    waiter = asyncio.create_task(stop_event.wait())

    done, _ = await asyncio.wait({serve_task, waiter}, return_when=asyncio.FIRST_COMPLETED)

    if waiter in done:
        app.logger.info("🔒 Received shutdown signal – shutting down server")
        serve_task.cancel()
        await serve_task

    app.logger.info("Server event loop complete, exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="HA URL")
    parser.add_argument("--key", type=str, help="HA long‑lived token")
    parser.add_argument("--no_response", action="store_true")
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    asyncio.run(main_with_server(args_dict))
