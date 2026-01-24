#!/usr/bin/env python3

import argparse
import asyncio
import copy
import logging
import os
import pathlib
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import version

import aiofiles
import numpy as np
import orjson
import pandas as pd

from emhass import utils
from emhass.forecast import Forecast
from emhass.machine_learning_forecaster import MLForecaster
from emhass.machine_learning_regressor import MLRegressor
from emhass.optimization import Optimization
from emhass.retrieve_hass import RetrieveHass

default_csv_filename = "opt_res_latest.csv"
default_pkl_suffix = "_mlf.pkl"
default_metadata_json = "metadata.json"
test_df_literal = "test_df_final.pkl"


@dataclass
class SetupContext:
    """
    A dataclass that serves as a context container for optimization preparation helpers.
    This context object encapsulates all necessary configuration and utility objects
    required for setting up and preparing optimization tasks.
    Attributes:
        retrieve_hass_conf (dict): Configuration dictionary for Home Assistant data retrieval.
        optim_conf (dict): Configuration dictionary for optimization parameters.
        plant_conf (dict): Configuration dictionary for plant/system parameters.
        emhass_conf (dict): Configuration dictionary for EMHASS settings.
        params (dict): Additional parameters dictionary.
        logger (logging.Logger): Logger instance for logging messages.
        get_data_from_file (bool): Flag indicating whether to retrieve data from file instead of live source.
        rh (RetrieveHass): RetrieveHass instance for retrieving Home Assistant data.
        fcst (Forecast | None): Optional Forecast object for weather or energy forecasting. Defaults to None.
    """

    retrieve_hass_conf: dict
    optim_conf: dict
    plant_conf: dict
    emhass_conf: dict
    params: dict
    logger: logging.Logger
    get_data_from_file: bool
    rh: RetrieveHass
    fcst: Forecast | None = None


@dataclass
def __init__(
    self, input_data_dict: dict, params: dict, idx: int, common_kwargs: dict, logger: logging.Logger
) -> None:
    """
    Initialize a PublishContext instance.
    Args:
        input_data_dict (dict): Dictionary containing input data with keys 'rh' (RetrieveHass),
            'opt' (Optimization), and 'fcst' (Forecast) objects.
        params (dict): Parameters dictionary for publishing configuration.
        idx (int): Index identifier for the current publishing operation.
        common_kwargs (dict): Common keyword arguments shared across publishing helpers.
        logger (logging.Logger): Logger instance for recording publishing operations.
    """


class PublishContext:
    """Context object for data publishing helpers."""

    input_data_dict: dict
    params: dict
    idx: int
    common_kwargs: dict
    logger: logging.Logger

    @property
    def rh(self) -> RetrieveHass:
        return self.input_data_dict["rh"]

    @property
    def opt(self) -> Optimization:
        return self.input_data_dict["opt"]

    @property
    def fcst(self) -> Forecast:
        return self.input_data_dict["fcst"]


async def _retrieve_from_file(
    emhass_conf: dict,
    test_df_literal: str,
    rh: RetrieveHass,
    retrieve_hass_conf: dict,
    optim_conf: dict,
) -> tuple[bool, object]:
    """Helper to retrieve data from a pickle file and configure variables."""
    async with aiofiles.open(emhass_conf["data_path"] / test_df_literal, "rb") as inp:
        content = await inp.read()
        rh.df_final, days_list, var_list, rh.ha_config = pickle.loads(content)
        rh.var_list = var_list
    # Assign variables based on set_type
    retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(var_list[0])
    if optim_conf.get("set_use_pv", True):
        retrieve_hass_conf["sensor_power_photovoltaics"] = str(var_list[1])
        retrieve_hass_conf["sensor_linear_interp"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"],
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
        ]
        retrieve_hass_conf["sensor_replace_zero"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"],
            var_list[2],
        ]
    else:
        retrieve_hass_conf["sensor_linear_interp"] = [
            retrieve_hass_conf["sensor_power_load_no_var_loads"]
        ]
        retrieve_hass_conf["sensor_replace_zero"] = []
    return True, days_list


async def _retrieve_from_hass(
    set_type: str,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    logger: logging.Logger | None,
) -> tuple[bool, object]:
    """Helper to retrieve live data from Home Assistant."""
    # Determine days_list based on set_type
    if set_type == "perfect-optim" or set_type == "adjust_pv":
        days_list = utils.get_days_list(retrieve_hass_conf["historic_days_to_retrieve"])
    elif set_type == "naive-mpc-optim":
        days_list = utils.get_days_list(1)
    else:
        days_list = None  # Not needed for dayahead
    var_list = [retrieve_hass_conf["sensor_power_load_no_var_loads"]]
    if optim_conf.get("set_use_pv", True):
        var_list.append(retrieve_hass_conf["sensor_power_photovoltaics"])
        if optim_conf.get("set_use_adjusted_pv", True):
            var_list.append(retrieve_hass_conf["sensor_power_photovoltaics_forecast"])
            if logger:
                logger.debug(f"Variable list for data retrieval: {var_list}")
    success = await rh.get_data(
        days_list, var_list, minimal_response=False, significant_changes_only=False
    )
    return success, days_list


async def retrieve_home_assistant_data(
    set_type: str,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: str,
    logger: logging.Logger | None = None,
) -> tuple[bool, pd.DataFrame | None, list | None]:
    """Retrieve data from Home Assistant or file and prepare it for optimization."""

    if get_data_from_file:
        success, days_list = await _retrieve_from_file(
            emhass_conf, test_df_literal, rh, retrieve_hass_conf, optim_conf
        )
    else:
        success, days_list = await _retrieve_from_hass(
            set_type, retrieve_hass_conf, optim_conf, rh, logger
        )
    if not success:
        return False, None, days_list
    rh.prepare_data(
        retrieve_hass_conf["sensor_power_load_no_var_loads"],
        load_negative=retrieve_hass_conf["load_negative"],
        set_zero_min=retrieve_hass_conf["set_zero_min"],
        var_replace_zero=retrieve_hass_conf["sensor_replace_zero"],
        var_interp=retrieve_hass_conf["sensor_linear_interp"],
    )
    return True, rh.df_final.copy(), days_list


def is_model_outdated(model_path: pathlib.Path, max_age_hours: int, logger: logging.Logger) -> bool:
    """
    Check if the saved model file is outdated based on its modification time.

    :param model_path: Path to the saved model file.
    :type model_path: pathlib.Path
    :param max_age_hours: Maximum age in hours before model is considered outdated.
    :type max_age_hours: int
    :param logger: Logger object for logging information.
    :type logger: logging.Logger
    :return: True if model is outdated or doesn't exist, False otherwise.
    :rtype: bool
    """
    if not model_path.exists():
        logger.info("Adjusted PV model file does not exist, will train new model")
        return True

    if max_age_hours <= 0:
        logger.info("adjusted_pv_model_max_age is set to 0, forcing model re-fit")
        return True

    model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
    model_age = datetime.now() - model_mtime
    max_age = timedelta(hours=max_age_hours)

    if model_age > max_age:
        logger.info(
            f"Adjusted PV model is outdated (age: {model_age.total_seconds() / 3600:.1f}h, "
            f"max: {max_age_hours}h), will train new model"
        )
        return True
    else:
        logger.info(
            f"Using existing adjusted PV model (age: {model_age.total_seconds() / 3600:.1f}h, "
            f"max: {max_age_hours}h)"
        )
        return False


async def _retrieve_and_fit_pv_model(
    fcst: Forecast,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: pd.DataFrame,
) -> bool:
    """
    Helper function to retrieve data and fit the PV adjustment model.

    :param fcst: Forecast object used for PV forecast adjustment.
    :type fcst: Forecast
    :param get_data_from_file: Whether to retrieve data from a file instead of Home Assistant.
    :type get_data_from_file: bool
    :param retrieve_hass_conf: Configuration dictionary for retrieving data from Home Assistant.
    :type retrieve_hass_conf: dict
    :param optim_conf: Configuration dictionary for optimization settings.
    :type optim_conf: dict
    :param rh: RetrieveHass object for interacting with Home Assistant.
    :type rh: RetrieveHass
    :param emhass_conf: Configuration dictionary for emhass paths and settings.
    :type emhass_conf: dict
    :param test_df_literal: DataFrame containing test data for debugging purposes.
    :type test_df_literal: pd.DataFrame
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    # Retrieve data from Home Assistant
    success, df_input_data, _ = await retrieve_home_assistant_data(
        "adjust_pv",
        get_data_from_file,
        retrieve_hass_conf,
        optim_conf,
        rh,
        emhass_conf,
        test_df_literal,
    )
    if not success:
        return False
    # Call data preparation method
    fcst.adjust_pv_forecast_data_prep(df_input_data)
    # Call the fit method
    await fcst.adjust_pv_forecast_fit(
        n_splits=5,
        regression_model=optim_conf["adjusted_pv_regression_model"],
    )
    return True


async def adjust_pv_forecast(
    logger: logging.Logger,
    fcst: Forecast,
    p_pv_forecast: pd.Series,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: pd.DataFrame,
) -> pd.Series:
    """
    Adjust the photovoltaic (PV) forecast using historical data and a regression model.

    This method retrieves historical data, prepares it for model fitting, trains a regression
    model, and adjusts the provided PV forecast based on the trained model.

    :param logger: Logger object for logging information and errors.
    :type logger: logging.Logger
    :param fcst: Forecast object used for PV forecast adjustment.
    :type fcst: Forecast
    :param p_pv_forecast: The initial PV forecast to be adjusted.
    :type p_pv_forecast: pd.Series
    :param get_data_from_file: Whether to retrieve data from a file instead of Home Assistant.
    :type get_data_from_file: bool
    :param retrieve_hass_conf: Configuration dictionary for retrieving data from Home Assistant.
    :type retrieve_hass_conf: dict
    :param optim_conf: Configuration dictionary for optimization settings.
    :type optim_conf: dict
    :param rh: RetrieveHass object for interacting with Home Assistant.
    :type rh: RetrieveHass
    :param emhass_conf: Configuration dictionary for emhass paths and settings.
    :type emhass_conf: dict
    :param test_df_literal: DataFrame containing test data for debugging purposes.
    :type test_df_literal: pd.DataFrame
    :return: The adjusted PV forecast as a pandas Series.
    :rtype: pd.Series
    """
    # Normalize data_path to Path object for safety (handles both str and Path types)
    data_path = pathlib.Path(emhass_conf["data_path"])
    model_filename = "adjust_pv_regressor.pkl"
    model_path = data_path / model_filename
    max_age_hours = optim_conf.get("adjusted_pv_model_max_age", 24)
    # Check if model needs to be re-fitted
    if is_model_outdated(model_path, max_age_hours, logger):
        logger.info("Adjusting PV forecast, retrieving history data for model fit")
        success = await _retrieve_and_fit_pv_model(
            fcst,
            get_data_from_file,
            retrieve_hass_conf,
            optim_conf,
            rh,
            emhass_conf,
            test_df_literal,
        )
        if not success:
            return False
    else:
        # Load existing model
        logger.info("Loading existing adjusted PV model from file")
        try:
            async with aiofiles.open(model_path, "rb") as inp:
                content = await inp.read()
                fcst.model_adjust_pv = pickle.loads(content)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
            logger.error(f"Failed to load existing adjusted PV model: {type(e).__name__}: {str(e)}")
            logger.warning(
                "Model file may be corrupted or incompatible. Falling back to re-fitting the model."
            )
            # Use helper function to retrieve data and re-fit model
            success = await _retrieve_and_fit_pv_model(
                fcst,
                get_data_from_file,
                retrieve_hass_conf,
                optim_conf,
                rh,
                emhass_conf,
                test_df_literal,
            )
            if not success:
                logger.error("Failed to retrieve data for model re-fit after load error")
                return False
            logger.info("Successfully re-fitted model after load failure")
        except Exception as e:
            logger.error(
                f"Unexpected error loading adjusted PV model: {type(e).__name__}: {str(e)}"
            )
            logger.error("Cannot recover from this error")
            return False
    # Call the predict method
    p_pv_forecast = p_pv_forecast.rename("forecast").to_frame()
    p_pv_forecast = fcst.adjust_pv_forecast_predict(forecasted_pv=p_pv_forecast)
    # Update the PV forecast
    return p_pv_forecast["adjusted_forecast"].rename(None)


async def _prepare_perfect_optim(ctx: SetupContext):
    """Helper to prepare data for perfect optimization."""
    success, df_input_data, days_list = await retrieve_home_assistant_data(
        "perfect-optim",
        ctx.get_data_from_file,
        ctx.retrieve_hass_conf,
        ctx.optim_conf,
        ctx.rh,
        ctx.emhass_conf,
        test_df_literal,
        ctx.logger,
    )
    if not success:
        return None
    return {
        "df_input_data": df_input_data,
        "days_list": days_list,
    }


async def _get_dayahead_pv_forecast(ctx: SetupContext):
    """Helper to retrieve and optionally adjust PV forecast."""
    # Check if we should calculate PV forecast
    if not (
        ctx.optim_conf["set_use_pv"]
        or ctx.optim_conf.get("weather_forecast_method", None) == "list"
    ):
        return pd.Series(0, index=ctx.fcst.forecast_dates), None
    # Get weather forecast
    df_weather = await ctx.fcst.get_weather_forecast(
        method=ctx.optim_conf["weather_forecast_method"]
    )
    if isinstance(df_weather, bool) and not df_weather:
        return None, None
    p_pv_forecast = ctx.fcst.get_power_from_weather(df_weather)
    # Adjust PV forecast if needed
    if ctx.optim_conf["set_use_adjusted_pv"]:
        p_pv_forecast = await adjust_pv_forecast(
            ctx.logger,
            ctx.fcst,
            p_pv_forecast,
            ctx.get_data_from_file,
            ctx.retrieve_hass_conf,
            ctx.optim_conf,
            ctx.rh,
            ctx.emhass_conf,
            test_df_literal,
        )
    return p_pv_forecast, df_weather


def _apply_df_freq_horizon(
    df: pd.DataFrame, retrieve_hass_conf: dict, prediction_horizon: int | None
) -> pd.DataFrame:
    """Helper to apply frequency adjustment and prediction horizon slicing."""
    # Handle Frequency
    if retrieve_hass_conf.get("optimization_time_step"):
        step = retrieve_hass_conf["optimization_time_step"]
        if not isinstance(step, pd._libs.tslibs.timedeltas.Timedelta):
            step = pd.to_timedelta(step, "minute")
        df = df.asfreq(step)
    else:
        df = utils.set_df_index_freq(df)
    # Handle Prediction Horizon
    if prediction_horizon:
        # Slice the dataframe up to the horizon
        df = copy.deepcopy(df)[df.index[0] : df.index[prediction_horizon - 1]]
    return df


async def _prepare_dayahead_optim(ctx: SetupContext):
    """Helper to prepare data for day-ahead optimization."""
    # Get PV Forecast
    p_pv_forecast, df_weather = await _get_dayahead_pv_forecast(ctx)
    if p_pv_forecast is None:
        return None
    # Get Load Forecast
    p_load_forecast = await ctx.fcst.get_load_forecast(
        days_min_load_forecast=ctx.optim_conf["delta_forecast_daily"].days,
        method=ctx.optim_conf["load_forecast_method"],
    )
    if isinstance(p_load_forecast, bool) and not p_load_forecast:
        ctx.logger.error("Unable to get load forecast.")
        return None
    # Build Input DataFrame
    df_input_data_dayahead = pd.DataFrame(
        np.transpose(np.vstack([p_pv_forecast.values, p_load_forecast.values])),
        index=p_pv_forecast.index,
        columns=["p_pv_forecast", "p_load_forecast"],
    )
    # Apply Frequency and Prediction Horizon
    # Use explicitly passed horizon, avoiding JSON re-parsing
    prediction_horizon = ctx.params["passed_data"].get("prediction_horizon")
    df_input_data_dayahead = _apply_df_freq_horizon(
        df_input_data_dayahead, ctx.retrieve_hass_conf, prediction_horizon
    )
    return {
        "df_input_data_dayahead": df_input_data_dayahead,
        "df_weather": df_weather,
        "p_pv_forecast": p_pv_forecast,
        "p_load_forecast": p_load_forecast,
    }


async def _get_naive_mpc_history(ctx: SetupContext):
    """Helper to retrieve historical data for Naive MPC."""
    # Check if we need to skip historical data retrieval
    is_list_forecast = ctx.optim_conf.get("load_forecast_method") == "list"
    is_list_weather = ctx.optim_conf.get("weather_forecast_method") == "list"
    no_pv = not ctx.optim_conf["set_use_pv"]

    if (is_list_forecast and is_list_weather) or (is_list_forecast and no_pv):
        return True, None, None, False  # success, df, days_list, set_mix_forecast
    # Retrieve data from Home Assistant
    success, df_input_data, days_list = await retrieve_home_assistant_data(
        "naive-mpc-optim",
        ctx.get_data_from_file,
        ctx.retrieve_hass_conf,
        ctx.optim_conf,
        ctx.rh,
        ctx.emhass_conf,
        test_df_literal,
        ctx.logger,
    )
    return success, df_input_data, days_list, True


async def _get_naive_mpc_pv_forecast(ctx: SetupContext, set_mix_forecast, df_input_data):
    """Helper to generate PV forecast for Naive MPC."""
    # If PV is disabled and no weather list, return zero series
    if not (
        ctx.optim_conf["set_use_pv"] or ctx.optim_conf.get("weather_forecast_method") == "list"
    ):
        return pd.Series(0, index=ctx.fcst.forecast_dates), None
    # Get weather forecast
    df_weather = await ctx.fcst.get_weather_forecast(
        method=ctx.optim_conf["weather_forecast_method"]
    )
    if isinstance(df_weather, bool) and not df_weather:
        return None, None
    # Calculate PV power
    p_pv_forecast = ctx.fcst.get_power_from_weather(
        df_weather, set_mix_forecast=set_mix_forecast, df_now=df_input_data
    )
    # Adjust PV forecast if needed
    if ctx.optim_conf["set_use_adjusted_pv"]:
        p_pv_forecast = await adjust_pv_forecast(
            ctx.logger,
            ctx.fcst,
            p_pv_forecast,
            ctx.get_data_from_file,
            ctx.retrieve_hass_conf,
            ctx.optim_conf,
            ctx.rh,
            ctx.emhass_conf,
            test_df_literal,
        )
    return p_pv_forecast, df_weather


async def _prepare_naive_mpc_optim(ctx: SetupContext):
    """Helper to prepare data for Naive MPC optimization."""
    # Retrieve Historical Data
    success, df_input_data, days_list, set_mix_forecast = await _get_naive_mpc_history(ctx)
    if not success:
        return None
    # Get PV Forecast
    p_pv_forecast, df_weather = await _get_naive_mpc_pv_forecast(
        ctx, set_mix_forecast, df_input_data
    )
    if p_pv_forecast is None:
        return None
    # Get Load Forecast
    p_load_forecast = await ctx.fcst.get_load_forecast(
        days_min_load_forecast=ctx.optim_conf["delta_forecast_daily"].days,
        method=ctx.optim_conf["load_forecast_method"],
        set_mix_forecast=set_mix_forecast,
        df_now=df_input_data,
    )
    if isinstance(p_load_forecast, bool) and not p_load_forecast:
        return None
    # Build and Format Input DataFrame
    df_input_data_dayahead = pd.concat([p_pv_forecast, p_load_forecast], axis=1)
    df_input_data_dayahead.columns = ["p_pv_forecast", "p_load_forecast"]
    # Reuse freq/horizon helper
    prediction_horizon = ctx.params["passed_data"].get("prediction_horizon")
    df_input_data_dayahead = _apply_df_freq_horizon(
        df_input_data_dayahead, ctx.retrieve_hass_conf, prediction_horizon
    )
    return {
        "df_input_data": df_input_data,
        "days_list": days_list,
        "df_input_data_dayahead": df_input_data_dayahead,
        "df_weather": df_weather,
        "p_pv_forecast": p_pv_forecast,
        "p_load_forecast": p_load_forecast,
    }


async def _prepare_ml_fit_predict(ctx: SetupContext):
    """Helper to prepare data for ML fit/predict/tune."""
    days_to_retrieve = ctx.params["passed_data"]["historic_days_to_retrieve"]
    model_type = ctx.params["passed_data"]["model_type"]
    var_model = ctx.params["passed_data"]["var_model"]
    if ctx.get_data_from_file:
        filename = model_type + ".pkl"
        filename_path = ctx.emhass_conf["data_path"] / filename
        async with aiofiles.open(filename_path, "rb") as inp:
            content = await inp.read()
            df_input_data, _, _, _ = pickle.loads(content)
        df_input_data = df_input_data[df_input_data.index[-1] - pd.offsets.Day(days_to_retrieve) :]
        return {"df_input_data": df_input_data}
    else:
        days_list = utils.get_days_list(days_to_retrieve)
        var_list = [var_model]
        if not await ctx.rh.get_data(days_list, var_list):
            return None
        ctx.rh.prepare_data(
            var_model,
            load_negative=ctx.retrieve_hass_conf.get("load_negative", False),
            set_zero_min=ctx.retrieve_hass_conf.get("set_zero_min", True),
            var_replace_zero=ctx.retrieve_hass_conf.get("sensor_replace_zero", []),
            var_interp=ctx.retrieve_hass_conf.get("sensor_linear_interp", []),
            skip_renaming=True,
        )
        return {"df_input_data": ctx.rh.df_final.copy()}


def _prepare_regressor_fit(ctx: SetupContext):
    """Helper to prepare data for Regressor fit/predict."""
    csv_file = ctx.params["passed_data"].get("csv_file", None)
    if not csv_file:
        ctx.logger.error("csv_file is required for regressor actions but was not provided.")
        return None
    if ctx.get_data_from_file:
        base_path = ctx.emhass_conf["data_path"]
        filename_path = pathlib.Path(base_path) / csv_file
    else:
        filename_path = ctx.emhass_conf["data_path"] / csv_file
    if filename_path.is_file():
        df_input_data = pd.read_csv(filename_path, parse_dates=True)
    else:
        ctx.logger.error(
            f"The CSV file {csv_file} was not found in path: {ctx.emhass_conf['data_path']}"
        )
        return None
    # Validate columns
    required_columns = []
    if "features" in ctx.params["passed_data"]:
        required_columns.extend(ctx.params["passed_data"]["features"])
    if "target" in ctx.params["passed_data"]:
        required_columns.append(ctx.params["passed_data"]["target"])
    if "timestamp" in ctx.params["passed_data"]:
        required_columns.append(ctx.params["passed_data"]["timestamp"])
    if not set(required_columns).issubset(df_input_data.columns):
        ctx.logger.error(
            f"The csv file does not contain the required columns: {', '.join(required_columns)}"
        )
        return None
    return {"df_input_data": df_input_data}


async def set_input_data_dict(
    emhass_conf: dict,
    costfun: str,
    params: str,
    runtimeparams: str,
    set_type: str,
    logger: logging.Logger,
    get_data_from_file: bool | None = False,
) -> dict:
    """
    Set up some of the data needed for the different actions.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param costfun: The type of cost function to use for optimization problem
    :type costfun: str
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param runtimeparams: Runtime optimization parameters passed as a dictionary
    :type runtimeparams: str
    :param set_type: Set the type of setup based on following type of optimization
    :type set_type: str
    :param logger: The passed logger object
    :type logger: logging object
    :param get_data_from_file: Use data from saved CSV file (useful for debug)
    :type get_data_from_file: bool, optional
    :return: A dictionnary with multiple data used by the action functions
    :rtype: dict

    """
    logger.info("Setting up needed data")
    # Parse Parameters
    if (params is not None) and (params != "null"):
        if isinstance(params, str):
            params = dict(orjson.loads(params))
    else:
        params = {}
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
    if type(retrieve_hass_conf) is bool:
        return False
    (
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
    ) = await utils.treat_runtimeparams(
        runtimeparams,
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        set_type,
        logger,
        emhass_conf,
    )
    if isinstance(params, str):
        params = dict(orjson.loads(params))
    # Initialize Core Objects
    rh = RetrieveHass(
        retrieve_hass_conf["hass_url"],
        retrieve_hass_conf["long_lived_token"],
        retrieve_hass_conf["optimization_time_step"],
        retrieve_hass_conf["time_zone"],
        params,
        emhass_conf,
        logger,
        get_data_from_file=get_data_from_file,
    )
    # Retrieve HA config
    if get_data_from_file:
        async with aiofiles.open(emhass_conf["data_path"] / test_df_literal, "rb") as inp:
            content = await inp.read()
            _, _, _, rh.ha_config = pickle.loads(content)
    elif not await rh.get_ha_config():
        return False
    if isinstance(params, dict):
        params_str = orjson.dumps(params).decode("utf-8")
        params = utils.update_params_with_ha_config(params_str, rh.ha_config)
    else:
        params = utils.update_params_with_ha_config(params, rh.ha_config)
    if isinstance(params, str):
        params = dict(orjson.loads(params))
    costfun = optim_conf.get("costfun", costfun)
    fcst = Forecast(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        params,
        emhass_conf,
        logger,
        get_data_from_file=get_data_from_file,
    )
    opt = Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        fcst.var_load_cost,
        fcst.var_prod_price,
        costfun,
        emhass_conf,
        logger,
    )
    # Create SetupContext
    ctx = SetupContext(
        retrieve_hass_conf=retrieve_hass_conf,
        optim_conf=optim_conf,
        plant_conf=plant_conf,
        emhass_conf=emhass_conf,
        params=params,
        logger=logger,
        get_data_from_file=get_data_from_file,
        rh=rh,
        fcst=fcst,
    )
    # Initialize Default Return Data
    data_results = {
        "df_input_data": None,
        "df_input_data_dayahead": None,
        "df_weather": None,
        "p_pv_forecast": None,
        "p_load_forecast": None,
        "days_list": None,
    }
    # Delegate to Helpers based on set_type
    result = None
    if set_type == "perfect-optim":
        result = await _prepare_perfect_optim(ctx)
    elif set_type == "dayahead-optim":
        result = await _prepare_dayahead_optim(ctx)
    elif set_type == "naive-mpc-optim":
        result = await _prepare_naive_mpc_optim(ctx)
    elif set_type in ["forecast-model-fit", "forecast-model-predict", "forecast-model-tune"]:
        result = await _prepare_ml_fit_predict(ctx)
    elif set_type == "regressor-model-fit":
        result = _prepare_regressor_fit(ctx)
    elif set_type == "regressor-model-predict":
        if get_data_from_file:
            result = _prepare_regressor_fit(ctx)
        else:
            result = {}
    elif set_type == "publish-data" or set_type == "export-influxdb-to-csv":
        result = {}
    else:
        logger.error(f"The passed action set_type parameter '{set_type}' is not valid")
        result = {}
    if result is None:
        return False
    data_results.update(result)
    # Build Final Dictionary
    input_data_dict = {
        "emhass_conf": emhass_conf,
        "retrieve_hass_conf": retrieve_hass_conf,
        "rh": rh,
        "opt": opt,
        "fcst": fcst,
        "costfun": costfun,
        "params": params,
        **data_results,
    }
    return input_data_dict


async def weather_forecast_cache(
    emhass_conf: dict, params: str, runtimeparams: str, logger: logging.Logger
) -> bool:
    """
    Perform a call to get forecast function, intend to save results to cache.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param runtimeparams: Runtime optimization parameters passed as a dictionary
    :type runtimeparams: str
    :param logger: The passed logger object
    :type logger: logging object
    :return: A bool for function completion
    :rtype: bool

    """
    # Parsing yaml
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
    # Treat runtimeparams
    (
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
    ) = await utils.treat_runtimeparams(
        runtimeparams,
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "forecast",
        logger,
        emhass_conf,
    )
    # Make sure weather_forecast_cache is true
    if (params is not None) and (params != "null"):
        params = orjson.loads(params)
    else:
        params = {}
    params["passed_data"]["weather_forecast_cache"] = True
    params = orjson.dumps(params).decode("utf-8")
    # Create Forecast object
    fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf, params, emhass_conf, logger)
    result = await fcst.get_weather_forecast(optim_conf["weather_forecast_method"])
    if isinstance(result, bool) and not result:
        return False

    return True


async def perfect_forecast_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = True,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the perfect forecast optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    logger.info("Performing perfect forecast optimization")
    # Load cost and prod price forecast
    df_input_data = input_data_dict["fcst"].get_load_cost_forecast(
        input_data_dict["df_input_data"],
        method=input_data_dict["fcst"].optim_conf["load_cost_forecast_method"],
        list_and_perfect=True,
    )
    if isinstance(df_input_data, bool) and not df_input_data:
        return False
    df_input_data = input_data_dict["fcst"].get_prod_price_forecast(
        df_input_data,
        method=input_data_dict["fcst"].optim_conf["production_price_forecast_method"],
        list_and_perfect=True,
    )
    if isinstance(df_input_data, bool) and not df_input_data:
        return False
    opt_res = input_data_dict["opt"].perform_perfect_forecast_optim(
        df_input_data, input_data_dict["days_list"]
    )
    # Save CSV file for analysis
    if save_data_to_file:
        filename = "opt_res_perfect_optim_" + input_data_dict["costfun"] + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )
    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save perfect results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        # Trigger the publish function, save entity data and not post to HA
        await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    return opt_res


def prepare_forecast_and_weather_data(
    input_data_dict: dict,
    logger: logging.Logger,
    warn_on_resolution: bool = False,
) -> pd.DataFrame | bool:
    """
    Prepare forecast data with load costs, production prices, outdoor temperature, and GHI.

    This helper function eliminates duplication between dayahead_forecast_optim and naive_mpc_optim.

    :param input_data_dict: Dictionary with forecast and input data
    :type input_data_dict: dict
    :param logger: Logger object
    :type logger: logging.Logger
    :param warn_on_resolution: Whether to warn about GHI resolution mismatch
    :type warn_on_resolution: bool
    :return: Prepared DataFrame or False on error
    :rtype: pd.DataFrame | bool
    """
    # Get load cost forecast
    df_input_data_dayahead = input_data_dict["fcst"].get_load_cost_forecast(
        input_data_dict["df_input_data_dayahead"],
        method=input_data_dict["fcst"].optim_conf["load_cost_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False

    # Get production price forecast
    df_input_data_dayahead = input_data_dict["fcst"].get_prod_price_forecast(
        df_input_data_dayahead,
        method=input_data_dict["fcst"].optim_conf["production_price_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False

    # Add outdoor temperature if provided
    if "outdoor_temperature_forecast" in input_data_dict["params"]["passed_data"]:
        df_input_data_dayahead["outdoor_temperature_forecast"] = input_data_dict["params"][
            "passed_data"
        ]["outdoor_temperature_forecast"]

    # Auto-fallback to temp_air from Open-Meteo weather forecast
    elif (
        input_data_dict["df_weather"] is not None
        and "temp_air" in input_data_dict["df_weather"].columns
    ):
        dayahead_index = df_input_data_dayahead.index
        # Align temp_air data to dayahead index using interpolation
        df_input_data_dayahead["temp_air"] = (
            input_data_dict["df_weather"]["temp_air"]
            .reindex(dayahead_index)
            .interpolate(method="time", limit_direction="both")
        )

    # Merge GHI (Global Horizontal Irradiance) from weather forecast if available
    if input_data_dict["df_weather"] is not None and "ghi" in input_data_dict["df_weather"].columns:
        dayahead_index = df_input_data_dayahead.index

        # Check time resolution if requested
        if (
            warn_on_resolution
            and len(input_data_dict["df_weather"].index) > 1
            and len(dayahead_index) > 1
        ):
            weather_index = input_data_dict["df_weather"].index
            weather_freq = (weather_index[1] - weather_index[0]).total_seconds()
            dayahead_freq = (dayahead_index[1] - dayahead_index[0]).total_seconds()
            if weather_freq > 2 * dayahead_freq:
                logger.warning(
                    "Weather data time resolution (%.0fs) is much coarser than dayahead index (%.0fs). "
                    "Step changes in GHI may occur.",
                    weather_freq,
                    dayahead_freq,
                )

        # Align GHI data to dayahead index using interpolation
        df_input_data_dayahead["ghi"] = (
            input_data_dict["df_weather"]["ghi"]
            .reindex(dayahead_index)
            .interpolate(method="time", limit_direction="both")
        )
        logger.debug(
            "Merged GHI data into optimization input: mean=%.1f W/m², max=%.1f W/m²",
            df_input_data_dayahead["ghi"].mean(),
            df_input_data_dayahead["ghi"].max(),
        )

    return df_input_data_dayahead


async def dayahead_forecast_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the day-ahead optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    logger.info("Performing day-ahead forecast optimization")
    # Prepare forecast data with costs, prices, outdoor temp, and GHI
    df_input_data_dayahead = prepare_forecast_and_weather_data(
        input_data_dict, logger, warn_on_resolution=False
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    opt_res_dayahead = input_data_dict["opt"].perform_dayahead_forecast_optim(
        df_input_data_dayahead,
        input_data_dict["p_pv_forecast"],
        input_data_dict["p_load_forecast"],
    )
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_dayahead_" + today.strftime("%Y_%m_%d") + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res_dayahead.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )

    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save day_ahead results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        # Trigger the publish function, save entity data and not post to HA
        await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    return opt_res_dayahead


async def naive_mpc_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the naive Model Predictive Controller optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    logger.info("Performing naive MPC optimization")
    # Prepare forecast data with costs, prices, outdoor temp, and GHI (with resolution warning)
    df_input_data_dayahead = prepare_forecast_and_weather_data(
        input_data_dict, logger, warn_on_resolution=True
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    # The specifics params for the MPC at runtime
    prediction_horizon = input_data_dict["params"]["passed_data"]["prediction_horizon"]
    soc_init = input_data_dict["params"]["passed_data"]["soc_init"]
    soc_final = input_data_dict["params"]["passed_data"]["soc_final"]
    def_total_hours = input_data_dict["params"]["optim_conf"].get(
        "operating_hours_of_each_deferrable_load", None
    )
    def_total_timestep = input_data_dict["params"]["optim_conf"].get(
        "operating_timesteps_of_each_deferrable_load", None
    )
    def_start_timestep = input_data_dict["params"]["optim_conf"][
        "start_timesteps_of_each_deferrable_load"
    ]
    def_end_timestep = input_data_dict["params"]["optim_conf"][
        "end_timesteps_of_each_deferrable_load"
    ]
    opt_res_naive_mpc = input_data_dict["opt"].perform_naive_mpc_optim(
        df_input_data_dayahead,
        input_data_dict["p_pv_forecast"],
        input_data_dict["p_load_forecast"],
        prediction_horizon,
        soc_init,
        soc_final,
        def_total_hours,
        def_total_timestep,
        def_start_timestep,
        def_end_timestep,
    )
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_naive_mpc_" + today.strftime("%Y_%m_%d") + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res_naive_mpc.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )

    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save mpc results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        # Trigger the publish function, save entity data and not post to HA
        await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    return opt_res_naive_mpc


async def forecast_model_fit(
    input_data_dict: dict, logger: logging.Logger, debug: bool | None = False
) -> tuple[pd.DataFrame, pd.DataFrame, MLForecaster]:
    """Perform a forecast model fit from training data retrieved from Home Assistant.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :return: The DataFrame containing the forecast data results without and with backtest and the `mlforecaster` object
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, mlforecaster]
    """
    data = copy.deepcopy(input_data_dict["df_input_data"])
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    var_model = input_data_dict["params"]["passed_data"]["var_model"]
    sklearn_model = input_data_dict["params"]["passed_data"]["sklearn_model"]
    num_lags = input_data_dict["params"]["passed_data"]["num_lags"]
    split_date_delta = input_data_dict["params"]["passed_data"]["split_date_delta"]
    perform_backtest = input_data_dict["params"]["passed_data"]["perform_backtest"]
    # The ML forecaster object
    mlf = MLForecaster(
        data,
        model_type,
        var_model,
        sklearn_model,
        num_lags,
        input_data_dict["emhass_conf"],
        logger,
    )
    # Fit the ML model
    df_pred, df_pred_backtest = await mlf.fit(
        split_date_delta=split_date_delta, perform_backtest=perform_backtest
    )
    # Save model
    if not debug:
        filename = model_type + default_pkl_suffix
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlf, pickle.HIGHEST_PROTOCOL))
            logger.debug("saved model to " + str(filename_path))
    return df_pred, df_pred_backtest, mlf


async def forecast_model_predict(
    input_data_dict: dict,
    logger: logging.Logger,
    use_last_window: bool | None = True,
    debug: bool | None = False,
    mlf: MLForecaster | None = None,
) -> pd.DataFrame:
    r"""Perform a forecast model predict using a previously trained skforecast model.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param use_last_window: True if the 'last_window' option should be used for the \
        custom machine learning forecast model. The 'last_window=True' means that the data \
        that will be used to generate the new forecast will be freshly retrieved from \
        Home Assistant. This data is needed because the forecast model is an auto-regressive \
        model with lags. If 'False' then the data using during the model train is used. Defaults to True
    :type use_last_window: Optional[bool], optional
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
        and unit testing. In production the actual model will be read from a saved pickle file. Defaults to None
    :type mlf: Optional[mlforecaster], optional
    :return: The DataFrame containing the forecast prediction data
    :rtype: pd.DataFrame
    """
    # Load model
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    filename = model_type + default_pkl_suffix
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlf = pickle.loads(content)
                logger.debug("loaded saved model from " + str(filename_path))
        else:
            logger.error(
                "The ML forecaster file ("
                + str(filename_path)
                + ") was not found, please run a model fit method before this predict method",
            )
            return
    # Make predictions
    if use_last_window:
        data_last_window = copy.deepcopy(input_data_dict["df_input_data"])
    else:
        data_last_window = None
    predictions = await mlf.predict(data_last_window)
    # Publish data to a Home Assistant sensor
    model_predict_publish = input_data_dict["params"]["passed_data"]["model_predict_publish"]
    model_predict_entity_id = input_data_dict["params"]["passed_data"]["model_predict_entity_id"]
    model_predict_device_class = input_data_dict["params"]["passed_data"][
        "model_predict_device_class"
    ]
    model_predict_unit_of_measurement = input_data_dict["params"]["passed_data"][
        "model_predict_unit_of_measurement"
    ]
    model_predict_friendly_name = input_data_dict["params"]["passed_data"][
        "model_predict_friendly_name"
    ]
    publish_prefix = input_data_dict["params"]["passed_data"]["publish_prefix"]
    if model_predict_publish is True:
        # Estimate the current index
        now_precise = datetime.now(input_data_dict["retrieve_hass_conf"]["time_zone"]).replace(
            second=0, microsecond=0
        )
        if input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "nearest":
            idx_closest = predictions.index.get_indexer([now_precise], method="nearest")[0]
        elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "first":
            idx_closest = predictions.index.get_indexer([now_precise], method="ffill")[0]
        elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "last":
            idx_closest = predictions.index.get_indexer([now_precise], method="bfill")[0]
        if idx_closest == -1:
            idx_closest = predictions.index.get_indexer([now_precise], method="nearest")[0]
        # Publish Load forecast
        await input_data_dict["rh"].post_data(
            predictions,
            idx_closest,
            model_predict_entity_id,
            model_predict_device_class,
            model_predict_unit_of_measurement,
            model_predict_friendly_name,
            type_var="mlforecaster",
            publish_prefix=publish_prefix,
        )
    return predictions


async def forecast_model_tune(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
    mlf: MLForecaster | None = None,
) -> tuple[pd.DataFrame, MLForecaster]:
    """Tune a forecast model hyperparameters using bayesian optimization.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
        and unit testing. In production the actual model will be read from a saved pickle file. Defaults to None
    :type mlf: Optional[mlforecaster], optional
    :return: The DataFrame containing the forecast data results using the optimized model
    :rtype: pd.DataFrame
    """
    # Load model
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    filename = model_type + default_pkl_suffix
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlf = pickle.loads(content)
                logger.debug("loaded saved model from " + str(filename_path))
        else:
            logger.error(
                "The ML forecaster file ("
                + str(filename_path)
                + ") was not found, please run a model fit method before this tune method",
            )
            return None, None
    # Tune the model
    split_date_delta = input_data_dict["params"]["passed_data"]["split_date_delta"]
    if debug:
        n_trials = 5
    else:
        n_trials = input_data_dict["params"]["passed_data"]["n_trials"]
    df_pred_optim = await mlf.tune(
        split_date_delta=split_date_delta, n_trials=n_trials, debug=debug
    )
    # Save model
    if not debug:
        filename = model_type + default_pkl_suffix
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlf, pickle.HIGHEST_PROTOCOL))
            logger.debug("Saved model to " + str(filename_path))
    return df_pred_optim, mlf


async def regressor_model_fit(
    input_data_dict: dict, logger: logging.Logger, debug: bool | None = False
) -> MLRegressor:
    """Perform a forecast model fit from training data retrieved from Home Assistant.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    """
    data = copy.deepcopy(input_data_dict["df_input_data"])
    if "model_type" in input_data_dict["params"]["passed_data"]:
        model_type = input_data_dict["params"]["passed_data"]["model_type"]
    else:
        logger.error("parameter: 'model_type' not passed")
        return False
    if "regression_model" in input_data_dict["params"]["passed_data"]:
        regression_model = input_data_dict["params"]["passed_data"]["regression_model"]
    else:
        logger.error("parameter: 'regression_model' not passed")
        return False
    if "features" in input_data_dict["params"]["passed_data"]:
        features = input_data_dict["params"]["passed_data"]["features"]
    else:
        logger.error("parameter: 'features' not passed")
        return False
    if "target" in input_data_dict["params"]["passed_data"]:
        target = input_data_dict["params"]["passed_data"]["target"]
    else:
        logger.error("parameter: 'target' not passed")
        return False
    if "timestamp" in input_data_dict["params"]["passed_data"]:
        timestamp = input_data_dict["params"]["passed_data"]["timestamp"]
    else:
        logger.error("parameter: 'timestamp' not passed")
        return False
    if "date_features" in input_data_dict["params"]["passed_data"]:
        date_features = input_data_dict["params"]["passed_data"]["date_features"]
    else:
        logger.error("parameter: 'date_features' not passed")
        return False
    # The MLRegressor object
    mlr = MLRegressor(data, model_type, regression_model, features, target, timestamp, logger)
    # Fit the ML model
    fit = await mlr.fit(date_features=date_features)
    if not fit:
        return False
    # Save model
    if not debug:
        filename = model_type + "_mlr.pkl"
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlr, pickle.HIGHEST_PROTOCOL))
    return mlr


async def regressor_model_predict(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
    mlr: MLRegressor | None = None,
) -> np.ndarray:
    """Perform a prediction from csv file.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    """
    if "model_type" in input_data_dict["params"]["passed_data"]:
        model_type = input_data_dict["params"]["passed_data"]["model_type"]
    else:
        logger.error("parameter: 'model_type' not passed")
        return False
    filename = model_type + "_mlr.pkl"
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlr = pickle.loads(content)
        else:
            logger.error(
                "The ML forecaster file was not found, please run a model fit method before this predict method",
            )
            return False
    if "new_values" in input_data_dict["params"]["passed_data"]:
        new_values = input_data_dict["params"]["passed_data"]["new_values"]
    else:
        logger.error("parameter: 'new_values' not passed")
        return False
    # Predict from csv file
    prediction = await mlr.predict(new_values)
    mlr_predict_entity_id = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_entity_id", "sensor.mlr_predict"
    )
    mlr_predict_device_class = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_device_class", "power"
    )
    mlr_predict_unit_of_measurement = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_unit_of_measurement", "W"
    )
    mlr_predict_friendly_name = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_friendly_name", "mlr predictor"
    )
    # Publish prediction
    idx = 0
    if not debug:
        await input_data_dict["rh"].post_data(
            prediction,
            idx,
            mlr_predict_entity_id,
            mlr_predict_device_class,
            mlr_predict_unit_of_measurement,
            mlr_predict_friendly_name,
            type_var="mlregressor",
        )
    return prediction


async def export_influxdb_to_csv(
    input_data_dict: dict | None,
    logger: logging.Logger,
    emhass_conf: dict | None = None,
    params: str | None = None,
    runtimeparams: str | None = None,
) -> bool:
    """Export data from InfluxDB to CSV file.

    This function can be called in two ways:
    1. With input_data_dict (from web_server via set_input_data_dict)
    2. Without input_data_dict (direct call from command line or web_server before set_input_data_dict)

    :param input_data_dict: Dictionary containing configuration and parameters (optional)
    :type input_data_dict: dict | None
    :param logger: Logger object
    :type logger: logging.Logger
    :param emhass_conf: Dictionary containing EMHASS configuration paths (used when input_data_dict is None)
    :type emhass_conf: dict | None
    :param params: JSON string of params (used when input_data_dict is None)
    :type params: str | None
    :param runtimeparams: JSON string of runtime parameters (used when input_data_dict is None)
    :type runtimeparams: str | None
    :return: Success status
    :rtype: bool
    """
    # Handle two calling modes
    if input_data_dict is None:
        # Direct mode: parse params and create RetrieveHass
        if emhass_conf is None or params is None:
            logger.error("emhass_conf and params are required when input_data_dict is None")
            return False
        # Parse params
        if isinstance(params, str):
            params = orjson.loads(params)
        if isinstance(runtimeparams, str):
            runtimeparams = orjson.loads(runtimeparams)
        # Get configuration
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
        if isinstance(retrieve_hass_conf, bool):
            return False
        # Treat runtime params
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            orjson.dumps(runtimeparams).decode("utf-8") if runtimeparams else "{}",
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "export-influxdb-to-csv",
            logger,
            emhass_conf,
        )
        # Parse params again if it's a string
        if isinstance(params, str):
            params = orjson.loads(params)
        # Create RetrieveHass object
        rh = RetrieveHass(
            retrieve_hass_conf["hass_url"],
            retrieve_hass_conf["long_lived_token"],
            retrieve_hass_conf["optimization_time_step"],
            retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
        )
        time_zone = rh.time_zone
        data_path = emhass_conf["data_path"]
    else:
        # Standard mode: use input_data_dict
        params = input_data_dict["params"]
        if isinstance(params, str):
            params = orjson.loads(params)
        rh = input_data_dict["rh"]
        time_zone = rh.time_zone
        data_path = input_data_dict["emhass_conf"]["data_path"]
    # Extract parameters from passed_data
    if "sensor_list" not in params.get("passed_data", {}):
        logger.error("parameter: 'sensor_list' not passed")
        return False
    sensor_list = params["passed_data"]["sensor_list"]
    if "csv_filename" not in params.get("passed_data", {}):
        logger.error("parameter: 'csv_filename' not passed")
        return False
    csv_filename = params["passed_data"]["csv_filename"]
    if "start_time" not in params.get("passed_data", {}):
        logger.error("parameter: 'start_time' not passed")
        return False
    start_time = params["passed_data"]["start_time"]
    # Optional parameters with defaults
    end_time = params["passed_data"].get("end_time", None)
    resample_freq = params["passed_data"].get("resample_freq", "1h")
    timestamp_col = params["passed_data"].get("timestamp_col_name", "timestamp")
    decimal_places = params["passed_data"].get("decimal_places", 2)
    handle_nan = params["passed_data"].get("handle_nan", "keep")
    # Check if InfluxDB is enabled
    if not rh.use_influxdb:
        logger.error(
            "InfluxDB is not enabled in configuration. Set use_influxdb: true in config.json"
        )
        return False
    # Parse time range
    start_dt, end_dt = utils.parse_export_time_range(start_time, end_time, time_zone, logger)
    if start_dt is False:
        return False
    # Create days list for data retrieval
    days_list = pd.date_range(start=start_dt.date(), end=end_dt.date(), freq="D", tz=time_zone)
    if len(days_list) == 0:
        logger.error("No days to retrieve. Check start_time and end_time.")
        return False
    logger.info(
        f"Retrieving {len(sensor_list)} sensors from {start_dt} to {end_dt} ({len(days_list)} days)"
    )
    logger.info(f"Sensors: {sensor_list}")
    # Retrieve data from InfluxDB
    success = rh.get_data(days_list, sensor_list)
    if not success or rh.df_final is None or rh.df_final.empty:
        logger.error("Failed to retrieve data from InfluxDB")
        return False
    # Filter and resample data
    df_export = utils.resample_and_filter_data(rh.df_final, start_dt, end_dt, resample_freq, logger)
    if df_export is False:
        return False
    # Reset index to make timestamp a column
    # Handle custom index names by renaming the index first
    df_export = df_export.rename_axis(timestamp_col).reset_index()
    # Clean column names
    df_export = utils.clean_sensor_column_names(df_export, timestamp_col)
    # Handle NaN values
    df_export = utils.handle_nan_values(df_export, handle_nan, timestamp_col, logger)
    # Round numeric columns to specified decimal places
    numeric_cols = df_export.select_dtypes(include=[np.number]).columns
    df_export[numeric_cols] = df_export[numeric_cols].round(decimal_places)
    # Save to CSV
    csv_path = pathlib.Path(data_path) / csv_filename
    df_export.to_csv(csv_path, index=False)
    logger.info(f"✓ Successfully exported to {csv_filename}")
    logger.info(f"  Rows: {df_export.shape[0]}")
    logger.info(f"  Columns: {list(df_export.columns)}")
    logger.info(
        f"  Time range: {df_export[timestamp_col].min()} to {df_export[timestamp_col].max()}"
    )
    logger.info(f"  File location: {csv_path}")
    return True


def _get_params(input_data_dict: dict) -> dict:
    """Helper to extract params from input_data_dict."""
    if input_data_dict:
        if not isinstance(input_data_dict.get("params", {}), dict):
            return orjson.loads(input_data_dict["params"])
        return input_data_dict.get("params", {})
    return {}


async def _publish_from_saved_entities(
    input_data_dict: dict, logger: logging.Logger, params: dict
) -> pd.DataFrame | None:
    """
    Helper to publish data from saved entity JSON files if publish_prefix is set.
    Returns DataFrame if successful, None if fallback to CSV is needed.
    """
    publish_prefix = params["passed_data"].get("publish_prefix", "")
    entity_path = input_data_dict["emhass_conf"]["data_path"] / "entities"
    if not entity_path.exists() or not os.listdir(entity_path):
        logger.warning(f"No saved entity json files in path: {entity_path}")
        logger.warning("Falling back to opt_res_latest")
        return None
    entity_path_contents = os.listdir(entity_path)
    matches_prefix = any(publish_prefix in entity for entity in entity_path_contents)
    if not (matches_prefix or publish_prefix == "all"):
        logger.warning(f"No saved entity json files that match prefix: {publish_prefix}")
        logger.warning("Falling back to opt_res_latest")
        return None
    opt_res_list = []
    opt_res_list_names = []
    for entity in entity_path_contents:
        if entity == default_metadata_json:
            continue
        if publish_prefix == "all" or publish_prefix in entity:
            entity_data = await publish_json(entity, input_data_dict, entity_path, logger)
            if isinstance(entity_data, bool):
                return None  # Error occurred
            opt_res_list.append(entity_data)
            opt_res_list_names.append(entity.replace(".json", ""))
    opt_res = pd.concat(opt_res_list, axis=1)
    opt_res.columns = opt_res_list_names
    return opt_res


def _load_opt_res_latest(
    input_data_dict: dict, logger: logging.Logger, save_data_to_file: bool
) -> pd.DataFrame | None:
    """Helper to load the optimization results DataFrame from CSV."""
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_dayahead_" + today.strftime("%Y_%m_%d") + ".csv"
    else:
        filename = default_csv_filename
    file_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not file_path.exists():
        logger.error("File not found error, run an optimization task first.")
        return None
    opt_res_latest = pd.read_csv(file_path, index_col="timestamp")
    opt_res_latest.index = pd.to_datetime(opt_res_latest.index)
    opt_res_latest.index.freq = input_data_dict["retrieve_hass_conf"]["optimization_time_step"]
    return opt_res_latest


def _get_closest_index(retrieve_hass_conf: dict, index: pd.DatetimeIndex) -> int:
    """Helper to find the closest index in the DataFrame to the current time."""
    now_precise = datetime.now(retrieve_hass_conf["time_zone"]).replace(second=0, microsecond=0)
    method = retrieve_hass_conf["method_ts_round"]
    if method == "nearest":
        return index.get_indexer([now_precise], method="nearest")[0]
    elif method == "first":
        return index.get_indexer([now_precise], method="ffill")[0]
    elif method == "last":
        return index.get_indexer([now_precise], method="bfill")[0]
    return index.get_indexer([now_precise], method="nearest")[0]


async def _publish_standard_forecasts(
    ctx: PublishContext, opt_res_latest: pd.DataFrame
) -> list[str]:
    """Publish PV, Load, Curtailment, and Hybrid Inverter data."""
    cols = []
    # PV Forecast
    custom_pv = ctx.params["passed_data"]["custom_pv_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_PV"],
        ctx.idx,
        custom_pv["entity_id"],
        "power",
        custom_pv["unit_of_measurement"],
        custom_pv["friendly_name"],
        type_var="power",
        **ctx.common_kwargs,
    )
    cols.append("P_PV")
    # Load Forecast
    custom_load = ctx.params["passed_data"]["custom_load_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_Load"],
        ctx.idx,
        custom_load["entity_id"],
        "power",
        custom_load["unit_of_measurement"],
        custom_load["friendly_name"],
        type_var="power",
        **ctx.common_kwargs,
    )
    cols.append("P_Load")
    # Curtailment
    if ctx.fcst.plant_conf["compute_curtailment"]:
        custom_curt = ctx.params["passed_data"]["custom_pv_curtailment_id"]
        await ctx.rh.post_data(
            opt_res_latest["P_PV_curtailment"],
            ctx.idx,
            custom_curt["entity_id"],
            "power",
            custom_curt["unit_of_measurement"],
            custom_curt["friendly_name"],
            type_var="power",
            **ctx.common_kwargs,
        )
        cols.append("P_PV_curtailment")
    # Hybrid Inverter
    if ctx.fcst.plant_conf["inverter_is_hybrid"]:
        custom_inv = ctx.params["passed_data"]["custom_hybrid_inverter_id"]
        await ctx.rh.post_data(
            opt_res_latest["P_hybrid_inverter"],
            ctx.idx,
            custom_inv["entity_id"],
            "power",
            custom_inv["unit_of_measurement"],
            custom_inv["friendly_name"],
            type_var="power",
            **ctx.common_kwargs,
        )
        cols.append("P_hybrid_inverter")
    return cols


async def _publish_deferrable_loads(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish data for all deferrable loads."""
    cols = []
    custom_def = ctx.params["passed_data"]["custom_deferrable_forecast_id"]
    for k in range(ctx.opt.optim_conf["number_of_deferrable_loads"]):
        col_name = f"P_deferrable{k}"
        if col_name not in opt_res_latest.columns:
            ctx.logger.error(f"{col_name} was not found in results DataFrame.")
            continue
        await ctx.rh.post_data(
            opt_res_latest[col_name],
            ctx.idx,
            custom_def[k]["entity_id"],
            "power",
            custom_def[k]["unit_of_measurement"],
            custom_def[k]["friendly_name"],
            type_var="deferrable",
            **ctx.common_kwargs,
        )
        cols.append(col_name)
    return cols


async def _publish_thermal_variable(
    rh, opt_res_latest, idx, k, custom_ids, col_prefix, type_var, unit_type, kwargs
) -> str | None:
    """Helper to publish a single thermal variable if valid."""
    if custom_ids and k < len(custom_ids):
        col_name = f"{col_prefix}{k}"
        if col_name in opt_res_latest.columns:
            entity_conf = custom_ids[k]
            await rh.post_data(
                opt_res_latest[col_name],
                idx,
                entity_conf["entity_id"],
                unit_type,
                entity_conf["unit_of_measurement"],
                entity_conf["friendly_name"],
                type_var=type_var,
                **kwargs,
            )
            return col_name
    return None


async def _publish_thermal_loads(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish predicted temperature and heating demand for thermal loads."""
    cols = []
    if "custom_predicted_temperature_id" not in ctx.params["passed_data"]:
        return cols
    custom_temp = ctx.params["passed_data"]["custom_predicted_temperature_id"]
    custom_heat = ctx.params["passed_data"].get("custom_heating_demand_id")
    def_load_config = ctx.opt.optim_conf.get("def_load_config", [])
    if not isinstance(def_load_config, list):
        def_load_config = []
    for k in range(ctx.opt.optim_conf["number_of_deferrable_loads"]):
        if k >= len(def_load_config):
            continue
        load_cfg = def_load_config[k]
        if "thermal_config" not in load_cfg and "thermal_battery" not in load_cfg:
            continue
        col_t = await _publish_thermal_variable(
            ctx.rh,
            opt_res_latest,
            ctx.idx,
            k,
            custom_temp,
            "predicted_temp_heater",
            "temperature",
            "temperature",
            ctx.common_kwargs,
        )
        if col_t:
            cols.append(col_t)
        col_h = await _publish_thermal_variable(
            ctx.rh,
            opt_res_latest,
            ctx.idx,
            k,
            custom_heat,
            "heating_demand_heater",
            "energy",
            "energy",
            ctx.common_kwargs,
        )
        if col_h:
            cols.append(col_h)
    return cols


async def _publish_battery_data(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish Battery Power and SOC."""
    cols = []
    if not ctx.opt.optim_conf["set_use_battery"]:
        return cols
    if "P_batt" not in opt_res_latest.columns:
        ctx.logger.error("P_batt was not found in results DataFrame.")
        return cols
    # Power
    custom_batt = ctx.params["passed_data"]["custom_batt_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_batt"],
        ctx.idx,
        custom_batt["entity_id"],
        "power",
        custom_batt["unit_of_measurement"],
        custom_batt["friendly_name"],
        type_var="batt",
        **ctx.common_kwargs,
    )
    cols.append("P_batt")
    # SOC
    custom_soc = ctx.params["passed_data"]["custom_batt_soc_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["SOC_opt"] * 100,
        ctx.idx,
        custom_soc["entity_id"],
        "battery",
        custom_soc["unit_of_measurement"],
        custom_soc["friendly_name"],
        type_var="SOC",
        **ctx.common_kwargs,
    )
    cols.append("SOC_opt")
    return cols


async def _publish_grid_and_costs(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish Grid Power, Costs, and Optimization Status."""
    cols = []
    # Grid
    custom_grid = ctx.params["passed_data"]["custom_grid_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_grid"],
        ctx.idx,
        custom_grid["entity_id"],
        "power",
        custom_grid["unit_of_measurement"],
        custom_grid["friendly_name"],
        type_var="power",
        **ctx.common_kwargs,
    )
    cols.append("P_grid")
    # Cost Function
    custom_cost = ctx.params["passed_data"]["custom_cost_fun_id"]
    col_cost_fun = [i for i in opt_res_latest.columns if "cost_fun_" in i]
    await ctx.rh.post_data(
        opt_res_latest[col_cost_fun],
        ctx.idx,
        custom_cost["entity_id"],
        "monetary",
        custom_cost["unit_of_measurement"],
        custom_cost["friendly_name"],
        type_var="cost_fun",
        **ctx.common_kwargs,
    )
    # Optim Status
    custom_status = ctx.params["passed_data"]["custom_optim_status_id"]
    if "optim_status" not in opt_res_latest:
        opt_res_latest["optim_status"] = "Optimal"
        ctx.logger.warning("no optim_status in opt_res_latest")
    status_val = opt_res_latest["optim_status"]
    await ctx.rh.post_data(
        status_val,
        ctx.idx,
        custom_status["entity_id"],
        "",
        "",
        custom_status["friendly_name"],
        type_var="optim_status",
        **ctx.common_kwargs,
    )
    cols.append("optim_status")
    # Unit Costs
    for key, var_name in [
        ("custom_unit_load_cost_id", "unit_load_cost"),
        ("custom_unit_prod_price_id", "unit_prod_price"),
    ]:
        custom_id = ctx.params["passed_data"][key]
        await ctx.rh.post_data(
            opt_res_latest[var_name],
            ctx.idx,
            custom_id["entity_id"],
            "monetary",
            custom_id["unit_of_measurement"],
            custom_id["friendly_name"],
            type_var=var_name,
            **ctx.common_kwargs,
        )
        cols.append(var_name)
    return cols


async def publish_data(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    opt_res_latest: pd.DataFrame | None = None,
    entity_save: bool | None = False,
    dont_post: bool | None = False,
) -> pd.DataFrame:
    """
    Publish the data obtained from the optimization results.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: If True we will read data from optimization results in dayahead CSV file
    :type save_data_to_file: bool, optional
    :return: The output data of the optimization readed from a CSV file in the data folder
    :rtype: pd.DataFrame
    :param entity_save: Save built entities to data_path/entities
    :type entity_save: bool, optional
    :param dont_post: Do not post to Home Assistant. Works with entity_save
    :type dont_post: bool, optional

    """
    logger.info("Publishing data to HASS instance")
    # Parse Parameters
    params = _get_params(input_data_dict)
    # Check for Entity Publishing (Prefix mode)
    publish_prefix = params["passed_data"].get("publish_prefix", "")
    if not save_data_to_file and publish_prefix != "" and not dont_post:
        opt_res = await _publish_from_saved_entities(input_data_dict, logger, params)
        if opt_res is not None:
            return opt_res
    # Load Optimization Results (if not passed)
    if opt_res_latest is None:
        opt_res_latest = _load_opt_res_latest(input_data_dict, logger, save_data_to_file)
        if opt_res_latest is None:
            return None
    # Determine Closest Index
    idx_closest = _get_closest_index(input_data_dict["retrieve_hass_conf"], opt_res_latest.index)
    # Create Context
    common_kwargs = {
        "publish_prefix": publish_prefix,
        "save_entities": entity_save,
        "dont_post": dont_post,
    }
    ctx = PublishContext(
        input_data_dict=input_data_dict,
        params=params,
        idx=idx_closest,
        common_kwargs=common_kwargs,
        logger=logger,
    )
    # Publish Data Components
    cols_published = []
    cols_published.extend(await _publish_standard_forecasts(ctx, opt_res_latest))
    cols_published.extend(await _publish_deferrable_loads(ctx, opt_res_latest))
    cols_published.extend(await _publish_thermal_loads(ctx, opt_res_latest))
    cols_published.extend(await _publish_battery_data(ctx, opt_res_latest))
    cols_published.extend(await _publish_grid_and_costs(ctx, opt_res_latest))
    # Return Summary DataFrame
    opt_res = opt_res_latest[cols_published].loc[[opt_res_latest.index[idx_closest]]]
    return opt_res


async def continual_publish(
    input_data_dict: dict, entity_path: pathlib.Path, logger: logging.Logger
):
    """
    If continual_publish is true and a entity file saved in /data_path/entities, continually publish sensor on freq rate, updating entity current state value based on timestamp

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param entity_path: Path for entities folder in data_path
    :type entity_path: Path
    :param logger: The passed logger object
    :type logger: logging.Logger
    """
    logger.info("Continual publish thread service started")
    freq = input_data_dict["retrieve_hass_conf"].get(
        "optimization_time_step", pd.to_timedelta(1, "minutes")
    )
    while True:
        # Sleep for x seconds (using current time as a reference for time left)
        time_zone = input_data_dict["retrieve_hass_conf"]["time_zone"]
        timestamp_diff = freq.total_seconds() - (datetime.now(time_zone).timestamp() % 60)
        sleep_seconds = max(0.0, min(timestamp_diff, 60.0))
        await asyncio.sleep(sleep_seconds)
        # Delegate processing to helper function to reduce complexity
        freq = await _publish_and_update_freq(input_data_dict, entity_path, logger, freq)
    return False


async def _publish_and_update_freq(input_data_dict, entity_path, logger, current_freq):
    """
    Helper to process entity publishing and frequency updates.
    Returns the (potentially updated) frequency.
    """
    # Guard clause: if path doesn't exist, do nothing and return current freq
    if not os.path.exists(entity_path):
        return current_freq
    entity_path_contents = os.listdir(entity_path)
    # Guard clause: if directory is empty, do nothing
    if not entity_path_contents:
        return current_freq
    # Loop through all saved entity files
    for entity in entity_path_contents:
        if entity != default_metadata_json:
            await publish_json(
                entity,
                input_data_dict,
                entity_path,
                logger,
                "continual_publish",
            )
    # Retrieve entity metadata from file
    metadata_file = entity_path / default_metadata_json
    if os.path.isfile(metadata_file):
        async with aiofiles.open(metadata_file) as file:
            content = await file.read()
            metadata = orjson.loads(content)
            # Check if freq should be shorter
            if metadata.get("lowest_time_step") is not None:
                return pd.to_timedelta(metadata["lowest_time_step"], "minutes")
    return current_freq


async def publish_json(
    entity: dict,
    input_data_dict: dict,
    entity_path: pathlib.Path,
    logger: logging.Logger,
    reference: str | None = "",
):
    """
    Extract saved entity data from .json (in data_path/entities), build entity, post results to post_data

    :param entity: json file containing entity data
    :type entity: dict
    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param entity_path: Path for entities folder in data_path
    :type entity_path: Path
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param reference: String for identifying who ran the function
    :type reference: str, optional

    """
    # Retrieve entity metadata from file
    if os.path.isfile(entity_path / default_metadata_json):
        async with aiofiles.open(entity_path / default_metadata_json) as file:
            content = await file.read()
            metadata = orjson.loads(content)
    else:
        logger.error("unable to located metadata.json in:" + entity_path)
        return False
    # Round current timecode (now)
    now_precise = datetime.now(input_data_dict["retrieve_hass_conf"]["time_zone"]).replace(
        second=0, microsecond=0
    )
    # Retrieve entity data from file
    entity_data = pd.read_json(entity_path / entity, orient="index")
    # Remove ".json" from string for entity_id
    entity_id = entity.replace(".json", "")
    # Adjust Dataframe from received entity json file
    entity_data.columns = [metadata[entity_id]["name"]]
    entity_data.index.name = "timestamp"
    entity_data.index = pd.to_datetime(entity_data.index).tz_convert(
        input_data_dict["retrieve_hass_conf"]["time_zone"]
    )
    entity_data.index.freq = pd.to_timedelta(
        int(metadata[entity_id]["optimization_time_step"]), "minutes"
    )
    # Calculate the current state value
    if input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "nearest":
        idx_closest = entity_data.index.get_indexer([now_precise], method="nearest")[0]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "first":
        idx_closest = entity_data.index.get_indexer([now_precise], method="ffill")[0]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "last":
        idx_closest = entity_data.index.get_indexer([now_precise], method="bfill")[0]
    if idx_closest == -1:
        idx_closest = entity_data.index.get_indexer([now_precise], method="nearest")[0]
    # Call post data
    if reference == "continual_publish":
        logger.debug("Auto Published sensor:")
        logger_levels = "DEBUG"
    else:
        logger_levels = "INFO"
    # post/save entity
    await input_data_dict["rh"].post_data(
        data_df=entity_data[metadata[entity_id]["name"]],
        idx=idx_closest,
        entity_id=entity_id,
        device_class=dict.get(metadata[entity_id], "device_class"),
        unit_of_measurement=metadata[entity_id]["unit_of_measurement"],
        friendly_name=metadata[entity_id]["friendly_name"],
        type_var=metadata[entity_id].get("type_var", ""),
        save_entities=False,
        logger_levels=logger_levels,
    )
    return entity_data[metadata[entity_id]["name"]]


async def main():
    r"""Define the main command line entry function.

    This function may take several arguments as inputs. You can type `emhass --help` to see the list of options:

    - action: Set the desired action, options are: perfect-optim, dayahead-optim,
      naive-mpc-optim, publish-data, forecast-model-fit, forecast-model-predict, forecast-model-tune

    - config: Define path to the config.yaml file

    - costfun: Define the type of cost function, options are: profit, cost, self-consumption

    - log2file: Define if we should log to a file or not

    - params: Configuration parameters passed from data/options.json if using the add-on

    - runtimeparams: Pass runtime optimization parameters as dictionnary

    - debug: Use True for testing purposes

    """
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        help="Set the desired action, options are: perfect-optim, dayahead-optim,\
        naive-mpc-optim, publish-data, forecast-model-fit, forecast-model-predict, forecast-model-tune",
    )
    parser.add_argument(
        "--config", type=str, help="Define path to the config.json/defaults.json file"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="String of configuration parameters passed",
    )
    parser.add_argument("--data", type=str, help="Define path to the Data files (.csv & .pkl)")
    parser.add_argument("--root", type=str, help="Define path emhass root")
    parser.add_argument(
        "--costfun",
        type=str,
        default="profit",
        help="Define the type of cost function, options are: profit, cost, self-consumption",
    )
    parser.add_argument(
        "--log2file",
        type=bool,
        default=False,
        help="Define if we should log to a file or not",
    )
    parser.add_argument(
        "--secrets",
        type=str,
        default=None,
        help="Define secret parameter file (secrets_emhass.yaml) path",
    )
    parser.add_argument(
        "--runtimeparams",
        type=str,
        default=None,
        help="Pass runtime optimization parameters as dictionnary",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Use True for testing purposes",
    )
    args = parser.parse_args()

    # The path to the configuration files
    if args.config is not None:
        config_path = pathlib.Path(args.config)
    else:
        config_path = pathlib.Path(str(utils.get_root(__file__, num_parent=3) / "config.json"))
    if args.data is not None:
        data_path = pathlib.Path(args.data)
    else:
        data_path = config_path.parent / "data/"
    if args.root is not None:
        root_path = pathlib.Path(args.root)
    else:
        root_path = utils.get_root(__file__, num_parent=1)
    if args.secrets is not None:
        secrets_path = pathlib.Path(args.secrets)
    else:
        secrets_path = pathlib.Path(config_path.parent / "secrets_emhass.yaml")

    associations_path = root_path / "data/associations.csv"
    defaults_path = root_path / "data/config_defaults.json"

    emhass_conf = {}
    emhass_conf["config_path"] = config_path
    emhass_conf["data_path"] = data_path
    emhass_conf["root_path"] = root_path
    emhass_conf["associations_path"] = associations_path
    emhass_conf["defaults_path"] = defaults_path
    # create logger
    logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=bool(args.log2file))

    # Check paths
    logger.debug("config path: " + str(config_path))
    logger.debug("data path: " + str(data_path))
    logger.debug("root path: " + str(root_path))
    if not associations_path.exists():
        logger.error("Could not find associations.csv file in: " + str(associations_path))
        logger.error("Try setting config file path with --associations")
        return False
    if not config_path.exists():
        logger.warning("Could not find config.json file in: " + str(config_path))
        logger.warning("Try setting config file path with --config")
    if not secrets_path.exists():
        logger.warning("Could not find secrets file in: " + str(secrets_path))
        logger.warning("Try setting secrets file path with --secrets")
    if not os.path.isdir(data_path):
        logger.error("Could not find data folder in: " + str(data_path))
        logger.error("Try setting data path with --data")
        return False
    if not os.path.isdir(root_path):
        logger.error("Could not find emhass/src folder in: " + str(root_path))
        logger.error("Try setting emhass root path with --root")
        return False

    # Additional argument
    try:
        parser.add_argument(
            "--version",
            action="version",
            version="%(prog)s " + version("emhass"),
        )
        args = parser.parse_args()
    except Exception:
        logger.info(
            "Version not found for emhass package. Or importlib exited with PackageNotFoundError.",
        )

    # Setup config
    config = {}
    # Check if passed config file is yaml of json, build config accordingly
    if config_path.exists():
        # Safe: Use pathlib's suffix instead of regex to avoid ReDoS
        file_extension = config_path.suffix.lstrip(".").lower()

        if file_extension:
            match file_extension:
                case "json":
                    config = await utils.build_config(
                        emhass_conf, logger, defaults_path, config_path
                    )
                case "yaml" | "yml":
                    config = await utils.build_config(
                        emhass_conf, logger, defaults_path, config_path=config_path
                    )
                case _:
                    logger.warning(
                        f"Unsupported config file format: .{file_extension}, building parameters with only defaults"
                    )
                    config = await utils.build_config(emhass_conf, logger, defaults_path)
        else:
            logger.warning("Config file has no extension, building parameters with only defaults")
            config = await utils.build_config(emhass_conf, logger, defaults_path)
    else:
        # If unable to find config file, use only defaults_config.json
        logger.warning("Unable to obtain config.json file, building parameters with only defaults")
        config = await utils.build_config(emhass_conf, logger, defaults_path)
    if type(config) is bool and not config:
        raise Exception("Failed to find default config")

    # Obtain secrets from secrets_emhass.yaml?
    params_secrets = {}
    emhass_conf, built_secrets = await utils.build_secrets(
        emhass_conf, logger, secrets_path=secrets_path
    )
    params_secrets.update(built_secrets)

    # Build params
    params = await utils.build_params(emhass_conf, params_secrets, config, logger)
    if type(params) is bool:
        raise Exception("A error has occurred while building parameters")
    # Add any passed params from args to params
    if args.params:
        params.update(orjson.loads(args.params))

    input_data_dict = await set_input_data_dict(
        emhass_conf,
        args.costfun,
        orjson.dumps(params).decode("utf-8"),
        args.runtimeparams,
        args.action,
        logger,
        args.debug,
    )
    if type(input_data_dict) is bool:
        raise Exception("A error has occurred while creating action objects")

    # Perform selected action
    if args.action == "perfect-optim":
        opt_res = await perfect_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "dayahead-optim":
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "naive-mpc-optim":
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "forecast-model-fit":
        df_fit_pred, df_fit_pred_backtest, mlf = await forecast_model_fit(
            input_data_dict, logger, debug=args.debug
        )
        opt_res = None
    elif args.action == "forecast-model-predict":
        if args.debug:
            _, _, mlf = await forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred = await forecast_model_predict(input_data_dict, logger, debug=args.debug, mlf=mlf)
        opt_res = None
    elif args.action == "forecast-model-tune":
        if args.debug:
            _, _, mlf = await forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred_optim, mlf = await forecast_model_tune(
            input_data_dict, logger, debug=args.debug, mlf=mlf
        )
        opt_res = None
    elif args.action == "regressor-model-fit":
        mlr = await regressor_model_fit(input_data_dict, logger, debug=args.debug)
        opt_res = None
    elif args.action == "regressor-model-predict":
        if args.debug:
            mlr = await regressor_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlr = None
        prediction = await regressor_model_predict(
            input_data_dict, logger, debug=args.debug, mlr=mlr
        )
        opt_res = None
    elif args.action == "export-influxdb-to-csv":
        success = await export_influxdb_to_csv(input_data_dict, logger)
        opt_res = None
    elif args.action == "publish-data":
        opt_res = await publish_data(input_data_dict, logger)
    else:
        logger.error("The passed action argument is not valid")
        logger.error(
            "Try setting --action: perfect-optim, dayahead-optim, naive-mpc-optim, forecast-model-fit, forecast-model-predict, forecast-model-tune, export-influxdb-to-csv or publish-data"
        )
        opt_res = None
    logger.info(opt_res)
    # Flush the logger
    ch.close()
    logger.removeHandler(ch)
    if (
        args.action == "perfect-optim"
        or args.action == "dayahead-optim"
        or args.action == "naive-mpc-optim"
        or args.action == "publish-data"
    ):
        return opt_res
    elif args.action == "forecast-model-fit":
        return df_fit_pred, df_fit_pred_backtest, mlf
    elif args.action == "forecast-model-predict":
        return df_pred
    elif args.action == "regressor-model-fit":
        return mlr
    elif args.action == "regressor-model-predict":
        return prediction
    elif args.action == "export-influxdb-to-csv":
        return success
    elif args.action == "forecast-model-tune":
        return df_pred_optim, mlf
    else:
        return opt_res


def main_sync():
    """Sync wrapper for async main function - used as CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
