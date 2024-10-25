#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
from typing import Tuple, Optional
from datetime import datetime, timedelta, timezone
import logging
import pathlib
import json
import copy
import numpy as np
import pandas as pd
from requests import get
import yaml
import pytz
import ast

import plotly.express as px

pd.options.plotting.backend = "plotly"

from emhass.machine_learning_forecaster import MLForecaster

def get_root(file: str, num_parent: Optional[int] = 3) -> str:
    """
    Get the root absolute path of the working directory.

    :param file: The passed file path with __file__
    :return: The root path
    :param num_parent: The number of parents levels up to desired root folder
    :type num_parent: int, optional
    :rtype: str

    """
    if num_parent == 3:
        root = pathlib.Path(file).resolve().parent.parent.parent
    elif num_parent == 2:
        root = pathlib.Path(file).resolve().parent.parent
    elif num_parent == 1:
        root = pathlib.Path(file).resolve().parent
    else:
        raise ValueError("num_parent value not valid, must be between 1 and 3")
    return root


def get_logger(fun_name: str, emhass_conf: dict, save_to_file: Optional[bool] = True,
               logging_level: Optional[str] = "DEBUG") -> Tuple[logging.Logger, logging.StreamHandler]:
    """
    Create a simple logger object.

    :param fun_name: The Python function object name where the logger will be used
    :type fun_name: str
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param save_to_file: Write log to a file, defaults to True
    :type save_to_file: bool, optional
    :return: The logger object and the handler
    :rtype: object

    """
    # create logger object
    logger = logging.getLogger(fun_name)
    logger.propagate = True
    logger.fileSetting = save_to_file
    if save_to_file:
        if  os.path.isdir(emhass_conf['data_path']):
            ch = logging.FileHandler(emhass_conf['data_path'] / 'logger_emhass.log')
        else:
            raise Exception("Unable to access data_path: "+emhass_conf['data_path'])  
    else:
        ch = logging.StreamHandler()
    if logging_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        logger.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        logger.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, ch


def get_forecast_dates(freq: int, delta_forecast: int, time_zone: datetime.tzinfo, timedelta_days: Optional[int] = 0
                       ) -> pd.core.indexes.datetimes.DatetimeIndex:
    """
    Get the date_range list of the needed future dates using the delta_forecast parameter.

    :param freq: Optimization time step.
    :type freq: int
    :param delta_forecast: Number of days to forecast in the future to be used for the optimization.
    :type delta_forecast: int
    :param timedelta_days: Number of truncated days needed for each optimization iteration, defaults to 0
    :type timedelta_days: Optional[int], optional
    :return: A list of future forecast dates.
    :rtype: pd.core.indexes.datetimes.DatetimeIndex

    """
    freq = pd.to_timedelta(freq, "minutes")
    start_forecast = pd.Timestamp(datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)
    end_forecast = (start_forecast + pd.Timedelta(days=delta_forecast)).replace(microsecond=0)
    forecast_dates = pd.date_range(start=start_forecast, 
        end=end_forecast+timedelta(days=timedelta_days)-freq, 
        freq=freq, tz=time_zone).tz_convert('utc').round(freq, ambiguous='infer', nonexistent='shift_forward').tz_convert(time_zone)
    return forecast_dates


def treat_runtimeparams(runtimeparams: str, params: str, retrieve_hass_conf: dict, optim_conf: dict,
                        plant_conf: dict, set_type: str, logger: logging.Logger
                        ) -> Tuple[str, dict]:
    """
    Treat the passed optimization runtime parameters.

    :param runtimeparams: Json string containing the runtime parameters dict.
    :type runtimeparams: str
    :param params: Built configuration parameters
    :type params: str
    :param retrieve_hass_conf: Container for data retrieving parameters.
    :type retrieve_hass_conf: dict
    :param optim_conf: Container for optimization parameters.
    :type optim_conf: dict
    :param plant_conf: Container for technical plant parameters.
    :type plant_conf: dict
    :param set_type: The type of action to be performed.
    :type set_type: str
    :param logger: The logger object.
    :type logger: logging.Logger
    :return: Returning the params and optimization parameter container.
    :rtype: Tuple[str, dict]

    """
    # check if passed params is a dict
    if (params != None) and (params != "null"):
        if type(params) is str:
            params = json.loads(params)
    else:
        params = {}

    # Some default data needed
    custom_deferrable_forecast_id = []
    custom_predicted_temperature_id = []
    for k in range(optim_conf['number_of_deferrable_loads']):
        custom_deferrable_forecast_id.append(
            {
                "entity_id": "sensor.p_deferrable{}".format(k),
                "unit_of_measurement": "W",
                "friendly_name": "Deferrable Load {}".format(k),
            }
        )
        custom_predicted_temperature_id.append(
            {
                "entity_id": "sensor.temp_predicted{}".format(k),
                "unit_of_measurement": "°C",
                "friendly_name": "Predicted temperature {}".format(k),
            }
        )
    default_passed_dict = {
        "custom_pv_forecast_id": {
            "entity_id": "sensor.p_pv_forecast",
            "unit_of_measurement": "W",
            "friendly_name": "PV Power Forecast",
        },
        "custom_load_forecast_id": {
            "entity_id": "sensor.p_load_forecast",
            "unit_of_measurement": "W",
            "friendly_name": "Load Power Forecast",
        },
        "custom_pv_curtailment_id": {
            "entity_id": "sensor.p_pv_curtailment",
            "unit_of_measurement": "W",
            "friendly_name": "PV Power Curtailment",
        },
        "custom_hybrid_inverter_id": {
            "entity_id": "sensor.p_hybrid_inverter",
            "unit_of_measurement": "W",
            "friendly_name": "PV Hybrid Inverter",
        },
        "custom_batt_forecast_id": {
            "entity_id": "sensor.p_batt_forecast",
            "unit_of_measurement": "W",
            "friendly_name": "Battery Power Forecast",
        },
        "custom_batt_soc_forecast_id": {
            "entity_id": "sensor.soc_batt_forecast",
            "unit_of_measurement": "%",
            "friendly_name": "Battery SOC Forecast",
        },
        "custom_grid_forecast_id": {
            "entity_id": "sensor.p_grid_forecast",
            "unit_of_measurement": "W",
            "friendly_name": "Grid Power Forecast",
        },
        "custom_cost_fun_id": {
            "entity_id": "sensor.total_cost_fun_value",
            "unit_of_measurement": "",
            "friendly_name": "Total cost function value",
        },
        "custom_optim_status_id": {
            "entity_id": "sensor.optim_status",
            "unit_of_measurement": "",
            "friendly_name": "EMHASS optimization status",
        },
        "custom_unit_load_cost_id": {
            "entity_id": "sensor.unit_load_cost",
            "unit_of_measurement": "€/kWh",
            "friendly_name": "Unit Load Cost",
        },
        "custom_unit_prod_price_id": {
            "entity_id": "sensor.unit_prod_price",
            "unit_of_measurement": "€/kWh",
            "friendly_name": "Unit Prod Price",
        },
        "custom_deferrable_forecast_id": custom_deferrable_forecast_id,
        "custom_predicted_temperature_id": custom_predicted_temperature_id,
        "publish_prefix": "",
    }
    if "passed_data" in params.keys():
        for key, value in default_passed_dict.items():
            params["passed_data"][key] = value
    else:
        params["passed_data"] = default_passed_dict

    if runtimeparams is not None:
        if type(runtimeparams) is str:
            runtimeparams = json.loads(runtimeparams)
       # Format required date/time parameters 
        optimization_time_step = int(
            retrieve_hass_conf['optimization_time_step'].seconds / 60.0)
        delta_forecast = int(optim_conf['delta_forecast_daily'].days)
        time_zone = retrieve_hass_conf["time_zone"]
        forecast_dates = get_forecast_dates(
            optimization_time_step, delta_forecast, time_zone)
       
       # regressor-model-fit
        if set_type == "regressor-model-fit":
            if "csv_file" in runtimeparams:
                csv_file = runtimeparams["csv_file"]
                params["passed_data"]["csv_file"] = csv_file
            if "features" in runtimeparams:
                features = runtimeparams["features"]
                params["passed_data"]["features"] = features
            if "target" in runtimeparams:
                target = runtimeparams["target"]
                params["passed_data"]["target"] = target
            if "timestamp" not in runtimeparams:
                params["passed_data"]["timestamp"] = None
            else:
                timestamp = runtimeparams["timestamp"]
                params["passed_data"]["timestamp"] = timestamp
            if "date_features" not in runtimeparams:
                params["passed_data"]["date_features"] = []
            else:
                date_features = runtimeparams["date_features"]
                params["passed_data"]["date_features"] = date_features
        
        # regressor-model-predict
        if set_type == "regressor-model-predict":
            if "new_values" in runtimeparams:
                new_values = runtimeparams["new_values"]
                params["passed_data"]["new_values"] = new_values
            if "csv_file" in runtimeparams:
                csv_file = runtimeparams["csv_file"]
                params["passed_data"]["csv_file"] = csv_file
            if "features" in runtimeparams:
                features = runtimeparams["features"]
                params["passed_data"]["features"] = features
            if "target" in runtimeparams:
                target = runtimeparams["target"]
                params["passed_data"]["target"] = target
        
        # Treating special data passed for MPC control case
        if set_type == "naive-mpc-optim":
            if "prediction_horizon" not in runtimeparams.keys():
                prediction_horizon = 10  # 10 time steps by default
            else:
                prediction_horizon = runtimeparams["prediction_horizon"]
            params["passed_data"]["prediction_horizon"] = prediction_horizon
            if "soc_init" not in runtimeparams.keys():
                soc_init = plant_conf['battery_target_state_of_charge']
            else:
                soc_init = runtimeparams["soc_init"]
            params["passed_data"]["soc_init"] = soc_init
            if "soc_final" not in runtimeparams.keys():
                soc_final = plant_conf['battery_target_state_of_charge']
            else:
                soc_final = runtimeparams["soc_final"]
            params["passed_data"]["soc_final"] = soc_final
            if 'operating_hours_of_each_deferrable_load' not in runtimeparams.keys():
                def_total_hours = optim_conf['operating_hours_of_each_deferrable_load']
            else:
                def_total_hours = runtimeparams['operating_hours_of_each_deferrable_load']
            params["passed_data"]['operating_hours_of_each_deferrable_load'] = def_total_hours
            if 'start_timesteps_of_each_deferrable_load' in runtimeparams.keys():
                def_start_timestep = runtimeparams['start_timesteps_of_each_deferrable_load']
            else:
                def_start_timestep = runtimeparams.get(
                    'def_start_timestep', optim_conf['start_timesteps_of_each_deferrable_load'])
            params["passed_data"]['start_timesteps_of_each_deferrable_load'] = def_start_timestep
            if 'end_timesteps_of_each_deferrable_load' in runtimeparams.keys():
                def_end_timestep = runtimeparams['end_timesteps_of_each_deferrable_load']
            else:
                def_end_timestep = runtimeparams.get(
                    'def_end_timestep', optim_conf['end_timesteps_of_each_deferrable_load'])
            params["passed_data"]['end_timesteps_of_each_deferrable_load'] = def_end_timestep
            forecast_dates = copy.deepcopy(forecast_dates)[
                0:prediction_horizon]
        else:
            params["passed_data"]["prediction_horizon"] = None
            params["passed_data"]["soc_init"] = None
            params["passed_data"]["soc_final"] = None
            params["passed_data"]['operating_hours_of_each_deferrable_load'] = None
            params["passed_data"]['start_timesteps_of_each_deferrable_load'] = None
            params["passed_data"]['end_timesteps_of_each_deferrable_load'] = None
        # Treat passed forecast data lists
        list_forecast_key = ['pv_power_forecast', 'load_power_forecast',
                             'load_cost_forecast', 'prod_price_forecast', 'outdoor_temperature_forecast']
        forecast_methods = ['weather_forecast_method', 'load_forecast_method', 'load_cost_forecast_method',
                            'production_price_forecast_method', 'outdoor_temperature_forecast_method']
        
        # Loop forecasts, check if value is a list and greater than or equal to forecast_dates
        for method, forecast_key in enumerate(list_forecast_key):
            if forecast_key in runtimeparams.keys():
                if type(runtimeparams[forecast_key]) == list and len(runtimeparams[forecast_key]) >= len(forecast_dates):
                    params['passed_data'][forecast_key] = runtimeparams[forecast_key]
                    optim_conf[forecast_methods[method]] = 'list'
                else:
                    logger.error(
                        f"ERROR: The passed data is either not a list or the length is not correct, length should be {str(len(forecast_dates))}")
                    logger.error(
                        f"Passed type is {str(type(runtimeparams[forecast_key]))} and length is {str(len(runtimeparams[forecast_key]))}")
                # Check if string contains list, if so extract
                if type(runtimeparams[forecast_key]) == str:
                    if type(ast.literal_eval(runtimeparams[forecast_key])) == list:
                        runtimeparams[forecast_key] = ast.literal_eval(runtimeparams[forecast_key])
                list_non_digits = [x for x in runtimeparams[forecast_key] if not (
                    isinstance(x, int) or isinstance(x, float))]
                if len(list_non_digits) > 0:
                    logger.warning(
                        f"There are non numeric values on the passed data for {forecast_key}, check for missing values (nans, null, etc)")
                    for x in list_non_digits:
                        logger.warning(
                            f"This value in {forecast_key} was detected as non digits: {str(x)}")
            else:
                params['passed_data'][forecast_key] = None
       
        # Treat passed data for forecast model fit/predict/tune at runtime
        if 'historic_days_to_retrieve' in runtimeparams.keys():
            days_to_retrieve = runtimeparams['historic_days_to_retrieve']
        else:
            days_to_retrieve = runtimeparams.get('days_to_retrieve', 9)
        params["passed_data"]['historic_days_to_retrieve'] = days_to_retrieve
        if "model_type" not in runtimeparams.keys():
            model_type = "load_forecast"
        else:
            model_type = runtimeparams["model_type"]
        params["passed_data"]["model_type"] = model_type
        if "var_model" not in runtimeparams.keys():
            var_model = "sensor.power_load_no_var_loads"
        else:
            var_model = runtimeparams["var_model"]
        params["passed_data"]["var_model"] = var_model
        if "sklearn_model" not in runtimeparams.keys():
            sklearn_model = "KNeighborsRegressor"
        else:
            sklearn_model = runtimeparams["sklearn_model"]
        params["passed_data"]["sklearn_model"] = sklearn_model
        if "regression_model" not in runtimeparams.keys():
            regression_model = "AdaBoostRegression"
        else:
            regression_model = runtimeparams["regression_model"]
        params["passed_data"]["regression_model"] = regression_model
        if "num_lags" not in runtimeparams.keys():
            num_lags = 48
        else:
            num_lags = runtimeparams["num_lags"]
        params["passed_data"]["num_lags"] = num_lags
        if "split_date_delta" not in runtimeparams.keys():
            split_date_delta = "48h"
        else:
            split_date_delta = runtimeparams["split_date_delta"]
        params["passed_data"]["split_date_delta"] = split_date_delta
        if "perform_backtest" not in runtimeparams.keys():
            perform_backtest = False
        else:
            perform_backtest = ast.literal_eval(
                str(runtimeparams["perform_backtest"]).capitalize())
        params["passed_data"]["perform_backtest"] = perform_backtest
        if "model_predict_publish" not in runtimeparams.keys():
            model_predict_publish = False
        else:
            model_predict_publish = ast.literal_eval(
                str(runtimeparams["model_predict_publish"]).capitalize())
        params["passed_data"]["model_predict_publish"] = model_predict_publish
        if "model_predict_entity_id" not in runtimeparams.keys():
            model_predict_entity_id = "sensor.p_load_forecast_custom_model"
        else:
            model_predict_entity_id = runtimeparams["model_predict_entity_id"]
        params["passed_data"]["model_predict_entity_id"] = model_predict_entity_id
        if "model_predict_unit_of_measurement" not in runtimeparams.keys():
            model_predict_unit_of_measurement = "W"
        else:
            model_predict_unit_of_measurement = runtimeparams["model_predict_unit_of_measurement"]
        params["passed_data"]["model_predict_unit_of_measurement"] = model_predict_unit_of_measurement
        if "model_predict_friendly_name" not in runtimeparams.keys():
            model_predict_friendly_name = "Load Power Forecast custom ML model"
        else:
            model_predict_friendly_name = runtimeparams["model_predict_friendly_name"]
        params["passed_data"]["model_predict_friendly_name"] = model_predict_friendly_name
        if "mlr_predict_entity_id" not in runtimeparams.keys():
            mlr_predict_entity_id = "sensor.mlr_predict"
        else:
            mlr_predict_entity_id = runtimeparams["mlr_predict_entity_id"]
        params["passed_data"]["mlr_predict_entity_id"] = mlr_predict_entity_id
        if "mlr_predict_unit_of_measurement" not in runtimeparams.keys():
            mlr_predict_unit_of_measurement = None
        else:
            mlr_predict_unit_of_measurement = runtimeparams["mlr_predict_unit_of_measurement"]
        params["passed_data"]["mlr_predict_unit_of_measurement"] = mlr_predict_unit_of_measurement
        if "mlr_predict_friendly_name" not in runtimeparams.keys():
            mlr_predict_friendly_name = "mlr predictor"
        else:
            mlr_predict_friendly_name = runtimeparams["mlr_predict_friendly_name"]
        params["passed_data"]["mlr_predict_friendly_name"] = mlr_predict_friendly_name
        
        # Treat passed data for other parameters
        if "alpha" not in runtimeparams.keys():
            alpha = 0.5
        else:
            alpha = runtimeparams["alpha"]
        params["passed_data"]["alpha"] = alpha
        if "beta" not in runtimeparams.keys():
            beta = 0.5
        else:
            beta = runtimeparams["beta"]
        params["passed_data"]["beta"] = beta
        # Param to save forecast cache (i.e. Solcast)
        if "weather_forecast_cache" not in runtimeparams.keys():
            weather_forecast_cache = False
        else:
            weather_forecast_cache = runtimeparams["weather_forecast_cache"]
        params["passed_data"]["weather_forecast_cache"] = weather_forecast_cache
        # Param to make sure optimization only uses cached data. (else produce error)
        if "weather_forecast_cache_only" not in runtimeparams.keys():
            weather_forecast_cache_only = False
        else:
            weather_forecast_cache_only = runtimeparams["weather_forecast_cache_only"]
        params["passed_data"]["weather_forecast_cache_only"] = weather_forecast_cache_only
        # A condition to manually save entity data under data_path/entities after optimization
        if "entity_save" not in runtimeparams.keys():
            entity_save = ""
        else:
            entity_save = runtimeparams["entity_save"]
        params["passed_data"]["entity_save"] = entity_save
        # A condition to put a prefix on all published data, or check for saved data under prefix name
        if "publish_prefix" not in runtimeparams.keys():
            publish_prefix = ""
        else:
            publish_prefix = runtimeparams["publish_prefix"]
        params["passed_data"]["publish_prefix"] = publish_prefix

        # Treat optimization (optim_conf) configuration parameters passed at runtime
        if 'number_of_deferrable_loads' in runtimeparams.keys():
            optim_conf['number_of_deferrable_loads'] = runtimeparams['number_of_deferrable_loads']
        if 'num_def_loads' in runtimeparams.keys():
            optim_conf['number_of_deferrable_loads'] = runtimeparams['num_def_loads']
        if 'nominal_power_of_deferrable_loads' in runtimeparams.keys():
            optim_conf['nominal_power_of_deferrable_loads'] = runtimeparams['nominal_power_of_deferrable_loads']
        if 'P_deferrable_nom' in runtimeparams.keys():
            optim_conf['nominal_power_of_deferrable_loads'] = runtimeparams['P_deferrable_nom']
        if 'operating_hours_of_each_deferrable_load' in runtimeparams.keys():
            optim_conf['operating_hours_of_each_deferrable_load'] = runtimeparams['operating_hours_of_each_deferrable_load']
        if 'def_total_hours' in runtimeparams.keys():
            optim_conf['operating_hours_of_each_deferrable_load'] = runtimeparams['def_total_hours']
        if 'start_timesteps_of_each_deferrable_load' in runtimeparams.keys():
            optim_conf['start_timesteps_of_each_deferrable_load'] = runtimeparams['start_timesteps_of_each_deferrable_load']
        if 'end_timesteps_of_each_deferrable_load' in runtimeparams.keys():
            optim_conf['end_timesteps_of_each_deferrable_load'] = runtimeparams['end_timesteps_of_each_deferrable_load']
        if "def_current_state" in runtimeparams.keys():
            optim_conf["def_current_state"] = [
                bool(s) for s in runtimeparams["def_current_state"]]
        if 'treat_deferrable_load_as_semi_cont' in runtimeparams.keys():
            optim_conf['treat_deferrable_load_as_semi_cont'] = [
                ast.literal_eval(str(k).capitalize())
                for k in runtimeparams['treat_deferrable_load_as_semi_cont']
            ]
        if 'treat_def_as_semi_cont' in runtimeparams.keys():
            optim_conf['treat_deferrable_load_as_semi_cont'] = [
                ast.literal_eval(str(k).capitalize())
                for k in runtimeparams['treat_def_as_semi_cont']
            ]
        if 'set_deferrable_load_single_constant' in runtimeparams.keys():
            optim_conf['set_deferrable_load_single_constant'] = [
                ast.literal_eval(str(k).capitalize()) for k in runtimeparams['set_deferrable_load_single_constant']
            ]
        if 'set_def_constant' in runtimeparams.keys():
            optim_conf['set_deferrable_load_single_constant'] = [
                ast.literal_eval(str(k).capitalize()) for k in runtimeparams['set_def_constant']
            ]
        if 'set_deferrable_startup_penalty' in runtimeparams.keys():
            optim_conf['set_deferrable_startup_penalty'] = [
                ast.literal_eval(str(k).capitalize()) for k in runtimeparams['set_deferrable_startup_penalty']
            ]
        if 'def_start_penalty' in runtimeparams.keys():
            optim_conf['set_deferrable_startup_penalty'] = [
                ast.literal_eval(str(k).capitalize()) for k in runtimeparams['def_start_penalty']
            ]
        if 'def_load_config' in runtimeparams:
            optim_conf["def_load_config"] = runtimeparams['def_load_config']
        if 'weight_battery_discharge' in runtimeparams.keys():
            optim_conf['weight_battery_discharge'] = runtimeparams[
                'weight_battery_discharge'
            ]
        if 'weight_battery_charge' in runtimeparams.keys():
            optim_conf['weight_battery_charge'] = runtimeparams['weight_battery_charge']

        # Treat retrieve data from Home Assistant (retrieve_hass_conf) configuration parameters passed at runtime
        if 'optimization_time_step' in runtimeparams.keys():
            retrieve_hass_conf['optimization_time_step'] = pd.to_timedelta(
                runtimeparams['optimization_time_step'], "minutes")
        if 'continual_publish' in runtimeparams.keys():
            retrieve_hass_conf['continual_publish'] = bool(
                runtimeparams['continual_publish'])
        if "solcast_api_key" in runtimeparams.keys():
            retrieve_hass_conf["solcast_api_key"] = runtimeparams["solcast_api_key"]
            optim_conf['weather_forecast_method'] = "solcast"
        if "solcast_rooftop_id" in runtimeparams.keys():
            retrieve_hass_conf["solcast_rooftop_id"] = runtimeparams[
                "solcast_rooftop_id"
            ]
            optim_conf['weather_forecast_method'] = "solcast"
        if "solar_forecast_kwp" in runtimeparams.keys():
            retrieve_hass_conf["solar_forecast_kwp"] = runtimeparams[
                "solar_forecast_kwp"
            ]
            optim_conf['weather_forecast_method'] = "solar.forecast"

        # Treat system model parameters (plant) configuration parameters passed at runtime
        if 'battery_minimum_state_of_charge' in runtimeparams.keys() or 'SOCmin' in runtimeparams.keys():
            plant_conf['battery_minimum_state_of_charge'] = runtimeparams.get(
                'battery_minimum_state_of_charge', runtimeparams.get('SOCmin'))
        if 'battery_maximum_state_of_charge' in runtimeparams.keys() or 'SOCmax' in runtimeparams.keys():
            plant_conf['battery_maximum_state_of_charge'] = runtimeparams.get(
                'battery_maximum_state_of_charge', runtimeparams.get('SOCmax'))
        if 'battery_target_state_of_charge' in runtimeparams.keys() or 'SOCtarget' in runtimeparams.keys():
            plant_conf['battery_target_state_of_charge'] = runtimeparams.get(
                'battery_target_state_of_charge', runtimeparams.get('SOCtarget'))
        if 'battery_discharge_power_max' in runtimeparams.keys() or 'Pd_max' in runtimeparams.keys():
            plant_conf['battery_discharge_power_max'] = runtimeparams.get(
                'battery_discharge_power_max', runtimeparams.get('Pd_max'))
        if 'battery_charge_power_max' in runtimeparams.keys() or 'Pc_max' in runtimeparams.keys():
            plant_conf['battery_charge_power_max'] = runtimeparams.get(
                'battery_charge_power_max', runtimeparams.get('Pc_max'))
        
        # Treat custom entities id's and friendly names for variables
        if "custom_pv_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_pv_forecast_id"] = runtimeparams[
                "custom_pv_forecast_id"
            ]
        if "custom_load_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_load_forecast_id"] = runtimeparams[
                "custom_load_forecast_id"
            ]
        if "custom_pv_curtailment_id" in runtimeparams.keys():
            params["passed_data"]["custom_pv_curtailment_id"] = runtimeparams[
                "custom_pv_curtailment_id"
            ]
        if "custom_hybrid_inverter_id" in runtimeparams.keys():
            params["passed_data"]["custom_hybrid_inverter_id"] = runtimeparams[
                "custom_hybrid_inverter_id"
            ]
        if "custom_batt_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_batt_forecast_id"] = runtimeparams[
                "custom_batt_forecast_id"
            ]
        if "custom_batt_soc_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_batt_soc_forecast_id"] = runtimeparams[
                "custom_batt_soc_forecast_id"
            ]
        if "custom_grid_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_grid_forecast_id"] = runtimeparams[
                "custom_grid_forecast_id"
            ]
        if "custom_cost_fun_id" in runtimeparams.keys():
            params["passed_data"]["custom_cost_fun_id"] = runtimeparams[
                "custom_cost_fun_id"
            ]
        if "custom_optim_status_id" in runtimeparams.keys():
            params["passed_data"]["custom_optim_status_id"] = runtimeparams[
                "custom_optim_status_id"
            ]
        if "custom_unit_load_cost_id" in runtimeparams.keys():
            params["passed_data"]["custom_unit_load_cost_id"] = runtimeparams[
                "custom_unit_load_cost_id"
            ]
        if "custom_unit_prod_price_id" in runtimeparams.keys():
            params["passed_data"]["custom_unit_prod_price_id"] = runtimeparams[
                "custom_unit_prod_price_id"
            ]
        if "custom_deferrable_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_deferrable_forecast_id"] = runtimeparams[
                "custom_deferrable_forecast_id"
            ]
        if "custom_predicted_temperature_id" in runtimeparams.keys():
            params["passed_data"]["custom_predicted_temperature_id"] = runtimeparams[
                "custom_predicted_temperature_id"
            ]
       
    # Serialize the final params
    params = json.dumps(params, default=str)
    return params, retrieve_hass_conf, optim_conf, plant_conf


def get_yaml_parse(params: str, logger: logging.Logger) -> Tuple[dict, dict, dict]:
    """
    Perform parsing of the params into the configuration catagories 
    
    :param params: Built configuration parameters
    :type params: str
    :param logger: The logger object
    :type logger: logging.Logger
    :return: A tuple with the dictionaries containing the parsed data
    :rtype: tuple(dict)

    """
    if params:
        if type(params) is str:
            input_conf = json.loads(params)
        else:
            input_conf = params
    else:
        input_conf = {}
        logger.error("No params have been detected for get_yaml_parse")
        return False, False, False

    optim_conf = input_conf.get("optim_conf", {})

    retrieve_hass_conf = input_conf.get("retrieve_hass_conf", {})

    plant_conf = input_conf.get("plant_conf", {})

    # Format time parameters
    if optim_conf.get('delta_forecast_daily',None) is not None:
        optim_conf['delta_forecast_daily'] = pd.Timedelta(days=optim_conf['delta_forecast_daily'])
    if retrieve_hass_conf.get('optimization_time_step',None) is not None:
        retrieve_hass_conf['optimization_time_step'] = pd.to_timedelta(retrieve_hass_conf['optimization_time_step'], "minutes")
    if retrieve_hass_conf.get('time_zone',None) is not None:
        retrieve_hass_conf["time_zone"] = pytz.timezone(retrieve_hass_conf["time_zone"])    

    return retrieve_hass_conf, optim_conf, plant_conf


def get_injection_dict(df: pd.DataFrame, plot_size: Optional[int] = 1366) -> dict:
    """
    Build a dictionary with graphs and tables for the webui.

    :param df: The optimization result DataFrame
    :type df: pd.DataFrame
    :param plot_size: Size of the plot figure in pixels, defaults to 1366
    :type plot_size: Optional[int], optional
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict

    """
    cols_p = [i for i in df.columns.to_list() if "P_" in i]
    # Let's round the data in the DF
    optim_status = df["optim_status"].unique().item()
    df.drop("optim_status", axis=1, inplace=True)
    cols_else = [i for i in df.columns.to_list() if "P_" not in i]
    df = df.apply(pd.to_numeric)
    df[cols_p] = df[cols_p].astype(int)
    df[cols_else] = df[cols_else].round(3)
    # Create plots
    n_colors = len(cols_p)
    colors = px.colors.sample_colorscale(
        "jet", [n / (n_colors - 1) for n in range(n_colors)]
    )
    fig_0 = px.line(
        df[cols_p],
        title="Systems powers schedule after optimization results",
        template="presentation",
        line_shape="hv",
        color_discrete_sequence=colors,
    )
    fig_0.update_layout(xaxis_title="Timestamp", yaxis_title="System powers (W)")
    if "SOC_opt" in df.columns.to_list():
        fig_1 = px.line(
            df["SOC_opt"],
            title="Battery state of charge schedule after optimization results",
            template="presentation",
            line_shape="hv",
            color_discrete_sequence=colors,
        )
        fig_1.update_layout(xaxis_title="Timestamp", yaxis_title="Battery SOC (%)")
    cols_cost = [i for i in df.columns.to_list() if "cost_" in i or "unit_" in i]
    n_colors = len(cols_cost)
    colors = px.colors.sample_colorscale(
        "jet", [n / (n_colors - 1) for n in range(n_colors)]
    )
    fig_2 = px.line(
        df[cols_cost],
        title="Systems costs obtained from optimization results",
        template="presentation",
        line_shape="hv",
        color_discrete_sequence=colors,
    )
    fig_2.update_layout(xaxis_title="Timestamp", yaxis_title="System costs (currency)")
    # Get full path to image
    image_path_0 = fig_0.to_html(full_html=False, default_width="75%")
    if "SOC_opt" in df.columns.to_list():
        image_path_1 = fig_1.to_html(full_html=False, default_width="75%")
    image_path_2 = fig_2.to_html(full_html=False, default_width="75%")
    # The tables
    table1 = df.reset_index().to_html(classes="mystyle", index=False)
    cost_cols = [i for i in df.columns if "cost_" in i]
    table2 = df[cost_cols].reset_index().sum(numeric_only=True)
    table2["optim_status"] = optim_status
    table2 = (
        table2.to_frame(name="Value")
        .reset_index(names="Variable")
        .to_html(classes="mystyle", index=False)
    )
    # The dict of plots
    injection_dict = {}
    injection_dict["title"] = "<h2>EMHASS optimization results</h2>"
    injection_dict["subsubtitle0"] = "<h4>Plotting latest optimization results</h4>"
    injection_dict["figure_0"] = image_path_0
    if "SOC_opt" in df.columns.to_list():
        injection_dict["figure_1"] = image_path_1
    injection_dict["figure_2"] = image_path_2
    injection_dict["subsubtitle1"] = "<h4>Last run optimization results table</h4>"
    injection_dict["table1"] = table1
    injection_dict["subsubtitle2"] = (
        "<h4>Summary table for latest optimization results</h4>"
    )
    injection_dict["table2"] = table2
    return injection_dict


def get_injection_dict_forecast_model_fit(df_fit_pred: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF fit case.

    :param df_fit_pred: The fit result DataFrame
    :type df_fit_pred: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
    fig = df_fit_pred.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=mlf.model_type)
    fig.update_xaxes(title_text="Time")
    image_path_0 = fig.to_html(full_html=False, default_width="75%")
    # The dict of plots
    injection_dict = {}
    injection_dict["title"] = "<h2>Custom machine learning forecast model fit</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Plotting train/test forecast model results for " + mlf.model_type + "</h4>"
    )
    injection_dict["subsubtitle0"] = (
        "<h4>Forecasting variable " + mlf.var_model + "</h4>"
    )
    injection_dict["figure_0"] = image_path_0
    return injection_dict


def get_injection_dict_forecast_model_tune(df_pred_optim: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF tune case.

    :param df_pred_optim: The tune result DataFrame
    :type df_pred_optim: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
    fig = df_pred_optim.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=mlf.model_type)
    fig.update_xaxes(title_text="Time")
    image_path_0 = fig.to_html(full_html=False, default_width="75%")
    # The dict of plots
    injection_dict = {}
    injection_dict["title"] = "<h2>Custom machine learning forecast model tune</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Performed a tuning routine using bayesian optimization for "
        + mlf.model_type
        + "</h4>"
    )
    injection_dict["subsubtitle0"] = (
        "<h4>Forecasting variable " + mlf.var_model + "</h4>"
    )
    injection_dict["figure_0"] = image_path_0
    return injection_dict

def build_config(emhass_conf: dict, logger: logging.Logger, defaults_path: str, config_path: Optional[str] = None, 
                 legacy_config_path: Optional[str] = None) -> dict:
    """
    Retrieve parameters from configuration files. 
    priority order (low - high) = defaults_path, config_path legacy_config_path

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :param defaults_path: path to config file for parameter defaults (config_defaults.json)
    :type defaults_path: str
    :param config_path: path to the main configuration file (config.json)
    :type config_path: str
    :param legacy_config_path: path to legacy config file (config_emhass.yaml)
    :type legacy_config_path: str
    :return: The built config dictionary
    :rtype: dict
    """

    # Read default parameters (default root_path/data/config_defaults.json)
    if defaults_path and pathlib.Path(defaults_path).is_file():
        with defaults_path.open('r') as data:
            config = json.load(data)
    else:
        logger.error("config_defaults.json. does not exist ")
        return False
    
    # Read user config parameters if provided (default /share/config.json)
    if config_path and pathlib.Path(config_path).is_file():
        with config_path.open('r') as data:
            # Set override default parameters (config_defaults) with user given parameters (config.json)
            logger.info("Obtaining parameters from config.json:")
            config.update(json.load(data))
    else:
        logger.info("config.json does not exist, or has not been passed")
        logger.info("you may like to generate the config.json file on the configuration page")

    # Check to see if legacy config_emhass.yaml was provided (default /app/config_emhass.yaml)
    # Convert legacy parameter definitions/format to match config.json
    if legacy_config_path and pathlib.Path(legacy_config_path).is_file():
        with open(legacy_config_path, 'r') as data:
            legacy_config = yaml.load(data, Loader=yaml.FullLoader)
            legacy_config_parameters = build_legacy_config_params(emhass_conf,legacy_config,logger)
            if type(legacy_config_parameters) is not bool:
                logger.info("Obtaining parameters from config_emhass.yaml:")
                config.update(legacy_config_parameters)     

    return config


def build_legacy_config_params(emhass_conf: dict, legacy_config: dict,
                 logger: logging.Logger) -> dict:
    """
    Build a config dictionary with legacy config_emhass.yaml file. 
    Uses the associations file to convert parameter naming conventions (to config.json/config_defaults.json).
    Extracts the parameter values and formats to match config.json.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param legacy_config: The legacy config dictionary
    :type legacy_config: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built config dictionary
    :rtype: dict
    """


    # Association file key reference
    # association[0] = config catagories
    # association[1] = legacy parameter name
    # association[2] = parameter (config.json/config_defaults.json)
    # association[3] = parameter list name if exists (not used, from legacy options.json) 

    # Check each config catagories exists, else create blank dict for categories (avoid errors)
    legacy_config['retrieve_hass_conf'] = legacy_config.get('retrieve_hass_conf',{})
    legacy_config['optim_conf'] = legacy_config.get('optim_conf',{})
    legacy_config['plant_conf'] = legacy_config.get('plant_conf',{})
    config = {}

    # Use associations list to map legacy parameter name with config.json parameter name
    if emhass_conf['associations_path'].exists():
            with emhass_conf['associations_path'].open('r') as data:
                associations = list(csv.reader(data, delimiter=","))
    else:
        logger.error("Cant find associations file (associations.csv) in: "  + str(emhass_conf['associations_path']))
        return False
    
    # Loop through all parameters in association file
    # Append config with existing legacy config parameters (converting alternative parameter naming conventions with associations list)
    for association in associations:
        # if legacy config catagories exists and if legacy parameter exists in config catagories
        if legacy_config.get(association[0],None) is not None and legacy_config[association[0]].get(association[1],None) is not None:
            config[association[2]] = legacy_config[association[0]][association[1]]
            
            # If config now has load_peak_hour_periods, extract from list of dict
            if association[2] == "load_peak_hour_periods" and type(config[association[2]]) is list:
                config[association[2]] = dict((key, d[key]) for d in config[association[2]] for key in d)
                
    return config
    # params['associations_dict'] = associations_dict

def param_to_config(param: dict,
                 logger: logging.Logger) -> dict:
    """
    A function that extracts the parameters from param back to the config.json format.
    Extracts parameters from config catagories.
    Attempts to exclude secrets hosed in retrieve_hass_conf.
    
    :param params: Built configuration parameters
    :type param: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built config dictionary
    :rtype: dict
    """ 
    logger.debug("Converting param to config")

    return_config = {}

    config_catagories = ["retrieve_hass_conf","optim_conf","plant_conf"]
    secret_params = ["hass_url", "time_zone", "Latitude", "Longitude", "Altitude", "long_lived_token", "solcast_api_key", "solcast_rooftop_id", "solar_forecast_kwp"]
    
    # Loop through config catagories that contain config params, and extract
    for config in config_catagories:
        for parameter in param[config]:
                # If parameter is not a secret, append to return_config
                if parameter not in secret_params:
                    return_config[str(parameter)] = param[config][parameter]
    
    return return_config

def build_secrets(emhass_conf: dict, logger: logging.Logger, argument: Optional[dict] = {}, options_path: Optional[str] = None, 
                 secrets_path: Optional[str] = None, no_response: Optional[bool] = False) -> Tuple[dict, dict]:    
    """
    Retrieve and build parameters from secrets locations (ENV, ARG, Secrets file (secrets_emhass.yaml/options.json) and/or Home Assistant (via API))
    priority order (lwo to high) = Defaults (written in function), ENV, Options json file, Home Assistant API,  Secrets yaml file, Arguments
    
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :param argument: dictionary of secrets arguments passed (url,key)
    :type argument: dict
    :param options_path: path to the options file (options.json) (usually provided bt EMHASS-Add-on)
    :type options_path: str
    :param secrets_path: path to secrets file (secrets_emhass.yaml)
    :type secrets_path: str
    :param no_response: bypass get request to Home Assistant (json response errors)
    :type no_response: bool
    :return: Updated emhass_conf, the built secrets dictionary
    :rtype: Tuple[dict, dict]:
    """

    #Set defaults to be overwritten
    params_secrets = {
        "hass_url": "https://myhass.duckdns.org/",
        "long_lived_token": "thatverylongtokenhere",
        "time_zone": "Europe/Paris",
        "Latitude": 45.83,
        "Longitude": 6.86,
        "Altitude": 4807.8,
        "solcast_api_key": "yoursecretsolcastapikey",
        "solcast_rooftop_id": "yourrooftopid",
        "solar_forecast_kwp": 5
    }

    # Obtain Secrets from ENV?
    params_secrets['hass_url'] = os.getenv("EMHASS_URL",params_secrets['hass_url'])
    params_secrets['long_lived_token'] = os.getenv("SUPERVISOR_TOKEN", params_secrets['long_lived_token'])   
    params_secrets['time_zone'] = os.getenv("TIME_ZONE", params_secrets['time_zone'])
    params_secrets['Latitude'] = float(os.getenv("LAT", params_secrets['Latitude']))
    params_secrets['Longitude'] = float(os.getenv("LON",  params_secrets['Longitude']))
    params_secrets['Altitude'] = float(os.getenv("ALT", params_secrets['Altitude']))      

    # Obtain secrets from options.json (Generated from EMHASS-Add-on, Home Assistant addon Configuration page) or Home Assistant API (from local Supervisor API)?
    # Use local supervisor API to obtain secrets from Home Assistant if hass_url in options.json is empty and SUPERVISOR_TOKEN ENV exists (provided by Home Assistant when running the container as addon)
    options = {}
    if options_path and pathlib.Path(options_path).is_file():
        with options_path.open('r') as data:
            options = json.load(data)
            
            # Obtain secrets from Home Assistant?
            url_from_options = options.get('hass_url', 'empty')
            key_from_options = options.get('long_lived_token', 'empty') 

            # If data path specified by options.json, overwrite emhass_conf['data_path']
            if options.get('data_path', None) != None and pathlib.Path(options['data_path']).exists():
                emhass_conf['data_path'] = pathlib.Path(options['data_path']);  
            
            # Check to use Home Assistant local API
            if not no_response and \
                (url_from_options == 'empty' or url_from_options == '' or url_from_options == "http://supervisor/core/api") and \
                os.getenv("SUPERVISOR_TOKEN", None) is not None:
                
                params_secrets['long_lived_token'] = os.getenv("SUPERVISOR_TOKEN",None)
                params_secrets['hass_url'] = "http://supervisor/core/api"
                headers = {
                "Authorization": "Bearer " + params_secrets['long_lived_token'],
                "content-type": "application/json"
                }
                # Obtain secrets from Home Assistant via API
                logger.debug("Obtaining secrets from Home Assistant Supervisor API")
                response = get((params_secrets['hass_url'] + "/config"), headers=headers)
                if response.status_code < 400:
                    config_hass = response.json()
                    params_secrets = {
                    'hass_url': params_secrets['hass_url'],
                    'long_lived_token': params_secrets['long_lived_token'],
                    'time_zone': config_hass['time_zone'],
                    'Latitude': config_hass['latitude'],
                    'Longitude': config_hass['longitude'],
                    'Altitude': config_hass['elevation']
                    }
                else: 
                    # Obtain the url and key secrets if any from options.json (default /app/options.json)
                    logger.warning("Error obtaining secrets from Home Assistant Supervisor API")
                    logger.debug("Obtaining url and key secrets from options.json")
                    if url_from_options != 'empty' and url_from_options != '':
                        params_secrets['hass_url'] = url_from_options
                    if key_from_options != 'empty' and key_from_options != '':
                        params_secrets['long_lived_token'] = key_from_options
                    if options.get('time_zone',"empty") != "empty" and options['time_zone'] != '':
                        params_secrets['time_zone'] = options['time_zone']
                    if options.get('Latitude',None) is not None and bool(options['Latitude']):
                        params_secrets['Latitude'] = options['Latitude']
                    if options.get('Longitude',None) is not None and bool(options['Longitude']):
                        params_secrets['Longitude'] = options['Longitude']         
                    if options.get('Altitude',None) is not None and bool(options['Altitude']):
                        params_secrets['Altitude'] = options['Altitude']
            else:
                # Obtain the url and key secrets if any from options.json (default /app/options.json)
                logger.debug("Obtaining url and key secrets from options.json")
                if url_from_options != 'empty' and url_from_options != '':
                    params_secrets['hass_url'] = url_from_options
                if key_from_options != 'empty' and key_from_options != '':
                    params_secrets['long_lived_token'] = key_from_options
                if options.get('time_zone',"empty") != "empty" and options['time_zone'] != '':
                    params_secrets['time_zone'] = options['time_zone']
                if options.get('Latitude',None) is not None and bool(options['Latitude']):
                    params_secrets['Latitude'] = options['Latitude']
                if options.get('Longitude',None) is not None and bool(options['Longitude']):
                    params_secrets['Longitude'] = options['Longitude']
                if options.get('Altitude',None) is not None and bool(options['Altitude']):
                    params_secrets['Altitude'] = options['Altitude']
                
            # Obtain the forecast secrets (if any) from options.json (default /app/options.json)
            forecast_secrets = ["solcast_api_key","solcast_rooftop_id","solar_forecast_kwp"]
            if any(x in forecast_secrets for x in list(options.keys())):
                logger.debug("Obtaining forecast secrets from options.json")
                if options.get('solcast_api_key',"empty") != "empty" and options['solcast_api_key'] != '':
                    params_secrets['solcast_api_key'] = options['solcast_api_key']
                if options.get('solcast_rooftop_id',"empty") != "empty" and options['solcast_rooftop_id'] != '':
                    params_secrets['solcast_rooftop_id'] = options['solcast_rooftop_id']
                if options.get('solar_forecast_kwp',None) and bool(options['solar_forecast_kwp']):
                    params_secrets['solar_forecast_kwp'] = options['solar_forecast_kwp']
     
    # Obtain secrets from secrets_emhass.yaml? (default /app/secrets_emhass.yaml)
    if secrets_path and pathlib.Path(secrets_path).is_file():
        logger.debug("Obtaining secrets from secrets file")
        with open(pathlib.Path(secrets_path), 'r') as file:
            params_secrets.update(yaml.load(file, Loader=yaml.FullLoader))

    # Receive key and url from ARG/arguments? 
    if argument.get('url',None) is not None:
       params_secrets['hass_url'] = argument['url']
       logger.debug("Obtaining url from passed argument")  
    if argument.get('key',None) is not None:
        params_secrets['long_lived_token'] = argument['key']  
        logger.debug("Obtaining long_lived_token from passed argument") 
 
    return emhass_conf, params_secrets 
 
   

def build_params(emhass_conf: dict, params_secrets: dict, config: dict,
                 logger: logging.Logger) -> dict:
    """
    Build the main params dictionary from the config and secrets
    Appends configuration catagories used by emhass to the parameters. (with use of the associations file as a reference)
    
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param params_secrets: The dictionary containing the built secret variables
    :type params_secrets: dict
    :param config: The dictionary of built config parameters 
    :type config: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built param dictionary
    :rtype: dict
    """
    if type(params_secrets) is not dict:
        params_secrets = {}
    
    params = {}
    #Start with blank config catagories
    params['retrieve_hass_conf'] = {}
    params['params_secrets'] = {}
    params['optim_conf'] = {}
    params['plant_conf'] = {}

    # Obtain associations to categorize parameters to their corresponding config catagories 
    if emhass_conf.get('associations_path', get_root(__file__, num_parent=2) / 'data/associations.csv').exists():
        with emhass_conf['associations_path'].open('r') as data:
            associations = list(csv.reader(data, delimiter=","))
    else:
        logger.error("Unable to obtain the associations file (associations.csv) in: "  + str(emhass_conf['associations_path']))
        return False

    # Association file key reference
    # association[0] = config catagories
    # association[1] = legacy parameter name
    # association[2] = parameter (config.json/config_defaults.json)
    # association[3] = parameter list name if exists (not used, from legacy options.json) 

    # Use association list to append parameters from config into params (with corresponding config catagories)
    for association in associations:
        # If parameter has list_ name and parameter in config is presented with its list name 
        # (ie, config parameter is in legacy options.json format)
        if len(association) == 4 and config.get(association[3],None) is not None:
            # Extract lists of dictionaries
            if config[association[3]] and type(config[association[3]][0]) is dict:
                params[association[0]][association[2]] = [i[association[2]] for i in config[association[3]]]
            else:
                params[association[0]][association[2]] = config[association[3]]
        # Else, directly set value of config parameter to param  
        elif config.get(association[2],None) is not None:
            params[association[0]][association[2]] = config[association[2]]

    # Check if we need to create `list_hp_periods` from config (ie. legacy options.json format)
    if params.get('optim_conf',None) is not None and config.get("list_peak_hours_periods_start_hours", None) is not None and config.get("list_peak_hours_periods_end_hours", None) is not None:
            start_hours_list = [i["peak_hours_periods_start_hours"] for i in config["list_peak_hours_periods_start_hours"]]
            end_hours_list = [i["peak_hours_periods_end_hours"] for i in config["list_peak_hours_periods_end_hours"]]
            num_peak_hours = len(start_hours_list)
            list_hp_periods_list = {'period_hp_'+str(i+1):[{'start':start_hours_list[i]},{'end':end_hours_list[i]}] for i in range(num_peak_hours)}
            params['optim_conf']['load_peak_hour_periods'] = list_hp_periods_list
    else:
        # Else, check param already contains load_peak_hour_periods from config
        if params['optim_conf'].get('load_peak_hour_periods',None) is None:
            logger.warning("Unable to detect or create load_peak_hour_periods parameter")

    # Format load_peak_hour_periods list to dict if necessary
    if params['optim_conf'].get('load_peak_hour_periods',None) is not None and isinstance(params['optim_conf']['load_peak_hour_periods'], list):
        params['optim_conf']['load_peak_hour_periods'] = dict((key, d[key]) for d in params['optim_conf']['load_peak_hour_periods'] for key in d)

    # Call function to check parameter lists that require the same length as deferrable loads
    # If not, set defaults it fill in gaps
    if params['optim_conf'].get('number_of_deferrable_loads',None) is not None:
        num_def_loads = params['optim_conf']['number_of_deferrable_loads']
        params['optim_conf']['start_timesteps_of_each_deferrable_load'] = check_def_loads(num_def_loads,params['optim_conf'],0,'start_timesteps_of_each_deferrable_load',logger)
        params['optim_conf']['end_timesteps_of_each_deferrable_load'] = check_def_loads(num_def_loads,params['optim_conf'],0,'end_timesteps_of_each_deferrable_load',logger)
        params['optim_conf']['set_deferrable_load_single_constant'] = check_def_loads(num_def_loads,params['optim_conf'],False,'set_deferrable_load_single_constant',logger)
        params['optim_conf']['treat_deferrable_load_as_semi_cont'] = check_def_loads(num_def_loads,params['optim_conf'],True,'treat_deferrable_load_as_semi_cont',logger)
        params['optim_conf']['set_deferrable_startup_penalty'] = check_def_loads(num_def_loads,params['optim_conf'],0.0,'set_deferrable_startup_penalty',logger)
        params['optim_conf']['operating_hours_of_each_deferrable_load'] = check_def_loads(num_def_loads,params['optim_conf'],0,'operating_hours_of_each_deferrable_load',logger)
        params['optim_conf']['nominal_power_of_deferrable_loads'] = check_def_loads(num_def_loads,params['optim_conf'],0,'nominal_power_of_deferrable_loads',logger)
    else:
        logger.warning("unable to obtain parameter: number_of_deferrable_loads")
    # historic_days_to_retrieve should be no less then 2
    if params["retrieve_hass_conf"].get('historic_days_to_retrieve',None) is not None:
        if params["retrieve_hass_conf"]['historic_days_to_retrieve'] < 2:
            params["retrieve_hass_conf"]['historic_days_to_retrieve'] = 2
            logger.warning("days_to_retrieve should not be lower then 2, setting days_to_retrieve to 2. Make sure your sensors also have at least 2 days of history")
    else:
        logger.warning("unable to obtain parameter: historic_days_to_retrieve")

    # Configure secrets, set params to correct config categorie
    # retrieve_hass_conf
    params['retrieve_hass_conf']["hass_url"] = params_secrets.get("hass_url",None)
    params['retrieve_hass_conf']["long_lived_token"] = params_secrets.get("long_lived_token",None)
    params['retrieve_hass_conf']["time_zone"] = params_secrets.get("time_zone",None)
    params['retrieve_hass_conf']['Latitude'] = params_secrets.get('Latitude',None)
    params['retrieve_hass_conf']['Longitude'] = params_secrets.get('Longitude',None)
    params['retrieve_hass_conf']['Altitude'] = params_secrets.get('Altitude',None)
    # Update optional param secrets
    if params["optim_conf"].get('weather_forecast_method',None) is not None:
        if params["optim_conf"]['weather_forecast_method'] == "solcast":
            params["retrieve_hass_conf"]["solcast_api_key"] = params_secrets.get("solcast_api_key", "123456")
            params["params_secrets"]["solcast_api_key"] = params_secrets.get("solcast_api_key", "123456")
            params["retrieve_hass_conf"]["solcast_rooftop_id"] = params_secrets.get("solcast_rooftop_id", "123456")
            params["params_secrets"]["solcast_rooftop_id"] = params_secrets.get("solcast_rooftop_id", "123456")
        elif params["optim_conf"]['weather_forecast_method'] == "solar.forecast":
            params["retrieve_hass_conf"]["solar_forecast_kwp"] = params_secrets.get("solar_forecast_kwp", 5)
            params["params_secrets"]["solar_forecast_kwp"] = params_secrets.get("solar_forecast_kwp", 5)
    else:
        logger.warning("Unable to detect weather_forecast_method parameter")   
    #  Check if secrets parameters still defaults values
    secret_params = ["https://myhass.duckdns.org/","thatverylongtokenhere",45.83,6.86,4807.8]
    if any(x in secret_params for x in params['retrieve_hass_conf'].values()):
        logger.warning("Some secret parameters values are still matching their defaults")
    

    # Set empty dict objects for params passed_data
    # To be latter populated with runtime parameters (treat_runtimeparams)
    params["passed_data"] = {
        "pv_power_forecast": None,
        "load_power_forecast": None,
        "load_cost_forecast": None,
        "prod_price_forecast": None,
        "prediction_horizon": None,
        "soc_init": None,
        "soc_final": None,
        'operating_hours_of_each_deferrable_load': None,
        'start_timesteps_of_each_deferrable_load': None,
        'end_timesteps_of_each_deferrable_load': None,
        "alpha": None,
        "beta": None,
    }

    return params

def check_def_loads(num_def_loads: int, parameter: list[dict], default, parameter_name: str, logger):
    """
    Check parameter lists with deferrable loads number, if they do not match, enlarge to fit.

    :param num_def_loads: Total number deferrable loads
    :type num_def_loads: int
    :param parameter: parameter config dict containing paramater
    :type: list[dict]
    :param default: default value for parameter to pad missing
    :type: obj
    :param parameter_name: name of parameter
    :type logger: str
    :param logger: The logger object
    :type logger: logging.Logger
    return: parameter list 
    :rtype: list[dict]

    """
    if parameter.get(parameter_name,None) is not None and type(parameter[parameter_name]) is list and num_def_loads > len(parameter[parameter_name]):
        logger.warning(parameter_name + " does not match number in num_def_loads, adding default values ("+ str(default) + ") to parameter")
        for x in range(len(parameter[parameter_name]), num_def_loads):
            parameter[parameter_name].append(default)
    return parameter[parameter_name]


def get_days_list(days_to_retrieve: int) -> pd.date_range:
    """
    Get list of past days from today to days_to_retrieve.

    :param days_to_retrieve: Total number of days to retrieve from the past
    :type days_to_retrieve: int
    :return: The list of days
    :rtype: pd.date_range

    """
    today = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    d = (today - timedelta(days=days_to_retrieve)).isoformat()
    days_list = pd.date_range(start=d, end=today.isoformat(), freq="D")
    return days_list


def set_df_index_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the freq of a DataFrame DateTimeIndex.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Input DataFrame with freq defined
    :rtype: pd.DataFrame

    """
    idx_diff = np.diff(df.index)
    # Sometimes there are zero values in this list.
    idx_diff = idx_diff[np.nonzero(idx_diff)]
    sampling = pd.to_timedelta(np.median(idx_diff))
    df = df[~df.index.duplicated()]
    return df.asfreq(sampling)
