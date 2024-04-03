#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pathlib
import logging
import json
import copy
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Tuple
from distutils.util import strtobool

from importlib.metadata import version
from emhass.retrieve_hass import RetrieveHass
from emhass.forecast import Forecast
from emhass.machine_learning_forecaster import MLForecaster
from emhass.optimization import Optimization
from emhass import utils


def set_input_data_dict(config_path: pathlib.Path, base_path: str, costfun: str, 
    params: str, runtimeparams: str, set_type: str, logger: logging.Logger,
    get_data_from_file: Optional[bool] = False) -> dict:
    """
    Set up some of the data needed for the different actions.
    
    :param config_path: The complete absolute path where the config.yaml file is located
    :type config_path: pathlib.Path
    :param base_path: The parent folder of the config_path
    :type base_path: str
    :param costfun: The type of cost function to use for optimization problem
    :type costfun: str
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param runtimeparams: Runtime optimization parameters passed as a dictionnary
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
    # Parsing yaml
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
        config_path, use_secrets=not(get_data_from_file), params=params)
    # Treat runtimeparams
    params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
        runtimeparams, params, retrieve_hass_conf, 
        optim_conf, plant_conf, set_type, logger)
    # Define main objects
    rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                      retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'], 
                      params, base_path, logger, get_data_from_file=get_data_from_file)
    fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf,
                    params, base_path, logger, get_data_from_file=get_data_from_file)
    opt = Optimization(retrieve_hass_conf, optim_conf, plant_conf, 
                       fcst.var_load_cost, fcst.var_prod_price, 
                       costfun, base_path, logger)
    # Perform setup based on type of action
    if set_type == "perfect-optim":
        # Retrieve data from hass
        if get_data_from_file:
            with open(pathlib.Path(base_path) / 'data' / 'test_df_final.pkl', 'rb') as inp:
                rh.df_final, days_list, var_list = pickle.load(inp)
        else:
            days_list = utils.get_days_list(retrieve_hass_conf['days_to_retrieve'])
            var_list = [retrieve_hass_conf['var_load'], retrieve_hass_conf['var_PV']]
            if not rh.get_data(days_list, var_list,
                               minimal_response=False, significant_changes_only=False):
                return False 
        if not rh.prepare_data(retrieve_hass_conf['var_load'], load_negative = retrieve_hass_conf['load_negative'],
                               set_zero_min = retrieve_hass_conf['set_zero_min'], 
                               var_replace_zero = retrieve_hass_conf['var_replace_zero'], 
                               var_interp = retrieve_hass_conf['var_interp']):
            return False
        df_input_data = rh.df_final.copy()
        # What we don't need for this type of action
        P_PV_forecast, P_load_forecast, df_input_data_dayahead = None, None, None
    elif set_type == "dayahead-optim":
        # Get PV and load forecasts
        df_weather = fcst.get_weather_forecast(method=optim_conf['weather_forecast_method'])
        P_PV_forecast = fcst.get_power_from_weather(df_weather)
        P_load_forecast = fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
        if isinstance(P_load_forecast,bool) and not P_load_forecast:
            logger.error("Unable to get sensor power photovoltaics, or sensor power load no var loads. Check HA sensors and their daily data")
            return False
        df_input_data_dayahead = pd.DataFrame(np.transpose(np.vstack([P_PV_forecast.values,P_load_forecast.values])),
                                              index=P_PV_forecast.index,
                                              columns=['P_PV_forecast', 'P_load_forecast'])
        df_input_data_dayahead = utils.set_df_index_freq(df_input_data_dayahead)
        params = json.loads(params)
        if 'prediction_horizon' in params['passed_data'] and params['passed_data']['prediction_horizon'] is not None:
            prediction_horizon = params['passed_data']['prediction_horizon']
            df_input_data_dayahead = copy.deepcopy(df_input_data_dayahead)[df_input_data_dayahead.index[0]:df_input_data_dayahead.index[prediction_horizon-1]]
        # What we don't need for this type of action
        df_input_data, days_list = None, None
    elif set_type == "naive-mpc-optim":
        # Retrieve data from hass
        if get_data_from_file:
            with open(pathlib.Path(base_path) / 'data' / 'test_df_final.pkl', 'rb') as inp:
                rh.df_final, days_list, var_list = pickle.load(inp)
        else:
            days_list = utils.get_days_list(1)
            var_list = [retrieve_hass_conf['var_load'], retrieve_hass_conf['var_PV']]
            if not rh.get_data(days_list, var_list,
                               minimal_response=False, significant_changes_only=False):
                return False
        if not rh.prepare_data(retrieve_hass_conf['var_load'], load_negative = retrieve_hass_conf['load_negative'],
                               set_zero_min = retrieve_hass_conf['set_zero_min'], 
                               var_replace_zero = retrieve_hass_conf['var_replace_zero'], 
                               var_interp = retrieve_hass_conf['var_interp']):
            return False
        df_input_data = rh.df_final.copy()
        # Get PV and load forecasts
        df_weather = fcst.get_weather_forecast(method=optim_conf['weather_forecast_method'])
        P_PV_forecast = fcst.get_power_from_weather(df_weather, set_mix_forecast=True, df_now=df_input_data)
        P_load_forecast = fcst.get_load_forecast(method=optim_conf['load_forecast_method'], set_mix_forecast=True, df_now=df_input_data)
        df_input_data_dayahead = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
        df_input_data_dayahead = utils.set_df_index_freq(df_input_data_dayahead)
        df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        params = json.loads(params)
        if 'prediction_horizon' in params['passed_data'] and params['passed_data']['prediction_horizon'] is not None:
            prediction_horizon = params['passed_data']['prediction_horizon']
            df_input_data_dayahead = copy.deepcopy(df_input_data_dayahead)[df_input_data_dayahead.index[0]:df_input_data_dayahead.index[prediction_horizon-1]]
    elif set_type == "forecast-model-fit" or set_type == "forecast-model-predict" or set_type == "forecast-model-tune":
        df_input_data_dayahead = None
        P_PV_forecast, P_load_forecast = None, None
        params = json.loads(params)
        # Retrieve data from hass
        days_to_retrieve = params['passed_data']['days_to_retrieve']
        model_type = params['passed_data']['model_type']
        var_model = params['passed_data']['var_model']
        if get_data_from_file:
            days_list = None
            filename = 'data_train_'+model_type+'.pkl'
            data_path = pathlib.Path(base_path) / 'data' / filename
            with open(data_path, 'rb') as inp:
                df_input_data, _ = pickle.load(inp)
            df_input_data = df_input_data[df_input_data.index[-1] - pd.offsets.Day(days_to_retrieve):]
        else:
            days_list = utils.get_days_list(days_to_retrieve)
            var_list = [var_model]
            if not rh.get_data(days_list, var_list):
                return False
            df_input_data = rh.df_final.copy()
    elif set_type == "publish-data":
        df_input_data, df_input_data_dayahead = None, None
        P_PV_forecast, P_load_forecast = None, None
        days_list = None
    else:
        logger.error("The passed action argument and hence the set_type parameter for setup is not valid")
        df_input_data, df_input_data_dayahead = None, None
        P_PV_forecast, P_load_forecast = None, None
        days_list = None

    # The input data dictionnary to return
    input_data_dict = {
        'root': base_path,
        'retrieve_hass_conf': retrieve_hass_conf,
        'rh': rh,
        'opt': opt,
        'fcst': fcst,
        'df_input_data': df_input_data,
        'df_input_data_dayahead': df_input_data_dayahead,
        'P_PV_forecast': P_PV_forecast,
        'P_load_forecast': P_load_forecast,
        'costfun': costfun,
        'params': params,
        'days_list': days_list
    }
    return input_data_dict
    
def perfect_forecast_optim(input_data_dict: dict, logger: logging.Logger,
    save_data_to_file: Optional[bool] = True, debug: Optional[bool] = False) -> pd.DataFrame:
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
    df_input_data = input_data_dict['fcst'].get_load_cost_forecast(
        input_data_dict['df_input_data'], 
        method=input_data_dict['fcst'].optim_conf['load_cost_forecast_method'],
        list_and_perfect=True)
    df_input_data = input_data_dict['fcst'].get_prod_price_forecast(
        df_input_data, method=input_data_dict['fcst'].optim_conf['prod_price_forecast_method'],
        list_and_perfect=True)
    opt_res = input_data_dict['opt'].perform_perfect_forecast_optim(df_input_data, input_data_dict['days_list'])
    # Save CSV file for analysis
    if save_data_to_file:
        filename = 'opt_res_perfect_optim_'+input_data_dict['costfun']+'.csv'
    else: # Just save the latest optimization results
        filename = 'opt_res_latest.csv'
    if not debug:
        opt_res.to_csv(pathlib.Path(input_data_dict['root']) / filename, index_label='timestamp')
    return opt_res
    
def dayahead_forecast_optim(input_data_dict: dict, logger: logging.Logger,
    save_data_to_file: Optional[bool] = False, debug: Optional[bool] = False) -> pd.DataFrame:
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
    # Load cost and prod price forecast
    df_input_data_dayahead = input_data_dict['fcst'].get_load_cost_forecast(
        input_data_dict['df_input_data_dayahead'],
        method=input_data_dict['fcst'].optim_conf['load_cost_forecast_method'])
    df_input_data_dayahead = input_data_dict['fcst'].get_prod_price_forecast(
        df_input_data_dayahead, 
        method=input_data_dict['fcst'].optim_conf['prod_price_forecast_method'])
    opt_res_dayahead = input_data_dict['opt'].perform_dayahead_forecast_optim(
        df_input_data_dayahead, input_data_dict['P_PV_forecast'], input_data_dict['P_load_forecast'])
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = 'opt_res_dayahead_'+today.strftime("%Y_%m_%d")+'.csv'
    else: # Just save the latest optimization results
        filename = 'opt_res_latest.csv'
    if not debug:
        opt_res_dayahead.to_csv(pathlib.Path(input_data_dict['root']) / filename, index_label='timestamp')
    return opt_res_dayahead

def naive_mpc_optim(input_data_dict: dict, logger: logging.Logger,
    save_data_to_file: Optional[bool] = False, debug: Optional[bool] = False) -> pd.DataFrame:
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
    # Load cost and prod price forecast
    df_input_data_dayahead = input_data_dict['fcst'].get_load_cost_forecast(
        input_data_dict['df_input_data_dayahead'],
        method=input_data_dict['fcst'].optim_conf['load_cost_forecast_method'])
    df_input_data_dayahead = input_data_dict['fcst'].get_prod_price_forecast(
        df_input_data_dayahead, method=input_data_dict['fcst'].optim_conf['prod_price_forecast_method'])
    # The specifics params for the MPC at runtime
    prediction_horizon = input_data_dict['params']['passed_data']['prediction_horizon']
    soc_init = input_data_dict['params']['passed_data']['soc_init']
    soc_final = input_data_dict['params']['passed_data']['soc_final']
    def_total_hours = input_data_dict['params']['passed_data']['def_total_hours']
    def_start_timestep = input_data_dict['params']['passed_data']['def_start_timestep']
    def_end_timestep = input_data_dict['params']['passed_data']['def_end_timestep']
    opt_res_naive_mpc = input_data_dict['opt'].perform_naive_mpc_optim(
        df_input_data_dayahead, input_data_dict['P_PV_forecast'], input_data_dict['P_load_forecast'],
        prediction_horizon, soc_init, soc_final, def_total_hours, def_start_timestep, def_end_timestep)
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = 'opt_res_naive_mpc_'+today.strftime("%Y_%m_%d")+'.csv'
    else: # Just save the latest optimization results
        filename = 'opt_res_latest.csv'
    if not debug:
        opt_res_naive_mpc.to_csv(pathlib.Path(input_data_dict['root']) / filename, index_label='timestamp')
    return opt_res_naive_mpc

def forecast_model_fit(input_data_dict: dict, logger: logging.Logger,
    debug: Optional[bool] = False) -> Tuple[pd.DataFrame, pd.DataFrame, MLForecaster]:
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
    data = copy.deepcopy(input_data_dict['df_input_data'])
    model_type = input_data_dict['params']['passed_data']['model_type']
    var_model = input_data_dict['params']['passed_data']['var_model']
    sklearn_model = input_data_dict['params']['passed_data']['sklearn_model']
    num_lags = input_data_dict['params']['passed_data']['num_lags']
    split_date_delta = input_data_dict['params']['passed_data']['split_date_delta']
    perform_backtest = input_data_dict['params']['passed_data']['perform_backtest']
    root = input_data_dict['root']
    # The ML forecaster object
    mlf = MLForecaster(data, model_type, var_model, sklearn_model, num_lags, root, logger)
    # Fit the ML model
    df_pred, df_pred_backtest = mlf.fit(split_date_delta=split_date_delta, 
                                        perform_backtest=perform_backtest)
    # Save model
    if not debug:
        filename = model_type+'_mlf.pkl'
        with open(pathlib.Path(root) / filename, 'wb') as outp:
            pickle.dump(mlf, outp, pickle.HIGHEST_PROTOCOL)
    return df_pred, df_pred_backtest, mlf

def forecast_model_predict(input_data_dict: dict, logger: logging.Logger,
    use_last_window: Optional[bool] = True, debug: Optional[bool] = False,
    mlf: Optional[MLForecaster] = None) -> pd.DataFrame:
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
    model_type = input_data_dict['params']['passed_data']['model_type']
    root = input_data_dict['root']
    filename = model_type+'_mlf.pkl'
    filename_path = pathlib.Path(root) / filename
    if not debug:
        if filename_path.is_file():
            with open(filename_path, 'rb') as inp:
                mlf = pickle.load(inp)
        else:
            logger.error("The ML forecaster file was not found, please run a model fit method before this predict method")
            return
    # Make predictions
    if use_last_window:
        data_last_window = copy.deepcopy(input_data_dict['df_input_data'])
    else:
        data_last_window = None
    predictions = mlf.predict(data_last_window)
    # Publish data to a Home Assistant sensor
    model_predict_publish = input_data_dict['params']['passed_data']['model_predict_publish']
    model_predict_entity_id = input_data_dict['params']['passed_data']['model_predict_entity_id']
    model_predict_unit_of_measurement = input_data_dict['params']['passed_data']['model_predict_unit_of_measurement']
    model_predict_friendly_name = input_data_dict['params']['passed_data']['model_predict_friendly_name']
    publish_prefix = input_data_dict['params']['passed_data']['publish_prefix']
    if model_predict_publish is True:
        # Estimate the current index
        now_precise = datetime.now(input_data_dict['retrieve_hass_conf']['time_zone']).replace(second=0, microsecond=0)
        if input_data_dict['retrieve_hass_conf']['method_ts_round'] == 'nearest':
            idx_closest = predictions.index.get_indexer([now_precise], method='nearest')[0]
        elif input_data_dict['retrieve_hass_conf']['method_ts_round'] == 'first':
            idx_closest = predictions.index.get_indexer([now_precise], method='ffill')[0]
        elif input_data_dict['retrieve_hass_conf']['method_ts_round'] == 'last':
            idx_closest = predictions.index.get_indexer([now_precise], method='bfill')[0]
        if idx_closest == -1:
            idx_closest = predictions.index.get_indexer([now_precise], method='nearest')[0]
        # Publish Load forecast
        input_data_dict['rh'].post_data(predictions, idx_closest, 
                                        model_predict_entity_id,
                                        model_predict_unit_of_measurement, 
                                        model_predict_friendly_name,
                                        type_var = 'mlforecaster',
                                        publish_prefix=publish_prefix)
    return predictions

def forecast_model_tune(input_data_dict: dict, logger: logging.Logger,
    debug: Optional[bool] = False, mlf: Optional[MLForecaster] = None
    ) -> Tuple[pd.DataFrame, MLForecaster]:
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
    model_type = input_data_dict['params']['passed_data']['model_type']
    root = input_data_dict['root']
    filename = model_type+'_mlf.pkl'
    filename_path = pathlib.Path(root) / filename
    if not debug:
        if filename_path.is_file():
            with open(filename_path, 'rb') as inp:
                mlf = pickle.load(inp)
        else:
            logger.error("The ML forecaster file was not found, please run a model fit method before this tune method")
            return None, None
    # Tune the model
    df_pred_optim = mlf.tune(debug=debug)
    # Save model
    if not debug:
        filename = model_type+'_mlf.pkl'
        with open(pathlib.Path(root) / filename, 'wb') as outp:
            pickle.dump(mlf, outp, pickle.HIGHEST_PROTOCOL)
    return df_pred_optim, mlf

def publish_data(input_data_dict: dict, logger: logging.Logger,
    save_data_to_file: Optional[bool] = False, 
    opt_res_latest: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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

    """
    logger.info("Publishing data to HASS instance")
    # Check if a day ahead optimization has been performed (read CSV file)
    if save_data_to_file:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = 'opt_res_dayahead_'+today.strftime("%Y_%m_%d")+'.csv'
    else:
        filename = 'opt_res_latest.csv'
    if opt_res_latest is None:
        if not os.path.isfile(pathlib.Path(input_data_dict['root']) / filename):
            logger.error("File not found error, run an optimization task first.")
            return
        else:
            opt_res_latest = pd.read_csv(pathlib.Path(input_data_dict['root']) / filename, index_col='timestamp')
            opt_res_latest.index = pd.to_datetime(opt_res_latest.index)
            opt_res_latest.index.freq = input_data_dict['retrieve_hass_conf']['freq']
    # Estimate the current index
    now_precise = datetime.now(input_data_dict['retrieve_hass_conf']['time_zone']).replace(second=0, microsecond=0)
    if input_data_dict['retrieve_hass_conf']['method_ts_round'] == 'nearest':
        idx_closest = opt_res_latest.index.get_indexer([now_precise], method='nearest')[0]
    elif input_data_dict['retrieve_hass_conf']['method_ts_round'] == 'first':
        idx_closest = opt_res_latest.index.get_indexer([now_precise], method='ffill')[0]
    elif input_data_dict['retrieve_hass_conf']['method_ts_round'] == 'last':
        idx_closest = opt_res_latest.index.get_indexer([now_precise], method='bfill')[0]
    if idx_closest == -1:
        idx_closest = opt_res_latest.index.get_indexer([now_precise], method='nearest')[0]
    # Publish the data
    params = json.loads(input_data_dict['params'])
    publish_prefix = params['passed_data']['publish_prefix']
    # Publish PV forecast
    custom_pv_forecast_id = params['passed_data']['custom_pv_forecast_id']
    input_data_dict['rh'].post_data(opt_res_latest['P_PV'], idx_closest, 
                                    custom_pv_forecast_id["entity_id"], 
                                    custom_pv_forecast_id["unit_of_measurement"],
                                    custom_pv_forecast_id["friendly_name"],
                                    type_var = 'power',
                                    publish_prefix = publish_prefix)
    # Publish Load forecast
    custom_load_forecast_id = params['passed_data']['custom_load_forecast_id']
    input_data_dict['rh'].post_data(opt_res_latest['P_Load'], idx_closest, 
                                    custom_load_forecast_id["entity_id"], 
                                    custom_load_forecast_id["unit_of_measurement"],
                                    custom_load_forecast_id["friendly_name"],
                                    type_var = 'power',
                                    publish_prefix = publish_prefix)
    cols_published = ['P_PV', 'P_Load']
    # Publish deferrable loads
    custom_deferrable_forecast_id = params['passed_data']['custom_deferrable_forecast_id']
    for k in range(input_data_dict['opt'].optim_conf['num_def_loads']):
        if "P_deferrable{}".format(k) not in opt_res_latest.columns:
            logger.error("P_deferrable{}".format(k)+" was not found in results DataFrame. Optimization task may need to be relaunched or it did not converge to a solution.")
        else:
            input_data_dict['rh'].post_data(opt_res_latest["P_deferrable{}".format(k)], idx_closest, 
                                            custom_deferrable_forecast_id[k]["entity_id"], 
                                            custom_deferrable_forecast_id[k]["unit_of_measurement"],
                                            custom_deferrable_forecast_id[k]["friendly_name"],
                                            type_var = 'deferrable',
                                            publish_prefix = publish_prefix)
            cols_published = cols_published+["P_deferrable{}".format(k)]
    # Publish battery power
    if input_data_dict['opt'].optim_conf['set_use_battery']:
        if 'P_batt' not in opt_res_latest.columns:
            logger.error("P_batt was not found in results DataFrame. Optimization task may need to be relaunched or it did not converge to a solution.")
        else:
            custom_batt_forecast_id = params['passed_data']['custom_batt_forecast_id']
            input_data_dict['rh'].post_data(opt_res_latest['P_batt'], idx_closest,
                                            custom_batt_forecast_id["entity_id"], 
                                            custom_batt_forecast_id["unit_of_measurement"],
                                            custom_batt_forecast_id["friendly_name"],
                                            type_var = 'batt',
                                            publish_prefix = publish_prefix)
            cols_published = cols_published+["P_batt"]
            custom_batt_soc_forecast_id = params['passed_data']['custom_batt_soc_forecast_id']
            input_data_dict['rh'].post_data(opt_res_latest['SOC_opt']*100, idx_closest,
                                            custom_batt_soc_forecast_id["entity_id"], 
                                            custom_batt_soc_forecast_id["unit_of_measurement"],
                                            custom_batt_soc_forecast_id["friendly_name"],
                                            type_var = 'SOC',
                                            publish_prefix = publish_prefix)
            cols_published = cols_published+["SOC_opt"]
    # Publish grid power
    custom_grid_forecast_id = params['passed_data']['custom_grid_forecast_id']
    input_data_dict['rh'].post_data(opt_res_latest['P_grid'], idx_closest, 
                                    custom_grid_forecast_id["entity_id"], 
                                    custom_grid_forecast_id["unit_of_measurement"],
                                    custom_grid_forecast_id["friendly_name"],
                                    type_var = 'power',
                                    publish_prefix = publish_prefix)
    cols_published = cols_published+["P_grid"]
    # Publish total value of cost function
    custom_cost_fun_id = params['passed_data']['custom_cost_fun_id']
    col_cost_fun = [i for i in opt_res_latest.columns if 'cost_fun_' in i]
    input_data_dict['rh'].post_data(opt_res_latest[col_cost_fun], idx_closest, 
                                    custom_cost_fun_id["entity_id"], 
                                    custom_cost_fun_id["unit_of_measurement"],
                                    custom_cost_fun_id["friendly_name"],
                                    type_var = 'cost_fun',
                                    publish_prefix = publish_prefix)
    # Publish the optimization status
    custom_cost_fun_id = params['passed_data']['custom_optim_status_id']
    if "optim_status" not in opt_res_latest:
        opt_res_latest["optim_status"] = 'Optimal'
        logger.warning("no optim_status in opt_res_latest, run an optimization task first")
    input_data_dict['rh'].post_data(opt_res_latest['optim_status'], idx_closest, 
                                    custom_cost_fun_id["entity_id"], 
                                    custom_cost_fun_id["unit_of_measurement"],
                                    custom_cost_fun_id["friendly_name"],
                                    type_var = 'optim_status',
                                    publish_prefix = publish_prefix)
    cols_published = cols_published+["optim_status"]
    # Publish unit_load_cost
    custom_unit_load_cost_id = params['passed_data']['custom_unit_load_cost_id']
    input_data_dict['rh'].post_data(opt_res_latest['unit_load_cost'], idx_closest, 
                                    custom_unit_load_cost_id["entity_id"], 
                                    custom_unit_load_cost_id["unit_of_measurement"],
                                    custom_unit_load_cost_id["friendly_name"],
                                    type_var = 'unit_load_cost',
                                    publish_prefix = publish_prefix)
    cols_published = cols_published+["unit_load_cost"]
    # Publish unit_prod_price
    custom_unit_prod_price_id = params['passed_data']['custom_unit_prod_price_id']
    input_data_dict['rh'].post_data(opt_res_latest['unit_prod_price'], idx_closest, 
                                    custom_unit_prod_price_id["entity_id"], 
                                    custom_unit_prod_price_id["unit_of_measurement"],
                                    custom_unit_prod_price_id["friendly_name"],
                                    type_var = 'unit_prod_price',
                                    publish_prefix = publish_prefix)
    cols_published = cols_published+["unit_prod_price"]
    # Create a DF resuming what has been published
    opt_res = opt_res_latest[cols_published].loc[[opt_res_latest.index[idx_closest]]]
    return opt_res
    
        
def main():
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
    parser.add_argument('--action', type=str, help='Set the desired action, options are: perfect-optim, dayahead-optim,\
        naive-mpc-optim, publish-data, forecast-model-fit, forecast-model-predict, forecast-model-tune')
    parser.add_argument('--config', type=str, help='Define path to the config.yaml file')
    parser.add_argument('--costfun', type=str, default='profit', help='Define the type of cost function, options are: profit, cost, self-consumption')
    parser.add_argument('--log2file', type=strtobool, default='False', help='Define if we should log to a file or not')
    parser.add_argument('--params', type=str, default=None, help='Configuration parameters passed from data/options.json')
    parser.add_argument('--runtimeparams', type=str, default=None, help='Pass runtime optimization parameters as dictionnary')
    parser.add_argument('--debug', type=strtobool, default='False', help='Use True for testing purposes')
    args = parser.parse_args()
    # The path to the configuration files
    config_path = pathlib.Path(args.config)
    base_path = str(config_path.parent)
    # create logger
    logger, ch = utils.get_logger(__name__, base_path, save_to_file=bool(args.log2file))
    # Additionnal argument
    try:
        parser.add_argument('--version', action='version', version='%(prog)s '+version('emhass'))
        args = parser.parse_args()
    except Exception:
        logger.info("Version not found for emhass package. Or importlib exited with PackageNotFoundError.")
    # Setup parameters
    input_data_dict = set_input_data_dict(config_path, base_path, 
                                          args.costfun, args.params, args.runtimeparams, args.action, 
                                          logger, args.debug)
    # Perform selected action
    if args.action == 'perfect-optim':
        opt_res = perfect_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == 'dayahead-optim':
        opt_res = dayahead_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == 'naive-mpc-optim':
        opt_res = naive_mpc_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == 'forecast-model-fit':
        df_fit_pred, df_fit_pred_backtest, mlf = forecast_model_fit(input_data_dict, logger, debug=args.debug)
        opt_res = None
    elif args.action == 'forecast-model-predict':
        if args.debug:
            _, _, mlf = forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred = forecast_model_predict(input_data_dict, logger, debug=args.debug, mlf=mlf)
        opt_res = None
    elif args.action == 'forecast-model-tune':
        if args.debug:
            _, _, mlf = forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred_optim, mlf = forecast_model_tune(input_data_dict, logger, debug=args.debug, mlf=mlf)
        opt_res = None
    elif args.action == 'publish-data':
        opt_res = publish_data(input_data_dict, logger)
    else:
        logger.error("The passed action argument is not valid")
        opt_res = None
    logger.info(opt_res)
    # Flush the logger
    ch.close()
    logger.removeHandler(ch)
    if args.action == 'perfect-optim' or args.action == 'dayahead-optim' or \
        args.action == 'naive-mpc-optim' or args.action == 'publish-data':
        return opt_res
    elif args.action == 'forecast-model-fit':
        return df_fit_pred, df_fit_pred_backtest, mlf
    elif args.action == 'forecast-model-predict':
        return df_pred
    elif args.action == 'forecast-model-tune':
        return df_pred_optim, mlf

if __name__ == '__main__':
    main()
