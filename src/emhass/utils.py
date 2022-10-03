#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional
import numpy as np, pandas as pd
import yaml, pytz, logging, pathlib, json, copy
from datetime import datetime, timedelta, timezone


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

def get_logger(fun_name: str, config_path: str, save_to_file: Optional[bool] = True) -> Tuple[logging.Logger, logging.StreamHandler]:
    """
    Create a simple logger object.
    
    :param fun_name: The Python function object name where the logger will be used
    :type fun_name: str
    :param config_path: The path to the yaml configuration file
    :type config_path: str
    :param save_to_file: Write log to a file, defaults to True
    :type save_to_file: bool, optional
    :return: The logger object and the handler
    :rtype: object
    
    """
	# create logger object
    logger = logging.getLogger(fun_name)
    logger.propagate = True
    logger.setLevel(logging.DEBUG)
    logger.fileSetting = save_to_file
    if save_to_file:
        ch = logging.FileHandler(config_path + '/data/logger_emhass.log')
    else:
        ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, ch

def get_forecast_dates(freq: int, delta_forecast: int, 
                       timedelta_days: Optional[int] = 0) -> pd.core.indexes.datetimes.DatetimeIndex:
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
        freq=freq).round(freq)
    return forecast_dates

def treat_runtimeparams(runtimeparams: str, params:str, retrieve_hass_conf: dict, optim_conf: dict, plant_conf: dict,
                        set_type: str, logger: logging.Logger) -> Tuple[str, dict]:
    """
    Treat the passed optimization runtime parameters. 
    
    :param runtimeparams: Json string containing the runtime parameters dict.
    :type runtimeparams: str
    :param params: Configuration parameters passed from data/options.json
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
    if runtimeparams is not None:
        runtimeparams = json.loads(runtimeparams)
        if (params != None) and (params != 'null'):
            params = json.loads(params)
        else:
            params = {'passed_data':{'pv_power_forecast':None,'load_power_forecast':None,'load_cost_forecast':None,'prod_price_forecast':None,
                                     'prediction_horizon':None,'soc_init':None,'soc_final':None,'def_total_hours':None,'alpha':None,'beta':None}}
        freq = int(retrieve_hass_conf['freq'].seconds/60.0)
        delta_forecast = int(optim_conf['delta_forecast'].days)
        forecast_dates = get_forecast_dates(freq, delta_forecast)
        # Treating special data passed for MPC control case
        if set_type == 'naive-mpc-optim':
            if 'prediction_horizon' not in runtimeparams.keys():
                prediction_horizon = 10 # 10 time steps by default
            else:
                prediction_horizon = runtimeparams['prediction_horizon']
            params['passed_data']['prediction_horizon'] = prediction_horizon
            if 'soc_init' not in runtimeparams.keys():
                soc_init = plant_conf['SOCtarget']
            else:
                soc_init = runtimeparams['soc_init']
            params['passed_data']['soc_init'] = soc_init
            if 'soc_final' not in runtimeparams.keys():
                soc_final = plant_conf['SOCtarget']
            else:
                soc_final = runtimeparams['soc_final']
            params['passed_data']['soc_final'] = soc_final
            if 'def_total_hours' not in runtimeparams.keys():
                def_total_hours = optim_conf['def_total_hours']
            else:
                def_total_hours = runtimeparams['def_total_hours']
            params['passed_data']['def_total_hours'] = def_total_hours
            if 'alpha' not in runtimeparams.keys():
                alpha = 0.5
            else:
                alpha = runtimeparams['alpha']
            params['passed_data']['alpha'] = alpha
            if 'beta' not in runtimeparams.keys():
                beta = 0.5
            else:
                beta = runtimeparams['beta']
            params['passed_data']['beta'] = beta
            forecast_dates = copy.deepcopy(forecast_dates)[0:prediction_horizon]
        # Treat passed forecast data lists
        if 'pv_power_forecast' in runtimeparams.keys():
            if type(runtimeparams['pv_power_forecast']) == list and len(runtimeparams['pv_power_forecast']) >= len(forecast_dates):
                params['passed_data']['pv_power_forecast'] = runtimeparams['pv_power_forecast']
                optim_conf['weather_forecast_method'] = 'list'
            else:
                logger.error("ERROR: The passed data is either not a list or the length is not correct, length should be "+str(len(forecast_dates)))
                logger.error("Passed type is "+str(type(runtimeparams['pv_power_forecast']))+" and length is "+str(len(forecast_dates)))
            if len([x for x in runtimeparams['pv_power_forecast'] if not str(x).isdigit()]) > 0:
                logger.warning("There are non numeric values on the passed data, check for missing values (nans, null, etc)")
        if 'load_power_forecast' in runtimeparams.keys():
            if type(runtimeparams['load_power_forecast']) == list and len(runtimeparams['load_power_forecast']) >= len(forecast_dates):
                params['passed_data']['load_power_forecast'] = runtimeparams['load_power_forecast']
                optim_conf['load_forecast_method'] = 'list'
            else:
                logger.error("ERROR: The passed data is either not a list or the length is not correct, length should be "+str(len(forecast_dates)))
                logger.error("Passed type is "+str(type(runtimeparams['load_power_forecast']))+" and length is "+str(len(forecast_dates)))
            if len([x for x in runtimeparams['load_power_forecast'] if not str(x).isdigit()]) > 0:
                logger.warning("There are non numeric values on the passed data, check for missing values (nans, null, etc)")
        if 'load_cost_forecast' in runtimeparams.keys():
            if type(runtimeparams['load_cost_forecast']) == list and len(runtimeparams['load_cost_forecast']) >= len(forecast_dates):
                params['passed_data']['load_cost_forecast'] = runtimeparams['load_cost_forecast']
                optim_conf['load_cost_forecast_method'] = 'list'
            else:
                logger.error("ERROR: The passed data is either not a list or the length is not correct, length should be "+str(len(forecast_dates)))
                logger.error("Passed type is "+str(type(runtimeparams['load_cost_forecast']))+" and length is "+str(len(forecast_dates)))
            if len([x for x in runtimeparams['load_cost_forecast'] if not str(x).isdigit()]) > 0:
                logger.warning("There are non numeric values on the passed data, check for missing values (nans, null, etc)")
        if 'prod_price_forecast' in runtimeparams.keys():
            if type(runtimeparams['prod_price_forecast']) == list and len(runtimeparams['prod_price_forecast']) >= len(forecast_dates):
                params['passed_data']['prod_price_forecast'] = runtimeparams['prod_price_forecast']
                optim_conf['prod_price_forecast_method'] = 'list'
            else:
                logger.error("ERROR: The passed data is either not a list or the length is not correct, length should be "+str(len(forecast_dates)))
                logger.error("Passed type is "+str(type(runtimeparams['prod_price_forecast']))+" and length is "+str(len(forecast_dates)))
            if len([x for x in runtimeparams['prod_price_forecast'] if not str(x).isdigit()]) > 0:
                logger.warning("There are non numeric values on the passed data, check for missing values (nans, null, etc)")
        # Treat optimization configuration parameters passed at runtime 
        if 'num_def_loads' in runtimeparams.keys():
            optim_conf['num_def_loads'] = runtimeparams['num_def_loads']
        if 'P_deferrable_nom' in runtimeparams.keys():
            optim_conf['P_deferrable_nom'] = runtimeparams['P_deferrable_nom']
        if 'def_total_hours' in runtimeparams.keys():
            optim_conf['def_total_hours'] = runtimeparams['def_total_hours']
        if 'treat_def_as_semi_cont' in runtimeparams.keys():
            optim_conf['treat_def_as_semi_cont'] = runtimeparams['treat_def_as_semi_cont']
        if 'set_def_constant' in runtimeparams.keys():
            optim_conf['set_def_constant'] = runtimeparams['set_def_constant']
        if 'solcast_api_key' in runtimeparams.keys():
            retrieve_hass_conf['solcast_api_key'] = runtimeparams['solcast_api_key']
            optim_conf['weather_forecast_method'] = 'solcast'
        if 'solcast_rooftop_id' in runtimeparams.keys():
            retrieve_hass_conf['solcast_rooftop_id'] = runtimeparams['solcast_rooftop_id']
            optim_conf['weather_forecast_method'] = 'solcast'
        if 'solar_forecast_kwp' in runtimeparams.keys():
            retrieve_hass_conf['solar_forecast_kwp'] = runtimeparams['solar_forecast_kwp']
            optim_conf['weather_forecast_method'] = 'solar.forecast'
        params = json.dumps(params)
    return params, retrieve_hass_conf, optim_conf

def get_yaml_parse(config_path: str, use_secrets: Optional[bool] = True,
                   params: Optional[str] = None) -> Tuple[dict, dict, dict]:
    """
    Perform parsing of the config.yaml file.
    
    :param config_path: The path to the yaml configuration file
    :type config_path: str
    :param use_secrets: Indicate if we should use a secrets file or not.
        Set to False for unit tests.
    :type use_secrets: bool, optional
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :return: A tuple with the dictionaries containing the parsed data
    :rtype: tuple(dict)

    """
    base = config_path.parent
    if params is None:
        with open(config_path, 'r') as file:
            input_conf = yaml.load(file, Loader=yaml.FullLoader)
    else:
        input_conf = json.loads(params)
    if use_secrets:
        if params is None:
            with open(base / 'secrets_emhass.yaml', 'r') as file:
                input_secrets = yaml.load(file, Loader=yaml.FullLoader)
        else:
            input_secrets = input_conf.pop('params_secrets', None)
        
    retrieve_hass_conf = dict((key,d[key]) for d in input_conf['retrieve_hass_conf'] for key in d)
    if use_secrets:
        retrieve_hass_conf = {**retrieve_hass_conf, **input_secrets}
    else:
        retrieve_hass_conf['hass_url'] = 'http://supervisor/core/api'
        retrieve_hass_conf['long_lived_token'] = '${SUPERVISOR_TOKEN}'
        retrieve_hass_conf['time_zone'] = 'Europe/Paris'
        retrieve_hass_conf['lat'] = 45.83
        retrieve_hass_conf['lon'] = 6.86
        retrieve_hass_conf['alt'] = 4807.8
    retrieve_hass_conf['freq'] = pd.to_timedelta(retrieve_hass_conf['freq'], "minutes")
    retrieve_hass_conf['time_zone'] = pytz.timezone(retrieve_hass_conf['time_zone'])
    
    optim_conf = dict((key,d[key]) for d in input_conf['optim_conf'] for key in d)
    optim_conf['list_hp_periods'] = dict((key,d[key]) for d in optim_conf['list_hp_periods'] for key in d)
    optim_conf['delta_forecast'] = pd.Timedelta(days=optim_conf['delta_forecast'])
    
    plant_conf = dict((key,d[key]) for d in input_conf['plant_conf'] for key in d)
    
    return retrieve_hass_conf, optim_conf, plant_conf

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
    days_list = pd.date_range(start=d, end=today.isoformat(), freq='D')
    
    return days_list

def set_df_index_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the freq of a DataFrame DateTimeIndex.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: Input DataFrame with freq defined
    
    """
    idx_diff = np.diff(df.index)
    sampling = pd.to_timedelta(np.median(idx_diff))
    df = df[~df.index.duplicated()]
    df = df.asfreq(sampling)
    return df
