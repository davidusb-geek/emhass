#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, pathlib, logging
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from importlib.metadata import version
import emhass
from emhass.retrieve_hass import retrieve_hass
from emhass.forecast import forecast
from emhass.optimization import optimization
from emhass.utils import get_yaml_parse, get_days_list, get_logger

def setUp(config_path: pathlib.Path, base_path: str, costfun: str, 
    params: str, logger: logging.Logger) -> dict:
    """
    Set up some of the data needed for the different actions.
    
    :param config_path: The absolute path where the config.yaml file is located
    :type config_path: str
    :param costfun: The type of cost function to use for optimization problem
    :type costfun: str
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param logger: The passed logger object
    :type logger: logging object
    :return: A dictionnary with multiple data used by the action functions
    :rtype: dict

    """
    logger.info("Setting up needed data")
    # Parsing yaml
    retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(config_path, params=params)
    # Retrieve data from hass
    days_list = get_days_list(retrieve_hass_conf['days_to_retrieve'])
    var_list = [retrieve_hass_conf['var_load'], retrieve_hass_conf['var_PV']]
    rh = retrieve_hass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                       retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'], 
                       params, base_path, logger)
    rh.get_data(days_list, var_list,
                minimal_response=False, significant_changes_only=False)
    rh.prepare_data(retrieve_hass_conf['var_load'], load_negative = retrieve_hass_conf['load_negative'],
                    set_zero_min = retrieve_hass_conf['set_zero_min'], 
                    var_replace_zero = retrieve_hass_conf['var_replace_zero'], 
                    var_interp = retrieve_hass_conf['var_interp'])
    df_input_data = rh.df_final.copy()
    # Initialize objects
    fcst = forecast(retrieve_hass_conf, optim_conf, plant_conf,
                    params, base_path, logger)
    df_weather = fcst.get_weather_forecast(method=optim_conf['weather_forecast_method'])
    P_PV_forecast = fcst.get_power_from_weather(df_weather)
    P_load_forecast = fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
    df_input_data_dayahead = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
    df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
    opt = optimization(retrieve_hass_conf, optim_conf, plant_conf, 
                       fcst.var_load_cost, fcst.var_prod_price, days_list, 
                       costfun, base_path, logger)
    # The input data dictionnary to return
    input_data_dict = {
        'root': base_path,
        'retrieve_hass_conf': retrieve_hass_conf,
        'df_input_data': df_input_data,
        'df_input_data_dayahead': df_input_data_dayahead,
        'opt': opt,
        'rh': rh,
        'fcst': fcst,
        'P_PV_forecast': P_PV_forecast,
        'P_load_forecast': P_load_forecast,
        'costfun': costfun,
        'params': params
    }
    return input_data_dict
    
def perfect_forecast_optim(input_data_dict: dict, logger: logging.Logger,
    save_data_to_file: Optional[bool] = True) -> pd.DataFrame:
    """
    Perform a call to the perfect forecast optimization routine.
    
    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    logger.info("Performing perfect forecast optimization")
    df_input_data = input_data_dict['fcst'].get_load_cost_forecast(
        input_data_dict['df_input_data'], 
        method=input_data_dict['fcst'].optim_conf['load_cost_forecast_method'])
    df_input_data = input_data_dict['fcst'].get_prod_price_forecast(
        df_input_data, method=input_data_dict['fcst'].optim_conf['prod_price_forecast_method'])
    opt_res = input_data_dict['opt'].perform_perfect_forecast_optim(df_input_data)
    # Save CSV file for analysis
    if save_data_to_file:
        filename = 'opt_res_perfect_optim_'+input_data_dict['costfun']
    else: # Just save the latest optimization results
        filename = 'opt_res_perfect_optim_latest'
    opt_res.to_csv(input_data_dict['root'] + '/data/' + filename + '.csv', index_label='timestamp')
    return opt_res
    
def dayahead_forecast_optim(input_data_dict: dict, logger: logging.Logger,
    save_data_to_file: Optional[bool] = False) -> pd.DataFrame:
    """
    Perform a call to the day-ahead optimization routine.
    
    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    logger.info("Performing day-ahead forecast optimization")
    df_input_data_dayahead = input_data_dict['fcst'].get_load_cost_forecast(
        input_data_dict['df_input_data_dayahead'],
        method=input_data_dict['fcst'].optim_conf['load_cost_forecast_method'])
    df_input_data_dayahead = input_data_dict['fcst'].get_prod_price_forecast(
        df_input_data_dayahead, method=input_data_dict['fcst'].optim_conf['prod_price_forecast_method'])
    opt_res_dayahead = input_data_dict['opt'].perform_dayahead_forecast_optim(
        df_input_data_dayahead, input_data_dict['P_PV_forecast'], input_data_dict['P_load_forecast'])
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = 'opt_res_dayahead_'+today.strftime("%Y_%m_%d")
    else: # Just save the latest optimization results
        filename = 'opt_res_dayahead_latest'
    opt_res_dayahead.to_csv(input_data_dict['root'] + '/data/' + filename + '.csv', index_label='timestamp')
    return opt_res_dayahead
    
def publish_data(input_data_dict: dict, logger: logging.Logger,
    save_data_to_file: Optional[bool] = False) -> pd.DataFrame:
    """
    Publish the data obtained from the optimization results.
    
    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :return: The output data of the optimization readed from a CSV file in the data folder
    :rtype: pd.DataFrame

    """
    logger.info("Publishing data to HASS instance")
    # Check if a day ahead optimization has been performed (read CSV file)
    if save_data_to_file:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = 'opt_res_dayahead_'+today.strftime("%Y_%m_%d")
    else:
        filename = 'opt_res_dayahead_latest'
    if not os.path.isfile(input_data_dict['root'] + '/data/' + filename + '.csv'):
        logger.error("File not found error, run the dayahead_forecast_optim first.")
    else:
        opt_res_dayahead = pd.read_csv(input_data_dict['root'] + '/data/' + filename + '.csv', index_col='timestamp')
        opt_res_dayahead.index = pd.to_datetime(opt_res_dayahead.index)
        opt_res_dayahead.index.freq = input_data_dict['retrieve_hass_conf']['freq']
    # Estimate the current index
    now_precise = datetime.now(input_data_dict['retrieve_hass_conf']['time_zone']).replace(second=0, microsecond=0)
    idx_closest = opt_res_dayahead.index.get_indexer([now_precise], method='ffill')[0]
    if idx_closest == -1:
        idx_closest = opt_res_dayahead.index.get_indexer([now_precise], method='nearest')[0]
    # Publish PV forecast
    input_data_dict['rh'].post_data(opt_res_dayahead['P_PV'], idx_closest, 
                                    'sensor.p_pv_forecast', "W", "PV Power Forecast")
    # Publish Load forecast
    input_data_dict['rh'].post_data(opt_res_dayahead['P_Load'], idx_closest, 
                                    'sensor.p_load_forecast', "W", "Load Power Forecast")
    # Publish deferrable loads
    for k in range(input_data_dict['opt'].optim_conf['num_def_loads']):
        input_data_dict['rh'].post_data(opt_res_dayahead["P_deferrable{}".format(k)], idx_closest, 
                                        'sensor.p_deferrable{}'.format(k), "W", "Deferrable Load {}".format(k))
    # Publish battery power
    if input_data_dict['opt'].optim_conf['set_use_battery']:
        input_data_dict['rh'].post_data(opt_res_dayahead['P_batt'], idx_closest,
                                        'sensor.p_batt_forecast', "W", "Battery Power Forecast")
    # Create a DF resuming what has been published
    opt_res = opt_res_dayahead.loc[opt_res_dayahead.index[idx_closest], ['P_PV', 'P_Load']]
    return opt_res
    
        
def main():
    """Define the main command line entry function."""
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, help='Set the desired action, options are: perfect-optim, dayahead-optim and publish-data')
    parser.add_argument('--config', type=str, help='Define path to the config.yaml file')
    parser.add_argument('--costfun', type=str, default='profit', help='Define the type of cost function, options are: profit, cost, self-consumption')
    parser.add_argument('--log2file', type=bool, default=False, help='Define if we should log to a file or not')
    parser.add_argument('--params', type=str, default=None, help='Configuration parameters passed from data/options.json')
    parser.add_argument('--version', action='version', version='%(prog)s '+version('emhass'))
    args = parser.parse_args()
    # The path to the configuration files
    config_path = pathlib.Path(args.config)
    base_path = str(config_path.parent)
    # create logger
    logger, ch = get_logger(__name__, base_path, save_to_file=args.log2file)
    # Setup parameters
    params = args.params
    input_data_dict = setUp(config_path, base_path, args.costfun, params, logger)
    # Perform selected action
    if args.action == 'perfect-optim':
        opt_res = perfect_forecast_optim(input_data_dict, logger)
    elif args.action == 'dayahead-optim':
        opt_res = dayahead_forecast_optim(input_data_dict, logger)
    elif args.action == 'publish-data':
        opt_res = publish_data(input_data_dict, logger)
    else:
        logger.error("The passed action argument is not valid")
        opt_res = None
    logger.info(opt_res)
    # Flush the logger
    ch.close()
    logger.removeHandler(ch)

if __name__ == '__main__':
    main()
