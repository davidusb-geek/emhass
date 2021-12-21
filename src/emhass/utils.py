#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Tuple,
    Optional,
)
import pandas as pd
import yaml, pytz, logging, os, pathlib
from datetime import datetime, timedelta, timezone

def get_root(file: str, num_parent: Optional[int] = 3) -> str:
    """
    Get the root absolute path of the working directory.
    
    :param file: The passed file path with __file__
    :return: The root path
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

def get_logger(fun_name: str, config_path: str, file: Optional[bool] = True) -> Tuple[logging.Logger, logging.StreamHandler]:
    """
    Create a simple logger object.
    
    :param fun_name: The Python function object name where the logger will be used
    :type fun_name: str
    :param config_path: The path to the yaml configuration file
    :type config_path: str
    :param file: Write log to a file, defaults to True
    :type file: bool, optional
    :return: The logger object and the handler
    :rtype: object
    
    """
	# create logger object
    logger = logging.getLogger(fun_name)
    logger.propagate = True
    logger.setLevel(logging.DEBUG)
    logger.fileSetting = file
    if file:
        ch = logging.FileHandler(config_path + '/data/emhass_logger.log')
    else:
        ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, ch

def get_yaml_parse(config_path: str) -> Tuple[dict, dict, dict]:
    """
    Perform parsing of the config.yaml file.
    
    :param config_path: The path to the yaml configuration file
    :type config_path: str
    :return: A tuple with the dictionaries containing the parsed data
    :rtype: tuple(dict)

    """
    with open(config_path + '/config.yaml', 'r') as file:
        input_conf = yaml.load(file, Loader=yaml.FullLoader)
    with open(config_path + '/secrets.yaml', 'r') as file:
        input_secrets = yaml.load(file, Loader=yaml.FullLoader)
        
    retrieve_hass_conf = dict((key,d[key]) for d in input_conf['retrieve_hass_conf'] for key in d)
    retrieve_hass_conf = {**retrieve_hass_conf, **input_secrets}
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