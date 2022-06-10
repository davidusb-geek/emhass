#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import pathlib, json, yaml, copy

from emhass.command_line import set_input_data_dict
from emhass.command_line import perfect_forecast_optim, dayahead_forecast_optim, naive_mpc_optim
from emhass.command_line import publish_data
from emhass import utils

# the root folder
root = str(utils.get_root(__file__, num_parent=2))
# create logger
logger, ch = utils.get_logger(__name__, root, save_to_file=False)

class TestCommandLineUtils(unittest.TestCase):

    def setUp(self):
        with open(root+'/config_emhass.yaml', 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        params.update({
            'params_secrets': {
                'hass_url': 'http://supervisor/core/api',
                'long_lived_token': '${SUPERVISOR_TOKEN}',
                'time_zone': 'Europe/Paris',
                'lat': 45.83,
                'lon': 6.86,
                'alt': 8000.0
            }
            })
        runtimeparams = {
            'pv_power_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
            'load_power_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
            'load_cost_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
            'prod_price_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
        }
        self.runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf'][7]['weather_forecast_method'] = 'list'
        params['optim_conf'][8]['load_forecast_method'] = 'list'
        params['optim_conf'][9]['load_cost_forecast_method'] = 'list'
        params['optim_conf'][13]['prod_price_forecast_method'] = 'list'
        self.params_json = json.dumps(params)
        
    def test_get_yaml_parse(self):
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), use_secrets=False)
        self.assertIsInstance(retrieve_hass_conf, dict)
        self.assertIsInstance(optim_conf, dict)
        self.assertIsInstance(plant_conf, dict)
        self.assertTrue(retrieve_hass_conf['alt'] == 4807.8)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), use_secrets=True, params=self.params_json)
        self.assertTrue(retrieve_hass_conf['alt'] == 8000.0)
        
    def test_get_forecast_dates(self):
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), use_secrets=True, params=self.params_json)
        freq = int(retrieve_hass_conf['freq'].seconds/60.0)
        delta_forecast = int(optim_conf['delta_forecast'].days)
        forecast_dates = utils.get_forecast_dates(freq, delta_forecast)
        self.assertIsInstance(forecast_dates, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertTrue(len(forecast_dates)==int(delta_forecast*60*24/freq))
        
    def test_treat_runtimeparams(self):
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            pathlib.Path(root+'/config_emhass.yaml'), use_secrets=True, params=self.params_json)
        set_type = 'dayahead-optim'
        params, optim_conf = utils.treat_runtimeparams(self.runtimeparams_json, self.params_json, 
                                                       retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        self.assertIsInstance(params, str)
        params = json.loads(params)
        self.assertIsInstance(params['passed_data']['pv_power_forecast'], list)
        self.assertIsInstance(params['passed_data']['load_power_forecast'], list)
        self.assertIsInstance(params['passed_data']['load_cost_forecast'], list)
        self.assertIsInstance(params['passed_data']['prod_price_forecast'], list)
        self.assertTrue(optim_conf['weather_forecast_method'] == 'list')
        self.assertTrue(optim_conf['load_forecast_method'] == 'list')
        self.assertTrue(optim_conf['load_cost_forecast_method'] == 'list')
        self.assertTrue(optim_conf['prod_price_forecast_method'] == 'list')
        set_type = 'naive-mpc-optim'
        params, optim_conf = utils.treat_runtimeparams(self.runtimeparams_json, self.params_json, 
                                                       retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        self.assertIsInstance(params, str)
        params = json.loads(params)
        self.assertTrue(params['passed_data']['prediction_horizon'] == 10)
        self.assertTrue(params['passed_data']['soc_init'] == plant_conf['SOCtarget'])
        self.assertTrue(params['passed_data']['soc_final'] == plant_conf['SOCtarget'])
        self.assertTrue(params['passed_data']['def_total_hours'] == optim_conf['def_total_hours'])
        # This will be the case when using emhass in standalone mode
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            pathlib.Path(root+'/config_emhass.yaml'), use_secrets=True, params=self.params_json)
        params = json.dumps(None)
        params, optim_conf = utils.treat_runtimeparams(self.runtimeparams_json, params, 
                                                       retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        self.assertIsInstance(params, str)
        params = json.loads(params)
        self.assertIsInstance(params['passed_data']['pv_power_forecast'], list)
        self.assertIsInstance(params['passed_data']['load_power_forecast'], list)
        self.assertIsInstance(params['passed_data']['load_cost_forecast'], list)
        self.assertIsInstance(params['passed_data']['prod_price_forecast'], list)
        self.assertTrue(optim_conf['weather_forecast_method'] == 'list')
        self.assertTrue(optim_conf['load_forecast_method'] == 'list')
        self.assertTrue(optim_conf['load_cost_forecast_method'] == 'list')
        self.assertTrue(optim_conf['prod_price_forecast_method'] == 'list')
        
    def test_set_input_data_dict(self):
        config_path = pathlib.Path(root+'/config_emhass.yaml')
        base_path = str(config_path.parent)
        costfun = 'profit'
        action = 'dayahead-optim'
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertIsInstance(input_data_dict, dict)
        self.assertTrue(input_data_dict['df_input_data'] == None)
        self.assertIsInstance(input_data_dict['df_input_data_dayahead'], pd.DataFrame)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].index.freq is not None)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].isnull().sum().sum()==0)
        action = 'publish-data'
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertTrue(input_data_dict['df_input_data'] == None)
        self.assertTrue(input_data_dict['df_input_data_dayahead'] == None)
        self.assertTrue(input_data_dict['P_PV_forecast'] == None)
        self.assertTrue(input_data_dict['P_load_forecast'] == None)
        action = 'naive-mpc-optim'
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertIsInstance(input_data_dict, dict)
        self.assertTrue(input_data_dict['df_input_data'] == None)
        self.assertIsInstance(input_data_dict['df_input_data_dayahead'], pd.DataFrame)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].index.freq is not None)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].isnull().sum().sum()==0)
        self.assertTrue(len(input_data_dict['df_input_data_dayahead'])==10) # The default value for prediction_horizon
        runtimeparams = {
            'pv_power_forecast':[1,2,3,4,5,6,7,8,9,10],
            'load_power_forecast':[1,2,3,4,5,6,7,8,9,10],
            'load_cost_forecast':[1,2,3,4,5,6,7,8,9,10],
            'prod_price_forecast':[1,2,3,4,5,6,7,8,9,10]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params = copy.deepcopy(json.loads(self.params_json))
        params['passed_data'] = runtimeparams
        params['optim_conf'][7]['weather_forecast_method'] = 'list'
        params['optim_conf'][8]['load_forecast_method'] = 'list'
        params['optim_conf'][9]['load_cost_forecast_method'] = 'list'
        params['optim_conf'][13]['prod_price_forecast_method'] = 'list'
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertIsInstance(input_data_dict, dict)
        self.assertTrue(input_data_dict['df_input_data'] == None)
        self.assertIsInstance(input_data_dict['df_input_data_dayahead'], pd.DataFrame)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].index.freq is not None)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].isnull().sum().sum()==0)
        self.assertTrue(len(input_data_dict['df_input_data_dayahead'])==10) # The default value for prediction_horizon
        
    def test_dayahead_forecast_optim(self):
        config_path = pathlib.Path(root+'/config_emhass.yaml')
        base_path = str(config_path.parent)
        costfun = 'profit'
        action = 'dayahead-optim'
        params = copy.deepcopy(json.loads(self.params_json))
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = dayahead_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum()==0)
        self.assertTrue(len(opt_res)==len(params['passed_data']['pv_power_forecast']))
        
    def test_naive_mpc_optim(self):
        config_path = pathlib.Path(root+'/config_emhass.yaml')
        base_path = str(config_path.parent)
        costfun = 'profit'
        action = 'naive-mpc-optim'
        params = copy.deepcopy(json.loads(self.params_json))
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum()==0)
        self.assertTrue(len(opt_res)==10)
        # A test similar to the docs
        runtimeparams = {"pv_power_forecast":
            [1,2,3,4,5,6,7,8,9,10], 
            "prediction_horizon":10, "soc_init":0.5,"soc_final":0.6,"def_total_hours":[1,3]}
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf'][7]['weather_forecast_method'] = 'list'
        params['optim_conf'][8]['load_forecast_method'] = 'naive'
        params['optim_conf'][9]['load_cost_forecast_method'] = 'hp_hc_periods'
        params['optim_conf'][13]['prod_price_forecast_method'] = 'constant'
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        
        opt_res = naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum()==0)
        self.assertTrue(len(opt_res)==10)
        # Test publish after passing the forecast as list
        opt_res_pub = publish_data(input_data_dict, logger)
        self.assertTrue(len(opt_res_pub)==1)
        
    def test_publish_data(self):
        config_path = pathlib.Path(root+'/config_emhass.yaml')
        base_path = str(config_path.parent)
        costfun = 'profit'
        action = 'naive-mpc-optim'
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = dayahead_forecast_optim(input_data_dict, logger, debug=True)
        opt_res_first = publish_data(input_data_dict, logger)
        self.assertTrue(len(opt_res_first)==1)
        params = copy.deepcopy(json.loads(self.params_json))
        params['retrieve_hass_conf'][8]['method_ts_round'] = 'last'
        params['optim_conf'][0]['set_use_battery'] = True
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = dayahead_forecast_optim(input_data_dict, logger, debug=True)
        opt_res_last = publish_data(input_data_dict, logger)
        self.assertTrue(len(opt_res_last)==1)
        
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
