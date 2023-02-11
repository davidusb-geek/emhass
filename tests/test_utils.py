#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import pathlib, json, yaml, copy

from emhass import utils

# the root folder
root = str(utils.get_root(__file__, num_parent=2))
# create logger
logger, ch = utils.get_logger(__name__, root, save_to_file=False)

class TestCommandLineUtils(unittest.TestCase):
    
    @staticmethod
    def get_test_params():
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
        return params

    def setUp(self):
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            'pv_power_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
            'load_power_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
            'load_cost_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
            'prod_price_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
        }
        self.runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf']['weather_forecast_method'] = 'list'
        params['optim_conf']['load_forecast_method'] = 'list'
        params['optim_conf']['load_cost_forecast_method'] = 'list'
        params['optim_conf']['prod_price_forecast_method'] = 'list'
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
        params, retrieve_hass_conf, optim_conf = utils.treat_runtimeparams(
            self.runtimeparams_json, self.params_json,
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
        params, retrieve_hass_conf, optim_conf = utils.treat_runtimeparams(
            self.runtimeparams_json, self.params_json,
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
        params, retrieve_hass_conf, optim_conf = utils.treat_runtimeparams(
            self.runtimeparams_json, params,
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
        # Test passing optimization configuration parameters at runtime 
        runtimeparams = json.loads(self.runtimeparams_json)
        runtimeparams.update({'num_def_loads':3})
        runtimeparams.update({'P_deferrable_nom':[3000.0, 750.0, 2500.0]})
        runtimeparams.update({'def_total_hours':[5, 8, 10]})
        runtimeparams.update({'treat_def_as_semi_cont':[True, True, True]})
        runtimeparams.update({'set_def_constant':[False, False, False]})
        runtimeparams.update({'solcast_api_key':'yoursecretsolcastapikey'})
        runtimeparams.update({'solcast_rooftop_id':'yourrooftopid'})
        runtimeparams.update({'solar_forecast_kwp':5.0})
        runtimeparams_json = json.dumps(runtimeparams)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            pathlib.Path(root+'/config_emhass.yaml'), use_secrets=True, params=self.params_json)
        set_type = 'dayahead-optim'
        params, retrieve_hass_conf, optim_conf = utils.treat_runtimeparams(
            runtimeparams_json, self.params_json,
            retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        self.assertIsInstance(params, str)
        params = json.loads(params)
        self.assertIsInstance(params['passed_data']['pv_power_forecast'], list)
        self.assertIsInstance(params['passed_data']['load_power_forecast'], list)
        self.assertIsInstance(params['passed_data']['load_cost_forecast'], list)
        self.assertIsInstance(params['passed_data']['prod_price_forecast'], list)
        self.assertTrue(optim_conf['num_def_loads'] == 3)
        self.assertTrue(optim_conf['P_deferrable_nom'] == [3000.0, 750.0, 2500.0])
        self.assertTrue(optim_conf['def_total_hours'] == [5, 8, 10])
        self.assertTrue(optim_conf['treat_def_as_semi_cont'] == [True, True, True])
        self.assertTrue(optim_conf['set_def_constant'] == [False, False, False])
    
    def test_treat_runtimeparams_failed(self):
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            'pv_power_forecast':[1,2,3,4,5,'nan',7,8,9,10],
            'load_power_forecast':[1,2,'nan',4,5,6,7,8,9,10],
            'load_cost_forecast':[1,2,3,4,5,6,7,8,'nan',10],
            'prod_price_forecast':[1,2,3,4,'nan',6,7,8,9,10]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf']['weather_forecast_method'] = 'list'
        params['optim_conf']['load_forecast_method'] = 'list'
        params['optim_conf']['load_cost_forecast_method'] = 'list'
        params['optim_conf']['prod_price_forecast_method'] = 'list'
        params_json = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            pathlib.Path(root+'/config_emhass.yaml'), use_secrets=True, params=params_json)
        set_type = 'dayahead-optim'
        params, retrieve_hass_conf, optim_conf = utils.treat_runtimeparams(
            runtimeparams_json, params_json,
            retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        params = json.loads(params)
        runtimeparams = json.loads(runtimeparams_json)
        self.assertTrue(len([x for x in runtimeparams['pv_power_forecast'] if not str(x).isdigit()])>0)
        self.assertTrue(len([x for x in runtimeparams['load_power_forecast'] if not str(x).isdigit()])>0)
        self.assertTrue(len([x for x in runtimeparams['load_cost_forecast'] if not str(x).isdigit()])>0)
        self.assertTrue(len([x for x in runtimeparams['prod_price_forecast'] if not str(x).isdigit()])>0)
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            'pv_power_forecast':'[1,2,3,4,5,6,7,8,9,10]',
            'load_power_forecast':'[1,2,3,4,5,6,7,8,9,10]',
            'load_cost_forecast':'[1,2,3,4,5,6,7,8,9,10]',
            'prod_price_forecast':'[1,2,3,4,5,6,7,8,9,10]'
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf']['weather_forecast_method'] = 'list'
        params['optim_conf']['load_forecast_method'] = 'list'
        params['optim_conf']['load_cost_forecast_method'] = 'list'
        params['optim_conf']['prod_price_forecast_method'] = 'list'
        params_json = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            pathlib.Path(root+'/config_emhass.yaml'), use_secrets=True, params=params_json)
        set_type = 'dayahead-optim'
        params, retrieve_hass_conf, optim_conf = utils.treat_runtimeparams(
            runtimeparams_json, params_json,
            retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        params = json.loads(params)
        runtimeparams = json.loads(runtimeparams_json)
        self.assertIsInstance(runtimeparams['pv_power_forecast'], str)
        self.assertIsInstance(runtimeparams['load_power_forecast'], str)
        self.assertIsInstance(runtimeparams['load_cost_forecast'], str)
        self.assertIsInstance(runtimeparams['prod_price_forecast'], str)
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
