#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import pathlib, json

from emhass import utils

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf['data_path'] = root / 'data/'
emhass_conf['root_path'] = root / 'src/emhass/'
emhass_conf['options_path'] = root / 'options.json'
emhass_conf['config_path'] = root / 'config.json'
emhass_conf['secrets_path'] = root / 'secrets_emhass(example).yaml'
emhass_conf['legacy_config_path'] = pathlib.Path(utils.get_root(__file__, num_parent=1)) / 'config_emhass.yaml' 
emhass_conf['defaults_path'] = emhass_conf['root_path']  / 'data/config_defaults.json'
emhass_conf['associations_path'] = emhass_conf['root_path']  / 'data/associations.csv'

# Create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)

class TestCommandLineUtils(unittest.TestCase):
    
    @staticmethod
    def get_test_params():
        print(emhass_conf['legacy_config_path'])
        # Build params with default config and secrets
        if emhass_conf['defaults_path'].exists():
            config = utils.build_config(emhass_conf,logger,emhass_conf['defaults_path'])
            _,secrets = utils.build_secrets(emhass_conf,logger,no_response=True)
            # Add Altitude secret manually for testing get_yaml_parse
            secrets['Altitude'] = 8000.0
            params =  utils.build_params(emhass_conf,secrets,config,logger)
        else:
            raise Exception("config_defaults. does not exist in path: "+str(emhass_conf['defaults_path'] ))
                
        return params

    def setUp(self):
        params = TestCommandLineUtils.get_test_params()
        # Add runtime parameters for forecast lists
        runtimeparams = {
            'pv_power_forecast':[i+1 for i in range(48)],
            'load_power_forecast':[i+1 for i in range(48)],
            'load_cost_forecast':[i+1 for i in range(48)],
            'prod_price_forecast':[i+1 for i in range(48)]
        }
        self.runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf']['weather_forecast_method'] = 'list'
        params['optim_conf']['load_forecast_method'] = 'list'
        params['optim_conf']['load_cost_forecast_method'] = 'list'
        params['optim_conf']['production_price_forecast_method'] = 'list'
        self.params_json = json.dumps(params)

    def test_build_config(self):
        # Test building with the different config methods
        config = {}
        params = {}
        # Test with defaults 
        config = utils.build_config(emhass_conf,logger,emhass_conf['defaults_path'])
        params = utils.build_params(emhass_conf,{},config,logger)
        self.assertTrue(params['optim_conf']['lp_solver'] == "default")
        self.assertTrue(params['optim_conf']['lp_solver_path'] == "empty")
        self.assertTrue(config['load_peak_hour_periods'] == {'period_hp_1': [{'start': '02:54'}, {'end': '15:24'}], 'period_hp_2': [{'start': '17:24'}, {'end': '20:24'}]})
        self.assertTrue(params['retrieve_hass_conf']['sensor_replace_zero'] == ['sensor.power_photovoltaics','sensor.power_load_no_var_loads'])
        # Test with config.json 
        config = utils.build_config(emhass_conf,logger,emhass_conf['defaults_path'],emhass_conf['config_path'])
        params = utils.build_params(emhass_conf,{},config,logger)
        self.assertTrue(params['optim_conf']['lp_solver'] == "default")
        self.assertTrue(params['optim_conf']['lp_solver_path'] == "empty")
        # Test with legacy config_emhass yaml
        config = utils.build_config(emhass_conf,logger,emhass_conf['defaults_path'],legacy_config_path=emhass_conf['legacy_config_path'])
        params = utils.build_params(emhass_conf,{},config,logger)
        self.assertTrue(params['retrieve_hass_conf']['sensor_replace_zero'] == ['sensor.power_photovoltaics'])
        self.assertTrue(config['load_peak_hour_periods'] == {'period_hp_1': [{'start': '02:54'}, {'end': '15:24'}], 'period_hp_2': [{'start': '17:24'}, {'end': '20:24'}]})
        self.assertTrue(params['plant_conf']['battery_charge_efficiency'] == 0.95)

        
    def test_get_yaml_parse(self):
        # Test get_yaml_parse with only secrets
        params = {}
        updated_emhass_conf, secrets = utils.build_secrets(emhass_conf,logger)
        emhass_conf.update(updated_emhass_conf)
        params.update(utils.build_params(emhass_conf, secrets, {}, logger))
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(json.dumps(params),logger)
        self.assertIsInstance(retrieve_hass_conf, dict)
        self.assertIsInstance(optim_conf, dict)
        self.assertIsInstance(plant_conf, dict)
        self.assertTrue(retrieve_hass_conf['Altitude'] == 4807.8)
        # Test get_yaml_parse with built params in get_test_params
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json,logger)
        self.assertTrue(retrieve_hass_conf['Altitude'] == 8000.0)
        
    def test_get_forecast_dates(self):
        retrieve_hass_conf, optim_conf, _ = utils.get_yaml_parse(self.params_json,logger)
        freq = int(retrieve_hass_conf['optimization_time_step'].seconds/60.0)
        delta_forecast = int(optim_conf['delta_forecast_daily'].days)
        time_zone = retrieve_hass_conf['time_zone']
        forecast_dates = utils.get_forecast_dates(freq, delta_forecast, time_zone)
        self.assertIsInstance(forecast_dates, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertTrue(len(forecast_dates)==int(delta_forecast*60*24/freq))
        
    def test_treat_runtimeparams(self):
        # Test dayahead runtime params
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json,logger)
        set_type = 'dayahead-optim'
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
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
        self.assertTrue(optim_conf['production_price_forecast_method'] == 'list')
        # Test naive MPC runtime params
        set_type = 'naive-mpc-optim'
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            self.runtimeparams_json, self.params_json,
            retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        self.assertIsInstance(params, str)
        params = json.loads(params)
        self.assertTrue(params['passed_data']['prediction_horizon'] == 10)
        self.assertTrue(params['passed_data']['soc_init'] == plant_conf['battery_target_state_of_charge'])
        self.assertTrue(params['passed_data']['soc_final'] == plant_conf['battery_target_state_of_charge'])
        self.assertTrue(params['passed_data']['operating_hours_of_each_deferrable_load'] == optim_conf['operating_hours_of_each_deferrable_load'])     
        # Test passing optimization and plant configuration parameters at runtime        
        runtimeparams = json.loads(self.runtimeparams_json)
        runtimeparams.update({'number_of_deferrable_loads':3})
        runtimeparams.update({'nominal_power_of_deferrable_loads':[3000.0, 750.0, 2500.0]})
        runtimeparams.update({'operating_hours_of_each_deferrable_load':[5, 8, 10]})
        runtimeparams.update({'treat_deferrable_load_as_semi_cont':[True, True, True]})
        runtimeparams.update({'set_deferrable_load_single_constant':[False, False, False]})
        runtimeparams.update({'weight_battery_discharge':2.0})
        runtimeparams.update({'weight_battery_charge':2.0})
        runtimeparams.update({'solcast_api_key':'yoursecretsolcastapikey'})
        runtimeparams.update({'solcast_rooftop_id':'yourrooftopid'})
        runtimeparams.update({'solar_forecast_kwp':5.0})
        runtimeparams.update({'battery_target_state_of_charge':0.4})
        runtimeparams.update({'publish_prefix':'emhass_'})
        runtimeparams.update({'custom_pv_forecast_id':'my_custom_pv_forecast_id'})
        runtimeparams.update({'custom_load_forecast_id':'my_custom_load_forecast_id'})
        runtimeparams.update({'custom_batt_forecast_id':'my_custom_batt_forecast_id'})
        runtimeparams.update({'custom_batt_soc_forecast_id':'my_custom_batt_soc_forecast_id'})
        runtimeparams.update({'custom_grid_forecast_id':'my_custom_grid_forecast_id'})
        runtimeparams.update({'custom_cost_fun_id':'my_custom_cost_fun_id'})
        runtimeparams.update({'custom_optim_status_id':'my_custom_optim_status_id'})
        runtimeparams.update({'custom_unit_load_cost_id':'my_custom_unit_load_cost_id'})
        runtimeparams.update({'custom_unit_prod_price_id':'my_custom_unit_prod_price_id'})
        runtimeparams.update({'custom_deferrable_forecast_id':'my_custom_deferrable_forecast_id'})
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json,logger)
        set_type = 'dayahead-optim'
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams, self.params_json,
            retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        self.assertIsInstance(params, str)
        params = json.loads(params)
        self.assertIsInstance(params['passed_data']['pv_power_forecast'], list)
        self.assertIsInstance(params['passed_data']['load_power_forecast'], list)
        self.assertIsInstance(params['passed_data']['load_cost_forecast'], list)
        self.assertIsInstance(params['passed_data']['prod_price_forecast'], list)
        self.assertTrue(optim_conf['number_of_deferrable_loads'] == 3)
        self.assertTrue(optim_conf['nominal_power_of_deferrable_loads'] == [3000.0, 750.0, 2500.0])
        self.assertTrue(optim_conf['operating_hours_of_each_deferrable_load'] == [5, 8, 10])
        self.assertTrue(optim_conf['treat_deferrable_load_as_semi_cont'] == [True, True, True])
        self.assertTrue(optim_conf['set_deferrable_load_single_constant'] == [False, False, False])
        self.assertTrue(optim_conf['weight_battery_discharge'] == 2.0)
        self.assertTrue(optim_conf['weight_battery_charge'] == 2.0)
        self.assertTrue(retrieve_hass_conf['solcast_api_key'] == 'yoursecretsolcastapikey')
        self.assertTrue(retrieve_hass_conf['solcast_rooftop_id'] == 'yourrooftopid')
        self.assertTrue(retrieve_hass_conf['solar_forecast_kwp'] == 5.0)
        self.assertTrue(plant_conf['battery_target_state_of_charge'] == 0.4)
        self.assertTrue(params['passed_data']['publish_prefix'] == 'emhass_')
        self.assertTrue(params['passed_data']['custom_pv_forecast_id'] == 'my_custom_pv_forecast_id')
        self.assertTrue(params['passed_data']['custom_load_forecast_id'] == 'my_custom_load_forecast_id')
        self.assertTrue(params['passed_data']['custom_batt_forecast_id'] == 'my_custom_batt_forecast_id')
        self.assertTrue(params['passed_data']['custom_batt_soc_forecast_id'] == 'my_custom_batt_soc_forecast_id')
        self.assertTrue(params['passed_data']['custom_grid_forecast_id'] == 'my_custom_grid_forecast_id')
        self.assertTrue(params['passed_data']['custom_cost_fun_id'] == 'my_custom_cost_fun_id')
        self.assertTrue(params['passed_data']['custom_optim_status_id'] == 'my_custom_optim_status_id')
        self.assertTrue(params['passed_data']['custom_unit_load_cost_id'] == 'my_custom_unit_load_cost_id')
        self.assertTrue(params['passed_data']['custom_unit_prod_price_id'] == 'my_custom_unit_prod_price_id')
        self.assertTrue(params['passed_data']['custom_deferrable_forecast_id'] == 'my_custom_deferrable_forecast_id')
    
    def test_treat_runtimeparams_failed(self):
        # Test treatment of nan values
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            'pv_power_forecast':[1,2,3,4,5,'nan',7,8,9,10],
            'load_power_forecast':[1,2,'nan',4,5,6,7,8,9,10],
            'load_cost_forecast':[1,2,3,4,5,6,7,8,'nan',10],
            'prod_price_forecast':[1,2,3,4,'nan',6,7,8,9,10]
        }
        params['passed_data'] = runtimeparams
        params['optim_conf']['weather_forecast_method'] = 'list'
        params['optim_conf']['load_forecast_method'] = 'list'
        params['optim_conf']['load_cost_forecast_method'] = 'list'
        params['optim_conf']['production_price_forecast_method'] = 'list'
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params,logger)
        set_type = 'dayahead-optim'
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams, params,
            retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        
        self.assertTrue(len([x for x in runtimeparams['pv_power_forecast'] if not str(x).isdigit()])>0)
        self.assertTrue(len([x for x in runtimeparams['load_power_forecast'] if not str(x).isdigit()])>0)
        self.assertTrue(len([x for x in runtimeparams['load_cost_forecast'] if not str(x).isdigit()])>0)
        self.assertTrue(len([x for x in runtimeparams['prod_price_forecast'] if not str(x).isdigit()])>0)
        # Test list embedded into a string
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            'pv_power_forecast':'[1,2,3,4,5,6,7,8,9,10]',
            'load_power_forecast':'[1,2,3,4,5,6,7,8,9,10]',
            'load_cost_forecast':'[1,2,3,4,5,6,7,8,9,10]',
            'prod_price_forecast':'[1,2,3,4,5,6,7,8,9,10]'
        }
        params['passed_data'] = runtimeparams
        params['optim_conf']['weather_forecast_method'] = 'list'
        params['optim_conf']['load_forecast_method'] = 'list'
        params['optim_conf']['load_cost_forecast_method'] = 'list'
        params['optim_conf']['production_price_forecast_method'] = 'list'
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params,logger)
        set_type = 'dayahead-optim'
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams, params,
            retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        self.assertIsInstance(runtimeparams['pv_power_forecast'], list)
        self.assertIsInstance(runtimeparams['load_power_forecast'], list)
        self.assertIsInstance(runtimeparams['load_cost_forecast'], list)
        self.assertIsInstance(runtimeparams['prod_price_forecast'], list)
        # Test string of numbers
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            'pv_power_forecast':'1,2,3,4,5,6,7,8,9,10',
            'load_power_forecast':'1,2,3,4,5,6,7,8,9,10',
            'load_cost_forecast':'1,2,3,4,5,6,7,8,9,10',
            'prod_price_forecast':'1,2,3,4,5,6,7,8,9,10'
        }
        params['passed_data'] = runtimeparams
        params['optim_conf']['weather_forecast_method'] = 'list'
        params['optim_conf']['load_forecast_method'] = 'list'
        params['optim_conf']['load_cost_forecast_method'] = 'list'
        params['optim_conf']['production_price_forecast_method'] = 'list'
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params,logger)
        set_type = 'dayahead-optim'
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams, params,
            retrieve_hass_conf, optim_conf, plant_conf, set_type, logger)
        self.assertIsInstance(runtimeparams['pv_power_forecast'], str)
        self.assertIsInstance(runtimeparams['load_power_forecast'], str)
        self.assertIsInstance(runtimeparams['load_cost_forecast'], str)
        self.assertIsInstance(runtimeparams['prod_price_forecast'], str)
    
    def test_build_secrets(self):
        # Test the build_secrets defaults from get_test_params()
        params = TestCommandLineUtils.get_test_params()
        expected_keys = ['retrieve_hass_conf', 'params_secrets', 'optim_conf', 'plant_conf', 'passed_data']
        for key in expected_keys:
            self.assertTrue(key in params.keys())
        self.assertTrue(params['retrieve_hass_conf']['time_zone'] == "Europe/Paris") 
        self.assertTrue(params['retrieve_hass_conf']['hass_url'] == "https://myhass.duckdns.org/")
        self.assertTrue(params['retrieve_hass_conf']['long_lived_token'] == "thatverylongtokenhere")      
        # Test Secrets from options.json
        params = {}
        secrets = {}
        _, secrets = utils.build_secrets(emhass_conf,logger,options_path=emhass_conf["options_path"],secrets_path="",no_response=True)
        params = utils.build_params(emhass_conf, secrets, {}, logger)
        for key in expected_keys:
            self.assertTrue(key in params.keys())
        self.assertTrue(params['retrieve_hass_conf']['time_zone'] == "Europe/Paris")
        self.assertTrue(params['retrieve_hass_conf']['hass_url'] == "https://myhass.duckdns.org/")
        self.assertTrue(params['retrieve_hass_conf']['long_lived_token'] == "thatverylongtokenhere")
        # Test Secrets from secrets_emhass(example).yaml
        params = {}
        secrets = {}
        _, secrets = utils.build_secrets(emhass_conf,logger,secrets_path=emhass_conf["secrets_path"])
        params = utils.build_params(emhass_conf, secrets, {}, logger)
        for key in expected_keys:
            self.assertTrue(key in params.keys())
        self.assertTrue(params['retrieve_hass_conf']['time_zone'] == "Europe/Paris")
        self.assertTrue(params['retrieve_hass_conf']['hass_url'] == "https://myhass.duckdns.org/")
        self.assertTrue(params['retrieve_hass_conf']['long_lived_token'] == "thatverylongtokenhere")
        # Test Secrets from arguments (command_line cli)
        params = {}
        secrets = {}
        _, secrets = utils.build_secrets(emhass_conf,logger,{"url":"test.url", "key":"test.key" },secrets_path="")
        logger.debug("Obtaining long_lived_token from passed argument") 
        params = utils.build_params(emhass_conf, secrets, {}, logger)
        for key in expected_keys:
            self.assertTrue(key in params.keys())
        self.assertTrue(params['retrieve_hass_conf']['time_zone'] == "Europe/Paris")
        self.assertTrue(params['retrieve_hass_conf']['hass_url'] == "test.url")
        self.assertTrue(params['retrieve_hass_conf']['long_lived_token'] == "test.key")
        
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)