#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import unittest
import requests_mock 
import numpy as np, pandas as pd
import pytz, pathlib, pickle, json, copy
import bz2
import pickle as cPickle

from emhass import utils
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import get_yaml_parse, get_days_list, get_logger

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf['data_path'] = root / 'data/'
emhass_conf['root_path'] = root / 'src/emhass/'
emhass_conf['options_path'] = root / 'options.json'
emhass_conf['secrets_path'] = root / 'secrets_emhass(example).yaml'
emhass_conf['defaults_path'] = emhass_conf['root_path']  / 'data/config_defaults.json'
emhass_conf['associations_path'] = emhass_conf['root_path']  / 'data/associations.csv'

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)

class TestRetrieveHass(unittest.TestCase):

    def setUp(self):
        get_data_from_file = True
        save_data_to_file = False

        # Build params with default secrets (no config)
        if emhass_conf['defaults_path'].exists():   
            _,secrets = utils.build_secrets(emhass_conf,logger,no_response=True)
            params =  utils.build_params(emhass_conf,secrets,{},logger)
            retrieve_hass_conf, _, _ = get_yaml_parse(params,logger)
        else:
            raise Exception("config_defaults. does not exist in path: "+str(emhass_conf['defaults_path'] ))

        # Force config params for testing
        retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(30, "minutes")
        retrieve_hass_conf['sensor_power_photovoltaics'] = 'sensor.power_photovoltaics'
        retrieve_hass_conf['sensor_power_load_no_var_loads'] = 'sensor.power_load_no_var_loads'
        retrieve_hass_conf['sensor_replace_zero'] = ['sensor.power_photovoltaics']
        retrieve_hass_conf['sensor_linear_interp'] = ['sensor.power_photovoltaics','sensor.power_load_no_var_loads'] 
        retrieve_hass_conf['set_zero_min'] = True
        retrieve_hass_conf['load_negative'] = True

        self.retrieve_hass_conf = retrieve_hass_conf
        self.rh = RetrieveHass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                               self.retrieve_hass_conf['optimization_time_step'], self.retrieve_hass_conf['time_zone'],
                               params, emhass_conf, logger, get_data_from_file=get_data_from_file)
        # Obtain sensor values from saved file
        if get_data_from_file:
            with open(emhass_conf['data_path'] / 'test_df_final.pkl', 'rb') as inp:
                self.rh.df_final, self.days_list, self.var_list = pickle.load(inp)
        # Else obtain sensor values from HA
        else:
            self.days_list = get_days_list(self.retrieve_hass_conf['historic_days_to_retrieve'])
            self.var_list = [self.retrieve_hass_conf['sensor_power_load_no_var_loads'], self.retrieve_hass_conf['sensor_power_photovoltaics']]
            self.rh.get_data(self.days_list, self.var_list,
                             minimal_response=False, significant_changes_only=False)
            # Check to save updated data to file 
            if save_data_to_file:
                with open(emhass_conf['data_path'] / 'test_df_final.pkl', 'wb') as outp:
                    pickle.dump((self.rh.df_final, self.days_list, self.var_list), 
                                outp, pickle.HIGHEST_PROTOCOL)
        self.df_raw = self.rh.df_final.copy()
        
    # Check yaml parse in setUp worked
    def test_get_yaml_parse(self):
        self.assertIsInstance(self.retrieve_hass_conf, dict)
        self.assertTrue('hass_url' in self.retrieve_hass_conf.keys())
        self.assertTrue(self.retrieve_hass_conf['hass_url'] == 'https://myhass.duckdns.org/')
    
    # Check yaml parse worked   
    def test_yaml_parse_web_server(self):
        params = {}
        if emhass_conf['defaults_path'].exists():
                with emhass_conf['defaults_path'].open('r') as data:
                    defaults = json.load(data)
                    params.update(utils.build_params(emhass_conf, {}, defaults, logger))
        _, optim_conf, _ = get_yaml_parse(params,logger)
        # Just check forecast methods
        self.assertFalse(optim_conf.get('weather_forecast_method') == None)
        self.assertFalse(optim_conf.get('load_forecast_method') == None)
        self.assertFalse(optim_conf.get('load_cost_forecast_method') == None)
        self.assertFalse(optim_conf.get('production_price_forecast_method') == None)

    # Assume get_data to HA fails
    def test_get_data_failed(self):
        days_list = get_days_list(1)
        var_list = [self.retrieve_hass_conf['sensor_power_load_no_var_loads']]
        response = self.rh.get_data(days_list, var_list)
        self.assertFalse(response)

    # Test with html mock response
    def test_get_data_mock(self):
        with requests_mock.mock() as m:
            days_list = get_days_list(1)
            var_list = [self.retrieve_hass_conf['sensor_power_load_no_var_loads']]
            data = bz2.BZ2File(str(emhass_conf['data_path'] / 'test_response_get_data_get_method.pbz2'), "rb")
            data = cPickle.load(data)
            m.get(self.retrieve_hass_conf['hass_url'], json=data.json())
            self.rh.get_data(days_list, var_list,
                             minimal_response=False, significant_changes_only=False,
                             test_url=self.retrieve_hass_conf['hass_url'])
            self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
            self.assertIsInstance(self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(len(self.rh.df_final.columns), len(var_list))
            self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['optimization_time_step'])
            self.assertEqual(self.rh.df_final.index.tz, datetime.timezone.utc)
        
    
    # Check the dataframe was formatted correctly
    def test_prepare_data(self):
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertIsInstance(self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(len(self.rh.df_final.columns), len(self.var_list))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), len(self.days_list))
        self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['optimization_time_step'])
        self.assertEqual(self.rh.df_final.index.tz, datetime.timezone.utc)
        self.rh.prepare_data(self.retrieve_hass_conf['sensor_power_load_no_var_loads'], 
                             load_negative = self.retrieve_hass_conf['load_negative'],
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['sensor_replace_zero'], 
                             var_interp = self.retrieve_hass_conf['sensor_linear_interp'])
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), self.df_raw.index.isin(self.days_list).sum())
        self.assertEqual(len(self.rh.df_final.columns), len(self.df_raw.columns))
        self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['optimization_time_step'])
        self.assertEqual(self.rh.df_final.index.tz, self.retrieve_hass_conf['time_zone'])
        
    # Test negative load 
    def test_prepare_data_negative_load(self):
        self.rh.df_final[self.retrieve_hass_conf['sensor_power_load_no_var_loads']] = -self.rh.df_final[self.retrieve_hass_conf['sensor_power_load_no_var_loads']]
        self.rh.prepare_data(self.retrieve_hass_conf['sensor_power_load_no_var_loads'], 
                             load_negative = True,
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['sensor_replace_zero'], 
                             var_interp = None)
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), self.df_raw.index.isin(self.days_list).sum())
        self.assertEqual(len(self.rh.df_final.columns), len(self.df_raw.columns))
        self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['optimization_time_step'])
        self.assertEqual(self.rh.df_final.index.tz, self.retrieve_hass_conf['time_zone'])
        
    # Test publish data
    def test_publish_data(self):
        response, data = self.rh.post_data(self.df_raw[self.df_raw.columns[0]], 
                                           25, 'sensor.p_pv_forecast', "Unit", "Variable",
                                           type_var = 'power')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['state']==str(np.round(self.df_raw.loc[self.df_raw.index[25],self.df_raw.columns[0]],2)))
        self.assertTrue(data['attributes']['unit_of_measurement']=='Unit')
        self.assertTrue(data['attributes']['friendly_name']=='Variable')
        # Lets test publishing a forecast with more added attributes
        df = copy.deepcopy(self.df_raw.iloc[0:30])
        df.columns = ['P_PV', 'P_Load']
        df["P_batt"] = 1000.0
        df["SOC_opt"] = 0.5
        response, data = self.rh.post_data(df["P_PV"], 25, 'sensor.p_pv_forecast', "W", "PV Forecast",
                                           type_var = 'power')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['state']==str(np.round(df.loc[df.index[25],df.columns[0]],2)))
        self.assertTrue(data['attributes']['unit_of_measurement']=='W')
        self.assertTrue(data['attributes']['friendly_name']=='PV Forecast')
        self.assertIsInstance(data['attributes']['forecasts'], list)
        response, data = self.rh.post_data(df["P_batt"], 25, 'sensor.p_batt_forecast', "W", "Battery Power Forecast",
                                           type_var = 'batt')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['attributes']['unit_of_measurement']=='W')
        self.assertTrue(data['attributes']['friendly_name']=='Battery Power Forecast')
        response, data = self.rh.post_data(df["SOC_opt"], 25, 'sensor.SOC_forecast', "%", "Battery SOC Forecast",
                                           type_var = 'SOC')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['attributes']['unit_of_measurement']=='%')
        self.assertTrue(data['attributes']['friendly_name']=='Battery SOC Forecast')
        
    
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)