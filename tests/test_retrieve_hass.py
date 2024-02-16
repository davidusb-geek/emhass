#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import unittest
import requests_mock
import numpy as np, pandas as pd
import pytz, pathlib, pickle, json, yaml, copy
import bz2
import pickle as cPickle

from emhass.retrieve_hass import RetrieveHass
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger

# the root folder
root = str(get_root(__file__, num_parent=2))
# create logger
logger, ch = get_logger(__name__, root, save_to_file=False)

class TestRetrieveHass(unittest.TestCase):

    def setUp(self):
        get_data_from_file = True
        save_data_to_file = False
        params = None
        retrieve_hass_conf, _, _ = get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), use_secrets=False)
        self.retrieve_hass_conf = retrieve_hass_conf
        self.rh = RetrieveHass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                               self.retrieve_hass_conf['freq'], self.retrieve_hass_conf['time_zone'],
                               params, root, logger, get_data_from_file=get_data_from_file)
        if get_data_from_file:
            with open(pathlib.Path(root+'/data/test_df_final.pkl'), 'rb') as inp:
                self.rh.df_final, self.days_list, self.var_list = pickle.load(inp)
        else:
            self.days_list = get_days_list(self.retrieve_hass_conf['days_to_retrieve'])
            self.var_list = [self.retrieve_hass_conf['var_load'], self.retrieve_hass_conf['var_PV']]
            self.rh.get_data(self.days_list, self.var_list,
                             minimal_response=False, significant_changes_only=False)
            if save_data_to_file:
                with open(pathlib.Path(root+'/data/test_df_final.pkl'), 'wb') as outp:
                    pickle.dump((self.rh.df_final, self.days_list, self.var_list), 
                                outp, pickle.HIGHEST_PROTOCOL)
        self.df_raw = self.rh.df_final.copy()
        
    def test_get_yaml_parse(self):
        with open(root+'/config_emhass.yaml', 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        params.update({
            'params_secrets': {
                'hass_url': 'http://supervisor/core/api',
                'long_lived_token': '${SUPERVISOR_TOKEN}',
                'time_zone': 'Europe/Paris',
                'lat': 45.83,
                'lon': 6.86,
                'alt': 4807.8
            }
            })
        params = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), 
                                                                    use_secrets=True, params=params)
        self.assertIsInstance(retrieve_hass_conf, dict)
        self.assertTrue('hass_url' in retrieve_hass_conf.keys())
        self.assertTrue(retrieve_hass_conf['hass_url'] == 'http://supervisor/core/api')
        self.assertIsInstance(optim_conf, dict)
        self.assertIsInstance(plant_conf, dict)
    
    def test_yaml_parse_wab_server(self):
        with open(pathlib.Path(root) / "config_emhass.yaml", 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        retrieve_hass_conf = config['retrieve_hass_conf']
        optim_conf = config['optim_conf']
        plant_conf = config['plant_conf']
        params = {}
        params['retrieve_hass_conf'] = retrieve_hass_conf
        params['optim_conf'] = optim_conf
        params['plant_conf'] = plant_conf
        # Just check forecast methods
        self.assertFalse(params['optim_conf'].get('weather_forecast_method') == None)
        self.assertFalse(params['optim_conf'].get('load_forecast_method') == None)
        self.assertFalse(params['optim_conf'].get('load_cost_forecast_method') == None)
        self.assertFalse(params['optim_conf'].get('prod_price_forecast_method') == None)

    def test_get_data_failed(self):
        days_list = get_days_list(1)
        var_list = [self.retrieve_hass_conf['var_load']]
        response = self.rh.get_data(days_list, var_list)
        self.assertFalse(response)

    def test_get_data_mock(self):
        with requests_mock.mock() as m:
            days_list = get_days_list(1)
            var_list = [self.retrieve_hass_conf['var_load']]
            data = bz2.BZ2File(str(pathlib.Path(root+'/data/test_response_get_data_get_method.pbz2')), "rb")
            data = cPickle.load(data)
            m.get(self.retrieve_hass_conf['hass_url'], json=data.json())
            self.rh.get_data(days_list, var_list,
                             minimal_response=False, significant_changes_only=False,
                             test_url=self.retrieve_hass_conf['hass_url'])
            self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
            self.assertIsInstance(self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(len(self.rh.df_final.columns), len(var_list))
            self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['freq'])
            self.assertEqual(self.rh.df_final.index.tz, datetime.timezone.utc)
        
    def test_prepare_data(self):
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertIsInstance(self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(len(self.rh.df_final.columns), len(self.var_list))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), len(self.days_list))
        self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['freq'])
        self.assertEqual(self.rh.df_final.index.tz, pytz.UTC)
        self.rh.prepare_data(self.retrieve_hass_conf['var_load'], 
                             load_negative = self.retrieve_hass_conf['load_negative'],
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['var_replace_zero'], 
                             var_interp = self.retrieve_hass_conf['var_interp'])
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), self.df_raw.index.isin(self.days_list).sum())
        self.assertEqual(len(self.rh.df_final.columns), len(self.df_raw.columns))
        self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['freq'])
        self.assertEqual(self.rh.df_final.index.tz, self.retrieve_hass_conf['time_zone'])
        
    def test_prepare_data_negative_load(self):
        self.rh.df_final[self.retrieve_hass_conf['var_load']] = -self.rh.df_final[self.retrieve_hass_conf['var_load']]
        self.rh.prepare_data(self.retrieve_hass_conf['var_load'], 
                             load_negative = True,
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['var_replace_zero'], 
                             var_interp = None)
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), self.df_raw.index.isin(self.days_list).sum())
        self.assertEqual(len(self.rh.df_final.columns), len(self.df_raw.columns))
        self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['freq'])
        self.assertEqual(self.rh.df_final.index.tz, self.retrieve_hass_conf['time_zone'])
        
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