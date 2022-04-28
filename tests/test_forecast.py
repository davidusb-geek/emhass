#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import pathlib
import pickle
import json
import copy

from emhass.retrieve_hass import retrieve_hass
from emhass.forecast import forecast
from emhass.optimization import optimization
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger
from emhass.command_line import dayahead_forecast_optim

# the root folder
root = str(get_root(__file__, num_parent=2))
# create logger
logger, ch = get_logger(__name__, root, save_to_file=False)

class TestForecast(unittest.TestCase):

    def setUp(self):
        self.get_data_from_file = True
        params = None
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), use_secrets=False)
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
        self.rh = retrieve_hass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                           self.retrieve_hass_conf['freq'], self.retrieve_hass_conf['time_zone'],
                           params, root, logger)
        if self.get_data_from_file:
            with open(pathlib.Path(root+'/data/test_df_final.pkl'), 'rb') as inp:
                self.rh.df_final, self.days_list, self.var_list = pickle.load(inp)
        else:
            self.days_list = get_days_list(self.retrieve_hass_conf['days_to_retrieve'])
            self.var_list = [self.retrieve_hass_conf['var_load'], self.retrieve_hass_conf['var_PV']]
            self.rh.get_data(self.days_list, self.var_list,
                            minimal_response=False, significant_changes_only=False)
        self.rh.prepare_data(self.retrieve_hass_conf['var_load'], load_negative = self.retrieve_hass_conf['load_negative'],
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['var_replace_zero'], 
                             var_interp = self.retrieve_hass_conf['var_interp'])
        self.df_input_data = self.rh.df_final.copy()
        
        self.fcst = forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                             params, root, logger, get_data_from_file=self.get_data_from_file)
        self.df_weather_scrap = self.fcst.get_weather_forecast(method='scrapper')
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_scrap)
        self.P_load_forecast = self.fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
        self.df_input_data_dayahead = pd.concat([self.P_PV_forecast, self.P_load_forecast], axis=1)
        self.df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        self.opt = optimization(retrieve_hass_conf, optim_conf, plant_conf, 
                       self.fcst.var_load_cost, self.fcst.var_prod_price, self.days_list, 
                       'profit', root, logger)
        self.input_data_dict = {
            'root': root,
            'retrieve_hass_conf': self.retrieve_hass_conf,
            'df_input_data': self.df_input_data,
            'df_input_data_dayahead': self.df_input_data_dayahead,
            'opt': self.opt,
            'rh': self.rh,
            'fcst': self.fcst,
            'P_PV_forecast': self.P_PV_forecast,
            'P_load_forecast': self.P_load_forecast,
            'params': params
        }
    
    def test_get_weather_forecast(self):
        self.assertTrue(self.df_input_data.isnull().sum().sum()==0)
        self.assertIsInstance(self.df_weather_scrap, type(pd.DataFrame()))
        self.assertTrue(col in self.df_weather_scrap.columns for col in ['ghi', 'dni', 'dhi', 'temp_air'])
        self.assertIsInstance(self.df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.df_weather_scrap.index.tz, self.fcst.time_zone)
        self.assertTrue(self.fcst.start_forecast < ts for ts in self.df_weather_scrap.index)
        self.assertEqual(len(self.df_weather_scrap), 
                         int(self.optim_conf['delta_forecast'].total_seconds()/3600/self.fcst.timeStep))
        print(">> The length of the wheater forecast = "+str(len(self.df_weather_scrap)))
        self.df_weather_csv = self.fcst.get_weather_forecast(method='csv')
        self.assertEqual(self.fcst.weather_forecast_method, 'csv')
        self.assertIsInstance(self.df_weather_csv, type(pd.DataFrame()))
        self.assertIsInstance(self.df_weather_csv.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.df_weather_csv.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.df_weather_csv.index.tz, self.fcst.time_zone)
        self.assertTrue(self.fcst.start_forecast < ts for ts in self.df_weather_csv.index)
        self.assertEqual(len(self.df_weather_csv), 
                         int(self.optim_conf['delta_forecast'].total_seconds()/3600/self.fcst.timeStep))
        P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_csv)
        self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_csv), len(P_PV_forecast))

    def test_get_forecasts_with_lists(self):
        with open(root+'/config_emhass.json', 'r') as read_file:
            params = json.load(read_file)
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
        params['passed_data'] = {
            'pv_power_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
            'load_power_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
            'load_cost_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
            'prod_price_forecast':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
        }
        params['optim_conf'][7]['weather_forecast_method'] = 'list'
        params['optim_conf'][8]['load_forecast_method'] = 'list'
        params['optim_conf'][9]['load_cost_forecast_method'] = 'list'
        params['optim_conf'][13]['prod_price_forecast_method'] = 'list'
        params_json = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), 
                                                                    use_secrets=False, params=params_json)
        rh = retrieve_hass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                           retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'],
                           params, root, logger)
        if self.get_data_from_file:
            with open(pathlib.Path(root+'/data/test_df_final.pkl'), 'rb') as inp:
                rh.df_final, days_list, var_list = pickle.load(inp)
        else:
            days_list = get_days_list(retrieve_hass_conf['days_to_retrieve'])
            var_list = [retrieve_hass_conf['var_load'], retrieve_hass_conf['var_PV']]
            rh.get_data(days_list, var_list,
                        minimal_response=False, significant_changes_only=False)
        rh.prepare_data(retrieve_hass_conf['var_load'], load_negative = retrieve_hass_conf['load_negative'],
                        set_zero_min = retrieve_hass_conf['set_zero_min'], 
                        var_replace_zero = retrieve_hass_conf['var_replace_zero'], 
                        var_interp = retrieve_hass_conf['var_interp'])
        df_input_data = rh.df_final.copy()
        
        fcst = forecast(retrieve_hass_conf, optim_conf, plant_conf, 
                        params_json, root, logger, get_data_from_file=True)
        df_input_data = copy.deepcopy(df_input_data).iloc[-49:-1]
        P_PV_forecast = fcst.get_weather_forecast(method='list')
        df_input_data.index = P_PV_forecast.index
        df_input_data.index.freq = rh.df_final.index.freq
        self.assertIsInstance(P_PV_forecast, type(pd.DataFrame()))
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertTrue(self.fcst.start_forecast < ts for ts in P_PV_forecast.index)
        self.assertTrue(P_PV_forecast.values[0][0] == 1)
        self.assertTrue(P_PV_forecast.values[-1][0] == 48)
        P_load_forecast = fcst.get_load_forecast(method='list')
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, fcst.time_zone)
        self.assertEqual(len(P_PV_forecast), len(P_load_forecast))
        self.assertTrue(P_load_forecast.values[0] == 1)
        self.assertTrue(P_load_forecast.values[-1] == 48)
        df_input_data = fcst.get_load_cost_forecast(df_input_data, method='list')
        self.assertTrue(fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        self.assertTrue(df_input_data['unit_load_cost'].values[0] == 1)
        self.assertTrue(df_input_data['unit_load_cost'].values[-1] == 48)
        df_input_data = fcst.get_prod_price_forecast(df_input_data, method='list')
        self.assertTrue(fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        self.assertTrue(df_input_data['unit_prod_price'].values[0] == 1)
        self.assertTrue(df_input_data['unit_prod_price'].values[-1] == 48)
        
    def test_get_power_from_weather(self):
        self.assertIsInstance(self.P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(self.P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(self.P_PV_forecast))
    
    def test_get_load_forecast(self):
        self.P_load_forecast = self.fcst.get_load_forecast()
        self.assertIsInstance(self.P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(self.P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.P_PV_forecast), len(self.P_load_forecast))
        print(">> The length of the load forecast = "+str(len(self.P_load_forecast)))
        
    def test_get_load_cost_forecast(self):
        df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.assertTrue(self.fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data, method='csv')
        self.assertTrue(self.fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        
    def test_get_prod_price_forecast(self):
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data, method='csv')
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)