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
            'pv_power_forecast':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89,
            1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75,
            804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'load_power_forecast':[227.66, 228.33, 254.70, 255.97, 241.44, 255.99, 239.28, 184.23, 250.35, 241.04, 238.89, 254.29,
            211.50, 209.48, 274.47, 385.9, 295.01, 246.9, 195.79, 268.3, 244.7, 220.18, 556.34, 783.07, 848.68, 464.60, 80.53, 
            302.53, 174.05, 332.86, 252.25, 179.33, 169.94, 199.88, 114.03, 245.19, 903.83, 1222.35, 1343.13, 1148.32, 428.25,
            356.49, 304.61, 239.95, 324.07, 299.97, 273.16, 328.05],
            'load_cost_forecast':[0.22, 0.32, 0.29, 0.32, 0.32, 0.27, 0.21, 0.27, 0.21, 0.21, 
            0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.21, 0.25, 0.32, 0.32, 0.29, 0.3, 0.29, 
            0.13, 0.21, 0.13, 0.11, 0.1, 0.1, 0.1, 0.1, 0.11, 0.11, 0.1, 0.13, 0.29, 0.32, 0.32, 
            0.32, 0.29, 0.33, 0.37, 0.51, 0.51, 0.37, 0.37, 0.37],
            'prod_price_forecast':[0.33, 0.44, 0.42, 0.44, 0.44, 0.39, 0.33, 0.39, 0.33, 0.33, 
            0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.33, 0.38, 0.44, 0.44, 0.42, 0.42, 0.42, 
            0.24, 0.33, 0.24, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.24, 0.42, 0.44, 
            0.45, 0.44, 0.42, 0.46, 0.5, 0.66, 0.66, 0.5, 0.5, 0.5]
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
        P_PV_forecast = fcst.get_weather_forecast(method='list')
        self.assertIsInstance(P_PV_forecast, type(pd.DataFrame()))
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertTrue(self.fcst.start_forecast < ts for ts in P_PV_forecast.index)
        P_load_forecast = fcst.get_load_forecast(method='list')
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, fcst.time_zone)
        self.assertEqual(len(P_PV_forecast), len(P_load_forecast))
        df_input_data = fcst.get_load_cost_forecast(df_input_data, method='list')
        self.assertTrue(fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        df_input_data = fcst.get_prod_price_forecast(df_input_data, method='list')
        self.assertTrue(fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        # Test the dayahead_forecast_optim from command_line
        pp = copy.deepcopy(params)
        pp['passed_data'] = {'pv_power_forecast':None,'load_power_forecast':None,'load_cost_forecast':None,'prod_price_forecast':None}
        pp['passed_data']['pv_power_forecast'] = params['passed_data']['pv_power_forecast']
        pp['passed_data']['load_power_forecast'] = params['passed_data']['load_power_forecast']
        pp['passed_data']['load_cost_forecast'] = None
        pp['passed_data']['prod_price_forecast'] = params['passed_data']['prod_price_forecast']
        pp['optim_conf'][7]['weather_forecast_method'] = 'list'
        pp['optim_conf'][8]['load_forecast_method'] = 'list'
        pp['optim_conf'][9]['load_cost_forecast_method'] = "hp_hc_periods"
        pp['optim_conf'][13]['prod_price_forecast_method'] = 'list'
        pp_json = json.dumps(pp)
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), 
                                                                    use_secrets=False, params=pp_json)
        fcst = forecast(retrieve_hass_conf, optim_conf, plant_conf, 
                        pp_json, root, logger, get_data_from_file=True)
        df_weather = fcst.get_weather_forecast(method=optim_conf['weather_forecast_method'])
        P_PV_forecast = fcst.get_power_from_weather(df_weather)
        P_load_forecast = fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
        df_input_data_dayahead = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
        df_input_data_dayahead.index.freq=df_input_data.index.freq
        df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        opt = optimization(retrieve_hass_conf, optim_conf, plant_conf, 
                           fcst.var_load_cost, fcst.var_prod_price,  
                           days_list, 'profit', root, logger)
        input_data_dict = {
                'root': root,
                'retrieve_hass_conf': retrieve_hass_conf,
                'df_input_data': rh.df_final.copy(),
                'df_input_data_dayahead': df_input_data_dayahead,
                'opt': opt,
                'rh': rh,
                'fcst': fcst,
                'P_PV_forecast': P_PV_forecast,
                'P_load_forecast': P_load_forecast,
                'params': pp
            }
        opt_res = dayahead_forecast_optim(input_data_dict, logger)
        self.assertIsInstance(opt_res, type(pd.DataFrame()))
        
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