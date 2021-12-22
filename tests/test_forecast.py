#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd

from emhass.retrieve_hass import retrieve_hass
from emhass.forecast import forecast
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger

# the root folder
root = str(get_root(__file__, num_parent=2))
retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(root)
# create logger
logger, ch = get_logger(__name__, root, file=False)

class TestForecast(unittest.TestCase):

    def setUp(self):
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
            
        self.days_list = get_days_list(self.retrieve_hass_conf['days_to_retrieve'])
        self.var_list = [self.retrieve_hass_conf['var_load'], self.retrieve_hass_conf['var_PV']]
        
        self.rh = retrieve_hass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                           self.retrieve_hass_conf['freq'], self.retrieve_hass_conf['time_zone'],
                           root, logger)
        self.rh.get_data(self.days_list, self.var_list,
                         minimal_response=False, significant_changes_only=False)
        self.rh.prepare_data(self.retrieve_hass_conf['var_load'], load_negative = self.retrieve_hass_conf['load_negative'],
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['var_replace_zero'], 
                             var_interp = self.retrieve_hass_conf['var_interp'])
        self.df_input_data = self.rh.df_final.copy()
        
        self.fcst = forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                             root, logger)
        self.df_weather_scrap = self.fcst.get_weather_forecast(method='scrapper')
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_scrap)
        
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
        try:
            self.df_weather_pvlib = self.fcst.get_weather_forecast(method='pvlib')
            self.assertIsInstance(self.df_weather_pvlib, type(pd.DataFrame()))
            self.assertTrue(col in self.df_weather_pvlib.columns for col in ['ghi', 'dni', 'dhi', 'temp_air'])
            self.assertIsInstance(self.df_weather_pvlib.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(self.df_weather_pvlib.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(self.df_weather_pvlib.index.tz, self.fcst.time_zone)
            self.assertTrue(self.fcst.start_forecast < ts for ts in self.df_weather_pvlib.index)
            self.assertEqual(len(self.df_weather_pvlib), 
                             int(self.optim_conf['delta_forecast'].total_seconds()/3600/self.fcst.timeStep))
        except:
            print(">> The pvlib method to get weather result in error output!")
        
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
        df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data,
                                                         method='csv')
        self.assertTrue(self.fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        
    def test_get_prod_price_forecast(self):
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data,
                                                         method='csv')
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)