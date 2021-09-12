#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase
import pandas as pd

from emhass.forecast import forecast
from emhass.utils import get_root, get_root_2pardir, get_yaml_parse

class TestForecast(TestCase):

    def setUp(self):
        try:
            root = get_root()
            retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(root)
        except:
            root = get_root_2pardir()
            retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(root)

        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
        self.fcst = forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, root)
        
    def get_weather_forecast(self):
        self.df_weather_scrap = self.fcst.get_weather_forecast(method='scrapper')
        self.assertIsInstance(self.df_weather_scrap, type(pd.DataFrame()))
        self.assertTrue(col in self.df_weather_scrap.columns for col in ['ghi', 'dni', 'dhi', 'temp_air'])
        self.assertIsInstance(self.df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.df_weather_scrap.index.tz, self.fcst.time_zone)
        self.assertTrue(self.fcst.start_forecast < ts for ts in self.df_weather_scrap.index)
        self.assertEqual(len(self.df_weather_scrap), int(self.optim_conf['delta_forecast'].total_seconds()/3600/self.fcst.timeStep))
        print(">> The lenght of the wheater forecast = "+str(len(self.df_weather_scrap)))
        try:
            self.df_weather_pvlib = self.fcst.get_weather_forecast(method='pvlib')
        except:
            print(">> The pvlib method to get weather result in error output!")
        
    def get_power_from_weather(self):
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_scrap)
        self.assertIsInstance(self.P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(self.P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(self.P_PV_forecast))
    
    def get_load_forecast(self):
        self.P_load_forecast = self.fcst.get_load_forecast()
        self.assertIsInstance(self.P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(self.P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.P_PV_forecast), len(self.P_load_forecast))
        print(">> The lenght of the load forecast = "+str(len(self.P_load_forecast)))
        
if __name__ == '__main__':
    tf=TestForecast()
    tf.setUp()
    tf.get_weather_forecast()
    tf.get_power_from_weather()
    tf.get_load_forecast()