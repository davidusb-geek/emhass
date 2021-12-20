#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd

from emhass.forecast import forecast
from emhass.utils import get_root, get_yaml_parse, get_logger

# the root folder
root = str(get_root(__file__, num_parent=2))
retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(root)
# create logger
logger, ch = get_logger(__name__, root, file=False)

class TestForecast(unittest.TestCase):

    def setUp(self):
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
        self.fcst = forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                             root, logger)
        self.df_weather_scrap = self.fcst.get_weather_forecast(method='scrapper')
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_scrap)
        
    def test_get_weather_forecast(self):
        self.assertIsInstance(self.df_weather_scrap, type(pd.DataFrame()))
        self.assertTrue(col in self.df_weather_scrap.columns for col in ['ghi', 'dni', 'dhi', 'temp_air'])
        self.assertIsInstance(self.df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.df_weather_scrap.index.tz, self.fcst.time_zone)
        self.assertTrue(self.fcst.start_forecast < ts for ts in self.df_weather_scrap.index)
        self.assertEqual(len(self.df_weather_scrap), int(self.optim_conf['delta_forecast'].total_seconds()/3600/self.fcst.timeStep))
        print(">> The length of the wheater forecast = "+str(len(self.df_weather_scrap)))
        try:
            self.df_weather_pvlib = self.fcst.get_weather_forecast(method='pvlib')
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
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)