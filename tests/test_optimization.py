#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase
import pandas as pd

from emhass.retrieve_hass import retrieve_hass
from emhass.optimization import optimization
from emhass.forecast import forecast
from emhass.utils import get_root, get_root_2pardir, get_yaml_parse, get_days_list

class TestOptimization(TestCase):

    def setUp(self):
        try:
            root = get_root()
            retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(root)
        except:
            root = get_root_2pardir()
            retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(root)

        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
        
        self.days_list = get_days_list(self.retrieve_hass_conf['days_to_retrieve'])
        self.var_list = [self.retrieve_hass_conf['var_load'], self.retrieve_hass_conf['var_PV']]
        
        self.rh = retrieve_hass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                           self.retrieve_hass_conf['freq'], self.retrieve_hass_conf['time_zone'])
        self.rh.get_data(self.days_list, self.var_list,
                         minimal_response=False, significant_changes_only=False)
        self.rh.prepare_data(self.retrieve_hass_conf['var_load'], load_negative = self.retrieve_hass_conf['load_negative'],
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['var_replace_zero'], 
                             var_interp = self.retrieve_hass_conf['var_interp'])
        self.df_input_data = self.rh.df_final.copy()
        
        self.fcst = forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf)
        self.df_weather = self.fcst.get_weather_forecast(method='scrapper')
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather)
        self.P_load_forecast = self.fcst.get_load_forecast()
        self.df_input_data_dayahead = pd.concat([self.P_PV_forecast, self.P_load_forecast], axis=1)
        self.df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        
        self.opt = optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, self.days_list)
        
    def test_perform_perfect_forecast_optim(self):
        self.df_input_data = self.opt.get_load_unit_cost(self.df_input_data)
        self.opt_res = self.opt.perform_perfect_forecast_optim(self.df_input_data)
        self.assertIsInstance(self.opt_res, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun' in self.opt_res.columns)
        
    def test_perform_dayahead_forecast_optim(self):
        self.df_input_data_dayahead = self.opt.get_load_unit_cost(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun' in self.opt_res_dayahead.columns)
        
    def test_publish_data(self):
        self.rh.post_data(self.P_PV_forecast, 0, 'sensor.p_pv_forecast',
                          "W", "PV Power Forecast")

if __name__ == '__main__':
    topt=TestOptimization()
    topt.setUp()
    topt.test_perform_perfect_forecast_optim()
    topt.test_perform_dayahead_forecast_optim()
    topt.test_publish_data()
