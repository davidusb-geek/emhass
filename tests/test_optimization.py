#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import pathlib
import pickle

from emhass.retrieve_hass import retrieve_hass
from emhass.optimization import optimization
from emhass.forecast import forecast
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger

# the root folder
root = str(get_root(__file__, num_parent=2))
# create logger
logger, ch = get_logger(__name__, root, file=False)

class TestOptimization(unittest.TestCase):

    def setUp(self):
        get_data_from_file = True
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), use_secrets=False)
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
        self.rh = retrieve_hass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                           self.retrieve_hass_conf['freq'], self.retrieve_hass_conf['time_zone'],
                           root, logger)
        if get_data_from_file:
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
                             root, logger, get_data_from_file=get_data_from_file)
        self.df_weather = self.fcst.get_weather_forecast(method=optim_conf['weather_forecast_method'])
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather)
        self.P_load_forecast = self.fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
        self.df_input_data_dayahead = pd.concat([self.P_PV_forecast, self.P_load_forecast], axis=1)
        self.df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        
        self.costfun = 'profit'
        self.opt = optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.days_list, self.costfun, root, logger)
        self.df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        
    def test_perform_perfect_forecast_optim(self):
        self.opt_res = self.opt.perform_perfect_forecast_optim(self.df_input_data)
        self.assertIsInstance(self.opt_res, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_'+self.costfun in self.opt_res.columns)
        
    def test_perform_dayahead_forecast_optim(self):
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_'+self.costfun in self.opt_res_dayahead.columns)
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
