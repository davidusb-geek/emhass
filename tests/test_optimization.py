#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import pandas as pd
import numpy as np
import pathlib
import pickle
from datetime import datetime, timezone

from emhass.retrieve_hass import RetrieveHass
from emhass.optimization import Optimization
from emhass.forecast import Forecast
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger

# the root folder
root = str(get_root(__file__, num_parent=2))
# create logger
logger, ch = get_logger(__name__, root, save_to_file=False)

class TestOptimization(unittest.TestCase):

    def setUp(self):
        get_data_from_file = True
        params = None
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), use_secrets=False)
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
        self.rh = RetrieveHass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                               self.retrieve_hass_conf['freq'], self.retrieve_hass_conf['time_zone'],
                               params, root, logger)
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
        
        self.fcst = Forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf,
                             params, root, logger, get_data_from_file=get_data_from_file)
        self.df_weather = self.fcst.get_weather_forecast(method='csv')
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather)
        self.P_load_forecast = self.fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
        self.df_input_data_dayahead = pd.concat([self.P_PV_forecast, self.P_load_forecast], axis=1)
        self.df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        
        self.costfun = 'profit'
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        self.df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.input_data_dict = {
            'retrieve_hass_conf': retrieve_hass_conf,
        }
        
    def test_perform_perfect_forecast_optim(self):
        self.opt_res = self.opt.perform_perfect_forecast_optim(self.df_input_data, self.days_list)
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
        self.assertTrue(self.opt_res_dayahead['P_deferrable0'].sum()*(
            self.retrieve_hass_conf['freq'].seconds/3600) == self.optim_conf['P_deferrable_nom'][0]*self.optim_conf['def_total_hours'][0])
        # Test the battery, dynamics and grid exchange contraints
        self.optim_conf.update({'set_use_battery': True})
        self.optim_conf.update({'set_nocharge_from_grid': True})
        self.optim_conf.update({'set_battery_dynamic': True})
        self.optim_conf.update({'set_nodischarge_to_grid': True})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertTrue('P_batt' in self.opt_res_dayahead.columns)
        self.assertTrue('SOC_opt' in self.opt_res_dayahead.columns)
        self.assertAlmostEqual(self.opt_res_dayahead.loc[self.opt_res_dayahead.index[-1],'SOC_opt'], self.plant_conf['SOCtarget'])
        # Test table conversion
        opt_res = pd.read_csv(root+'/data/opt_res_latest.csv', index_col='timestamp')
        cost_cols = [i for i in opt_res.columns if 'cost_' in i]
        table = opt_res[cost_cols].reset_index().sum(numeric_only=True).to_frame(name='Cost Totals').reset_index()
        self.assertTrue(table.columns[0]=='index')
        self.assertTrue(table.columns[1]=='Cost Totals')
        # Check status
        self.assertTrue('optim_status' in self.opt_res_dayahead.columns)
        # Test treat_def_as_semi_cont and set_def_constant constraints
        self.optim_conf.update({'treat_def_as_semi_cont': [True, True]})
        self.optim_conf.update({'set_def_constant': [True, True]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        self.optim_conf.update({'treat_def_as_semi_cont': [False, True]})
        self.optim_conf.update({'set_def_constant': [True, True]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        self.optim_conf.update({'treat_def_as_semi_cont': [False, True]})
        self.optim_conf.update({'set_def_constant': [False, True]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        self.optim_conf.update({'treat_def_as_semi_cont': [False, False]})
        self.optim_conf.update({'set_def_constant': [False, True]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        self.optim_conf.update({'treat_def_as_semi_cont': [False, False]})
        self.optim_conf.update({'set_def_constant': [False, False]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        # Test with different default solver, debug mode and batt SOC conditions
        del self.optim_conf['lp_solver']
        del self.optim_conf['lp_solver_path']
        self.optim_conf['set_use_battery'] = True
        soc_init = None
        soc_final = 0.3
        self.optim_conf['set_total_pv_sell'] = True
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values
        self.opt_res_dayahead = self.opt.perform_optimization(
            self.df_input_data_dayahead, self.P_PV_forecast.values.ravel(), 
            self.P_load_forecast.values.ravel(), unit_load_cost, unit_prod_price,
            soc_init = soc_init, soc_final = soc_final, debug = True)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_'+self.costfun in self.opt_res_dayahead.columns)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        
    def test_perform_dayahead_forecast_optim_costfun_selfconso(self):
        costfun = 'self-consumption'
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                costfun, root, logger)
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_selfcons' in self.opt_res_dayahead.columns)
        
    def test_perform_dayahead_forecast_optim_costfun_cost(self):
        costfun = 'cost'
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                costfun, root, logger)
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_cost' in self.opt_res_dayahead.columns)
        
    def test_perform_dayahead_forecast_optim_aux(self):
        self.optim_conf['treat_def_as_semi_cont'] = [False, False]
        self.optim_conf['set_total_pv_sell'] = True
        self.optim_conf['set_def_constant'] = [True, True]
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        import pulp as pl
        solver_list = pl.listSolvers(onlyAvailable=True)
        for solver in solver_list:
            self.optim_conf['lp_solver'] = solver
            if os.getenv("LP_SOLVER_PATH", default=None) == None:
                self.optim_conf['lp_solver_path'] = os.getenv("LP_SOLVER_PATH", default=None)            
            self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                    self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                    self.costfun, root, logger)
            self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
            self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
            self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
                self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
            self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
            self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        
    def test_perform_naive_mpc_optim(self):
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        # Test the battery
        self.optim_conf.update({'set_use_battery': True})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, root, logger)
        prediction_horizon = 10
        soc_init = 0.4
        soc_final = 0.6
        def_total_hours = [2, 3]
        def_start_timestep = [-5, 0]
        def_end_timestep = [4, 0]
        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast, prediction_horizon,
            soc_init=soc_init, soc_final=soc_final, def_total_hours=def_total_hours, def_start_timestep=def_start_timestep, def_end_timestep=def_end_timestep)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertTrue('P_batt' in self.opt_res_dayahead.columns)
        self.assertTrue('SOC_opt' in self.opt_res_dayahead.columns)
        self.assertTrue(np.abs(self.opt_res_dayahead.loc[self.opt_res_dayahead.index[-1],'SOC_opt']-soc_final)<1e-3)
        term1 = self.optim_conf['P_deferrable_nom'][0]*def_total_hours[0]
        term2 = self.opt_res_dayahead['P_deferrable0'].sum()*(self.retrieve_hass_conf['freq'].seconds/3600)
        self.assertTrue(np.abs(term1-term2)<1e-3)
        soc_init = 0.8
        soc_final = 0.5
        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast, prediction_horizon,
            soc_init=soc_init, soc_final=soc_final, def_total_hours=def_total_hours, def_start_timestep=def_start_timestep, def_end_timestep=def_end_timestep)
        self.assertAlmostEqual(self.opt_res_dayahead.loc[self.opt_res_dayahead.index[-1],'SOC_opt'], soc_final)
        
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
