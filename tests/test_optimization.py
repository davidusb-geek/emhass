#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import unittest
import pandas as pd
import numpy as np
import pathlib
import pickle
import random
from datetime import datetime, timezone

from emhass.retrieve_hass import RetrieveHass
from emhass.optimization import Optimization
from emhass.forecast import Forecast
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger,build_config, build_params, build_secrets
from pandas.testing import assert_series_equal

# The root folder
root = pathlib.Path(get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf['data_path'] = root / 'data/'
emhass_conf['root_path'] = root / 'src/emhass/'
emhass_conf['defaults_path'] = emhass_conf['root_path']  / 'data/config_defaults.json'
emhass_conf['associations_path'] = emhass_conf['root_path']  / 'data/associations.csv'

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)

class TestOptimization(unittest.TestCase):

    def setUp(self):
        get_data_from_file = True
        params = {}
        # Build params with default config and secrets
        if emhass_conf['defaults_path'].exists():
            config = build_config(emhass_conf,logger,emhass_conf['defaults_path'])
            _,secrets = build_secrets(emhass_conf,logger,no_response=True)
            params =  build_params(emhass_conf,secrets,config,logger)
        else:
            raise Exception("config_defaults. does not exist in path: "+str(emhass_conf['defaults_path'] ))
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(json.dumps(params),logger)
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
        #Build RetrieveHass object
        self.rh = RetrieveHass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                               self.retrieve_hass_conf['optimization_time_step'], self.retrieve_hass_conf['time_zone'],
                               params, emhass_conf, logger)
        # Obtain sensor values from saved file
        if get_data_from_file:
            with open(emhass_conf['data_path'] / 'test_df_final.pkl', 'rb') as inp:
                self.rh.df_final, self.days_list, self.var_list = pickle.load(inp)
            self.retrieve_hass_conf['sensor_power_load_no_var_loads'] = str(self.var_list[0])
            self.retrieve_hass_conf['sensor_power_photovoltaics'] = str(self.var_list[1])
            self.retrieve_hass_conf['sensor_linear_interp'] = [retrieve_hass_conf['sensor_power_photovoltaics'], retrieve_hass_conf['sensor_power_load_no_var_loads']]
            self.retrieve_hass_conf['sensor_replace_zero'] = [retrieve_hass_conf['sensor_power_photovoltaics']]
        # Else obtain sensor values from HA
        else:
            self.days_list = get_days_list(self.retrieve_hass_conf['historic_days_to_retrieve'])
            self.var_list = [self.retrieve_hass_conf['sensor_power_load_no_var_loads'], self.retrieve_hass_conf['sensor_power_photovoltaics']]
            self.rh.get_data(self.days_list, self.var_list,
                            minimal_response=False, significant_changes_only=False)
        # Prepare data for optimization
        self.rh.prepare_data(self.retrieve_hass_conf['sensor_power_load_no_var_loads'], load_negative = self.retrieve_hass_conf['load_negative'],
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['sensor_replace_zero'], 
                             var_interp = self.retrieve_hass_conf['sensor_linear_interp'])
        self.df_input_data = self.rh.df_final.copy()
        #Build Forecast object
        self.fcst = Forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf,
                             params, emhass_conf, logger, get_data_from_file=get_data_from_file)
        self.df_weather = self.fcst.get_weather_forecast(method='csv')
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather)
        self.P_load_forecast = self.fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
        self.df_input_data_dayahead = pd.concat([self.P_PV_forecast, self.P_load_forecast], axis=1)
        self.df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        #Build Optimization object
        self.costfun = 'profit'
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, emhass_conf, logger)
        self.df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.input_data_dict = {
            'retrieve_hass_conf': retrieve_hass_conf,
        }
        
    # Check formatting of output from perfect optimization
    def test_perform_perfect_forecast_optim(self):
        self.opt_res = self.opt.perform_perfect_forecast_optim(self.df_input_data, self.days_list)
        self.assertIsInstance(self.opt_res, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_'+self.costfun in self.opt_res.columns)
        
    
    def test_perform_dayahead_forecast_optim(self):
        # Check formatting of output from dayahead optimization
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_'+self.costfun in self.opt_res_dayahead.columns)
        self.assertTrue(self.opt_res_dayahead['P_deferrable0'].sum()*(
            self.retrieve_hass_conf['optimization_time_step'].seconds/3600) == self.optim_conf['nominal_power_of_deferrable_loads'][0]*self.optim_conf['operating_hours_of_each_deferrable_load'][0])
        # Test the battery, dynamics and grid exchange contraints
        self.optim_conf.update({'set_use_battery': True})
        self.optim_conf.update({'set_nocharge_from_grid': True})
        self.optim_conf.update({'set_battery_dynamic': True})
        self.optim_conf.update({'set_nodischarge_to_grid': True})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, emhass_conf, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertTrue('P_batt' in self.opt_res_dayahead.columns)
        self.assertTrue('SOC_opt' in self.opt_res_dayahead.columns)
        self.assertAlmostEqual(self.opt_res_dayahead.loc[self.opt_res_dayahead.index[-1],'SOC_opt'], self.plant_conf['battery_target_state_of_charge'])
        # Test table conversion
        opt_res = pd.read_csv(emhass_conf['data_path'] / 'opt_res_latest.csv', index_col='timestamp')
        cost_cols = [i for i in opt_res.columns if 'cost_' in i]
        table = opt_res[cost_cols].reset_index().sum(numeric_only=True).to_frame(name='Cost Totals').reset_index()
        self.assertTrue(table.columns[0]=='index')
        self.assertTrue(table.columns[1]=='Cost Totals')
        # Check status
        self.assertTrue('optim_status' in self.opt_res_dayahead.columns)
        # Test treat_def_as_semi_cont and set_def_constant constraints
        self.optim_conf.update({'treat_deferrable_load_as_semi_cont': [True, True]})
        self.optim_conf.update({'set_deferrable_load_single_constant': [True, True]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, emhass_conf, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        self.optim_conf.update({'treat_deferrable_load_as_semi_cont': [False, True]})
        self.optim_conf.update({'set_deferrable_load_single_constant': [True, True]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, emhass_conf, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        self.optim_conf.update({'treat_deferrable_load_as_semi_cont': [False, True]})
        self.optim_conf.update({'set_deferrable_load_single_constant': [False, True]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, emhass_conf, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        self.optim_conf.update({'treat_deferrable_load_as_semi_cont': [False, False]})
        self.optim_conf.update({'set_deferrable_load_single_constant': [False, True]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, emhass_conf, logger)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        self.optim_conf.update({'treat_deferrable_load_as_semi_cont': [False, False]})
        self.optim_conf.update({'set_deferrable_load_single_constant': [False, False]})
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, emhass_conf, logger)
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
                                self.costfun, emhass_conf, logger)
        
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
        
    
    # Check formatting of output from dayahead optimization in self-consumption
    def test_perform_dayahead_forecast_optim_costfun_selfconsumption(self):
        costfun = 'self-consumption'
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                costfun, emhass_conf, logger)
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_selfcons' in self.opt_res_dayahead.columns)
        
    # Check formatting of output from dayahead optimization in cost
    def test_perform_dayahead_forecast_optim_costfun_cost(self):
        costfun = 'cost'
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                costfun, emhass_conf, logger)
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_cost' in self.opt_res_dayahead.columns)
        
    #
    def test_perform_dayahead_forecast_optim_aux(self):
        self.optim_conf['treat_deferrable_load_as_semi_cont'] = [False, False]
        self.optim_conf['set_total_pv_sell'] = True
        self.optim_conf['set_deferrable_load_single_constant'] = [True, True]
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, emhass_conf, logger)
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        # Test dayahead optimization using different solvers
        import pulp as pl
        solver_list = pl.listSolvers(onlyAvailable=True)
        for solver in solver_list:
            self.optim_conf['lp_solver'] = solver
            if os.getenv('lp_solver_path', default=None) == None:
                self.optim_conf['lp_solver_path'] = os.getenv('lp_solver_path', default=None)            
            self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                    self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                    self.costfun, emhass_conf, logger)
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
                                self.costfun, emhass_conf, logger)
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
        term1 = self.optim_conf['nominal_power_of_deferrable_loads'][0]*def_total_hours[0]
        term2 = self.opt_res_dayahead['P_deferrable0'].sum()*(self.retrieve_hass_conf['optimization_time_step'].seconds/3600)
        self.assertTrue(np.abs(term1-term2)<1e-3)
        #
        soc_init = 0.8
        soc_final = 0.5
        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead, self.P_PV_forecast, self.P_load_forecast, prediction_horizon,
            soc_init=soc_init, soc_final=soc_final, def_total_hours=def_total_hours, def_start_timestep=def_start_timestep, def_end_timestep=def_end_timestep)
        self.assertAlmostEqual(self.opt_res_dayahead.loc[self.opt_res_dayahead.index[-1],'SOC_opt'], soc_final)
    
    # Test format output of dayahead optimization with a thermal deferrable load
    def test_thermal_load_optim(self):
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead['outdoor_temperature_forecast'] = [random.normalvariate(10.0, 3.0) for _ in range(48)]
        runtimeparams = {
            'def_load_config': [
                {},
                {'thermal_config': {
                    'heating_rate': 5.0,
                    'cooling_constant': 0.1,
                    'overshoot_temperature': 24.0,
                    'start_temperature': 20,
                    'desired_temperatures': [21]*48,
                    }
                }
            ]
        }
        self.optim_conf["def_load_config"] = runtimeparams['def_load_config']
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price,  
                                self.costfun, emhass_conf, logger)
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values # €/kWh
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values # €/kWh
        self.opt_res_dayahead = self.opt.perform_optimization(self.df_input_data_dayahead, 
                                                              self.P_PV_forecast.values.ravel(), 
                                                              self.P_load_forecast.values.ravel(), 
                                                              unit_load_cost, unit_prod_price)
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_'+self.costfun in self.opt_res_dayahead.columns)
        self.assertTrue(self.opt.optim_status == 'Optimal')
        
    # Setup function to run dayahead optimization for the following tests
    def run_penalty_test_forecast(self):
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf,
                                self.fcst.var_load_cost, self.fcst.var_prod_price,
                                self.costfun, emhass_conf, logger)
        def_total_hours = [5 * self.retrieve_hass_conf['optimization_time_step'].seconds / 3600.0]
        def_start_timestep = [0]
        def_end_timestep = [0]
        prediction_horizon = 10
        self.optim_conf.update({'number_of_deferrable_loads': 1})

        self.fcst.params["passed_data"]["prod_price_forecast"] = [0 for i in range(prediction_horizon)]
        self.fcst.params["passed_data"]["solar_forecast_kwp"] = [
            0 for i in range(prediction_horizon)
        ]
        self.fcst.params["passed_data"]["prediction_horizon"] = prediction_horizon

        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(
            self.df_input_data_dayahead, method="list"
        )
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(
            self.df_input_data_dayahead, method="list"
        )

        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.P_PV_forecast,
            self.P_load_forecast,
            prediction_horizon,
            def_total_hours=def_total_hours,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep
        )

    # Test load is constant
    def test_constant_load(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [2,1,1,1,1,1.5,1.1,2,2,2]
        self.optim_conf.update({'set_deferrable_load_single_constant': [True]})

        self.run_penalty_test_forecast()

        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf['nominal_power_of_deferrable_loads'][0]
            * pd.Series(
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], index=self.opt_res_dayahead.index
            ),
            check_names=False,
        )

    # Test no startup penalty when bump is small
    def test_startup_penalty_continuous_with_small_bump(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [2,1,1,1,1,1.5,1.1,2,2,2]
        self.optim_conf.update({'set_deferrable_startup_penalty': [100.0]})

        self.run_penalty_test_forecast()

        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf['nominal_power_of_deferrable_loads'][0]
            * pd.Series(
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], index=self.opt_res_dayahead.index
            ),
            check_names=False,
        )

    # Test startup penalty
    def test_startup_penalty_discontinuity_when_justified(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [2,1,1,1,1,1.5,1.1,2,2,2]

        self.optim_conf.update({'set_deferrable_startup_penalty': [0.1]})

        self.run_penalty_test_forecast()

        assert_series_equal(self.opt_res_dayahead["P_deferrable0"],
                        self.optim_conf['nominal_power_of_deferrable_loads'][0] *
                        pd.Series([0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
                                  index=self.opt_res_dayahead.index),
                        check_names=False)

    # Test penalty continuity when deferrable load is already on
    def test_startup_penalty_no_discontinuity_at_start(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [1.2,1,1,1,1,1.1,2,2,2,2]

        self.optim_conf.update({
            'set_deferrable_startup_penalty': [100.0],
            "def_current_state": [True],
        })

        self.run_penalty_test_forecast()

        assert_series_equal(self.opt_res_dayahead["P_deferrable0"],
                        self.optim_conf['nominal_power_of_deferrable_loads'][0] *
                        pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  index=self.opt_res_dayahead.index),
                        check_names=False)

    # Test delay start
    def test_startup_penalty_delayed_start(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [1.2,1,1,1,1,1.1,2,2,2,2]

        self.optim_conf.update(
            {
                'set_deferrable_startup_penalty': [100.0],
                "def_current_state": [False],
            }
        )

        self.run_penalty_test_forecast()

        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf['nominal_power_of_deferrable_loads'][0]
            * pd.Series(
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], index=self.opt_res_dayahead.index
            ),
            check_names=False,
        )


if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
