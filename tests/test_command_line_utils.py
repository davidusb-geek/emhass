#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch
import pandas as pd
import pathlib, json, copy
import numpy as np

from emhass.command_line import set_input_data_dict
from emhass.command_line import (
    perfect_forecast_optim,
    dayahead_forecast_optim,
    naive_mpc_optim,
)
from emhass.command_line import (
    forecast_model_fit,
    forecast_model_predict,
    forecast_model_tune,
    regressor_model_fit,
    regressor_model_predict,
)
from emhass.command_line import publish_data
from emhass.command_line import main
from emhass import utils

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf['data_path'] = root / 'data/'
emhass_conf['root_path'] = root / 'src/emhass/'
emhass_conf['config_path'] = root / 'config.json'
emhass_conf['defaults_path'] = emhass_conf['root_path']  / 'data/config_defaults.json'
emhass_conf['associations_path'] = emhass_conf['root_path']  / 'data/associations.csv'


# create loggerW
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)

class TestCommandLineUtils(unittest.TestCase):
    
    @staticmethod
    def get_test_params():
        # Build params with default config and secrets
        if emhass_conf['defaults_path'].exists():
            config = utils.build_config(emhass_conf,logger,emhass_conf['defaults_path'])
            _,secrets = utils.build_secrets(emhass_conf,logger,no_response=True)
            params =  utils.build_params(emhass_conf,secrets,config,logger)
        else:
            raise Exception("config_defaults. does not exist in path: "+str(emhass_conf['defaults_path'] )) 
        return params

    def setUp(self):
        params = TestCommandLineUtils.get_test_params()
        # Add runtime parameters for forecast lists
        runtimeparams = {
            'pv_power_forecast':[i+1 for i in range(48)],
            'load_power_forecast':[i+1 for i in range(48)],
            'load_cost_forecast':[i+1 for i in range(48)],
            'prod_price_forecast':[i+1 for i in range(48)]
        }
        self.runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        self.params_json = json.dumps(params)
        
    # Test input data for actions (using data from file)
    def test_set_input_data_dict(self):
        costfun = 'profit'
        # Test dayahead
        action = 'dayahead-optim'
        input_data_dict = set_input_data_dict(emhass_conf, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertIsInstance(input_data_dict, dict)
        self.assertTrue(input_data_dict['df_input_data'] == None)
        self.assertIsInstance(input_data_dict['df_input_data_dayahead'], pd.DataFrame)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].index.freq is not None)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].isnull().sum().sum()==0)
        self.assertTrue(input_data_dict['fcst'].optim_conf['weather_forecast_method']=='list')
        self.assertTrue(input_data_dict['fcst'].optim_conf['load_forecast_method']=='list')
        self.assertTrue(input_data_dict['fcst'].optim_conf['load_cost_forecast_method']=='list')
        self.assertTrue(input_data_dict['fcst'].optim_conf['production_price_forecast_method']=='list')
        # Test publish data
        action = 'publish-data'
        input_data_dict = set_input_data_dict(emhass_conf, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertTrue(input_data_dict['df_input_data'] == None)
        self.assertTrue(input_data_dict['df_input_data_dayahead'] == None)
        self.assertTrue(input_data_dict['P_PV_forecast'] == None)
        self.assertTrue(input_data_dict['P_load_forecast'] == None)
        # Test naive mpc
        action = 'naive-mpc-optim'
        input_data_dict = set_input_data_dict(emhass_conf, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertIsInstance(input_data_dict, dict)
        self.assertIsInstance(input_data_dict['df_input_data_dayahead'], pd.DataFrame)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].index.freq is not None)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].isnull().sum().sum()==0)
        self.assertTrue(len(input_data_dict['df_input_data_dayahead'])==10) # The default value for prediction_horizon
        # Test Naive mpc with a shorter forecast =
        runtimeparams = {
            'pv_power_forecast':[1,2,3,4,5,6,7,8,9,10],
            'load_power_forecast':[1,2,3,4,5,6,7,8,9,10],
            'load_cost_forecast':[1,2,3,4,5,6,7,8,9,10],
            'prod_price_forecast':[1,2,3,4,5,6,7,8,9,10]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params = copy.deepcopy(json.loads(self.params_json))
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertIsInstance(input_data_dict, dict)
        self.assertIsInstance(input_data_dict['df_input_data_dayahead'], pd.DataFrame)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].index.freq is not None)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].isnull().sum().sum()==0)
        self.assertTrue(len(input_data_dict['df_input_data_dayahead'])==10) # The default value for prediction_horizon
        # Test naive mpc with a shorter forecast and prediction horizon = 10
        action = 'naive-mpc-optim'
        runtimeparams['prediction_horizon'] = 10
        runtimeparams_json = json.dumps(runtimeparams)
        params = copy.deepcopy(json.loads(self.params_json))
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertIsInstance(input_data_dict, dict)
        self.assertIsInstance(input_data_dict['df_input_data_dayahead'], pd.DataFrame)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].index.freq is not None)
        self.assertTrue(input_data_dict['df_input_data_dayahead'].isnull().sum().sum()==0)
        self.assertTrue(len(input_data_dict['df_input_data_dayahead'])==10) # The fixed value for prediction_horizon
        # Test passing just load cost and prod price as lists
        action = 'dayahead-optim'
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            'load_cost_forecast':[i+1 for i in range(48)],
            'prod_price_forecast':[i+1 for i in range(48)]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertTrue(input_data_dict['fcst'].optim_conf['load_cost_forecast_method']=='list')
        self.assertTrue(input_data_dict['fcst'].optim_conf['production_price_forecast_method']=='list')
    
    # Test day-ahead optimization
    def test_webserver_get_injection_dict(self):
        costfun = 'profit'
        action = 'dayahead-optim'
        input_data_dict = set_input_data_dict(emhass_conf, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = dayahead_forecast_optim(input_data_dict, logger, debug=True)
        injection_dict = utils.get_injection_dict(opt_res)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict['table1'], str)
        self.assertIsInstance(injection_dict['table2'], str)
    
    # Test data formatting of dayahead optimization with load cost and prod price as lists
    def test_dayahead_forecast_optim(self):
        # Test dataframe output of profit dayahead optimization
        costfun = 'profit'
        action = 'dayahead-optim'
        params = copy.deepcopy(json.loads(self.params_json))
        input_data_dict = set_input_data_dict(emhass_conf, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = dayahead_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum()==0)
        self.assertTrue(len(opt_res)==len(params['passed_data']['pv_power_forecast']))
        # Test dayahead output, passing just load cost and prod price as runtime lists (costfun=profit)
        action = 'dayahead-optim'
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            'load_cost_forecast':[i+1 for i in range(48)],
            'prod_price_forecast':[i+1 for i in range(48)]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = dayahead_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum()==0)
        self.assertTrue(input_data_dict['fcst'].optim_conf['load_cost_forecast_method']=='list')
        self.assertTrue(input_data_dict['fcst'].optim_conf['production_price_forecast_method']=='list')
        self.assertEqual(opt_res['unit_load_cost'].values.tolist(), runtimeparams['load_cost_forecast'])
        self.assertEqual(opt_res['unit_prod_price'].values.tolist(), runtimeparams['prod_price_forecast'])
        
    # Test dataframe outpit of perfect forecast optimization
    def test_perfect_forecast_optim(self):
        costfun = 'profit'
        action = 'perfect-optim'
        input_data_dict = set_input_data_dict(emhass_conf, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = perfect_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum()==0)
        self.assertIsInstance(opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(opt_res.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue('cost_fun_'+input_data_dict["costfun"] in opt_res.columns)
        
    # Test naive mpc optimization
    def test_naive_mpc_optim(self):
        # Test mpc optimization
        costfun = 'profit'
        action = 'naive-mpc-optim'
        params = copy.deepcopy(json.loads(self.params_json))
        input_data_dict = set_input_data_dict(emhass_conf, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum()==0)
        self.assertTrue(len(opt_res)==10)
        # Test mpc optimization with runtime parameters similar to the documentation
        runtimeparams = {"pv_power_forecast":
            [1,2,3,4,5,6,7,8,9,10], 
            "prediction_horizon":10, "soc_init":0.5,"soc_final":0.6,'operating_hours_of_each_deferrable_load':[1,3],'start_timesteps_of_each_deferrable_load':[-3,0],'end_timesteps_of_each_deferrable_load':[8,0]}
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf']['weather_forecast_method'] = 'list'
        params['optim_conf']['load_forecast_method'] = 'naive'
        params['optim_conf']['load_cost_forecast_method'] = 'hp_hc_periods'
        params['optim_conf']['production_price_forecast_method'] = 'constant'
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum()==0)
        self.assertTrue(len(opt_res)==10)
        # Test publish after passing the forecast as list
        # with method_ts_round=first
        costfun = 'profit'
        action = 'naive-mpc-optim'
        params = copy.deepcopy(json.loads(self.params_json))
        params['retrieve_hass_conf']['method_ts_round'] = 'first'
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = naive_mpc_optim(input_data_dict, logger, debug=True)
        action = 'publish-data'
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, None, 
                                              action, logger, get_data_from_file=True)
        opt_res_first = publish_data(input_data_dict, logger, opt_res_latest=opt_res)
        self.assertTrue(len(opt_res_first)==1)
        # test mpc and publish with method_ts_round=last and set_use_battery=true
        action = 'naive-mpc-optim'
        params = copy.deepcopy(json.loads(self.params_json))
        params['retrieve_hass_conf']['method_ts_round'] = 'last'
        params['optim_conf']['set_use_battery'] = True
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        opt_res = naive_mpc_optim(input_data_dict, logger, debug=True)
        action = 'publish-data'
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, None, 
                                              action, logger, get_data_from_file=True)
        opt_res_last = publish_data(input_data_dict, logger, opt_res_latest=opt_res)
        self.assertTrue(len(opt_res_last)==1)
            # Reproduce when trying to publish data params=None and runtimeparams=None
            # action = 'publish-data'
            # input_data_dict = set_input_data_dict(emhass_conf, costfun, None, None, 
            #                                       action, logger, get_data_from_file=True)
            # opt_res_last = publish_data(input_data_dict, logger, opt_res_latest=opt_res)
            # self.assertTrue(len(opt_res_last)==1)
        # Check if status is published
        from datetime import datetime
        now_precise = datetime.now(input_data_dict['retrieve_hass_conf']['time_zone']).replace(second=0, microsecond=0)
        idx_closest = opt_res.index.get_indexer([now_precise], method='nearest')[0]
        custom_cost_fun_id = {"entity_id": "sensor.optim_status", "unit_of_measurement": "", "friendly_name": "EMHASS optimization status"}
        publish_prefix = ""
        response, data = input_data_dict['rh'].post_data(opt_res['optim_status'], idx_closest, 
                                        custom_cost_fun_id["entity_id"], 
                                        custom_cost_fun_id["unit_of_measurement"],
                                        custom_cost_fun_id["friendly_name"],
                                        type_var = 'optim_status',
                                        publish_prefix = publish_prefix)
        self.assertTrue(hasattr(response, '__class__'))
        self.assertTrue(data['attributes']['friendly_name'] == 'EMHASS optimization status')
    
    # Test outputs of fit, predict and tune
    def test_forecast_model_fit_predict_tune(self):
        costfun = 'profit'
        action = 'forecast-model-fit' # fit, predict and tune methods
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            'historic_days_to_retrieve': 20,
            "model_type": "load_forecast",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": '48h',
            "perform_backtest": False,
            "model_predict_publish": True,
            "model_predict_entity_id": "sensor.p_load_forecast_knn",
            "model_predict_unit_of_measurement": "W",
            "model_predict_friendly_name": "Load Power Forecast KNN regressor"
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['optim_conf']['load_forecast_method'] = 'skforecast'
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertTrue(input_data_dict['params']['passed_data']['model_type'] == 'load_forecast')
        self.assertTrue(input_data_dict['params']['passed_data']['sklearn_model'] == 'KNeighborsRegressor')
        self.assertTrue(input_data_dict['params']['passed_data']['perform_backtest'] == False)
        # Check that the default params are loaded
        input_data_dict = set_input_data_dict(emhass_conf, costfun, self.params_json, self.runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        self.assertTrue(input_data_dict['params']['passed_data']['model_type'] == 'load_forecast')
        self.assertTrue(input_data_dict['params']['passed_data']['sklearn_model'] == 'KNeighborsRegressor')
        self.assertIsInstance(input_data_dict['df_input_data'], pd.DataFrame)
        # Test the fit method
        df_fit_pred, df_fit_pred_backtest, mlf = forecast_model_fit(input_data_dict, logger, debug=True)
        self.assertIsInstance(df_fit_pred, pd.DataFrame)
        self.assertTrue(df_fit_pred_backtest == None)
        # Test ijection_dict for fit method on webui
        injection_dict = utils.get_injection_dict_forecast_model_fit(df_fit_pred, mlf)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict['figure_0'], str)
        # Test the predict method on observations following the train period
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        df_pred = forecast_model_predict(input_data_dict, logger, use_last_window=False, debug=True, mlf=mlf)
        self.assertIsInstance(df_pred, pd.Series)
        self.assertTrue(df_pred.isnull().sum().sum() == 0)
        # Now a predict using last_window
        df_pred = forecast_model_predict(input_data_dict, logger, debug=True, mlf=mlf)
        self.assertIsInstance(df_pred, pd.Series)
        self.assertTrue(df_pred.isnull().sum().sum() == 0)
        # Test the tune method
        df_pred_optim, mlf = forecast_model_tune(
            input_data_dict, logger, debug=True, mlf=mlf
        )
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertTrue(mlf.is_tuned == True)
        # Test injection_dict for tune method on webui
        injection_dict = utils.get_injection_dict_forecast_model_tune(df_fit_pred, mlf)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict["figure_0"], str)

    # Test data formatting of regressor model fit amd predict
    def test_regressor_model_fit_predict(self):
        costfun = "profit"
        action = "regressor-model-fit"  # fit and predict methods
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "AdaBoostRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "mlr_predict_entity_id": "sensor.predicted_hours_test",
            "mlr_predict_unit_of_measurement": "h",
            "mlr_predict_friendly_name": "Predicted hours",
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["model_type"]
            == "heating_hours_degreeday",
        )
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["regression_model"]
            == "AdaBoostRegression",
        )
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["csv_file"]
            == "heating_prediction.csv",
        )
        mlr = regressor_model_fit(input_data_dict, logger, debug=True)

        # def test_regressor_model_predict(self):
        costfun = "profit"
        action = "regressor-model-predict"  # predict methods
        params = TestCommandLineUtils.get_test_params()
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "AdaBoostRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "mlr_predict_entity_id": "sensor.predicted_hours_test",
            "mlr_predict_unit_of_measurement": "h",
            "mlr_predict_friendly_name": "Predicted hours",
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params["passed_data"] = runtimeparams
        params_json = json.dumps(params)

        input_data_dict = set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["model_type"]
            == "heating_hours_degreeday",
        )
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["mlr_predict_friendly_name"]
            == "Predicted hours",
        )

        regressor_model_predict(input_data_dict, logger, debug=True, mlr=mlr)

    
    # CLI test action that does not exist
    @patch('sys.argv', ['main', '--action', 'test', '--config', str(emhass_conf['config_path']), 
                        '--debug', 'True'])
    def test_main_wrong_action(self):
        opt_res = main()
        self.assertEqual(opt_res, None)
        
    # CLI test action perfect-optim action 
    @patch('sys.argv', ['main', '--action', 'perfect-optim', '--config', str(emhass_conf['config_path']), 
                        '--debug', 'True', '--params', json.dumps(get_test_params())])
    def test_main_perfect_forecast_optim(self):
        opt_res = main()
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertIsInstance(opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            opt_res.index.dtype,
            pd.core.dtypes.dtypes.DatetimeTZDtype,
        )

    # CLI test dayahead forecast optimzation action
    def test_main_dayahead_forecast_optim(self):
        with patch('sys.argv', ['main', '--action', 'dayahead-optim', '--config', str(emhass_conf['config_path']), 
                                '--params', self.params_json, '--runtimeparams', self.runtimeparams_json,
                                '--debug', 'True']):
            opt_res = main()
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)

    # CLI test naive mpc optimzation action
    def test_main_naive_mpc_optim(self):
        with patch('sys.argv', ['main', '--action', 'naive-mpc-optim', '--config', str(emhass_conf['config_path']), 
                                '--params', self.params_json, '--runtimeparams', self.runtimeparams_json,
                                '--debug', 'True']):
            opt_res = main()
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertTrue(len(opt_res) == 10)

    # CLI test forecast model fit action
    def test_main_forecast_model_fit(self):
        params = copy.deepcopy(json.loads(self.params_json))
        runtimeparams = {
            'historic_days_to_retrieve': 20,
            "model_type": "load_forecast",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": '48h',
            "perform_backtest": False
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf']['load_forecast_method'] = 'skforecast'
        params_json = json.dumps(params)
        with patch('sys.argv', ['main', '--action', 'forecast-model-fit', '--config', str(emhass_conf['config_path']), 
                                '--params', params_json, '--runtimeparams', runtimeparams_json,
                                '--debug', 'True']):
            df_fit_pred, df_fit_pred_backtest, mlf = main()
        self.assertIsInstance(df_fit_pred, pd.DataFrame)
        self.assertTrue(df_fit_pred_backtest == None)
        
    # CLI test forecast model predict action
    def test_main_forecast_model_predict(self):
        params = copy.deepcopy(json.loads(self.params_json))
        runtimeparams = {
            'historic_days_to_retrieve': 20,
            "model_type": "load_forecast",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params["passed_data"] = runtimeparams
        params["optim_conf"]['load_forecast_method'] = "skforecast"
        params_json = json.dumps(params)
        with patch('sys.argv', ['main', '--action', 'forecast-model-predict', '--config', str(emhass_conf['config_path']), 
                                '--params', params_json, '--runtimeparams', runtimeparams_json,
                                '--debug', 'True']):
            df_pred = main()
        self.assertIsInstance(df_pred, pd.Series)
        self.assertTrue(df_pred.isnull().sum().sum() == 0)

    # CLI test forecast model tune action
    def test_main_forecast_model_tune(self):
        params = copy.deepcopy(json.loads(self.params_json))
        runtimeparams = {
            'historic_days_to_retrieve': 20,
            "model_type": "load_forecast",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params["passed_data"] = runtimeparams
        params["optim_conf"]['load_forecast_method'] = "skforecast"
        params_json = json.dumps(params)
        with patch('sys.argv', ['main', '--action', 'forecast-model-tune', '--config', str(emhass_conf['config_path']), 
                                '--params', params_json, '--runtimeparams', runtimeparams_json,
                                '--debug', 'True']):
            df_pred_optim, mlf = main()
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertTrue(mlf.is_tuned == True)

    # CLI test regressor model fit action
    def test_main_regressor_model_fit(self):
        params = copy.deepcopy(json.loads(self.params_json))
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "AdaBoostRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params["passed_data"] = runtimeparams
        params_json = json.dumps(params)
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "regressor-model-fit",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            mlr = main()

    # CLI test regressor model predict action
    def test_main_regressor_model_predict(self):
        params = copy.deepcopy(json.loads(self.params_json))
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "AdaBoostRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params["passed_data"] = runtimeparams
        params["optim_conf"]['load_forecast_method'] = "skforecast"
        params_json = json.dumps(params)
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "regressor-model-predict",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            prediction = main()
        self.assertIsInstance(prediction, np.ndarray)
        
    # CLI test publish data action
    @patch('sys.argv', ['main', '--action', 'publish-data', '--config', str(emhass_conf['config_path']), 
                        '--debug', 'True'])
    def test_main_publish_data(self):
        opt_res = main()
        self.assertFalse(opt_res.empty)
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
