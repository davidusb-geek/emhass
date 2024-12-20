#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os
import requests_mock
import pandas as pd
import pathlib, pickle, json, copy
import bz2
import _pickle as cPickle

from emhass.retrieve_hass import RetrieveHass
from emhass.command_line import set_input_data_dict
from emhass.machine_learning_forecaster import MLForecaster
from emhass.forecast import Forecast
from emhass.optimization import Optimization
from emhass import utils

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf['data_path'] = root / 'data/'
emhass_conf['root_path'] = root / 'src/emhass/'
emhass_conf['defaults_path'] = emhass_conf['root_path']  / 'data/config_defaults.json'
emhass_conf['associations_path'] = emhass_conf['root_path']  / 'data/associations.csv'


# create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)

class TestForecast(unittest.TestCase):
    
    @staticmethod
    def get_test_params():
        params = {}
        # Build params with default config and secrets
        if emhass_conf['defaults_path'].exists():
            config = utils.build_config(emhass_conf,logger,emhass_conf['defaults_path'])
            _,secrets = utils.build_secrets(emhass_conf,logger,no_response=True)
            params =  utils.build_params(emhass_conf,secrets,config,logger)
        else:
            raise Exception("config_defaults. does not exist in path: "+str(emhass_conf['defaults_path'] ))
        return params

    def setUp(self):
        self.get_data_from_file = True
        params = json.dumps(TestForecast.get_test_params())
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params,logger)
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
        # Create RetrieveHass object
        self.rh = RetrieveHass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                               self.retrieve_hass_conf['optimization_time_step'], self.retrieve_hass_conf['time_zone'],
                               params, emhass_conf, logger)
        # Obtain sensor values from saved file
        if self.get_data_from_file:
            with open(emhass_conf['data_path'] / 'test_df_final.pkl', 'rb') as inp:
                self.rh.df_final, self.days_list, self.var_list = pickle.load(inp)
            self.retrieve_hass_conf['sensor_power_load_no_var_loads'] = str(self.var_list[0])
            self.retrieve_hass_conf['sensor_power_photovoltaics'] = str(self.var_list[1])
            self.retrieve_hass_conf['sensor_linear_interp'] = [retrieve_hass_conf['sensor_power_photovoltaics'], retrieve_hass_conf['sensor_power_load_no_var_loads']]
            self.retrieve_hass_conf['sensor_replace_zero'] = [retrieve_hass_conf['sensor_power_photovoltaics']]
        # Else obtain sensor values from HA
        else:
            self.days_list = utils.get_days_list(self.retrieve_hass_conf['historic_days_to_retrieve'])
            self.var_list = [self.retrieve_hass_conf['sensor_power_load_no_var_loads'], self.retrieve_hass_conf['sensor_power_photovoltaics']]
            self.rh.get_data(self.days_list, self.var_list,
                            minimal_response=False, significant_changes_only=False)
        # Prepare data for optimization
        self.rh.prepare_data(self.retrieve_hass_conf['sensor_power_load_no_var_loads'], load_negative = self.retrieve_hass_conf['load_negative'],
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['sensor_replace_zero'], 
                             var_interp = self.retrieve_hass_conf['sensor_linear_interp'])
        self.df_input_data = self.rh.df_final.copy()
        # Create forecast Object
        self.fcst = Forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                             params, emhass_conf, logger, get_data_from_file=self.get_data_from_file)
        # The default for test is csv read
        self.df_weather_scrap = self.fcst.get_weather_forecast(method='csv')
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_scrap)
        self.P_load_forecast = self.fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
        self.df_input_data_dayahead = pd.concat([self.P_PV_forecast, self.P_load_forecast], axis=1)
        self.df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        self.opt = Optimization(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price, 
                                'profit', emhass_conf, logger)
        # Manually create input data (from formatted parameter) dictionary
        self.input_data_dict = {
            'emhass_conf': emhass_conf,
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
    
    
    # Test weather forecast dataframe output based on saved csv file  
    def test_get_weather_forecast_csv(self):
        # Test dataframe from get weather forecast
        self.df_weather_csv = self.fcst.get_weather_forecast(method='csv')
        self.assertEqual(self.fcst.weather_forecast_method, 'csv')
        self.assertIsInstance(self.df_weather_csv, type(pd.DataFrame()))
        self.assertIsInstance(self.df_weather_csv.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.df_weather_csv.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.df_weather_csv.index.tz, self.fcst.time_zone)
        self.assertTrue(self.fcst.start_forecast < ts for ts in self.df_weather_csv.index)
        self.assertEqual(len(self.df_weather_csv), 
                         int(self.optim_conf['delta_forecast_daily'].total_seconds()/3600/self.fcst.timeStep))
        # Test dataframe from get power from weather
        P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_csv)
        self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_csv), len(P_PV_forecast))
        df_weather_none = self.fcst.get_weather_forecast(method='none')
        self.assertTrue(df_weather_none == None)
    
    # Test output weather forecast using scrapper with mock get request data
    def test_get_weather_forecast_scrapper_method_mock(self):
        with requests_mock.mock() as m:
            data = bz2.BZ2File(str(emhass_conf['data_path'] / 'test_response_scrapper_get_method.pbz2'), "rb")
            data = cPickle.load(data)
            get_url = "https://clearoutside.com/forecast/"+str(round(self.fcst.lat, 2))+"/"+str(round(self.fcst.lon, 2))+"?desktop=true"
            m.get(get_url, content=data)
            # Test dataframe output from get weather forecast
            df_weather_scrap = self.fcst.get_weather_forecast(method='scrapper')
            self.assertIsInstance(df_weather_scrap, type(pd.DataFrame()))
            self.assertIsInstance(df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(df_weather_scrap.index.tz, self.fcst.time_zone)
            self.assertTrue(self.fcst.start_forecast < ts for ts in df_weather_scrap.index)
            self.assertEqual(len(df_weather_scrap), 
                            int(self.optim_conf['delta_forecast_daily'].total_seconds()/3600/self.fcst.timeStep))
            # Test dataframe output from get power from weather forecast
            P_PV_forecast = self.fcst.get_power_from_weather(df_weather_scrap)
            self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
            self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
            self.assertEqual(len(df_weather_scrap), len(P_PV_forecast))
            # Test dataframe output from get power from weather forecast (with 2 PV plant's)
            self.plant_conf['pv_module_model'] = [self.plant_conf['pv_module_model'][0], self.plant_conf['pv_module_model'][0]]
            self.plant_conf['pv_inverter_model'] = [self.plant_conf['pv_inverter_model'][0], self.plant_conf['pv_inverter_model'][0]]
            self.plant_conf['surface_tilt'] = [30, 45]
            self.plant_conf['surface_azimuth'] = [270, 90]
            self.plant_conf['modules_per_string'] = [8, 8]
            self.plant_conf['strings_per_inverter'] = [1, 1]
            P_PV_forecast = self.fcst.get_power_from_weather(df_weather_scrap)
            self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
            self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
            self.assertEqual(len(df_weather_scrap), len(P_PV_forecast))
    
    # Test output weather forecast using Solcast with mock get request data   
    def test_get_weather_forecast_solcast_method_mock(self):
        self.fcst.params = {'passed_data': {'weather_forecast_cache': False, 'weather_forecast_cache_only': False}}
        self.fcst.retrieve_hass_conf['solcast_api_key'] = "123456"
        self.fcst.retrieve_hass_conf['solcast_rooftop_id'] =  "123456"
        if os.path.isfile(emhass_conf['data_path'] / "weather_forecast_data.pkl"): 
            os.rename(emhass_conf['data_path'] / "weather_forecast_data.pkl", emhass_conf['data_path'] / "temp_weather_forecast_data.pkl")
        with requests_mock.mock() as m:
            data = bz2.BZ2File(str(emhass_conf['data_path'] / 'test_response_solcast_get_method.pbz2'), "rb")
            data = cPickle.load(data)
            get_url = "https://api.solcast.com.au/rooftop_sites/123456/forecasts?hours=24"
            m.get(get_url, json=data.json())
            df_weather_scrap = self.fcst.get_weather_forecast(method='solcast')
            self.assertIsInstance(df_weather_scrap, type(pd.DataFrame()))
            self.assertIsInstance(df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(df_weather_scrap.index.tz, self.fcst.time_zone)
            self.assertTrue(self.fcst.start_forecast < ts for ts in df_weather_scrap.index)
            self.assertEqual(len(df_weather_scrap), 
                            int(self.optim_conf['delta_forecast_daily'].total_seconds()/3600/self.fcst.timeStep))
            if os.path.isfile(emhass_conf['data_path'] / "temp_weather_forecast_data.pkl"): 
                os.rename(emhass_conf['data_path'] / "temp_weather_forecast_data.pkl", emhass_conf['data_path'] / "weather_forecast_data.pkl")
        
    # Test output weather forecast using Forecast.Solar with mock get request data
    def test_get_weather_forecast_solarforecast_method_mock(self):
        with requests_mock.mock() as m:
            data = bz2.BZ2File(str(emhass_conf['data_path'] / 'test_response_solarforecast_get_method.pbz2'), "rb")
            data = cPickle.load(data)
            for i in range(len(self.plant_conf['pv_module_model'])):
                get_url = "https://api.forecast.solar/estimate/"+str(round(self.fcst.lat, 2))+"/"+str(round(self.fcst.lon, 2))+\
                    "/"+str(self.plant_conf['surface_tilt'][i])+"/"+str(self.plant_conf['surface_azimuth'][i]-180)+\
                    "/"+str(5)
                m.get(get_url, json=data)
                df_weather_solarforecast = self.fcst.get_weather_forecast(method='solar.forecast')
                self.assertIsInstance(df_weather_solarforecast, type(pd.DataFrame()))
                self.assertIsInstance(df_weather_solarforecast.index, pd.core.indexes.datetimes.DatetimeIndex)
                self.assertIsInstance(df_weather_solarforecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
                self.assertEqual(df_weather_solarforecast.index.tz, self.fcst.time_zone)
                self.assertTrue(self.fcst.start_forecast < ts for ts in df_weather_solarforecast.index)
                self.assertEqual(len(df_weather_solarforecast), 
                                int(self.optim_conf['delta_forecast_daily'].total_seconds()/3600/self.fcst.timeStep))

    #  Test output weather forecast using passed runtime lists   
    def test_get_forecasts_with_lists(self):
        # Load default params
        params = {}
        if emhass_conf['defaults_path'].exists():
            with emhass_conf['defaults_path'].open('r') as data:
                defaults = json.load(data)
                updated_emhass_conf, built_secrets = utils.build_secrets(emhass_conf,logger)
                emhass_conf.update(updated_emhass_conf)
                params.update(utils.build_params(emhass_conf, built_secrets, defaults, logger))
        else:
            raise Exception("config_defaults. does not exist in path: "+str(emhass_conf['defaults_path'] ))
        # Create 48 (1 day of data) long lists runtime forecasts parameters
        runtimeparams = {
            'pv_power_forecast':[i+1 for i in range(48)],
            'load_power_forecast':[i+1 for i in range(48)],
            'load_cost_forecast':[i+1 for i in range(48)],
            'prod_price_forecast':[i+1 for i in range(48)]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json,logger)
        set_type = "dayahead-optim"
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams_json, params_json, retrieve_hass_conf, 
            optim_conf, plant_conf, set_type, logger)
        # Build RetrieveHass Object
        rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                          retrieve_hass_conf['optimization_time_step'], retrieve_hass_conf['time_zone'],
                          params, emhass_conf, logger)
        # Obtain sensor values from saved file
        if self.get_data_from_file:
            with open((emhass_conf['data_path'] / 'test_df_final.pkl'), 'rb') as inp:
                rh.df_final, days_list, var_list = pickle.load(inp)
            retrieve_hass_conf['sensor_power_load_no_var_loads'] = str(self.var_list[0])
            retrieve_hass_conf['sensor_power_photovoltaics'] = str(self.var_list[1])
            retrieve_hass_conf['sensor_linear_interp'] = [retrieve_hass_conf['sensor_power_photovoltaics'], retrieve_hass_conf['sensor_power_load_no_var_loads']]
            retrieve_hass_conf['sensor_replace_zero'] = [retrieve_hass_conf['sensor_power_photovoltaics']]
        # Else obtain sensor values from HA
        else:
            days_list = utils.get_days_list(retrieve_hass_conf['historic_days_to_retrieve'])
            var_list = [retrieve_hass_conf['sensor_power_load_no_var_loads'], retrieve_hass_conf['sensor_power_photovoltaics']]
            rh.get_data(days_list, var_list,
                        minimal_response=False, significant_changes_only=False)
        # Prepare data for optimization
        rh.prepare_data(retrieve_hass_conf['sensor_power_load_no_var_loads'], load_negative = retrieve_hass_conf['load_negative'],
                        set_zero_min = retrieve_hass_conf['set_zero_min'], 
                        var_replace_zero = retrieve_hass_conf['sensor_replace_zero'], 
                        var_interp = retrieve_hass_conf['sensor_linear_interp'])
        df_input_data = rh.df_final.copy()
        # Build Forecast Object
        fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf, 
                        params, emhass_conf, logger, get_data_from_file=True)
        # Obtain only 48 rows of data and remove last column for input
        df_input_data = copy.deepcopy(df_input_data).iloc[-49:-1]
        # Get Weather forecast with list, check dataframe output
        P_PV_forecast = fcst.get_weather_forecast(method='list')
        df_input_data.index = P_PV_forecast.index
        df_input_data.index.freq = rh.df_final.index.freq
        self.assertIsInstance(P_PV_forecast, type(pd.DataFrame()))
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, fcst.time_zone)
        self.assertTrue(fcst.start_forecast < ts for ts in P_PV_forecast.index)
        self.assertTrue(P_PV_forecast.values[0][0] == 1)
        self.assertTrue(P_PV_forecast.values[-1][0] == 48)
        # Get load forecast with list, check dataframe output
        P_load_forecast = fcst.get_load_forecast(method='list')
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, fcst.time_zone)
        self.assertEqual(len(P_PV_forecast), len(P_load_forecast))
        self.assertTrue(P_load_forecast.values[0] == 1)
        self.assertTrue(P_load_forecast.values[-1] == 48)
        # Get load cost forecast with list, check dataframe output
        df_input_data = fcst.get_load_cost_forecast(df_input_data, method='list')
        self.assertTrue(fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        self.assertTrue(df_input_data['unit_load_cost'].values[0] == 1)
        self.assertTrue(df_input_data['unit_load_cost'].values[-1] == 48)
        # Get production price forecast with list, check dataframe output
        df_input_data = fcst.get_prod_price_forecast(df_input_data, method='list')
        self.assertTrue(fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        self.assertTrue(df_input_data['unit_prod_price'].values[0] == 1)
        self.assertTrue(df_input_data['unit_prod_price'].values[-1] == 48)
    
    # Test output weather forecast using longer passed runtime lists   
    def test_get_forecasts_with_longer_lists(self):
        # Load default params
        params = {}
        set_type = "dayahead-optim"
        if emhass_conf['defaults_path'].exists():
            with emhass_conf['defaults_path'].open('r') as data:
                defaults = json.load(data)
                updated_emhass_conf, built_secrets = utils.build_secrets(emhass_conf,logger)
                emhass_conf.update(updated_emhass_conf)
                params.update(utils.build_params(emhass_conf, built_secrets, defaults, logger))
        else:
            raise Exception("config_defaults. does not exist in path: "+str(emhass_conf['defaults_path'] ))
        # Create 3*48 (3 days of data) long lists runtime forecasts parameters
        runtimeparams = {
            'pv_power_forecast':[i+1 for i in range(3*48)],
            'load_power_forecast':[i+1 for i in range(3*48)],
            'load_cost_forecast':[i+1 for i in range(3*48)],
            'prod_price_forecast':[i+1 for i in range(3*48)]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json,logger)
        optim_conf['delta_forecast_daily'] = pd.Timedelta(days=3)
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams_json, params_json, retrieve_hass_conf, 
            optim_conf, plant_conf, set_type, logger)
        # Create Forecast Object
        fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf, 
                        params, emhass_conf, logger, get_data_from_file=True)
        # Get weather forecast with list, check dataframe output
        P_PV_forecast = fcst.get_weather_forecast(method='list')
        self.assertIsInstance(P_PV_forecast, type(pd.DataFrame()))
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, fcst.time_zone)
        self.assertTrue(fcst.start_forecast < ts for ts in P_PV_forecast.index)
        self.assertTrue(P_PV_forecast.values[0][0] == 1)
        self.assertTrue(P_PV_forecast.values[-1][0] == 3*48)
        # Get load forecast with list, check dataframe output
        P_load_forecast = fcst.get_load_forecast(method='list')
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, fcst.time_zone)
        self.assertEqual(len(P_PV_forecast), len(P_load_forecast))
        self.assertTrue(P_load_forecast.values[0] == 1)
        self.assertTrue(P_load_forecast.values[-1] == 3*48)
        df_input_data_dayahead = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
        df_input_data_dayahead = utils.set_df_index_freq(df_input_data_dayahead)
        df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        # Get load cost forecast with list, check dataframe output
        df_input_data_dayahead = fcst.get_load_cost_forecast(df_input_data_dayahead, method='list')
        self.assertTrue(fcst.var_load_cost in df_input_data_dayahead.columns)
        self.assertTrue(df_input_data_dayahead.isnull().sum().sum()==0)
        self.assertTrue(df_input_data_dayahead[fcst.var_load_cost].iloc[0] == 1)
        self.assertTrue(df_input_data_dayahead[fcst.var_load_cost].iloc[-1] == 3*48)
        # Get production price forecast with list, check dataframe output
        df_input_data_dayahead = fcst.get_prod_price_forecast(df_input_data_dayahead, method='list')
        self.assertTrue(fcst.var_prod_price in df_input_data_dayahead.columns)
        self.assertTrue(df_input_data_dayahead.isnull().sum().sum()==0)
        self.assertTrue(df_input_data_dayahead[fcst.var_prod_price].iloc[0] == 1)
        self.assertTrue(df_input_data_dayahead[fcst.var_prod_price].iloc[-1] == 3*48)

    # Test output values of weather forecast using passed runtime lists and saved sensor data
    def test_get_forecasts_with_lists_special_case(self):
        # Load default params
        params = {}
        if emhass_conf['defaults_path'].exists():
            config = utils.build_config(emhass_conf,logger,emhass_conf['defaults_path'])
            _,secrets = utils.build_secrets(emhass_conf,logger,no_response=True)
            params =  utils.build_params(emhass_conf,secrets,config,logger)
        else:
            raise Exception("config_defaults. does not exist in path: "+str(emhass_conf['defaults_path'] ))
        # Create 48 (1 day of data) long lists runtime forecasts parameters
        runtimeparams = {
            'load_cost_forecast':[i+1 for i in range(48)],
            'prod_price_forecast':[i+1 for i in range(48)]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json,logger)
        set_type = "dayahead-optim"
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams_json, params_json, retrieve_hass_conf, 
            optim_conf, plant_conf, set_type, logger)
        # Create RetrieveHass Object
        rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                          retrieve_hass_conf['optimization_time_step'], retrieve_hass_conf['time_zone'],
                          params, emhass_conf, logger)
        # Obtain sensor values from saved file
        if self.get_data_from_file:
            with open(emhass_conf['data_path'] / 'test_df_final.pkl', 'rb') as inp:
                rh.df_final, days_list, var_list = pickle.load(inp)
            retrieve_hass_conf['sensor_power_load_no_var_loads'] = str(self.var_list[0])
            retrieve_hass_conf['sensor_power_photovoltaics'] = str(self.var_list[1])
            retrieve_hass_conf['sensor_linear_interp'] = [retrieve_hass_conf['sensor_power_photovoltaics'], retrieve_hass_conf['sensor_power_load_no_var_loads']]
            retrieve_hass_conf['sensor_replace_zero'] = [retrieve_hass_conf['sensor_power_photovoltaics']]
        # Else obtain sensor values from HA
        else:
            days_list = utils.get_days_list(retrieve_hass_conf['historic_days_to_retrieve'])
            var_list = [retrieve_hass_conf['sensor_power_load_no_var_loads'], retrieve_hass_conf['sensor_power_photovoltaics']]
            rh.get_data(days_list, var_list,
                        minimal_response=False, significant_changes_only=False)
        # Prepare data for optimization
        rh.prepare_data(retrieve_hass_conf['sensor_power_load_no_var_loads'], load_negative = retrieve_hass_conf['load_negative'],
                        set_zero_min = retrieve_hass_conf['set_zero_min'], 
                        var_replace_zero = retrieve_hass_conf['sensor_replace_zero'], 
                        var_interp = retrieve_hass_conf['sensor_linear_interp'])
        df_input_data = rh.df_final.copy()
        # Create forecast object
        fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf, 
                        params, emhass_conf, logger, get_data_from_file=True)
        # Obtain only 48 rows of data and remove last column for input
        df_input_data = copy.deepcopy(df_input_data).iloc[-49:-1]
        # Get weather forecast with list 
        P_PV_forecast = fcst.get_weather_forecast()
        df_input_data.index = P_PV_forecast.index
        df_input_data.index.freq = rh.df_final.index.freq
        # Get load cost forecast with list, check values from output
        df_input_data = fcst.get_load_cost_forecast(
            df_input_data, method='list')
        self.assertTrue(fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        self.assertTrue(df_input_data['unit_load_cost'].values[0] == 1)
        self.assertTrue(df_input_data['unit_load_cost'].values[-1] == 48)
        # Get production price forecast with list, check values from output
        df_input_data = fcst.get_prod_price_forecast(
            df_input_data, method='list')
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
        # Test passing a lists of PV params
        self.plant_conf['pv_module_model'] = [self.plant_conf['pv_module_model'], self.plant_conf['pv_module_model']]
        self.plant_conf['pv_inverter_model'] = [self.plant_conf['pv_inverter_model'], self.plant_conf['pv_inverter_model']]
        self.plant_conf['surface_tilt'] = [30, 45]
        self.plant_conf['surface_azimuth'] = [270, 90]
        self.plant_conf['modules_per_string'] = [8, 8]
        self.plant_conf['strings_per_inverter'] = [1, 1]
        self.fcst = Forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                             None, emhass_conf, logger, get_data_from_file=self.get_data_from_file)
        df_weather_scrap = self.fcst.get_weather_forecast(method='csv')
        P_PV_forecast = self.fcst.get_power_from_weather(df_weather_scrap)
        self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(P_PV_forecast))
        # Test the mixed forecast
        params = json.dumps({'passed_data':{'alpha':0.5,'beta':0.5}})
        df_input_data = self.input_data_dict['rh'].df_final.copy()
        self.fcst = Forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                             params, emhass_conf, logger, get_data_from_file=self.get_data_from_file)
        df_weather_scrap = self.fcst.get_weather_forecast(method='csv')
        P_PV_forecast = self.fcst.get_power_from_weather(df_weather_scrap, set_mix_forecast=True, df_now=df_input_data)
        self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(P_PV_forecast))
    
    # Test dataframe output of load forecast
    def test_get_load_forecast(self):
        P_load_forecast = self.fcst.get_load_forecast()
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))
        print(">> The length of the load forecast = "+str(len(P_load_forecast)))
        # Test the mixed forecast
        params = json.dumps({'passed_data':{'alpha':0.5,'beta':0.5}})
        df_input_data = self.input_data_dict['rh'].df_final.copy()
        self.fcst = Forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                             params, emhass_conf, logger, get_data_from_file=self.get_data_from_file)
        P_load_forecast = self.fcst.get_load_forecast(set_mix_forecast=True, df_now=df_input_data)
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))
        # Test load forecast from csv
        P_load_forecast = self.fcst.get_load_forecast(method="csv")
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))
        
    # Test dataframe output of ml load forecast
    def test_get_load_forecast_mlforecaster(self):
        params = TestForecast.get_test_params()
        params_json = json.dumps(params)
        costfun = 'profit'
        action = 'forecast-model-fit' # fit, predict and tune methods
        params = copy.deepcopy(json.loads(params_json))
        # pass custom runtime parameters
        runtimeparams = {
            'historic_days_to_retrieve': 20,
            "model_type": "load_forecast",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf']['load_forecast_method'] = 'mlforecaster'
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(emhass_conf, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        data = copy.deepcopy(input_data_dict['df_input_data'])
        # Create MLForecaster Object
        model_type = input_data_dict['params']['passed_data']['model_type']
        var_model = input_data_dict['params']['passed_data']['var_model']
        sklearn_model = input_data_dict['params']['passed_data']['sklearn_model']
        num_lags = input_data_dict['params']['passed_data']['num_lags']
        mlf = MLForecaster(data, model_type, var_model, sklearn_model, num_lags, emhass_conf, logger)
        mlf.fit()
        # Get load forecast using mlforecaster 
        P_load_forecast = input_data_dict['fcst'].get_load_forecast(method="mlforecaster", use_last_window=False, 
                                                                    debug=True, mlf=mlf)
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertTrue((P_load_forecast.index == self.fcst.forecast_dates).all())
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))
        
    # Test load cost forecast dataframe output using saved csv referece file
    def test_get_load_cost_forecast(self):
        df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.assertTrue(self.fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data, method='csv',
                                                         csv_path='data_load_cost_forecast.csv')
        self.assertTrue(self.fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        
    # Test production price forecast dataframe output using saved csv referece file
    def test_get_prod_price_forecast(self):
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data, method='csv',
                                                          csv_path='data_load_cost_forecast.csv')
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
