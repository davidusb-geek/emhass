#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import requests_mock
import pandas as pd
import pathlib, pickle, json, copy, yaml
import bz2
import _pickle as cPickle

from emhass.retrieve_hass import RetrieveHass
from emhass.command_line import set_input_data_dict
from emhass.machine_learning_forecaster import MLForecaster
from emhass.forecast import Forecast
from emhass.optimization import Optimization
from emhass import utils

# the root folder
root = str(utils.get_root(__file__, num_parent=2))
# create logger
logger, ch = utils.get_logger(__name__, root, save_to_file=False)

class TestForecast(unittest.TestCase):
    
    @staticmethod
    def get_test_params():
        with open(root+'/config_emhass.yaml', 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        params.update({
            'params_secrets': {
                'hass_url': 'http://supervisor/core/api',
                'long_lived_token': '${SUPERVISOR_TOKEN}',
                'time_zone': 'Europe/Paris',
                'lat': 45.83,
                'lon': 6.86,
                'alt': 8000.0
            }
            })
        return params

    def setUp(self):
        self.get_data_from_file = True
        params = None
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(pathlib.Path(root) / 'config_emhass.yaml', use_secrets=False)
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = \
            retrieve_hass_conf, optim_conf, plant_conf
        self.rh = RetrieveHass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                               self.retrieve_hass_conf['freq'], self.retrieve_hass_conf['time_zone'],
                               params, root, logger)
        if self.get_data_from_file:
            with open(pathlib.Path(root) / 'data' / 'test_df_final.pkl', 'rb') as inp:
                self.rh.df_final, self.days_list, self.var_list = pickle.load(inp)
        else:
            self.days_list = utils.get_days_list(self.retrieve_hass_conf['days_to_retrieve'])
            self.var_list = [self.retrieve_hass_conf['var_load'], self.retrieve_hass_conf['var_PV']]
            self.rh.get_data(self.days_list, self.var_list,
                            minimal_response=False, significant_changes_only=False)
        self.rh.prepare_data(self.retrieve_hass_conf['var_load'], load_negative = self.retrieve_hass_conf['load_negative'],
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['var_replace_zero'], 
                             var_interp = self.retrieve_hass_conf['var_interp'])
        self.df_input_data = self.rh.df_final.copy()
        
        self.fcst = Forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                             params, root, logger, get_data_from_file=self.get_data_from_file)
        # The default for test is csv read
        self.df_weather_scrap = self.fcst.get_weather_forecast(method='csv')
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_scrap)
        self.P_load_forecast = self.fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
        self.df_input_data_dayahead = pd.concat([self.P_PV_forecast, self.P_load_forecast], axis=1)
        self.df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
        self.opt = Optimization(retrieve_hass_conf, optim_conf, plant_conf, 
                                self.fcst.var_load_cost, self.fcst.var_prod_price, 
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
    
    def test_get_weather_forecast_csv(self):
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
        df_weather_none = self.fcst.get_weather_forecast(method='none')
        self.assertTrue(df_weather_none == None)
    
    def test_get_weather_forecast_scrapper_method_mock(self):
        with requests_mock.mock() as m:
            data = bz2.BZ2File(str(pathlib.Path(root+'/data/test_response_scrapper_get_method.pbz2')), "rb")
            data = cPickle.load(data)
            get_url = "https://clearoutside.com/forecast/"+str(round(self.fcst.lat, 2))+"/"+str(round(self.fcst.lon, 2))+"?desktop=true"
            m.get(get_url, content=data)
            df_weather_scrap = self.fcst.get_weather_forecast(method='scrapper')
            self.assertIsInstance(df_weather_scrap, type(pd.DataFrame()))
            self.assertIsInstance(df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(df_weather_scrap.index.tz, self.fcst.time_zone)
            self.assertTrue(self.fcst.start_forecast < ts for ts in df_weather_scrap.index)
            self.assertEqual(len(df_weather_scrap), 
                            int(self.optim_conf['delta_forecast'].total_seconds()/3600/self.fcst.timeStep))
            P_PV_forecast = self.fcst.get_power_from_weather(df_weather_scrap)
            self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
            self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
            self.assertEqual(len(df_weather_scrap), len(P_PV_forecast))
            self.plant_conf['module_model'] = [self.plant_conf['module_model'][0], self.plant_conf['module_model'][0]]
            self.plant_conf['inverter_model'] = [self.plant_conf['inverter_model'][0], self.plant_conf['inverter_model'][0]]
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

    def test_get_weather_forecast_solcast_method_mock(self):
        with requests_mock.mock() as m:
            data = bz2.BZ2File(str(pathlib.Path(root+'/data/test_response_solcast_get_method.pbz2')), "rb")
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
                            int(self.optim_conf['delta_forecast'].total_seconds()/3600/self.fcst.timeStep))
        
    def test_get_weather_forecast_solarforecast_method_mock(self):
        with requests_mock.mock() as m:
            data = bz2.BZ2File(str(pathlib.Path(root+'/data/test_response_solarforecast_get_method.pbz2')), "rb")
            data = cPickle.load(data)
            for i in range(len(self.plant_conf['module_model'])):
                get_url = "https://api.forecast.solar/estimate/"+str(round(self.fcst.lat, 2))+"/"+str(round(self.fcst.lon, 2))+\
                    "/"+str(self.plant_conf["surface_tilt"][i])+"/"+str(self.plant_conf["surface_azimuth"][i]-180)+\
                    "/"+str(5)
                m.get(get_url, json=data)
                df_weather_solarforecast = self.fcst.get_weather_forecast(method='solar.forecast')
                self.assertIsInstance(df_weather_solarforecast, type(pd.DataFrame()))
                self.assertIsInstance(df_weather_solarforecast.index, pd.core.indexes.datetimes.DatetimeIndex)
                self.assertIsInstance(df_weather_solarforecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
                self.assertEqual(df_weather_solarforecast.index.tz, self.fcst.time_zone)
                self.assertTrue(self.fcst.start_forecast < ts for ts in df_weather_solarforecast.index)
                self.assertEqual(len(df_weather_solarforecast), 
                                int(self.optim_conf['delta_forecast'].total_seconds()/3600/self.fcst.timeStep))

    def test_get_forecasts_with_lists(self):
        with open(root+'/config_emhass.yaml', 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
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
        runtimeparams = {
            'pv_power_forecast':[i+1 for i in range(48)],
            'load_power_forecast':[i+1 for i in range(48)],
            'load_cost_forecast':[i+1 for i in range(48)],
            'prod_price_forecast':[i+1 for i in range(48)]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), 
                                                                    use_secrets=False, params=params_json)
        set_type = "dayahead-optim"
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams_json, params_json, retrieve_hass_conf, 
            optim_conf, plant_conf, set_type, logger)
        rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                          retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'],
                          params, root, logger)
        if self.get_data_from_file:
            with open(pathlib.Path(root+'/data/test_df_final.pkl'), 'rb') as inp:
                rh.df_final, days_list, var_list = pickle.load(inp)
        else:
            days_list = utils.get_days_list(retrieve_hass_conf['days_to_retrieve'])
            var_list = [retrieve_hass_conf['var_load'], retrieve_hass_conf['var_PV']]
            rh.get_data(days_list, var_list,
                        minimal_response=False, significant_changes_only=False)
        rh.prepare_data(retrieve_hass_conf['var_load'], load_negative = retrieve_hass_conf['load_negative'],
                        set_zero_min = retrieve_hass_conf['set_zero_min'], 
                        var_replace_zero = retrieve_hass_conf['var_replace_zero'], 
                        var_interp = retrieve_hass_conf['var_interp'])
        df_input_data = rh.df_final.copy()
        fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf, 
                        params_json, root, logger, get_data_from_file=True)
        df_input_data = copy.deepcopy(df_input_data).iloc[-49:-1]
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
        P_load_forecast = fcst.get_load_forecast(method='list')
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, fcst.time_zone)
        self.assertEqual(len(P_PV_forecast), len(P_load_forecast))
        self.assertTrue(P_load_forecast.values[0] == 1)
        self.assertTrue(P_load_forecast.values[-1] == 48)
        df_input_data = fcst.get_load_cost_forecast(df_input_data, method='list')
        self.assertTrue(fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        self.assertTrue(df_input_data['unit_load_cost'].values[0] == 1)
        self.assertTrue(df_input_data['unit_load_cost'].values[-1] == 48)
        df_input_data = fcst.get_prod_price_forecast(df_input_data, method='list')
        self.assertTrue(fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        self.assertTrue(df_input_data['unit_prod_price'].values[0] == 1)
        self.assertTrue(df_input_data['unit_prod_price'].values[-1] == 48)
        # Test with longer lists
        with open(root+'/config_emhass.yaml', 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
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
        runtimeparams = {
            'pv_power_forecast':[i+1 for i in range(3*48)],
            'load_power_forecast':[i+1 for i in range(3*48)],
            'load_cost_forecast':[i+1 for i in range(3*48)],
            'prod_price_forecast':[i+1 for i in range(3*48)]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), 
                                                                          use_secrets=False, params=params_json)
        optim_conf['delta_forecast'] = pd.Timedelta(days=3)
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams_json, params_json, retrieve_hass_conf, 
            optim_conf, plant_conf, set_type, logger)
        fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf, 
                        params_json, root, logger, get_data_from_file=True)
        P_PV_forecast = fcst.get_weather_forecast(method='list')
        self.assertIsInstance(P_PV_forecast, type(pd.DataFrame()))
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, fcst.time_zone)
        self.assertTrue(fcst.start_forecast < ts for ts in P_PV_forecast.index)
        self.assertTrue(P_PV_forecast.values[0][0] == 1)
        self.assertTrue(P_PV_forecast.values[-1][0] == 3*48)
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
        df_input_data_dayahead = fcst.get_load_cost_forecast(df_input_data_dayahead, method='list')
        self.assertTrue(fcst.var_load_cost in df_input_data_dayahead.columns)
        self.assertTrue(df_input_data_dayahead.isnull().sum().sum()==0)
        self.assertTrue(df_input_data_dayahead[fcst.var_load_cost].iloc[0] == 1)
        self.assertTrue(df_input_data_dayahead[fcst.var_load_cost].iloc[-1] == 3*48)
        df_input_data_dayahead = fcst.get_prod_price_forecast(df_input_data_dayahead, method='list')
        self.assertTrue(fcst.var_prod_price in df_input_data_dayahead.columns)
        self.assertTrue(df_input_data_dayahead.isnull().sum().sum()==0)
        self.assertTrue(df_input_data_dayahead[fcst.var_prod_price].iloc[0] == 1)
        self.assertTrue(df_input_data_dayahead[fcst.var_prod_price].iloc[-1] == 3*48)
        
    def test_get_forecasts_with_lists_special_case(self):
        with open(root+'/config_emhass.yaml', 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
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
        runtimeparams = {
            'load_cost_forecast':[i+1 for i in range(48)],
            'prod_price_forecast':[i+1 for i in range(48)]
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params_json = json.dumps(params)
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), 
                                                                    use_secrets=False, params=params_json)
        set_type = "dayahead-optim"
        params, retrieve_hass_conf, optim_conf, plant_conf = utils.treat_runtimeparams(
            runtimeparams_json, params_json, retrieve_hass_conf, 
            optim_conf, plant_conf, set_type, logger)
        rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                          retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'],
                          params, root, logger)
        if self.get_data_from_file:
            with open(pathlib.Path(root+'/data/test_df_final.pkl'), 'rb') as inp:
                rh.df_final, days_list, var_list = pickle.load(inp)
        else:
            days_list = utils.get_days_list(retrieve_hass_conf['days_to_retrieve'])
            var_list = [retrieve_hass_conf['var_load'], retrieve_hass_conf['var_PV']]
            rh.get_data(days_list, var_list,
                        minimal_response=False, significant_changes_only=False)
        rh.prepare_data(retrieve_hass_conf['var_load'], load_negative = retrieve_hass_conf['load_negative'],
                        set_zero_min = retrieve_hass_conf['set_zero_min'], 
                        var_replace_zero = retrieve_hass_conf['var_replace_zero'], 
                        var_interp = retrieve_hass_conf['var_interp'])
        df_input_data = rh.df_final.copy()
        fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf, 
                        params_json, root, logger, get_data_from_file=True)
        df_input_data = copy.deepcopy(df_input_data).iloc[-49:-1]
        P_PV_forecast = fcst.get_weather_forecast()
        df_input_data.index = P_PV_forecast.index
        df_input_data.index.freq = rh.df_final.index.freq
        df_input_data = fcst.get_load_cost_forecast(
            df_input_data, method='list')
        self.assertTrue(fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        self.assertTrue(df_input_data['unit_load_cost'].values[0] == 1)
        self.assertTrue(df_input_data['unit_load_cost'].values[-1] == 48)
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
        # Lets test passing a lists of PV params
        self.plant_conf['module_model'] = [self.plant_conf['module_model'], self.plant_conf['module_model']]
        self.plant_conf['inverter_model'] = [self.plant_conf['inverter_model'], self.plant_conf['inverter_model']]
        self.plant_conf['surface_tilt'] = [30, 45]
        self.plant_conf['surface_azimuth'] = [270, 90]
        self.plant_conf['modules_per_string'] = [8, 8]
        self.plant_conf['strings_per_inverter'] = [1, 1]
        self.fcst = Forecast(self.retrieve_hass_conf, self.optim_conf, self.plant_conf, 
                             None, root, logger, get_data_from_file=self.get_data_from_file)
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
                             params, root, logger, get_data_from_file=self.get_data_from_file)
        df_weather_scrap = self.fcst.get_weather_forecast(method='csv')
        P_PV_forecast = self.fcst.get_power_from_weather(df_weather_scrap, set_mix_forecast=True, df_now=df_input_data)
        self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(P_PV_forecast))
    
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
                             params, root, logger, get_data_from_file=self.get_data_from_file)
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
        
    def test_get_load_forecast_mlforecaster(self):
        params = TestForecast.get_test_params()
        params_json = json.dumps(params)
        config_path = pathlib.Path(root+'/config_emhass.yaml')
        base_path = str(config_path.parent)
        costfun = 'profit'
        action = 'forecast-model-fit' # fit, predict and tune methods
        params = copy.deepcopy(json.loads(params_json))
        runtimeparams = {
            "days_to_retrieve": 20,
            "model_type": "load_forecast",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params['passed_data'] = runtimeparams
        params['optim_conf']['load_forecast_method'] = 'mlforecaster'
        params_json = json.dumps(params)
        input_data_dict = set_input_data_dict(config_path, base_path, costfun, params_json, runtimeparams_json, 
                                              action, logger, get_data_from_file=True)
        data = copy.deepcopy(input_data_dict['df_input_data'])
        model_type = input_data_dict['params']['passed_data']['model_type']
        var_model = input_data_dict['params']['passed_data']['var_model']
        sklearn_model = input_data_dict['params']['passed_data']['sklearn_model']
        num_lags = input_data_dict['params']['passed_data']['num_lags']
        mlf = MLForecaster(data, model_type, var_model, sklearn_model, num_lags, root, logger)
        mlf.fit()
        P_load_forecast = input_data_dict['fcst'].get_load_forecast(method="mlforecaster", use_last_window=False, 
                                                                    debug=True, mlf=mlf)
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertTrue((P_load_forecast.index == self.fcst.forecast_dates).all())
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))
        
    def test_get_load_cost_forecast(self):
        df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.assertTrue(self.fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data, method='csv',
                                                         csv_path='/data/data_load_cost_forecast.csv')
        self.assertTrue(self.fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        
    def test_get_prod_price_forecast(self):
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data, method='csv',
                                                          csv_path='/data/data_load_cost_forecast.csv')
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum()==0)
        
if __name__ == '__main__':
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
