#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import os
import pickle
import copy
import logging
import json
from typing import Optional
import bz2
import pickle as cPickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from requests import get
from bs4 import BeautifulSoup
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.irradiance import disc

from emhass.retrieve_hass import RetrieveHass
from emhass.machine_learning_forecaster import MLForecaster
from emhass.utils import get_days_list, set_df_index_freq


class Forecast(object):
    r"""
    Generate weather, load and costs forecasts needed as inputs to the optimization.
    
    In EMHASS we have basically 4 forecasts to deal with:
    
    - PV power production forecast (internally based on the weather forecast and the
      characteristics of your PV plant). This is given in Watts.
    
    - Load power forecast: how much power your house will demand on the next 24h. This
      is given in Watts.
    
    - PV production selling price forecast: at what price are you selling your excess
      PV production on the next 24h. This is given in EUR/kWh.
    
    - Load cost forecast: the price of the energy from the grid on the next 24h. This
      is given in EUR/kWh.
    
    There are methods that are generalized to the 4 forecast needed. For all there
    forecasts it is possible to pass the data either as a passed list of values or by
    reading from a CSV file. With these methods it is then possible to use data from
    external forecast providers.
    
    Then there are the methods that are specific to each type of forecast and that 
    proposed forecast treated and generated internally by this EMHASS forecast class.
    For the weather forecast a first method (`scrapper`) uses a scrapping to the 
    ClearOutside webpage which proposes detailed forecasts based on Lat/Lon locations. 
    This method seems stable but as with any scrape method it will fail if any changes 
    are made to the webpage API. Another method (`solcast`) is using the SolCast PV 
    production forecast service. A final method (`solar.forecast`) is using another 
    external service: Solar.Forecast, for which just the nominal PV peak installed 
    power should be provided. Search the forecast section on the documentation for examples 
    on how to implement these different methods.
    
    The `get_power_from_weather` method is proposed here to convert from irradiance
    data to electrical power. The PVLib module is used to model the PV plant.
    
    The specific methods for the load forecast are a first method (`naive`) that uses 
    a naive approach, also called persistance. It simply assumes that the forecast for 
    a future period will be equal to the observed values in a past period. The past 
    period is controlled using parameter `delta_forecast`. A second method (`mlforecaster`)
    uses an internal custom forecasting model using machine learning. There is a section
    in the documentation explaining how to use this method.
    
    .. note:: This custom machine learning model is introduced from v0.4.0. EMHASS \
        proposed this new `mlforecaster` class with `fit`, `predict` and `tune` methods. \
        Only the `predict` method is used here to generate new forecasts, but it is \
        necessary to previously fit a forecaster model and it is a good idea to \
        optimize the model hyperparameters using the `tune` method. See the dedicated \
        section in the documentation for more help.
    
    For the PV production selling price and Load cost forecasts the privileged method
    is a direct read from a user provided list of values. The list should be passed
    as a runtime parameter during the `curl` to the EMHASS API.
    
    I reading from a CSV file, it should contain no header and the timestamped data 
    should have the following format:
    
    2021-04-29 00:00:00+00:00,287.07
    
    2021-04-29 00:30:00+00:00,274.27
    
    2021-04-29 01:00:00+00:00,243.38
    
    ...
    
    The data columns in these files will correspond to the data in the units expected
    for each forecasting method.
    
    """

    def __init__(self, retrieve_hass_conf: dict, optim_conf: dict, plant_conf: dict, 
                 params: str, emhass_conf: dict, logger: logging.Logger, 
                 opt_time_delta: Optional[int] = 24,
                 get_data_from_file: Optional[bool] = False) -> None:
        """
        Define constructor for the forecast class.
        
        :param retrieve_hass_conf: Dictionary containing the needed configuration
            data from the configuration file, specific to retrieve data from HASS
        :type retrieve_hass_conf: dict
        :param optim_conf: Dictionary containing the needed configuration
            data from the configuration file, specific for the optimization task
        :type optim_conf: dict
        :param plant_conf: Dictionary containing the needed configuration
            data from the configuration file, specific for the modeling of the PV plant
        :type plant_conf: dict
        :param params: Configuration parameters passed from data/options.json
        :type params: str
        :param emhass_conf: Dictionary containing the needed emhass paths
        :type emhass_conf: dict
        :param logger: The passed logger object
        :type logger: logging object
        :param opt_time_delta: The time delta in hours used to generate forecasts, 
            a value of 24 will generate 24 hours of forecast data, defaults to 24
        :type opt_time_delta: int, optional
        :param get_data_from_file: Select if data should be retrieved from a 
            previously saved pickle useful for testing or directly from connection to
            hass database
        :type get_data_from_file: bool, optional

        """
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf
        self.freq = self.retrieve_hass_conf['optimization_time_step']
        self.time_zone = self.retrieve_hass_conf['time_zone']
        self.method_ts_round = self.retrieve_hass_conf['method_ts_round']
        self.timeStep = self.freq.seconds/3600 # in hours
        self.time_delta = pd.to_timedelta(opt_time_delta, "hours")
        self.var_PV = self.retrieve_hass_conf['sensor_power_photovoltaics']
        self.var_load = self.retrieve_hass_conf['sensor_power_load_no_var_loads']
        self.var_load_new = self.var_load+'_positive'
        self.lat = self.retrieve_hass_conf['Latitude'] 
        self.lon = self.retrieve_hass_conf['Longitude']
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.get_data_from_file = get_data_from_file
        self.var_load_cost = 'unit_load_cost'
        self.var_prod_price = 'unit_prod_price'
        if (params == None) or (params == "null"):
            self.params = {}
        elif type(params) is dict:
            self.params = params
        else:
            self.params = json.loads(params)
        if self.method_ts_round == 'nearest':
            self.start_forecast = pd.Timestamp(datetime.now(), tz=self.time_zone).replace(microsecond=0)
        elif self.method_ts_round == 'first':
            self.start_forecast = pd.Timestamp(datetime.now(), tz=self.time_zone).replace(microsecond=0).floor(freq=self.freq)
        elif self.method_ts_round == 'last':
            self.start_forecast = pd.Timestamp(datetime.now(), tz=self.time_zone).replace(microsecond=0).ceil(freq=self.freq)
        else:
            self.logger.error("Wrong method_ts_round passed parameter")
        self.end_forecast = (self.start_forecast + self.optim_conf['delta_forecast_daily']).replace(microsecond=0)
        self.forecast_dates = pd.date_range(start=self.start_forecast, 
                                            end=self.end_forecast-self.freq, 
                                            freq=self.freq, tz=self.time_zone).tz_convert('utc').round(self.freq, ambiguous='infer', nonexistent='shift_forward').tz_convert(self.time_zone)
        if params is not None:
            if 'prediction_horizon' in list(self.params['passed_data'].keys()):
                if self.params['passed_data']['prediction_horizon'] is not None:
                    self.forecast_dates = self.forecast_dates[0:self.params['passed_data']['prediction_horizon']]
        
        
    def get_weather_forecast(self, method: Optional[str] = 'scrapper',
                             csv_path: Optional[str] = "data_weather_forecast.csv") -> pd.DataFrame:
        r"""
        Get and generate weather forecast data.
        
        :param method: The desired method, options are 'scrapper', 'csv', 'list', 'solcast' and \
            'solar.forecast'. Defaults to 'scrapper'.
        :type method: str, optional
        :return: The DataFrame containing the forecasted data
        :rtype: pd.DataFrame
        
        """
        csv_path  = self.emhass_conf['data_path'] / csv_path
        w_forecast_cache_path = os.path.abspath(self.emhass_conf['data_path'] / "weather_forecast_data.pkl")

        self.logger.info("Retrieving weather forecast data using method = "+method)
        self.weather_forecast_method = method # Saving this attribute for later use to identify csv method usage
        if method == 'scrapper':
            freq_scrap = pd.to_timedelta(60, "minutes") # The scrapping time step is 60min on clearoutside
            forecast_dates_scrap = pd.date_range(start=self.start_forecast,
                                                 end=self.end_forecast-freq_scrap, 
                                                 freq=freq_scrap, tz=self.time_zone).tz_convert('utc').round(freq_scrap, ambiguous='infer', nonexistent='shift_forward').tz_convert(self.time_zone)
            # Using the clearoutside webpage
            response = get("https://clearoutside.com/forecast/"+str(round(self.lat, 2))+"/"+str(round(self.lon, 2))+"?desktop=true")
            '''import bz2 # Uncomment to save a serialized data for tests
            import _pickle as cPickle
            with bz2.BZ2File("data/test_response_scrapper_get_method.pbz2", "w") as f: 
                cPickle.dump(response.content, f)'''
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find_all(id='day_0')[0]
            list_names = table.find_all(class_='fc_detail_label')
            list_tables = table.find_all('ul')[1:]
            selected_cols = [0, 1, 2, 3, 10, 12, 15] # Selected variables
            col_names = [list_names[i].get_text() for i in selected_cols]
            list_tables = [list_tables[i] for i in selected_cols]
            # Building the raw DF container
            raw_data = pd.DataFrame(index=range(len(forecast_dates_scrap)), columns=col_names, dtype=float)
            for count_col, col in enumerate(col_names):
                list_rows = list_tables[count_col].find_all('li')
                for count_row, row in enumerate(list_rows):
                    raw_data.loc[count_row, col] = float(row.get_text())
            # Treating index
            raw_data.set_index(forecast_dates_scrap, inplace=True)
            raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
            raw_data = raw_data.reindex(self.forecast_dates)
            raw_data.interpolate(method='linear', axis=0, limit=None, 
                                 limit_direction='both', inplace=True)
            # Converting the cloud cover into Global Horizontal Irradiance with a PVLib method
            ghi_est = self.cloud_cover_to_irradiance(raw_data['Total Clouds (% Sky Obscured)'])
            data = ghi_est
            data['temp_air'] = raw_data['Temperature (°C)']
            data['wind_speed'] = raw_data['Wind Speed/Direction (mph)']*1.60934 # conversion to km/h
            data['relative_humidity'] = raw_data['Relative Humidity (%)']
            data['precipitable_water'] = pvlib.atmosphere.gueymard94_pw(
                data['temp_air'], data['relative_humidity'])
        elif method == 'solcast': # using Solcast API
            # Check if weather_forecast_cache is true or if forecast_data file does not exist
            if not os.path.isfile(w_forecast_cache_path):
                # Check if weather_forecast_cache_only is true, if so produce error for not finding cache file
                if not self.params["passed_data"].get("weather_forecast_cache_only",False):
                    # Retrieve data from the Solcast API
                    if 'solcast_api_key' not in self.retrieve_hass_conf:
                        self.logger.error("The solcast_api_key parameter was not defined")
                        return False
                    if 'solcast_rooftop_id' not in self.retrieve_hass_conf:
                        self.logger.error("The solcast_rooftop_id parameter was not defined")
                        return False
                    headers = {
                        'User-Agent': 'EMHASS',
                        "Authorization": "Bearer " + self.retrieve_hass_conf['solcast_api_key'],
                        "content-type": "application/json",
                        }
                    days_solcast = int(len(self.forecast_dates)*self.freq.seconds/3600)
                    # If weather_forecast_cache, set request days as twice as long to avoid length issues (add a buffer) 
                    if self.params["passed_data"].get("weather_forecast_cache",False):
                        days_solcast = min((days_solcast * 2), 336)
                    url = "https://api.solcast.com.au/rooftop_sites/"+self.retrieve_hass_conf['solcast_rooftop_id']+"/forecasts?hours="+str(days_solcast)
                    response = get(url, headers=headers)
                    '''import bz2 # Uncomment to save a serialized data for tests
                    import _pickle as cPickle
                    with bz2.BZ2File("data/test_response_solcast_get_method.pbz2", "w") as f: 
                        cPickle.dump(response, f)'''
                    # Verify the request passed
                    if int(response.status_code) == 200:
                        data = response.json()
                    elif int(response.status_code) == 402 or int(response.status_code) == 429:
                        self.logger.error("Solcast error: May have exceeded your subscription limit.")
                        return False  
                    elif int(response.status_code) >= 400 or int(response.status_code) >= 202:
                        self.logger.error("Solcast error: There was a issue with the solcast request, check solcast API key and rooftop ID.")
                        self.logger.error("Solcast error: Check that your subscription is valid and your network can connect to Solcast.")
                        return False  
                    data_list = []
                    for elm in data['forecasts']:
                        data_list.append(elm['pv_estimate']*1000) # Converting kW to W
                    # Check if the retrieved data has the correct length
                    if len(data_list) < len(self.forecast_dates):
                        self.logger.error("Not enough data retried from Solcast service, try increasing the time step or use MPC.")
                    else:
                        # If runtime weather_forecast_cache is true save forecast result to file as cache
                        if self.params["passed_data"].get("weather_forecast_cache",False):
                            # Add x2 forecast periods for cached results. This adds a extra delta_forecast amount of days for a buffer
                            cached_forecast_dates =  self.forecast_dates.union(pd.date_range(self.forecast_dates[-1], periods=(len(self.forecast_dates) +1), freq=self.freq)[1:])
                            cache_data_list = data_list[0:len(cached_forecast_dates)]
                            cache_data_dict = {'ts':cached_forecast_dates, 'yhat':cache_data_list}
                            data_cache = pd.DataFrame.from_dict(cache_data_dict)
                            data_cache.set_index('ts', inplace=True)
                            with open(w_forecast_cache_path, "wb") as file:    
                                cPickle.dump(data_cache, file)
                            if not os.path.isfile(w_forecast_cache_path):
                                self.logger.warning("Solcast forecast data could not be saved to file.")
                            else:
                                self.logger.info("Saved the Solcast results to cache, for later reference.")    
                        # Trim request results to forecast_dates        
                        data_list = data_list[0:len(self.forecast_dates)]
                        data_dict = {'ts':self.forecast_dates, 'yhat':data_list}
                        # Define DataFrame
                        data = pd.DataFrame.from_dict(data_dict)
                        # Define index
                        data.set_index('ts', inplace=True)
                # Else, notify user to update cache
                else:
                    self.logger.error("Unable to obtain Solcast cache file.")
                    self.logger.error("Try running optimization again with 'weather_forecast_cache_only': false")
                    self.logger.error("Optionally, obtain new Solcast cache with runtime parameter 'weather_forecast_cache': true in an optimization, or run the `weather-forecast-cache` action, to pull new data from Solcast and cache.")
                    return False
            # Else, open stored weather_forecast_data.pkl file for previous forecast data (cached data)
            else:
                with open(w_forecast_cache_path, "rb") as file:
                    data = cPickle.load(file)
                    if not isinstance(data, pd.DataFrame) or len(data) < len(self.forecast_dates):
                        self.logger.error("There has been a error obtaining cached Solcast forecast data.")
                        self.logger.error("Try running optimization again with 'weather_forecast_cache': true, or run action `weather-forecast-cache`, to pull new data from Solcast and cache.")
                        self.logger.warning("Removing old Solcast cache file. Next optimization will pull data from Solcast, unless 'weather_forecast_cache_only': true")
                        os.remove(w_forecast_cache_path)
                        return False
                    # Filter cached forecast data to match current forecast_dates start-end range (reduce forecast Dataframe size to appropriate length)
                    if self.forecast_dates[0] in data.index and self.forecast_dates[-1] in data.index:
                        data = data.loc[self.forecast_dates[0]:self.forecast_dates[-1]]
                        self.logger.info("Retrieved Solcast data from the previously saved cache.")
                    else:
                        self.logger.error("Unable to obtain cached Solcast forecast data within the requested timeframe range.")
                        self.logger.error("Try running optimization again (not using cache). Optionally, add runtime parameter 'weather_forecast_cache': true to pull new data from Solcast and cache.")
                        self.logger.warning("Removing old Solcast cache file. Next optimization will pull data from Solcast, unless 'weather_forecast_cache_only': true")
                        os.remove(w_forecast_cache_path)
                        return False    
        elif method == 'solar.forecast': # using the solar.forecast API
            # Retrieve data from the solar.forecast API
            if 'solar_forecast_kwp' not in self.retrieve_hass_conf:
                self.logger.warning("The solar_forecast_kwp parameter was not defined, using dummy values for testing")
                self.retrieve_hass_conf['solar_forecast_kwp'] = 5
            if  self.retrieve_hass_conf['solar_forecast_kwp'] == 0:
                self.logger.warning("The solar_forecast_kwp parameter is set to zero, setting to default 5")
                self.retrieve_hass_conf['solar_forecast_kwp'] = 5
            if self.optim_conf['delta_forecast_daily'].days > 1:
                self.logger.warning("The free public tier for solar.forecast only provides one day forecasts")
                self.logger.warning("Continuing with just the first day of data, the other days are filled with 0.0.")
                self.logger.warning("Use the other available methods for delta_forecast_daily > 1")
            headers = {
                "Accept": "application/json"
                }
            data = pd.DataFrame()
            for i in range(len(self.plant_conf['pv_module_model'])):
                url = "https://api.forecast.solar/estimate/"+str(round(self.lat, 2))+"/"+str(round(self.lon, 2))+\
                    "/"+str(self.plant_conf['surface_tilt'][i])+"/"+str(self.plant_conf['surface_azimuth'][i]-180)+\
                    "/"+str(self.retrieve_hass_conf["solar_forecast_kwp"])
                response = get(url, headers=headers)
                '''import bz2 # Uncomment to save a serialized data for tests
                import _pickle as cPickle
                with bz2.BZ2File("data/test_response_solarforecast_get_method.pbz2", "w") as f: 
                    cPickle.dump(response.json(), f)'''
                data_raw = response.json()
                data_dict = {'ts':list(data_raw['result']['watts'].keys()), 'yhat':list(data_raw['result']['watts'].values())}
                # Form the final DataFrame
                data_tmp = pd.DataFrame.from_dict(data_dict)
                data_tmp.set_index('ts', inplace=True)
                data_tmp.index = pd.to_datetime(data_tmp.index)
                data_tmp = data_tmp.tz_localize(self.forecast_dates.tz)
                data_tmp = data_tmp.reindex(index=self.forecast_dates)
                mask_up_data_df = data_tmp.copy(deep=True).fillna(method = "ffill").isnull()
                mask_down_data_df = data_tmp.copy(deep=True).fillna(method = "bfill").isnull()
                data_tmp.loc[data_tmp.index[mask_up_data_df['yhat']==True],:] = 0.0
                data_tmp.loc[data_tmp.index[mask_down_data_df['yhat']==True],:] = 0.0
                data_tmp.interpolate(inplace=True, limit=1)
                data_tmp = data_tmp.fillna(0.0)
                if len(data) == 0:
                    data = copy.deepcopy(data_tmp)
                else:
                    data = data + data_tmp
        elif method == 'csv': # reading from a csv file
            weather_csv_file_path = csv_path
            # Loading the csv file, we will consider that this is the PV power in W
            data = pd.read_csv(weather_csv_file_path, header=None, names=['ts', 'yhat'])
            # Check if the passed data has the correct length       
            if len(data) < len(self.forecast_dates):
                self.logger.error("Passed data from CSV is not long enough")
            else:
                # Ensure correct length
                data = data.loc[data.index[0:len(self.forecast_dates)],:]
                # Define index
                data.index = self.forecast_dates
                data.drop('ts', axis=1, inplace=True)
                data = data.copy().loc[self.forecast_dates]
        elif method == 'list': # reading a list of values
            # Loading data from passed list
            data_list = self.params['passed_data']['pv_power_forecast']
            # Check if the passed data has the correct length
            if len(data_list) < len(self.forecast_dates) and self.params['passed_data']['prediction_horizon'] is None:
                self.logger.error("Passed data from passed list is not long enough")
            else:
                # Ensure correct length
                data_list = data_list[0:len(self.forecast_dates)]
                # Define DataFrame
                data_dict = {'ts':self.forecast_dates, 'yhat':data_list}
                data = pd.DataFrame.from_dict(data_dict)
                # Define index
                data.set_index('ts', inplace=True)
        else:
            self.logger.error("Method %r is not valid", method)
            data = None
        return data
    
    def cloud_cover_to_irradiance(self, cloud_cover: pd.Series, 
                                  offset:Optional[int] = 35) -> pd.DataFrame:
        """
        Estimates irradiance from cloud cover in the following steps.
        
        1. Determine clear sky GHI using Ineichen model and
           climatological turbidity.
           
        2. Estimate cloudy sky GHI using a function of cloud_cover
           
        3. Estimate cloudy sky DNI using the DISC model.
        
        4. Calculate DHI from DNI and GHI.
        
        (This function was copied and modified from PVLib)

        :param cloud_cover: Cloud cover in %.
        :type cloud_cover: pd.Series
        :param offset: Determines the minimum GHI., defaults to 35
        :type offset: Optional[int], optional
        :return: Estimated GHI, DNI, and DHI.
        :rtype: pd.DataFrame
        """
        location = Location(latitude=self.lat, longitude=self.lon)
        solpos = location.get_solarposition(cloud_cover.index)
        cs = location.get_clearsky(cloud_cover.index, model='ineichen', 
                                   solar_position=solpos)
        # Using only the linear method
        offset = offset / 100.
        cloud_cover_unit = copy.deepcopy(cloud_cover) / 100.
        ghi = (offset + (1 - offset) * (1 - cloud_cover_unit)) * cs['ghi']
        # Using disc model
        dni = disc(ghi, solpos['zenith'], cloud_cover.index)['dni']
        dhi = ghi - dni * np.cos(np.radians(solpos['zenith']))
        irrads = pd.DataFrame({'ghi': ghi, 'dni': dni, 'dhi': dhi}).fillna(0)
        return irrads
    
    @staticmethod
    def get_mix_forecast(df_now: pd.DataFrame, df_forecast: pd.DataFrame, 
                         alpha:float, beta:float, col:str) -> pd.DataFrame:
        """A simple correction method for forecasted data using the current real values of a variable.

        :param df_now: The DataFrame containing the current/real values
        :type df_now: pd.DataFrame
        :param df_forecast: The DataFrame containing the forecast data
        :type df_forecast: pd.DataFrame
        :param alpha: A weight for the forecast data side
        :type alpha: float
        :param beta: A weight for the current/real values sied
        :type beta: float
        :param col: The column variable name
        :type col: str
        :return: The output DataFrame with the corrected values
        :rtype: pd.DataFrame
        """
        first_fcst = alpha*df_forecast.iloc[0] + beta*df_now[col].iloc[-1]
        df_forecast.iloc[0] = first_fcst
        return df_forecast
    
    def get_power_from_weather(self, df_weather: pd.DataFrame, 
                               set_mix_forecast:Optional[bool] = False,
                               df_now:Optional[pd.DataFrame] = pd.DataFrame()) -> pd.Series:
        r"""
        Convert wheater forecast data into electrical power.
        
        :param df_weather: The DataFrame containing the weather forecasted data. \
            This DF should be generated by the 'get_weather_forecast' method or at \
            least contain the same columns names filled with proper data.
        :type df_weather: pd.DataFrame
        :param set_mix_forecast: Use a mixed forcast strategy to integra now/current values.
        :type set_mix_forecast: Bool, optional
        :param df_now: The DataFrame containing the now/current data.
        :type df_now: pd.DataFrame
        :return: The DataFrame containing the electrical power in Watts
        :rtype: pd.DataFrame

        """
        # If using csv method we consider that yhat is the PV power in W
        if "solar_forecast_kwp" in self.retrieve_hass_conf.keys() and self.retrieve_hass_conf["solar_forecast_kwp"] == 0:
            P_PV_forecast = pd.Series(0, index=df_weather.index)
        else:
            if self.weather_forecast_method == 'solcast' or self.weather_forecast_method == 'solar.forecast' or \
                self.weather_forecast_method == 'csv' or self.weather_forecast_method == 'list':
                P_PV_forecast = df_weather['yhat']
                P_PV_forecast.name = None
            else: # We will transform the weather data into electrical power
                # Transform to power (Watts)
                # Setting the main parameters of the PV plant
                location = Location(latitude=self.lat, longitude=self.lon)
                temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['close_mount_glass_glass']
                cec_modules = bz2.BZ2File(self.emhass_conf['root_path'] / 'data' / 'cec_modules.pbz2', "rb")
                cec_modules = cPickle.load(cec_modules)
                cec_inverters = bz2.BZ2File(self.emhass_conf['root_path'] / 'data' /  'cec_inverters.pbz2', "rb")
                cec_inverters = cPickle.load(cec_inverters)
                if type(self.plant_conf['pv_module_model']) == list:
                    P_PV_forecast = pd.Series(0, index=df_weather.index)
                    for i in range(len(self.plant_conf['pv_module_model'])):
                        # Selecting correct module and inverter
                        module = cec_modules[self.plant_conf['pv_module_model'][i]]
                        inverter = cec_inverters[self.plant_conf['pv_inverter_model'][i]]
                        # Building the PV system in PVLib
                        system = PVSystem(surface_tilt=self.plant_conf['surface_tilt'][i], 
                                        surface_azimuth=self.plant_conf['surface_azimuth'][i],
                                        module_parameters=module,
                                        inverter_parameters=inverter,
                                        temperature_model_parameters=temp_params,
                                        modules_per_string=self.plant_conf['modules_per_string'][i],
                                        strings_per_inverter=self.plant_conf['strings_per_inverter'][i])
                        mc = ModelChain(system, location, aoi_model="physical")
                        # Run the model on the weather DF indexes
                        mc.run_model(df_weather)
                        # Extracting results for AC power
                        P_PV_forecast = P_PV_forecast + mc.results.ac
                else:
                    # Selecting correct module and inverter
                    module = cec_modules[self.plant_conf['pv_module_model']]
                    inverter = cec_inverters[self.plant_conf['pv_inverter_model']]
                    # Building the PV system in PVLib
                    system = PVSystem(surface_tilt=self.plant_conf['surface_tilt'], 
                                    surface_azimuth=self.plant_conf['surface_azimuth'],
                                    module_parameters=module,
                                    inverter_parameters=inverter,
                                    temperature_model_parameters=temp_params,
                                    modules_per_string=self.plant_conf['modules_per_string'],
                                    strings_per_inverter=self.plant_conf['strings_per_inverter'])
                    mc = ModelChain(system, location, aoi_model="physical")
                    # Run the model on the weather DF indexes
                    mc.run_model(df_weather)
                    # Extracting results for AC power
                    P_PV_forecast = mc.results.ac
        if set_mix_forecast:
            P_PV_forecast = Forecast.get_mix_forecast(
                df_now, P_PV_forecast, 
                self.params['passed_data']['alpha'], self.params['passed_data']['beta'], self.var_PV)
        return P_PV_forecast
    
    def get_forecast_days_csv(self, timedelta_days: Optional[int] = 1) -> pd.date_range:
        r"""
        Get the date range vector of forecast dates that will be used when loading a CSV file.
        
        :return: The forecast dates vector
        :rtype: pd.date_range

        """
        start_forecast_csv = pd.Timestamp(datetime.now(), tz=self.time_zone).replace(microsecond=0)
        if self.method_ts_round == 'nearest':
            start_forecast_csv = pd.Timestamp(datetime.now(), tz=self.time_zone).replace(microsecond=0)
        elif self.method_ts_round == 'first':
            start_forecast_csv = pd.Timestamp(datetime.now(), tz=self.time_zone).replace(microsecond=0).floor(freq=self.freq)
        elif self.method_ts_round == 'last':
            start_forecast_csv = pd.Timestamp(datetime.now(), tz=self.time_zone).replace(microsecond=0).ceil(freq=self.freq)
        else:
            self.logger.error("Wrong method_ts_round passed parameter")
        end_forecast_csv = (start_forecast_csv + self.optim_conf['delta_forecast_daily']).replace(microsecond=0)
        forecast_dates_csv = pd.date_range(start=start_forecast_csv, 
                                           end=end_forecast_csv+timedelta(days=timedelta_days)-self.freq, 
                                           freq=self.freq, tz=self.time_zone).tz_convert('utc').round(self.freq, ambiguous='infer', nonexistent='shift_forward').tz_convert(self.time_zone)
        if self.params is not None:
            if 'prediction_horizon' in list(self.params['passed_data'].keys()):
                if self.params['passed_data']['prediction_horizon'] is not None:
                    forecast_dates_csv = forecast_dates_csv[0:self.params['passed_data']['prediction_horizon']]
        return forecast_dates_csv
    
    def get_forecast_out_from_csv_or_list(self, df_final: pd.DataFrame, forecast_dates_csv: pd.date_range,
                                          csv_path: str, data_list: Optional[list] = None, 
                                          list_and_perfect: Optional[bool] = False) -> pd.DataFrame:
        r"""
        Get the forecast data as a DataFrame from a CSV file. 
        
        The data contained in the CSV file should be a 24h forecast with the same frequency as 
        the main 'optimization_time_step' parameter in the configuration file. The timestamp will not be used and 
        a new DateTimeIndex is generated to fit the timestamp index of the input data in 'df_final'.
        
        :param df_final: The DataFrame containing the input data.
        :type df_final: pd.DataFrame
        :param forecast_dates_csv: The forecast dates vector
        :type forecast_dates_csv: pd.date_range
        :param csv_path: The path to the CSV file
        :type csv_path: str
        :return: The data from the CSV file
        :rtype: pd.DataFrame

        """
        if csv_path is None:
            data_dict = {'ts':forecast_dates_csv, 'yhat':data_list}
            df_csv = pd.DataFrame.from_dict(data_dict)
            df_csv.index = forecast_dates_csv
            df_csv.drop(['ts'], axis=1, inplace=True)
            df_csv = set_df_index_freq(df_csv)
            if list_and_perfect:
                days_list = df_final.index.day.unique().tolist()
            else:
                days_list = df_csv.index.day.unique().tolist()
        else:
            if not os.path.exists(csv_path):
                csv_path = self.emhass_conf['data_path'] / csv_path
            load_csv_file_path = csv_path    
            df_csv = pd.read_csv(load_csv_file_path, header=None, names=['ts', 'yhat'])
            df_csv.index = forecast_dates_csv
            df_csv.drop(['ts'], axis=1, inplace=True)
            df_csv = set_df_index_freq(df_csv)
            days_list = df_final.index.day.unique().tolist()
        forecast_out = pd.DataFrame()
        for day in days_list:
            if csv_path is None:
                if list_and_perfect:
                    df_tmp = copy.deepcopy(df_final)
                else:
                    df_tmp = copy.deepcopy(df_csv)
            else:
                df_tmp = copy.deepcopy(df_final)
            first_elm_index = [i for i, x in enumerate(df_tmp.index.day == day) if x][0]
            last_elm_index = [i for i, x in enumerate(df_tmp.index.day == day) if x][-1]
            fcst_index = pd.date_range(start=df_tmp.index[first_elm_index],
                                    end=df_tmp.index[last_elm_index], 
                                    freq=df_tmp.index.freq)
            first_hour = str(df_tmp.index[first_elm_index].hour)+":"+str(df_tmp.index[first_elm_index].minute)
            last_hour = str(df_tmp.index[last_elm_index].hour)+":"+str(df_tmp.index[last_elm_index].minute)
            if len(forecast_out) == 0:
                if csv_path is None:
                    if list_and_perfect:
                        forecast_out = pd.DataFrame(
                            df_csv.between_time(first_hour, last_hour).values,
                            index=fcst_index)
                    else:
                        forecast_out = pd.DataFrame(
                            df_csv.loc[fcst_index,:].between_time(first_hour, last_hour).values,
                            index=fcst_index)
                else:
                    forecast_out = pd.DataFrame(
                        df_csv.between_time(first_hour, last_hour).values,
                        index=fcst_index)
            else:
                if csv_path is None:
                    if list_and_perfect:
                        forecast_tp = pd.DataFrame(
                            df_csv.between_time(first_hour, last_hour).values,
                            index=fcst_index)
                    else:
                        forecast_tp = pd.DataFrame(
                            df_csv.loc[fcst_index,:].between_time(first_hour, last_hour).values,
                            index=fcst_index)
                else:
                    forecast_tp = pd.DataFrame(
                        df_csv.between_time(first_hour, last_hour).values,
                        index=fcst_index)
                forecast_out = pd.concat([forecast_out, forecast_tp], axis=0)
        return forecast_out
    
    def get_load_forecast(self, days_min_load_forecast: Optional[int] = 3, method: Optional[str] = 'naive',
                          csv_path: Optional[str] = "data_load_forecast.csv",
                          set_mix_forecast:Optional[bool] = False, df_now:Optional[pd.DataFrame] = pd.DataFrame(),
                          use_last_window: Optional[bool] = True, mlf: Optional[MLForecaster] = None,
                          debug: Optional[bool] = False) -> pd.Series:
        r"""
        Get and generate the load forecast data.
        
        :param days_min_load_forecast: The number of last days to retrieve that \
            will be used to generate a naive forecast, defaults to 3
        :type days_min_load_forecast: int, optional
        :param method: The method to be used to generate load forecast, the options \
            are 'naive' for a persistance model, 'mlforecaster' for using a custom \
            previously fitted machine learning model, 'csv' to read the forecast from \
            a CSV file and 'list' to use data directly passed at runtime as a list of \
            values. Defaults to 'naive'.
        :type method: str, optional
        :param csv_path: The path to the CSV file used when method = 'csv', \
            defaults to "/data/data_load_forecast.csv"
        :type csv_path: str, optional
        :param set_mix_forecast: Use a mixed forcast strategy to integra now/current values.
        :type set_mix_forecast: Bool, optional
        :param df_now: The DataFrame containing the now/current data.
        :type df_now: pd.DataFrame, optional
        :param use_last_window: True if the 'last_window' option should be used for the \
            custom machine learning forecast model. The 'last_window=True' means that the data \
            that will be used to generate the new forecast will be freshly retrieved from \
            Home Assistant. This data is needed because the forecast model is an auto-regressive \
            model with lags. If 'False' then the data using during the model train is used.
        :type use_last_window: Bool, optional
        :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
            and unit testing. In production the actual model will be read from a saved pickle file.
        :type mlf: mlforecaster, optional
        :param debug: The DataFrame containing the now/current data.
        :type debug: Bool, optional
        :return: The DataFrame containing the electrical load power in Watts
        :rtype: pd.DataFrame

        """
        csv_path  = self.emhass_conf['data_path'] / csv_path
        
        if method == 'naive' or method == 'mlforecaster': # retrieving needed data for these methods
            self.logger.info("Retrieving data from hass for load forecast using method = "+method)
            var_list = [self.var_load]
            var_replace_zero = None
            var_interp = [self.var_load]
            time_zone_load_foreacast = None
            # We will need to retrieve a new set of load data according to the days_min_load_forecast parameter
            rh = RetrieveHass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                               self.freq, time_zone_load_foreacast, self.params, self.emhass_conf, self.logger)
            if self.get_data_from_file:
                filename_path = self.emhass_conf['data_path'] / 'test_df_final.pkl'
                with open(filename_path, 'rb') as inp:
                    rh.df_final, days_list, var_list = pickle.load(inp)
                    self.var_load = var_list[0]
                    self.retrieve_hass_conf['sensor_power_load_no_var_loads'] = self.var_load
                    var_interp = [var_list[0]]
                    self.var_list = [var_list[0]]
                    self.var_load_new = self.var_load+'_positive'
            else:
                days_list = get_days_list(days_min_load_forecast) 
                if not rh.get_data(days_list, var_list):
                    return False
            if  not rh.prepare_data(
                self.retrieve_hass_conf['sensor_power_load_no_var_loads'], load_negative = self.retrieve_hass_conf['load_negative'],
                set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                var_replace_zero = var_replace_zero, var_interp = var_interp):
                return False
            df = rh.df_final.copy()[[self.var_load_new]]
        if method == 'naive': # using a naive approach
            mask_forecast_out = (df.index > days_list[-1] - self.optim_conf['delta_forecast_daily'])
            forecast_out = df.copy().loc[mask_forecast_out]
            forecast_out = forecast_out.rename(columns={self.var_load_new: 'yhat'})
            # Force forecast_out length to avoid mismatches
            forecast_out = forecast_out.iloc[0:len(self.forecast_dates)]
            forecast_out.index = self.forecast_dates
        elif method == 'mlforecaster': # using a custom forecast model with machine learning
            # Load model
            model_type = self.params['passed_data']['model_type']
            filename = model_type+'_mlf.pkl'
            filename_path = self.emhass_conf['data_path'] / filename
            if not debug:
                if filename_path.is_file():
                    with open(filename_path, 'rb') as inp:
                        mlf = pickle.load(inp)
                else:
                    self.logger.error("The ML forecaster file was not found, please run a model fit method before this predict method")
                    return False
            # Make predictions
            if use_last_window:
                data_last_window = copy.deepcopy(df)
                data_last_window = data_last_window.rename(columns={self.var_load_new: self.var_load})
            else:
                data_last_window = None
            forecast_out = mlf.predict(data_last_window)
            # Force forecast length to avoid mismatches
            self.logger.debug("Number of ML predict forcast data generated (lags_opt): " + str(len(forecast_out.index)))
            self.logger.debug("Number of forcast dates obtained: " + str(len(self.forecast_dates)))
            if len(self.forecast_dates) < len(forecast_out.index):
                forecast_out = forecast_out.iloc[0:len(self.forecast_dates)]
            # To be removed once bug is fixed
            elif len(self.forecast_dates) > len(forecast_out.index):
                self.logger.error("Unable to obtain: " + str(len(self.forecast_dates))  + " lags_opt values from sensor: power load no var loads, check optimization_time_step/freq and historic_days_to_retrieve/days_to_retrieve parameters")
                return False
            # Define DataFrame
            data_dict = {'ts':self.forecast_dates, 'yhat':forecast_out.values.tolist()}
            data = pd.DataFrame.from_dict(data_dict)
            # Define index
            data.set_index('ts', inplace=True)
            forecast_out = data.copy().loc[self.forecast_dates]
        elif method == 'csv': # reading from a csv file
            load_csv_file_path = csv_path
            df_csv = pd.read_csv(load_csv_file_path, header=None, names=['ts', 'yhat'])
            if len(df_csv) < len(self.forecast_dates):
                self.logger.error("Passed data from CSV is not long enough")
            else:
                # Ensure correct length
                df_csv = df_csv.loc[df_csv.index[0:len(self.forecast_dates)],:]
                # Define index
                df_csv.index = self.forecast_dates
                df_csv.drop(['ts'], axis=1, inplace=True)
                forecast_out = df_csv.copy().loc[self.forecast_dates]
        elif method == 'list': # reading a list of values
            # Loading data from passed list
            data_list = self.params['passed_data']['load_power_forecast']
            # Check if the passed data has the correct length
            if len(data_list) < len(self.forecast_dates) and self.params['passed_data']['prediction_horizon'] is None:
                self.logger.error("Passed data from passed list is not long enough")
                return False
            else:
                # Ensure correct length
                data_list = data_list[0:len(self.forecast_dates)]
                # Define DataFrame
                data_dict = {'ts':self.forecast_dates, 'yhat':data_list}
                data = pd.DataFrame.from_dict(data_dict)
                # Define index
                data.set_index('ts', inplace=True)
                forecast_out = data.copy().loc[self.forecast_dates]
        else:
            self.logger.error("Passed method is not valid")
            return False
        P_Load_forecast = copy.deepcopy(forecast_out['yhat'])
        if set_mix_forecast:
            P_Load_forecast = Forecast.get_mix_forecast(
                df_now, P_Load_forecast, 
                self.params['passed_data']['alpha'], self.params['passed_data']['beta'], self.var_load_new)
        return P_Load_forecast
    
    def get_load_cost_forecast(self, df_final: pd.DataFrame, method: Optional[str] = 'hp_hc_periods',
                               csv_path: Optional[str] = "data_load_cost_forecast.csv",
                               list_and_perfect: Optional[bool] = False) -> pd.DataFrame:
        r"""
        Get the unit cost for the load consumption based on multiple tariff \
        periods. This is the cost of the energy from the utility in a vector \
        sampled at the fixed freq value.
        
        :param df_final: The DataFrame containing the input data.
        :type df_final: pd.DataFrame
        :param method: The method to be used to generate load cost forecast, \
            the options are 'hp_hc_periods' for peak and non-peak hours contracts\
            and 'csv' to load a CSV file, defaults to 'hp_hc_periods'
        :type method: str, optional
        :param csv_path: The path to the CSV file used when method = 'csv', \
            defaults to "data_load_cost_forecast.csv"
        :type csv_path: str, optional
        :return: The input DataFrame with one additionnal column appended containing
            the load cost for each time observation.
        :rtype: pd.DataFrame

        """
        csv_path  = self.emhass_conf['data_path'] / csv_path
        if method == 'hp_hc_periods':
            df_final[self.var_load_cost] = self.optim_conf['load_offpeak_hours_cost']
            list_df_hp = []
            for key, period_hp in self.optim_conf['load_peak_hour_periods'].items():
                list_df_hp.append(df_final[self.var_load_cost].between_time(
                    period_hp[0]['start'], period_hp[1]['end']))
            for df_hp in list_df_hp:
                df_final.loc[df_hp.index, self.var_load_cost] = self.optim_conf['load_peak_hours_cost']
        elif method == 'csv':
            forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
            forecast_out = self.get_forecast_out_from_csv_or_list(
                df_final, forecast_dates_csv, csv_path)
            df_final[self.var_load_cost] = forecast_out
        elif method == 'list': # reading a list of values
            # Loading data from passed list
            data_list = self.params['passed_data']['load_cost_forecast']
            # Check if the passed data has the correct length
            if len(data_list) < len(self.forecast_dates) and self.params['passed_data']['prediction_horizon'] is None:
                self.logger.error("Passed data from passed list is not long enough")
                return False
            else:
                # Ensure correct length
                data_list = data_list[0:len(self.forecast_dates)]
                # Define the correct dates
                forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
                forecast_out = self.get_forecast_out_from_csv_or_list(
                    df_final, forecast_dates_csv, None, data_list=data_list, list_and_perfect=list_and_perfect)
                # Fill the final DF
                df_final[self.var_load_cost] = forecast_out
        else:
            self.logger.error("Passed method is not valid")
            return False
        return df_final
    
    def get_prod_price_forecast(self, df_final: pd.DataFrame, method: Optional[str] = 'constant',
                                csv_path: Optional[str] = "data_prod_price_forecast.csv", 
                                list_and_perfect: Optional[bool] = False) -> pd.DataFrame:

        r"""
        Get the unit power production price for the energy injected to the grid.\
        This is the price of the energy injected to the utility in a vector \
        sampled at the fixed freq value.
        
        :param df_input_data: The DataFrame containing all the input data retrieved
            from hass
        :type df_input_data: pd.DataFrame
        :param method: The method to be used to generate the production price forecast, \
            the options are 'constant' for a fixed constant value and 'csv'\
            to load a CSV file, defaults to 'constant'
        :type method: str, optional
        :param csv_path: The path to the CSV file used when method = 'csv', \
            defaults to "/data/data_load_cost_forecast.csv"
        :type csv_path: str, optional
        :return: The input DataFrame with one additionnal column appended containing
            the power production price for each time observation.
        :rtype: pd.DataFrame

        """
        csv_path  = self.emhass_conf['data_path'] / csv_path
        if method == 'constant':
            df_final[self.var_prod_price] = self.optim_conf['photovoltaic_production_sell_price']
        elif method == 'csv':
            forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
            forecast_out = self.get_forecast_out_from_csv_or_list(
                df_final, forecast_dates_csv, csv_path)
            df_final[self.var_prod_price] = forecast_out
        elif method == 'list': # reading a list of values
            # Loading data from passed list
            data_list = self.params['passed_data']['prod_price_forecast']
            # Check if the passed data has the correct length
            if len(data_list) < len(self.forecast_dates) and self.params['passed_data']['prediction_horizon'] is None:
                self.logger.error("Passed data from passed list is not long enough")
                return False
            else:
                # Ensure correct length
                data_list = data_list[0:len(self.forecast_dates)]
                # Define the correct dates
                forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
                forecast_out = self.get_forecast_out_from_csv_or_list(
                    df_final, forecast_dates_csv, None, data_list=data_list, list_and_perfect=list_and_perfect)
                # Fill the final DF
                df_final[self.var_prod_price] = forecast_out
        else:
            self.logger.error("Passed method is not valid")
            return False
        return df_final