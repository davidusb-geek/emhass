#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import pathlib, pickle, copy, logging, json
import pandas as pd, numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.irradiance import disc

from emhass.retrieve_hass import retrieve_hass
from emhass.utils import get_days_list

class forecast:
    """
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
    
    The weather forecast is obtained from two methods. The first method
    uses a scrapper to the ClearOutside webpage which proposes detailed forecasts 
    based on Lat/Lon locations. This method seems quite stable but as with any scrape 
    method it will fail if any changes are made to the webpage API. The second method
    for weather forecast is using a direct read fro a CSV file. With this method we
    will consider that we are reading the PV power directly.
    
    The 'get_power_from_weather' method is proposed here to convert from irradiance
    data to electrical power. Again PVLib is used to model the PV plant.
    
    For the load forecast two methods are available. The first method allows the user 
    to use a CSV file with their own forecast. With this method a more powerful 
    external package for time series forecast may be used to create your own detailed 
    load forecast. The second method is a naive method, also called persistance.
    It simply assumes that the forecast for a future period will be equal to the
    observed values in a past period. The past period is controlled using 
    parameter 'delta_forecast'.
    
    For the PV production selling price and Load cost forecasts the privileged method
    is a direct read from a user provided CSV file. 
    
    For all the forecastin g methods, the CSV file should contain no header and the 
    timestamped data should have the following format:
    
    2021-04-29 00:00:00+00:00,287.07
    
    2021-04-29 00:30:00+00:00,274.27
    
    2021-04-29 01:00:00+00:00,243.38
    
    ...
    
    The data columns in these files will correspond to the data in the units expected
    for each forecasting method.
    
    """

    def __init__(self, retrieve_hass_conf: dict, optim_conf: dict, plant_conf: dict, 
                 params: str, config_path: str, logger: logging.Logger, 
                 opt_time_delta: Optional[int] = 24,
                 get_data_from_file: Optional[bool] = False) -> None:
        """
        Define constructor for the forecast class.
        
        :param retrieve_hass_conf: Dictionnary containing the needed configuration
            data from the configuration file, specific to retrieve data from HASS
        :type retrieve_hass_conf: dict
        :param optim_conf: Dictionnary containing the needed configuration
            data from the configuration file, specific for the optimization task
        :type optim_conf: dict
        :param plant_conf: Dictionnary containing the needed configuration
            data from the configuration file, specific for the modeling of the PV plant
        :type plant_conf: dict
        :param params: Configuration parameters passed from data/options.json
        :type params: str
        :param config_path: The path to the yaml configuration file
        :type config_path: str
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
        self.freq = self.retrieve_hass_conf['freq']
        self.time_zone = self.retrieve_hass_conf['time_zone']
        self.timeStep = self.freq.seconds/3600 # in hours
        self.time_delta = pd.to_timedelta(opt_time_delta, "hours") # The period of optimization
        self.var_PV = self.retrieve_hass_conf['var_PV']
        self.var_load = self.retrieve_hass_conf['var_load']
        self.var_load_new = self.var_load+'_positive'
        self.start_forecast = pd.Timestamp(datetime.now(), tz=self.time_zone).replace(microsecond=0)
        self.end_forecast = (self.start_forecast + self.optim_conf['delta_forecast']).replace(microsecond=0)
        self.lat = self.retrieve_hass_conf['lat'] 
        self.lon = self.retrieve_hass_conf['lon']
        self.root = config_path
        self.logger = logger
        self.get_data_from_file = get_data_from_file
        self.var_load_cost = 'unit_load_cost'
        self.var_prod_price = 'unit_prod_price'
        self.forecast_dates = pd.date_range(start=self.start_forecast, 
                                            end=self.end_forecast-self.freq, 
                                            freq=self.freq).round(self.freq)
        if params is None:
            self.params = params
        else:
            self.params = json.loads(params)
        
    def get_weather_forecast(self, method: Optional[str] = 'scrapper',
                             csv_path: Optional[str] = "/data/data_weather_forecast.csv") -> pd.DataFrame:
        """
        Get and generate weather forecast data.
        
        :param method: The desired method, options are 'scrapper' and 'csv', \
            defaults to 'scrapper'
        :type method: str, optional
        :return: The DataFrame containing the forecasted data
        :rtype: pd.DataFrame
        
        """
        self.logger.info("Retrieving weather forecast data using method = "+method)
        self.weather_forecast_method = method # Saving this attribute for later use to identify csv method usage
        if method == 'scrapper':
            freq_scrap = pd.to_timedelta(60, "minutes") # The scrapping time step is 60min
            forecast_dates_scrap = pd.date_range(start=self.start_forecast,
                                                 end=self.end_forecast-freq_scrap, 
                                                 freq=freq_scrap).round(freq_scrap)
            forecast_dates = pd.date_range(start=self.start_forecast, 
                                           end=self.end_forecast-self.freq, freq=self.freq).round(self.freq)
            # Using the clearoutside webpage
            response = requests.get("https://clearoutside.com/forecast/"+str(round(self.lat, 2))+"/"+str(round(self.lon, 2)))
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find_all(id='day_0')[0]
            list_names = table.find_all(class_='fc_detail_label')
            list_tables = table.find_all('ul')[1:]
            selected_cols = [0, 1, 2, 3, 10, 12, 15] # Selected variables
            col_names = [list_names[i].get_text() for i in selected_cols]
            list_tables = [list_tables[i] for i in selected_cols]
            # Building the raw DF container
            raw_data = pd.DataFrame(index=range(24), columns=col_names, dtype=float)
            for count_col, col in enumerate(col_names):
                list_rows = list_tables[count_col].find_all('li')
                for count_row, row in enumerate(list_rows):
                    raw_data.loc[count_row, col] = float(row.get_text())
            # Treating index
            raw_data.set_index(forecast_dates_scrap, inplace=True)
            raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
            raw_data = raw_data.reindex(forecast_dates)
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
        elif method == 'csv': # reading from a csv file
            forecast_dates_csv = self.get_forecast_days_csv()
            weather_csv_file_path = self.root + csv_path
            # Loading the csv file, we will consider that this is the PV power in W
            data = pd.read_csv(weather_csv_file_path, header=None, names=['ts', 'yhat'])
            data = pd.concat([data, data], axis=0)
            # Check if the passed data has the correct length
            if len(data) < len(forecast_dates_csv):
                data = None
                self.logger.error("Passed data from CSV is not long enough")
            else:
                # Define index and pick correct dates
                data.index = forecast_dates_csv
                data.drop(['ts'], axis=1, inplace=True)
                data = data.copy().loc[self.forecast_dates]
        elif method == 'list': # reading a list of values
            forecast_dates_csv = self.get_forecast_days_csv()
            # Loading data from passed list
            data_list = self.params['passed_data']['pv_power_forecast']
            data_list = data_list + data_list
            # Check if the passed data has the correct length
            if len(data_list) < len(forecast_dates_csv):
                data = None
                self.logger.error("Passed data from passed list is not long enough")
            else:
                # Define index and pick correct dates
                data_dict = {'ts':forecast_dates_csv, 'yhat':data_list}
                data = pd.DataFrame.from_dict(data_dict)
                data.index = forecast_dates_csv
                data.drop(['ts'], axis=1, inplace=True)
                data = data.copy().loc[self.forecast_dates]
        else:
            self.logger.error("Passed method is not valid")
        return data
    
    def cloud_cover_to_irradiance(self, cloud_cover: pd.Series, 
                                  offset:Optional[int] = 35) -> pd.DataFrame:
        """Estimates irradiance from cloud cover in the following steps:
        
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
    
    def get_power_from_weather(self, df_weather: pd.DataFrame) -> pd.Series:
        """
        Convert wheater forecast data into electrical power.
        
        :param df_weather: The DataFrame containing the weather forecasted data. \
            This DF should be generated by the 'get_weather_forecast' method or at \
            least contain the same columns names filled with proper data.
        :type df_weather: pd.DataFrame
        :return: The DataFrame containing the electrical power in Watts
        :rtype: pd.DataFrame

        """
        # If using csv method we consider that yhat is the PV power in W
        if self.weather_forecast_method == 'csv' or self.weather_forecast_method == 'list':
            P_PV_forecast = df_weather['yhat']
            P_PV_forecast.name = None
        else: # We will transform the weather data into electrical power
            # Transform to power (Watts)
            # Setting the main parameters of the PV plant
            location = Location(latitude=self.lat, longitude=self.lon)
            temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['close_mount_glass_glass']
            cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
            cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
            # Selecting correct module and inverter
            module = cec_modules[self.plant_conf['module_model']]
            inverter = cec_inverters[self.plant_conf['inverter_model']]
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
        
        return P_PV_forecast
    
    def get_forecast_days_csv(self, timedelta_days: Optional[int] = 1) -> pd.date_range:
        """
        Get the date range vector of forecast dates that will be used when \
        loading a CSV file.
        
        :return: The forecast dates vector
        :rtype: pd.date_range

        """
        start_forecast_csv = pd.Timestamp(datetime.now(), tz=self.time_zone).replace(hour=0, minute=0, second=0, microsecond=0)
        end_forecast_csv = (start_forecast_csv + self.optim_conf['delta_forecast']).replace(microsecond=0)
        forecast_dates_csv = pd.date_range(start=start_forecast_csv, 
                                           end=end_forecast_csv+timedelta(days=timedelta_days)-self.freq, 
                                           freq=self.freq).round(self.freq)
        return forecast_dates_csv
    
    def get_forecast_out_from_csv(self, df_final: pd.DataFrame, forecast_dates_csv: pd.date_range,
                                  csv_path: str, data_list: Optional[list] = None) -> pd.DataFrame:
        """
        Get the forecast data as a DataFrame from a CSV file. The data contained \
            in the CSV file should be a 24h forecast with the same frequency as \
            the main 'freq' parameter in the configuration file. The timestamp \
            will not be used and a new DateTimeIndex is generated to fit the \
            timestamp index of the input data in 'df_final'.
        
        :param df_final: The DataFrame containing the input data.
        :type df_final: pd.DataFrame
        :param forecast_dates_csv: The forecast dates vector
        :type forecast_dates_csv: pd.date_range
        :param csv_path: The path to the CSV file
        :type csv_path: str
        :return: The data from the CSV file
        :rtype: pd.DataFrame

        """
        days_list = pd.date_range(start=df_final.index[0], 
                                  end=df_final.index[-1], 
                                  freq='D')
        
        if csv_path is None:
            data_dict = {'ts':forecast_dates_csv, 'yhat':data_list}
            df_csv = pd.DataFrame.from_dict(data_dict)
            df_csv.index = forecast_dates_csv
            df_csv.drop(['ts'], axis=1, inplace=True)
        else:
            load_csv_file_path = self.root + csv_path
            df_csv = pd.read_csv(load_csv_file_path, header=None, names=['ts', 'yhat'])
            df_csv.index = forecast_dates_csv
            df_csv.drop(['ts'], axis=1, inplace=True)
        
        forecast_out = pd.DataFrame()
        for day in days_list:
            first_elm_index = [i for i, x in enumerate(df_final.index.day == day.day) if x][0]
            last_elm_index = [i for i, x in enumerate(df_final.index.day == day.day) if x][-1]
            fcst_index = pd.date_range(start=df_final.index[first_elm_index],
                                       end=df_final.index[last_elm_index], 
                                       freq=df_final.index.freq)
            first_hour = str(df_final.index[first_elm_index].hour)+":"+str(df_final.index[first_elm_index].minute)
            last_hour = str(df_final.index[last_elm_index].hour)+":"+str(df_final.index[last_elm_index].minute)
            if len(forecast_out) == 0:
                forecast_out = pd.DataFrame(
                    df_csv.between_time(first_hour, last_hour).values,
                    index=fcst_index)
            else:
                forecast_tp = pd.DataFrame(
                    df_csv.between_time(first_hour, last_hour).values,
                    index=fcst_index)
                forecast_out = pd.concat([forecast_out, forecast_tp], axis=0)
        
        return forecast_out
    
    def get_load_forecast(self, days_min_load_forecast: Optional[int] = 3, method: Optional[str] = 'naive',
                          csv_path: Optional[str] = "/data/data_load_forecast.csv") -> pd.Series:
        """
        Get and generate the load forecast data.
        
        :param days_min_load_forecast: The number of last days to retrieve that \
            will be used to generate a naive forecast, defaults to 3
        :type days_min_load_forecast: int, optional
        :param method: The method to be used to generate load forecast, the options \
            are 'csv' to load a CSV file or 'naive' for a persistance model, defaults to 'naive'
        :type method: str, optional
        :param csv_path: The path to the CSV file used when method = 'csv', \
            defaults to "/data/data_load_forecast.csv"
        :type csv_path: str, optional
        :return: The DataFrame containing the electrical load power in Watts
        :rtype: pd.DataFrame

        """
        
        if method == 'naive': # using a naive approach
            self.logger.info("Retrieving data from hass for load forecast using method = "+method)
            var_list = [self.var_load]
            var_replace_zero = None
            var_interp = [self.var_load]
            time_zone_load_foreacast = None
            # We will need to retrieve a new set of load data according to the days_min_load_forecast parameter
            rh = retrieve_hass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                               self.freq, time_zone_load_foreacast, self.params, self.root, self.logger)
            if self.get_data_from_file:
                with open(pathlib.Path(self.root+'/data/test_df_final.pkl'), 'rb') as inp:
                    rh.df_final, days_list, _ = pickle.load(inp)
            else:
                days_list = get_days_list(days_min_load_forecast)
                rh.get_data(days_list, var_list)
            rh.prepare_data(self.retrieve_hass_conf['var_load'], load_negative = self.retrieve_hass_conf['load_negative'],
                            set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                            var_replace_zero = var_replace_zero, 
                            var_interp = var_interp)
            
            df = rh.df_final.copy()[[self.var_load_new]]
            
            mask_forecast_out = (df.index > days_list[-1] - self.optim_conf['delta_forecast'])
            forecast_out = df.copy().loc[mask_forecast_out]
            forecast_out = forecast_out.rename(columns={self.var_load_new: 'yhat'})
            # Force forecast_out length to avoid mismatches
            forecast_out = forecast_out.iloc[0:len(self.forecast_dates)]
            forecast_out.index = self.forecast_dates
        
        elif method == 'csv': # reading from a csv file
            forecast_dates_csv = self.get_forecast_days_csv()
            load_csv_file_path = self.root + csv_path
            df_csv = pd.read_csv(load_csv_file_path, header=None, names=['ts', 'yhat'])
            df_csv = pd.concat([df_csv, df_csv], axis=0)
            if len(df_csv) < len(forecast_dates_csv):
                forecast_out = {'yhat':None}
                self.logger.error("Passed data from CSV is not long enough")
            else:
                df_csv.index = forecast_dates_csv
                df_csv.drop(['ts'], axis=1, inplace=True)
                forecast_out = df_csv.copy().loc[self.forecast_dates]
        
        elif method == 'list': # reading a list of values
            forecast_dates_csv = self.get_forecast_days_csv()
            # Loading data from passed list
            data_list = self.params['passed_data']['load_power_forecast']
            data_list = data_list + data_list
            # Check if the passed data has the correct length
            if len(data_list) < len(forecast_dates_csv):
                forecast_out = {'yhat':None}
                self.logger.error("Passed data from passed list is not long enough")
            else:
                # Define index and pick correct dates
                data_dict = {'ts':forecast_dates_csv, 'yhat':data_list}
                data = pd.DataFrame.from_dict(data_dict)
                data.index = forecast_dates_csv
                data.drop(['ts'], axis=1, inplace=True)
                forecast_out = data.copy().loc[self.forecast_dates]

        else:
            self.logger.error("Passed method is not valid")

        return forecast_out['yhat']
    
    def get_load_cost_forecast(self, df_final: pd.DataFrame, method: Optional[str] = 'hp_hc_periods',
                               csv_path: Optional[str] = "/data/data_load_cost_forecast.csv") -> pd.DataFrame:
        """
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
            defaults to "/data/data_load_cost_forecast.csv"
        :type csv_path: str, optional
        :return: The input DataFrame with one additionnal column appended containing
            the load cost for each time observation.
        :rtype: pd.DataFrame

        """
        if method == 'hp_hc_periods':
            df_final[self.var_load_cost] = self.optim_conf['load_cost_hc']
            list_df_hp = []
            for key, period_hp in self.optim_conf['list_hp_periods'].items():
                list_df_hp.append(df_final[self.var_load_cost].between_time(
                    period_hp[0]['start'], period_hp[1]['end']))
            for df_hp in list_df_hp:
                df_final.loc[df_hp.index, self.var_load_cost] = self.optim_conf['load_cost_hp']
        elif method == 'csv':
            forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
            forecast_out = self.get_forecast_out_from_csv(df_final,
                                                          forecast_dates_csv,
                                                          csv_path)
            df_final[self.var_load_cost] = forecast_out
        elif method == 'list': # reading a list of values
            forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
            data_list = self.params['passed_data']['load_cost_forecast']
            if len(data_list) < len(forecast_dates_csv):
                df_final = None
                self.logger.error("Passed data from passed list is not long enough")
            else:
                forecast_out = self.get_forecast_out_from_csv(df_final,
                                                            forecast_dates_csv,
                                                            None,
                                                            data_list=data_list)
                df_final[self.var_load_cost] = forecast_out
        else:
            self.logger.error("Passed method is not valid")
            
        return df_final
    
    def get_prod_price_forecast(self, df_final: pd.DataFrame, method: Optional[str] = 'constant',
                               csv_path: Optional[str] = "/data/data_prod_price_forecast.csv") -> pd.DataFrame:
        """
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
        if method == 'constant':
            df_final[self.var_prod_price] = self.optim_conf['prod_sell_price']
        elif method == 'csv':
            forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
            forecast_out = self.get_forecast_out_from_csv(df_final,
                                                          forecast_dates_csv,
                                                          csv_path)
            df_final[self.var_prod_price] = forecast_out
        elif method == 'list': # reading a list of values
            forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
            data_list = self.params['passed_data']['prod_price_forecast']
            if len(data_list) < len(forecast_dates_csv):
                df_final = None
                self.logger.error("Passed data from passed list is not long enough")
            else:
                forecast_out = self.get_forecast_out_from_csv(df_final,
                                                            forecast_dates_csv,
                                                            None,
                                                            data_list=data_list)
                df_final[self.var_prod_price] = forecast_out
        else:
            self.logger.error("Passed method is not valid")
            
        return df_final
    
