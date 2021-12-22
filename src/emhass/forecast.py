#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import logging
import pvlib
from pvlib.forecast import GFS
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from emhass.retrieve_hass import retrieve_hass
from emhass.utils import get_days_list

class forecast:
    """
    Generate weather and load forecasts needed as inputs to the optimization.
    
    The weather forecast is obtained from two methods. In the first method
    the tools proposed in the PVLib package are use to retrieve data from the
    GFS global model. However this API proposed by PVLib is highly experimental
    and prone to changes. A second method uses a scrapper to the ClearOutside
    webpage which proposes detailed forecasts based on Lat/Lon locations. This
    methods seems quite stable but as any scrape method it will fail if any changes
    are made to the webpage API.
    
    The 'get_power_from_weather' method is proposed here to convert from irradiance
    data to electrical power. Again PVLib is used to model the PV plant.
    For the load forecast two methods are available.
    
    The first method allows the user to use a CSV file with their own forecast.
    With this method a more powerful external package for time series forecast
    may be used to create your own detailed load forecast.
    
    The CSV should contain no header and the timestamped data should have the
    following format:
        
    2021-04-29 00:00:00+00:00,287.07
    
    2021-04-29 00:30:00+00:00,274.27
    
    2021-04-29 01:00:00+00:00,243.38
        
    ...
    
    The second method is a naive method, also called persistance.
    It simply assumes that the forecast for a future period will be equal to the
    observed values in a past period. The past period is controlled using 
    parameter 'delta_forecast'.
    
    """

    def __init__(self, retrieve_hass_conf: dict, optim_conf: dict, plant_conf: dict, 
                 config_path: str, logger: logging.Logger, opt_time_delta: Optional[int] = 24) -> None:
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
        :param config_path: The path to the yaml configuration file
        :type config_path: str
        :param logger: The passed logger object
        :type logger: logging object
        :param opt_time_delta: The time delta in hours used to generate forecasts, 
        a value of 24 will generate 24 hours of forecast data, defaults to 24
        :type opt_time_delta: int, optional

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
        self.var_load_cost = 'unit_load_cost'
        self.var_prod_price = 'unit_prod_price'
        self.forecast_dates = pd.date_range(start=self.start_forecast, 
                                            end=self.end_forecast-self.freq, 
                                            freq=self.freq).round(self.freq)
        
        
    def get_weather_forecast(self, method: Optional[str] = 'scrapper') -> pd.DataFrame:
        """
        Get and generate weather forecast data.
        
        :param method: The desired method, options are 'scrapper' and 'pvlib', \
            defaults to 'scrapper'
        :type method: str, optional
        :return: The DataFrame containing the forecasted data
        :rtype: pd.DataFrame
        
        """
        model = GFS()
        self.logger.info("Retrieving weather forecast data using method = "+method)
        if method == 'scrapper':
            freq_scrap = pd.to_timedelta(60, "minutes")
            forecast_dates_scrap = pd.date_range(start=self.start_forecast,
                                                 end=self.end_forecast-freq_scrap, 
                                                 freq=freq_scrap).round(freq_scrap)
            forecast_dates = pd.date_range(start=self.start_forecast, 
                                           end=self.end_forecast-self.freq, freq=self.freq).round(self.freq)
            response = requests.get("https://clearoutside.com/forecast/"+str(round(self.lat, 2))+"/"+str(round(self.lon, 2)))
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find_all(id='day_0')[0]
            list_names = table.find_all(class_='fc_detail_label')
            list_tables = table.find_all('ul')[1:]
            selected_cols = [0, 1, 2, 3, 10, 12, 15] # Selected variables
            col_names = [list_names[i].get_text() for i in selected_cols]
            list_tables = [list_tables[i] for i in selected_cols]
            raw_data = pd.DataFrame(index=range(24), columns=col_names, dtype=float)
            for count_col, col in enumerate(col_names):
                list_rows = list_tables[count_col].find_all('li')
                for count_row, row in enumerate(list_rows):
                    raw_data.loc[count_row, col] = float(row.get_text())
            raw_data.set_index(forecast_dates_scrap, inplace=True)
            raw_data = raw_data.reindex(forecast_dates)
            raw_data.interpolate(method='linear', axis=0, limit=None, 
                                 limit_direction='both', inplace=True)
            model.set_location(self.time_zone, self.lat, self.lon)
            ghi_est = model.cloud_cover_to_irradiance(raw_data['Total Clouds (% Sky Obscured)'])
            data = ghi_est
            data['temp_air'] = raw_data['Temperature (°C)']
            data['wind_speed'] = raw_data['Wind Speed/Direction (mph)']*1.60934 # conversion to km/h
            data['relative_humidity'] = raw_data['Relative Humidity (%)']
            data['precipitable_water'] = pvlib.atmosphere.gueymard94_pw(
                data['temp_air'], data['relative_humidity'])
        elif method == 'pvlib':
            """
            This is experimental, the GHI forecast from PVLib is unstable and prone to
            changes. The advantage of this approach is that the resulting DF automatically
            contains the needed variables for the PV plant model
            """
            raw_data = model.get_data(self.lat, self.lon, self.start_forecast, self.end_forecast)
            data = raw_data.copy()
            data = model.rename(data)
            data['temp_air'] = model.kelvin_to_celsius(data['temp_air'])
            data['wind_speed'] = model.uv_to_speed(data)
            irrad_data = model.cloud_cover_to_irradiance(data['total_clouds'])
            data = data.join(irrad_data, how='outer')
            data = data[model.output_variables]
        else:
            self.logger.error("Passed method is not valid")
        return data
    
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
        # Transform to power (Watts)
        location = Location(latitude=self.lat, longitude=self.lon)
        temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['close_mount_glass_glass']
        cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
        cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
        module = cec_modules[self.plant_conf['module_model']]
        inverter = cec_inverters[self.plant_conf['inverter_model']]
        
        system = PVSystem(surface_tilt=self.plant_conf['surface_tilt'], 
                          surface_azimuth=self.plant_conf['surface_azimuth'],
                          module_parameters=module,
                          inverter_parameters=inverter,
                          temperature_model_parameters=temp_params,
                          modules_per_string=self.plant_conf['modules_per_string'],
                          strings_per_inverter=self.plant_conf['strings_per_inverter'])
        mc = ModelChain(system, location, aoi_model="physical")
        mc.run_model(df_weather)
        
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
    
    def get_forecast_out_from_csv(self, df_final: pd.DataFrame, 
                                  forecast_dates_csv: pd.date_range,
                                  csv_path: str) -> pd.DataFrame:
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
        days_list = get_days_list(days_min_load_forecast)
        
        if method == 'naive': # using a naive approach
            self.logger.info("Retrieving data from hass for load forecast using method = "+method)
            var_list = [self.var_load]
            var_replace_zero = None
            var_interp = [self.var_load]
            time_zone_load_foreacast = None
            
            rh = retrieve_hass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                               self.freq, time_zone_load_foreacast, self.root, self.logger)
            
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
            df_csv.index = forecast_dates_csv
            df_csv.drop(['ts'], axis=1, inplace=True)
            forecast_out = df_csv.copy().loc[self.forecast_dates]
            
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
        else:
            self.logger.error("Passed method is not valid")
            
        return df_final
    