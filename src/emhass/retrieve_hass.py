#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
import numpy as np, pandas as pd
from requests import get, post
import json

from emhass.utils import get_logger

class retrieve_hass(ABC):
    """
    Retrieve data from Home Assistant using the restful API.
    
    This class allows the user to retrieve data from a Home Assistant instance \
    using the provided restful API (https://developers.home-assistant.io/docs/api/rest/)
    
    This class methods are:
        
    - get_data: to retrieve the actual data from hass
    
    - prepare_data: to apply some data treatment in preparation for the optimization task
    
    - post_data: Post passed data to hass
    
    """

    def __init__(self, hass_url, long_lived_token, freq, time_zone, config_path, logger):
        """
        Define constructor for retrieve_hass class.
        
        :param hass_url: The URL of the Home Assistant instance
        :type hass_url: str
        :param long_lived_token: The long lived token retrieved from the configuration pane
        :type long_lived_token: str
        :param freq: The frequency of the data DateTimeIndexes
        :type freq: pd.TimeDelta
        :param time_zone: The time zone
        :type time_zone: datetime.timezone
        :param config_path: The path to the yaml configuration file
        :type config_path: str
        :param logger: The passed logger object
        :type logger: logging object

        """
        self.hass_url = hass_url
        self.long_lived_token = long_lived_token
        self.freq = freq
        self.time_zone = time_zone
        # create logger
        self.logger, self.ch = get_logger(__name__, config_path, file=logger.fileSetting)


    def get_data(self, days_list, var_list, minimal_response = False,
                 significant_changes_only = False):
        """
        Retrieve the actual data from hass.
        
        :param days_list: A list of days to retrieve. The ISO format should be used \
            and the timezone is UTC. The frequency of the data_range should be freq='D'
        :type days_list: pandas.date_range
        :param var_list: The list of variables to retrive from hass. These should \
            be the exact name of the sensor in Home Assistant. \
            For example: ['sensor.home_load', 'sensor.home_pv']
        :type var_list: list
        :param minimal_response: Retrieve a minimal response using the hass \
            restful API, defaults to False
        :type minimal_response: bool, optional
        :param significant_changes_only: Retrieve significant changes only \
            using the hass restful API, defaults to False
        :type significant_changes_only: bool, optional
        :return: The DataFrame populated with the retrieved data from hass
        :rtype: pandas.DataFrame
        
        .. warning:: The minimal_response and significant_changes_only options \
            are experimental
        """
        self.logger.info("Retrieve hass get data method initiated...")
        self.df_final = pd.DataFrame()
        for day in days_list:
        
            for i, var in enumerate(var_list):
                
                url = self.hass_url+"api/history/period/"+day.isoformat()+"?filter_entity_id="+var
                if minimal_response:
                    url = url + "?minimal_response"
                if significant_changes_only:
                    url = url + "?significant_changes_only"
                headers = {
                    "Authorization": "Bearer " + self.long_lived_token,
                    "content-type": "application/json",
                }
                response = get(url, headers=headers)
                try:
                    data = response.json()[0]
                except IndexError:
                    self.logger.error("The retrieved JSON is empty, check that correct day or variable names are passed")
                    break
                df_raw = pd.DataFrame.from_dict(data)
                if len(df_raw) == 0:
                    self.logger.error("Retrieved empty Dataframe, check that correct day or variable names are passed")
                if i == 0:
                    from_date = pd.to_datetime(df_raw['last_changed']).min()
                    to_date = pd.to_datetime(df_raw['last_changed']).max()
                    ts = pd.to_datetime(pd.date_range(start=from_date, end=to_date, freq=self.freq), 
                                        format='%Y-%d-%m %H:%M').round(self.freq)
                    df_day = pd.DataFrame(index = ts)
                df_tp = df_raw.copy()[['state']].replace(
                    ['unknown', 'unavailable', ''], np.nan).astype(float).rename(columns={'state': var})
                df_tp.set_index(pd.to_datetime(df_raw['last_changed']), inplace=True)
                df_tp = df_tp.resample(self.freq).mean()
                df_day = pd.concat([df_day, df_tp], axis=1)
            
            self.df_final = pd.concat([self.df_final, df_day], axis=0)
        self.df_final.index.freq = self.freq

    
    def prepare_data(self, var_load, load_negative = False, set_zero_min = True,
                     var_replace_zero = None, var_interp = None):
        """
        Apply some data treatment in preparation for the optimization task.
        
        :param var_load: The name of the variable for the household load consumption.
        :type var_load: str
        :param load_negative: Set to True if the retrived load variable is \
            negative by convention, defaults to False
        :type load_negative: bool, optional
        :param set_zero_min: A special treatment for a minimum value saturation \
            to zero. Values below zero are replaced by nans, defaults to True
        :type set_zero_min: bool, optional
        :param var_replace_zero: A list of retrived variables that we would want \
            to replace nans with zeros, defaults to None
        :type var_replace_zero: list, optional
        :param var_interp: A list of retrived variables that we would want to \
            interpolate nan values using linear interpolation, defaults to None
        :type var_interp: list, optional
        :return: The DataFrame populated with the retrieved data from hass and \
            after the data treatment
        :rtype: pandas.DataFrame
        
        """
        if load_negative:
            self.df_final[var_load+'_positive'] = -self.df_final[var_load]
        else:
            self.df_final[var_load+'_positive'] = self.df_final[var_load]
        self.df_final.drop([var_load], inplace=True, axis=1)
        if set_zero_min:
            self.df_final.clip(lower=0.0, inplace=True, axis=1)
            self.df_final.replace(to_replace=0.0, value=np.nan, inplace=True)
        new_var_replace_zero = []
        new_var_interp = []
        if var_replace_zero is not None:
            for string in var_replace_zero:
                new_string = string.replace(var_load, var_load+'_positive')
                new_var_replace_zero.append(new_string)
        else:
            new_var_replace_zero = None
        if var_interp is not None:
            for string in var_interp:
                new_string = string.replace(var_load, var_load+'_positive')
                new_var_interp.append(new_string)
        else:
            new_var_interp = None
        if new_var_replace_zero is not None:
            self.df_final[new_var_replace_zero] = self.df_final[new_var_replace_zero].fillna(0.0)
        if new_var_interp is not None:
            self.df_final[new_var_interp] = self.df_final[new_var_interp].interpolate(
                method='linear', axis=0, limit=None)
        if self.time_zone is not None:
            self.df_final.index = self.df_final.index.tz_convert(self.time_zone)
        # Drop datetimeindex duplicates on final DF
        self.df_final = self.df_final[~self.df_final.index.duplicated(keep='first')]
        
    def post_data(self, data_df, idx, entity_id, 
                  unit_of_measurement, friendly_name):
        """
        Post passed data to hass.
        
        :param data_df: The DataFrame containing the data that will be posted \
            to hass. This should be a one columns DF or a series.
        :type data_df: pd.DataFrame
        :param idx: The int index of the location of the data within the passed \
            DataFrame. We will post just one value at a time.
        :type idx: int
        :param entity_id: The unique entity_id of the sensor in hass.
        :type entity_id: str
        :param unit_of_measurement: The units of the sensor.
        :type unit_of_measurement: str
        :param friendly_name: The friendly name that will be used in the hass frontend.
        :type friendly_name: str

        """
        url = self.hass_url+"api/states/"+entity_id
        headers = {
            "Authorization": "Bearer " + self.long_lived_token,
            "content-type": "application/json",
        }
        data = {
            "state": str(data_df.loc[data_df.index[idx]]),
            "attributes": {
                "unit_of_measurement": unit_of_measurement,
                "friendly_name": friendly_name
            }
        }
        response = post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            self.logger.info("Successfully posted value in existing entity_id")
        elif response.status_code == 201:
            self.logger.info("Successfully posted value in a newly created entity_id")
        elif response.status_code == 400:
            self.logger.info("Error posting value to HASS: Bad Request")
        elif response.status_code == 401:
            self.logger.info("Error posting value to HASS: Unauthorized")
        elif response.status_code == 404:
            self.logger.info("Error posting value to HASS: Not Found")
        elif response.status_code == 405:
            self.logger.info("Error posting value to HASS: Method not allowed")
        else:
            self.logger.info("The received response code is not recognized")
        