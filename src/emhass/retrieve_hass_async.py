"""
Async WebSocket client for Home Assistant based on HomeAssistantAPI patterns.
This implementation provides persistent connections with automatic ping/pong,
graceful reconnection, and statistics support.
"""
import asyncio
import copy
import logging
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
import numpy as np
import orjson
import pandas as pd

from emhass.utils_async import set_df_index_freq
from emhass.connection_manager import get_websocket_client

logger = logging.getLogger(__name__)

class RetrieveHass:
    """
    Compatibility wrapper for EMHASS integration.
    Provides the same interface as the original RetrieveHass class.
    """

    def __init__(
        self,
        hass_url: str,
        long_lived_token: str,
        freq: pd.Timedelta,
        time_zone: timezone,
        params: str,
        emhass_conf: dict,
        logger: logging.Logger,
        get_data_from_file: bool = False,
        # auto_cleanup: bool = True,
    ):
        # print("RetrieveHass.__init__ - retrieve_hass_async")
        self.hass_url = hass_url
        self.long_lived_token = long_lived_token
        self.freq = freq
        self.time_zone = time_zone
        self.params = orjson.loads(params) if isinstance(params, str) else params
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.get_data_from_file = get_data_from_file
        # self.auto_cleanup = auto_cleanup

        # Internal state
        self._client = None

        # For compatibility with existing code
        self.var_list = []
        self.df_final = pd.DataFrame()
        self.df_weather = pd.DataFrame()
        self.df_forecast = pd.DataFrame()

    async def get_ha_config(self) -> Dict[str, Any]:
        """Get Home Assistant configuration."""
        # print("RetrieveHass.get_ha_config - retrieve_hass_async")
        # global websocket
        try:
            # print("get_ha_config_try")
            self._client = await get_websocket_client(self.hass_url, self.long_lived_token)
        except Exception as e:
            self.logger.error(f"Fout bij connectie opzetten: {e}")
            raise

        self.ha_config = await self._client.get_config()


    async def get_data(
        self,
        days_list: List[pd.Timestamp],
        var_list: List[str],
        minimal_response: bool = True,
        no_attributes: bool = True
    ) -> bool:
        """
        Retrieve data from Home Assistant.

        This method maintains compatibility with the original EMHASS interface
        while using the new WebSocket implementation.
        """
        # print("RetrieveHass.get_data - retrieve_hass_async")
        t0 = time.time()
        # if not self._client:
        #     try:
        #         await self.startup()
        #     except Exception as e:
        #         self.logger.error(f"Auto-startup failed for get_data: {e}")
        #         return False
        try:
            self._client = await asyncio.wait_for(
                get_websocket_client(self.hass_url, self.long_lived_token, self.logger),
                timeout=20.0
            )
        except asyncio.TimeoutError:
            self.logger.error("WebSocket connection timed out")
            return False
        except Exception as e:
            self.logger.error(f"Fout bij connectie opzetten: {e}")
            return False

        self.var_list = var_list

        if self.get_data_from_file:
            # Handle file-based data loading (unchanged)
            self.df_final = self._load_data_from_file()
            return True

        # Calculate time range
        start_time = min(days_list).to_pydatetime()
        end_time = datetime.now()

        # Try to get statistics data (which contains the actual historical data)
        try:
            # Get statistics data with 5-minute period for good resolution
            stats_data = await asyncio.wait_for(
                self._client.get_statistics(
                    start_time=start_time,
                    end_time=end_time,
                    statistic_ids=var_list,
                    period="5minute"
                ),
                timeout=30.0
            )

            # Convert statistics data to DataFrame
            self.df_final = self._convert_statistics_to_dataframe(stats_data, var_list)

            if self.df_final.empty:
                self.logger.warning("Statistics data conversion resulted in empty DataFrame")
                # Try with different period
                stats_data = await asyncio.wait_for(
                    self._client.get_statistics(
                        start_time=start_time,
                        end_time=end_time,
                        statistic_ids=var_list,
                        period="hour"
                    ),
                    timeout=30.0
                )
                self.df_final = self._convert_statistics_to_dataframe(stats_data, var_list)

            return not self.df_final.empty

        except Exception as e:
            self.logger.error(f"Failed to get data via WebSocket: {e}")


    async def get_statistics_data(
        self,
        days_list: List[pd.Timestamp],
        var_list: List[str],
        period: str = "hour"
    ) -> pd.DataFrame:
        """Get statistical data for entities."""
        # print("RetrieveHass.get_statistics_data - retrieve_hass_async")
        t0 = time.time()
        if not self._client:
            try:
                await self.startup()
            except Exception as e:
                self.logger.error(f"Auto-startup failed for get_statistics_data: {e}")
                raise ConnectionError(f"Failed to establish connection: {e}")

        start_time = min(days_list).to_pydatetime()
        end_time = max(days_list).to_pydatetime()

        stats_data = await self._client.get_statistics(
            start_time=start_time,
            end_time=end_time,
            statistic_ids=var_list,
            period=period
        )
        t1 = time.time()
        total = t1 - t0
        print("time_getting_statistics", total)

        return self._convert_statistics_to_dataframe(stats_data, var_list)

    async def _get_statistics_data(self, days_list: List[pd.Timestamp], var_list: List[str]) -> bool:
        """
        Retrieve data using Home Assistant statistics API for older data.
        Compatibility method for original interface.
        """
        # print("RetrieveHass._get_statistics_data - retrieve_hass_async")
        try:
            df_stats = await self.get_statistics_data(days_list, var_list)
            if not df_stats.empty:
                self.df_final = df_stats
                return True
            return False
        except Exception as e:
            self.logger.error(f"Statistics data retrieval failed: {e}")
            return False

    def prepare_data(
        self,
        var_load: str,
        load_negative: bool,
        set_zero_min: bool,
        var_replace_zero: List[str],
        var_interp: List[str],
    ) -> pd.DataFrame:
        """
        Prepare data for optimization.
        Compatibility method for original interface.
        """
        # print("RetrieveHass.prepare_data - retrieve_hass_async")
        try:
            if load_negative:  # Apply the correct sign to load power
                self.df_final[var_load + "_positive"] = -self.df_final[var_load]
            else:
                self.df_final[var_load + "_positive"] = self.df_final[var_load]
            self.df_final.drop([var_load], inplace=True, axis=1)
        except KeyError:
            self.logger.error(
                "Variable "
                + var_load
                + " was not found. This is typically because no data could be retrieved from Home Assistant"
            )
            return False
        except ValueError:
            self.logger.error(
                "sensor.power_photovoltaics and sensor.power_load_no_var_loads should not be the same"
            )
            return False
        # Confirm var_replace_zero & var_interp contain only sensors contained in var_list
        if isinstance(var_replace_zero, list):
            original_list = var_replace_zero[:]
            var_replace_zero = [
                item for item in var_replace_zero if item in self.var_list
            ]
            removed = set(original_list) - set(var_replace_zero)
            for item in removed:
                self.logger.warning(
                    f"Sensor '{item}' in var_replace_zero not found in self.var_list and has been removed."
                )
        else:
            var_replace_zero = []
        if isinstance(var_interp, list):
            original_list = var_interp[:]
            var_interp = [item for item in var_interp if item in self.var_list]
            removed = set(original_list) - set(var_interp)
            for item in removed:
                self.logger.warning(
                    f"Sensor '{item}' in var_interp not found in self.var_list and has been removed."
                )
        else:
            var_interp = []
        # Apply minimum values
        if set_zero_min:
            self.df_final.clip(lower=0.0, inplace=True, axis=1)
            self.df_final.replace(to_replace=0.0, value=np.nan, inplace=True)
        new_var_replace_zero = []
        new_var_interp = []
        # Just changing the names of variables to contain the fact that they are considered positive
        if var_replace_zero is not None:
            for string in var_replace_zero:
                new_string = string.replace(var_load, var_load + "_positive")
                new_var_replace_zero.append(new_string)
        else:
            self.logger.warning(
                "Unable to find all the sensors in sensor_replace_zero parameter"
            )
            self.logger.warning(
                "Confirm sure all sensors in sensor_replace_zero are sensor_power_photovoltaics and/or ensor_power_load_no_var_loads "
            )
            new_var_replace_zero = None
        if var_interp is not None:
            for string in var_interp:
                new_string = string.replace(var_load, var_load + "_positive")
                new_var_interp.append(new_string)
        else:
            new_var_interp = None
            self.logger.warning(
                "Unable to find all the sensors in sensor_linear_interp parameter"
            )
            self.logger.warning(
                "Confirm all sensors in sensor_linear_interp are sensor_power_photovoltaics and/or ensor_power_load_no_var_loads "
            )
        # Treating NaN replacement: either by zeros or by linear interpolation
        if new_var_replace_zero is not None:
            self.df_final[new_var_replace_zero] = self.df_final[
                new_var_replace_zero
            ].fillna(0.0)
        if new_var_interp is not None:
            self.df_final[new_var_interp] = self.df_final[new_var_interp].interpolate(
                method="linear", axis=0, limit=None
            )
            self.df_final[new_var_interp] = self.df_final[new_var_interp].fillna(0.0)
        # Setting the correct time zone on DF index
        if self.time_zone is not None:
            self.df_final.index = self.df_final.index.tz_convert(self.time_zone)
        # Drop datetimeindex duplicates on final DF
        self.df_final = self.df_final[~self.df_final.index.duplicated(keep="first")]
        return True

    @staticmethod
    def get_attr_data_dict(
        data_df: pd.DataFrame,
        idx: int,
        entity_id: str,
        device_class: str,
        unit_of_measurement: str,
        friendly_name: str,
        list_name: str,
        state: float,
    ) -> dict:
        # print("RetrieveHass.get_attr_data_dict - retrieve_hass_async")
        list_df = copy.deepcopy(data_df).loc[data_df.index[idx] :].reset_index()
        list_df.columns = ["timestamps", entity_id]
        ts_list = [str(i) for i in list_df["timestamps"].tolist()]
        vals_list = [str(np.round(i, 2)) for i in list_df[entity_id].tolist()]
        forecast_list = []
        for i, ts in enumerate(ts_list):
            datum = {}
            datum["date"] = ts
            datum[entity_id.split("sensor.")[1]] = vals_list[i]
            forecast_list.append(datum)
        data = {
            "state": f"{state:.2f}",
            "attributes": {
                "device_class": device_class,
                "unit_of_measurement": unit_of_measurement,
                "friendly_name": friendly_name,
                list_name: forecast_list,
            },
        }
        return data

    async def post_data(
        self,
        data_df: pd.DataFrame,
        idx: int,
        entity_id: str,
        device_class: str,
        unit_of_measurement: str,
        friendly_name: str,
        type_var: str,
        from_mlforecaster: bool | None = False,
        publish_prefix: str | None = "",
        save_entities: bool | None = False,
        logger_levels: str | None = "info",
        dont_post: bool | None = False,
    ) -> None:
        r"""
        Post passed data to hass.

        :param data_df: The DataFrame containing the data that will be posted \
            to hass. This should be a one columns DF or a series.
        :type data_df: pd.DataFrame
        :param idx: The int index of the location of the data within the passed \
            DataFrame. We will post just one value at a time.
        :type idx: int
        :param entity_id: The unique entity_id of the sensor in hass.
        :type entity_id: str
        :param device_class: The HASS device class for the sensor.
        :type device_class: str
        :param unit_of_measurement: The units of the sensor.
        :type unit_of_measurement: str
        :param friendly_name: The friendly name that will be used in the hass frontend.
        :type friendly_name: str
        :param type_var: A variable to indicate the type of variable: power, SOC, etc.
        :type type_var: str
        :param publish_prefix: A common prefix for all published data entity_id.
        :type publish_prefix: str, optional
        :param save_entities: if entity data should be saved in data_path/entities
        :type save_entities: bool, optional
        :param logger_levels: set logger level, info or debug, to output
        :type logger_levels: str, optional
        :param dont_post: dont post to HA
        :type dont_post: bool, optional

        """
        # print("RetrieveHass.post_data - retrieve_hass_async")
        # Add a possible prefix to the entity ID
        entity_id = entity_id.replace("sensor.", "sensor." + publish_prefix)
        # Set the URL
        if (
            self.hass_url == "http://supervisor/core/api"
        ):  # If we are using the supervisor API
            url = self.hass_url + "/states/" + entity_id
        else:  # Otherwise the Home Assistant Core API it is
            url = self.hass_url + "api/states/" + entity_id
        headers = {
            "Authorization": "Bearer " + self.long_lived_token,
            "content-type": "application/json",
        }
        # Preparing the data dict to be published
        if type_var == "cost_fun":
            if isinstance(data_df.iloc[0], pd.Series):  # if Series extract
                data_df = data_df.iloc[:, 0]
            state = np.round(data_df.sum(), 2)
        elif type_var == "unit_load_cost" or type_var == "unit_prod_price":
            state = np.round(data_df.loc[data_df.index[idx]], 4)
        elif type_var == "optim_status":
            state = data_df.loc[data_df.index[idx]]
        elif type_var == "mlregressor":
            state = data_df[idx]
        else:
            state = np.round(data_df.loc[data_df.index[idx]], 2)
        if type_var == "power":
            data = RetrieveHass.get_attr_data_dict(
                data_df,
                idx,
                entity_id,
                device_class,
                unit_of_measurement,
                friendly_name,
                "forecasts",
                state,
            )
        elif type_var == "deferrable":
            data = RetrieveHass.get_attr_data_dict(
                data_df,
                idx,
                entity_id,
                device_class,
                unit_of_measurement,
                friendly_name,
                "deferrables_schedule",
                state,
            )
        elif type_var == "temperature":
            data = RetrieveHass.get_attr_data_dict(
                data_df,
                idx,
                entity_id,
                device_class,
                unit_of_measurement,
                friendly_name,
                "predicted_temperatures",
                state,
            )
        elif type_var == "batt":
            data = RetrieveHass.get_attr_data_dict(
                data_df,
                idx,
                entity_id,
                device_class,
                unit_of_measurement,
                friendly_name,
                "battery_scheduled_power",
                state,
            )
        elif type_var == "SOC":
            data = RetrieveHass.get_attr_data_dict(
                data_df,
                idx,
                entity_id,
                device_class,
                unit_of_measurement,
                friendly_name,
                "battery_scheduled_soc",
                state,
            )
        elif type_var == "unit_load_cost":
            data = RetrieveHass.get_attr_data_dict(
                data_df,
                idx,
                entity_id,
                device_class,
                unit_of_measurement,
                friendly_name,
                "unit_load_cost_forecasts",
                state,
            )
        elif type_var == "unit_prod_price":
            data = RetrieveHass.get_attr_data_dict(
                data_df,
                idx,
                entity_id,
                device_class,
                unit_of_measurement,
                friendly_name,
                "unit_prod_price_forecasts",
                state,
            )
        elif type_var == "mlforecaster":
            data = RetrieveHass.get_attr_data_dict(
                data_df,
                idx,
                entity_id,
                device_class,
                unit_of_measurement,
                friendly_name,
                "scheduled_forecast",
                state,
            )
        elif type_var == "optim_status":
            data = {
                "state": state,
                "attributes": {
                    "device_class": device_class,
                    "unit_of_measurement": unit_of_measurement,
                    "friendly_name": friendly_name,
                },
            }
        elif type_var == "mlregressor":
            data = {
                "state": state,
                "attributes": {
                    "device_class": device_class,
                    "unit_of_measurement": unit_of_measurement,
                    "friendly_name": friendly_name,
                },
            }
        else:
            data = {
                "state": f"{state:.2f}",
                "attributes": {
                    "device_class": device_class,
                    "unit_of_measurement": unit_of_measurement,
                    "friendly_name": friendly_name,
                },
            }
        # Actually post the data
        if self.get_data_from_file or dont_post:

            class response:
                pass

            response.status_code = 200
            response.ok = True
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=orjson.dumps(data)) as response:
                    # Store response data since we need to access it after the context manager
                    response_ok = response.ok
                    response_status_code = response.status

        # Treating the response status and posting them on the logger
        if response.ok if self.get_data_from_file or dont_post else response_ok:
            if logger_levels == "DEBUG":
                self.logger.debug(
                    "Successfully posted to " + entity_id + " = " + str(state)
                )
            else:
                self.logger.info(
                    "Successfully posted to " + entity_id + " = " + str(state)
                )

            # If save entities is set, save entity data to /data_path/entities
            if save_entities:
                entities_path = self.emhass_conf["data_path"] / "entities"

                # Clarify folder exists
                pathlib.Path(entities_path).mkdir(parents=True, exist_ok=True)

                # Save entity data to json file
                result = data_df.to_json(
                    index="timestamp", orient="index", date_unit="s", date_format="iso"
                )
                parsed = orjson.loads(result)
                async with aiofiles.open(entities_path / (entity_id + ".json"), "w") as file:
                    await file.write(orjson.dumps(parsed, option=orjson.OPT_INDENT_2).decode())

                # Save the required metadata to json file
                metadata_path = entities_path / "metadata.json"
                if os.path.isfile(metadata_path):
                    async with aiofiles.open(metadata_path, "r") as file:
                        content = await file.read()
                        metadata = orjson.loads(content)
                else:
                    metadata = {}

                async with aiofiles.open(metadata_path, "w") as file:
                    # Save entity metadata, key = entity_id
                    metadata[entity_id] = {
                        "name": data_df.name,
                        "device_class": device_class,
                        "unit_of_measurement": unit_of_measurement,
                        "friendly_name": friendly_name,
                        "type_var": type_var,
                        "optimization_time_step": int(self.freq.seconds / 60),
                    }

                    # Find lowest frequency to set for continual loop freq
                    if metadata.get("lowest_time_step", None) is None or metadata[
                        "lowest_time_step"
                    ] > int(self.freq.seconds / 60):
                        metadata["lowest_time_step"] = int(self.freq.seconds / 60)
                    await file.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode())

                    self.logger.debug("Saved " + entity_id + " to json file")

        else:
            self.logger.warning(
                "The status code for received curl command response is: "
                + str(response.status_code if self.get_data_from_file or dont_post else response_status_code)
            )
        return response, data


    def _load_data_from_file(self) -> pd.DataFrame:
        """Load data from pickle file (unchanged from original)."""
        # print("RetrieveHass._load_data_from_file retrieve_hass_async")
        import pickle
        from pathlib import Path

        data_path = Path(self.emhass_conf.get("data_path", "/tmp"))
        pickle_file = data_path / "data_retrieve_hass.pkl"

        if pickle_file.exists():
            with open(pickle_file, "rb") as f:
                df_final = pickle.load(f)
            self.logger.info(f"Loaded data from file: {pickle_file}")
            return df_final
        else:
            raise FileNotFoundError(f"Data file not found: {pickle_file}")

    def _convert_history_to_dataframe(
        self,
        history_data: List[List[Dict[str, Any]]],
        var_list: List[str]
    ) -> pd.DataFrame:
        """Convert WebSocket history data to DataFrame."""
        # print("RetrieveHass._convert_history_to_dataframe retrieve_hass_async")
        import pandas as pd
        import numpy as np
        from datetime import datetime

        # Initialize empty DataFrame with DatetimeIndex
        df_final = pd.DataFrame()

        # Check if we have any data
        if not history_data:
            self.logger.warning("No history data received")
            return df_final

        # Process each entity's history
        for i, entity_history in enumerate(history_data):
            if i >= len(var_list):
                break

            entity_id = var_list[i]

            if not entity_history:
                self.logger.warning(f"No history data for {entity_id}")
                continue

            # Convert to DataFrame
            entity_data = []
            for state in entity_history:
                try:
                    # Handle both string and dict formats
                    if isinstance(state, str):
                        # Skip string entries that are not state objects
                        continue

                    # Handle WebSocket compressed state format
                    if "last_changed" in state:
                        # REST API format
                        timestamp = pd.to_datetime(state["last_changed"])
                        value = state["state"]
                    elif "last_updated" in state:
                        # WebSocket compressed format - use timestamp
                        if isinstance(state["last_updated"], (int, float)):
                            timestamp = pd.to_datetime(state["last_updated"], unit='s')
                        else:
                            timestamp = pd.to_datetime(state["last_updated"])
                        value = state["state"]
                    elif "lu" in state:
                        # WebSocket compressed format - 'lu' = last_updated, 's' = state
                        if isinstance(state["lu"], (int, float)):
                            timestamp = pd.to_datetime(state["lu"], unit='s')
                        else:
                            timestamp = pd.to_datetime(state["lu"])
                        value = state["s"]
                    else:
                        # Skip if no timestamp available
                        continue

                    # Try to convert to numeric
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        # Handle non-numeric states
                        if value in ["on", "off"]:
                            value = 1 if value == "on" else 0
                        elif value in ["unknown", "unavailable", ""]:
                            value = np.nan
                        else:
                            continue

                    entity_data.append({
                        "timestamp": timestamp,
                        entity_id: value
                    })
                except (KeyError, ValueError, TypeError) as e:
                    self.logger.debug(f"Skipping invalid state for {entity_id}: {e}")
                    continue

            if entity_data:
                entity_df = pd.DataFrame(entity_data)
                entity_df.set_index("timestamp", inplace=True)

                if df_final.empty:
                    df_final = entity_df
                else:
                    df_final = df_final.join(entity_df, how="outer")

        # Process the final DataFrame
        if not df_final.empty:
            # Ensure timezone awareness
            if df_final.index.tz is None:
                df_final.index = df_final.index.tz_localize(self.time_zone)
            else:
                df_final.index = df_final.index.tz_convert(self.time_zone)

            # Sort by index
            df_final = df_final.sort_index()

            # Resample to frequency
            df_final = df_final.resample(self.freq).mean()

            # Forward fill missing values
            df_final = df_final.ffill()

            # Set frequency for the DataFrame index
            df_final = set_df_index_freq(df_final)

        return df_final

    def _convert_statistics_to_dataframe(
        self,
        stats_data: Dict[str, Any],
        var_list: List[str]
    ) -> pd.DataFrame:
        """Convert WebSocket statistics data to DataFrame."""
        # print("RetrieveHass._convert_statistics_to_dataframe retrieve_hass_async")
        import pandas as pd

        # Initialize empty DataFrame
        df_final = pd.DataFrame()

        # The websocket manager already extracts the 'result' portion
        # so stats_data should be directly the entity data dictionary

        for entity_id in var_list:
            if entity_id not in stats_data:
                self.logger.warning(f"No statistics data for {entity_id}")
                continue

            entity_stats = stats_data[entity_id]


            if not entity_stats:
                continue

            # Convert statistics to DataFrame
            entity_data = []
            for i, stat in enumerate(entity_stats):
                try:

                    # Handle timestamp from start time (milliseconds or ISO string)
                    if isinstance(stat["start"], (int, float)):
                        # Convert from milliseconds to datetime with UTC timezone
                        timestamp = pd.to_datetime(stat["start"], unit='ms', utc=True)
                    else:
                        # Assume ISO string
                        timestamp = pd.to_datetime(stat["start"], utc=True)

                    # Use mean, max, min or sum depending on what's available
                    value = None
                    if "mean" in stat and stat["mean"] is not None:
                        value = stat["mean"]
                    elif "sum" in stat and stat["sum"] is not None:
                        value = stat["sum"]
                    elif "max" in stat and stat["max"] is not None:
                        value = stat["max"]
                    elif "min" in stat and stat["min"] is not None:
                        value = stat["min"]

                    if value is not None:
                        try:
                            value = float(value)
                            entity_data.append({
                                "timestamp": timestamp,
                                entity_id: value
                            })
                        except (ValueError, TypeError):
                            self.logger.debug(f"Could not convert value to float: {value}")

                except (KeyError, ValueError, TypeError) as e:
                    self.logger.debug(f"Skipping invalid statistic for {entity_id}: {e}")
                    continue


            if entity_data:
                entity_df = pd.DataFrame(entity_data)
                entity_df.set_index("timestamp", inplace=True)

                if df_final.empty:
                    df_final = entity_df
                else:
                    df_final = df_final.join(entity_df, how="outer")

        # Process the final DataFrame
        if not df_final.empty:
            # Ensure timezone awareness - timestamps should already be UTC from conversion above
            if df_final.index.tz is None:
                # If somehow still naive, localize as UTC first then convert
                df_final.index = df_final.index.tz_localize('UTC').tz_convert(self.time_zone)
            else:
                # Convert from existing timezone to target timezone
                df_final.index = df_final.index.tz_convert(self.time_zone)

            # Sort by index
            df_final = df_final.sort_index()

            # Resample to frequency if needed
            try:
                original_shape = df_final.shape
                df_final = df_final.resample(self.freq).mean()
            except Exception as e:
                self.logger.warning(f"Could not resample data to {self.freq}: {e}")


            # Forward fill missing values
            df_final = df_final.ffill()

            # Set frequency for the DataFrame index
            df_final = set_df_index_freq(df_final)

        return df_final
