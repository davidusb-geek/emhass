import asyncio
import copy
import logging
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Any

import aiofiles
import aiohttp
import numpy as np
import orjson
import pandas as pd

from emhass.connection_manager import get_websocket_client
from emhass.utils import set_df_index_freq

logger = logging.getLogger(__name__)


class RetrieveHass:
    r"""
    Retrieve data from Home Assistant using the restful API.

    This class allows the user to retrieve data from a Home Assistant instance \
    using the provided restful API (https://developers.home-assistant.io/docs/api/rest/)

    This class methods are:

    - get_data: to retrieve the actual data from hass

    - prepare_data: to apply some data treatment in preparation for the optimization task

    - post_data: Post passed data to hass

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
        get_data_from_file: bool | None = False,
    ) -> None:
        """
        Define constructor for RetrieveHass class.

        :param hass_url: The URL of the Home Assistant instance
        :type hass_url: str
        :param long_lived_token: The long lived token retrieved from the configuration pane
        :type long_lived_token: str
        :param freq: The frequency of the data DateTimeIndexes
        :type freq: pd.TimeDelta
        :param time_zone: The time zone
        :type time_zone: datetime.timezone
        :param params: Configuration parameters passed from data/options.json
        :type params: str
        :param emhass_conf: Dictionary containing the needed emhass paths
        :type emhass_conf: dict
        :param logger: The passed logger object
        :type logger: logging object
        :param get_data_from_file: Select if data should be retrieved from a
        previously saved pickle useful for testing or directly from connection to
        hass database
        :type get_data_from_file: bool, optional

        """
        self.hass_url = hass_url
        self.long_lived_token = long_lived_token
        self.freq = freq
        self.time_zone = time_zone
        if (params is None) or (params == "null"):
            self.params = {}
        elif type(params) is dict:
            self.params = params
        else:
            self.params = orjson.loads(params)
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.get_data_from_file = get_data_from_file
        self.var_list = []
        self.use_websocket = self.params.get("retrieve_hass_conf", {}).get("use_websocket", False)
        if self.use_websocket:
            self._client = None
        else:
            self.logger.debug("Websocket integration disabled, using Home Assistant API")
        # Initialize InfluxDB configuration
        self.use_influxdb = self.params.get("retrieve_hass_conf", {}).get("use_influxdb", False)
        if self.use_influxdb:
            influx_conf = self.params.get("retrieve_hass_conf", {})
            self.influxdb_host = influx_conf.get("influxdb_host", "localhost")
            self.influxdb_port = influx_conf.get("influxdb_port", 8086)
            self.influxdb_username = influx_conf.get("influxdb_username", "")
            self.influxdb_password = influx_conf.get("influxdb_password", "")
            self.influxdb_database = influx_conf.get("influxdb_database", "homeassistant")
            self.influxdb_measurement = influx_conf.get("influxdb_measurement", "W")
            self.influxdb_retention_policy = influx_conf.get("influxdb_retention_policy", "autogen")
            self.influxdb_use_ssl = influx_conf.get("influxdb_use_ssl", False)
            self.influxdb_verify_ssl = influx_conf.get("influxdb_verify_ssl", False)
            self.logger.info(
                f"InfluxDB integration enabled: {self.influxdb_host}:{self.influxdb_port}/{self.influxdb_database}"
            )
        else:
            self.logger.debug("InfluxDB integration disabled, using Home Assistant API")

    async def get_ha_config(self):
        """
        Extract some configuration data from HA.

        :rtype: bool
        """
        # Initialize empty config immediately for safety
        self.ha_config = {}

        # Check if variables are None, empty strings, or explicitly set to "empty"
        if (
            not self.hass_url
            or self.hass_url == "empty"
            or not self.long_lived_token
            or self.long_lived_token == "empty"
        ):
            self.logger.info(
                "No Home Assistant URL or Long Lived Token found. Using only local configuration file."
            )
            return True

        # Use WebSocket if configured
        if self.use_websocket:
            return await self.get_ha_config_websocket()

        self.logger.info("get HA config from rest api.")

        # Set up headers
        headers = {
            "Authorization": "Bearer " + self.long_lived_token,
            "content-type": "application/json",
        }

        # Construct the URL (incorporating the PR's helpful checks)
        # The Supervisor API sometimes uses a different path structure
        if self.hass_url == "http://supervisor/core/api":
            url = self.hass_url + "/config"
        else:
            # Helpful check for users who forget the trailing slash
            if not self.hass_url.endswith("/"):
                self.logger.warning(
                    "The defined HA URL is missing a trailing slash </>. Appending it, but please fix your configuration."
                )
                self.hass_url = self.hass_url + "/"
            url = self.hass_url + "api/config"

        # Attempt the connection
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    # Check for HTTP errors (404, 401, 500) before trying to parse JSON
                    response.raise_for_status()
                    data = await response.read()
                    self.ha_config = orjson.loads(data)
                    return True

        except Exception as e:
            # Granular Error Logging
            # We log the specific error 'e' so the user knows if it's a Timeout, Connection Refused, or 401 Auth error
            self.logger.error(f"Unable to obtain configuration from Home Assistant at: {url}")
            self.logger.error(f"Error details: {e}")

            # Helpful hint for Add-on users without confusing Docker users
            if "supervisor" in self.hass_url:
                self.logger.error(
                    "If using the add-on, try setting url and token to 'empty' to force local config."
                )

            return False

    async def get_ha_config_websocket(self) -> dict[str, Any]:
        """Get Home Assistant configuration."""
        try:
            self._client = await get_websocket_client(
                self.hass_url, self.long_lived_token, self.logger
            )
            self.ha_config = await self._client.get_config()
            return self.ha_config
        except Exception as e:
            self.logger.error(
                f"EMHASS was unable to obtain configuration data from Home Assistant through websocket: {e}"
            )
            raise

    async def get_data(
        self,
        days_list: pd.date_range,
        var_list: list,
        minimal_response: bool | None = False,
        significant_changes_only: bool | None = False,
        test_url: str | None = "empty",
    ) -> None:
        r"""
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
        # Use WebSockets if configured, otherwise use Home Assistant REST API
        if self.use_websocket:
            success = await self.get_data_websocket(days_list, var_list)
            if not success:
                self.logger.warning("WebSocket data retrieval failed, falling back to REST API")
                # Fall back to REST API if websocket fails
                return await self._get_data_rest_api(
                    days_list,
                    var_list,
                    minimal_response,
                    significant_changes_only,
                    test_url,
                )
            return success

        # Use InfluxDB if configured, otherwise use Home Assistant API
        if self.use_influxdb:
            return self.get_data_influxdb(days_list, var_list)

        self.logger.info("Using REST API for data retrieval")
        return await self._get_data_rest_api(
            days_list, var_list, minimal_response, significant_changes_only, test_url
        )

    async def _get_data_rest_api(
        self,
        days_list: pd.date_range,
        var_list: list,
        minimal_response: bool | None = False,
        significant_changes_only: bool | None = False,
        test_url: str | None = "empty",
    ) -> None:
        """Internal method to handle REST API data retrieval."""
        self.logger.info("Retrieve hass get data method initiated...")
        headers = {
            "Authorization": "Bearer " + self.long_lived_token,
            "content-type": "application/json",
        }
        # Remove empty strings from var_list
        var_list = [var for var in var_list if var != ""]
        # Looping on each day from days list
        self.df_final = pd.DataFrame()

        async with aiohttp.ClientSession() as session:
            x = 0  # iterate based on days

            for day in days_list:
                for i, var in enumerate(var_list):
                    if test_url == "empty":
                        if (
                            self.hass_url == "http://supervisor/core/api"
                        ):  # If we are using the supervisor API
                            url = (
                                self.hass_url
                                + "/history/period/"
                                + day.isoformat()
                                + "?filter_entity_id="
                                + var
                            )
                        else:  # Otherwise the Home Assistant Core API it is
                            if self.hass_url[-1] != "/":
                                self.logger.warning(
                                    "Missing slash </> at the end of the defined URL, appending a slash but please fix your URL"
                                )
                                self.hass_url = self.hass_url + "/"
                            url = (
                                self.hass_url
                                + "api/history/period/"
                                + day.isoformat()
                                + "?filter_entity_id="
                                + var
                            )
                        if minimal_response:  # A support for minimal response
                            url = url + "?minimal_response"
                        if significant_changes_only:  # And for signicant changes only (check the HASS restful API for more info)
                            url = url + "?significant_changes_only"
                    else:
                        url = test_url

                    try:
                        async with session.get(url, headers=headers) as response:
                            response.raise_for_status()
                            data = await response.read()
                            data_list = orjson.loads(data)
                    except Exception:
                        self.logger.error("Unable to access Home Assistant instance, check URL")
                        self.logger.error("If using addon, try setting url and token to 'empty'")
                        return False
                    else:
                        if response.status == 401:
                            self.logger.error("Unable to access Home Assistant instance, TOKEN/KEY")
                            self.logger.error(
                                "If using addon, try setting url and token to 'empty'"
                            )
                            return False
                        if response.status > 299:
                            self.logger.error(
                                f"Home assistant request GET error: {response.status} for var {var}"
                            )
                            return False
                    """import bz2 # Uncomment to save a serialized data for tests
                    import _pickle as cPickle
                    with bz2.BZ2File("data/test_response_get_data_get_method.pbz2", "w") as f:
                        cPickle.dump(response, f)"""

                    try:  # Sometimes when there are connection problems we need to catch empty retrieved json
                        data = data_list[0]
                    except IndexError:
                        if x == 0:
                            self.logger.error(
                                "The retrieved JSON is empty, A sensor:"
                                + var
                                + " may have 0 days of history, passed sensor may not be correct, or days to retrieve is set too high. Check your Logger configuration, ensuring the sensors are in the include list."
                            )
                        else:
                            self.logger.error(
                                "The retrieved JSON is empty for day:"
                                + str(day)
                                + ", days_to_retrieve may be larger than the recorded history of sensor:"
                                + var
                                + " (check your recorder settings)"
                            )
                        return False

                    df_raw = pd.DataFrame.from_dict(data)
                    if len(df_raw) == 0:
                        if x == 0:
                            self.logger.error(
                                "The retrieved Dataframe is empty, A sensor:"
                                + var
                                + " may have 0 days of history or passed sensor may not be correct"
                            )
                        else:
                            self.logger.error(
                                "Retrieved empty Dataframe for day:"
                                + str(day)
                                + ", days_to_retrieve may be larger than the recorded history of sensor:"
                                + var
                                + " (check your recorder settings)"
                            )
                        return False

                    if (
                        len(df_raw) < ((60 / (self.freq.seconds / 60)) * 24)
                        and x != len(days_list) - 1
                    ):  # check if there is enough Dataframes for passed frequency per day (not inc current day)
                        self.logger.debug(
                            "sensor:"
                            + var
                            + " retrieved Dataframe count: "
                            + str(len(df_raw))
                            + ", on day: "
                            + str(day)
                            + ". This is less than freq value passed: "
                            + str(self.freq)
                        )

                    if i == 0:  # Defining the DataFrame container
                        from_date = pd.to_datetime(df_raw["last_changed"], format="ISO8601").min()
                        to_date = pd.to_datetime(df_raw["last_changed"], format="ISO8601").max()
                        ts = pd.to_datetime(
                            pd.date_range(start=from_date, end=to_date, freq=self.freq),
                            format="%Y-%d-%m %H:%M",
                        ).round(self.freq, ambiguous="infer", nonexistent="shift_forward")
                        df_day = pd.DataFrame(index=ts)

                    # Caution with undefined string data: unknown, unavailable, etc.
                    df_tp = (
                        df_raw.copy()[["state"]]
                        .replace(["unknown", "unavailable", ""], np.nan)
                        .astype(float)
                        .rename(columns={"state": var})
                    )
                    # Setting index, resampling and concatenation
                    df_tp.set_index(
                        pd.to_datetime(df_raw["last_changed"], format="ISO8601"),
                        inplace=True,
                    )
                    df_tp = df_tp.resample(self.freq).mean()
                    df_day = pd.concat([df_day, df_tp], axis=1)
                self.df_final = pd.concat([self.df_final, df_day], axis=0)
                x += 1

        self.df_final = set_df_index_freq(self.df_final)
        if self.df_final.index.freq != self.freq:
            self.logger.error(
                "The inferred freq:"
                + str(self.df_final.index.freq)
                + " from data is not equal to the defined freq in passed:"
                + str(self.freq)
            )
            return False
        self.var_list = var_list
        return True

    async def get_data_websocket(
        self,
        days_list: pd.date_range,
        var_list: list[str],
    ) -> bool:
        r"""
        Retrieve the actual data from hass.

        :param days_list: A list of days to retrieve. The ISO format should be used \
            and the timezone is UTC. The frequency of the data_range should be freq='D'
        :type days_list: pandas.date_range
        :param var_list: The list of variables to retrive from hass. These should \
            be the exact name of the sensor in Home Assistant. \
            For example: ['sensor.home_load', 'sensor.home_pv']
        :type var_list: list
        :return: The DataFrame populated with the retrieved data from hass
        :rtype: pandas.DataFrame
        """
        try:
            self._client = await asyncio.wait_for(
                get_websocket_client(self.hass_url, self.long_lived_token, self.logger),
                timeout=20.0,
            )
        except TimeoutError:
            self.logger.error("WebSocket connection timed out")
            return False
        except Exception as e:
            self.logger.error(f"Websocket connection error: {e}")
            return False

        self.var_list = var_list

        # Calculate time range
        start_time = min(days_list).to_pydatetime()
        end_time = datetime.now()

        # Try to get statistics data (which contains the actual historical data)
        try:
            # Get statistics data with 5-minute period for good resolution
            t0 = time.time()
            stats_data = await asyncio.wait_for(
                self._client.get_statistics(
                    start_time=start_time,
                    end_time=end_time,
                    statistic_ids=var_list,
                    period="5minute",
                ),
                timeout=30.0,
            )

            # Convert statistics data to DataFrame
            self.df_final = self._convert_statistics_to_dataframe(stats_data, var_list)

            t1 = time.time()
            self.logger.info(f"Statistics data retrieval took {t1 - t0:.2f} seconds")

            return not self.df_final.empty

        except Exception as e:
            self.logger.error(f"Failed to get data via WebSocket: {e}")
            return False

    def get_data_influxdb(
        self,
        days_list: pd.date_range,
        var_list: list,
    ) -> bool:
        """
        Retrieve data from InfluxDB database.

        This method provides an alternative data source to Home Assistant API,
        enabling longer historical data retention for better machine learning model training.

        :param days_list: A list of days to retrieve data for
        :type days_list: pandas.date_range
        :param var_list: List of sensor entity IDs to retrieve
        :type var_list: list
        :return: Success status of data retrieval
        :rtype: bool
        """
        self.logger.info("Retrieve InfluxDB get data method initiated...")

        # Check for empty inputs
        if not days_list.size:
            self.logger.error("Empty days_list provided")
            return False

        client = self._init_influx_client()
        if not client:
            return False

        # Convert all timestamps to UTC for comparison, then make naive for InfluxDB
        # This ensures we compare actual instants in time, not wall clock times
        # InfluxDB queries expect naive UTC timestamps (with 'Z' suffix)

        # Normalize start_time to pd.Timestamp in UTC
        start_time = pd.Timestamp(days_list[0])
        if start_time.tz is not None:
            start_time = start_time.tz_convert("UTC").tz_localize(None)
        # If naive, assume it's already UTC

        # Get current time in UTC
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)

        # Normalize requested_end to pd.Timestamp in UTC
        requested_end = pd.Timestamp(days_list[-1]) + pd.Timedelta(days=1)
        if requested_end.tz is not None:
            requested_end = requested_end.tz_convert("UTC").tz_localize(None)
        # If naive, assume it's already UTC

        # Cap end_time at current time to avoid querying future data
        # This prevents FILL(previous) from creating fake future datapoints
        end_time = min(now, requested_end)
        total_days = (end_time - start_time).days

        self.logger.info(f"Retrieving {len(var_list)} sensors over {total_days} days from InfluxDB")
        self.logger.debug(f"Time range: {start_time} to {end_time}")
        if end_time < requested_end:
            self.logger.debug(f"End time capped at current time (requested: {requested_end})")

        # Collect sensor dataframes
        sensor_dfs = []
        global_min_time = None
        global_max_time = None

        for sensor in filter(None, var_list):
            df_sensor = self._fetch_sensor_data(client, sensor, start_time, end_time)
            if df_sensor is not None:
                sensor_dfs.append(df_sensor)
                # Track global time range
                sensor_min = df_sensor.index.min()
                sensor_max = df_sensor.index.max()
                global_min_time = min(global_min_time or sensor_min, sensor_min)
                global_max_time = max(global_max_time or sensor_max, sensor_max)

        client.close()

        if not sensor_dfs:
            self.logger.error("No data retrieved from InfluxDB")
            return False

        # Create complete time index covering all sensors
        if global_min_time is not None and global_max_time is not None:
            complete_index = pd.date_range(
                start=global_min_time, end=global_max_time, freq=self.freq
            )
            self.df_final = pd.DataFrame(index=complete_index)

            # Merge all sensor dataframes
            for df_sensor in sensor_dfs:
                self.df_final = pd.concat([self.df_final, df_sensor], axis=1)

        # Set frequency and validate with error handling
        try:
            self.df_final = set_df_index_freq(self.df_final)
        except Exception as e:
            self.logger.error(f"Exception occurred while setting DataFrame index frequency: {e}")
            return False

        if self.df_final.index.freq != self.freq:
            self.logger.warning(
                f"InfluxDB data frequency ({self.df_final.index.freq}) differs from expected ({self.freq})"
            )

        self.var_list = var_list
        self.logger.info(f"InfluxDB data retrieval completed: {self.df_final.shape}")
        return True

    def _init_influx_client(self):
        """Initialize InfluxDB client connection."""
        try:
            from influxdb import InfluxDBClient
        except ImportError:
            self.logger.error("InfluxDB client not installed. Install with: pip install influxdb")
            return None

        try:
            client = InfluxDBClient(
                host=self.influxdb_host,
                port=self.influxdb_port,
                username=self.influxdb_username or None,
                password=self.influxdb_password or None,
                database=self.influxdb_database,
                ssl=self.influxdb_use_ssl,
                verify_ssl=self.influxdb_verify_ssl,
            )
            # Test connection
            client.ping()
            self.logger.debug(
                f"Successfully connected to InfluxDB at {self.influxdb_host}:{self.influxdb_port}"
            )

            # Initialize measurement cache
            if not hasattr(self, "_measurement_cache"):
                self._measurement_cache = {}

            return client
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {e}")
            return None

    def _discover_entity_measurement(self, client, entity_id: str) -> str:
        """Auto-discover which measurement contains the given entity."""
        # Check cache first
        if entity_id in self._measurement_cache:
            return self._measurement_cache[entity_id]

        try:
            # Get all available measurements
            measurements_query = "SHOW MEASUREMENTS"
            measurements_result = client.query(measurements_query)
            measurements = [m["name"] for m in measurements_result.get_points()]

            # Priority order: check common sensor types first
            priority_measurements = ["EUR/kWh", "€/kWh", "W", "EUR", "€", "%", "A", "V"]
            all_measurements = priority_measurements + [
                m for m in measurements if m not in priority_measurements
            ]

            self.logger.debug(
                f"Searching for entity '{entity_id}' across {len(measurements)} measurements"
            )

            # Search for entity in each measurement
            for measurement in all_measurements:
                if measurement not in measurements:
                    continue  # Skip if measurement doesn't exist

                try:
                    # Use SHOW TAG VALUES to get all entity_ids in this measurement
                    tag_query = f'SHOW TAG VALUES FROM "{measurement}" WITH KEY = "entity_id"'
                    self.logger.debug(
                        f"Checking measurement '{measurement}' with tag query: {tag_query}"
                    )
                    result = client.query(tag_query)
                    points = list(result.get_points())

                    # Check if our target entity_id is in the tag values
                    for point in points:
                        if point.get("value") == entity_id:
                            self.logger.debug(
                                f"Found entity '{entity_id}' in measurement '{measurement}'"
                            )
                            # Cache the result
                            self._measurement_cache[entity_id] = measurement
                            return measurement

                except Exception as query_error:
                    self.logger.debug(
                        f"Tag query failed for measurement '{measurement}': {query_error}"
                    )
                    continue

        except Exception as e:
            self.logger.error(f"Error discovering measurement for entity {entity_id}: {e}")

        # Fallback to default measurement if not found
        self.logger.warning(
            f"Entity '{entity_id}' not found in any measurement, using default: {self.influxdb_measurement}"
        )
        return self.influxdb_measurement

    def _build_influx_query_for_measurement(
        self, entity_id: str, measurement: str, start_time, end_time
    ) -> str:
        """Build InfluxQL query for specific measurement and entity."""
        # Convert frequency to InfluxDB interval
        freq_minutes = int(self.freq.total_seconds() / 60)
        interval = f"{freq_minutes}m"

        # Format times properly for InfluxDB
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Use FILL(previous) instead of FILL(linear) for compatibility with open-source InfluxDB
        query = f"""
        SELECT mean("value") AS "mean_value"
        FROM "{self.influxdb_database}"."{self.influxdb_retention_policy}"."{measurement}"
        WHERE time >= '{start_time_str}'
        AND time < '{end_time_str}'
        AND "entity_id"='{entity_id}'
        GROUP BY time({interval}) FILL(previous)
        """
        return query

    def _build_influx_query(self, sensor: str, start_time, end_time) -> str:
        """Build InfluxQL query for sensor data retrieval (legacy method)."""
        # Convert sensor name: sensor.sec_pac_solar -> sec_pac_solar
        entity_id = sensor.replace("sensor.", "") if sensor.startswith("sensor.") else sensor

        # Use default measurement (for backward compatibility)
        return self._build_influx_query_for_measurement(
            entity_id, self.influxdb_measurement, start_time, end_time
        )

    def _fetch_sensor_data(self, client, sensor: str, start_time, end_time):
        """Fetch and process data for a single sensor with auto-discovery."""
        self.logger.debug(f"Retrieving sensor: {sensor}")

        # Clean sensor name (remove sensor. prefix if present)
        entity_id = sensor.replace("sensor.", "") if sensor.startswith("sensor.") else sensor

        # Auto-discover which measurement contains this entity
        measurement = self._discover_entity_measurement(client, entity_id)
        if not measurement:
            self.logger.warning(f"Entity '{entity_id}' not found in any InfluxDB measurement")
            return None

        try:
            query = self._build_influx_query_for_measurement(
                entity_id, measurement, start_time, end_time
            )
            self.logger.debug(f"InfluxDB query: {query}")

            # Execute query
            result = client.query(query)
            points = list(result.get_points())

            if not points:
                self.logger.warning(
                    f"No data found for entity: {entity_id} in measurement: {measurement}"
                )
                return None

            self.logger.info(f"Retrieved {len(points)} data points for {sensor}")

            # Create DataFrame from points
            df_sensor = pd.DataFrame(points)

            # Convert time column and set as index with timezone awareness
            df_sensor["time"] = pd.to_datetime(df_sensor["time"], utc=True)
            df_sensor.set_index("time", inplace=True)

            # Rename value column to original sensor name
            if "mean_value" in df_sensor.columns:
                df_sensor = df_sensor[["mean_value"]].rename(columns={"mean_value": sensor})
            else:
                self.logger.error(
                    f"Expected 'mean_value' column not found for {sensor} in measurement {measurement}"
                )
                return None

            # Handle non-numeric data with NaN ratio warning
            df_sensor[sensor] = pd.to_numeric(df_sensor[sensor], errors="coerce")

            # Check proportion of NaNs and log warning if high
            nan_count = df_sensor[sensor].isna().sum()
            total_count = len(df_sensor[sensor])
            if total_count > 0:
                nan_ratio = nan_count / total_count
                if nan_ratio > 0.2:
                    self.logger.warning(
                        f"Entity '{entity_id}' has {nan_count}/{total_count} ({nan_ratio:.1%}) non-numeric values coerced to NaN."
                    )

            self.logger.debug(
                f"Successfully retrieved {len(df_sensor)} data points for '{entity_id}' from measurement '{measurement}'"
            )
            return df_sensor

        except Exception as e:
            self.logger.error(
                f"Failed to query entity {entity_id} from measurement {measurement}: {e}"
            )
            return None

    def prepare_data(
        self,
        var_load: str,
        load_negative: bool,
        set_zero_min: bool,
        var_replace_zero: list[str],
        var_interp: list[str],
    ) -> bool:
        r"""
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
        self.logger.debug("prepare_data self.var_list=%s", self.var_list)
        self.logger.debug("prepare_data var_load=%s", var_load)
        self.logger.debug("prepare_data load_negative=%s", load_negative)
        self.logger.debug("prepare_data set_zero_min=%s", set_zero_min)
        self.logger.debug("prepare_data var_replace_zero=%s", var_replace_zero)
        self.logger.debug("prepare_data var_interp=%s", var_interp)
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
            var_replace_zero = [item for item in var_replace_zero if item in self.var_list]
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
            self.logger.warning("Unable to find all the sensors in sensor_replace_zero parameter")
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
            self.logger.warning("Unable to find all the sensors in sensor_linear_interp parameter")
            self.logger.warning(
                "Confirm all sensors in sensor_linear_interp are sensor_power_photovoltaics and/or ensor_power_load_no_var_loads "
            )
        # Treating NaN replacement: either by zeros or by linear interpolation
        if new_var_replace_zero is not None:
            self.df_final[new_var_replace_zero] = self.df_final[new_var_replace_zero].fillna(0.0)
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
        decimals: int = 2,
    ) -> dict:
        list_df = copy.deepcopy(data_df).loc[data_df.index[idx] :].reset_index()
        list_df.columns = ["timestamps", entity_id]
        ts_list = [i.isoformat() for i in list_df["timestamps"].tolist()]
        vals_list = [str(np.round(i, decimals)) for i in list_df[entity_id].tolist()]
        forecast_list = []
        for i, ts in enumerate(ts_list):
            datum = {}
            datum["date"] = ts
            datum[entity_id.split("sensor.")[1]] = vals_list[i]
            forecast_list.append(datum)
        data = {
            "state": f"{state:.{decimals}f}",
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
        Post passed data to hass using REST API.

        .. note:: This method ALWAYS uses the REST API for posting data to Home Assistant,
                  regardless of the use_websocket setting. WebSocket is only used for
                  data retrieval, not for publishing/posting data.

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
        # Add a possible prefix to the entity ID
        entity_id = entity_id.replace("sensor.", "sensor." + publish_prefix)
        # Set the URL
        if self.hass_url == "http://supervisor/core/api":  # If we are using the supervisor API
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
            state = float(data_df[idx])
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
                decimals=4,
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
                decimals=4,
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
        elif type_var == "energy":
            data = RetrieveHass.get_attr_data_dict(
                data_df,
                idx,
                entity_id,
                device_class,
                unit_of_measurement,
                friendly_name,
                "heating_demand_forecast",
                state,
            )
        elif type_var == "optim_status":
            data = {
                "state": state,
                "attributes": {
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
            # Create mock response for file mode or dont_post mode
            self.logger.debug(
                f"Skipping actual POST (get_data_from_file={self.get_data_from_file}, dont_post={dont_post})"
            )
            response_ok = True
            response_status_code = 200
        else:
            # Always use REST API for posting data, regardless of use_websocket setting
            self.logger.debug(f"Posting data to URL: {url}")
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers, data=orjson.dumps(data).decode("utf-8")
                    ) as response:
                        # Store response data since we need to access it after the context manager
                        response_ok = response.ok
                        response_status_code = response.status
                        self.logger.debug(
                            f"HTTP POST response: ok={response_ok}, status={response_status_code}"
                        )
            except Exception as e:
                self.logger.error(f"Failed to post data to {entity_id}: {e}")
                response_ok = False
                response_status_code = 500

        # Treating the response status and posting them on the logger
        if response_ok:
            if logger_levels == "DEBUG" or dont_post:
                self.logger.debug("Successfully posted to " + entity_id + " = " + str(state))
            else:
                self.logger.info("Successfully posted to " + entity_id + " = " + str(state))

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
                    async with aiofiles.open(metadata_path) as file:
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
                    if metadata.get("lowest_time_step") is None or metadata[
                        "lowest_time_step"
                    ] > int(self.freq.seconds / 60):
                        metadata["lowest_time_step"] = int(self.freq.seconds / 60)
                    await file.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode())

                    self.logger.debug("Saved " + entity_id + " to json file")

        else:
            self.logger.warning(
                f"Failed to post data to {entity_id}. Status code: {response_status_code}"
            )

        # Create a response object to maintain compatibility
        class MockResponse:
            def __init__(self, ok, status_code):
                self.ok = ok
                self.status_code = status_code

        mock_response = MockResponse(response_ok, response_status_code)
        self.logger.debug(f"Completed post_data for {entity_id}")
        return mock_response, data

    def _convert_statistics_to_dataframe(
        self, stats_data: dict[str, Any], var_list: list[str]
    ) -> pd.DataFrame:
        """Convert WebSocket statistics data to DataFrame."""
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
            for _i, stat in enumerate(entity_stats):
                try:
                    # Handle timestamp from start time (milliseconds or ISO string)
                    if isinstance(stat["start"], int | float):
                        # Convert from milliseconds to datetime with UTC timezone
                        timestamp = pd.to_datetime(stat["start"], unit="ms", utc=True)
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
                            entity_data.append({"timestamp": timestamp, entity_id: value})
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
                df_final.index = df_final.index.tz_localize("UTC").tz_convert(self.time_zone)
            else:
                # Convert from existing timezone to target timezone
                df_final.index = df_final.index.tz_convert(self.time_zone)

            # Sort by index
            df_final = df_final.sort_index()

            # Resample to frequency if needed
            try:
                df_final = df_final.resample(self.freq).mean()
            except Exception as e:
                self.logger.warning(f"Could not resample data to {self.freq}: {e}")

            # Forward fill missing values
            df_final = df_final.ffill()

            # Set frequency for the DataFrame index
            df_final = set_df_index_freq(df_final)

        return df_final
