import os
import aiohttp
import pandas as pd
import copy
from itertools import zip_longest
import re

class WeatherForecastHelper:
    def __init__(self, forecast_instance):
        self.forecast_instance = forecast_instance

    async def get_open_meteo_forecast(self, w_forecast_cache_path):
        if not os.path.isfile(w_forecast_cache_path):
            data_raw = await self.forecast_instance.get_cached_open_meteo_forecast_json(
                self.forecast_instance.optim_conf["open_meteo_cache_max_age"],
                self.forecast_instance.optim_conf["delta_forecast_daily"].days
            )
            data_15min = pd.DataFrame.from_dict(data_raw["minutely_15"])
            data_15min["time"] = pd.to_datetime(data_15min["time"])
            data_15min.set_index("time", inplace=True)
            data_15min.index = data_15min.index.tz_localize(self.forecast_instance.time_zone)

            data_15min = data_15min.rename(
                columns={
                    "temperature_2m": "temp_air",
                    "relative_humidity_2m": "relative_humidity",
                    "rain": "precipitable_water",
                    "cloud_cover": "cloud_cover",
                    "wind_speed_10m": "wind_speed",
                    "shortwave_radiation_instant": "ghi",
                    "diffuse_radiation_instant": "dhi",
                    "direct_normal_irradiance_instant": "dni",
                }
            )

            data = data_15min.reindex(self.forecast_instance.forecast_dates)
            data.interpolate(
                method="linear",
                axis=0,
                limit=None,
                limit_direction="both",
                inplace=True,
            )
            data = set_df_index_freq(data)
            index_utc = data.index.tz_convert("utc")
            index_tz = index_utc.round(
                freq=data.index.freq, ambiguous="infer", nonexistent="shift_forward"
            ).tz_convert(self.forecast_instance.time_zone)
            data.index = index_tz
            data = set_df_index_freq(data)

            # Convert mm to cm and clip the minimum value to 0.1 cm as expected by PVLib
            data["precipitable_water"] = (data["precipitable_water"] / 10).clip(
                lower=0.1
            )

            if self.forecast_instance.use_legacy_pvlib:
                # Converting the cloud cover into Global Horizontal Irradiance with a PVLib method
                data = data.drop(columns=["ghi", "dhi", "dni"])
                ghi_est = self.forecast_instance.cloud_cover_to_irradiance(data["cloud_cover"])
                data["ghi"] = ghi_est["ghi"]
                data["dni"] = ghi_est["dni"]
                data["dhi"] = ghi_est["dhi"]

            data = await self.cache_forecast_data(w_forecast_cache_path, data)
        else:
            data = await self.forecast_instance.get_cached_forecast_data(w_forecast_cache_path)

        return data

    async def get_solcast_forecast(self, w_forecast_cache_path):
        if os.path.isfile(w_forecast_cache_path):
            data = await self.forecast_instance.get_cached_forecast_data(w_forecast_cache_path)
        else:
            if self.forecast_instance.params["passed_data"].get("weather_forecast_cache_only", False):
                self.forecast_instance.logger.error("Unable to obtain Solcast cache file.")
                self.forecast_instance.logger.error(
                    "Try running optimization again with 'weather_forecast_cache_only': false"
                )
                self.forecast_instance.logger.error(
                    "Optionally, obtain new Solcast cache with runtime parameter 'weather_forecast_cache': true in an optimization, or run the `weather-forecast-cache` action, to pull new data from Solcast and cache."
                )
                return False
            else:
                if "solcast_api_key" not in self.forecast_instance.retrieve_hass_conf:
                    self.forecast_instance.logger.error(
                        "The solcast_api_key parameter was not defined"
                    )
                    return False
                if "solcast_rooftop_id" not in self.forecast_instance.retrieve_hass_conf:
                    self.forecast_instance.logger.error(
                        "The solcast_rooftop_id parameter was not defined"
                    )
                    return False
                headers = {
                    "User-Agent": "EMHASS",
                    "Authorization": "Bearer "
                    + self.forecast_instance.retrieve_hass_conf["solcast_api_key"],
                    "content-type": "application/json",
                }
                days_solcast = int(
                    len(self.forecast_instance.forecast_dates) * self.forecast_instance.freq.seconds / 3600
                )
                roof_ids = re.split(
                    r"[,\s]+", self.forecast_instance.retrieve_hass_conf["solcast_rooftop_id"].strip()
                )
                total_data_list = [0] * len(self.forecast_instance.forecast_dates)
                async with aiohttp.ClientSession() as session:
                    for roof_id in roof_ids:
                        url = f"https://api.solcast.com.au/rooftop_sites/{roof_id}/forecasts?hours={days_solcast}"
                        async with session.get(url, headers=headers) as response:
                            if int(response.status) == 200:
                                data = await response.json()
                            elif (
                                int(response.status) == 402
                                or int(response.status) == 429
                            ):
                                self.forecast_instance.logger.error(
                                    "Solcast error: May have exceeded your subscription limit."
                                )
                                return False
                            elif int(response.status) >= 400 or (
                                int(response.status) >= 202
                                and int(response.status) <= 299
                            ):
                                self.forecast_instance.logger.error(
                                    "Solcast error: There was a issue with the solcast request, check solcast API key and rooftop ID."
                                )
                                self.forecast_instance.logger.error(
                                    "Solcast error: Check that your subscription is valid and your network can connect to Solcast."
                                )
                                return False
                            data_list = []
                            for elm in data["forecasts"]:
                                data_list.append(
                                    elm["pv_estimate"] * 1000
                                )
                            if len(data_list) < len(self.forecast_instance.forecast_dates):
                                self.forecast_instance.logger.error(
                                    "Not enough data retrieved from Solcast service, try increasing the time step or use MPC."
                                )
                                return False
                            total_data_list = [
                                total + current
                                for total, current in zip_longest(
                                    total_data_list, data_list, fillvalue=0
                                )
                            ]

                    total_data_list = total_data_list[0 : len(self.forecast_instance.forecast_dates)]
                    data_dict = {"ts": self.forecast_instance.forecast_dates, "yhat": total_data_list}
                    data = pd.DataFrame.from_dict(data_dict)
                    data.set_index("ts", inplace=True)

                    data = await self.cache_forecast_data(w_forecast_cache_path, data)

        return data

    async def get_solar_forecast(self, w_forecast_cache_path):
        if os.path.isfile(w_forecast_cache_path):
            data = await self.forecast_instance.get_cached_forecast_data(w_forecast_cache_path)
        else:
            if "solar_forecast_kwp" not in self.forecast_instance.retrieve_hass_conf:
                self.forecast_instance.logger.warning(
                    "The solar_forecast_kwp parameter was not defined, using dummy values for testing"
                )
                self.forecast_instance.retrieve_hass_conf["solar_forecast_kwp"] = 5
            if self.forecast_instance.retrieve_hass_conf["solar_forecast_kwp"] == 0:
                self.forecast_instance.logger.warning(
                    "The solar_forecast_kwp parameter is set to zero, setting to default 5"
                )
                self.forecast_instance.retrieve_hass_conf["solar_forecast_kwp"] = 5
            if self.forecast_instance.optim_conf["delta_forecast_daily"].days > 1:
                self.forecast_instance.logger.warning(
                    "The free public tier for solar.forecast only provides one day forecasts"
                )
                self.forecast_instance.logger.warning(
                    "Continuing with just the first day of data, the other days are filled with 0.0."
                )
                self.forecast_instance.logger.warning(
                    "Use the other available methods for delta_forecast_daily > 1"
                )
            headers = {"Accept": "application/json"}
            data = pd.DataFrame()

            async with aiohttp.ClientSession() as session:
                for i in range(len(self.forecast_instance.plant_conf["pv_module_model"])):
                    url = (
                        "https://api.forecast.solar/estimate/"
                        + str(round(self.forecast_instance.lat, 2))
                        + "/"
                        + str(round(self.forecast_instance.lon, 2))
                        + "/"
                        + str(self.forecast_instance.plant_conf["surface_tilt"][i])
                        + "/"
                        + str(self.forecast_instance.plant_conf["surface_azimuth"][i] - 180)
                        + "/"
                        + str(self.forecast_instance.retrieve_hass_conf["solar_forecast_kwp"])
                    )
                    async with session.get(url, headers=headers) as response:
                        data_raw = await response.json()
                        data_dict = {
                            "ts": list(data_raw["result"]["watts"].keys()),
                            "yhat": list(data_raw["result"]["watts"].values()),
                        }
                        data_tmp = pd.DataFrame.from_dict(data_dict)
                        data_tmp.set_index("ts", inplace=True)
                        data_tmp.index = pd.to_datetime(data_tmp.index)
                        data_tmp = data_tmp.tz_localize(self.forecast_instance.forecast_dates.tz)
                        data_tmp = data_tmp.reindex(index=self.forecast_instance.forecast_dates)
                        mask_up_data_df = (
                            data_tmp.copy(deep=True).fillna(method="ffill").isnull()
                        )
                        mask_down_data_df = (
                            data_tmp.copy(deep=True).fillna(method="bfill").isnull()
                        )
                        data_tmp.loc[mask_up_data_df["yhat"], :] = 0.0
                        data_tmp.loc[mask_down_data_df["yhat"], :] = 0.0
                        data_tmp.interpolate(inplace=True, limit=1)
                        data_tmp = data_tmp.fillna(0.0)
                        if len(data) == 0:
                            data = copy.deepcopy(data_tmp)
                        else:
                            data = data + data_tmp

                        data = await self.cache_forecast_data(w_forecast_cache_path, data)

        return data

    def get_csv_forecast(self, csv_path):
        weather_csv_file_path = csv_path
        data = pd.read_csv(weather_csv_file_path, header=None, names=["ts", "yhat"])
        if len(data) < len(self.forecast_instance.forecast_dates):
            self.forecast_instance.logger.error("Passed data from CSV is not long enough")
        else:
            data = data.loc[data.index[0 : len(self.forecast_instance.forecast_dates)], :]
            data.index = self.forecast_instance.forecast_dates
            data.drop("ts", axis=1, inplace=True)
            data = data.copy().loc[self.forecast_instance.forecast_dates]

        return data

    def get_list_forecast(self):
        data_list = self.forecast_instance.params["passed_data"]["pv_power_forecast"]
        if (
            len(data_list) < len(self.forecast_instance.forecast_dates)
            and self.forecast_instance.params["passed_data"]["prediction_horizon"] is None
        ):
            self.forecast_instance.logger.error("Passed data from passed list is not long enough")
        else:
            data_list = data_list[0 : len(self.forecast_instance.forecast_dates)]
            data_dict = {"ts": self.forecast_instance.forecast_dates, "yhat": data_list}
            data = pd.DataFrame.from_dict(data_dict)
            data.set_index("ts", inplace=True)

        return data

    async def cache_forecast_data(self, w_forecast_cache_path, data):
        if self.forecast_instance.params["passed_data"].get("weather_forecast_cache", False):
            data = await self.forecast_instance.set_cached_forecast_data(w_forecast_cache_path, data)
        return data
