#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from emhass.retrieve_hass import RetrieveHass
from emhass.utils import (
    build_params,
    build_secrets,
    get_days_list,
    get_logger,
    get_root,
    get_yaml_parse,
)

pio.renderers.default = "browser"
pd.options.plotting.backend = "plotly"

# the root folder
root = pathlib.Path(str(get_root(__file__, num_parent=2)))
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["config_path"] = root / "config.json"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=True)


def load_forecast(data, forecast_date, freq, template):
    """
    Forecast the load profile for the next day based on historic data.

    Parameters:
    - data: pd.DataFrame with a DateTimeIndex containing the historic load data. Must include a 'load' column.
    - forecast_date: pd.Timestamp for the date of the forecast.
    - freq: frequency of the time series (e.g., '1H' for hourly).

    Returns:
    - forecast: pd.Series with the forecasted load profile for the next day.
    - used_days: list of days used to calculate the forecast.
    """
    # Ensure the 'load' column exists
    if "load" not in data.columns:
        raise ValueError("Data must have a 'load' column.")

    # Filter historic data for the same month and day of the week
    month = forecast_date.month
    day_of_week = forecast_date.dayofweek
    historic_data = data[
        (data.index.month == month) & (data.index.dayofweek == day_of_week)
    ]
    used_days = np.unique(historic_data.index.date)

    # Align all historic data to the forecast day
    aligned_data = []
    for day in used_days:
        daily_data = data[data.index.date == pd.Timestamp(day).date()]
        aligned_daily_data = daily_data.copy()
        aligned_daily_data.index = aligned_daily_data.index.map(
            lambda x: x.replace(
                year=forecast_date.year,
                month=forecast_date.month,
                day=forecast_date.day,
            )
        )
        aligned_data.append(aligned_daily_data)

    # Combine all aligned historic data into a single DataFrame
    combined_data = pd.concat(aligned_data)

    # Compute the mean load for each timestamp
    forecast = combined_data.groupby(combined_data.index).mean()

    # Plot the results
    fig = go.Figure()
    for day in used_days:
        daily_data = data[data.index.date == pd.Timestamp(day).date()]
        aligned_daily_data = daily_data.copy()
        aligned_daily_data.index = aligned_daily_data.index.map(
            lambda x: x.replace(
                year=forecast_date.year,
                month=forecast_date.month,
                day=forecast_date.day,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=aligned_daily_data.index,
                y=aligned_daily_data["load"],
                mode="lines",
                name=f"Historic day: {day}",
                line=dict(width=1),
                opacity=0.6,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["load"],
            mode="lines",
            name="Forecast (Mean)",
            line=dict(color="red", width=3),
        )
    )
    fig.update_layout(
        title=f"Load Forecast for {forecast_date.date()}",
        xaxis_title="Time",
        yaxis_title="Load (kW)",
        legend_title="Legend",
        template=template,
        xaxis=dict(range=[forecast.index.min(), forecast.index.max()]),
        yaxis=dict(autorange=True),
    )
    fig.show()

    return forecast, used_days


if __name__ == "__main__":
    days_to_retrieve = 365
    model_type = "load_clustering"
    var_model = "sensor.power_load_positive"

    # Build params with no config and default secrets
    data_path = emhass_conf["data_path"] / str("data_train_" + model_type + ".pkl")
    template = "presentation"

    if data_path.is_file():
        logger.info("Loading a previous data file")
        _, secrets = build_secrets(emhass_conf, logger, no_response=True)
        params = build_params(emhass_conf, secrets, {}, logger)
        with open(data_path, "rb") as fid:
            data, var_model = pickle.load(fid)
    else:
        logger.info(
            "Using EMHASS methods to retrieve the new forecast model train data"
        )
        secrets_path = root / "secrets_emhass.yaml"
        emhass_conf, secrets = build_secrets(
            emhass_conf, logger, secrets_path=secrets_path
        )
        params = build_params(emhass_conf, secrets, {}, logger)
        retrieve_hass_conf, _, _ = get_yaml_parse(params, logger)
        retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(30, "minutes")

        rh = RetrieveHass(
            retrieve_hass_conf["hass_url"],
            retrieve_hass_conf["long_lived_token"],
            retrieve_hass_conf["optimization_time_step"],
            retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
            get_data_from_file=False,
        )

        days_list = get_days_list(days_to_retrieve)
        var_list = [var_model]
        rh.get_data(days_list, var_list)

        with open(data_path, "wb") as fid:
            pickle.dump((rh.df_final, var_model), fid, pickle.HIGHEST_PROTOCOL)

        data = copy.deepcopy(rh.df_final)

    logger.info(data.describe())

    # Plot the input data
    fig = data.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text="Power (W)")
    fig.update_xaxes(title_text="Time")
    fig.show()

    # Define forecast date and frequency
    forecast_date = pd.Timestamp("2023-07-15")
    freq = pd.to_timedelta(30, "minutes")

    # Call the forecasting method
    data.columns = ["load"]
    forecast, used_days = load_forecast(data, forecast_date, freq, template)