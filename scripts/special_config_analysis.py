"""
This is a script for analysis plot.
To use this script you will need plotly and kaleido. Install them using:
    pip install plotly
    pip install kaleido
Before running this script you should perform a perfect optimization for each type of cost function:
profit, cost and self-consumption
"""

import pathlib
import pickle

import pandas as pd
import plotly.io as pio

from emhass.forecast import Forecast
from emhass.optimization import Optimization
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import (
    build_config,
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
emhass_conf["scripts_path"] = root / "scripts/"
emhass_conf["config_path"] = root / "config.json"
emhass_conf["secrets_path"] = root / "secrets_emhass.yaml"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"
# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)


def get_forecast_optim_objects(retrieve_hass_conf, optim_conf, plant_conf, params, get_data_from_file):
    fcst = Forecast(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        params,
        emhass_conf,
        logger,
        get_data_from_file=get_data_from_file,
    )
    df_weather = fcst.get_weather_forecast(method="solar.forecast")
    P_PV_forecast = fcst.get_power_from_weather(df_weather)
    P_load_forecast = fcst.get_load_forecast(method=optim_conf["load_forecast_method"])
    df_input_data_dayahead = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
    df_input_data_dayahead.columns = ["P_PV_forecast", "P_load_forecast"]
    opt = Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        fcst.var_load_cost,
        fcst.var_prod_price,
        "cost",
        emhass_conf,
        logger,
    )
    return fcst, P_PV_forecast, P_load_forecast, df_input_data_dayahead, opt


if __name__ == "__main__":
    get_data_from_file = False

    # Build params with defaults, secret file, and added special config and secrets
    config = build_config(
        emhass_conf,
        logger,
        emhass_conf["defaults_path"],
        emhass_conf["scripts_path"] / "special_options.json",
    )
    emhass_conf, secrets = build_secrets(
        emhass_conf,
        logger,
        options_path=emhass_conf["scripts_path"] / "special_options.json",
        secrets_path=emhass_conf["secrets_path"],
        no_response=True,
    )
    params = build_params(emhass_conf, secrets, config, logger)

    pv_power_forecast = [
        0,
        8,
        27,
        42,
        47,
        41,
        25,
        7,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        19,
        52,
        73,
        74,
        68,
        44,
        12,
        0,
        0,
        0,
        0,
    ]
    load_power_forecast = [
        2850,
        3021,
        3107,
        3582,
        2551,
        2554,
        1856,
        2505,
        1768,
        2540,
        1722,
        2463,
        1670,
        1379,
        1165,
        1000,
        1641,
        1181,
        1861,
        1414,
        1467,
        1344,
        1209,
        1531,
    ]
    load_cost_forecast = [
        17.836,
        19.146,
        18.753,
        17.838,
        17.277,
        16.282,
        16.736,
        16.047,
        17.004,
        19.982,
        17.17,
        16.968,
        16.556,
        16.21,
        12.333,
        10.937,
    ]
    prod_price_forecast = [
        6.651,
        7.743,
        7.415,
        6.653,
        6.185,
        5.356,
        5.734,
        5.16,
        5.958,
        8.439,
        6.096,
        5.928,
        5.584,
        5.296,
        4.495,
        3.332,
    ]
    prediction_horizon = 16
    soc_init = 0.98
    soc_final = 0.3
    operating_hours_of_each_deferrable_load = [0]
    alpha = 1
    beta = 0

    params["passed_data"] = {
        "pv_power_forecast": pv_power_forecast,
        "load_power_forecast": load_power_forecast,
        "load_cost_forecast": load_cost_forecast,
        "prod_price_forecast": prod_price_forecast,
        "prediction_horizon": prediction_horizon,
        "soc_init": soc_init,
        "soc_final": soc_final,
        "operating_hours_of_each_deferrable_load": operating_hours_of_each_deferrable_load,
        "alpha": alpha,
        "beta": beta,
    }

    params["optim_conf"]["weather_forecast_method"] = "list"
    params["optim_conf"]["load_forecast_method"] = "list"
    params["optim_conf"]["load_cost_forecast_method"] = "list"
    params["optim_conf"]["production_price_forecast_method"] = "list"

    data_path = emhass_conf["scripts_path"] / "data_temp.pkl"

    retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(params, logger)

    if data_path.is_file():
        logger.info("Loading a previous data file")
        with open(data_path, "rb") as fid:
            (
                fcst,
                P_PV_forecast,
                P_load_forecast,
                df_input_data_dayahead,
                opt,
                df_input_data,
            ) = pickle.load(fid)
    else:
        rh = RetrieveHass(
            retrieve_hass_conf["hass_url"],
            retrieve_hass_conf["long_lived_token"],
            retrieve_hass_conf["optimization_time_step"],
            retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
        )
        days_list = get_days_list(retrieve_hass_conf["historic_days_to_retrieve"])
        var_list = [
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
            retrieve_hass_conf["sensor_power_photovoltaics"],
        ]
        rh.get_data(days_list, var_list, minimal_response=False, significant_changes_only=False)
        rh.prepare_data(
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=retrieve_hass_conf["load_negative"],
            set_zero_min=retrieve_hass_conf["set_zero_min"],
            var_replace_zero=retrieve_hass_conf["sensor_replace_zero"],
            var_interp=retrieve_hass_conf["sensor_linear_interp"],
        )
        df_input_data = rh.df_final.copy()
        fcst, P_PV_forecast, P_load_forecast, df_input_data_dayahead, opt = get_forecast_optim_objects(
            retrieve_hass_conf, optim_conf, plant_conf, params, get_data_from_file
        )
        df_input_data = fcst.get_load_cost_forecast(df_input_data)
        df_input_data = fcst.get_prod_price_forecast(df_input_data)

        with open(data_path, "wb") as fid:
            pickle.dump(
                (
                    fcst,
                    P_PV_forecast,
                    P_load_forecast,
                    df_input_data_dayahead,
                    opt,
                    df_input_data,
                ),
                fid,
                pickle.HIGHEST_PROTOCOL,
            )

    template = "presentation"
    y_axis_title = "Power (W)"

    # Let's plot the input data
    fig_inputs1 = df_input_data[
        [
            str(retrieve_hass_conf["sensor_power_photovoltaics"]),
            str(retrieve_hass_conf["sensor_power_load_no_var_loads"] + "_positive"),
        ]
    ].plot()
    fig_inputs1.layout.template = template
    fig_inputs1.update_yaxes(title_text=y_axis_title)
    fig_inputs1.update_xaxes(title_text="Time")
    fig_inputs1.show()

    fig_inputs2 = df_input_data[["unit_load_cost", "unit_prod_price"]].plot()
    fig_inputs2.layout.template = template
    fig_inputs2.update_yaxes(title_text="Load cost and production sell price (EUR)")
    fig_inputs2.update_xaxes(title_text="Time")
    fig_inputs2.show()

    fig_inputs_dah = df_input_data_dayahead.plot()
    fig_inputs_dah.layout.template = template
    fig_inputs_dah.update_yaxes(title_text=y_axis_title)
    fig_inputs_dah.update_xaxes(title_text="Time")
    fig_inputs_dah.show()

    # Perform a dayahead optimization
    """df_input_data_dayahead = fcst.get_load_cost_forecast(df_input_data_dayahead)
    df_input_data_dayahead = fcst.get_prod_price_forecast(df_input_data_dayahead)
    opt_res_dah = opt.perform_dayahead_forecast_optim(df_input_data_dayahead, P_PV_forecast, P_load_forecast)
    fig_res_dah = opt_res_dah[['P_deferrable0', 'P_deferrable1', 'P_grid']].plot()
    fig_res_dah.layout.template = template
    fig_res_dah.update_yaxes(title_text = y_axis_title)
    fig_res_dah.update_xaxes(title_text = "Time")
    fig_res_dah.show()"""

    '''post_mpc_optim: "curl -i -H \"Content-Type: application/json\" -X POST -d '{
        \"load_cost_forecast\":[17.836, 19.146, 18.753, 17.838, 17.277, 16.282, 16.736, 16.047, 17.004, 19.982, 17.17, 16.968, 16.556, 16.21, 12.333, 10.937],
        \"prod_price_forecast\":[6.651, 7.743, 7.415, 6.653, 6.185, 5.356, 5.734, 5.16, 5.958, 8.439, 6.096, 5.928, 5.584, 5.296, 4.495, 3.332],
        \"prediction_horizon\":16,
        \"pv_power_forecast\": [0, 8, 27, 42, 47, 41, 25, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 52, 73, 74, 68, 44, 12, 0, 0, 0, 0],
        \"alpha\": 1, \"beta\": 0, \"soc_init\":0.98, \"soc_final\":0.3, \"operating_hours_of_each_deferrable_load\":[0]
        }' http://localhost:5000/action/naive-mpc-optim"'''

    # Perform a MPC optimization
    df_input_data_dayahead["unit_load_cost"] = load_cost_forecast
    df_input_data_dayahead.loc[
        df_input_data_dayahead.index[2] : df_input_data_dayahead.index[6],
        "unit_load_cost",
    ] = 150
    df_input_data_dayahead["unit_prod_price"] = prod_price_forecast

    opt.optim_conf["weight_battery_discharge"] = 0.0
    opt.optim_conf["weight_battery_charge"] = 0.0
    opt.optim_conf["battery_dynamic_max"] = 0.9
    opt.optim_conf["set_nocharge_from_grid"] = False
    opt.optim_conf["set_nodischarge_to_grid"] = False
    opt.optim_conf["set_total_pv_sell"] = False

    opt_res_dayahead = opt.perform_naive_mpc_optim(
        df_input_data_dayahead,
        P_PV_forecast,
        P_load_forecast,
        prediction_horizon,
        soc_init=soc_init,
        soc_final=soc_final,
        def_total_hours=operating_hours_of_each_deferrable_load,
    )
    fig_res_mpc = opt_res_dayahead[["P_batt", "P_grid"]].plot()
    fig_res_mpc.layout.template = template
    fig_res_mpc.update_yaxes(title_text=y_axis_title)
    fig_res_mpc.update_xaxes(title_text="Time")
    fig_res_mpc.show()
