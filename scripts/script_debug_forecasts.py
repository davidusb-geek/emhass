import asyncio
import pathlib

import pandas as pd
import plotly.io as pio

from emhass.forecast import Forecast
from emhass.utils import (
    build_config,
    build_params,
    build_secrets,
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
emhass_conf["secrets_path"] = root / "secrets_emhass.yaml"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)


async def main():
    get_data_from_file = True
    template = "presentation"

    methods_list = ["solar.forecast", "solcast", "scrapper"]  #

    for k, method in enumerate(methods_list):
        # Build params with default config, weather_forecast_method=method and default secrets
        config = await build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        config["weather_forecast_method"] = method
        _, secrets = await build_secrets(
            emhass_conf,
            logger,
            secrets_path=emhass_conf["secrets_path"],
            no_response=True,
        )
        params = await build_params(emhass_conf, secrets, config, logger)

        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(params, logger)
        optim_conf["delta_forecast_daily"] = pd.Timedelta(days=2)
        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params,
            emhass_conf,
            logger,
            get_data_from_file=get_data_from_file,
        )
        df_weather = await fcst.get_weather_forecast(method=method)
        P_PV_forecast = fcst.get_power_from_weather(df_weather)
        P_PV_forecast = P_PV_forecast.to_frame(name=f"PV_forecast {method}")
        if k == 0:
            res_df = P_PV_forecast
        else:
            res_df = pd.concat([res_df, P_PV_forecast], axis=1)

    # Plot the PV data
    fig = res_df.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text="Powers (W)")
    fig.update_xaxes(title_text="Time")
    fig.show()


if __name__ == "__main__":
    asyncio.run(main())
