# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import pathlib
import plotly.express as px
import plotly.subplots as sp
import plotly.io as pio
pio.renderers.default = 'browser'
pd.options.plotting.backend = "plotly"

from emhass.forecast import Forecast
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger

# the root folder
root = str(get_root(__file__, num_parent=2))
emhass_conf = {}
emhass_conf['config_path'] = pathlib.Path(root) / 'config_emhass.yaml'
emhass_conf['data_path'] = pathlib.Path(root) / 'data/'
emhass_conf['root_path'] = pathlib.Path(root)

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)

if __name__ == '__main__':

    get_data_from_file = True
    params = None
    template = 'presentation'
    
    methods_list = ['solar.forecast', 'solcast', 'scrapper'] # 
    
    for k, method in enumerate(methods_list):
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(emhass_conf)
        optim_conf['delta_forecast'] = pd.Timedelta(days=2)
        fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf,
                        params, emhass_conf, logger, get_data_from_file=get_data_from_file)
        df_weather = fcst.get_weather_forecast(method=method)
        P_PV_forecast = fcst.get_power_from_weather(df_weather)
        P_PV_forecast = P_PV_forecast.to_frame(name=f'PV_forecast {method}')
        if k == 0:
            res_df = P_PV_forecast
        else:
            res_df = pd.concat([res_df, P_PV_forecast], axis=1)
    
    # Plot the PV data
    fig = res_df.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Powers (W)")
    fig.update_xaxes(title_text = "Time")
    fig.show()

