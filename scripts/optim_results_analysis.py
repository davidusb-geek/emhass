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

from emhass.retrieve_hass import retrieve_hass
from emhass.optimization import optimization
from emhass.forecast import forecast
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger

# the root folder
root = str(get_root(__file__, num_parent=2))
# create logger
logger, ch = get_logger(__name__, root, save_to_file=False)

def get_forecast_optim_objects(retrieve_hass_conf, optim_conf, plant_conf,
                               params, get_data_from_file):
    fcst = forecast(retrieve_hass_conf, optim_conf, plant_conf,
                    params, root, logger, get_data_from_file=get_data_from_file)
    df_weather = fcst.get_weather_forecast(method='csv')
    P_PV_forecast = fcst.get_power_from_weather(df_weather)
    P_load_forecast = fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
    df_input_data_dayahead = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
    df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
    opt = optimization(retrieve_hass_conf, optim_conf, plant_conf, 
                       fcst.var_load_cost, fcst.var_prod_price,  
                       'profit', root, logger)
    return fcst, P_PV_forecast, P_load_forecast, df_input_data_dayahead, opt

if __name__ == '__main__':
    show_figures = False
    save_figures = False
    get_data_from_file = True
    params = None
    retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), use_secrets=False)
    retrieve_hass_conf, optim_conf, plant_conf = \
        retrieve_hass_conf, optim_conf, plant_conf
    rh = retrieve_hass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                        retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'],
                        params, root, logger)
    if get_data_from_file:
        with open(pathlib.Path(root+'/data/test_df_final.pkl'), 'rb') as inp:
            rh.df_final, days_list, var_list = pickle.load(inp)
    else:
        days_list = get_days_list(retrieve_hass_conf['days_to_retrieve'])
        var_list = [retrieve_hass_conf['var_load'], retrieve_hass_conf['var_PV']]
        rh.get_data(days_list, var_list,
                        minimal_response=False, significant_changes_only=False)
    rh.prepare_data(retrieve_hass_conf['var_load'], load_negative = retrieve_hass_conf['load_negative'],
                            set_zero_min = retrieve_hass_conf['set_zero_min'], 
                            var_replace_zero = retrieve_hass_conf['var_replace_zero'], 
                            var_interp = retrieve_hass_conf['var_interp'])
    df_input_data = rh.df_final.copy()
    
    fcst, P_PV_forecast, P_load_forecast, df_input_data_dayahead, opt = \
        get_forecast_optim_objects(retrieve_hass_conf, optim_conf, plant_conf,
                                   params, get_data_from_file)
    df_input_data = fcst.get_load_cost_forecast(df_input_data)
    df_input_data = fcst.get_prod_price_forecast(df_input_data)
    
    template = 'presentation'
    
    # Let's plot the input data
    fig_inputs1 = df_input_data[['sensor.power_photovoltaics',
                                'sensor.power_load_no_var_loads_positive']].plot()
    fig_inputs1.layout.template = template
    fig_inputs1.update_yaxes(title_text = "Powers (W)")
    fig_inputs1.update_xaxes(title_text = "Time")
    if show_figures:
        fig_inputs1.show()
    if save_figures:
        fig_inputs1.write_image(root + "/docs/images/inputs_power.svg", 
                                width=1080, height=0.8*1080)
    
    fig_inputs_dah = df_input_data_dayahead.plot()
    fig_inputs_dah.layout.template = template
    fig_inputs_dah.update_yaxes(title_text = "Powers (W)")
    fig_inputs_dah.update_xaxes(title_text = "Time")
    if show_figures:
        fig_inputs_dah.show()
    if save_figures:
        fig_inputs_dah.write_image(root + "/docs/images/inputs_dayahead.svg", 
                                   width=1080, height=0.8*1080)
    
    # And then perform a dayahead optimization
    df_input_data_dayahead = fcst.get_load_cost_forecast(df_input_data_dayahead)
    df_input_data_dayahead = fcst.get_prod_price_forecast(df_input_data_dayahead)
    optim_conf['treat_def_as_semi_cont'] = [True, True]
    optim_conf['set_def_constant'] = [True, True]
    unit_load_cost = df_input_data[opt.var_load_cost].values
    unit_prod_price = df_input_data[opt.var_prod_price].values
    opt_res_dah = opt.perform_optimization(df_input_data_dayahead, P_PV_forecast.values.ravel(), 
                                           P_load_forecast.values.ravel(), 
                                           unit_load_cost, unit_prod_price,
                                           debug = True)
    # opt_res_dah = opt.perform_dayahead_forecast_optim(df_input_data_dayahead, P_PV_forecast, P_load_forecast)
    opt_res_dah['P_PV'] = df_input_data_dayahead[['P_PV_forecast']]
    fig_res_dah = opt_res_dah[['P_deferrable0', 'P_deferrable1', 'P_grid', 'P_PV',
                               'P_def_start_0', 'P_def_start_1', 'P_def_bin2_0', 'P_def_bin2_1']].plot()
    fig_res_dah.layout.template = template
    fig_res_dah.update_yaxes(title_text = "Powers (W)")
    fig_res_dah.update_xaxes(title_text = "Time")
    # if show_figures:
    fig_res_dah.show()
    if save_figures:
        fig_res_dah.write_image(root + "/docs/images/optim_results_PV_defLoads_dayaheadOptim.svg", 
                                width=1080, height=0.8*1080)
    
    print("System with: PV, two deferrable loads, dayahead optimization, profit >> total cost function sum: "+\
        str(opt_res_dah['cost_profit'].sum()))
    
    print(opt_res_dah)
    opt_res_dah.to_html('opt_res_dah.html')