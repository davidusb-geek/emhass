# -*- coding: utf-8 -*-
import pickle
import random
import numpy as np
import pandas as pd
import pathlib
import plotly.express as px
import plotly.subplots as sp
import plotly.io as pio
pio.renderers.default = 'browser'
pd.options.plotting.backend = "plotly"

from emhass.retrieve_hass import RetrieveHass
from emhass.optimization import Optimization
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
    show_figures = True
    template = 'presentation'
    
    retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(emhass_conf, use_secrets=False)
    retrieve_hass_conf, optim_conf, plant_conf = \
        retrieve_hass_conf, optim_conf, plant_conf
    rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                            retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'],
                            params, emhass_conf, logger)
    if get_data_from_file:
        with open(emhass_conf['data_path'] / 'test_df_final.pkl', 'rb') as inp:
            rh.df_final, days_list, var_list = pickle.load(inp)
        retrieve_hass_conf['var_load'] = str(var_list[0])
        retrieve_hass_conf['var_PV'] = str(var_list[1])
        retrieve_hass_conf['var_interp'] = [retrieve_hass_conf['var_PV'], retrieve_hass_conf['var_load']]
        retrieve_hass_conf['var_replace_zero'] = [retrieve_hass_conf['var_PV']]
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
    
    fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf,
                    params, emhass_conf, logger, get_data_from_file=get_data_from_file)
    df_weather = fcst.get_weather_forecast(method='csv')
    P_PV_forecast = fcst.get_power_from_weather(df_weather)
    P_load_forecast = fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
    df_input_data = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
    df_input_data.columns = ['P_PV_forecast', 'P_load_forecast']
    
    df_input_data = fcst.get_load_cost_forecast(df_input_data)
    df_input_data = fcst.get_prod_price_forecast(df_input_data)
    input_data_dict = {'retrieve_hass_conf': retrieve_hass_conf}
    
    # Set special debug cases
    
    # Solver configurations
    optim_conf.update({'lp_solver': 'PULP_CBC_CMD'}) # set the name of the linear programming solver that will be used. Options are 'PULP_CBC_CMD', 'GLPK_CMD' and 'COIN_CMD'. 
    optim_conf.update({'lp_solver_path': 'empty'})  # set the path to the LP solver, COIN_CMD default is /usr/bin/cbc
    
    # Config for a single thermal model
    optim_conf.update({'num_def_loads': 1})
    optim_conf.update({'P_deferrable_nom': [1000.0]})
    optim_conf.update({'def_total_hours': [0]})
    optim_conf.update({'def_start_timestep': [0]})
    optim_conf.update({'def_end_timestep': [0]})
    optim_conf.update({'treat_def_as_semi_cont': [False]})
    optim_conf.update({'set_def_constant': [False]})
    optim_conf.update({'def_start_penalty': [0.0]})
    
    # Thermal modeling
    df_input_data['outdoor_temperature_forecast'] = [random.normalvariate(10.0, 3.0) for _ in range(48)]
    
    runtimeparams = {
        'def_load_config': [
            {'thermal_config': {
                'heating_rate': 5.0,
                'cooling_constant': 0.1,
                'overshoot_temperature': 24.0,
                'start_temperature': 20,
                'desired_temperatures': [21]*48,
                }
            }
        ]
    }
    if 'def_load_config' in runtimeparams:
        optim_conf["def_load_config"] = runtimeparams['def_load_config']

    costfun = 'profit'
    opt = Optimization(retrieve_hass_conf, optim_conf, plant_conf, 
                       fcst.var_load_cost, fcst.var_prod_price,  
                       costfun, emhass_conf, logger)
    P_PV_forecast.loc[:] = 0
    P_load_forecast.loc[:] = 0
    
    df_input_data.loc[df_input_data.index[25:30],'unit_load_cost'] = 2.0 # A price peak
    unit_load_cost = df_input_data[opt.var_load_cost].values # €/kWh
    unit_prod_price = df_input_data[opt.var_prod_price].values # €/kWh
    
    
    opt_res_dayahead = opt.perform_optimization(df_input_data, P_PV_forecast.values.ravel(), 
                                                P_load_forecast.values.ravel(), 
                                                unit_load_cost, unit_prod_price, debug=True)
    
    # Let's plot the input data
    fig_inputs_dah = df_input_data.plot()
    fig_inputs_dah.layout.template = template
    fig_inputs_dah.update_yaxes(title_text = "Powers (W) and Costs(EUR)")
    fig_inputs_dah.update_xaxes(title_text = "Time")
    if show_figures:
        fig_inputs_dah.show()
    
    vars_to_plot = ['P_deferrable0', 'unit_load_cost', 'predicted_temp_heater0', 'target_temp_heater0', 'P_def_start_0']
    if plant_conf['inverter_is_hybrid']:
        vars_to_plot = vars_to_plot + ['P_hybrid_inverter']
    if plant_conf['compute_curtailment']:
        vars_to_plot = vars_to_plot + ['P_PV_curtailment']
    if optim_conf['set_use_battery']:
        vars_to_plot = vars_to_plot + ['P_batt'] + ['SOC_opt']
    fig_res_dah = opt_res_dayahead[vars_to_plot].plot() # 'P_def_start_0', 'P_def_start_1', 'P_def_bin2_0', 'P_def_bin2_1'
    fig_res_dah.layout.template = template
    fig_res_dah.update_yaxes(title_text = "Powers (W)")
    fig_res_dah.update_xaxes(title_text = "Time")
    if show_figures:
        fig_res_dah.show()
    
    print("System with: PV, two deferrable loads, dayahead optimization, profit >> total cost function sum: "+\
        str(opt_res_dayahead['cost_profit'].sum())+", Status: "+opt_res_dayahead['optim_status'].unique().item())
    
    print(opt_res_dayahead[vars_to_plot])
    