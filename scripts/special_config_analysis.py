# -*- coding: utf-8 -*-
'''
    This is a script for analysis plot.
    To use this script you will need plotly and kaleido. Install them using: 
        pip install plotly
        pip install kaleido
    Before running this script you should perform a perfect optimization for each type of cost function:
    profit, cost and self-consumption 
'''
import numpy as np
import pandas as pd
import pathlib
import yaml
import json
import pickle
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
# create logger
logger, ch = get_logger(__name__, root, save_to_file=False)

def get_forecast_optim_objects(retrieve_hass_conf, optim_conf, plant_conf,
                               params, get_data_from_file):
    fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf,
                    params, root, logger, get_data_from_file=get_data_from_file)
    df_weather = fcst.get_weather_forecast(method='solar.forecast')
    P_PV_forecast = fcst.get_power_from_weather(df_weather)
    P_load_forecast = fcst.get_load_forecast(method=optim_conf['load_forecast_method'])
    df_input_data_dayahead = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
    df_input_data_dayahead.columns = ['P_PV_forecast', 'P_load_forecast']
    opt = Optimization(retrieve_hass_conf, optim_conf, plant_conf, 
                       fcst.var_load_cost, fcst.var_prod_price,  
                       'cost', root, logger)
    return fcst, P_PV_forecast, P_load_forecast, df_input_data_dayahead, opt

def build_params(params, options):
    # Updating variables in retrieve_hass_conf
    params['retrieve_hass_conf'][0]['freq'] = options['optimization_time_step']
    params['retrieve_hass_conf'][1]['days_to_retrieve'] = options['historic_days_to_retrieve']
    params['retrieve_hass_conf'][2]['var_PV'] = options['sensor_power_photovoltaics']
    params['retrieve_hass_conf'][3]['var_load'] = options['sensor_power_load_no_var_loads']
    params['retrieve_hass_conf'][6]['var_replace_zero'] = [options['sensor_power_photovoltaics']]
    params['retrieve_hass_conf'][7]['var_interp'] = [options['sensor_power_photovoltaics'], options['sensor_power_load_no_var_loads']]
    params['retrieve_hass_conf'][8]['method_ts_round'] = options['method_ts_round']
    # Updating variables in optim_conf
    params['optim_conf'][0]['set_use_battery'] = options['set_use_battery']
    params['optim_conf'][2]['num_def_loads'] = options['number_of_deferrable_loads']
    params['optim_conf'][3]['P_deferrable_nom'] = [i['nominal_power_of_deferrable_loads'] for i in options['list_nominal_power_of_deferrable_loads']]
    params['optim_conf'][4]['def_total_hours'] = [i['operating_hours_of_each_deferrable_load'] for i in options['list_operating_hours_of_each_deferrable_load']]
    params['optim_conf'][5]['treat_def_as_semi_cont'] = [i['treat_deferrable_load_as_semi_cont'] for i in options['list_treat_deferrable_load_as_semi_cont']]
    params['optim_conf'][6]['set_def_constant'] = [False for i in range(len(params['optim_conf'][3]['P_deferrable_nom']))]
    params['optim_conf'][8]['load_forecast_method'] = options['load_forecast_method']
    start_hours_list = [i['peak_hours_periods_start_hours'] for i in options['list_peak_hours_periods_start_hours']]
    end_hours_list = [i['peak_hours_periods_end_hours'] for i in options['list_peak_hours_periods_end_hours']]
    num_peak_hours = len(start_hours_list)
    list_hp_periods_list = [{'period_hp_'+str(i+1):[{'start':start_hours_list[i]},{'end':end_hours_list[i]}]} for i in range(num_peak_hours)]
    params['optim_conf'][10]['list_hp_periods'] = list_hp_periods_list
    params['optim_conf'][11]['load_cost_hp'] = options['load_peak_hours_cost']
    params['optim_conf'][12]['load_cost_hc'] = options['load_offpeak_hours_cost']
    params['optim_conf'][14]['prod_sell_price'] = options['photovoltaic_production_sell_price']
    params['optim_conf'][15]['set_total_pv_sell'] = options['set_total_pv_sell']
    params['optim_conf'][16]['lp_solver'] = options['lp_solver']
    params['optim_conf'][17]['lp_solver_path'] = options['lp_solver_path']
    params['optim_conf'][18]['set_nocharge_from_grid'] = options['set_nocharge_from_grid']
    params['optim_conf'][19]['set_nodischarge_to_grid'] = options['set_nodischarge_to_grid']
    params['optim_conf'][20]['set_battery_dynamic'] = options['set_battery_dynamic']
    params['optim_conf'][21]['battery_dynamic_max'] = options['battery_dynamic_max']
    params['optim_conf'][22]['battery_dynamic_min'] = options['battery_dynamic_min']
    params['optim_conf'][23]['weight_battery_discharge'] = options['weight_battery_discharge']
    params['optim_conf'][24]['weight_battery_charge'] = options['weight_battery_charge']
    # Updating variables in plant_conf
    params['plant_conf'][0]['P_grid_max'] = options['maximum_power_from_grid']
    params['plant_conf'][1]['module_model'] = [i['pv_module_model'] for i in options['list_pv_module_model']]
    params['plant_conf'][2]['inverter_model'] = [i['pv_inverter_model'] for i in options['list_pv_inverter_model']]
    params['plant_conf'][3]['surface_tilt'] = [i['surface_tilt'] for i in options['list_surface_tilt']]
    params['plant_conf'][4]['surface_azimuth'] = [i['surface_azimuth'] for i in options['list_surface_azimuth']]
    params['plant_conf'][5]['modules_per_string'] = [i['modules_per_string'] for i in options['list_modules_per_string']]
    params['plant_conf'][6]['strings_per_inverter'] = [i['strings_per_inverter'] for i in options['list_strings_per_inverter']]
    params['plant_conf'][7]['Pd_max'] = options['battery_discharge_power_max']
    params['plant_conf'][8]['Pc_max'] = options['battery_charge_power_max']
    params['plant_conf'][9]['eta_disch'] = options['battery_discharge_efficiency']
    params['plant_conf'][10]['eta_ch'] = options['battery_charge_efficiency']
    params['plant_conf'][11]['Enom'] = options['battery_nominal_energy_capacity']
    params['plant_conf'][12]['SOCmin'] = options['battery_minimum_state_of_charge']
    params['plant_conf'][13]['SOCmax'] = options['battery_maximum_state_of_charge']
    params['plant_conf'][14]['SOCtarget'] = options['battery_target_state_of_charge']
    return params

if __name__ == '__main__':
    get_data_from_file = False
    config_path = pathlib.Path(root+'/config_emhass.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    retrieve_hass_conf = config['retrieve_hass_conf']
    optim_conf = config['optim_conf']
    plant_conf = config['plant_conf']
    
    params = {}
    params['retrieve_hass_conf'] = retrieve_hass_conf
    params['optim_conf'] = optim_conf
    params['plant_conf'] = plant_conf
    
    options_json = pathlib.Path(root+'/scripts/special_options.json')
    with options_json.open('r') as data:
        options = json.load(data)
    
    params = build_params(params, options)
    
    with open(pathlib.Path(root) / 'secrets_emhass.yaml', 'r') as file:
        input_secrets = yaml.load(file, Loader=yaml.FullLoader)
        
    params['params_secrets'] = input_secrets
    
    pv_power_forecast = [0, 8, 27, 42, 47, 41, 25, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 52, 73, 74, 68, 44, 12, 0, 0, 0, 0]
    load_power_forecast = [2850, 3021, 3107, 3582, 2551, 2554, 1856, 2505, 1768, 2540, 1722, 2463, 1670, 1379, 1165, 1000, 1641, 1181, 1861, 1414, 1467, 1344, 1209, 1531]
    load_cost_forecast = [17.836, 19.146, 18.753, 17.838, 17.277, 16.282, 16.736, 16.047, 17.004, 19.982, 17.17, 16.968, 16.556, 16.21, 12.333, 10.937]
    prod_price_forecast = [6.651, 7.743, 7.415, 6.653, 6.185, 5.356, 5.734, 5.16, 5.958, 8.439, 6.096, 5.928, 5.584, 5.296, 4.495, 3.332]
    prediction_horizon = 16
    soc_init = 0.98
    soc_final = 0.3
    def_total_hours = [0]
    alpha = 1
    beta = 0
    
    params['passed_data'] = {'pv_power_forecast':pv_power_forecast,'load_power_forecast':load_power_forecast,
                             'load_cost_forecast':load_cost_forecast,'prod_price_forecast':prod_price_forecast,
                             'prediction_horizon':prediction_horizon,'soc_init':soc_init,'soc_final':soc_final,
                             'def_total_hours':def_total_hours,'alpha':alpha,'beta':beta}
    
    optim_conf[7]['weather_forecast_method'] = 'list'
    optim_conf[8]['load_forecast_method'] = 'list'
    optim_conf[9]['load_cost_forecast_method'] = 'list'
    optim_conf[13]['prod_price_forecast_method'] = 'list'
    
    data_path = pathlib.Path(root+'/scripts/data_temp.pkl')
    
    if data_path.is_file():
        logger.info("Loading a previous data file")
        with open(data_path, "rb") as fid:
            fcst, P_PV_forecast, P_load_forecast, df_input_data_dayahead, opt, df_input_data = pickle.load(fid)
    else:
    
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(config_path, use_secrets=True, params = json.dumps(params))
        rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                          retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'],
                          params, root, logger)
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
                                    json.dumps(params), get_data_from_file)
        df_input_data = fcst.get_load_cost_forecast(df_input_data)
        df_input_data = fcst.get_prod_price_forecast(df_input_data)
    
        with open(data_path, 'wb') as fid:
            pickle.dump((fcst, P_PV_forecast, P_load_forecast, df_input_data_dayahead, opt, df_input_data), fid, pickle.HIGHEST_PROTOCOL)
    
    template = 'presentation'
    
    # Let's plot the input data
    fig_inputs1 = df_input_data[['sensor.power_photovoltaics',
                                 'sensor.power_load_no_var_loads_positive']].plot()
    fig_inputs1.layout.template = template
    fig_inputs1.update_yaxes(title_text = "Powers (W)")
    fig_inputs1.update_xaxes(title_text = "Time")
    fig_inputs1.show()
    
    fig_inputs2 = df_input_data[['unit_load_cost',
                                 'unit_prod_price']].plot()
    fig_inputs2.layout.template = template
    fig_inputs2.update_yaxes(title_text = "Load cost and production sell price (EUR)")
    fig_inputs2.update_xaxes(title_text = "Time")
    fig_inputs2.show()
    
    fig_inputs_dah = df_input_data_dayahead.plot()
    fig_inputs_dah.layout.template = template
    fig_inputs_dah.update_yaxes(title_text = "Powers (W)")
    fig_inputs_dah.update_xaxes(title_text = "Time")
    fig_inputs_dah.show()
    
    # Perform a dayahead optimization
    '''df_input_data_dayahead = fcst.get_load_cost_forecast(df_input_data_dayahead)
    df_input_data_dayahead = fcst.get_prod_price_forecast(df_input_data_dayahead)
    opt_res_dah = opt.perform_dayahead_forecast_optim(df_input_data_dayahead, P_PV_forecast, P_load_forecast)
    fig_res_dah = opt_res_dah[['P_deferrable0', 'P_deferrable1', 'P_grid']].plot()
    fig_res_dah.layout.template = template
    fig_res_dah.update_yaxes(title_text = "Powers (W)")
    fig_res_dah.update_xaxes(title_text = "Time")
    fig_res_dah.show()'''
    
    '''post_mpc_optim: "curl -i -H \"Content-Type: application/json\" -X POST -d '{ 
        \"load_cost_forecast\":[17.836, 19.146, 18.753, 17.838, 17.277, 16.282, 16.736, 16.047, 17.004, 19.982, 17.17, 16.968, 16.556, 16.21, 12.333, 10.937],  
        \"prod_price_forecast\":[6.651, 7.743, 7.415, 6.653, 6.185, 5.356, 5.734, 5.16, 5.958, 8.439, 6.096, 5.928, 5.584, 5.296, 4.495, 3.332], 
        \"prediction_horizon\":16, 
        \"pv_power_forecast\": [0, 8, 27, 42, 47, 41, 25, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 52, 73, 74, 68, 44, 12, 0, 0, 0, 0], 
        \"alpha\": 1, \"beta\": 0, \"soc_init\":0.98, \"soc_final\":0.3, \"def_total_hours\":[0]        
        }' http://localhost:5000/action/naive-mpc-optim"'''
    
    # Perform a MPC optimization
    df_input_data_dayahead['unit_load_cost'] = load_cost_forecast
    df_input_data_dayahead.loc[df_input_data_dayahead.index[2]:df_input_data_dayahead.index[6],'unit_load_cost'] = 150
    df_input_data_dayahead['unit_prod_price'] = prod_price_forecast
    
    opt.optim_conf['weight_battery_discharge'] = 0.0
    opt.optim_conf['weight_battery_charge'] = 0.0
    opt.optim_conf['battery_dynamic_max'] = 0.9
    opt.optim_conf['set_nocharge_from_grid'] = False
    opt.optim_conf['set_nodischarge_to_grid'] = False
    opt.optim_conf['set_total_pv_sell'] = False
    
    opt_res_dayahead = opt.perform_naive_mpc_optim(
        df_input_data_dayahead, P_PV_forecast, P_load_forecast, prediction_horizon,
        soc_init=soc_init, soc_final=soc_final, def_total_hours=def_total_hours)
    fig_res_mpc = opt_res_dayahead[['P_batt', 'P_grid']].plot()
    fig_res_mpc.layout.template = template
    fig_res_mpc.update_yaxes(title_text = "Powers (W)")
    fig_res_mpc.update_xaxes(title_text = "Time")
    fig_res_mpc.show()