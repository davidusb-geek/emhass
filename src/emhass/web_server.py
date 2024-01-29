#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from flask import Flask, request, make_response, render_template
from jinja2 import Environment, PackageLoader
from requests import get
from waitress import serve
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import os, json, argparse, pickle, yaml, logging
from distutils.util import strtobool
import pandas as pd
import plotly.express as px
pd.options.plotting.backend = "plotly"
from emhass.command_line import set_input_data_dict
from emhass.command_line import perfect_forecast_optim, dayahead_forecast_optim, naive_mpc_optim
from emhass.command_line import forecast_model_fit, forecast_model_predict, forecast_model_tune
from emhass.command_line import publish_data

# Define the Flask instance
app = Flask(__name__)

def get_injection_dict(df, plot_size = 1366):
    cols_p = [i for i in df.columns.to_list() if 'P_' in i]
    # Let's round the data in the DF
    optim_status = df['optim_status'].unique().item()
    df.drop('optim_status', axis=1, inplace=True)
    cols_else = [i for i in df.columns.to_list() if 'P_' not in i]
    df = df.apply(pd.to_numeric)
    df[cols_p] = df[cols_p].astype(int)
    df[cols_else] = df[cols_else].round(3)
    # Create plots
    n_colors = len(cols_p)
    colors = px.colors.sample_colorscale("jet", [n/(n_colors -1) for n in range(n_colors)])
    fig_0 = px.line(df[cols_p], title='Systems powers schedule after optimization results', 
                    template='presentation', line_shape="hv",
                    color_discrete_sequence=colors)
    fig_0.update_layout(xaxis_title='Timestamp', yaxis_title='System powers (W)')
    if 'SOC_opt' in df.columns.to_list():
        fig_1 = px.line(df['SOC_opt'], title='Battery state of charge schedule after optimization results', 
                        template='presentation',  line_shape="hv",
                        color_discrete_sequence=colors)
        fig_1.update_layout(xaxis_title='Timestamp', yaxis_title='Battery SOC (%)')
    cols_cost = [i for i in df.columns.to_list() if 'cost_' in i or 'unit_' in i]
    n_colors = len(cols_cost)
    colors = px.colors.sample_colorscale("jet", [n/(n_colors -1) for n in range(n_colors)])
    fig_2 = px.line(df[cols_cost], title='Systems costs obtained from optimization results', 
                    template='presentation', line_shape="hv",
                    color_discrete_sequence=colors)
    fig_2.update_layout(xaxis_title='Timestamp', yaxis_title='System costs (currency)')
    # Get full path to image
    image_path_0 = fig_0.to_html(full_html=False, default_width='75%')
    if 'SOC_opt' in df.columns.to_list():
        image_path_1 = fig_1.to_html(full_html=False, default_width='75%')
    image_path_2 = fig_2.to_html(full_html=False, default_width='75%')
    # The tables
    table1 = df.reset_index().to_html(classes='mystyle', index=False)
    cost_cols = [i for i in df.columns if 'cost_' in i]
    table2 = df[cost_cols].reset_index().sum(numeric_only=True)
    table2['optim_status'] = optim_status
    table2 = table2.to_frame(name='Value').reset_index(names='Variable').to_html(classes='mystyle', index=False)
    # The dict of plots
    injection_dict = {}
    injection_dict['title'] = '<h2>EMHASS optimization results</h2>'
    injection_dict['subsubtitle0'] = '<h4>Plotting latest optimization results</h4>'
    injection_dict['figure_0'] = image_path_0
    if 'SOC_opt' in df.columns.to_list():
        injection_dict['figure_1'] = image_path_1
    injection_dict['figure_2'] = image_path_2
    injection_dict['subsubtitle1'] = '<h4>Last run optimization results table</h4>'
    injection_dict['table1'] = table1
    injection_dict['subsubtitle2'] = '<h4>Cost totals for latest optimization results</h4>'
    injection_dict['table2'] = table2
    return injection_dict

def get_injection_dict_forecast_model_fit(df_fit_pred, mlf):
    fig = df_fit_pred.plot()
    fig.layout.template = 'presentation'
    fig.update_yaxes(title_text = mlf.model_type)
    fig.update_xaxes(title_text = "Time")
    image_path_0 = fig.to_html(full_html=False, default_width='75%')
    # The dict of plots
    injection_dict = {}
    injection_dict['title'] = '<h2>Custom machine learning forecast model fit</h2>'
    injection_dict['subsubtitle0'] = '<h4>Plotting train/test forecast model results for '+mlf.model_type+'</h4>'
    injection_dict['subsubtitle0'] = '<h4>Forecasting variable '+mlf.var_model+'</h4>'
    injection_dict['figure_0'] = image_path_0
    return injection_dict

def get_injection_dict_forecast_model_tune(df_pred_optim, mlf):
    fig = df_pred_optim.plot()
    fig.layout.template = 'presentation'
    fig.update_yaxes(title_text = mlf.model_type)
    fig.update_xaxes(title_text = "Time")
    image_path_0 = fig.to_html(full_html=False, default_width='75%')
    # The dict of plots
    injection_dict = {}
    injection_dict['title'] = '<h2>Custom machine learning forecast model tune</h2>'
    injection_dict['subsubtitle0'] = '<h4>Performed a tuning routine using bayesian optimization for '+mlf.model_type+'</h4>'
    injection_dict['subsubtitle0'] = '<h4>Forecasting variable '+mlf.var_model+'</h4>'
    injection_dict['figure_0'] = image_path_0
    return injection_dict

def build_params(params, params_secrets, options, addon):
    if addon == 1:
        # Updating variables in retrieve_hass_conf
        params['retrieve_hass_conf']['freq'] = options.get('optimization_time_step',params['retrieve_hass_conf']['freq'])
        params['retrieve_hass_conf']['days_to_retrieve'] = options.get('historic_days_to_retrieve',params['retrieve_hass_conf']['days_to_retrieve'])
        params['retrieve_hass_conf']['var_PV'] = options.get('sensor_power_photovoltaics',params['retrieve_hass_conf']['var_PV'])
        params['retrieve_hass_conf']['var_load'] = options.get('sensor_power_load_no_var_loads',params['retrieve_hass_conf']['var_load'])
        params['retrieve_hass_conf']['load_negative'] = options.get('load_negative',params['retrieve_hass_conf']['load_negative'])
        params['retrieve_hass_conf']['set_zero_min'] = options.get('set_zero_min',params['retrieve_hass_conf']['set_zero_min'])
        params['retrieve_hass_conf']['var_replace_zero'] = options.get('sensor_power_photovoltaics',params['retrieve_hass_conf']['var_replace_zero'])
        params['retrieve_hass_conf']['var_interp'] = options.get('sensor_power_photovoltaics',params['retrieve_hass_conf']['var_PV']), options.get('sensor_power_load_no_var_loads',params['retrieve_hass_conf']['var_load'])
        params['retrieve_hass_conf']['method_ts_round'] = options.get('method_ts_round',params['retrieve_hass_conf']['method_ts_round'])
        # Update params Secrets if specified
        params['params_secrets'] = params_secrets
        params['params_secrets']['time_zone'] = options.get('time_zone',params_secrets['time_zone'])
        params['params_secrets']['lat'] = options.get('Latitude',params_secrets['lat'])
        params['params_secrets']['lon'] = options.get('Longitude',params_secrets['lon'])
        params['params_secrets']['alt'] = options.get('Altitude',params_secrets['alt'])
        # Updating variables in optim_conf
        params['optim_conf']['set_use_battery'] = options.get('set_use_battery',params['optim_conf']['set_use_battery'])
        params['optim_conf']['num_def_loads'] = options.get('number_of_deferrable_loads',params['optim_conf']['num_def_loads'])
        try:
            params['optim_conf']['P_deferrable_nom'] = [i['nominal_power_of_deferrable_loads'] for i in options.get('list_nominal_power_of_deferrable_loads')]
        except:
            app.logger.debug("no list_nominal_power_of_deferrable_loads, defaulting to config file")
        try:    
            params['optim_conf']['def_total_hours'] = [i['operating_hours_of_each_deferrable_load'] for i in options.get('list_operating_hours_of_each_deferrable_load')]
        except:
             app.logger.debug("no list_operating_hours_of_each_deferrable_load, defaulting to config file")
        try:
            params['optim_conf']['treat_def_as_semi_cont'] = [i['treat_deferrable_load_as_semi_cont'] for i in options.get('list_treat_deferrable_load_as_semi_cont')]
        except:
             app.logger.debug("no list_treat_deferrable_load_as_semi_cont, defaulting to config file")
        params['optim_conf']['weather_forecast_method'] = options.get('weather_forecast_method',params['optim_conf']['weather_forecast_method'])
        if params['optim_conf']['weather_forecast_method'] == "solcast":
            params['retrieve_hass_conf']['solcast_api_key'] = options.get('optional_solcast_api_key',params['retrieve_hass_conf']['solcast_api_key'])
            params['retrieve_hass_conf']['solcast_rooftop_id'] = options.get('optional_solcast_rooftop_id',params['retrieve_hass_conf']['solcast_rooftop_id'])
        elif params['optim_conf']['weather_forecast_method'] == "solar.forecast":    
            params['retrieve_hass_conf']['solar_forecast_kwp'] = options.get('optional_solar_forecast_kwp',params['retrieve_hass_conf']['solar_forecast_kwp'])
        params['optim_conf']['load_forecast_method'] = options.get('load_forecast_method',params['optim_conf']['load_forecast_method'])
        params['optim_conf']['delta_forecast'] = options.get('delta_forecast_daily',params['optim_conf']['delta_forecast'])
        params['optim_conf']['load_cost_forecast_method'] = options.get('load_cost_forecast_method',params['optim_conf']['load_cost_forecast_method'])
        try:
            params['optim_conf']['set_def_constant'] = [i['set_deferrable_load_single_constant'] for i in options.get('list_set_deferrable_load_single_constant')]
        except:
            app.logger.debug("no list_set_deferrable_load_single_constant, defaulting to config file")
        try:
            start_hours_list = [i['peak_hours_periods_start_hours'] for i in options['list_peak_hours_periods_start_hours']]
            end_hours_list = [i['peak_hours_periods_end_hours'] for i in options['list_peak_hours_periods_end_hours']]
            num_peak_hours = len(start_hours_list)
            list_hp_periods_list = [{'period_hp_'+str(i+1):[{'start':start_hours_list[i]},{'end':end_hours_list[i]}]} for i in range(num_peak_hours)]
            params['optim_conf']['list_hp_periods'] = list_hp_periods_list
        except:
            app.logger.debug("no list_peak_hours_periods_start_hours, defaulting to config file")
        params['optim_conf']['load_cost_hp'] = options.get('load_peak_hours_cost',params['optim_conf']['load_cost_hp'])
        params['optim_conf']['load_cost_hc'] = options.get('load_offpeak_hours_cost', params['optim_conf']['load_cost_hc'])
        params['optim_conf']['prod_price_forecast_method'] = options.get('production_price_forecast_method', params['optim_conf']['prod_price_forecast_method'])
        params['optim_conf']['prod_sell_price'] = options.get('photovoltaic_production_sell_price',params['optim_conf']['prod_sell_price'])
        params['optim_conf']['set_total_pv_sell'] = options.get('set_total_pv_sell',params['optim_conf']['set_total_pv_sell'])
        params['optim_conf']['lp_solver'] = options.get('lp_solver',params['optim_conf']['lp_solver'])
        params['optim_conf']['lp_solver_path'] = options.get('lp_solver_path',params['optim_conf']['lp_solver_path'])
        params['optim_conf']['set_nocharge_from_grid'] = options.get('set_nocharge_from_grid',params['optim_conf']['set_nocharge_from_grid'])
        params['optim_conf']['set_nodischarge_to_grid'] = options.get('set_nodischarge_to_grid',params['optim_conf']['set_nodischarge_to_grid'])
        params['optim_conf']['set_battery_dynamic'] = options.get('set_battery_dynamic',params['optim_conf']['set_battery_dynamic'])
        params['optim_conf']['battery_dynamic_max'] = options.get('battery_dynamic_max',params['optim_conf']['battery_dynamic_max'])
        params['optim_conf']['battery_dynamic_min'] = options.get('battery_dynamic_min',params['optim_conf']['battery_dynamic_min'])
        params['optim_conf']['weight_battery_discharge'] = options.get('weight_battery_discharge',params['optim_conf']['weight_battery_discharge'])
        params['optim_conf']['weight_battery_charge'] = options.get('weight_battery_charge',params['optim_conf']['weight_battery_charge'])
        try:
            params['optim_conf']['def_start_timestep'] = [i['start_timesteps_of_each_deferrable_load'] for i in options.get('list_start_timesteps_of_each_deferrable_load')]
        except:
            app.logger.debug("no list_start_timesteps_of_each_deferrable_load, defaulting to config file")
        try:
            params['optim_conf']['def_end_timestep'] = [i['end_timesteps_of_each_deferrable_load'] for i in options.get('list_end_timesteps_of_each_deferrable_load')]
        except:
            app.logger.debug("no list_end_timesteps_of_each_deferrable_load, defaulting to config file")
        # Updating variables in plant_con
        params['plant_conf']['P_grid_max'] = options.get('maximum_power_from_grid',params['plant_conf']['P_grid_max'])
        try:
            params['plant_conf']['module_model'] = [i['pv_module_model'] for i in options.get('list_pv_module_model')]
        except:
            app.logger.debug("no list_pv_module_model, defaulting to config file")
        try:
            params['plant_conf']['inverter_model'] = [i['pv_inverter_model'] for i in options.get('list_pv_inverter_model')]
        except:
            app.logger.debug("no list_pv_inverter_model, defaulting to config file")
        try:
            params['plant_conf']['surface_tilt'] = [i['surface_tilt'] for i in options.get('list_surface_tilt')]
        except:
            app.logger.debug("no list_surface_tilt, defaulting to config file")
        try:
            params['plant_conf']['surface_azimuth'] = [i['surface_azimuth'] for i in options.get('list_surface_azimuth')]
        except:
            app.logger.debug("no list_surface_azimuth, defaulting to config file")
        try:
            params['plant_conf']['modules_per_string'] = [i['modules_per_string'] for i in options.get('list_modules_per_string')]
        except:
            app.logger.debug("no list_modules_per_string, defaulting to config file")
        try:
            params['plant_conf']['strings_per_inverter'] = [i['strings_per_inverter'] for i in options.get('list_strings_per_inverter')]
        except:
            app.logger.debug("no list_strings_per_inverter, defaulting to config file")
        params['plant_conf']['Pd_max'] = options.get('battery_discharge_power_max',params['plant_conf']['Pd_max']) 
        params['plant_conf']['Pc_max'] = options.get('battery_charge_power_max',params['plant_conf']['Pc_max'])
        params['plant_conf']['eta_disch'] = options.get('battery_discharge_efficiency',params['plant_conf']['eta_disch'])
        params['plant_conf']['eta_ch'] = options.get('battery_charge_efficiency',params['plant_conf']['eta_ch'])
        params['plant_conf']['Enom'] = options.get('battery_nominal_energy_capacity',params['plant_conf']['Enom'])
        params['plant_conf']['SOCmin'] = options.get('battery_minimum_state_of_charge',params['plant_conf']['SOCmin']) 
        params['plant_conf']['SOCmax'] = options.get('battery_maximum_state_of_charge',params['plant_conf']['SOCmax']) 
        params['plant_conf']['SOCtarget'] = options.get('battery_target_state_of_charge',params['plant_conf']['SOCtarget'])
    else:
        params['params_secrets'] = params_secrets
    # The params dict
    params['passed_data'] = {'pv_power_forecast':None,'load_power_forecast':None,'load_cost_forecast':None,'prod_price_forecast':None,
                             'prediction_horizon':None,'soc_init':None,'soc_final':None,'def_total_hours':None,'def_start_timestep':None,'def_end_timestep':None,'alpha':None,'beta':None}
    return params

@app.route('/')
def index():
    app.logger.info("EMHASS server online, serving index.html...")
    # Load HTML template
    file_loader = PackageLoader('emhass', 'templates')
    env = Environment(loader=file_loader)
    template = env.get_template('index.html')
    # Load cache dict
    if (data_path / 'injection_dict.pkl').exists():
        with open(str(data_path / 'injection_dict.pkl'), "rb") as fid:
            injection_dict = pickle.load(fid)
    else:
        app.logger.warning("The data container dictionary is empty... Please launch an optimization task")
        injection_dict={}
    basename = request.headers.get("X-Ingress-Path", "")
    return make_response(template.render(injection_dict=injection_dict, basename=basename))

@app.route('/action/<action_name>', methods=['POST'])
def action_call(action_name):
    with open(str(data_path / 'params.pkl'), "rb") as fid:
        config_path, params = pickle.load(fid)
    runtimeparams = request.get_json(force=True)
    params = json.dumps(params)
    runtimeparams = json.dumps(runtimeparams)
    input_data_dict = set_input_data_dict(config_path, str(data_path), costfun, 
        params, runtimeparams, action_name, app.logger)
    if action_name == 'publish-data':
        app.logger.info(" >> Publishing data...")
        _ = publish_data(input_data_dict, app.logger)
        msg = f'EMHASS >> Action publish-data executed... \n'
        return make_response(msg, 201)
    elif action_name == 'perfect-optim':
        app.logger.info(" >> Performing perfect optimization...")
        opt_res = perfect_forecast_optim(input_data_dict, app.logger)
        injection_dict = get_injection_dict(opt_res)
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action perfect-optim executed... \n'
        return make_response(msg, 201)
    elif action_name == 'dayahead-optim':
        app.logger.info(" >> Performing dayahead optimization...")
        opt_res = dayahead_forecast_optim(input_data_dict, app.logger)
        injection_dict = get_injection_dict(opt_res)
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action dayahead-optim executed... \n'
        return make_response(msg, 201)
    elif action_name == 'naive-mpc-optim':
        app.logger.info(" >> Performing naive MPC optimization...")
        opt_res = naive_mpc_optim(input_data_dict, app.logger)
        injection_dict = get_injection_dict(opt_res)
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action naive-mpc-optim executed... \n'
        return make_response(msg, 201)
    elif action_name == 'forecast-model-fit':
        app.logger.info(" >> Performing a machine learning forecast model fit...")
        df_fit_pred, _, mlf = forecast_model_fit(input_data_dict, app.logger)
        injection_dict = get_injection_dict_forecast_model_fit(
            df_fit_pred, mlf)
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action forecast-model-fit executed... \n'
        return make_response(msg, 201)
    elif action_name == 'forecast-model-predict':
        app.logger.info(" >> Performing a machine learning forecast model predict...")
        df_pred = forecast_model_predict(input_data_dict, app.logger)
        table1 = df_pred.reset_index().to_html(classes='mystyle', index=False)
        injection_dict = {}
        injection_dict['title'] = '<h2>Custom machine learning forecast model predict</h2>'
        injection_dict['subsubtitle0'] = '<h4>Performed a prediction using a pre-trained model</h4>'
        injection_dict['table1'] = table1
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action forecast-model-predict executed... \n'
        return make_response(msg, 201)
    elif action_name == 'forecast-model-tune':
        app.logger.info(" >> Performing a machine learning forecast model tune...")
        df_pred_optim, mlf = forecast_model_tune(input_data_dict, app.logger)
        injection_dict = get_injection_dict_forecast_model_tune(
            df_pred_optim, mlf)
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action forecast-model-tune executed... \n'
        return make_response(msg, 201)
    else:
        app.logger.error("ERROR: passed action is not valid")
        msg = f'EMHASS >> ERROR: Passed action is not valid... \n'
        return make_response(msg, 400)

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, help='The URL to your Home Assistant instance, ex the external_url in your hass configuration')
    parser.add_argument('--key', type=str, help='Your access key. If using EMHASS in standalone this should be a Long-Lived Access Token')
    parser.add_argument('--addon', type=strtobool, default='False', help='Define if we are usinng EMHASS with the add-on or in standalone mode')
    args = parser.parse_args()
    
    # Define the paths
    DATA_PATH = os.getenv("DATA_PATH", default="/app/data/")
    data_path = Path(DATA_PATH)
    if args.addon==1:
        OPTIONS_PATH = os.getenv('OPTIONS_PATH', default=data_path / "options.json")
        options_json = Path(OPTIONS_PATH)
        CONFIG_PATH = os.getenv("CONFIG_PATH", default="/app/config_emhass.yaml")
        hass_url = args.url
        key = args.key
        # Read options info
        if options_json.exists():
            with options_json.open('r') as data:
                options = json.load(data)
        else:
            app.logger.error("options.json does not exists")
        
        DATA_PATH = "/share/" #"/data/"
    else:
        CONFIG_PATH = os.getenv("CONFIG_PATH", default="/app/config_emhass.yaml")
        options = None
        DATA_PATH = os.getenv("DATA_PATH", default="/app/data/")

    config_path = Path(CONFIG_PATH)
    
    # Read example config file
    if config_path.exists():
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        retrieve_hass_conf = config['retrieve_hass_conf']
        optim_conf = config['optim_conf']
        plant_conf = config['plant_conf']
    else:
        app.logger.error("config_emhass.json does not exists")
        app.logger.info("Failed config_path: "+str(config_path))

    params = {}
    params['retrieve_hass_conf'] = retrieve_hass_conf
    params['optim_conf'] = optim_conf
    params['plant_conf'] = plant_conf
    web_ui_url = '0.0.0.0'

    # Initialize this global dict
    if (data_path / 'injection_dict.pkl').exists():
        with open(str(data_path / 'injection_dict.pkl'), "rb") as fid:
            injection_dict = pickle.load(fid)
    else:
        injection_dict = None
    
    if args.addon==1:
        # The cost function
        costfun = options.get('costfun', 'profit')
        # Some data from options
        logging_level = options.get('logging_level','INFO')
        url_from_options = options.get('hass_url', 'empty')
        if url_from_options == 'empty' or url_from_options == '':
            url = hass_url+"/config"
        else:
            hass_url = url_from_options
            url = hass_url+"/api/config"
        token_from_options = options.get('long_lived_token', 'empty')
        if token_from_options == 'empty' or token_from_options == '':
            long_lived_token = key
        else:
            long_lived_token = token_from_options
        headers = {
            "Authorization": "Bearer " + long_lived_token,
            "content-type": "application/json"
        }
        response = get(url, headers=headers)  
        try:
            config_hass = response.json()
            
            params_secrets = {
                'hass_url': hass_url,
                'long_lived_token': long_lived_token,
                'time_zone': config_hass['time_zone'],
                'lat': config_hass['latitude'],
                'lon': config_hass['longitude'],
                'alt': config_hass['elevation']
            }
        except: #if addon testing (use secrets)
            with open(os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml'), 'r') as file:
                params_secrets = yaml.load(file, Loader=yaml.FullLoader)
                params_secrets['hass_url'] = hass_url
                params_secrets['long_lived_token'] = long_lived_token
    else:
        costfun = os.getenv('LOCAL_COSTFUN', default='profit')
        logging_level = os.getenv('LOGGING_LEVEL', default='INFO')
        with open(os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml'), 'r') as file:
            params_secrets = yaml.load(file, Loader=yaml.FullLoader)
        hass_url = params_secrets['hass_url']
        
    # Build params
    params = build_params(params, params_secrets, options, args.addon)
    with open(str(data_path / 'params.pkl'), "wb") as fid:
        pickle.dump((config_path, params), fid)

    # Define logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    if logging_level == "DEBUG":
        app.logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        app.logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        app.logger.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        app.logger.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)
    else:
        app.logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    app.logger.propagate = False
    app.logger.addHandler(ch)
    
    # Launch server
    port = int(os.environ.get('PORT', 5000))
    app.logger.info("Launching the emhass webserver at: http://"+web_ui_url+":"+str(port))
    app.logger.info("Home Assistant data fetch will be performed using url: "+hass_url)
    app.logger.info("The data path is: "+str(data_path))
    try:
        app.logger.info("Using core emhass version: "+version('emhass'))
    except PackageNotFoundError:
        app.logger.info("Using development emhass version")
    serve(app, host=web_ui_url, port=port, threads=8)
