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
from collections import defaultdict
from emhass.command_line import set_input_data_dict
from emhass.command_line import perfect_forecast_optim, dayahead_forecast_optim, naive_mpc_optim
from emhass.command_line import publish_data


# Define the Flask instance
app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.logger.propagate = False

def get_injection_dict(df, plot_size = 1366):
    # Create plots
    cols_p = [i for i in df.columns.to_list() if 'P_' in i]
    fig_0 = px.line(df[cols_p], title='Systems powers schedule after optimization results', 
                    template='plotly_white', width=plot_size, height=0.5*plot_size, line_shape="hv")
    fig_0.update_layout(xaxis_title='Timestamp', yaxis_title='System powers (W)')
    if 'SOC_opt' in df.columns.to_list():
        fig_1 = px.line(df['SOC_opt'], title='Battery state of charge schedule after optimization results', 
                        template='plotly_white', width=plot_size, height=0.5*plot_size, line_shape="hv")
        fig_1.update_layout(xaxis_title='Timestamp', yaxis_title='Battery SOC (%)')
    cols_cost = [i for i in df.columns.to_list() if 'cost_' in i or 'unit_' in i]
    fig_2 = px.line(df[cols_cost], title='Systems costs obtained from optimization results', 
                      template='plotly_white', width=plot_size, height=0.5*plot_size, line_shape="hv")
    fig_2.update_layout(xaxis_title='Timestamp', yaxis_title='System costs (currency)')
    # Get full path to image
    image_path_0 = fig_0.to_html(full_html=False, default_width='75%')
    if 'SOC_opt' in df.columns.to_list():
        image_path_1 = fig_1.to_html(full_html=False, default_width='75%')
    image_path_2 = fig_2.to_html(full_html=False, default_width='75%')
    # The tables
    table1 = df.reset_index().to_html(classes='mystyle', index=False)
    cost_cols = [i for i in df.columns if 'cost_' in i]
    table2 = df[cost_cols].reset_index().sum(numeric_only=True).to_frame(name='Cost Totals').reset_index().to_html(classes='mystyle', index=False)
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

def build_params(params, options, addon):
    if addon:
        # Updating variables in retrieve_hass_conf
        params['retrieve_hass_conf']['freq'] = options['optimization_time_step']
        params['retrieve_hass_conf']['days_to_retrieve'] = options['historic_days_to_retrieve']
        params['retrieve_hass_conf']['var_PV'] = options['sensor_power_photovoltaics']
        params['retrieve_hass_conf']['var_load'] = options['sensor_power_load_no_var_loads']
        params['retrieve_hass_conf']['var_replace_zero'] = [options['sensor_power_photovoltaics']]
        params['retrieve_hass_conf']['var_interp'] = [options['sensor_power_photovoltaics'], options['sensor_power_load_no_var_loads']]
        params['retrieve_hass_conf']['method_ts_round'] = options['method_ts_round']
        # Updating variables in optim_conf
        params['optim_conf']['set_use_battery'] = options['set_use_battery']
        params['optim_conf']['num_def_loads'] = options['number_of_deferrable_loads']
        params['optim_conf']['P_deferrable_nom'] = [i['nominal_power_of_deferrable_loads'] for i in options['list_nominal_power_of_deferrable_loads']]
        params['optim_conf']['def_total_hours'] = [i['operating_hours_of_each_deferrable_load'] for i in options['list_operating_hours_of_each_deferrable_load']]
        params['optim_conf']['treat_def_as_semi_cont'] = [i['treat_deferrable_load_as_semi_cont'] for i in options['list_treat_deferrable_load_as_semi_cont']]
        params['optim_conf']['set_def_constant'] = [False for i in range(len(params['optim_conf']['P_deferrable_nom']))]
        start_hours_list = [i['peak_hours_periods_start_hours'] for i in options['list_peak_hours_periods_start_hours']]
        end_hours_list = [i['peak_hours_periods_end_hours'] for i in options['list_peak_hours_periods_end_hours']]
        num_peak_hours = len(start_hours_list)
        list_hp_periods_list = [{'period_hp_'+str(i+1):[{'start':start_hours_list[i]},{'end':end_hours_list[i]}]} for i in range(num_peak_hours)]
        params['optim_conf']['list_hp_periods'] = list_hp_periods_list
        params['optim_conf']['load_cost_hp'] = options['load_peak_hours_cost']
        params['optim_conf']['load_cost_hc'] = options['load_offpeak_hours_cost']
        params['optim_conf']['prod_sell_price'] = options['photovoltaic_production_sell_price']
        params['optim_conf']['set_total_pv_sell'] = options['set_total_pv_sell']
        params['optim_conf']['lp_solver'] = options['lp_solver']
        params['optim_conf']['lp_solver_path'] = options['lp_solver_path']
        params['optim_conf']['set_nocharge_from_grid'] = options['set_nocharge_from_grid']
        # Updating variables in plant_conf
        params['plant_conf']['P_grid_max'] = options['maximum_power_from_grid']
        params['plant_conf']['module_model'] = [i['pv_module_model'] for i in options['list_pv_module_model']]
        params['plant_conf']['inverter_model'] = [i['pv_inverter_model'] for i in options['list_pv_inverter_model']]
        params['plant_conf']['surface_tilt'] = [i['surface_tilt'] for i in options['list_surface_tilt']]
        params['plant_conf']['surface_azimuth'] = [i['surface_azimuth'] for i in options['list_surface_azimuth']]
        params['plant_conf']['modules_per_string'] = [i['modules_per_string'] for i in options['list_modules_per_string']]
        params['plant_conf']['strings_per_inverter'] = [i['strings_per_inverter'] for i in options['list_strings_per_inverter']]
        params['plant_conf']['Pd_max'] = options['battery_discharge_power_max']
        params['plant_conf']['Pc_max'] = options['battery_charge_power_max']
        params['plant_conf']['eta_disch'] = options['battery_discharge_efficiency']
        params['plant_conf']['eta_ch'] = options['battery_charge_efficiency']
        params['plant_conf']['Enom'] = options['battery_nominal_energy_capacity']
        params['plant_conf']['SOCmin'] = options['battery_minimum_state_of_charge']
        params['plant_conf']['SOCmax'] = options['battery_maximum_state_of_charge']
        params['plant_conf']['SOCtarget'] = options['battery_target_state_of_charge']
    # The params dict
    params['params_secrets'] = params_secrets
    params['passed_data'] = {'pv_power_forecast':None,'load_power_forecast':None,'load_cost_forecast':None,'prod_price_forecast':None,
                             'prediction_horizon':None,'soc_init':None,'soc_final':None,'def_total_hours':None,'alpha':None,'beta':None}
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
    return make_response(template.render(injection_dict=injection_dict))

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
    if args.addon:
        OPTIONS_PATH = os.getenv('OPTIONS_PATH', default=data_path / "options.json")
        options_json = Path(OPTIONS_PATH)
        CONFIG_PATH = os.getenv("CONFIG_PATH", default="/usr/src/config_emhass.yaml")
        hass_url = args.url
        key = args.key
        # Read options info
        if options_json.exists():
            with options_json.open('r') as data:
                options = json.load(data)
        else:
            app.logger.warning("options.json does not exists")
            options = defaultdict(lambda: 0)
            options['list_nominal_power_of_deferrable_loads'] = []
            options['list_operating_hours_of_each_deferrable_load'] = []
            options['list_treat_deferrable_load_as_semi_cont'] = []
            options['list_peak_hours_periods_start_hours'] = []
            options['list_peak_hours_periods_end_hours'] = []
            options['list_pv_module_model'] = []
            options['list_pv_inverter_model'] = []
            options['list_surface_tilt'] = []
            options['list_surface_azimuth'] = []
            options['list_modules_per_string'] = []
            options['list_strings_per_inverter'] = []

        DATA_PATH = "/share/" #"/data/"
    else:
        CONFIG_PATH = os.getenv("CONFIG_PATH", default="/app/config_emhass.yaml")
        options = {}

    config_path = Path(CONFIG_PATH)
    
    # Read example config file
    try:
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        retrieve_hass_conf = config['retrieve_hass_conf']
        optim_conf = config['optim_conf']
        plant_conf = config['plant_conf']
    except FileNotFoundError:
        app.logger.error("CONFIG_PATH: %r does not exist", str(config_path))
        sys.exit(1)

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
    
    if hass_url:
        # The cost function
        costfun = options.get('costfun', 'profit')
        # Some data from options
        url_from_options = options.get('hass_url', 'empty')
        if url_from_options == 'empty':
            url = hass_url+"/config"
        else:
            hass_url = url_from_options
            url = hass_url+"/api/config"
        token_from_options = options.get('long_lived_token', 'empty')
        if token_from_options == 'empty':
            long_lived_token = key
        else:
            long_lived_token = token_from_options
        headers = {
            "Authorization": "Bearer " + long_lived_token,
            "content-type": "application/json"
        }
        response = get(url, headers=headers)
        config_hass = response.json()
        params_secrets = {
            'hass_url': hass_url,
            'long_lived_token': long_lived_token,
            'time_zone': config_hass['time_zone'],
            'lat': config_hass['latitude'],
            'lon': config_hass['longitude'],
            'alt': config_hass['elevation']
        }
    else:
        costfun = os.getenv('LOCAL_COSTFUN', default='profit')
        with open(os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml'), 'r') as file:
            params_secrets = yaml.load(file, Loader=yaml.FullLoader)
        hass_url = params_secrets['hass_url']

        
    # Build params
    params = build_params(params, options, args.addon)
    with open(str(data_path / 'params.pkl'), "wb") as fid:
        pickle.dump((config_path, params), fid)

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
