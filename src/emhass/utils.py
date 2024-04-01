#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional
import numpy as np, pandas as pd
import yaml, pytz, logging, pathlib, json, copy
from datetime import datetime, timedelta, timezone
import plotly.express as px
pd.options.plotting.backend = "plotly"

from emhass.machine_learning_forecaster import MLForecaster


def get_root(file: str, num_parent: Optional[int] = 3) -> str:
    """
    Get the root absolute path of the working directory.
    
    :param file: The passed file path with __file__
    :return: The root path
    :param num_parent: The number of parents levels up to desired root folder
    :type num_parent: int, optional
    :rtype: str
    
    """
    if num_parent == 3:
        root = pathlib.Path(file).resolve().parent.parent.parent
    elif num_parent == 2:
        root = pathlib.Path(file).resolve().parent.parent
    elif num_parent == 1:
        root = pathlib.Path(file).resolve().parent
    else:
        raise ValueError("num_parent value not valid, must be between 1 and 3")
    return root

def get_logger(fun_name: str, config_path: str, save_to_file: Optional[bool] = True,
               logging_level: Optional[str] = "DEBUG") -> Tuple[logging.Logger, logging.StreamHandler]:
    """
    Create a simple logger object.
    
    :param fun_name: The Python function object name where the logger will be used
    :type fun_name: str
    :param config_path: The path to the yaml configuration file
    :type config_path: str
    :param save_to_file: Write log to a file, defaults to True
    :type save_to_file: bool, optional
    :return: The logger object and the handler
    :rtype: object
    
    """
	# create logger object
    logger = logging.getLogger(fun_name)
    logger.propagate = True
    logger.fileSetting = save_to_file
    if save_to_file:
        ch = logging.FileHandler(config_path + '/data/logger_emhass.log')
    else:
        ch = logging.StreamHandler()
    if logging_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        logger.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        logger.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, ch

def get_forecast_dates(freq: int, delta_forecast: int, 
                       timedelta_days: Optional[int] = 0) -> pd.core.indexes.datetimes.DatetimeIndex:
    """
    Get the date_range list of the needed future dates using the delta_forecast parameter.

    :param freq: Optimization time step.
    :type freq: int
    :param delta_forecast: Number of days to forecast in the future to be used for the optimization.
    :type delta_forecast: int
    :param timedelta_days: Number of truncated days needed for each optimization iteration, defaults to 0
    :type timedelta_days: Optional[int], optional
    :return: A list of future forecast dates.
    :rtype: pd.core.indexes.datetimes.DatetimeIndex
    
    """
    freq = pd.to_timedelta(freq, "minutes")
    start_forecast = pd.Timestamp(datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)
    end_forecast = (start_forecast + pd.Timedelta(days=delta_forecast)).replace(microsecond=0)
    forecast_dates = pd.date_range(start=start_forecast, 
        end=end_forecast+timedelta(days=timedelta_days)-freq, 
        freq=freq).round(freq, ambiguous='infer', nonexistent='shift_forward')
    return forecast_dates

def treat_runtimeparams(runtimeparams: str, params: str, retrieve_hass_conf: dict, optim_conf: dict, plant_conf: dict,
                        set_type: str, logger: logging.Logger) -> Tuple[str, dict]:
    """
    Treat the passed optimization runtime parameters. 
    
    :param runtimeparams: Json string containing the runtime parameters dict.
    :type runtimeparams: str
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param retrieve_hass_conf: Container for data retrieving parameters.
    :type retrieve_hass_conf: dict
    :param optim_conf: Container for optimization parameters.
    :type optim_conf: dict
    :param plant_conf: Container for technical plant parameters.
    :type plant_conf: dict
    :param set_type: The type of action to be performed.
    :type set_type: str
    :param logger: The logger object.
    :type logger: logging.Logger
    :return: Returning the params and optimization parameter container.
    :rtype: Tuple[str, dict]
    
    """
    if (params != None) and (params != 'null'):
        params = json.loads(params)
    else:
        params = {}
    # Some default data needed
    custom_deferrable_forecast_id = []
    for k in range(optim_conf['num_def_loads']):
        custom_deferrable_forecast_id.append({
            "entity_id": "sensor.p_deferrable{}".format(k), 
            "unit_of_measurement": "W", 
            "friendly_name": "Deferrable Load {}".format(k)
        })
    default_passed_dict = {'custom_pv_forecast_id': {"entity_id": "sensor.p_pv_forecast", "unit_of_measurement": "W", "friendly_name": "PV Power Forecast"},
                           'custom_load_forecast_id': {"entity_id": "sensor.p_load_forecast", "unit_of_measurement": "W", "friendly_name": "Load Power Forecast"},
                           'custom_batt_forecast_id': {"entity_id": "sensor.p_batt_forecast", "unit_of_measurement": "W", "friendly_name": "Battery Power Forecast"},
                           'custom_batt_soc_forecast_id': {"entity_id": "sensor.soc_batt_forecast", "unit_of_measurement": "%", "friendly_name": "Battery SOC Forecast"},
                           'custom_grid_forecast_id': {"entity_id": "sensor.p_grid_forecast", "unit_of_measurement": "W", "friendly_name": "Grid Power Forecast"},
                           'custom_cost_fun_id': {"entity_id": "sensor.total_cost_fun_value", "unit_of_measurement": "", "friendly_name": "Total cost function value"},
                           'custom_optim_status_id': {"entity_id": "sensor.optim_status", "unit_of_measurement": "", "friendly_name": "EMHASS optimization status"},
                           'custom_unit_load_cost_id': {"entity_id": "sensor.unit_load_cost", "unit_of_measurement": "€/kWh", "friendly_name": "Unit Load Cost"},
                           'custom_unit_prod_price_id': {"entity_id": "sensor.unit_prod_price", "unit_of_measurement": "€/kWh", "friendly_name": "Unit Prod Price"},
                           'custom_deferrable_forecast_id': custom_deferrable_forecast_id,
                           'publish_prefix': ""}
    if 'passed_data' in params.keys():
        for key, value in default_passed_dict.items():
            params['passed_data'][key] = value
    else:
        params['passed_data'] = default_passed_dict
    if runtimeparams is not None:
        runtimeparams = json.loads(runtimeparams)
        freq = int(retrieve_hass_conf['freq'].seconds/60.0)
        delta_forecast = int(optim_conf['delta_forecast'].days)
        forecast_dates = get_forecast_dates(freq, delta_forecast)
        # Treating special data passed for MPC control case
        if set_type == 'naive-mpc-optim':
            if 'prediction_horizon' not in runtimeparams.keys():
                prediction_horizon = 10 # 10 time steps by default
            else:
                prediction_horizon = runtimeparams['prediction_horizon']
            params['passed_data']['prediction_horizon'] = prediction_horizon
            if 'soc_init' not in runtimeparams.keys():
                soc_init = plant_conf['SOCtarget']
            else:
                soc_init = runtimeparams['soc_init']
            params['passed_data']['soc_init'] = soc_init
            if 'soc_final' not in runtimeparams.keys():
                soc_final = plant_conf['SOCtarget']
            else:
                soc_final = runtimeparams['soc_final']
            params['passed_data']['soc_final'] = soc_final
            if 'def_total_hours' not in runtimeparams.keys():
                def_total_hours = optim_conf['def_total_hours']
            else:
                def_total_hours = runtimeparams['def_total_hours']
            params['passed_data']['def_total_hours'] = def_total_hours
            if 'def_start_timestep' not in runtimeparams.keys():
                def_start_timestep = optim_conf['def_start_timestep']
            else:
                def_start_timestep = runtimeparams['def_start_timestep']
            params['passed_data']['def_start_timestep'] = def_start_timestep
            if 'def_end_timestep' not in runtimeparams.keys():
                def_end_timestep = optim_conf['def_end_timestep']
            else:
                def_end_timestep = runtimeparams['def_end_timestep']
            params['passed_data']['def_end_timestep'] = def_end_timestep
            if 'alpha' not in runtimeparams.keys():
                alpha = 0.5
            else:
                alpha = runtimeparams['alpha']
            params['passed_data']['alpha'] = alpha
            if 'beta' not in runtimeparams.keys():
                beta = 0.5
            else:
                beta = runtimeparams['beta']
            params['passed_data']['beta'] = beta
            forecast_dates = copy.deepcopy(forecast_dates)[0:prediction_horizon]
        else:
            params['passed_data']['prediction_horizon'] = None
            params['passed_data']['soc_init'] = None
            params['passed_data']['soc_final'] = None
            params['passed_data']['def_total_hours'] = None
            params['passed_data']['def_start_timestep'] = None
            params['passed_data']['def_end_timestep'] = None
            params['passed_data']['alpha'] = None
            params['passed_data']['beta'] = None
        # Treat passed forecast data lists
        list_forecast_key = ['pv_power_forecast', 'load_power_forecast', 'load_cost_forecast', 'prod_price_forecast']
        forecast_methods = ['weather_forecast_method', 'load_forecast_method', 'load_cost_forecast_method', 'prod_price_forecast_method']
        for method, forecast_key in enumerate(list_forecast_key):
            if forecast_key in runtimeparams.keys():
                if type(runtimeparams[forecast_key]) == list and len(runtimeparams[forecast_key]) >= len(forecast_dates):
                    params['passed_data'][forecast_key] = runtimeparams[forecast_key]
                    optim_conf[forecast_methods[method]] = 'list'
                else:
                    logger.error(f"ERROR: The passed data is either not a list or the length is not correct, length should be {str(len(forecast_dates))}")
                    logger.error(f"Passed type is {str(type(runtimeparams[forecast_key]))} and length is {str(len(runtimeparams[forecast_key]))}")
                list_non_digits = [x for x in runtimeparams[forecast_key] if not (isinstance(x, int) or isinstance(x, float))]
                if len(list_non_digits) > 0:
                    logger.warning(f"There are non numeric values on the passed data for {forecast_key}, check for missing values (nans, null, etc)")
                    for x in list_non_digits:
                        logger.warning(f"This value in {forecast_key} was detected as non digits: {str(x)}")
            else:
                params['passed_data'][forecast_key] = None
        # Treat passed data for forecast model fit/predict/tune at runtime
        if 'days_to_retrieve' not in runtimeparams.keys():
            days_to_retrieve = 9
        else:
            days_to_retrieve = runtimeparams['days_to_retrieve']
        params['passed_data']['days_to_retrieve'] = days_to_retrieve
        if 'model_type' not in runtimeparams.keys():
            model_type = "load_forecast"
        else:
            model_type = runtimeparams['model_type']
        params['passed_data']['model_type'] = model_type
        if 'var_model' not in runtimeparams.keys():
            var_model = "sensor.power_load_no_var_loads"
        else:
            var_model = runtimeparams['var_model']
        params['passed_data']['var_model'] = var_model
        if 'sklearn_model' not in runtimeparams.keys():
            sklearn_model = "KNeighborsRegressor"
        else:
            sklearn_model = runtimeparams['sklearn_model']
        params['passed_data']['sklearn_model'] = sklearn_model
        if 'num_lags' not in runtimeparams.keys():
            num_lags = 48
        else:
            num_lags = runtimeparams['num_lags']
        params['passed_data']['num_lags'] = num_lags
        if 'split_date_delta' not in runtimeparams.keys():
            split_date_delta = '48h'
        else:
            split_date_delta = runtimeparams['split_date_delta']
        params['passed_data']['split_date_delta'] = split_date_delta
        if 'perform_backtest' not in runtimeparams.keys():
            perform_backtest = False
        else:
            perform_backtest = eval(str(runtimeparams['perform_backtest']).capitalize())
        params['passed_data']['perform_backtest'] = perform_backtest
        if 'model_predict_publish' not in runtimeparams.keys():
            model_predict_publish = False
        else:
            model_predict_publish = eval(str(runtimeparams['model_predict_publish']).capitalize())
        params['passed_data']['model_predict_publish'] = model_predict_publish
        if 'model_predict_entity_id' not in runtimeparams.keys():
            model_predict_entity_id = "sensor.p_load_forecast_custom_model"
        else:
            model_predict_entity_id = runtimeparams['model_predict_entity_id']
        params['passed_data']['model_predict_entity_id'] = model_predict_entity_id
        if 'model_predict_unit_of_measurement' not in runtimeparams.keys():
            model_predict_unit_of_measurement = "W"
        else:
            model_predict_unit_of_measurement = runtimeparams['model_predict_unit_of_measurement']
        params['passed_data']['model_predict_unit_of_measurement'] = model_predict_unit_of_measurement
        if 'model_predict_friendly_name' not in runtimeparams.keys():
            model_predict_friendly_name = "Load Power Forecast custom ML model"
        else:
            model_predict_friendly_name = runtimeparams['model_predict_friendly_name']
        params['passed_data']['model_predict_friendly_name'] = model_predict_friendly_name
        # Treat optimization configuration parameters passed at runtime 
        if 'num_def_loads' in runtimeparams.keys():
            optim_conf['num_def_loads'] = runtimeparams['num_def_loads']
        if 'P_deferrable_nom' in runtimeparams.keys():
            optim_conf['P_deferrable_nom'] = runtimeparams['P_deferrable_nom']
        if 'def_total_hours' in runtimeparams.keys():
            optim_conf['def_total_hours'] = runtimeparams['def_total_hours']
        if 'def_start_timestep' in runtimeparams.keys():
            optim_conf['def_start_timestep'] = runtimeparams['def_start_timestep']
        if 'def_end_timestep' in runtimeparams.keys():
            optim_conf['def_end_timestep'] = runtimeparams['def_end_timestep']
        if 'treat_def_as_semi_cont' in runtimeparams.keys():
            optim_conf['treat_def_as_semi_cont'] = [eval(str(k).capitalize()) for k in runtimeparams['treat_def_as_semi_cont']]
        if 'set_def_constant' in runtimeparams.keys():
            optim_conf['set_def_constant'] = [eval(str(k).capitalize()) for k in runtimeparams['set_def_constant']]
        if 'solcast_api_key' in runtimeparams.keys():
            retrieve_hass_conf['solcast_api_key'] = runtimeparams['solcast_api_key']
            optim_conf['weather_forecast_method'] = 'solcast'
        if 'solcast_rooftop_id' in runtimeparams.keys():
            retrieve_hass_conf['solcast_rooftop_id'] = runtimeparams['solcast_rooftop_id']
            optim_conf['weather_forecast_method'] = 'solcast'
        if 'solar_forecast_kwp' in runtimeparams.keys():
            retrieve_hass_conf['solar_forecast_kwp'] = runtimeparams['solar_forecast_kwp']
            optim_conf['weather_forecast_method'] = 'solar.forecast'
        if 'weight_battery_discharge' in runtimeparams.keys():
            optim_conf['weight_battery_discharge'] = runtimeparams['weight_battery_discharge']
        if 'weight_battery_charge' in runtimeparams.keys():
            optim_conf['weight_battery_charge'] = runtimeparams['weight_battery_charge']
        # Treat plant configuration parameters passed at runtime
        if 'SOCtarget' in runtimeparams.keys():
            plant_conf['SOCtarget'] = runtimeparams['SOCtarget']
        # Treat custom entities id's and friendly names for variables
        if 'custom_pv_forecast_id' in runtimeparams.keys():
            params['passed_data']['custom_pv_forecast_id'] = runtimeparams['custom_pv_forecast_id']
        if 'custom_load_forecast_id' in runtimeparams.keys():
            params['passed_data']['custom_load_forecast_id'] = runtimeparams['custom_load_forecast_id']
        if 'custom_batt_forecast_id' in runtimeparams.keys():
            params['passed_data']['custom_batt_forecast_id'] = runtimeparams['custom_batt_forecast_id']
        if 'custom_batt_soc_forecast_id' in runtimeparams.keys():
            params['passed_data']['custom_batt_soc_forecast_id'] = runtimeparams['custom_batt_soc_forecast_id']
        if 'custom_grid_forecast_id' in runtimeparams.keys():
            params['passed_data']['custom_grid_forecast_id'] = runtimeparams['custom_grid_forecast_id']
        if 'custom_cost_fun_id' in runtimeparams.keys():
            params['passed_data']['custom_cost_fun_id'] = runtimeparams['custom_cost_fun_id']
        if 'custom_optim_status_id' in runtimeparams.keys():
            params['passed_data']['custom_optim_status_id'] = runtimeparams['custom_optim_status_id']
        if 'custom_unit_load_cost_id' in runtimeparams.keys():
            params['passed_data']['custom_unit_load_cost_id'] = runtimeparams['custom_unit_load_cost_id']
        if 'custom_unit_prod_price_id' in runtimeparams.keys():
            params['passed_data']['custom_unit_prod_price_id'] = runtimeparams['custom_unit_prod_price_id']
        if 'custom_deferrable_forecast_id' in runtimeparams.keys():
            params['passed_data']['custom_deferrable_forecast_id'] = runtimeparams['custom_deferrable_forecast_id']
        # A condition to put a prefix on all published data
        if 'publish_prefix' not in runtimeparams.keys():
            publish_prefix = ""
        else:
            publish_prefix = runtimeparams['publish_prefix']
        params['passed_data']['publish_prefix'] = publish_prefix
    # Serialize the final params
    params = json.dumps(params)
    return params, retrieve_hass_conf, optim_conf, plant_conf

def get_yaml_parse(config_path: str, use_secrets: Optional[bool] = True,
                   params: Optional[str] = None) -> Tuple[dict, dict, dict]:
    """
    Perform parsing of the config.yaml file.
    
    :param config_path: The path to the yaml configuration file
    :type config_path: str
    :param use_secrets: Indicate if we should use a secrets file or not.
        Set to False for unit tests.
    :type use_secrets: bool, optional
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :return: A tuple with the dictionaries containing the parsed data
    :rtype: tuple(dict)

    """
    base = config_path.parent
    if params is None:
        with open(config_path, 'r') as file:
            input_conf = yaml.load(file, Loader=yaml.FullLoader)
    else:
        input_conf = json.loads(params)
    if use_secrets:
        if params is None:
            with open(base / 'secrets_emhass.yaml', 'r') as file:
                input_secrets = yaml.load(file, Loader=yaml.FullLoader)
        else:
            input_secrets = input_conf.pop('params_secrets', None)
   
    if (type(input_conf['retrieve_hass_conf']) == list): #if using old config version 
        retrieve_hass_conf = dict({key:d[key] for d in input_conf['retrieve_hass_conf'] for key in d})
    else:
        retrieve_hass_conf = input_conf.get('retrieve_hass_conf', {})
        
    if use_secrets:
        retrieve_hass_conf.update(input_secrets)
    else:
        retrieve_hass_conf['hass_url'] = 'http://supervisor/core/api'
        retrieve_hass_conf['long_lived_token'] = '${SUPERVISOR_TOKEN}'
        retrieve_hass_conf['time_zone'] = 'Europe/Paris'
        retrieve_hass_conf['lat'] = 45.83
        retrieve_hass_conf['lon'] = 6.86
        retrieve_hass_conf['alt'] = 4807.8
    retrieve_hass_conf['freq'] = pd.to_timedelta(retrieve_hass_conf['freq'], "minutes")
    retrieve_hass_conf['time_zone'] = pytz.timezone(retrieve_hass_conf['time_zone'])
    
    if (type(input_conf['optim_conf']) == list):
        optim_conf = dict({key:d[key] for d in input_conf['optim_conf'] for key in d})
    else:
        optim_conf = input_conf.get('optim_conf', {})

    optim_conf['list_hp_periods'] = dict((key,d[key]) for d in optim_conf['list_hp_periods'] for key in d)
    optim_conf['delta_forecast'] = pd.Timedelta(days=optim_conf['delta_forecast'])
    
    if (type(input_conf['plant_conf']) == list):
        plant_conf = dict({key:d[key] for d in input_conf['plant_conf'] for key in d})
    else:
        plant_conf = input_conf.get('plant_conf', {})
    
    return retrieve_hass_conf, optim_conf, plant_conf

def get_injection_dict(df: pd.DataFrame, plot_size: Optional[int] = 1366) -> dict:
    """
    Build a dictionary with graphs and tables for the webui.

    :param df: The optimization result DataFrame
    :type df: pd.DataFrame
    :param plot_size: Size of the plot figure in pixels, defaults to 1366
    :type plot_size: Optional[int], optional
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    
    """
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
    injection_dict['subsubtitle2'] = '<h4>Summary table for latest optimization results</h4>'
    injection_dict['table2'] = table2
    return injection_dict

def get_injection_dict_forecast_model_fit(df_fit_pred: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF fit case.

    :param df_fit_pred: The fit result DataFrame
    :type df_fit_pred: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
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

def get_injection_dict_forecast_model_tune(df_pred_optim: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF tune case.

    :param df_pred_optim: The tune result DataFrame
    :type df_pred_optim: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
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

def build_params(params: dict, params_secrets: dict, options: dict, addon: int, logger: logging.Logger) -> dict:
    """
    Build the main params dictionary from the loaded options.json when using the add-on.

    :param params: The main params dictionary
    :type params: dict
    :param params_secrets: The dictionary containing the secret protected variables
    :type params_secrets: dict
    :param options: The load dictionary from options.json
    :type options: dict
    :param addon: A "bool" to select if we are using the add-on
    :type addon: int
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The builded dictionary
    :rtype: dict
    """
    if addon == 1:
        # Updating variables in retrieve_hass_conf
        params['retrieve_hass_conf']['freq'] = options.get('optimization_time_step',params['retrieve_hass_conf']['freq'])
        params['retrieve_hass_conf']['days_to_retrieve'] = options.get('historic_days_to_retrieve',params['retrieve_hass_conf']['days_to_retrieve'])
        params['retrieve_hass_conf']['var_PV'] = options.get('sensor_power_photovoltaics',params['retrieve_hass_conf']['var_PV'])
        params['retrieve_hass_conf']['var_load'] = options.get('sensor_power_load_no_var_loads',params['retrieve_hass_conf']['var_load'])
        params['retrieve_hass_conf']['load_negative'] = options.get('load_negative',params['retrieve_hass_conf']['load_negative'])
        params['retrieve_hass_conf']['set_zero_min'] = options.get('set_zero_min',params['retrieve_hass_conf']['set_zero_min'])
        params['retrieve_hass_conf']['var_replace_zero'] = [options.get('sensor_power_photovoltaics',params['retrieve_hass_conf']['var_replace_zero'])]
        params['retrieve_hass_conf']['var_interp'] = [options.get('sensor_power_photovoltaics',params['retrieve_hass_conf']['var_PV']), options.get('sensor_power_load_no_var_loads',params['retrieve_hass_conf']['var_load'])]
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
        if options.get('list_nominal_power_of_deferrable_loads',None) != None: 
            params['optim_conf']['P_deferrable_nom'] = [i['nominal_power_of_deferrable_loads'] for i in options.get('list_nominal_power_of_deferrable_loads')]
        if options.get('list_operating_hours_of_each_deferrable_load',None) != None: 
            params['optim_conf']['def_total_hours'] = [i['operating_hours_of_each_deferrable_load'] for i in options.get('list_operating_hours_of_each_deferrable_load')]
        if options.get('list_treat_deferrable_load_as_semi_cont',None) != None: 
            params['optim_conf']['treat_def_as_semi_cont'] = [i['treat_deferrable_load_as_semi_cont'] for i in options.get('list_treat_deferrable_load_as_semi_cont')]
        params['optim_conf']['weather_forecast_method'] = options.get('weather_forecast_method',params['optim_conf']['weather_forecast_method'])
        # Update optional param secrets
        if params['optim_conf']['weather_forecast_method'] == "solcast":
            params['params_secrets']['solcast_api_key'] = options.get('optional_solcast_api_key',params_secrets.get('solcast_api_key',"123456"))
            params['params_secrets']['solcast_rooftop_id'] = options.get('optional_solcast_rooftop_id',params_secrets.get('solcast_rooftop_id',"123456"))
        elif params['optim_conf']['weather_forecast_method'] == "solar.forecast":    
            params['params_secrets']['solar_forecast_kwp'] = options.get('optional_solar_forecast_kwp',params_secrets.get('solar_forecast_kwp',5))
        params['optim_conf']['load_forecast_method'] = options.get('load_forecast_method',params['optim_conf']['load_forecast_method'])
        params['optim_conf']['delta_forecast'] = options.get('delta_forecast_daily',params['optim_conf']['delta_forecast'])
        params['optim_conf']['load_cost_forecast_method'] = options.get('load_cost_forecast_method',params['optim_conf']['load_cost_forecast_method'])
        if options.get('list_set_deferrable_load_single_constant',None) != None: 
            params['optim_conf']['set_def_constant'] = [i['set_deferrable_load_single_constant'] for i in options.get('list_set_deferrable_load_single_constant')]
        if options.get('list_peak_hours_periods_start_hours',None) != None and options.get('list_peak_hours_periods_end_hours',None) != None:
            start_hours_list = [i['peak_hours_periods_start_hours'] for i in options['list_peak_hours_periods_start_hours']]
            end_hours_list = [i['peak_hours_periods_end_hours'] for i in options['list_peak_hours_periods_end_hours']]
            num_peak_hours = len(start_hours_list)
            list_hp_periods_list = [{'period_hp_'+str(i+1):[{'start':start_hours_list[i]},{'end':end_hours_list[i]}]} for i in range(num_peak_hours)]
            params['optim_conf']['list_hp_periods'] = list_hp_periods_list
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
        if options.get('list_start_timesteps_of_each_deferrable_load',None) != None: 
            params['optim_conf']['def_start_timestep'] = [i['start_timesteps_of_each_deferrable_load'] for i in options.get('list_start_timesteps_of_each_deferrable_load')]
        if options.get('list_end_timesteps_of_each_deferrable_load',None) != None: 
            params['optim_conf']['def_end_timestep'] = [i['end_timesteps_of_each_deferrable_load'] for i in options.get('list_end_timesteps_of_each_deferrable_load')]
        # Updating variables in plant_conf
        params['plant_conf']['P_from_grid_max'] = options.get('maximum_power_from_grid',params['plant_conf']['P_from_grid_max'])
        params['plant_conf']['P_to_grid_max'] = options.get('maximum_power_to_grid',params['plant_conf']['P_to_grid_max'])
        if options.get('list_pv_module_model',None) != None:         
            params['plant_conf']['module_model'] = [i['pv_module_model'] for i in options.get('list_pv_module_model')]
        if options.get('list_pv_inverter_model',None) != None:        
            params['plant_conf']['inverter_model'] = [i['pv_inverter_model'] for i in options.get('list_pv_inverter_model')]
        if options.get('list_surface_tilt',None) != None:        
            params['plant_conf']['surface_tilt'] = [i['surface_tilt'] for i in options.get('list_surface_tilt')]
        if options.get('list_surface_azimuth',None) != None:         
            params['plant_conf']['surface_azimuth'] = [i['surface_azimuth'] for i in options.get('list_surface_azimuth')]
        if options.get('list_modules_per_string',None) != None:         
            params['plant_conf']['modules_per_string'] = [i['modules_per_string'] for i in options.get('list_modules_per_string')]
        if options.get('list_strings_per_inverter',None) != None: 
            params['plant_conf']['strings_per_inverter'] = [i['strings_per_inverter'] for i in options.get('list_strings_per_inverter')]
        params['plant_conf']['Pd_max'] = options.get('battery_discharge_power_max',params['plant_conf']['Pd_max']) 
        params['plant_conf']['Pc_max'] = options.get('battery_charge_power_max',params['plant_conf']['Pc_max'])
        params['plant_conf']['eta_disch'] = options.get('battery_discharge_efficiency',params['plant_conf']['eta_disch'])
        params['plant_conf']['eta_ch'] = options.get('battery_charge_efficiency',params['plant_conf']['eta_ch'])
        params['plant_conf']['Enom'] = options.get('battery_nominal_energy_capacity',params['plant_conf']['Enom'])
        params['plant_conf']['SOCmin'] = options.get('battery_minimum_state_of_charge',params['plant_conf']['SOCmin']) 
        params['plant_conf']['SOCmax'] = options.get('battery_maximum_state_of_charge',params['plant_conf']['SOCmax']) 
        params['plant_conf']['SOCtarget'] = options.get('battery_target_state_of_charge',params['plant_conf']['SOCtarget'])
        # Check parameter lists have the same amounts as deferrable loads
        # If not, set defaults it fill in gaps
        if params['optim_conf']['num_def_loads'] is not len(params['optim_conf']['def_start_timestep']):
            logger.warning("def_start_timestep / list_start_timesteps_of_each_deferrable_load does not match number in num_def_loads, adding default values to parameter")
            for x in range(len(params['optim_conf']['def_start_timestep']), params['optim_conf']['num_def_loads']):
                params['optim_conf']['def_start_timestep'].append(0)
        if params['optim_conf']['num_def_loads'] is not len(params['optim_conf']['def_end_timestep']):
            logger.warning("def_end_timestep / list_end_timesteps_of_each_deferrable_load does not match number in num_def_loads, adding default values to parameter")
            for x in range(len(params['optim_conf']['def_end_timestep']), params['optim_conf']['num_def_loads']):
                params['optim_conf']['def_end_timestep'].append(0)
        if params['optim_conf']['num_def_loads'] is not len(params['optim_conf']['set_def_constant']):
            logger.warning("set_def_constant / list_set_deferrable_load_single_constant does not match number in num_def_loads, adding default values to parameter")
            for x in range(len(params['optim_conf']['set_def_constant']), params['optim_conf']['num_def_loads']):
                params['optim_conf']['set_def_constant'].append(False)
        if params['optim_conf']['num_def_loads'] is not len(params['optim_conf']['treat_def_as_semi_cont']):
            logger.warning("treat_def_as_semi_cont / list_treat_deferrable_load_as_semi_cont does not match number in num_def_loads, adding default values to parameter")
            for x in range(len(params['optim_conf']['treat_def_as_semi_cont']), params['optim_conf']['num_def_loads']):
                params['optim_conf']['treat_def_as_semi_cont'].append(True)        
        if params['optim_conf']['num_def_loads'] is not len(params['optim_conf']['def_total_hours']):
            logger.warning("def_total_hours / list_operating_hours_of_each_deferrable_load does not match number in num_def_loads, adding default values to parameter")
            for x in range(len(params['optim_conf']['def_total_hours']), params['optim_conf']['num_def_loads']):
                params['optim_conf']['def_total_hours'].append(0)                   
        if params['optim_conf']['num_def_loads'] is not len(params['optim_conf']['P_deferrable_nom']):
            logger.warning("P_deferrable_nom / list_nominal_power_of_deferrable_loads does not match number in num_def_loads, adding default values to parameter")
            for x in range(len(params['optim_conf']['P_deferrable_nom']), params['optim_conf']['num_def_loads']):
                params['optim_conf']['P_deferrable_nom'].append(0)   
        # days_to_retrieve should be no less then 2
        if params['retrieve_hass_conf']['days_to_retrieve'] < 2:
            params['retrieve_hass_conf']['days_to_retrieve'] = 2
            logger.warning("days_to_retrieve should not be lower then 2, setting days_to_retrieve to 2. Make sure your sensors also have at least 2 days of history")
    else:
        params['params_secrets'] = params_secrets
    # The params dict
    params['passed_data'] = {'pv_power_forecast':None,'load_power_forecast':None,'load_cost_forecast':None,'prod_price_forecast':None,
                             'prediction_horizon':None,'soc_init':None,'soc_final':None,'def_total_hours':None,'def_start_timestep':None,'def_end_timestep':None,'alpha':None,'beta':None}
    return params

def get_days_list(days_to_retrieve: int) -> pd.date_range:
    """
    Get list of past days from today to days_to_retrieve.
    
    :param days_to_retrieve: Total number of days to retrieve from the past
    :type days_to_retrieve: int
    :return: The list of days
    :rtype: pd.date_range

    """
    today = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    d = (today - timedelta(days=days_to_retrieve)).isoformat()
    days_list = pd.date_range(start=d, end=today.isoformat(), freq='D')
    
    return days_list

def set_df_index_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the freq of a DataFrame DateTimeIndex.
    
    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Input DataFrame with freq defined
    :rtype: pd.DataFrame
    
    """
    idx_diff = np.diff(df.index)
    sampling = pd.to_timedelta(np.median(idx_diff))
    df = df[~df.index.duplicated()]
    df = df.asfreq(sampling)
    return df
