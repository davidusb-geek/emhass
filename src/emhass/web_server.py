#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, make_response, render_template
from jinja2 import Environment, PackageLoader
from requests import get
from waitress import serve
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import os, json, argparse, pickle, yaml, logging,  re, threading
from distutils.util import strtobool

from emhass.command_line import set_input_data_dict
from emhass.command_line import perfect_forecast_optim, dayahead_forecast_optim, naive_mpc_optim
from emhass.command_line import forecast_model_fit, forecast_model_predict, forecast_model_tune, weather_forecast_cache
from emhass.command_line import regressor_model_fit, regressor_model_predict
from emhass.command_line import publish_data, continual_publish
from emhass.utils import get_injection_dict, get_injection_dict_forecast_model_fit, \
    get_injection_dict_forecast_model_tune,  build_config, build_secrets, build_params, \
    param_to_config

# Define the Flask instance
app = Flask(__name__)
emhass_conf = {}

# Check logfile for error, anything after string match if provided 
def checkFileLog(refString=None):
    if (refString is not None): 
       logArray = grabLog(refString) #grab reduced log array
    else: 
        if ((emhass_conf['data_path'] / 'actionLogs.txt')).exists():
            with open(str(emhass_conf['data_path'] / 'actionLogs.txt'), "r") as fp:
                    logArray = fp.readlines()
    for logString in logArray:
            if (logString.split(' ', 1)[0] == "ERROR"):
                return True     
    return False

# Find string in logs, append all lines after to return
def grabLog(refString): 
    isFound = []
    output = []
    if ((emhass_conf['data_path'] / 'actionLogs.txt')).exists():
            with open(str(emhass_conf['data_path'] / 'actionLogs.txt'), "r") as fp:
                    logArray = fp.readlines()
            for x in range(len(logArray)-1): #find all matches and log key in isFound
                if (re.search(refString,logArray[x])):
                   isFound.append(x)
            if len(isFound) != 0:
                for x in range(isFound[-1],len(logArray)): #use isFound to extract last related action logs  
                    output.append(logArray[x])
    return output

# Clear the log file
def clearFileLog(): 
    if ((emhass_conf['data_path'] / 'actionLogs.txt')).exists():
        with open(str(emhass_conf['data_path'] / 'actionLogs.txt'), "w") as fp:
            fp.truncate()    

# Initial index page render
@app.route('/')
@app.route('/index')
def index():
    app.logger.info("EMHASS server online, serving index.html...")
    # Load HTML template
    file_loader = PackageLoader('emhass', 'templates')
    env = Environment(loader=file_loader)
    template = env.get_template('index.html')
    # Load cache dict
    if (emhass_conf['data_path'] / 'injection_dict.pkl').exists():
        with open(str(emhass_conf['data_path'] / 'injection_dict.pkl'), "rb") as fid:
            injection_dict = pickle.load(fid)
    else:
        app.logger.warning("The data container dictionary is empty... Please launch an optimization task")
        injection_dict={}

    # replace {{basename}} in html template html with path root  
    # basename = request.headers.get("X-Ingress-Path", "")
    # return make_response(template.render(injection_dict=injection_dict, basename=basename))
    
    return make_response(template.render(injection_dict=injection_dict))

# Configuration page actions:
# Render configuration page 
@app.route('/configuration')
def configuration():
    app.logger.info("serving configuration.html...")
    # Load HTML template
    file_loader = PackageLoader('emhass', 'templates')
    env = Environment(loader=file_loader)
    template = env.get_template('configuration.html')
    return make_response(template.render(config=params))

# Get latest built config
@app.route('/get-config', methods=['GET'])
def parameter_get():
    # Build config
    config = build_config(emhass_conf,app.logger,emhass_conf["defaults_path"],emhass_conf["config_path"],emhass_conf["legacy_config_path"])
    params = build_params(emhass_conf,{},config,app.logger)
    return_config = param_to_config(params,app.logger)
    # Make sure we do not send any secret parameters
    secret_params = ["hass_url", "time_zone", "Latitude", "Longitude", "Altitude", "long_lived_token", "solcast_api_key", "solcast_rooftop_id", "solar_forecast_kwp"]
    for key in list(return_config.keys()):
        if key in secret_params:
            del return_config[key]
    # Send config
    return make_response(return_config,201)

# Get default Config
@app.route('/get-config/defaults', methods=['GET'])
def config_get():
    #return default parameters, if options.json exists, return with options overwriting defaults (legacy support)
    config = build_config(emhass_conf,app.logger,emhass_conf["defaults_path"],emhass_conf["options_path"])
    params = build_params(emhass_conf,{},config,app.logger)
    return_config = param_to_config(params,app.logger)
    # Make sure we do not send any secret parameters
    secret_params = ["hass_url", "time_zone", "Latitude", "Longitude", "Altitude", "long_lived_token", "solcast_api_key", "solcast_rooftop_id", "solar_forecast_kwp"]
    for key in list(return_config.keys()):
        if key in secret_params:
            del return_config[key]

    # Send params
    return make_response(return_config,201)

# Save config to file (config.json and param.pkl)
@app.route('/set-config', methods=['POST'])
def parameter_set():
    config = {}
    # Load defaults as a reference point to override
    if emhass_conf['defaults_path'] and Path(emhass_conf['defaults_path']).is_file():
        with emhass_conf['defaults_path'].open('r') as data:
            defaults = json.load(data)
    
    # Retrieve send config
    config = request.get_json(force=True)
    # Merge defaults with config 
    config = defaults | config
    # check if data is empty
    if len(config) == 0:
        return make_response("failed to retrieve config",400)
    # Save config to config.json
    if emhass_conf['config_path']:
        with emhass_conf['config_path'].open('w') as f:
            json.dump(config, f, indent=4)
    else: 
        return make_response("failed to retrieve config file",400)
    
    # Update current param save (build and save param with config)
    if os.path.exists(str(emhass_conf['data_path'])): 
        with open(str(emhass_conf['data_path'] / 'params.pkl'), "rb") as fid:
            emhass_conf['config_path'], params = pickle.load(fid)
    else: 
        return make_response("failed to retrieve parameter file",400)
    # Build params from config
    web_Params = build_params(emhass_conf,params_secrets,config,app.logger)
    # Update (overwrite) params with params from the web
    params.update(web_Params)
    # Save updated params 
    with open(str(emhass_conf['data_path'] / 'params.pkl'), "wb") as fid:
        pickle.dump((config_path, params), fid)   
    
    app.logger.info("Saved parameters from webserver")
    return make_response({},201)
#

#get template actions 
@app.route('/template/<action_name>', methods=['GET'])
def template_action(action_name):
    app.logger.info(" >> Sending rendered template table data")
    if action_name == 'table-template':
        file_loader = PackageLoader('emhass', 'templates')
        env = Environment(loader=file_loader)
        template = env.get_template('template.html')
        if (emhass_conf['data_path'] / 'injection_dict.pkl').exists():
            with open(str(emhass_conf['data_path'] / 'injection_dict.pkl'), "rb") as fid:
                injection_dict = pickle.load(fid)
        else:
            app.logger.warning("The data container dictionary is empty... Please launch an optimization task")
            injection_dict={}        
        return make_response(template.render(injection_dict=injection_dict))

#post actions 
@app.route('/action/<action_name>', methods=['POST'])
def action_call(action_name):
    # Setting up parameters
    with open(str(emhass_conf['data_path'] / 'params.pkl'), "rb") as fid:
        emhass_conf['config_path'], params = pickle.load(fid)
    runtimeparams = request.get_json(force=True)
    params = json.dumps(params)
    if runtimeparams is not None and runtimeparams != '{}':
        app.logger.info("Passed runtime parameters: " + str(runtimeparams))
    runtimeparams = json.dumps(runtimeparams)
    
    # Run action if weather_forecast_cache
    if action_name == 'weather-forecast-cache':
        ActionStr = " >> Performing weather forecast, try to caching result"
        app.logger.info(ActionStr)
        weather_forecast_cache(emhass_conf, params, runtimeparams, app.logger)
        msg = f'EMHASS >> Weather Forecast has run and results possibly cached... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)

    ActionStr = " >> Setting input data dict"
    app.logger.info(ActionStr)
    input_data_dict = set_input_data_dict(emhass_conf, costfun, 
        params, runtimeparams, action_name, app.logger)
    if not input_data_dict:
        return make_response(grabLog(ActionStr), 400)
    
    # If continual_publish is True, start thread with loop function
    if len(continual_publish_thread) == 0 and input_data_dict['retrieve_hass_conf'].get('continual_publish',False):
        # Start Thread
        continualLoop = threading.Thread(name='continual_publish',target=continual_publish,args=[input_data_dict,entity_path,app.logger])
        continualLoop.start()
        continual_publish_thread.append(continualLoop)      
        
    # run action based on POST request
    if action_name == 'publish-data':
        ActionStr = " >> Publishing data..."
        app.logger.info(ActionStr)
        _ = publish_data(input_data_dict, app.logger)
        msg = f'EMHASS >> Action publish-data executed... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)
    elif action_name == 'perfect-optim':
        ActionStr = " >> Performing perfect optimization..."
        app.logger.info(ActionStr)
        opt_res = perfect_forecast_optim(input_data_dict, app.logger)
        injection_dict = get_injection_dict(opt_res)
        with open(str(emhass_conf['data_path'] / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action perfect-optim executed... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)
    elif action_name == 'dayahead-optim':
        ActionStr = " >> Performing dayahead optimization..."
        app.logger.info(ActionStr)
        opt_res = dayahead_forecast_optim(input_data_dict, app.logger)
        injection_dict = get_injection_dict(opt_res)
        with open(str(emhass_conf['data_path'] / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action dayahead-optim executed... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)
    elif action_name == 'naive-mpc-optim':
        ActionStr = " >> Performing naive MPC optimization..."
        app.logger.info(ActionStr)
        opt_res = naive_mpc_optim(input_data_dict, app.logger)
        injection_dict = get_injection_dict(opt_res)
        with open(str(emhass_conf['data_path'] / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action naive-mpc-optim executed... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)
    elif action_name == 'forecast-model-fit':
        ActionStr = " >> Performing a machine learning forecast model fit..."
        app.logger.info(ActionStr)
        df_fit_pred, _, mlf = forecast_model_fit(input_data_dict, app.logger)
        injection_dict = get_injection_dict_forecast_model_fit(
            df_fit_pred, mlf)
        with open(str(emhass_conf['data_path'] / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action forecast-model-fit executed... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)
    elif action_name == 'forecast-model-predict':
        ActionStr = " >> Performing a machine learning forecast model predict..."
        app.logger.info(ActionStr)
        df_pred = forecast_model_predict(input_data_dict, app.logger)
        if df_pred is None:
            return make_response(grabLog(ActionStr), 400)
        table1 = df_pred.reset_index().to_html(classes='mystyle', index=False)
        injection_dict = {}
        injection_dict['title'] = '<h2>Custom machine learning forecast model predict</h2>'
        injection_dict['subsubtitle0'] = '<h4>Performed a prediction using a pre-trained model</h4>'
        injection_dict['table1'] = table1
        with open(str(emhass_conf['data_path'] / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action forecast-model-predict executed... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)
    elif action_name == 'forecast-model-tune':
        ActionStr = " >> Performing a machine learning forecast model tune..."
        app.logger.info(ActionStr)
        df_pred_optim, mlf = forecast_model_tune(input_data_dict, app.logger)    
        if df_pred_optim is None or  mlf is None:
            return make_response(grabLog(ActionStr), 400)
        injection_dict = get_injection_dict_forecast_model_tune(
            df_pred_optim, mlf)
        with open(str(emhass_conf['data_path'] / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action forecast-model-tune executed... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)
    elif action_name == 'regressor-model-fit':
        ActionStr = " >> Performing a machine learning regressor fit..."
        app.logger.info(ActionStr)
        regressor_model_fit(input_data_dict, app.logger)
        msg = f'EMHASS >> Action regressor-model-fit executed... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)
    elif action_name == 'regressor-model-predict':
        ActionStr = " >> Performing a machine learning regressor predict..."
        app.logger.info(ActionStr)
        regressor_model_predict(input_data_dict, app.logger)
        msg = f'EMHASS >> Action regressor-model-predict executed... \n'
        if not checkFileLog(ActionStr):
            return make_response(msg, 201)
        return make_response(grabLog(ActionStr), 400)
    else:
        app.logger.error("ERROR: passed action is not valid")
        msg = f'EMHASS >> ERROR: Passed action is not valid... \n'
        return make_response(msg, 400)

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, help='The URL to your Home Assistant instance, ex the external_url in your hass configuration')
    parser.add_argument('--key', type=str, help='Your access key. If using EMHASS in standalone this should be a Long-Lived Access Token')
    parser.add_argument('--no_response', type=strtobool, default='False', help='This is set if json response errors occur')
    args = parser.parse_args()

    # Pre-built parameters (raw from config.json, legacy config (config_emhass.yaml), options.json (Home Assistant Secrets) and secrets_emhass.yaml parameter files)
    config = {} 
    # Secrets
    params_secrets = {}
    # Built parameters (inc. config + secrets)
    params = None 
    
    # Find env's, not not set defaults 
    DATA_PATH = os.getenv("DATA_PATH", default="/app/data/")
    ROOT_PATH = os.getenv("ROOT_PATH", default=str(Path(__file__).parent))
    CONFIG_PATH = os.getenv('CONFIG_PATH', default="/share/config.json")
    OPTIONS_PATH = os.getenv('OPTIONS_PATH', default="/app/options.json") 
    DEFAULTS_PATH = os.getenv('DEFAULTS_PATH', default=ROOT_PATH +"/data/config_defaults.json")
    ASSOCIATIONS_PATH = os.getenv('ASSOCIATIONS_PATH', default=ROOT_PATH + "/data/associations.csv")
    LEGACY_CONFIG_PATH = os.getenv("LEGACY_CONFIG_PATH", default="/app/config_emhass.yaml")

    # Define the paths
    config_path = Path(CONFIG_PATH)
    options_path = Path(OPTIONS_PATH)
    defaults_path = Path(DEFAULTS_PATH)
    associations_path = Path(ASSOCIATIONS_PATH)
    legacy_config_path = Path(LEGACY_CONFIG_PATH)
    data_path = Path(DATA_PATH)
    root_path = Path(ROOT_PATH)
    emhass_conf['config_path'] = config_path
    emhass_conf['options_path'] = options_path
    emhass_conf['defaults_path'] = defaults_path
    emhass_conf['associations_path'] = associations_path
    emhass_conf['legacy_config_path'] = legacy_config_path 
    emhass_conf['data_path'] = data_path
    emhass_conf['root_path'] = root_path 

    config = {}
    # Combine parameters from all configuration files (if exists)
    config.update(build_config(emhass_conf,app.logger,defaults_path,config_path,legacy_config_path))

    web_ui_url = '0.0.0.0'
    costfun = os.getenv('LOCAL_COSTFUN', config.get('costfun', 'profit'))
    logging_level = os.getenv('LOGGING_LEVEL', config.get('logging_level','INFO'))
        
    ## secrets
    argument = {}
    if args.url:
        argument['url'] = args.url
    if args.key:
        argument['key'] = args.key

    # Combine secrets from ENV,ARG, Secrets file and/or Home Assistant (if exist)
    emhass_conf, secrets = build_secrets(emhass_conf,app.logger,argument,options_path,os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml'), bool(args.no_response))
    params_secrets.update(secrets)

    # Initialize this global dict
    if (emhass_conf['data_path'] / 'injection_dict.pkl').exists():
        with open(str(emhass_conf['data_path'] / 'injection_dict.pkl'), "rb") as fid:
            injection_dict = pickle.load(fid)
    else:
        injection_dict = None

    # Build params from config and param_secrets, save result for later reference
    params = build_params(emhass_conf, params_secrets, config, app.logger)
    if type(params) is bool:
        raise Exception("A error has occurred while building parameters")   
    if os.path.exists(str(emhass_conf['data_path'])): 
        with open(str(emhass_conf['data_path'] / 'params.pkl'), "wb") as fid:
            pickle.dump((config_path, params), fid)
    else: 
        raise Exception("missing: " + str(emhass_conf['data_path']))   


    # Define loggers
    ch = logging.StreamHandler() 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # Action file logger
    fileLogger = logging.FileHandler(str(emhass_conf['data_path'] / 'actionLogs.txt')) 
    formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    fileLogger.setFormatter(formatter) # add format to Handler
    if logging_level == "DEBUG":
        app.logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
        fileLogger.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        app.logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
        fileLogger.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        app.logger.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
        fileLogger.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        app.logger.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)
        fileLogger.setLevel(logging.ERROR)
    else:
        app.logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
        fileLogger.setLevel(logging.DEBUG)
    app.logger.propagate = False
    app.logger.addHandler(ch)
    app.logger.addHandler(fileLogger)   
    # Clear Action File logger file, ready for new instance
    clearFileLog()

    # If entity_path exists, remove any entity/metadata files 
    entity_path = emhass_conf['data_path'] / "entities"
    if os.path.exists(entity_path): 
        entity_pathContents = os.listdir(entity_path)
        if len(entity_pathContents) > 0:
            for entity in entity_pathContents:
                os.remove(entity_path / entity)
        
    # Initialise continual publish thread list
    continual_publish_thread = []
    
    # Launch server
    port = int(os.environ.get('PORT', 5000))
    app.logger.info("Launching the emhass webserver at: http://"+web_ui_url+":"+str(port))
    app.logger.info("Home Assistant data fetch will be performed using url: "+params_secrets['hass_url'])
    app.logger.info("The data path is: "+str(emhass_conf['data_path']))
    app.logger.info("The logging is: "+str(logging_level))
    try:
        app.logger.info("Using core emhass version: "+version('emhass'))
    except PackageNotFoundError:
        app.logger.info("Using development emhass version")
    serve(app, host=web_ui_url, port=port, threads=8)

