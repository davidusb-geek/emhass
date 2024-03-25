#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, make_response, render_template
from jinja2 import Environment, PackageLoader
from requests import get
from waitress import serve
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import os, json, argparse, pickle, yaml, logging,  re
from distutils.util import strtobool

from emhass.command_line import set_input_data_dict
from emhass.command_line import perfect_forecast_optim, dayahead_forecast_optim, naive_mpc_optim
from emhass.command_line import forecast_model_fit, forecast_model_predict, forecast_model_tune
from emhass.command_line import publish_data
from emhass.utils import get_injection_dict, get_injection_dict_forecast_model_fit, \
    get_injection_dict_forecast_model_tune, build_params

# Define the Flask instance
app = Flask(__name__)

#check logfile for error, anything after string match if provided 
def checkFileLog(refString=None):
    if (refString is not None): 
       logArray = grabLog(refString) #grab reduced log array
    else: 
        if ((data_path / 'actionLogs.txt')).exists():
            with open(str(data_path / 'actionLogs.txt'), "r") as fp:
                    logArray = fp.readlines()
    for logString in logArray:
            if (logString.split(' ', 1)[0] == "ERROR"):
                return True     
    return False

#find string in logs, append all lines after to return
def grabLog(refString): 
    isFound = []
    output = []
    if ((data_path / 'actionLogs.txt')).exists():
            with open(str(data_path / 'actionLogs.txt'), "r") as fp:
                    logArray = fp.readlines()
            for x in range(len(logArray)-1): #find all matches and log key in isFound
                if (re.search(refString,logArray[x])):
                   isFound.append(x)
            if len(isFound) != 0:
                for x in range(isFound[-1],len(logArray)): #use isFound to extract last related action logs  
                    output.append(logArray[x])
    return output

#clear the log file
def clearFileLog(): 
    if ((data_path / 'actionLogs.txt')).exists():
        with open(str(data_path / 'actionLogs.txt'), "w") as fp:
            fp.truncate()    

#initial index page render
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

    # replace {{basename}} in html template html with path root  
    # basename = request.headers.get("X-Ingress-Path", "")
    # return make_response(template.render(injection_dict=injection_dict, basename=basename))
    
    return make_response(template.render(injection_dict=injection_dict))


#get actions 
@app.route('/template/<action_name>', methods=['GET'])
def template_action(action_name):
    app.logger.info(" >> Sending rendered template table data")
    if action_name == 'table-template':
        file_loader = PackageLoader('emhass', 'templates')
        env = Environment(loader=file_loader)
        template = env.get_template('template.html')
        if (data_path / 'injection_dict.pkl').exists():
            with open(str(data_path / 'injection_dict.pkl'), "rb") as fid:
                injection_dict = pickle.load(fid)
        else:
            app.logger.warning("The data container dictionary is empty... Please launch an optimization task")
            injection_dict={}        
        return make_response(template.render(injection_dict=injection_dict))

#post actions 
@app.route('/action/<action_name>', methods=['POST'])
def action_call(action_name):
    with open(str(data_path / 'params.pkl'), "rb") as fid:
        config_path, params = pickle.load(fid)
    runtimeparams = request.get_json(force=True)
    params = json.dumps(params)
    if runtimeparams is not None and runtimeparams != '{}':
        app.logger.info("Passed runtime parameters: " + str(runtimeparams))
    runtimeparams = json.dumps(runtimeparams)
    ActionStr = " >> Setting input data dict"
    app.logger.info(ActionStr)
    input_data_dict = set_input_data_dict(config_path, str(data_path), costfun, 
        params, runtimeparams, action_name, app.logger)
    if not input_data_dict:
        return make_response(grabLog(ActionStr), 400)
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
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
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
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
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
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
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
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
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
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
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
        with open(str(data_path / 'injection_dict.pkl'), "wb") as fid:
            pickle.dump(injection_dict, fid)
        msg = f'EMHASS >> Action forecast-model-tune executed... \n'
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
    parser.add_argument('--addon', type=strtobool, default='False', help='Define if we are usinng EMHASS with the add-on or in standalone mode')
    parser.add_argument('--no_response', type=strtobool, default='False', help='This is set if json response errors occur')
    args = parser.parse_args()
    
    #Obtain url and key from ENV or ARG (if any)
    hass_url = os.getenv("EMHASS_URL", default=args.url)
    key =  os.getenv("SUPERVISOR_TOKEN", default=args.key) 
    if hass_url != "http://supervisor/core/api":
        key =  os.getenv("EMHASS_KEY", key)  
    #If url or key is None, Set as empty string to reduce NoneType errors bellow
    if key is None: key = ""
    if hass_url is None: hass_url = ""
    
    #find env's, not not set defaults 
    use_options = os.getenv('USE_OPTIONS', default=False)
    CONFIG_PATH = os.getenv("CONFIG_PATH", default="/app/config_emhass.yaml")
    OPTIONS_PATH = os.getenv('OPTIONS_PATH', default="/app/options.json")
    DATA_PATH = os.getenv("DATA_PATH", default="/app/data/")
    
    #options None by default
    options = None 

    # Define the paths
    if args.addon==1:
        options_json = Path(OPTIONS_PATH)
        # Read options info
        if options_json.exists():
            with options_json.open('r') as data:
                options = json.load(data)
        else:
            app.logger.error("options.json does not exist")
            raise Exception("options.json does not exist in path: "+str(options_json)) 
    else:
        if use_options:
            options_json = Path(OPTIONS_PATH)
            # Read options info
            if options_json.exists():
                with options_json.open('r') as data:
                    options = json.load(data)
            else:
                app.logger.error("options.json does not exist")
                raise Exception("options.json does not exist in path: "+str(options_json)) 
        else:
            options = None       

    #if data path specified by options.json
    if options is not None:
        if options.get('data_path', None) != None and options.get('data_path', None) != "default":
            DATA_PATH = options.get('data_path', None);   

    config_path = Path(CONFIG_PATH)
    data_path = Path(DATA_PATH)
    
    # Read the example default config file
    if config_path.exists():
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        retrieve_hass_conf = config['retrieve_hass_conf']
        optim_conf = config['optim_conf']
        plant_conf = config['plant_conf']
    else:
        app.logger.error("Unable to open the default configuration yaml file")
        raise Exception("Failed to open config file, config_path: "+str(config_path)) 

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
        if url_from_options == 'empty' or url_from_options == '' or url_from_options == "http://supervisor/core/api":
            url = "http://supervisor/core/api/config"
        else:
            hass_url = url_from_options
            url = hass_url+"api/config"
        token_from_options = options.get('long_lived_token', 'empty')
        if token_from_options == 'empty' or token_from_options == '':
            long_lived_token = key
        else:
            long_lived_token = token_from_options
        headers = {
            "Authorization": "Bearer " + long_lived_token,
            "content-type": "application/json"
        }
        if not args.no_response==1:
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
        else: #if no_response is set to true
            costfun = os.getenv('LOCAL_COSTFUN', default='profit')
            logging_level = os.getenv('LOGGING_LEVEL', default='INFO')
            # check if secrets file exists
            if Path(os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml')).is_file(): 
                with open(os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml'), 'r') as file:
                    params_secrets = yaml.load(file, Loader=yaml.FullLoader)
                    app.logger.debug("Obtained secrets from secrets file")
            #If cant find secrets_emhass file, use env
            else: 
                app.logger.debug("Failed to find secrets file: "+str(os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml')))
                app.logger.debug("Setting location defaults")
                params_secrets = {} 
                #If no secrets file try args, else set some defaults 
                params_secrets['time_zone'] = os.getenv("TIME_ZONE", default="Europe/Paris")
                params_secrets['lat'] = float(os.getenv("LAT", default="45.83"))
                params_secrets['lon'] = float(os.getenv("LON", default="6.86"))
                params_secrets['alt'] = float(os.getenv("ALT", default="4807.8"))      
            #If ARG/ENV specify url and key, then override secrets file
            if hass_url != "":
                params_secrets['hass_url'] = hass_url
                app.logger.debug("Using URL obtained from ARG/ENV")
            else:
                hass_url = params_secrets.get('hass_url',"http://localhost:8123/")      
            if long_lived_token != "":
                params_secrets['long_lived_token'] = long_lived_token
                app.logger.debug("Using Key obtained from ARG/ENV")       
    else: #If addon is false
        costfun = os.getenv('LOCAL_COSTFUN', default='profit')
        logging_level = os.getenv('LOGGING_LEVEL', default='INFO')
        if Path(os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml')).is_file(): 
            with open(os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml'), 'r') as file:
                params_secrets = yaml.load(file, Loader=yaml.FullLoader)
            #Check if URL and KEY are provided by file. If not attempt using values from ARG/ENV
            if  params_secrets.get("hass_url", "empty") == "empty" or params_secrets['hass_url'] == "":
                app.logger.info("No specified Home Assistant URL in secrets_emhass.yaml. Attempting to get from ARG/ENV") 
                if hass_url != "":
                     params_secrets['hass_url'] = hass_url    
                else:
                    app.logger.error("Can not find Home Assistant URL from secrets_emhass.yaml or ARG/ENV")
                    raise Exception("Can not find Home Assistant URL from secrets_emhass.yaml or ARG/ENV")  
            else:
                hass_url = params_secrets['hass_url']
            if  params_secrets.get("long_lived_token", "empty") == "empty" or params_secrets['long_lived_token'] == "":
                app.logger.info("No specified Home Assistant KEY in secrets_emhass.yaml. Attempting to get from ARG/ENV") 
                if key != "":
                    params_secrets['long_lived_token'] = key
                else:
                    app.logger.error("Can not find Home Assistant KEY from secrets_emhass.yaml or ARG/ENV")
                    raise Exception("Can not find Home Assistant KEY from secrets_emhass.yaml or ARG/ENV")  
        else: #If no secrets file try args, else set some defaults 
            app.logger.info("Failed to find secrets_emhass.yaml in directory:" + os.getenv('SECRETS_PATH', default='/app/secrets_emhass.yaml') ) 
            app.logger.info("Attempting to use secrets from arguments or environment variables")        
            params_secrets = {} 
            params_secrets['time_zone'] = os.getenv("TIME_ZONE", default="Europe/Paris")
            params_secrets['lat'] = float(os.getenv("LAT", default="45.83"))
            params_secrets['lon'] = float(os.getenv("LON", default="6.86"))
            params_secrets['alt'] = float(os.getenv("ALT", default="4807.8"))      
            if hass_url != "":
                params_secrets['hass_url'] = hass_url
            else: #If cant find secrets_emhass and passed url ENV/ARG, then send error
                app.logger.error("No specified Home Assistant URL") 
                raise Exception("Can not find Home Assistant URL from secrets_emhass.yaml or ARG/ENV") 
            if key != "":
                params_secrets['long_lived_token'] = key
            else: #If cant find secrets_emhass and passed key ENV/ARG, then send error
                app.logger.error("No specified Home Assistant KEY")     
                raise Exception("Can not find Home Assistant KEY from secrets_emhass.yaml or ARG/ENV") 
    # Build params
    if use_options:
        params = build_params(params, params_secrets, options, 1, app.logger)
    else:
        params = build_params(params, params_secrets, options, args.addon, app.logger)
    if os.path.exists(str(data_path)): 
        with open(str(data_path / 'params.pkl'), "wb") as fid:
            pickle.dump((config_path, params), fid)
    else: 
        raise Exception("missing: " + str(data_path))        

    # Define logger
    #stream logger
    ch = logging.StreamHandler() 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    #Action File logger
    fileLogger = logging.FileHandler(str(data_path / 'actionLogs.txt')) 
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
    clearFileLog() #Clear Action File logger file, ready for new instance

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
