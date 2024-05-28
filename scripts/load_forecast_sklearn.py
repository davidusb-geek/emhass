#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pathlib, pickle, copy, time
import plotly.io as pio
pio.renderers.default = 'browser'
pd.options.plotting.backend = "plotly"

from emhass.retrieve_hass import RetrieveHass
from emhass.forecast import Forecast
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
# from skopt.space import Categorical, Real, Integer


# the root folder
root = str(get_root(__file__, num_parent=2))
emhass_conf = {}
emhass_conf['config_path'] = pathlib.Path(root) / 'config_emhass.yaml'
emhass_conf['data_path'] = pathlib.Path(root) / 'data/'
emhass_conf['root_path'] = pathlib.Path(root)

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=True)


def add_date_features(data):
    df = copy.deepcopy(data)
    df['year'] = [i.year for i in df.index]
    df['month'] = [i.month for i in df.index]
    df['day_of_week'] = [i.dayofweek for i in df.index]
    df['day_of_year'] = [i.dayofyear for i in df.index]
    df['day'] = [i.day for i in df.index]
    df['hour'] = [i.hour for i in df.index]
    return df

def neg_r2_score(y_true, y_pred):
    return -r2_score(y_true, y_pred)

if __name__ == '__main__':

    days_to_retrieve = 240
    model_type = "load_forecast"
    var_model = "sensor.power_load_no_var_loads"
    var_model_in_kw = False
    sklearn_model = "KNeighborsRegressor"
    num_lags = 48
    
    data_path = emhass_conf['data_path'] / str('data_train_'+model_type+'.pkl')
    params = None
    template = 'presentation'

    if data_path.is_file():
        logger.info("Loading a previous data file")
        with open(data_path, "rb") as fid:
            data, var_model = pickle.load(fid)
    else:
        logger.info("Using EMHASS methods to retrieve the new forecast model train data")
        retrieve_hass_conf, _, _ = get_yaml_parse(emhass_conf, use_secrets=True)
        rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
        retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'],
        params, emhass_conf, logger, get_data_from_file=False)

        days_list = get_days_list(days_to_retrieve)
        var_list = [var_model]
        sensors_in_kw = [var_model_in_kw]
        rh.get_data(days_list, var_list, minimal_response=False, significant_changes_only=False, sensor_in_kw_list=sensors_in_kw)
        
        with open(data_path, 'wb') as fid:
            pickle.dump((rh.df_final, var_model), fid, pickle.HIGHEST_PROTOCOL)

        data = copy.deepcopy(rh.df_final)
        
    logger.info(data.describe())
    fig = data.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.show()
    fig.write_image(emhass_conf['root_path'] / "docs/images/inputs_power_load_forecast.svg", width=1080, height=0.8*1080)

    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    data = data[~data.index.duplicated(keep='first')]
    
    data_exo = pd.DataFrame(index=data.index)
    data_exo = add_date_features(data_exo)
    data_exo[var_model] = data[var_model]
    data_exo = data_exo.interpolate(method='linear', axis=0, limit=None)
    
    date_train = data_exo.index[-1]-pd.Timedelta('15days')+data_exo.index.freq # The last 15 days
    date_split = data_exo.index[-1]-pd.Timedelta('48h')+data_exo.index.freq # The last 48h
    data_train = data_exo.loc[:date_split,:]
    data_test  = data_exo.loc[date_split:,:]
    steps = len(data_test)
    
    if sklearn_model == 'LinearRegression':
        base_model = LinearRegression()
    elif sklearn_model == 'ElasticNet':
        base_model = ElasticNet()
    elif sklearn_model == 'KNeighborsRegressor':
        base_model = KNeighborsRegressor()
    else:
        logger.error("Passed sklearn model "+sklearn_model+" is not valid")
    
    forecaster = ForecasterAutoreg(
        regressor = base_model,
        lags      = num_lags
        )

    logger.info("Training a KNN regressor")
    start_time = time.time()
    forecaster.fit(y=data_train[var_model], 
                   exog=data_train.drop(var_model, axis=1))
    logger.info(f"Elapsed time: {time.time() - start_time}")
    
    # Predictions
    predictions = forecaster.predict(steps=steps, exog=data_train.drop(var_model, axis=1))
    pred_metric = r2_score(data_test[var_model],predictions)
    logger.info(f"Prediction R2 score: {pred_metric}")
    
    # Plot
    df = pd.DataFrame(index=data_exo.index,columns=['train','test','pred'])
    df['train'] = data_train[var_model]
    df['test'] = data_test[var_model]
    df['pred'] = predictions
    fig = df.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.update_xaxes(range=[date_train+pd.Timedelta('10days'), data_exo.index[-1]])
    fig.show()
    fig.write_image(emhass_conf['root_path'] / "docs/images/load_forecast_knn_bare.svg", width=1080, height=0.8*1080)
    
    logger.info("Simple backtesting")
    start_time = time.time()
    metric, predictions_backtest = backtesting_forecaster(
        forecaster         = forecaster,
        y                  = data_train[var_model],
        exog               = data_train.drop(var_model, axis=1),
        initial_train_size = None,
        fixed_train_size   = False,
        steps              = num_lags,
        metric             = neg_r2_score,
        refit              = False,
        verbose            = False
    )
    logger.info(f"Elapsed time: {time.time() - start_time}")
    logger.info(f"Backtest R2 score: {-metric}")
    
    df = pd.DataFrame(index=data_exo.index,columns=['train','pred'])
    df['train'] = data_exo[var_model]
    df['pred'] = predictions_backtest
    fig = df.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.show()
    fig.write_image(emhass_conf['root_path'] / "docs/images/load_forecast_knn_bare_backtest.svg", width=1080, height=0.8*1080)
    
    # Bayesian search hyperparameter and lags with Skopt
    
    # Lags used as predictors
    lags_grid = [6, 12, 24, 36, 48, 60, 72]

    # Regressor hyperparameters search space
    search_space = {'n_neighbors': Integer(2, 20, "uniform", name='n_neighbors'),
                    'leaf_size': Integer(20, 40, "log-uniform", name='leaf_size'),
                    'weights': Categorical(['uniform', 'distance'], name='weights')
                    }
    logger.info("Backtesting and bayesian hyperparameter optimization")
    start_time = time.time()
    results, optimize_results_object = bayesian_search_forecaster(
        forecaster         = forecaster,
        y                  = data_train[var_model],
        exog               = data_train.drop(var_model, axis=1),
        lags_grid          = lags_grid,
        search_space       = search_space,
        steps              = num_lags,
        metric             = neg_r2_score,
        refit              = True,
        initial_train_size = len(data_exo.loc[:date_train]),
        fixed_train_size   = True,
        n_trials           = 10,
        random_state       = 123,
        return_best        = True,
        verbose            = False,
        engine             = 'skopt',
        kwargs_gp_minimize = {}
    )
    logger.info(f"Elapsed time: {time.time() - start_time}")
    logger.info(results)
    logger.info(optimize_results_object)
    
    save_forecaster(forecaster, file_name='forecaster.py', verbose=False)
    
    forecaster_loaded = load_forecaster('forecaster.py', verbose=False)
    predictions_loaded = forecaster.predict(steps=steps, exog=data_train.drop(var_model, axis=1))
    
    df = pd.DataFrame(index=data_exo.index,columns=['train','test','pred','pred_naive','pred_optim'])
    freq_hours = df.index.freq.delta.seconds/3600
    lags_opt = int(np.round(len(results.iloc[0]['lags'])))
    days_needed = int(np.round(lags_opt*freq_hours/24))
    shift = int(24/freq_hours)
    P_load_forecast_naive = pd.concat([data_exo.iloc[-shift:], data_exo.iloc[:-shift]])
    df['train'] = data_train[var_model]
    df['test'] = data_test[var_model]
    df['pred'] = predictions
    df['pred_naive'] = P_load_forecast_naive[var_model].values
    df['pred_optim'] = predictions_loaded
    fig = df.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.update_xaxes(range=[date_train+pd.Timedelta('10days'), data_exo.index[-1]])
    fig.show()
    fig.write_image(emhass_conf['root_path'] / "docs/images/load_forecast_knn_optimized.svg", width=1080, height=0.8*1080)
    
    logger.info("######################## Train/Test R2 score comparison ######################## ")
    pred_naive_metric_train = r2_score(df.loc[data_train.index,'train'],df.loc[data_train.index,'pred_naive'])
    logger.info(f"R2 score for naive prediction in train period (backtest): {pred_naive_metric_train}")
    pred_optim_metric_train = -results.iloc[0]['neg_r2_score']
    logger.info(f"R2 score for optimized prediction in train period: {pred_optim_metric_train}")
    
    pred_metric_test = r2_score(df.loc[data_test.index[1:-1],'test'],df.loc[data_test[1:-1].index,'pred'])
    logger.info(f"R2 score for non-optimized prediction in test period: {pred_metric_test}")
    pred_naive_metric_test = r2_score(df.loc[data_test.index[1:-1],'test'],df.loc[data_test[1:-1].index,'pred_naive'])
    logger.info(f"R2 score for naive persistance forecast in test period: {pred_naive_metric_test}")
    pred_optim_metric_test = r2_score(df.loc[data_test.index[1:-1],'test'],df.loc[data_test[1:-1].index,'pred_optim'])
    logger.info(f"R2 score for optimized prediction in test period: {pred_optim_metric_test}")
    logger.info("################################################################################ ")
    
    logger.info("Number of optimal lags obtained: "+str(lags_opt))
    logger.info("Prediction in production using last_window")
    
    # Let's perform a naive load forecast for comparison
    retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(emhass_conf, use_secrets=True)
    fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf,
                    params, emhass_conf, logger)
    P_load_forecast = fcst.get_load_forecast(method='naive')
    
    # Then retrieve some data and perform a prediction mocking a production env
    rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
                      retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'],
                      params, emhass_conf, logger, get_data_from_file=False)

    days_list = get_days_list(days_needed)
    var_model = retrieve_hass_conf['var_load']
    var_list = [var_model]
    sensor_in_kw_list = [retrieve_hass_conf['var_load_in_kw']]
    rh.get_data(days_list, var_list, minimal_response=False, significant_changes_only=False, sensor_in_kw_list)
    data_last_window = copy.deepcopy(rh.df_final)
    
    data_last_window = add_date_features(data_last_window)
    data_last_window = data_last_window.interpolate(method='linear', axis=0, limit=None)
    
    predictions_prod = forecaster.predict(steps=lags_opt, 
                                          last_window=data_last_window[var_model],
                                          exog=data_last_window.drop(var_model, axis=1))
    
    df = pd.DataFrame(index=P_load_forecast.index,columns=['pred_naive','pred_prod'])
    df['pred_naive'] = P_load_forecast
    df['pred_prod'] = predictions_prod
    fig = df.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.show()
    fig.write_image(emhass_conf['root_path'] / "docs/images/load_forecast_production.svg", width=1080, height=0.8*1080)
