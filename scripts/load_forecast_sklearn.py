#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pathlib, pickle, copy, time
import plotly.io as pio
pio.renderers.default = 'browser'
pd.options.plotting.backend = "plotly"

from emhass.retrieve_hass import retrieve_hass
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
from skopt.space import Categorical, Real, Integer


# the root folder
root = str(get_root(__file__, num_parent=1))
# create logger
logger, ch = get_logger(__name__, root, save_to_file=False)

if __name__ == '__main__':

    params = None
    data_path = pathlib.Path(root+'/data/data.pkl')
    template = 'presentation'

    if data_path.is_file():
        logger.info("Loading a previous data file")
        with open(data_path, "rb") as fid:
            data = pickle.load(fid)
    else:
        logger.info("Using EMHASS methods to retrieve the data")
        retrieve_hass_conf, _, _ = get_yaml_parse(pathlib.Path(root+'/config_emhass.yaml'), use_secrets=True)
        rh = retrieve_hass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
        retrieve_hass_conf['freq'], retrieve_hass_conf['time_zone'],
        params, root, logger, get_data_from_file=False)

        days_list = get_days_list(retrieve_hass_conf['days_to_retrieve'])
        var_model = retrieve_hass_conf['var_load']
        var_list = [var_model]
        rh.get_data(days_list, var_list)
        
        with open(data_path, 'wb') as fid:
            pickle.dump(rh.df_final, fid, pickle.HIGHEST_PROTOCOL)

        data = copy.deepcopy(rh.df_final)
        
    logger.info(data.describe())
    fig = data.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.show()
    fig.write_image(root + "/docs/images/inputs_power_load_forecast.png", width=1080, height=0.8*1080)

    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    data = data[~data.index.duplicated(keep='first')]
    
    data_exo = pd.DataFrame(index=data.index)
    data_exo['year'] = [i.year for i in data_exo.index]
    data_exo['month'] = [i.month for i in data_exo.index]
    data_exo['day_of_week'] = [i.dayofweek for i in data_exo.index]
    data_exo['day_of_year'] = [i.dayofyear for i in data_exo.index]
    data_exo['day'] = [i.day for i in data_exo.index]
    data_exo['hour'] = [i.hour for i in data_exo.index]
    data_exo[var_model] = data[var_model]
    data_exo = data_exo.interpolate(method='linear', axis=0, limit=None)
    
    date_train = data_exo.index[-1]-pd.Timedelta('15days')+data_exo.index.freq # The last 15 days
    date_split = data_exo.index[-1]-pd.Timedelta('48h')+data_exo.index.freq # The last 48h
    data_train = data_exo.loc[:date_split,:]
    data_test  = data_exo.loc[date_split:,:]
    steps = len(data_test)
    
    forecaster = ForecasterAutoreg(
        regressor = KNeighborsRegressor(),
        lags      = 48
        )

    logger.info("Training a KNN regressor")
    start_time = time.time()
    forecaster.fit(y=data_train[var_model], 
                   exog=data_train.drop(var_model, axis=1))
    logger.info(f"Elapsed time: {time.time() - start_time}")
    
    # Predictions
    predictions = forecaster.predict(steps=steps, exog=data_train.drop(var_model, axis=1))
    pred_metric = r2_score(data_test[var_model],predictions)
    logger.info(f"Prediction error: {pred_metric}")
    
    # Plot
    df = pd.DataFrame(index=data_exo.index,columns=['train','test','predictions'])
    df['train'] = data_train[var_model]
    df['test'] = data_test[var_model]
    df['predictions'] = predictions
    fig = df.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.update_xaxes(range=[date_train+pd.Timedelta('5days'), data_exo.index[-1]])
    fig.show()
    fig.write_image(root + "/docs/images/load_forecast_knn_bare.png", width=1080, height=0.8*1080)
    
    logger.info("Simple backtesting")
    start_time = time.time()
    metric, predictions_backtest = backtesting_forecaster(
        forecaster         = forecaster,
        y                  = data_train[var_model],
        exog               = data_train.drop(var_model, axis=1),
        initial_train_size = None,
        fixed_train_size   = False,
        steps              = 48, #10
        metric             = r2_score,
        refit              = False, #True
        verbose            = False
    )
    logger.info(f"Elapsed time: {time.time() - start_time}")
    logger.info(f"Backtest error: {metric}")
    
    df = pd.DataFrame(index=data_exo.index,columns=['train','predictions'])
    df['train'] = data_exo[var_model]
    df['predictions'] = predictions_backtest
    fig = df.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.show()
    fig.write_image(root + "/docs/images/load_forecast_knn_bare_backtest.png", width=1080, height=0.8*1080)
    
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
        steps              = 48,
        metric             = r2_score,
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
    pred_metric = r2_score(data_test[var_model],predictions_loaded)
    logger.info(f"Prediction loaded and optimized model error: {pred_metric}")
    
    df = pd.DataFrame(index=data_exo.index,columns=['train','test','predictions'])
    df['train'] = data_train[var_model]
    df['test'] = data_test[var_model]
    df['predictions'] = predictions
    df['predictions_optimized'] = predictions_loaded
    fig = df.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.update_xaxes(range=[date_train+pd.Timedelta('5days'), data_exo.index[-1]])
    fig.show()
    fig.write_image(root + "/docs/images/load_forecast_knn_optimized.png", width=1080, height=0.8*1080)