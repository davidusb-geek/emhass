#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pathlib, pickle, copy, time
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
pd.options.plotting.backend = "plotly"

from emhass.retrieve_hass import RetrieveHass
from emhass.forecast import Forecast
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger, build_secrets, build_params

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, silhouette_score

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
# from skopt.space import Categorical, Real, Integer

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler


# the root folder
root = pathlib.Path(str(get_root(__file__, num_parent=2)))
emhass_conf = {}
emhass_conf['data_path'] = root / 'data/'
emhass_conf['root_path'] = root / 'src/emhass/'
emhass_conf['config_path'] = root / 'config.json'
emhass_conf['defaults_path'] = emhass_conf['root_path']  / 'data/config_defaults.json'
emhass_conf['associations_path'] = emhass_conf['root_path']  / 'data/associations.csv'

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=True)

if __name__ == '__main__':

    days_to_retrieve = 240
    model_type = "load_clustering"
    var_model = "sensor.power_load_positive"

    # Build params with no config and default secrets
    data_path = emhass_conf['data_path'] / str('data_train_'+model_type+'.pkl')
    _,secrets = build_secrets(emhass_conf,logger,no_response=True)
    params =  build_params(emhass_conf,secrets,{},logger)
    template = 'presentation'

    if data_path.is_file():
        logger.info("Loading a previous data file")
        with open(data_path, "rb") as fid:
            data, var_model = pickle.load(fid)
    else:
        logger.info("Using EMHASS methods to retrieve the new forecast model train data")
        retrieve_hass_conf, _, _ = get_yaml_parse(params,logger)
        rh = RetrieveHass(retrieve_hass_conf['hass_url'], retrieve_hass_conf['long_lived_token'], 
        retrieve_hass_conf['optimization_time_step'], retrieve_hass_conf['time_zone'],
        params, emhass_conf, logger, get_data_from_file=False)

        days_list = get_days_list(days_to_retrieve)
        var_list = [var_model]
        rh.get_data(days_list, var_list)
        
        with open(data_path, 'wb') as fid:
            pickle.dump((rh.df_final, var_model), fid, pickle.HIGHEST_PROTOCOL)

        data = copy.deepcopy(rh.df_final)
        
    logger.info(data.describe())
    
    # Plot the input data
    fig = data.plot()
    fig.layout.template = template
    fig.update_yaxes(title_text = "Power (W)")
    fig.update_xaxes(title_text = "Time")
    fig.show()
    
    data_lag = pd.concat([data, data.shift()], axis=1)
    data_lag.columns = ['power_load y(t)', 'power_load y(t+1)']
    data_lag = data_lag.dropna()
    
    fig2 = data_lag.plot.scatter(x='power_load y(t)', y='power_load y(t+1)', c='DarkBlue')
    fig2.layout.template = template
    fig2.show()
    
    # Elbow method to check how many clusters
    # distortions = []
    # K = range(1,12)

    # for cluster_size in K:
    #     kmeans = KMeans(n_clusters=cluster_size, init='k-means++')
    #     kmeans = kmeans.fit(data_lag)
    #     distortions.append(kmeans.inertia_)
        
    # df = pd.DataFrame({'Clusters': K, 'Distortions': distortions})
    # fig = (px.line(df, x='Clusters', y='Distortions', template=template)).update_traces(mode='lines+markers')
    # fig.show()
    
    # The silouhette method
    silhouette_scores = []
    K = range(2,12)

    for cluster_size in K:
        kmeans = KMeans(n_clusters=cluster_size, init='k-means++', random_state=200)
        labels = kmeans.fit(data_lag).labels_
        silhouette_score_tmp = silhouette_score(data_lag, labels, metric='euclidean', 
                                                sample_size=1000, random_state=200)
        silhouette_scores.append(silhouette_score_tmp)

    df = pd.DataFrame({'Clusters': K, 'Silhouette Score': silhouette_scores})
    fig = (px.line(df, x='Clusters', y='Silhouette Score', template=template)).update_traces(mode='lines+markers')
    fig.show()
    
    # The clustering
    kmeans = KMeans(n_clusters=6, init='k-means++')
    kmeans = kmeans.fit(data_lag)
    data_lag['cluster_group'] = kmeans.labels_

    fig = px.scatter(data_lag, x='power_load y(t)', y='power_load y(t+1)', color='cluster_group', template=template)
    fig.show()
    
    km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=200)
    y_pred = km.fit_predict(data_lag)
    data_lag['cluster_group_tslearn_euclidean'] = y_pred
    
    fig = px.scatter(data_lag, x='power_load y(t)', y='power_load y(t+1)', color='cluster_group_tslearn_euclidean', template=template)
    fig.show()
    
    # dba_km = TimeSeriesKMeans(n_clusters=6, n_init=2, metric="dtw", verbose=True, max_iter_barycenter=10, random_state=200)
    # y_pred = dba_km.fit_predict(data_lag)
    # data_lag['cluster_group_tslearn_dba'] = y_pred
    
    # fig = px.scatter(data_lag, x='power_load y(t)', y='power_load y(t+1)', color='cluster_group_tslearn_dba', template=template)
    # fig.show()
    
    # sdtw_km = TimeSeriesKMeans(n_clusters=6, metric="softdtw", metric_params={"gamma": .01}, verbose=True, random_state=200)
    # y_pred = sdtw_km.fit_predict(data_lag)
    # data_lag['cluster_group_tslearn_sdtw'] = y_pred
    
    # fig = px.scatter(data_lag, x='power_load y(t)', y='power_load y(t+1)', color='cluster_group_tslearn_sdtw', template=template)
    # fig.show()