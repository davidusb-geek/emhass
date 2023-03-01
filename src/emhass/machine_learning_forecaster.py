#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import copy
import time
from typing import Optional, Tuple
import pandas as pd
import numpy as np

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


class mlforecaster:
    r"""
    A forecaster class using machine learning models with auto-regressive approach and features\
    based on timestamp information (hour, day week, etc).
    
    This class uses the `skforecast` module and the machine learning models are from `scikit-learn`.
    
    """

    def __init__(self, data: pd.DataFrame, model_type: str, var_model: str, sklearn_model: str,
                 num_lags: int, root: str, logger: logging.Logger) -> None:
        self.data = data
        self.model_type = model_type
        self.var_model = var_model
        self.sklearn_model = sklearn_model
        self.num_lags = num_lags
        self.root = root
        self.logger = logger
        self.is_tuned = False
        # A quick data preparation
        self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace=True)
        self.data = self.data[~self.data.index.duplicated(keep='first')]
    
    @staticmethod
    def add_date_features(data):
        df = copy.deepcopy(data)
        df['year'] = [i.year for i in df.index]
        df['month'] = [i.month for i in df.index]
        df['day_of_week'] = [i.dayofweek for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['day'] = [i.day for i in df.index]
        df['hour'] = [i.hour for i in df.index]
        return df

    @staticmethod
    def neg_r2_score(y_true, y_pred):
        return -r2_score(y_true, y_pred)
    
    def fit(self, split_date_delta: Optional[str] = '48h', perform_backtest: Optional[bool] = False
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Performing a forecast model fit for "+self.model_type)
        # Preparing the data: adding exogenous features
        self.data_exo = pd.DataFrame(index=self.data.index)
        self.data_exo = mlforecaster.add_date_features(self.data_exo)
        self.data_exo[self.var_model] = self.data[self.var_model]
        self.data_exo = self.data_exo.interpolate(method='linear', axis=0, limit=None)
        # train/test split
        self.date_train = self.data_exo.index[-1]-pd.Timedelta('15days')+self.data_exo.index.freq # The last 15 days
        self.date_split = self.data_exo.index[-1]-pd.Timedelta(split_date_delta)+self.data_exo.index.freq # The last 48h
        self.data_train = self.data_exo.loc[:self.date_split,:]
        self.data_test  = self.data_exo.loc[self.date_split:,:]
        self.steps = len(self.data_test)
        # Pick correct sklearn model
        if self.sklearn_model == 'LinearRegression':
            base_model = LinearRegression()
        elif self.sklearn_model == 'ElasticNet':
            base_model = ElasticNet()
        elif self.sklearn_model == 'KNeighborsRegressor':
            base_model = KNeighborsRegressor()
        else:
            self.logger.error("Passed sklearn model "+self.sklearn_model+" is not valid")
        # Define the forecaster object
        self.forecaster = ForecasterAutoreg(
            regressor = base_model,
            lags      = self.num_lags
            )
        # Fit and time it
        self.logger.info("Training a "+self.sklearn_model+" model")
        start_time = time.time()
        self.forecaster.fit(y=self.data_train[self.var_model], 
                            exog=self.data_train.drop(self.var_model, axis=1))
        self.logger.info(f"Elapsed time for model fit: {time.time() - start_time}")
        # Make a prediction to print metrics
        predictions = self.forecaster.predict(steps=self.steps, exog=self.data_train.drop(self.var_model, axis=1))
        pred_metric = r2_score(self.data_test[self.var_model],predictions)
        self.logger.info(f"Prediction R2 score of fitted model on test data: {pred_metric}")
        # Packing results in a DataFrame
        df_pred = pd.DataFrame(index=self.data_exo.index,columns=['train','test','pred'])
        df_pred['train'] = self.data_train[self.var_model]
        df_pred['test'] = self.data_test[self.var_model]
        df_pred['pred'] = predictions
        df_pred_backtest = None
        if perform_backtest:
            # Using backtesting tool to evaluate the model
            self.logger.info("Performing simple backtesting of fitted model")
            start_time = time.time()
            metric, predictions_backtest = backtesting_forecaster(
                forecaster         = self.forecaster,
                y                  = self.data_train[self.var_model],
                exog               = self.data_train.drop(self.var_model, axis=1),
                initial_train_size = None,
                fixed_train_size   = False,
                steps              = self.num_lags,
                metric             = mlforecaster.neg_r2_score,
                refit              = False,
                verbose            = False
            )
            self.logger.info(f"Elapsed backtesting time: {time.time() - start_time}")
            self.logger.info(f"Backtest R2 score: {-metric}")
            df_pred_backtest = pd.DataFrame(index=self.data_exo.index,columns=['train','pred'])
            df_pred_backtest['train'] = self.data_exo[self.var_model]
            df_pred_backtest['pred'] = predictions_backtest
        return df_pred, df_pred_backtest
    
    def predict(self, data_last_window: Optional[pd.DataFrame] = None
            ) -> pd.DataFrame:
        if data_last_window is None:
            predictions = self.forecaster.predict(steps=self.num_lags, exog=self.data_train.drop(self.var_model, axis=1))
        else:
            data_last_window = mlforecaster.add_date_features(data_last_window)
            data_last_window = data_last_window.interpolate(method='linear', axis=0, limit=None)
            if self.is_tuned:
                predictions = self.forecaster.predict(steps=self.lags_opt, 
                                                    last_window=data_last_window[self.var_model],
                                                    exog=data_last_window.drop(self.var_model, axis=1))
            else:
                predictions = self.forecaster.predict(steps=self.num_lags, 
                                                    last_window=data_last_window[self.var_model],
                                                    exog=data_last_window.drop(self.var_model, axis=1))
        return predictions
    
    def tune(self) -> None:
        # Bayesian search hyperparameter and lags with Skopt
        # Lags used as predictors
        lags_grid = [6, 12, 24, 36, 48, 60, 72]
        # Regressor hyperparameters search space
        if self.sklearn_model == 'LinearRegression':
            search_space = {'fit_intercept': Categorical(['True', 'False'], name='fit_intercept')}
        elif self.sklearn_model == 'ElasticNet':
            search_space = {'alpha': Real(0.0, 2.0, "uniform", name='alpha'),
                            'l1_ratio': Real(0.0, 1.0, "uniform", name='l1_ratio'),
                            'selection': Categorical(['cyclic', 'random'], name='selection')
                            }
        elif self.sklearn_model == 'KNeighborsRegressor':
            search_space = {'n_neighbors': Integer(2, 20, "uniform", name='n_neighbors'),
                            'leaf_size': Integer(20, 40, "log-uniform", name='leaf_size'),
                            'weights': Categorical(['uniform', 'distance'], name='weights')
                            }
        self.logger.info("Backtesting and bayesian hyperparameter optimization")
        start_time = time.time()
        self.optimize_results, self.optimize_results_object = bayesian_search_forecaster(
            forecaster         = self.forecaster,
            y                  = self.data_train[self.var_model],
            exog               = self.data_train.drop(self.var_model, axis=1),
            lags_grid          = lags_grid,
            search_space       = search_space,
            steps              = self.num_lags,
            metric             = mlforecaster.neg_r2_score,
            refit              = True,
            initial_train_size = len(self.data_exo.loc[:self.date_train]),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = True,
            verbose            = False,
            engine             = 'skopt',
            kwargs_gp_minimize = {}
        )
        self.logger.info(f"Elapsed time: {time.time() - start_time}")
        self.is_tuned = True
        predictions_opt = self.forecaster.predict(steps=self.num_lags, exog=self.data_train.drop(self.var_model, axis=1))
        freq_hours = self.data_exo.index.freq.delta.seconds/3600
        self.lags_opt = int(np.round(len(self.optimize_results.iloc[0]['lags'])))
        self.days_needed = int(np.round(self.lags_opt*freq_hours/24))
        df = pd.DataFrame(index=self.data_exo.index,columns=['train','test','pred_optim'])
        df['train'] = self.data_train[self.var_model]
        df['test'] = self.data_test[self.var_model]
        df['pred_optim'] = predictions_opt
        pred_optim_metric_train = -self.optimize_results.iloc[0]['neg_r2_score']
        self.logger.info(f"R2 score for optimized prediction in train period: {pred_optim_metric_train}")
        pred_optim_metric_test = r2_score(df.loc[self.data_test.index[1:-1],'test'],df.loc[self.data_test[1:-1].index,'pred_optim'])
        self.logger.info(f"R2 score for optimized prediction in test period: {pred_optim_metric_test}")
        self.logger.info("Number of optimal lags obtained: "+str(self.lags_opt))
        return df