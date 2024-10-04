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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class MLForecaster:
    r"""
    A forecaster class using machine learning models with auto-regressive approach and features\
    based on timestamp information (hour, day, week, etc).
    
    This class uses the `skforecast` module and the machine learning models are from `scikit-learn`.
    
    It exposes three main methods:
    
    - `fit`: to train a model with the passed data.
    
    - `predict`: to obtain a forecast from a pre-trained model.
    
    - `tune`: to optimize the models hyperparameters using bayesian optimization. 
    
    """

    def __init__(self, data: pd.DataFrame, model_type: str, var_model: str, sklearn_model: str,
                 num_lags: int, emhass_conf: dict, logger: logging.Logger) -> None:
        r"""Define constructor for the forecast class.

        :param data: The data that will be used for train/test
        :type data: pd.DataFrame
        :param model_type: A unique name defining this model and useful to identify \
            for what it will be used for.
        :type model_type: str
        :param var_model: The name of the sensor to retrieve data from Home Assistant. \
            Example: `sensor.power_load_no_var_loads`.
        :type var_model: str
        :param sklearn_model: The `scikit-learn` model that will be used. For now only \
            this options are possible: `LinearRegression`, `ElasticNet` and `KNeighborsRegressor`.
        :type sklearn_model: str
        :param num_lags: The number of auto-regression lags to consider. A good starting point \
            is to fix this as one day. For example if your time step is 30 minutes, then fix this \
            to 48, if the time step is 1 hour the fix this to 24 and so on.
        :type num_lags: int
        :param emhass_conf: Dictionary containing the needed emhass paths
        :type emhass_conf: dict
        :param logger: The passed logger object
        :type logger: logging.Logger
        """
        self.data = data
        self.model_type = model_type
        self.var_model = var_model
        self.sklearn_model = sklearn_model
        self.num_lags = num_lags
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.is_tuned = False
        # A quick data preparation
        self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace=True)
        self.data = self.data[~self.data.index.duplicated(keep='first')]
    
    @staticmethod
    def add_date_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add date features from the input DataFrame timestamp

        :param data: The input DataFrame
        :type data: pd.DataFrame
        :return: The DataFrame with the added features
        :rtype: pd.DataFrame
        """
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
        """The negative of the r2 score."""
        return -r2_score(y_true, y_pred)
    
    @staticmethod
    def generate_exog(data_last_window, periods, var_name):
        """Generate the exogenous data for future timestamps."""
        forecast_dates = pd.date_range(start=data_last_window.index[-1]+data_last_window.index.freq, 
                                       periods=periods, 
                                       freq=data_last_window.index.freq)
        exog = pd.DataFrame({var_name:[np.nan]*periods},
                            index=forecast_dates)
        exog = MLForecaster.add_date_features(exog)
        return exog
    
    def fit(self, split_date_delta: Optional[str] = '48h', perform_backtest: Optional[bool] = False
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        r"""The fit method to train the ML model.

        :param split_date_delta: The delta from now to `split_date_delta` that will be used \
            as the test period to evaluate the model, defaults to '48h'
        :type split_date_delta: Optional[str], optional
        :param perform_backtest: If `True` then a back testing routine is performed to evaluate \
            the performance of the model on the complete train set, defaults to False
        :type perform_backtest: Optional[bool], optional
        :return: The DataFrame containing the forecast data results without and with backtest
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        self.logger.info("Performing a forecast model fit for "+self.model_type)
        # Preparing the data: adding exogenous features
        self.data_exo = pd.DataFrame(index=self.data.index)
        self.data_exo = MLForecaster.add_date_features(self.data_exo)
        self.data_exo[self.var_model] = self.data[self.var_model]
        self.data_exo = self.data_exo.interpolate(method='linear', axis=0, limit=None)
        # train/test split
        self.date_train = self.data_exo.index[-1]-pd.Timedelta('5days')+self.data_exo.index.freq # The last 5 days
        self.date_split = self.data_exo.index[-1]-pd.Timedelta(split_date_delta)+self.data_exo.index.freq # The last 48h
        self.data_train = self.data_exo.loc[:self.date_split-self.data_exo.index.freq,:]
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
            self.logger.error("Passed sklearn model "+self.sklearn_model+" is not valid. Defaulting to KNeighborsRegressor")
            base_model = KNeighborsRegressor()
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
        predictions = self.forecaster.predict(steps=self.steps, exog=self.data_test.drop(self.var_model, axis=1))
        pred_metric = r2_score(self.data_test[self.var_model],predictions)
        self.logger.info(f"Prediction R2 score of fitted model on test data: {pred_metric}")
        # Packing results in a DataFrame
        df_pred = pd.DataFrame(index=self.data_exo.index,columns=['train','test','pred'])
        df_pred['train'] = self.data_train[self.var_model]
        df_pred['test'] = self.data_test[self.var_model]
        df_pred['pred'] = predictions
        df_pred_backtest = None
        if perform_backtest is True:
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
                metric             = MLForecaster.neg_r2_score,
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
            ) -> pd.Series:
        """The predict method to generate forecasts from a previously fitted ML model.

        :param data_last_window: The data that will be used to generate the new forecast, this \
            will be freshly retrieved from Home Assistant. This data is needed because the forecast \
            model is an auto-regressive model with lags. If not passed then the data used during the \
            model train is used, defaults to None
        :type data_last_window: Optional[pd.DataFrame], optional
        :return: A pandas series containing the generated forecasts.
        :rtype: pd.Series
        """
        if data_last_window is None:
            predictions = self.forecaster.predict(steps=self.num_lags, exog=self.data_test.drop(self.var_model, axis=1))
        else:
            data_last_window = data_last_window.interpolate(method='linear', axis=0, limit=None)
            if self.is_tuned:
                exog = MLForecaster.generate_exog(data_last_window, self.lags_opt, self.var_model)
                predictions = self.forecaster.predict(steps=self.lags_opt, 
                                                      last_window=data_last_window[self.var_model],
                                                      exog=exog.drop(self.var_model, axis=1))
            else:
                exog = MLForecaster.generate_exog(data_last_window, self.num_lags, self.var_model)
                predictions = self.forecaster.predict(steps=self.num_lags, 
                                                      last_window=data_last_window[self.var_model],
                                                      exog=exog.drop(self.var_model, axis=1))
        return predictions
    
    def tune(self, debug: Optional[bool] = False) -> pd.DataFrame:
        """Tuning a previously fitted model using bayesian optimization.

        :param debug: Set to True for testing and faster optimizations, defaults to False
        :type debug: Optional[bool], optional
        :return: The DataFrame with the forecasts using the optimized model.
        :rtype: pd.DataFrame
        """
        # Regressor hyperparameters search space
        if self.sklearn_model == 'LinearRegression':
            if debug:
                def search_space(trial):
                    search_space  = {'fit_intercept': trial.suggest_categorical('fit_intercept', [True]),
                                     'lags': trial.suggest_categorical('lags', [3])} 
                    return search_space
            else:
                def search_space(trial):
                    search_space  = {'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                                     'lags': trial.suggest_categorical('lags', [6, 12, 24, 36, 48, 60, 72])}
                    return search_space
        elif self.sklearn_model == 'ElasticNet':
            if debug:
                def search_space(trial):
                    search_space  = {'selection': trial.suggest_categorical('selection', ['random']),
                                     'lags': trial.suggest_categorical('lags', [3])} 
                    return search_space
            else:
                def search_space(trial):
                    search_space  = {'alpha': trial.suggest_float('alpha', 0.0, 2.0),
                                    'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                                    'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
                                    'lags': trial.suggest_categorical('lags', [6, 12, 24, 36, 48, 60, 72])} 
                    return search_space
        elif self.sklearn_model == 'KNeighborsRegressor':
            if debug:
                def search_space(trial):
                    search_space  = {'weights': trial.suggest_categorical('weights', ['uniform']),
                                     'lags': trial.suggest_categorical('lags', [3])} 
                    return search_space
            else:
                def search_space(trial):
                    search_space  = {'n_neighbors': trial.suggest_int('n_neighbors', 2, 20),
                                    'leaf_size': trial.suggest_int('leaf_size', 20, 40),
                                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                                    'lags': trial.suggest_categorical('lags', [6, 12, 24, 36, 48, 60, 72])} 
                    return search_space
        # Bayesian search hyperparameter and lags with skforecast/optuna
        # Lags used as predictors
        if debug:
            refit = False
            num_lags = 3
        else:
            refit = True
            num_lags = self.num_lags
        # The optimization routine call
        self.logger.info("Bayesian hyperparameter optimization with backtesting")
        start_time = time.time()
        self.optimize_results, self.optimize_results_object = bayesian_search_forecaster(
            forecaster         = self.forecaster,
            y                  = self.data_train[self.var_model],
            exog               = self.data_train.drop(self.var_model, axis=1),
            search_space       = search_space,
            steps              = num_lags,
            metric             = MLForecaster.neg_r2_score,
            refit              = refit,
            initial_train_size = len(self.data_exo.loc[:self.date_train]),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = True,
            verbose            = False,
            engine             = 'optuna'
        )
        self.logger.info(f"Elapsed time: {time.time() - start_time}")
        self.is_tuned = True
        predictions_opt = self.forecaster.predict(steps=self.num_lags, exog=self.data_test.drop(self.var_model, axis=1))
        freq_hours = self.data_exo.index.freq.delta.seconds/3600
        self.lags_opt = int(np.round(len(self.optimize_results.iloc[0]['lags'])))
        self.days_needed = int(np.round(self.lags_opt*freq_hours/24))
        df_pred_opt = pd.DataFrame(index=self.data_exo.index,columns=['train','test','pred_optim'])
        df_pred_opt['train'] = self.data_train[self.var_model]
        df_pred_opt['test'] = self.data_test[self.var_model]
        df_pred_opt['pred_optim'] = predictions_opt
        pred_optim_metric_train = -self.optimize_results.iloc[0]['neg_r2_score']
        self.logger.info(f"R2 score for optimized prediction in train period: {pred_optim_metric_train}")
        pred_optim_metric_test = r2_score(df_pred_opt.loc[predictions_opt.index,'test'],
                                          df_pred_opt.loc[predictions_opt.index,'pred_optim'])
        self.logger.info(f"R2 score for optimized prediction in test period: {pred_optim_metric_test}")
        self.logger.info("Number of optimal lags obtained: "+str(self.lags_opt))
        return df_pred_opt
