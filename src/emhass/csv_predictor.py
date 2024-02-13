#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import logging
import time
from typing import Optional
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import  r2_score

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=DeprecationWarning)

class CsvPredictor:
    r"""
    A forecaster class using machine learning models.
    
    This class uses the `sklearn` module and the machine learning models are from `scikit-learn`.
    
    It exposes two main methods:
    
    - `fit`: to train a model with the passed data.
    
    - `predict`: to obtain a forecast from a pre-trained model.
    
    """
    def __init__(self, data, model_type: str, independent_variables: list, dependent_variable: str, timestamp: str,
                logger: logging.Logger) -> None:
        r"""Define constructor for the forecast class.

        :param data: The data that will be used for train/test
        :type data: pd.DataFrame
        :param model_type: A unique name defining this model and useful to identify \
            for what it will be used for.
        :type model_type: str
        :param independent_variables: A list of independent variables. \
            Example: [`solar`, `degree_days`].
        :type independent_variables: list
        :param dependent_variable: The dependent variable(to be predicted). \
            Example: `hours`.
        :type dependent_variable: str
        :param timestamp: If defined, the column key that has to be used of timestamp.
        :type timestamp: str
        :param logger: The passed logger object
        :type logger: logging.Logger
        """
        self.data = data
        self.independent_variables = independent_variables
        self.dependent_variable = dependent_variable
        self.timestamp = timestamp
        self.model_type = model_type
        self.logger = logger
        self.data.sort_index(inplace=True)
        self.data = self.data[~self.data.index.duplicated(keep='first')]
    
    @staticmethod
    def add_date_features(data: pd.DataFrame, date_features: list, timestamp: str) -> pd.DataFrame:
        """Add date features from the input DataFrame timestamp

        :param data: The input DataFrame
        :type data: pd.DataFrame
        :param timestamp: The column containing the timestamp
        :type timestamp: str
        :return: The DataFrame with the added features
        :rtype: pd.DataFrame
        """
        df = copy.deepcopy(data)
        df[timestamp]= pd.to_datetime(df['timestamp'])
        if 'year' in date_features:
            df['year'] = [i.year for i in df['timestamp']]
        if 'month' in date_features:
            df['month'] = [i.month for i in df['timestamp']]
        if 'day_of_week' in date_features:
            df['day_of_week'] = [i.dayofweek for i in df['timestamp']]
        if 'day_of_year' in date_features:
            df['day_of_year'] = [i.dayofyear for i in df['timestamp']]
        if 'day' in date_features:
            df['day'] = [i.day for i in df['timestamp']]
        if 'hour' in date_features:
            df['hour'] = [i.day for i in df['timestamp']]

        return df

    def fit(self, date_features: Optional[list] = []) -> None:
        """
        Fit the model using the provided data.
        
        :param date_features: A list of 'date_features' to take into account when fitting the model.
        :type data: list
        """
        self.logger.info("Performing a csv model fit for "+self.model_type)
        self.data_exo = pd.DataFrame(self.data)
        self.data_exo[self.independent_variables] = self.data[self.independent_variables]
        self.data_exo[self.dependent_variable] = self.data[self.dependent_variable]
        keep_columns = []
        keep_columns.extend(self.independent_variables)
        if self.timestamp is not None:
            keep_columns.append(self.timestamp)
        keep_columns.append(self.dependent_variable)
        self.data_exo = self.data_exo[self.data_exo.columns.intersection(keep_columns)]
        self.data_exo.reset_index(drop=True, inplace=True)
        if len(date_features) > 0:
            if self.timestamp is not None:
                self.data_exo = CsvPredictor.add_date_features(self.data_exo, date_features, self.timestamp)
            else:
                self.logger.error("If no timestamp provided, you can't use date_features, going further without date_features.")

        y = self.data_exo[self.dependent_variable]
        self.data_exo = self.data_exo.drop(self.dependent_variable,axis=1)
        if self.timestamp is not None:
            self.data_exo = self.data_exo.drop(self.timestamp,axis=1)
        X = self.data_exo

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.steps = len(X_test)

        # Define the model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        # Define the parameters to tune
        param_grid = {
            'regressor__fit_intercept': [True, False],
            'regressor__positive': [True, False],
        }

        # Create a grid search object
        self.grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring=['r2', 'neg_mean_squared_error'], refit='r2', verbose=0, n_jobs=-1)
        # Fit the grid search object to the data
        self.logger.info("Fitting the model...")
        start_time = time.time()
        self.grid_search.fit(X_train.values, y_train.values)
        self.logger.info(f"Elapsed time for model fit: {time.time() - start_time}")

        self.model = self.grid_search.best_estimator_


        # Make predictions
        predictions = self.model.predict(X_test.values)
        predictions = pd.Series(predictions, index=X_test.index)
        pred_metric = r2_score(y_test,predictions)
        self.logger.info(f"Prediction R2 score of fitted model on test data: {pred_metric}")
        

    def predict(self, new_values:list) -> np.ndarray:
        r"""The predict method to generate a forecast from a csv file.


        :param new_values: The new values for the independent variables(in the same order as the independent variables list). \
            Example: [2.24, 5.68].
        :type new_values: list
        :return: The np.ndarray containing the predicted value.
        :rtype: np.ndarray
        """
        self.logger.info("Performing a prediction for "+self.model_type)
        new_values = np.array([new_values])

        return self.model.predict(new_values)
