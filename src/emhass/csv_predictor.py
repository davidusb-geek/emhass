#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import copy
import pathlib
import time
from typing import Optional
# from typing import Optional, Tuple
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import r2_score

# from skforecast.ForecasterAutoreg import ForecasterAutoreg
# from skforecast.model_selection import bayesian_search_forecaster
# from skforecast.model_selection import backtesting_forecaster

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class CsvPredictor:
    r"""
    A forecaster class using machine learning models.
    
    This class uses the `skforecast` module and the machine learning models are from `scikit-learn`.
    
    It exposes one main method:
    
    - `predict`: to obtain a forecast from a csv file.
    
    """

    # def __init__(self, data: pd.DataFrame, model_type: str, csv_file: str, independent_variables: list, dependent_variable: str, sklearn_model: str, new_values:list, root: str,
    #               logger: logging.Logger) -> None:
    def __init__(self, csv_file: str, independent_variables: list, dependent_variable: str, sklearn_model: str, new_values:list, root: str,
                  logger: logging.Logger) -> None:
        r"""Define constructor for the forecast class.

        :param data: The data that will be used for train/test
        :type data: pd.DataFrame
        :param model_type: A unique name defining this model and useful to identify \
            for what it will be used for.
        :type model_type: str
        :param csv_file: The name of the csv file to retrieve data from. \
            Example: `prediction.csv`.
        :type csv_file: str
        :param independent_variables: A list of independent variables. \
            Example: [`solar`, `degree_days`].
        :type independent_variables: list
        :param dependent_variable: The dependent variable(to be predicted). \
            Example: `hours`.
        :type dependent_variable: str
        :param sklearn_model: The `scikit-learn` model that will be used. For now only \
            this options are possible: `LinearRegression`, `ElasticNet` and `KNeighborsRegressor`.
        :type sklearn_model: str
        :param new_values: The new values for the independent variables(in the same order as the independent variables list). \
            Example: [2.24, 5.68].
        :type new_values: list
        :param root: The parent folder of the path where the config.yaml file is located
        :type root: str
        :param logger: The passed logger object
        :type logger: logging.Logger
        """
        # self.data = data
        # self.model_type = model_type
        self.csv_file = csv_file
        self.independent_variables = independent_variables
        self.dependent_variable = dependent_variable
        self.sklearn_model = sklearn_model
        self.new_values = new_values
        self.root = root
        self.logger = logger
        self.is_tuned = False

    
    def load_data(self):
        filename_path = pathlib.Path(self.root) / self.csv_file
        if filename_path.is_file():
            with open(filename_path, 'rb') as inp:
                data = pd.read_csv(filename_path)
        else:
            self.logger.error("The cvs file was not found.")
            return

        required_columns = self.independent_variables
        
        if not set(required_columns).issubset(data.columns):
            raise ValueError(
                f"CSV file should contain the following columns: {', '.join(required_columns)}"
            )
        print(type(data))
        return data
    
    def prepare_data(self, data):
        """
        Prepare the data.
        
        :param data: Input Data
        :return: Input DataFrame with freq defined
        :rtype: pd.DataFrame
        
        """
        X = data[self.independent_variables].values
        y = data[self.dependent_variable].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(type(X_train))
        print(type(y_train))
        
        return X_train, y_train
    
    
    # def predict(self, perform_backtest: Optional[bool] = False
    #         ) -> pd.Series:
    def predict(self):
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
        self.logger.info("Performing a prediction for "+self.csv_file)
        # Preparing the data: adding exogenous features
        data = self.load_data()
        X, y = self.prepare_data(data)
        
        if self.sklearn_model == 'LinearRegression':
            base_model = LinearRegression()
        elif self.sklearn_model == 'ElasticNet':
            base_model = ElasticNet()
        elif self.sklearn_model == 'KNeighborsRegressor':
            base_model = KNeighborsRegressor()
        else:
            self.logger.error("Passed sklearn model "+self.sklearn_model+" is not valid")
        # Define the forecaster object
        self.forecaster = base_model
        # Fit and time it
        self.logger.info("Training a "+self.sklearn_model+" model")
        start_time = time.time()
        self.forecaster.fit(X, y)
        self.logger.info(f"Elapsed time for model fit: {time.time() - start_time}")
        new_values = np.array([self.new_values])
        prediction = self.forecaster.predict(new_values)
        print(type(prediction))
        
        return prediction
    
    
    
    