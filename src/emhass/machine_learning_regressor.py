"""Machine learning regressor module."""

from __future__ import annotations

import copy
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)

REGRESSION_METHODS = {
    "LinearRegression": {
        "model": LinearRegression(),
        "param_grid": {
            "linearregression__fit_intercept": [True, False],
            "linearregression__positive": [True, False],
        },
    },
    "RidgeRegression": {
        "model": Ridge(),
        "param_grid": {"ridge__alpha": [0.1, 1.0, 10.0]},
    },
    "LassoRegression": {
        "model": Lasso(),
        "param_grid": {"lasso__alpha": [0.1, 1.0, 10.0]},
    },
    "RandomForestRegression": {
        "model": RandomForestRegressor(),
        "param_grid": {"randomforestregressor__n_estimators": [50, 100, 200]},
    },
    "GradientBoostingRegression": {
        "model": GradientBoostingRegressor(),
        "param_grid": {
            "gradientboostingregressor__n_estimators": [50, 100, 200],
            "gradientboostingregressor__learning_rate": [0.01, 0.1, 0.2],
        },
    },
    "AdaBoostRegression": {
        "model": AdaBoostRegressor(),
        "param_grid": {
            "adaboostregressor__n_estimators": [50, 100, 200],
            "adaboostregressor__learning_rate": [0.01, 0.1, 0.2],
        },
    },
}


class MLRegressor:
    r"""A forecaster class using machine learning models.

    This class uses the `sklearn` module and the machine learning models are \
        from `scikit-learn`.

    It exposes two main methods:

    - `fit`: to train a model with the passed data.

    - `predict`: to obtain a forecast from a pre-trained model.

    """

    def __init__(self: MLRegressor, data: pd.DataFrame, model_type: str, regression_model: str,
                 features: list, target: str, timestamp: str, logger: logging.Logger) -> None:
        r"""Define constructor for the forecast class.

        :param data: The data that will be used for train/test
        :type data: pd.DataFrame
        :param model_type: A unique name defining this model and useful to identify \
            for what it will be used for.
        :type model_type: str
        :param regression_model: The model that will be used. For now only \
            this options are possible: `LinearRegression`, `RidgeRegression`, \
            `LassoRegression`, `RandomForestRegression`, \
            `GradientBoostingRegression` and `AdaBoostRegression`.
        :type regression_model: str
        :param features: A list of features. \
            Example: [`solar_production`, `degree_days`].
        :type features: list
        :param target: The target(to be predicted). \
            Example: `heating_hours`.
        :type target: str
        :param timestamp: If defined, the column key that has to be used of timestamp.
        :type timestamp: str
        :param logger: The passed logger object
        :type logger: logging.Logger
        """
        self.data = data
        self.features = features
        self.target = target
        self.timestamp = timestamp
        self.model_type = model_type
        self.regression_model = regression_model
        self.logger = logger
        self.data = self.data.sort_index()
        self.data = self.data[~self.data.index.duplicated(keep="first")]
        self.data_exo = None
        self.steps = None
        self.model = None
        self.grid_search = None

    @staticmethod
    def add_date_features(data: pd.DataFrame, date_features: list, timestamp: str) -> pd.DataFrame:
        """Add date features from the input DataFrame timestamp.

        :param data: The input DataFrame
        :type data: pd.DataFrame
        :param timestamp: The column containing the timestamp
        :type timestamp: str
        :return: The DataFrame with the added features
        :rtype: pd.DataFrame
        """
        df = copy.deepcopy(data)  # noqa: PD901
        df[timestamp] = pd.to_datetime(df["timestamp"])
        if "year" in date_features:
            df["year"] = [i.year for i in df["timestamp"]]
        if "month" in date_features:
            df["month"] = [i.month for i in df["timestamp"]]
        if "day_of_week" in date_features:
            df["day_of_week"] = [i.dayofweek for i in df["timestamp"]]
        if "day_of_year" in date_features:
            df["day_of_year"] = [i.dayofyear for i in df["timestamp"]]
        if "day" in date_features:
            df["day"] = [i.day for i in df["timestamp"]]
        if "hour" in date_features:
            df["hour"] = [i.day for i in df["timestamp"]]
        return df

    def get_regression_model(self: MLRegressor) -> tuple[str, str]:
        r"""
        Get the base model and parameter grid for the specified regression model.
        Returns a tuple containing the base model and parameter grid corresponding to \
            the specified regression model.

        :param self: The instance of the MLRegressor class.
        :type self: MLRegressor
        :return: A tuple containing the base model and parameter grid.
        :rtype: tuple[str, str]
        """
        if self.regression_model == "LinearRegression":
            base_model = REGRESSION_METHODS["LinearRegression"]["model"]
            param_grid = REGRESSION_METHODS["LinearRegression"]["param_grid"]
        elif self.regression_model == "RidgeRegression":
            base_model = REGRESSION_METHODS["RidgeRegression"]["model"]
            param_grid = REGRESSION_METHODS["RidgeRegression"]["param_grid"]
        elif self.regression_model == "LassoRegression":
            base_model = REGRESSION_METHODS["LassoRegression"]["model"]
            param_grid = REGRESSION_METHODS["LassoRegression"]["param_grid"]
        elif self.regression_model == "RandomForestRegression":
            base_model = REGRESSION_METHODS["RandomForestRegression"]["model"]
            param_grid = REGRESSION_METHODS["RandomForestRegression"]["param_grid"]
        elif self.regression_model == "GradientBoostingRegression":
            base_model = REGRESSION_METHODS["GradientBoostingRegression"]["model"]
            param_grid = REGRESSION_METHODS["GradientBoostingRegression"]["param_grid"]
        elif self.regression_model == "AdaBoostRegression":
            base_model = REGRESSION_METHODS["AdaBoostRegression"]["model"]
            param_grid = REGRESSION_METHODS["AdaBoostRegression"]["param_grid"]
        else:
            self.logger.error(
                "Passed model %s is not valid",
                self.regression_model,
            )
            return None, None
        return base_model, param_grid

    def fit(self: MLRegressor, date_features: list | None = None) -> bool:
        r"""Fit the model using the provided data.

        :param date_features: A list of 'date_features' to take into account when \
            fitting the model.
        :type data: list
        :return: bool if successful
        :rtype: bool
        """
        self.logger.info("Performing a MLRegressor fit for %s", self.model_type)
        self.data_exo = pd.DataFrame(self.data)
        self.data_exo[self.features] = self.data[self.features]
        self.data_exo[self.target] = self.data[self.target]
        keep_columns = []
        keep_columns.extend(self.features)
        if self.timestamp is not None:
            keep_columns.append(self.timestamp)
        keep_columns.append(self.target)
        self.data_exo = self.data_exo[self.data_exo.columns.intersection(keep_columns)]
        self.data_exo = self.data_exo.reset_index(drop=True)
        if date_features is not None:
            if self.timestamp is not None:
                self.data_exo = MLRegressor.add_date_features(
                    self.data_exo,
                    date_features,
                    self.timestamp,
                )
            else:
                self.logger.error(
                    "If no timestamp provided, you can't use date_features, going \
                    further without date_features.",
                )
        y = self.data_exo[self.target]
        self.data_exo = self.data_exo.drop(self.target, axis=1)
        if self.timestamp is not None:
            self.data_exo = self.data_exo.drop(self.timestamp, axis=1)
        X = self.data_exo 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.steps = len(X_test)
        base_model, param_grid = self.get_regression_model()
        if base_model is None:
            return False
        self.model = make_pipeline(StandardScaler(), base_model)
        # Create a grid search object
        self.grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error",
                                        refit=True, verbose=0, n_jobs=-1)
        # Fit the grid search object to the data
        self.logger.info("Training a %s model", self.regression_model)
        start_time = time.time()
        self.grid_search.fit(X_train.values, y_train.values)
        self.logger.info("Elapsed time for model fit: %s", time.time() - start_time)
        self.model = self.grid_search.best_estimator_
        # Make predictions
        predictions = self.model.predict(X_test.values)
        predictions = pd.Series(predictions, index=X_test.index)
        pred_metric = r2_score(y_test, predictions)
        self.logger.info(
            "Prediction R2 score of fitted model on test data: %s",
            pred_metric,
        )
        return True

    def predict(self: MLRegressor, new_values: list) -> np.ndarray:
        """Predict a new value.

        :param new_values: The new values for the features \
            (in the same order as the features list). \
            Example: [2.24, 5.68].
        :type new_values: list
        :return: The np.ndarray containing the predicted value.
        :rtype: np.ndarray
        """
        self.logger.info("Performing a prediction for %s", self.model_type)
        new_values = np.array([new_values])
        return self.model.predict(new_values)
