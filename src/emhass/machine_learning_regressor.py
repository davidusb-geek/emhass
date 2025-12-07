"""Machine learning regressor module."""

from __future__ import annotations

import asyncio
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

from emhass import utils

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
        "param_grid": {"ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]},
    },
    "LassoRegression": {
        "model": Lasso(),
        "param_grid": {"lasso__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]},
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

    def __init__(
        self: MLRegressor,
        data: pd.DataFrame,
        model_type: str,
        regression_model: str,
        features: list[str],
        target: str,
        timestamp: str,
        logger: logging.Logger,
    ) -> None:
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
        self.data = data.sort_index()
        self.features = features
        self.target = target
        self.timestamp = timestamp
        self.model_type = model_type
        self.regression_model = regression_model
        self.logger = logger

        self.data = self.data[~self.data.index.duplicated(keep="first")]
        self.data_exo: pd.DataFrame | None = None
        self.steps: int | None = None
        self.model = None
        self.grid_search: GridSearchCV | None = None

    def _prepare_data(self, date_features: list[str] | None) -> tuple[pd.DataFrame, pd.Series]:
        self.data_exo = self.data.copy()
        self.data_exo[self.features] = self.data[self.features]
        self.data_exo[self.target] = self.data[self.target]

        keep_columns = list(self.features)
        if self.timestamp:
            keep_columns.append(self.timestamp)
        keep_columns.append(self.target)
        self.data_exo = self.data_exo[keep_columns].reset_index(drop=True)

        if date_features and self.timestamp:
            self.data_exo = utils.add_date_features(
                self.data_exo, timestamp=self.timestamp, date_features=date_features
            )
        elif date_features:
            self.logger.warning("Timestamp is required for date_features. Skipping date features.")

        y = self.data_exo[self.target]
        X = self.data_exo.drop(
            columns=[self.target, self.timestamp] if self.timestamp else [self.target]
        )
        return X, y

    def _get_model_and_params(self) -> tuple[GridSearchCV, dict] | tuple[None, None]:
        r"""
        Get the base model and parameter grid for the specified regression model.
        Returns a tuple containing the base model and parameter grid corresponding to \
            the specified regression model.

        :param self: The instance of the MLRegressor class.
        :type self: MLRegressor
        :return: A tuple containing the base model and parameter grid.
        :rtype: tuple[str, str]
        """
        method = REGRESSION_METHODS.get(self.regression_model)
        if not method:
            self.logger.error("Invalid regression model: %s", self.regression_model)
            return None, None

        pipeline = make_pipeline(StandardScaler(), method["model"])
        param_grid = method["param_grid"]
        return pipeline, param_grid

    async def fit(self: MLRegressor, date_features: list[str] | None = None) -> bool:
        r"""Fit the model using the provided data.

        :param date_features: A list of 'date_features' to take into account when \
            fitting the model.
        :type data: list
        :return: bool if successful
        :rtype: bool
        """
        self.logger.info("Fitting MLRegressor model for %s", self.model_type)

        X, y = self._prepare_data(date_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.steps = len(X_test)

        model_pipeline, param_grid = self._get_model_and_params()
        if model_pipeline is None:
            return False

        self.grid_search = GridSearchCV(
            model_pipeline,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            refit=True,
            verbose=0,
            n_jobs=-1,
        )

        self.logger.info("Training model: %s", self.regression_model)
        start = time.time()
        await asyncio.to_thread(self.grid_search.fit, X_train.values, y_train.values)
        self.logger.info("Model fit completed in %.2f seconds", time.time() - start)

        self.model = self.grid_search.best_estimator_

        predictions = await asyncio.to_thread(self.model.predict, X_test.values)
        r2 = r2_score(y_test, predictions)
        self.logger.info("R2 score on test set: %.4f", r2)
        return True

    async def predict(self: MLRegressor, new_values: list[float]) -> np.ndarray:
        """Predict a new value.

        :param new_values: The new values for the features \
            (in the same order as the features list). \
            Example: [2.24, 5.68].
        :type new_values: list
        :return: The np.ndarray containing the predicted value.
        :rtype: np.ndarray
        """
        self.logger.info("Making prediction with model %s", self.model_type)
        new_values_array = np.array([new_values])
        prediction = await asyncio.to_thread(self.model.predict, new_values_array)
        return prediction
