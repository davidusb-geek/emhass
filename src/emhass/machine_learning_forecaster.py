import asyncio
import logging
import time
import warnings

import numpy as np
import pandas as pd
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    bayesian_search_forecaster,
)
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from emhass import utils

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

    def __init__(
        self,
        data: pd.DataFrame,
        model_type: str,
        var_model: str,
        sklearn_model: str,
        num_lags: int,
        emhass_conf: dict,
        logger: logging.Logger,
    ) -> None:
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
        self.forecaster: ForecasterRecursive | None = None
        self.optimize_results: pd.DataFrame | None = None
        self.optimize_results_object = None

        # A quick data preparation
        self._prepare_data()

    def _prepare_data(self):
        """Prepare the input data by cleaning and sorting."""
        self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace=True)
        self.data = self.data[~self.data.index.duplicated(keep="first")]

    @staticmethod
    def neg_r2_score(y_true, y_pred):
        """The negative of the r2 score."""
        return -r2_score(y_true, y_pred)

    @staticmethod
    async def interpolate_async(data: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values asynchronously."""
        return await asyncio.to_thread(data.interpolate, method="linear", axis=0, limit=None)

    @staticmethod
    def get_lags_list_from_frequency(freq: pd.Timedelta) -> list[int]:
        """Calculate appropriate lag values based on data frequency.

        The lags represent different time horizons (6h, 12h, 1d, 1.5d, 2d, 2.5d, 3d).
        This method scales these horizons according to the actual data frequency.

        :param freq: The frequency of the data as a pandas Timedelta
        :type freq: pd.Timedelta
        :return: A list of lag values appropriate for the data frequency
        :rtype: list[int]
        """
        # Define target time horizons in hours
        target_horizons_hours = [6, 12, 24, 36, 48, 60, 72]

        # Calculate frequency in hours
        freq_hours = freq.total_seconds() / 3600

        # Calculate lags for each horizon
        lags = [int(round(horizon / freq_hours)) for horizon in target_horizons_hours]

        # Remove duplicates and ensure minimum value of 1
        lags = sorted({max(1, lag) for lag in lags})

        return lags

    @staticmethod
    async def generate_exog(data_last_window, periods, var_name):
        """Generate the exogenous data for future timestamps."""
        forecast_dates = pd.date_range(
            start=data_last_window.index[-1] + data_last_window.index.freq,
            periods=periods,
            freq=data_last_window.index.freq,
        )
        exog = pd.DataFrame({var_name: [np.nan] * periods}, index=forecast_dates)
        exog = utils.add_date_features(exog)
        return exog

    def _get_sklearn_model(self, model_name: str):
        """Get the sklearn model instance based on the model name."""
        models = {
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(),
            "LassoRegression": Lasso(),
            "ElasticNet": ElasticNet(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "SVR": SVR(),
            "RandomForestRegressor": RandomForestRegressor(),
            "ExtraTreesRegressor": ExtraTreesRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "AdaBoostRegressor": AdaBoostRegressor(),
            "MLPRegressor": MLPRegressor(),
        }

        if model_name not in models:
            self.logger.error(
                f"Passed sklearn model {model_name} is not valid. Defaulting to KNeighborsRegressor"
            )
            return KNeighborsRegressor()

        return models[model_name]

    async def fit(
        self,
        split_date_delta: str | None = "48h",
        perform_backtest: bool | None = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        try:
            self.logger.info("Performing a forecast model fit for " + self.model_type)

            # Check if variable exists in data
            if self.var_model not in self.data.columns:
                raise KeyError(
                    f"Variable '{self.var_model}' not found in data columns: {list(self.data.columns)}"
                )

            # Preparing the data: adding exogenous features
            self.data_exo = pd.DataFrame(index=self.data.index)
            self.data_exo = utils.add_date_features(self.data_exo)
            self.data_exo[self.var_model] = self.data[self.var_model]

            self.data_exo = await self.interpolate_async(self.data_exo)

            # train/test split
            self.date_train = (
                self.data_exo.index[-1] - pd.Timedelta("5days") + self.data_exo.index.freq
            )  # The last 5 days
            self.date_split = (
                self.data_exo.index[-1] - pd.Timedelta(split_date_delta) + self.data_exo.index.freq
            )  # The last 48h
            self.data_train = self.data_exo.loc[: self.date_split - self.data_exo.index.freq, :]
            self.data_test = self.data_exo.loc[self.date_split :, :]
            self.steps = len(self.data_test)

            # Pick correct sklearn model
            base_model = self._get_sklearn_model(self.sklearn_model)

            # Define the forecaster object
            self.forecaster = ForecasterRecursive(regressor=base_model, lags=self.num_lags)

            # Fit and time it
            self.logger.info("Training a " + self.sklearn_model + " model")
            start_time = time.time()

            await asyncio.to_thread(
                self.forecaster.fit,
                y=self.data_train[self.var_model],
                exog=self.data_train.drop(self.var_model, axis=1),
                store_in_sample_residuals=True,
            )

            fit_time = time.time() - start_time
            self.logger.info(f"Elapsed time for model fit: {fit_time}")

            # Make a prediction to print metrics
            predictions = await asyncio.to_thread(
                self.forecaster.predict,
                steps=self.steps,
                exog=self.data_test.drop(self.var_model, axis=1),
            )
            pred_metric = await asyncio.to_thread(
                r2_score, self.data_test[self.var_model], predictions
            )
            self.logger.info(f"Prediction R2 score of fitted model on test data: {pred_metric}")

            # Packing results in a DataFrame
            df_pred = pd.DataFrame(index=self.data_exo.index, columns=["train", "test", "pred"])

            df_pred["train"] = self.data_train[self.var_model]
            df_pred["test"] = self.data_test[self.var_model]
            df_pred["pred"] = predictions

            df_pred_backtest = None

            if perform_backtest is True:
                # Using backtesting tool to evaluate the model
                self.logger.info("Performing simple backtesting of fitted model")
                start_time = time.time()
                cv = TimeSeriesFold(
                    steps=self.num_lags,
                    initial_train_size=None,
                    fixed_train_size=False,
                    gap=0,
                    allow_incomplete_fold=True,
                    refit=False,
                )

                metric, predictions_backtest = await asyncio.to_thread(
                    backtesting_forecaster,
                    forecaster=self.forecaster,
                    y=self.data_train[self.var_model],
                    exog=self.data_train.drop(self.var_model, axis=1),
                    cv=cv,
                    metric=MLForecaster.neg_r2_score,
                    verbose=False,
                    show_progress=True,
                )

                backtest_time = time.time() - start_time
                backtest_r2 = -metric
                self.logger.info(f"Elapsed backtesting time: {backtest_time}")
                self.logger.info(f"Backtest R2 score: {backtest_r2}")
                df_pred_backtest = pd.DataFrame(
                    index=self.data_exo.index, columns=["train", "pred"]
                )
                df_pred_backtest["train"] = self.data_exo[self.var_model]
                # Handle skforecast 0.18.0+ DataFrame output with fold column
                if isinstance(predictions_backtest, pd.DataFrame):
                    # Extract the 'pred' column from the DataFrame
                    pred_values = (
                        predictions_backtest["pred"]
                        if "pred" in predictions_backtest.columns
                        else predictions_backtest.iloc[:, -1]
                    )
                else:
                    # If it's a Series, use it directly
                    pred_values = predictions_backtest

                # Use loc to align indices properly - only assign where indices match
                df_pred_backtest.loc[pred_values.index, "pred"] = pred_values

            return df_pred, df_pred_backtest

        except asyncio.CancelledError:
            self.logger.info("Model training was cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error during model fitting: {e}")
            raise

    async def predict(
        self,
        data_last_window: pd.DataFrame | None = None,
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
        try:
            if self.forecaster is None:
                raise ValueError("Model has not been fitted yet. Call fit() first.")

            if data_last_window is None:
                predictions = await asyncio.to_thread(
                    self.forecaster.predict,
                    steps=self.num_lags,
                    exog=self.data_test.drop(self.var_model, axis=1),
                )
            else:
                data_last_window = await self.interpolate_async(data_last_window)

                if self.is_tuned:
                    exog = await self.generate_exog(data_last_window, self.lags_opt, self.var_model)

                    predictions = await asyncio.to_thread(
                        self.forecaster.predict,
                        steps=self.lags_opt,
                        last_window=data_last_window[self.var_model],
                        exog=exog.drop(self.var_model, axis=1),
                    )
                else:
                    exog = await self.generate_exog(data_last_window, self.num_lags, self.var_model)

                    predictions = await asyncio.to_thread(
                        self.forecaster.predict,
                        steps=self.num_lags,
                        last_window=data_last_window[self.var_model],
                        exog=exog.drop(self.var_model, axis=1),
                    )

            return predictions

        except asyncio.CancelledError:
            self.logger.info("Prediction was cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def _get_search_space(self, debug: bool, lags_list: list[int] | None = None):
        """Get the hyperparameter search space for the given model.

        :param debug: If True, use simplified search space for faster testing
        :type debug: bool
        :param lags_list: List of lag values to use. If None, uses default values
        :type lags_list: list[int] | None
        """
        if lags_list is None:
            lags_list = [6, 12, 24, 36, 48, 60, 72]

        debug_lags = [3]

        def get_lags(trial):
            return trial.suggest_categorical("lags", debug_lags if debug else lags_list)

        def svr_search_space(trial):
            search = {
                "C": trial.suggest_float("C", 0.1, 1.0) if debug else trial.suggest_float("C", 1e-2, 100.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                "lags": get_lags(trial),
            }
            # Only tune gamma if kernel is rbf
            if search["kernel"] == "rbf":
                 search["gamma"] = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
            else:
                 search["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            return search

        # Registry of search space generators
        search_spaces = {
            "LinearRegression": lambda trial: {
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True] if debug else [True, False]),
                "lags": get_lags(trial),
            },
            "RidgeRegression": lambda trial: {
                "alpha": trial.suggest_float("alpha", 0.1, 1.0) if debug else trial.suggest_float("alpha", 1e-4, 100.0, log=True),
                "lags": get_lags(trial),
            },
            "LassoRegression": lambda trial: {
                "alpha": trial.suggest_float("alpha", 0.1, 1.0) if debug else trial.suggest_float("alpha", 1e-4, 100.0, log=True),
                "lags": get_lags(trial),
            },
            "ElasticNet": lambda trial: {
                "alpha": trial.suggest_float("alpha", 0.0, 2.0),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "selection": trial.suggest_categorical("selection", ["random"] if debug else ["cyclic", "random"]),
                "lags": get_lags(trial),
            },
            "KNeighborsRegressor": lambda trial: {
                "n_neighbors": 2 if debug else trial.suggest_int("n_neighbors", 2, 20),
                "leaf_size": 20 if debug else trial.suggest_int("leaf_size", 20, 40),
                "weights": trial.suggest_categorical("weights", ["uniform"] if debug else ["uniform", "distance"]),
                "lags": get_lags(trial),
            },
            "DecisionTreeRegressor": lambda trial: {
                "max_depth": trial.suggest_int("max_depth", 2, 5) if debug else trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "lags": get_lags(trial),
            },
            "SVR": svr_search_space,
            "RandomForestRegressor": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 20) if debug else trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "lags": get_lags(trial),
            },
            "ExtraTreesRegressor": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 20) if debug else trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "lags": get_lags(trial),
            },
            "GradientBoostingRegressor": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 20) if debug else trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "lags": get_lags(trial),
            },
            "AdaBoostRegressor": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 20) if debug else trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
                "lags": get_lags(trial),
            },
            "MLPRegressor": lambda trial: {
                "learning_rate_init": trial.suggest_float("learning_rate_init", 0.001, 0.01),
                "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50)]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "lags": get_lags(trial),
            },
        }

        if self.sklearn_model not in search_spaces:
            raise ValueError(f"Unsupported model for tuning: {self.sklearn_model}")

        return search_spaces[self.sklearn_model]

    async def tune(
        self, split_date_delta: str | None = "48h", debug: bool | None = False
    ) -> pd.DataFrame:
        """Tuning a previously fitted model using bayesian optimization.

        :param split_date_delta: The delta from now to `split_date_delta` that will be used \
            as the test period to evaluate the model, defaults to '48h'.\
            This define the training/validation split for the tuning process.
        :type split_date_delta: Optional[str], optional
        :param debug: Set to True for testing and faster optimizations, defaults to False
        :type debug: Optional[bool], optional
        :return: The DataFrame with the forecasts using the optimized model.
        :rtype: pd.DataFrame
        """
        try:
            # Calculate appropriate lags based on data frequency
            freq_timedelta = pd.Timedelta(self.data_exo.index.freq)
            lags_list = MLForecaster.get_lags_list_from_frequency(freq_timedelta)
            self.logger.info(
                f"Using lags list based on data frequency ({self.data_exo.index.freq}): {lags_list}"
            )
            if self.forecaster is None:
                raise ValueError("Model has not been fitted yet. Call fit() first.")

            # Get the search space for this model
            search_space = self._get_search_space(debug, lags_list)

            # Bayesian search hyperparameter and lags with skforecast/optuna
            if debug:
                refit = False
                num_lags = 3
            else:
                refit = True
                num_lags = self.num_lags
            # The optimization routine call
            self.logger.info("Bayesian hyperparameter optimization with backtesting")
            start_time = time.time()

            # Use the 'y' data that will be passed to the optimizer
            data_to_tune = self.data_train[self.var_model]

            # Calculate the new split date and initial_train_size based on the passed split_date_delta
            try:
                date_split = (
                    data_to_tune.index[-1]
                    - pd.Timedelta(split_date_delta)
                    + data_to_tune.index.freq
                )
                initial_train_size = len(data_to_tune.loc[: date_split - data_to_tune.index.freq])
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid split_date_delta: {split_date_delta}. Falling back to 5 days."
                )
                date_split = (
                    data_to_tune.index[-1] - pd.Timedelta("5days") + data_to_tune.index.freq
                )
                initial_train_size = len(data_to_tune.loc[: date_split - data_to_tune.index.freq])

            # Check if the calculated initial_train_size is valid
            window_size = num_lags  # This is what skforecast will use as window_size
            if debug:
                window_size = 3  # Match debug lags

            if initial_train_size <= window_size:
                self.logger.warning(
                    f"Calculated initial_train_size ({initial_train_size}) is <= window_size ({window_size})."
                )
                self.logger.warning(
                    "This is likely because split_date_delta is too large for the dataset."
                )
                self.logger.warning(
                    f"Adjusting initial_train_size to {window_size + 1} to attempt recovery."
                )
                initial_train_size = window_size + 1

            cv = TimeSeriesFold(
                steps=num_lags,
                initial_train_size=initial_train_size,
                fixed_train_size=True,
                gap=0,
                skip_folds=None,
                allow_incomplete_fold=True,
                refit=refit,
            )

            (
                self.optimize_results,
                self.optimize_results_object,
            ) = await asyncio.to_thread(
                bayesian_search_forecaster,
                forecaster=self.forecaster,
                y=self.data_train[self.var_model],
                exog=self.data_train.drop(self.var_model, axis=1),
                cv=cv,
                search_space=search_space,
                metric=MLForecaster.neg_r2_score,
                n_trials=10,
                random_state=123,
                return_best=True,
            )

            optimization_time = time.time() - start_time
            self.logger.info(f"Elapsed time: {optimization_time}")

            self.is_tuned = True

            predictions_opt = await asyncio.to_thread(
                self.forecaster.predict,
                steps=self.num_lags,
                exog=self.data_test.drop(self.var_model, axis=1),
            )

            freq_hours = self.data_exo.index.freq.delta.seconds / 3600
            self.lags_opt = int(np.round(len(self.optimize_results.iloc[0]["lags"])))
            self.days_needed = int(np.round(self.lags_opt * freq_hours / 24))

            df_pred_opt = pd.DataFrame(
                index=self.data_exo.index, columns=["train", "test", "pred_optim"]
            )
            df_pred_opt["train"] = self.data_train[self.var_model]
            df_pred_opt["test"] = self.data_test[self.var_model]
            df_pred_opt["pred_optim"] = predictions_opt

            pred_optim_metric_train = -self.optimize_results.iloc[0]["neg_r2_score"]
            self.logger.info(
                f"R2 score for optimized prediction in train period: {pred_optim_metric_train}"
            )

            pred_optim_metric_test = await asyncio.to_thread(
                r2_score,
                df_pred_opt.loc[predictions_opt.index, "test"],
                df_pred_opt.loc[predictions_opt.index, "pred_optim"],
            )
            self.logger.info(
                f"R2 score for optimized prediction in test period: {pred_optim_metric_test}"
            )
            self.logger.info("Number of optimal lags obtained: " + str(self.lags_opt))

            return df_pred_opt

        except asyncio.CancelledError:
            self.logger.info("Model tuning was cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error during model tuning: {e}")
            raise
