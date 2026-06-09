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
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from emhass import utils

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MLForecaster:
    r"""
    A forecaster class using machine learning models with auto-regressive approach and features\
    based on timestamp information (hour, day, week, etc).

    This class uses the `skforecast` module and the machine learning models are from `scikit-learn`.

    In addition to the always-present date features, the forecaster can optionally consume\
    extra *exogenous* covariates (for example weather columns such as the outside temperature)\
    that are known over the whole train/forecast window. These are passed through to\
    `skforecast`'s ``ForecasterRecursive`` ``exog`` argument unchanged. The columns to use are\
    listed in ``weather_features`` and must be present in the input ``data`` (for fit/tune) and\
    supplied for the forecast horizon at predict time (see ``predict``). When ``weather_features``\
    is empty the behaviour is identical to a date-features-only model.

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
        weather_features: list[str] | None = None,
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
        :param weather_features: An optional list of extra exogenous covariate column names \
            (for example weather columns such as ``temp_air``) that are known over the whole \
            train/forecast window and should be fed to the model in addition to the date \
            features. These columns must be present in ``data``. Defaults to None (date \
            features only, i.e. the historical behaviour).
        :type weather_features: Optional[list[str]], optional
        """
        self.data = data
        self.model_type = model_type
        self.var_model = var_model
        self.sklearn_model = sklearn_model
        self.num_lags = num_lags
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.weather_features = list(weather_features) if weather_features else []
        self.is_tuned = False
        self.forecaster: ForecasterRecursive | None = None
        self.optimize_results: pd.DataFrame | None = None
        self.optimize_results_object = None
        self.backtest_metrics_: dict | None = None

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
    async def generate_exog(
        data_last_window, periods, var_name, weather_features=None, weather_future=None
    ):
        """Generate the exogenous data for future timestamps.

        :param data_last_window: The last window of observed data, used to anchor the future date \
            range.
        :param periods: The number of future steps to generate exog for.
        :param var_name: The name of the (target) variable column to seed the frame with.
        :param weather_features: Optional list of extra covariate column names to carry over from \
            ``weather_future`` into the generated exog. Defaults to None (date features only).
        :param weather_future: Optional DataFrame holding the future values of the columns named \
            in ``weather_features``. It is aligned onto the generated future date range. Defaults \
            to None.
        """
        forecast_dates = pd.date_range(
            start=data_last_window.index[-1] + data_last_window.index.freq,
            periods=periods,
            freq=data_last_window.index.freq,
        )
        exog = pd.DataFrame({var_name: [np.nan] * periods}, index=forecast_dates)
        exog = utils.add_date_features(exog)
        if weather_features:
            exog = MLForecaster._merge_weather_exog(
                exog, weather_features, weather_future, "future"
            )
        return exog

    @staticmethod
    def _merge_weather_exog(exog, weather_features, weather_source, context):
        """Align and attach the configured weather covariate columns onto an exog frame.

        The columns are reindexed onto ``exog.index`` (so the same calendar instants are used) and
        any residual gaps are filled by interpolation then forward/backward fill, mirroring how the
        date features are always fully populated. A missing column or an empty/None source raises a
        clear ``KeyError``/``ValueError`` so the configuration mistake surfaces early rather than as
        an opaque skforecast "Missing columns in exog" error later.

        :param exog: The exog frame (already carrying the date features) to enrich.
        :param weather_features: List of weather covariate column names to attach.
        :param weather_source: A DataFrame holding those columns over (at least) the span of \
            ``exog.index``.
        :param context: A short label ('train' or 'future') used only for error messages.
        """
        if weather_source is None:
            raise ValueError(
                f"weather_features {weather_features} were configured but no weather data was "
                f"provided for the {context} window"
            )
        missing = [column for column in weather_features if column not in weather_source.columns]
        if missing:
            raise KeyError(
                f"Configured weather_features {missing} not found in the {context} weather data "
                f"columns: {list(weather_source.columns)}"
            )
        aligned = weather_source[weather_features].reindex(exog.index)
        aligned = aligned.interpolate(method="linear", axis=0, limit_direction="both")
        aligned = aligned.ffill().bfill()
        for column in weather_features:
            exog[column] = aligned[column].to_numpy()
        return exog

    def _get_sklearn_model(self, model_name: str):
        """Get the sklearn model instance based on the model name."""
        seed = 42
        models = {
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(),
            "LassoRegression": Lasso(random_state=seed),
            "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=seed),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "DecisionTreeRegressor": DecisionTreeRegressor(ccp_alpha=0.0, random_state=seed),
            "SVR": SVR(),
            "RandomForestRegressor": RandomForestRegressor(
                min_samples_leaf=1, max_features=1.0, random_state=seed
            ),
            "ExtraTreesRegressor": ExtraTreesRegressor(
                min_samples_leaf=1, max_features=1.0, random_state=seed
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                learning_rate=0.1, random_state=seed
            ),
            "AdaBoostRegressor": AdaBoostRegressor(learning_rate=1.0, random_state=seed),
            "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100,), random_state=seed),
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
            the performance of the model on the complete train set, defaults to False.
            When True, the ``backtest_metrics_`` attribute is populated with a dict containing
            MAE, RMSE, R2, and MAPE computed over the out-of-sample backtest folds.
        :type perform_backtest: Optional[bool], optional
        :return: The DataFrame containing the forecast data results without and with backtest
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        try:
            self.logger.info("Performing a forecast model fit for " + self.model_type)
            # Reset metrics so a reused instance never exposes stale results from a previous fit.
            self.backtest_metrics_ = None

            # Check if variable exists in data
            if self.var_model not in self.data.columns:
                raise KeyError(
                    f"Variable '{self.var_model}' not found in data columns: {list(self.data.columns)}"
                )

            # Preparing the data: adding exogenous features
            self.data_exo = pd.DataFrame(index=self.data.index)
            self.data_exo = utils.add_date_features(self.data_exo)
            # Optional extra exogenous covariates (e.g. weather). They must be present as columns
            # in the training data, aligned to the load history. When weather_features is empty
            # this block is a no-op and the model is date-features-only (historical behaviour).
            if self.weather_features:
                self.data_exo = MLForecaster._merge_weather_exog(
                    self.data_exo, self.weather_features, self.data, "train"
                )
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
            self.forecaster = ForecasterRecursive(estimator=base_model, lags=self.num_lags)

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

                # Compute goodness-of-fit metrics over the backtest fold predictions.
                # Only rows where the backtest produced a prediction are used; the leading
                # rows (the initial warm-up window) remain NaN and are excluded.
                actual_values = df_pred_backtest["train"].astype(float)
                predicted_values = df_pred_backtest["pred"].astype(float)
                valid_mask = predicted_values.notna() & actual_values.notna()
                n_valid_samples = int(valid_mask.sum())
                actual_valid = actual_values[valid_mask]
                predicted_valid = predicted_values[valid_mask]

                # Guard against degenerate cases (all-NaN predictions or single sample).
                # r2_score requires at least 2 samples; MAE/RMSE require at least 1.
                if n_valid_samples == 0:
                    self.backtest_metrics_ = {
                        "mae": float("nan"),
                        "rmse": float("nan"),
                        "r2": float("nan"),
                        "mape": float("nan"),
                        "n_samples": 0,
                    }
                    self.logger.warning(
                        "Backtest produced no valid predictions — metrics set to NaN"
                    )
                else:
                    backtest_mae = float(mean_absolute_error(actual_valid, predicted_valid))
                    backtest_rmse = float(
                        np.sqrt(mean_squared_error(actual_valid, predicted_valid))
                    )
                    # r2_score is undefined for a single sample (variance == 0)
                    backtest_r2_full = (
                        float(r2_score(actual_valid, predicted_valid))
                        if n_valid_samples > 1
                        else float("nan")
                    )
                    # MAPE: exclude zero actuals to avoid division by zero
                    nonzero_mask = actual_valid != 0
                    if nonzero_mask.sum() > 0:
                        backtest_mape = float(
                            np.mean(
                                np.abs(
                                    (actual_valid[nonzero_mask] - predicted_valid[nonzero_mask])
                                    / actual_valid[nonzero_mask]
                                )
                            )
                            * 100
                        )
                    else:
                        backtest_mape = float("nan")

                    self.backtest_metrics_ = {
                        "mae": backtest_mae,
                        "rmse": backtest_rmse,
                        "r2": backtest_r2_full,
                        "mape": backtest_mape,
                        "n_samples": n_valid_samples,
                    }
                    self.logger.info(
                        f"Backtest metrics — MAE: {backtest_mae:.4f}, "
                        f"RMSE: {backtest_rmse:.4f}, "
                        f"R2: {backtest_r2_full:.4f}, "
                        f"MAPE: {backtest_mape:.2f}%"
                    )

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
        weather_future: pd.DataFrame | None = None,
    ) -> pd.Series:
        """The predict method to generate forecasts from a previously fitted ML model.

        :param data_last_window: The data that will be used to generate the new forecast, this \
            will be freshly retrieved from Home Assistant. This data is needed because the forecast \
            model is an auto-regressive model with lags. If not passed then the data used during the \
            model train is used, defaults to None
        :type data_last_window: Optional[pd.DataFrame], optional
        :param weather_future: When the model was trained with ``weather_features``, this DataFrame \
            must provide the future values of those columns over the forecast horizon (the weather \
            forecast). It is ignored when no ``weather_features`` are configured. Defaults to None.
        :type weather_future: Optional[pd.DataFrame], optional
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
                steps = self.lags_opt if self.is_tuned else self.num_lags
                # getattr fallback keeps models pickled before weather_features existed working.
                weather_features = getattr(self, "weather_features", [])
                exog = await self.generate_exog(
                    data_last_window,
                    steps,
                    self.var_model,
                    weather_features=weather_features,
                    weather_future=weather_future,
                )

                predictions = await asyncio.to_thread(
                    self.forecaster.predict,
                    steps=steps,
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
            # Base SVR parameters
            search = {
                "C": trial.suggest_float("C", 0.1, 1.0)
                if debug
                else trial.suggest_float("C", 1e-2, 100.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                "gamma": trial.suggest_categorical(
                    "gamma", ["scale", "auto", 0.01, 0.1, 1.0, 10.0]
                ),
                "lags": get_lags(trial),
            }
            return search

        # Registry of search space generators
        search_spaces = {
            "LinearRegression": lambda trial: {
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True] if debug else [True, False]
                ),
                "lags": get_lags(trial),
            },
            "RidgeRegression": lambda trial: {
                "alpha": trial.suggest_float("alpha", 0.1, 1.0)
                if debug
                else trial.suggest_float("alpha", 1e-4, 100.0, log=True),
                "lags": get_lags(trial),
            },
            "LassoRegression": lambda trial: {
                "alpha": trial.suggest_float("alpha", 0.1, 1.0)
                if debug
                else trial.suggest_float("alpha", 1e-4, 100.0, log=True),
                "lags": get_lags(trial),
            },
            "ElasticNet": lambda trial: {
                "alpha": trial.suggest_float("alpha", 0.0, 2.0),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "selection": trial.suggest_categorical(
                    "selection", ["random"] if debug else ["cyclic", "random"]
                ),
                "lags": get_lags(trial),
            },
            "KNeighborsRegressor": lambda trial: {
                "n_neighbors": trial.suggest_int("n_neighbors", 2, 2)
                if debug
                else trial.suggest_int("n_neighbors", 2, 20),
                "leaf_size": trial.suggest_int("leaf_size", 20, 20)
                if debug
                else trial.suggest_int("leaf_size", 20, 40),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform"] if debug else ["uniform", "distance"]
                ),
                "lags": get_lags(trial),
            },
            "DecisionTreeRegressor": lambda trial: {
                "max_depth": trial.suggest_int("max_depth", 2, 5)
                if debug
                else trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "lags": get_lags(trial),
            },
            "SVR": svr_search_space,
            "RandomForestRegressor": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 20)
                if debug
                else trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "lags": get_lags(trial),
            },
            "ExtraTreesRegressor": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 20)
                if debug
                else trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "lags": get_lags(trial),
            },
            "GradientBoostingRegressor": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 20)
                if debug
                else trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "lags": get_lags(trial),
            },
            "AdaBoostRegressor": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 20)
                if debug
                else trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
                "lags": get_lags(trial),
            },
            "MLPRegressor": lambda trial: {
                "learning_rate_init": trial.suggest_float("learning_rate_init", 0.001, 0.01),
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes", [(50,), (100,), (50, 50)]
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "lags": get_lags(trial),
            },
        }

        if self.sklearn_model not in search_spaces:
            raise ValueError(f"Unsupported model for tuning: {self.sklearn_model}")

        return search_spaces[self.sklearn_model]

    async def tune(
        self,
        split_date_delta: str | None = "48h",
        n_trials: int = 10,
        debug: bool | None = False,
    ) -> pd.DataFrame:
        """Tuning a previously fitted model using bayesian optimization.

        :param split_date_delta: The delta from now to `split_date_delta` that will be used \
            as the test period to evaluate the model, defaults to '48h'.\
            This define the training/validation split for the tuning process.
        :type split_date_delta: Optional[str], optional
        :param debug: Set to True for testing and faster optimizations, defaults to False
        :type debug: Optional[bool], optional
        :param n_trials: Number of trials for bayesian optimization, defaults to 10
        :type n_trials: Optional[int], optional
        :return: The DataFrame with the forecasts using the optimized model.
        :rtype: pd.DataFrame
        """
        try:
            if self.forecaster is None:
                raise ValueError("Model has not been fitted yet. Call fit() first.")

            # Calculate appropriate lags based on data frequency
            freq_timedelta = pd.Timedelta(self.data_exo.index.freq)
            lags_list = MLForecaster.get_lags_list_from_frequency(freq_timedelta)
            self.logger.info(
                f"Using lags list based on data frequency ({self.data_exo.index.freq}): {lags_list}"
            )

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
                MIN_SAMPLES_FOR_KNN = 6
                new_train_size = window_size + MIN_SAMPLES_FOR_KNN
                self.logger.warning(
                    f"Adjusting initial_train_size to {new_train_size} to attempt recovery."
                )
                initial_train_size = new_train_size

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
                n_trials=n_trials,
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

            freq_hours = pd.to_timedelta(self.data_exo.index.freq).total_seconds() / 3600
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
