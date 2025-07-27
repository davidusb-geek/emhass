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
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from emhass import utils_async

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MLForecaster:
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
        self._prepare_data()

    def _prepare_data(self):
        self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace=True)
        self.data = self.data[~self.data.index.duplicated(keep="first")]

    @staticmethod
    def neg_r2_score(y_true, y_pred):
        return -r2_score(y_true, y_pred)

    @staticmethod
    async def interpolate_async(data: pd.DataFrame) -> pd.DataFrame:
        return await asyncio.to_thread(data.interpolate, method="linear", axis=0, limit=None)

    @staticmethod
    async def generate_exog(data_last_window, periods, var_name):
        forecast_dates = pd.date_range(
            start=data_last_window.index[-1] + data_last_window.index.freq,
            periods=periods,
            freq=data_last_window.index.freq,
        )
        exog = pd.DataFrame({var_name: [np.nan] * periods}, index=forecast_dates)
        exog = utils_async.add_date_features(exog)
        return exog

    @staticmethod
    def get_sklearn_model(name: str):
        models = {
            "linearregression": LinearRegression,
            "elasticnet": ElasticNet,
            "kneighborsregressor": KNeighborsRegressor,
        }
        name = name.lower()
        if name not in models:
            raise ValueError(f"Unknown sklearn model: {name}")
        return models[name]()

    async def fit(self, split_date_delta: str = "48h", perform_backtest: bool = False):
        self.logger.info(f"Fitting model {self.model_type} ({self.sklearn_model})")
        self.data_exo = utils_async.add_date_features(pd.DataFrame(index=self.data.index))

        if self.var_model not in self.data.columns:
            raise KeyError(f"Variable '{self.var_model}' not found in data columns: {list(self.data.columns)}")

        self.data_exo[self.var_model] = self.data[self.var_model]
        self.data_exo = await self.interpolate_async(self.data_exo)

        self.date_train = self.data_exo.index[-1] - pd.Timedelta("5days") + self.data_exo.index.freq
        self.date_split = self.data_exo.index[-1] - pd.Timedelta(split_date_delta) + self.data_exo.index.freq

        self.data_train = self.data_exo.loc[: self.date_split - self.data_exo.index.freq, :]
        self.data_test = self.data_exo.loc[self.date_split :, :]
        self.steps = len(self.data_test)

        base_model = self.get_sklearn_model(self.sklearn_model)
        self.forecaster = ForecasterRecursive(regressor=base_model, lags=self.num_lags)
        self.logger.info("Training a " + self.sklearn_model + " model")
        start_time = time.time()

        await asyncio.to_thread(
            self.forecaster.fit,
            y=self.data_train[self.var_model],
            exog=self.data_train.drop(self.var_model, axis=1)
        )

        self.logger.info(f"Elapsed time for model fit: {time.time() - start_time}")

        predictions = await asyncio.to_thread(
            self.forecaster.predict,
            steps=self.steps,
            exog=self.data_test.drop(self.var_model, axis=1)
        )

        score = await asyncio.to_thread(r2_score, self.data_test[self.var_model], predictions)
        self.logger.info(f"Prediction R2 score on test data: {score}")

        df_pred = pd.DataFrame(index=self.data_exo.index, columns=["train", "test", "pred"])
        df_pred["train"] = self.data_train[self.var_model]
        df_pred["test"] = self.data_test[self.var_model]
        df_pred["pred"] = predictions

        df_pred_backtest = None

        if perform_backtest:
            self.logger.info("Performing simple backtesting of fitted model")
            start_time = time.time()
            cv = TimeSeriesFold(steps=self.num_lags, allow_incomplete_fold=True, refit=False)
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
            self.logger.info(f"Elapsed backtesting time: {time.time() - start_time}")
            self.logger.info(f"Backtest R2 score: {-metric}")
            df_pred_backtest = pd.DataFrame(index=self.data_exo.index, columns=["train", "pred"])
            df_pred_backtest["train"] = self.data_exo[self.var_model]
            df_pred_backtest["pred"] = predictions_backtest

        return df_pred, df_pred_backtest

    async def predict(self, data_last_window: pd.DataFrame | None = None) -> pd.Series:
        if data_last_window is None:
            return await asyncio.to_thread(
                self.forecaster.predict,
                steps=self.num_lags,
                exog=self.data_test.drop(self.var_model, axis=1)
            )

        data_last_window = await self.interpolate_async(data_last_window)
        lags = self.lags_opt if self.is_tuned else self.num_lags
        exog = await self.generate_exog(data_last_window, lags, self.var_model)

        return await asyncio.to_thread(
            self.forecaster.predict,
            steps=lags,
            last_window=data_last_window[self.var_model],
            exog=exog.drop(self.var_model, axis=1),
        )

    async def tune(self, debug: bool = False) -> pd.DataFrame:
        self.logger.info("Starting Bayesian optimization")
        model = self.sklearn_model.lower()

        def get_search_space(debug):
            if model == "linearregression":
                return lambda trial: {
                    "fit_intercept": trial.suggest_categorical("fit_intercept", [True] if debug else [True, False]),
                    "lags": trial.suggest_categorical("lags", [3] if debug else [6, 12, 24, 36, 48, 60, 72]),
                }
            elif model == "elasticnet":
                return lambda trial: {
                    "alpha": trial.suggest_float("alpha", 0.0, 2.0) if not debug else 1.0,
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0) if not debug else 0.5,
                    "selection": trial.suggest_categorical("selection", ["random"] if debug else ["cyclic", "random"]),
                    "lags": trial.suggest_categorical("lags", [3] if debug else [6, 12, 24, 36, 48, 60, 72]),
                }
            elif model == "kneighborsregressor":
                return lambda trial: {
                    "n_neighbors": trial.suggest_int("n_neighbors", 2, 20) if not debug else 5,
                    "leaf_size": trial.suggest_int("leaf_size", 20, 40) if not debug else 30,
                    "weights": trial.suggest_categorical("weights", ["uniform"] if debug else ["uniform", "distance"]),
                    "lags": trial.suggest_categorical("lags", [3] if debug else [6, 12, 24, 36, 48, 60, 72]),
                }
            else:
                raise ValueError(f"Unsupported model for tuning: {model}")

        refit = not debug
        num_lags = 3 if debug else self.num_lags

        cv = TimeSeriesFold(
            steps=num_lags,
            initial_train_size=len(self.data_exo.loc[: self.date_train]),
            fixed_train_size=True,
            allow_incomplete_fold=True,
            refit=refit,
        )

        self.optimize_results, _ = await asyncio.to_thread(
            bayesian_search_forecaster,
            forecaster=self.forecaster,
            y=self.data_train[self.var_model],
            exog=self.data_train.drop(self.var_model, axis=1),
            cv=cv,
            search_space=get_search_space(debug),
            metric=MLForecaster.neg_r2_score,
            n_trials=10,
            random_state=123,
            return_best=True,
        )

        self.is_tuned = True
        self.lags_opt = int(np.round(len(self.optimize_results.iloc[0]["lags"])))
        freq_hours = self.data_exo.index.freq.delta.seconds / 3600
        self.days_needed = int(np.round(self.lags_opt * freq_hours / 24))

        predictions_opt = await asyncio.to_thread(
            self.forecaster.predict,
            steps=self.num_lags,
            exog=self.data_test.drop(self.var_model, axis=1)
        )

        df_pred_opt = pd.DataFrame(index=self.data_exo.index, columns=["train", "test", "pred_optim"])
        df_pred_opt["train"] = self.data_train[self.var_model]
        df_pred_opt["test"] = self.data_test[self.var_model]
        df_pred_opt["pred_optim"] = predictions_opt

        r2_train = -self.optimize_results.iloc[0]["neg_r2_score"]
        r2_test = await asyncio.to_thread(
            r2_score,
            df_pred_opt.loc[predictions_opt.index, "test"],
            df_pred_opt.loc[predictions_opt.index, "pred_optim"],
        )

        self.logger.info(f"R2 train (optimized): {r2_train}")
        self.logger.info(f"R2 test (optimized): {r2_test}")
        self.logger.info(f"Optimal lags: {self.lags_opt}")

        return df_pred_opt
