"""A native time-series forecaster for EMHASS built on the Darts library.

This module provides :class:`DartsForecaster`, an alternative to
:class:`emhass.machine_learning_forecaster.MLForecaster` that uses a global
gradient-boosted time-series model (LightGBM, via Darts) instead of the
recursive ``skforecast``/``scikit-learn`` approach. It is target-agnostic and
serves the load, load-cost and production-price forecasts.

Why a second forecaster
-----------------------
``MLForecaster`` is a recursive auto-regressive model: it predicts one step,
feeds that prediction back as a lag, and repeats. On long horizons (a day or
two of 5-minute steps) the recursion tends to flatten towards the mean and the
weather signal is limited to calendar features. ``DartsForecaster`` instead:

* fits a **single-shot** model (``output_chunk_length`` covers the whole
  horizon) so there is no recursive error accumulation;
* consumes **future covariates** (weather + calendar) that are known for the
  forecast window, directly addressing the long-standing request to make the
  ML load forecast weather-aware (GitHub issue #847);
* optionally emits **quantiles** (e.g. P10/P50/P90), giving a cheap measure of
  forecast uncertainty (GitHub issue #841).

Design goals
------------
* **Drop-in interface**: the same ``fit`` / ``predict`` / ``tune`` async method
  shapes as :class:`MLForecaster`, so ``command_line.py`` can dispatch to either
  forecaster behind the existing ``load_forecast_method`` config switch.
* **Generic, zero-config-friendly**: works on the load sensor alone (calendar
  covariates only). Weather covariates are used automatically *if* the caller
  passes them, but are never required.
* **Optional dependency**: Darts and LightGBM are heavyweight, so they are an
  ``emhass[darts]`` extra and imported lazily. EMHASS continues to install and
  run with its default forecaster if the extra is absent; selecting the
  ``darts`` method without the extra raises a clear, actionable error.

The weather covariates, when supplied, reuse EMHASS's existing Open-Meteo
forecast (see :meth:`emhass.forecast.Forecast.get_weather_forecast`) so no new
external API client is introduced in core.
"""

import asyncio
import logging
import warnings

import numpy as np
import pandas as pd

from emhass import utils

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Calendar features EMHASS already knows how to compute (see utils.add_date_features).
# Kept as a module constant so fit and predict stay in lockstep.
CALENDAR_FEATURES = ["month", "day_of_week", "day_of_year", "hour"]

# Default LightGBM hyper-parameters. Deliberately modest so a fit stays fast and
# low-memory on constrained hardware (e.g. a Raspberry Pi add-on). All are
# overridable via the ``model_kwargs`` argument.
DEFAULT_LGBM_KWARGS = {
    "n_estimators": 300,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbosity": -1,
}

# Default auto-regressive lags expressed in *steps*. These are scaled from a set
# of physical horizons (recent history + daily + weekly periodicity) to the
# actual data frequency in :meth:`DartsForecaster._default_lags`, so the same
# defaults are sensible whether the data is 30-minute or 5-minute.
_DEFAULT_LAG_HORIZONS_HOURS = [
    1 / 12,  # one step at 5-min data, clamped to >=1 step otherwise
    1 / 6,
    0.25,
    0.5,
    1,
    2,
    24,  # previous day, same time
    48,  # two days back
    168,  # previous week, same time
]

# A short window of *recent* covariate lags lets the model react to the latest
# weather/calendar without exploding the feature count on a long single-shot
# horizon. Expressed in steps relative to each output position.
_DEFAULT_FUTURE_COV_LAG_STEPS = [-12, -6, -3, -1, 0]

# Defensive upper bound for a household electrical load sample, in watts. Used
# only by the sanity check to flag a grossly broken forecast; never clips the
# returned values silently beyond the non-negativity floor.
DEFAULT_MAX_PLAUSIBLE_W = 100000.0


class DartsDependencyError(ImportError):
    """Raised when the optional ``darts``/``lightgbm`` extra is not installed."""


def _require_darts():
    """Import Darts lazily and raise a clear, actionable error if it is missing."""
    try:
        from darts import TimeSeries  # noqa: F401
        from darts.models import LightGBMModel  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via the dependency test
        raise DartsDependencyError(
            "The 'darts' load forecast method requires the optional Darts and "
            "LightGBM dependencies. Install them with:\n"
            "    pip install emhass[darts]\n"
            "or add 'darts' and 'lightgbm' to your environment."
        ) from exc


class DartsForecaster:
    r"""A single-shot, weather-aware time-series forecaster.

    The class mirrors the public surface of
    :class:`emhass.machine_learning_forecaster.MLForecaster`:

    - :meth:`fit` trains the model on historical data.
    - :meth:`predict` produces a forecast over the optimization horizon.
    - :meth:`tune` runs a light hyper-parameter search.

    Unlike ``MLForecaster``, it can consume future covariates (weather +
    calendar) that are known for the forecast window. Weather is optional: if
    the caller does not pass a ``future_covariates`` frame, only calendar
    features are used and the forecaster still works on the target sensor alone.

    The forecaster is **target-agnostic**: ``var_model`` names whatever series is
    being modelled, so the same class serves the load forecast *and* the load
    cost / production price forecasts (see the ``darts`` option on
    :meth:`emhass.forecast.Forecast.get_load_cost_forecast` and
    :meth:`~emhass.forecast.Forecast.get_prod_price_forecast`). Use
    ``non_negative=False`` for a price target, which can legitimately go
    negative.

    :param data: The training data. Must contain a column named ``var_model``
        (the target sensor) and a :class:`~pandas.DatetimeIndex`. May optionally
        contain extra columns to be used as future covariates (named via
        ``covariate_columns``).
    :type data: pd.DataFrame
    :param model_type: A unique name identifying this model (used for the saved
        pickle filename), mirroring ``MLForecaster``.
    :type model_type: str
    :param var_model: The name of the target column / Home Assistant sensor,
        e.g. ``sensor.power_load_no_var_loads`` for load or a price sensor for
        the cost / production-price forecasts.
    :type var_model: str
    :param num_lags: The longest auto-regressive lag to consider, in steps. Used
        only as a ceiling for the default lag set; the actual lags are derived in
        :meth:`_default_lags`. Kept for interface parity with ``MLForecaster``.
    :type num_lags: int
    :param emhass_conf: Dictionary of EMHASS paths.
    :type emhass_conf: dict
    :param logger: The logger object.
    :type logger: logging.Logger
    :param covariate_columns: Optional list of column names in ``data`` to use as
        future covariates (e.g. ``["temperature_2m", "relative_humidity_2m"]``).
        Calendar features are always added on top of these. Defaults to ``None``
        (calendar-only).
    :type covariate_columns: list[str] | None
    :param quantiles: Optional list of quantile levels for a probabilistic
        forecast, e.g. ``[0.1, 0.5, 0.9]``. When ``None`` a single deterministic
        forecast is produced. The P50 (or the deterministic point) is what
        :meth:`predict` returns; other quantiles are exposed on
        :attr:`last_quantiles` for callers that want uncertainty bands.
    :type quantiles: list[float] | None
    :param model_kwargs: Extra keyword arguments forwarded to the underlying
        ``darts.models.LightGBMModel`` (merged over :data:`DEFAULT_LGBM_KWARGS`).
    :type model_kwargs: dict | None
    :param non_negative: Whether the target is physically non-negative and the
        forecast should be floored at zero. ``True`` (the default) suits a load
        in watts; set it to ``False`` for a target that can legitimately go
        negative, e.g. a spot energy price (negative prices occur on grids such
        as Amber). Defaults to ``True``.
    :type non_negative: bool
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_type: str,
        var_model: str,
        num_lags: int,
        emhass_conf: dict,
        logger: logging.Logger,
        covariate_columns: list[str] | None = None,
        quantiles: list[float] | None = None,
        output_chunk_length: int | None = None,
        model_kwargs: dict | None = None,
        non_negative: bool = True,
    ) -> None:
        self.data = data
        self.model_type = model_type
        self.var_model = var_model
        self.num_lags = num_lags
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.covariate_columns = list(covariate_columns) if covariate_columns else []
        self.quantiles = list(quantiles) if quantiles else None
        self.model_kwargs = {**DEFAULT_LGBM_KWARGS, **(model_kwargs or {})}
        self.non_negative = non_negative

        self.model = None
        self.freq: pd.Timedelta | None = None
        # The single-shot forecast length. Defaults to a 48h horizon scaled to
        # the data frequency, which covers the typical EMHASS day-ahead window;
        # callers may override. Must be >= the optimization horizon at predict.
        self._requested_output_chunk_length = output_chunk_length
        self.output_chunk_length: int | None = None
        self.last_quantiles: pd.DataFrame | None = None
        self.is_tuned = False

        self._prepare_data()
        if self.output_chunk_length is None:
            self.output_chunk_length = self._default_output_chunk_length()

    # ------------------------------------------------------------------ helpers

    def _prepare_data(self) -> None:
        """Clean and sort the input data, and infer the sampling frequency."""
        self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.sort_index()
        self.data = self.data[~self.data.index.duplicated(keep="first")]
        inferred = pd.infer_freq(self.data.index)
        if inferred is not None:
            # pd.Timedelta(offset) works across pandas versions; the offset
            # `.delta` attribute was removed in pandas 2.2.
            self.freq = pd.Timedelta(pd.tseries.frequencies.to_offset(inferred))
        elif len(self.data.index) >= 2:
            self.freq = self.data.index.to_series().diff().median()
        else:
            self.freq = pd.Timedelta("30min")

    def _default_lags(self) -> list[int]:
        """Derive a sensible auto-regressive lag set scaled to the data frequency.

        Mirrors the spirit of ``MLForecaster.get_lags_list_from_frequency`` but
        adds the daily/weekly periodicity lags that make a single-shot model
        strong on long horizons. Returned as the **negative** step offsets Darts
        expects (e.g. ``[-1, -2, ...]``), clamped so the magnitude never exceeds
        ``num_lags`` (which therefore acts as a meaningful ceiling).
        """
        freq_hours = self.freq.total_seconds() / 3600
        magnitudes = {max(1, int(round(h / freq_hours))) for h in _DEFAULT_LAG_HORIZONS_HOURS}
        ceiling = max(1, int(self.num_lags))
        magnitudes = {min(mag, ceiling) for mag in magnitudes}
        return [-mag for mag in sorted(magnitudes)]

    def _default_output_chunk_length(self) -> int:
        """Default single-shot horizon: 48h scaled to the data frequency."""
        if self._requested_output_chunk_length:
            return int(self._requested_output_chunk_length)
        return max(1, int(round(pd.Timedelta("48h") / self.freq)))

    @staticmethod
    def _strip_tz(frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with a tz-naive index (Darts does not accept tz-aware)."""
        if frame.index.tz is not None:
            out = frame.copy()
            out.index = out.index.tz_localize(None)
            return out
        return frame

    def _build_calendar_covariates(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Calendar covariates over ``index`` using EMHASS's own feature helper."""
        frame = pd.DataFrame(index=index)
        frame = utils.add_date_features(frame, date_features=CALENDAR_FEATURES)
        return frame.astype("float32")

    def _build_future_covariates(self, source: pd.DataFrame) -> "object":
        """Assemble a Darts future-covariates TimeSeries spanning the horizon.

        Darts requires future covariates to extend ``output_chunk_length`` steps
        past the end of the target series. Calendar features are deterministic so
        they are generated over the extended index automatically. Weather columns
        present in ``source`` are reindexed onto the extended index; any portion
        of the horizon the caller did not supply weather for is forward-filled
        from the last known value ("old weather beats no weather"), so a
        calendar-only caller works out of the box and a weather-aware caller gets
        the real signal wherever it provided one.
        """
        from darts import TimeSeries

        # Extend the index by the single-shot horizon so covariates cover it.
        horizon = self.output_chunk_length or self._default_output_chunk_length()
        extended_index = pd.date_range(
            source.index[0],
            source.index[-1] + horizon * self.freq,
            freq=self.freq_str,
        )
        calendar = self._build_calendar_covariates(extended_index)

        cols = [c for c in self.covariate_columns if c in source.columns]
        if cols:
            weather = source[cols].astype("float32").reindex(extended_index)
            combined = weather.join(calendar)
        else:
            combined = calendar
        combined = combined.ffill().bfill()
        return TimeSeries.from_dataframe(combined, freq=self.freq_str)

    @property
    def freq_str(self) -> str:
        """The pandas frequency alias for the inferred sampling step."""
        return pd.tseries.frequencies.to_offset(self.freq).freqstr

    def _make_model(self, output_chunk_length: int):
        """Instantiate the underlying Darts ``LightGBMModel``."""
        from darts.models import LightGBMModel

        lags = self._default_lags()
        kwargs = dict(
            lags=lags,
            output_chunk_length=output_chunk_length,
            multi_models=False,
            # Calendar covariates are always present, so future-covariate lags
            # are always meaningful.
            lags_future_covariates=_DEFAULT_FUTURE_COV_LAG_STEPS,
            **self.model_kwargs,
        )
        if self.quantiles:
            kwargs["likelihood"] = "quantile"
            kwargs["quantiles"] = self.quantiles
        self.logger.info(
            f"Building DartsForecaster LightGBMModel: lags={lags}, "
            f"output_chunk_length={output_chunk_length}, "
            f"quantiles={self.quantiles or 'deterministic'}"
        )
        return LightGBMModel(**kwargs)

    @staticmethod
    def _quantile_column(pred_df: pd.DataFrame, level: float) -> pd.Series:
        """Pick the predicted-quantile column for ``level`` from a Darts frame."""
        import re

        for col in pred_df.columns:
            match = re.search(r"q([0-9.]+)$", str(col).lower())
            if match and abs(float(match.group(1)) - level) < 1e-6:
                return pred_df[col]
        # Single-column deterministic frame: return the only column.
        if pred_df.shape[1] == 1:
            return pred_df.iloc[:, 0]
        raise KeyError(f"quantile {level} not found among {list(pred_df.columns)}")

    def _build_target_series(self, frame: pd.DataFrame):
        """Build a Darts target ``TimeSeries`` from the ``var_model`` column."""
        from darts import TimeSeries

        return TimeSeries.from_series(frame[self.var_model].astype("float32"), freq=self.freq_str)

    def _prepare_target_and_covariates(self, frame: pd.DataFrame):
        """Build the Darts target + future-covariates pair from a tz-aware frame.

        Centralises the window preparation that :meth:`fit` and :meth:`predict`
        otherwise duplicated: tz stripping (Darts requires a tz-naive index),
        target ``TimeSeries`` construction and the future-covariate assembly,
        all over the same frame. Linear interpolation is applied by the caller.

        :param frame: A prepared frame containing at least the ``var_model``
            column (and any covariate columns).
        :return: ``(target, future_cov, original_tz)`` where ``target`` and
            ``future_cov`` are Darts ``TimeSeries`` and ``original_tz`` is the
            timezone stripped off (or ``None``), so callers can re-attach it.
        """
        original_tz = frame.index.tz
        naive = self._strip_tz(frame)  # Darts requires a tz-naive index
        target = self._build_target_series(naive)
        future_cov = self._build_future_covariates(naive)
        return target, future_cov, original_tz

    # --------------------------------------------------------------------- fit

    async def fit(
        self,
        split_date_delta: str | None = "48h",
        perform_backtest: bool | None = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        r"""Train the Darts model.

        :param split_date_delta: Test window length, kept for interface parity
            with :class:`MLForecaster`. Used to carve a hold-out tail for the
            reported in-sample metric.
        :type split_date_delta: str, optional
        :param perform_backtest: Reserved for interface parity. A historical
            backtest is not run here (Darts ``historical_forecasts`` is
            expensive); the hold-out metric is reported instead. Defaults to
            ``False``.
        :type perform_backtest: bool, optional
        :return: ``(df_pred, df_pred_backtest)`` where ``df_pred`` carries the
            train/test/pred columns over the hold-out window, mirroring
            ``MLForecaster.fit``. ``df_pred_backtest`` is ``None``.
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        _require_darts()

        if self.var_model not in self.data.columns:
            raise KeyError(
                f"Variable '{self.var_model}' not found in data columns: {list(self.data.columns)}"
            )

        self.logger.info("Performing a Darts forecast model fit for " + self.model_type)

        interpolated = await asyncio.to_thread(
            lambda: self.data.interpolate(method="linear", axis=0, limit_direction="both")
        )
        prepared = self._strip_tz(interpolated)  # Darts requires a tz-naive index
        original_tz = interpolated.index.tz

        # Cap the single-shot horizon to what the data can support.
        self.output_chunk_length = min(
            self._default_output_chunk_length(), max(1, len(prepared) // 2)
        )

        # Hold-out tail for the reported metric (parity with MLForecaster.fit).
        split_date = prepared.index[-1] - pd.Timedelta(split_date_delta) + self.freq
        train_df = prepared.loc[: split_date - self.freq]
        test_df = prepared.loc[split_date:]

        # Fit on the train portion; the single-shot model emits the whole horizon.
        # The future covariates are built over the full prepared frame so they
        # span the hold-out tail used for the reported metric below.
        target = self._build_target_series(train_df)
        future_cov = self._build_future_covariates(prepared)

        self.model = self._make_model(self.output_chunk_length)
        self.logger.info(f"Training a LightGBM (Darts) model on {len(target)} steps")
        await asyncio.to_thread(self.model.fit, target, future_covariates=future_cov)

        # Report a one-shot hold-out metric (best-effort; never fails the fit).
        df_pred = pd.DataFrame(index=interpolated.index, columns=["train", "test", "pred"])
        df_pred["train"] = self._reindex_like(train_df[self.var_model], original_tz)
        df_pred["test"] = self._reindex_like(test_df[self.var_model], original_tz)
        try:
            from sklearn.metrics import r2_score

            steps = min(self.output_chunk_length, len(test_df))
            if steps >= 1:
                pred = await asyncio.to_thread(
                    self.model.predict,
                    n=steps,
                    series=target,
                    future_covariates=future_cov,
                    predict_likelihood_parameters=bool(self.quantiles),
                )
                pred_df = pred.to_dataframe()
                point = (
                    self._quantile_column(pred_df, 0.5) if self.quantiles else pred_df.iloc[:, 0]
                )
                point = self._reindex_like(point, original_tz)
                common = df_pred.index.intersection(point.index)
                if len(common):
                    df_pred.loc[common, "pred"] = point.reindex(common)
                scored = df_pred.dropna(subset=["test", "pred"])
                if len(scored):
                    metric = r2_score(scored["test"], scored["pred"])
                    self.logger.info(f"Prediction R2 score of fitted model on test data: {metric}")
        except Exception as exc:  # noqa: BLE001 - metric is informational only
            self.logger.warning(f"Could not compute hold-out metric: {exc}")

        return df_pred, None

    def _reindex_like(self, series: pd.Series, tz) -> pd.Series:
        """Re-attach the original timezone to a (tz-naive) Darts-produced series."""
        out = series.copy()
        if tz is not None and out.index.tz is None:
            out.index = out.index.tz_localize(tz)
        return out

    # ----------------------------------------------------------------- predict

    async def predict(
        self,
        data_last_window: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Generate a forecast from the fitted model.

        :param data_last_window: Fresh recent data used to anchor the
            auto-regressive lags (typically just retrieved from Home Assistant).
            Must contain the ``var_model`` column and, if weather covariates were
            used at fit time, the same covariate columns spanning the forecast
            window. If ``None``, the model forecasts from the end of its training
            series.
        :type data_last_window: pd.DataFrame, optional
        :return: The point forecast (P50 if probabilistic) as a
            :class:`~pandas.Series` indexed by the forecast timestamps.
        :rtype: pd.Series
        """
        _require_darts()

        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        n = self.output_chunk_length
        original_tz = None
        if data_last_window is None:
            pred = await asyncio.to_thread(
                self.model.predict,
                n=n,
                predict_likelihood_parameters=bool(self.quantiles),
            )
        else:
            window = await asyncio.to_thread(
                lambda: data_last_window.interpolate(
                    method="linear", axis=0, limit_direction="both"
                )
            )
            target, future_cov, original_tz = self._prepare_target_and_covariates(window)
            pred = await asyncio.to_thread(
                self.model.predict,
                n=n,
                series=target,
                future_covariates=future_cov,
                predict_likelihood_parameters=bool(self.quantiles),
            )

        pred_df = pred.to_dataframe()
        if original_tz is not None and pred_df.index.tz is None:
            pred_df.index = pred_df.index.tz_localize(original_tz)
        if self.quantiles:
            self.last_quantiles = pred_df.copy()
            point = self._quantile_column(pred_df, 0.5)
        else:
            self.last_quantiles = None
            point = pred_df.iloc[:, 0]
        # Floor at zero only for physically non-negative targets (e.g. a load in
        # watts). Targets that can legitimately be negative — such as a spot
        # energy price — pass through unclipped.
        if self.non_negative:
            point = point.clip(lower=0.0)
        return point

    # -------------------------------------------------------------------- tune

    async def tune(
        self,
        split_date_delta: str | None = "48h",
        n_trials: int = 10,
        debug: bool | None = False,
    ) -> pd.DataFrame:
        """Light hyper-parameter search over the LightGBM learning rate / depth.

        A deliberately small grid (kept cheap so it runs on constrained
        hardware) over the most impactful LightGBM knobs. The best configuration
        is stored on the instance and the model is refitted with it.

        :param split_date_delta: Hold-out window for scoring, parity with
            ``MLForecaster``.
        :type split_date_delta: str, optional
        :param n_trials: Maximum number of grid points to evaluate.
        :type n_trials: int
        :param debug: When ``True`` evaluate a single trivial configuration for
            fast unit testing.
        :type debug: bool, optional
        :return: A DataFrame of the train/test/optimized-prediction columns.
        :rtype: pd.DataFrame
        """
        _require_darts()
        from sklearn.metrics import r2_score

        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        grid = (
            [{"learning_rate": 0.05, "num_leaves": 31}]
            if debug
            else [
                {"learning_rate": lr, "num_leaves": nl}
                for lr in (0.03, 0.05, 0.1)
                for nl in (31, 63)
            ][:n_trials]
        )

        best_score, best_kwargs = -np.inf, None
        for trial in grid:
            candidate = {**self.model_kwargs, **trial}
            saved = self.model_kwargs
            self.model_kwargs = candidate
            try:
                df_pred, _ = await self.fit(split_date_delta=split_date_delta)
                scored = df_pred.dropna(subset=["test", "pred"])
                score = r2_score(scored["test"], scored["pred"]) if len(scored) else -np.inf
            finally:
                self.model_kwargs = saved
            self.logger.info(f"Darts tune trial {trial} -> R2={score:.4f}")
            if score > best_score:
                best_score, best_kwargs = score, candidate

        if best_kwargs is not None:
            self.model_kwargs = best_kwargs
            self.is_tuned = True
            self.logger.info(f"Best Darts hyper-parameters: {best_kwargs} (R2={best_score:.4f})")
        df_pred_opt, _ = await self.fit(split_date_delta=split_date_delta)
        return df_pred_opt.rename(columns={"pred": "pred_optim"})

    # ------------------------------------------------------------- sanity check

    def sanity_check(
        self,
        forecast: pd.Series,
        history: pd.Series,
        max_plausible_w: float = DEFAULT_MAX_PLAUSIBLE_W,
    ) -> dict:
        r"""Cheap, defensive validation of a produced forecast.

        Returns a verdict dict; callers decide whether to fall back to another
        method. This is a *producer-side* gate that flags a subtly-broken
        forecast — it does not mutate the forecast.

        Two robustness lessons are baked in:

        * **History-only reference.** The reference daily level is computed from
          *recorded history only*, never from any horizon region that may have
          been forward-filled with the last observed value for the lag warm-up
          (which would make the reference track a single instantaneous reading).
        * **Robust statistic.** The reference uses the **median** of the recent
          window, not the mean, so a short load spike (or denser-cadence
          historical regions) cannot skew it.

        :param forecast: The produced forecast series.
        :param history: The recorded load history (real observations only).
        :param max_plausible_w: Defensive per-sample upper bound in watts.
        :return: ``{"ok": bool, "detail": str, ...}``.
        """
        problems = []
        values = forecast.to_numpy(dtype="float64")
        # Peak over real (finite) samples only. Computed once and guarded so an
        # empty or all-NaN forecast cannot raise from np.nanmax (which errors on
        # a zero-size / all-NaN input) when building the messages below.
        finite_values = values[np.isfinite(values)]
        peak = float(np.max(finite_values)) if finite_values.size else float("nan")
        if not np.all(np.isfinite(values)):
            problems.append("non-finite values")
        if np.any(values < 0):
            problems.append("negative values")
        if finite_values.size and np.any(finite_values > max_plausible_w):
            problems.append(f"value > {max_plausible_w:.0f}W (peak {peak:.0f})")

        steps_per_day = max(1, int(round(pd.Timedelta("24h") / self.freq)))
        clean_history = history.dropna()
        recent_level = (
            float(clean_history.tail(steps_per_day).median())
            if len(clean_history)
            else float("nan")
        )
        pred_mean = float(np.nanmean(finite_values)) if finite_values.size else float("nan")
        detail = (
            f"n={len(values)} mean={pred_mean:.0f}W peak={peak:.0f}W "
            f"recent_day_median={recent_level:.0f}W"
        )
        if np.isfinite(recent_level) and recent_level > 50:
            ratio = pred_mean / recent_level
            detail += f" ratio={ratio:.2f}"
            if ratio < 0.3 or ratio > 3.5:
                problems.append(
                    f"mean ratio {ratio:.2f} vs recent daily median (grossly diverging)"
                )

        ok = not problems
        if problems:
            detail += " | PROBLEMS: " + "; ".join(problems)
        return {
            "ok": ok,
            "detail": detail,
            "pred_mean": pred_mean,
            "recent_level": recent_level,
        }
