#!/usr/bin/env python3

"""
On-demand forecast calibration for the load forecast methods.

This module computes an accuracy report that compares the built-in load
forecast methods (``naive``, ``typical`` and ``mlforecaster``) against the
realised load, using only the history retrieved from Home Assistant. It is a
reporting tool with no side effects on any optimization: it lets a user see
which load forecaster tracks their own consumption best.

The realised history is split chronologically into three windows
(``train`` / ``test`` / ``val``). Every method is scored out-of-sample on the
``test`` and ``val`` windows with a day-ahead walk-forward: to predict a given
day a method may only see history strictly before that day, which is what keeps
the numbers honest (no look-ahead). ``mlforecaster`` is additionally scored
in-sample on the ``train`` window; ``naive`` and ``typical`` have no fit step so
their ``train`` cell is reported as ``N/A``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from emhass import utils
from emhass.forecast import Forecast

# The load column name used internally by the walk-forward loop.
_LOAD_COL = "load"

# Hard floor on the amount of history a calibration run needs. The typical
# method averages recurrences of each (month, day-of-week), so it needs many
# more days than the 9-day machine-learning floor to be meaningful.
CALIBRATION_MIN_DAYS = 60
# Default amount of history to retrieve when the caller does not ask for more.
CALIBRATION_DEFAULT_DAYS = 90
# Default size of the two held-out evaluation windows, in days.
CALIBRATION_VAL_DAYS = 14
CALIBRATION_TEST_DAYS = 14

DEFAULT_METHODS = ["naive", "typical", "mlforecaster"]
_NA = "N/A"


def _steps_per_day(freq: pd.Timedelta) -> int:
    """Number of time steps in one day at the given frequency."""
    return int(round(pd.Timedelta("24h") / freq))


def _split_days(day_list: list, test_days: int, val_days: int) -> dict:
    """Split an ordered list of calendar days into train/test/val blocks.

    ``val`` is the most recent block, ``test`` the block immediately before it,
    ``train`` everything older.

    :return: dict with keys ``train``, ``test``, ``val`` mapping to day lists.
    """
    val = day_list[-val_days:]
    test = day_list[-(val_days + test_days) : -val_days]
    train = day_list[: -(val_days + test_days)]
    return {"train": train, "test": test, "val": val}


def _naive_predict_day(history_before: pd.Series, target_dates: pd.DatetimeIndex) -> pd.Series:
    """Persistence: carry the block immediately before the target day forward.

    Mirrors the production naive rule (``forecast.py`` ``_get_load_forecast_naive``:
    take the last ``horizon`` observations) applied at a past date.
    """
    horizon = len(target_dates)
    if len(history_before) < horizon:
        return pd.Series(np.nan, index=target_dates)
    values = history_before.iloc[-horizon:].to_numpy()
    return pd.Series(values, index=target_dates)


def _typical_predict_day(history_before: pd.Series, target_dates: pd.DatetimeIndex) -> pd.Series:
    """Typical profile for the target day, derived only from prior history.

    Reuses the static ``Forecast.get_typical_load_forecast`` (mean load per
    (month, day-of-week) at each time of day). The caller passes only history
    strictly before the target day, so the static method (which has no temporal
    guard of its own) cannot leak the day it is predicting.
    """
    forecast_date = pd.Timestamp(target_dates[0].date())
    data = history_before.to_frame(name=_LOAD_COL)
    try:
        forecast, used_days = Forecast.get_typical_load_forecast(data, forecast_date)
    except (ValueError, IndexError):
        # No same-(month, day-of-week) history yet -> nothing to predict.
        return pd.Series(np.nan, index=target_dates)
    if used_days is None or len(used_days) == 0:
        return pd.Series(np.nan, index=target_dates)
    series = forecast[_LOAD_COL] if _LOAD_COL in forecast.columns else forecast.iloc[:, 0]
    return series.reindex(target_dates)


def _walk_forward(
    load: pd.Series,
    predict_day,
    days: list,
) -> pd.DataFrame:
    """Run a day-ahead walk-forward over ``days`` for one method.

    For each day the method sees only history with index strictly before the
    day's first timestamp. Returns a frame with aligned ``actual`` and ``pred``
    columns over every scored timestamp (may contain NaN where a method could
    not predict).
    """
    actual_parts = []
    pred_parts = []
    for day in days:
        target_dates = load.index[load.index.normalize() == pd.Timestamp(day)]
        if len(target_dates) == 0:
            continue
        day_start = target_dates[0]
        history_before = load.loc[load.index < day_start]
        if len(history_before) == 0:
            continue
        pred = predict_day(history_before, target_dates)
        actual_parts.append(load.loc[target_dates])
        pred_parts.append(pred.reindex(target_dates))
    if not actual_parts:
        return pd.DataFrame(columns=["actual", "pred"])
    return pd.DataFrame({"actual": pd.concat(actual_parts), "pred": pd.concat(pred_parts)})


async def _build_ml_predict_day(
    train_load: pd.Series,
    freq: pd.Timedelta,
    var_model: str,
    sklearn_model: str,
    emhass_conf: dict,
    logger: logging.Logger,
):
    """Fit an MLForecaster once on the train window and return a predict_day fn.

    The model is fitted fresh in memory here (the user's saved model pickle is
    never read), so calibration works even for a user who has never run
    ``forecast-model-fit``. Returns ``None`` if the fit fails, so the caller can
    mark the whole mlforecaster row ``N/A`` rather than failing the report.
    """
    from emhass.machine_learning_forecaster import MLForecaster

    num_lags = _steps_per_day(freq)
    try:
        mlf = MLForecaster(
            train_load.to_frame(name=var_model),
            "load_calibration",
            var_model,
            sklearn_model,
            num_lags,
            emhass_conf,
            logger,
        )
        await mlf.fit(perform_backtest=False)
    except Exception as exc:  # noqa: BLE001 - degrade the whole method, never 500
        logger.warning(f"Forecast calibration: mlforecaster fit failed ({exc}); skipping method")
        return None

    def ml_predict_day(history_before: pd.Series, target_dates: pd.DatetimeIndex) -> pd.Series:
        horizon = len(target_dates)
        if len(history_before) < num_lags:
            return pd.Series(np.nan, index=target_dates)
        # skforecast only needs the last num_lags observations to seed the
        # recursion; pass a little more (2x) as a safe margin so a short gap
        # near the boundary still leaves enough non-NaN lags.
        last_window = history_before.iloc[-num_lags * 2 :]
        try:
            # The fitted forecaster expects the same date-feature exog columns
            # produced at fit time (utils.add_date_features), over the horizon.
            preds = mlf.forecaster.predict(
                steps=horizon,
                last_window=last_window,
                exog=_ml_exog(target_dates),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"Forecast calibration: mlforecaster predict failed for {target_dates[0]}: {exc}"
            )
            return pd.Series(np.nan, index=target_dates)
        return pd.Series(np.asarray(preds), index=target_dates)

    return ml_predict_day


def _ml_exog(target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the date-feature exog frame the fitted forecaster expects."""
    return utils.add_date_features(pd.DataFrame(index=target_dates))


async def compute_forecast_calibration(
    load: pd.Series,
    freq: pd.Timedelta,
    emhass_conf: dict,
    logger: logging.Logger,
    methods: list[str] | None = None,
    sklearn_model: str = "LinearRegression",
    test_days: int = CALIBRATION_TEST_DAYS,
    val_days: int = CALIBRATION_VAL_DAYS,
    var_model: str = "sensor.power_load_no_var_loads",
) -> dict:
    """Compute the load forecast calibration report from realised history.

    :param load: The realised load history (tz-aware DatetimeIndex at ``freq``).
    :param freq: The time step of the history.
    :param methods: Subset of {naive, typical, mlforecaster}; default all three.
    :param var_model: The load column name (only a fallback for direct callers;
        the endpoint always passes the configured load sensor).
    :return: dict with keys ``table`` (metrics DataFrame), ``plot`` (val-window
        actual + per-method predictions DataFrame), ``val_window`` and
        ``caveats``; or ``{"error": ...}`` when there is not enough history.
    """
    methods = methods or list(DEFAULT_METHODS)
    load = load.dropna().sort_index()
    load = load[~load.index.duplicated(keep="first")]

    day_list = sorted({ts.normalize() for ts in load.index})
    n_days = len(day_list)
    if n_days < CALIBRATION_MIN_DAYS or n_days < (test_days + val_days + 1):
        msg = (
            f"Not enough load history for calibration: got {n_days} days, "
            f"need at least {max(CALIBRATION_MIN_DAYS, test_days + val_days + 1)}."
        )
        logger.error(msg)
        return {"error": msg}

    splits = _split_days(day_list, test_days, val_days)

    # Per-method day-ahead predictors. mlforecaster is fit once on the train window.
    predictors = {}
    if "naive" in methods:
        predictors["naive"] = _naive_predict_day
    if "typical" in methods:
        predictors["typical"] = _typical_predict_day
    if "mlforecaster" in methods:
        train_load = load.loc[load.index.normalize() <= splits["train"][-1]]
        ml_predict_day = await _build_ml_predict_day(
            train_load, freq, var_model, sklearn_model, emhass_conf, logger
        )
        if ml_predict_day is not None:
            predictors["mlforecaster"] = ml_predict_day

    # Walk-forward every method on its eval splits; mlforecaster also on train.
    # Keep the paired (actual, pred) frames so the skill score can be computed on
    # the SAME days a method and the naive baseline both cover (no cross-sample bias).
    paired_by = {}
    for method, predict_day in predictors.items():
        eval_splits = ["train", "test", "val"] if method == "mlforecaster" else ["test", "val"]
        paired_by[method] = {
            split: _walk_forward(load, predict_day, splits[split]) for split in eval_splits
        }

    metrics_rows = {}
    skills = {}
    for method in predictors:
        metrics_rows[method] = {
            split: (
                utils.compute_forecast_metrics(paired["actual"], paired["pred"], logger)
                if not paired.empty
                else None
            )
            for split, paired in paired_by[method].items()
        }
        naive_splits = paired_by.get("naive", {})
        skills[method] = {
            split: _skill_vs_naive(method, paired_by[method].get(split), naive_splits.get(split))
            for split in paired_by[method]
        }

    val_days_idx = load.index[load.index.normalize().isin(splits["val"])]
    plot_preds = {
        method: paired_by[method]["val"]["pred"]
        for method in predictors
        if not paired_by[method].get("val", pd.DataFrame()).empty
    }

    table = _build_table(metrics_rows, skills, methods)
    plot = _build_plot_frame(load, val_days_idx, plot_preds)

    caveats = (
        "Typical is derived from the retrieved history window, not the long-term "
        "typical profile used in production, and it requires prior same-month, "
        "same-weekday history so it may cover fewer days than the other methods on "
        "a short window (compare the n columns). The skill score compares each "
        "method against naive on the days they both cover. The train column is "
        "in-sample for mlforecaster and not applicable (no fit step) for naive and "
        "typical. Mlforecaster is fit fresh with a LinearRegression baseline."
    )
    return {
        "table": table,
        "plot": plot,
        "val_window": (str(splits["val"][0].date()), str(splits["val"][-1].date())),
        "caveats": caveats,
    }


def _skill_vs_naive(method: str, method_paired, naive_paired) -> float | None:
    """Persistence skill score of one method against naive, on shared days only.

    ``1 - method_MAE / naive_MAE`` computed over the timestamps where BOTH the
    method and naive produced a prediction, so a method that covers fewer days
    is never flattered by comparing to naive over a different (larger) sample.
    Returns 0.0 for naive itself, None when it cannot be computed.
    """
    if method == "naive":
        return 0.0
    if method_paired is None or method_paired.empty:
        return None
    if naive_paired is None or naive_paired.empty:
        return None
    joined = method_paired.join(naive_paired[["pred"]], how="inner", rsuffix="_naive")
    mask = joined["pred"].notna() & joined["pred_naive"].notna() & joined["actual"].notna()
    if mask.sum() == 0:
        return None
    actual = joined.loc[mask, "actual"]
    mae_method = float((actual - joined.loc[mask, "pred"]).abs().mean())
    mae_naive = float((actual - joined.loc[mask, "pred_naive"]).abs().mean())
    if mae_naive == 0:
        return None
    return 1 - mae_method / mae_naive


def _build_table(metrics_rows: dict, skills: dict, methods: list[str]) -> pd.DataFrame:
    """Flatten the per-method/per-split metrics into a display table.

    Rows = method; columns = split x {mae, rmse, r2, mape, skill, n}. Cells that
    could not be computed (no fit step, or no coverage) are shown as ``N/A``.
    """
    metric_keys = ["mae", "rmse", "r2", "mape"]
    rows = []
    ordered = [m for m in methods if m in metrics_rows]
    for method in ordered:
        row = {"method": method}
        for split in ["train", "test", "val"]:
            m = metrics_rows[method].get(split)
            if m is None:
                for k in metric_keys:
                    row[f"{split}_{k}"] = _NA
                row[f"{split}_skill"] = _NA
                row[f"{split}_n"] = _NA
                continue
            for k in metric_keys:
                row[f"{split}_{k}"] = round(m[k], 4) if m[k] == m[k] else _NA
            skill = skills.get(method, {}).get(split)
            row[f"{split}_skill"] = round(skill, 4) if skill is not None else _NA
            row[f"{split}_n"] = m["n_samples"]
        rows.append(row)
    return pd.DataFrame(rows)


def _build_plot_frame(load: pd.Series, val_idx: pd.DatetimeIndex, plot_preds: dict) -> pd.DataFrame:
    """Assemble the val-window actual + per-method prediction overlay frame."""
    frame = pd.DataFrame(index=val_idx)
    frame["actual"] = load.reindex(val_idx)
    for method, preds in plot_preds.items():
        frame[method] = preds.reindex(val_idx)
    return frame
