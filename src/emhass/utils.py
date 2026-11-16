from __future__ import annotations

import ast
import asyncio
import copy
import csv
import html
import itertools
import logging
import math
import numbers
import os
import pathlib
import re
import shutil
import stat
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

import aiofiles
import aiohttp
import numpy as np
import orjson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import yaml
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from emhass.coordinator.site import compile_site, site_problem, site_token_secret_usable

if TYPE_CHECKING:
    from emhass.machine_learning_forecaster import MLForecaster

pd.options.plotting.backend = "plotly"

# Unit conversion constants
W_TO_KW = 1000  # Watts to kilowatts conversion factor


def get_root(file: str, num_parent: int = 3) -> str:
    """
    Get the root absolute path of the working directory.

    :param file: The passed file path with __file__
    :return: The root path
    :param num_parent: The number of parents levels up to desired root folder
    :type num_parent: int, optional
    :rtype: str

    """
    if num_parent == 3:
        root = pathlib.Path(file).resolve().parent.parent.parent
    elif num_parent == 2:
        root = pathlib.Path(file).resolve().parent.parent
    elif num_parent == 1:
        root = pathlib.Path(file).resolve().parent
    else:
        raise ValueError("num_parent value not valid, must be between 1 and 3")
    return root


def get_logger(
    fun_name: str,
    emhass_conf: dict[str, pathlib.Path],
    save_to_file: bool = True,
    logging_level: str = "DEBUG",
) -> tuple[logging.Logger, logging.StreamHandler]:
    """
    Create a simple logger object.

    :param fun_name: The Python function object name where the logger will be used
    :type fun_name: str
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param save_to_file: Write log to a file, defaults to True
    :type save_to_file: bool, optional
    :return: The logger object and the handler
    :rtype: object

    """
    # create logger object
    logger = logging.getLogger(fun_name)
    logger.propagate = True
    logger.fileSetting = save_to_file
    if save_to_file:
        if os.path.isdir(emhass_conf["data_path"]):
            ch = logging.FileHandler(emhass_conf["data_path"] / "logger_emhass.log")
        else:
            raise Exception("Unable to access data_path: " + emhass_conf["data_path"])
    else:
        ch = logging.StreamHandler()
    if logging_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        logger.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        logger.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, ch


def _get_now() -> datetime:
    """Return the current instant as a timezone-aware UTC datetime. Separated out for easier mocking in tests."""
    return datetime.now(UTC)


def add_local_calendar_days(
    timestamp: pd.Timestamp | datetime,
    days: int,
    time_zone: datetime.tzinfo | str | None = None,
) -> pd.Timestamp:
    """Add local calendar days with deterministic pytz DST endpoint handling.

    EMHASS forecast horizons use local civil/calendar days, not fixed 24-hour
    durations. Valid wall times therefore retain the existing 23/24/25-hour
    behavior across DST. If the nominal target wall time does not exist, move
    it forward by the timezone's actual transition gap. If it is ambiguous,
    select the post-transition (chronologically later) occurrence.

    EMHASS uses either the configured IANA timezone name or a pytz timezone,
    depending on the calling path. IANA names are normalized with
    pytz.timezone(). When no timezone is configured and timestamp is naive,
    preserve the pre-existing naive calendar-day behavior.
    """
    ts = pd.Timestamp(timestamp)
    days = int(days)

    if time_zone is None and ts.tzinfo is None:
        return ts if days == 0 else ts + pd.DateOffset(days=days)

    tz = time_zone if time_zone is not None else ts.tz
    if isinstance(tz, str):
        # Some existing runtime-parameter paths carry the configured IANA
        # timezone name until get_yaml_parse() converts it. Preserve that
        # established contract and resolve it through EMHASS's pytz dependency.
        tz = pytz.timezone(tz)
    if not hasattr(tz, "localize") or not hasattr(tz, "normalize"):
        raise TypeError("time_zone must be an IANA timezone name or pytz timezone")

    if ts.tzinfo is None:
        ts = pd.Timestamp(tz.localize(ts.to_pydatetime(), is_dst=None))

    local = ts.tz_convert(tz)
    if days == 0:
        return local

    nominal = local.tz_localize(None) + pd.DateOffset(days=days)
    nominal_dt = pd.Timestamp(nominal).to_pydatetime()
    try:
        resolved = tz.localize(nominal_dt, is_dst=None)
    except pytz.NonExistentTimeError:
        # Attach the pre-transition offset, then normalize. This advances the
        # nominal wall time by the timezone's real gap (e.g. 60 min in Sydney,
        # 30 min on Lord Howe) rather than hard-coding an hour.
        resolved = tz.normalize(tz.localize(nominal_dt, is_dst=False))
    except pytz.AmbiguousTimeError:
        # The is_dst flag does not encode chronology (zones such as Africa/Casablanca
        # and Europe/Dublin use negative DST), so pick the later UTC instant instead.
        resolved = max(
            tz.localize(nominal_dt, is_dst=True),
            tz.localize(nominal_dt, is_dst=False),
        )
    return pd.Timestamp(resolved)


def get_forecast_start(
    now: pd.Timestamp | datetime,
    freq: pd.Timedelta,
    time_zone: datetime.tzinfo | str,
    method_ts_round: str = "nearest",
) -> pd.Timestamp:
    """Round a clock reading to the first optimization step, without wall-clock ambiguity.

    This is the single definition of the forecast start used by both the runtime
    parameters (``get_forecast_dates``) and ``Forecast``, so that they build the same grid.
    Whole seconds are kept in UTC and the result is returned in ``time_zone``; local wall
    times are never localized, so repeated (fall-back) and missing (spring-forward) hours
    cannot raise or select the wrong occurrence.

    - ``nearest``: round to the closest step on the UTC clock (ties to even).
    - ``first``: the latest step that is not after ``now``.
    - ``last``: the earliest step that is not before ``now``.

    ``first`` and ``last`` stay aligned to the local wall clock (e.g. :00/:30 in
    Asia/Kathmandu) using the UTC offset in force at ``now``.

    :param now: The clock reading; must be timezone-aware.
    :param freq: The optimization time step.
    :param time_zone: The configured site time zone the result is expressed in.
    :param method_ts_round: ``nearest``, ``first`` or ``last``.
    :raises ValueError: If ``method_ts_round`` is not one of the supported values.
    """
    # Whole seconds, rounded down as before: a request a fraction of a second after a step
    # boundary (a cron job firing at :00) still gets that boundary for "last".
    now_utc = pd.Timestamp(now).tz_convert("UTC").floor("s")
    if method_ts_round == "nearest":
        start_utc = now_utc.round(freq)
    elif method_ts_round in ("first", "last"):
        offset = now_utc.tz_convert(time_zone).utcoffset()
        wall = (now_utc + offset).tz_localize(None)
        wall = wall.floor(freq) if method_ts_round == "first" else wall.ceil(freq)
        start_utc = (wall - offset).tz_localize("UTC")
    else:
        raise ValueError(
            f"method_ts_round must be 'nearest', 'first' or 'last', got {method_ts_round!r}"
        )
    return start_utc.tz_convert(time_zone)


def get_forecast_dates(
    freq: int,
    delta_forecast: int,
    time_zone: datetime.tzinfo,
    timedelta_days: int | None = 0,
    method_ts_round: str = "nearest",
    now: datetime | None = None,
) -> list[str]:
    """
    Get the date_range list of the needed future dates using the delta_forecast parameter.

    :param freq: Optimization time step.
    :type freq: int
    :param delta_forecast: Number of local calendar days to forecast in the future for the optimization.
    :type delta_forecast: int
    :param timedelta_days: Additional local calendar days needed for the forecast range, defaults to 0
    :type timedelta_days: Optional[int], optional
    :param method_ts_round: How the first step is rounded, see :func:`get_forecast_start`.
        Callers that feed ``Forecast`` must pass the configured ``method_ts_round`` so both
        build the same grid. Defaults to ``nearest``, the configuration default, so a caller that
        omits it gets the grid a default-configured run uses.
    :type method_ts_round: str, optional
    :param now: The timezone-aware clock reading to build the grid from. Pass the same value to
        every call of one request so they share a grid; defaults to the current time.
    :type now: datetime, optional
    :return: The forecast dates as ISO 8601 strings, one per optimization step.
    :rtype: list[str]

    """
    freq = pd.to_timedelta(freq, "minutes")
    # _get_now() is the timezone-aware UTC instant; the start is derived from it in UTC so the
    # host clock zone is irrelevant. get_forecast_start() raises on a naive value.
    start_forecast = get_forecast_start(
        _get_now() if now is None else now, freq, time_zone, method_ts_round
    )
    total_days = int(delta_forecast) + int(timedelta_days or 0)
    final_end_date = add_local_calendar_days(start_forecast, total_days, time_zone) - freq

    forecast_dates = pd.date_range(
        start=start_forecast,
        end=final_end_date,
        freq=freq,
        tz=time_zone,
    )

    return [ts.isoformat() for ts in forecast_dates]


def normalize_heat_cool_mode(
    value: str,
    *,
    field_name: str = "mode",
    context: str | None = None,
) -> str:
    """Normalize heat/cool mode values and raise on invalid input."""
    mode_norm = str(value).strip().lower()
    if mode_norm not in {"heat", "cool"}:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}invalid {field_name} '{value}'. Expected 'heat' or 'cool'.")
    return mode_norm


def calculate_cop_heatpump(
    supply_temperature: float | np.ndarray | pd.Series,
    carnot_efficiency: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    mode: str = "heat",
) -> np.ndarray:
    r"""
    Calculate heat pump Coefficient of Performance (COP) for each timestep in the prediction horizon.

        The COP is calculated using a Carnot-based formula. Two modes are supported:

        - ``mode='heat'`` (default):

            .. math::
                    COP_{heat}(h) = \eta_{carnot} \times \frac{T_{supply\_K}}{T_{supply\_K} - T_{outdoor\_K}(h)}

        - ``mode='cool'``:

            .. math::
                    COP_{cool}(h) = \eta_{carnot} \times \frac{T_{supply\_K}}{T_{outdoor\_K}(h) - T_{supply\_K}}

    Where temperatures are converted to Kelvin (K = °C + 273.15).

    This formula models real heat pump behavior where COP decreases as the temperature lift
    (difference between supply and outdoor temperature) increases. The carnot_efficiency factor
    represents the real-world efficiency as a fraction of the ideal Carnot cycle efficiency.

    :param supply_temperature: The heat pump supply temperature in degrees Celsius. \
        Can be a scalar (constant supply T) or an array/Series matching the length of \
        ``outdoor_temperature_forecast`` for weather-compensated supply T (heating curve). \
        Typical scalar values: 30-40°C for underfloor heating, 50-70°C for radiator systems.
    :type supply_temperature: float or np.ndarray or pd.Series
    :param carnot_efficiency: Real-world efficiency factor as fraction of ideal Carnot cycle. \
        Typical range: 0.35-0.50 (35-50%). Default in thermal battery config: 0.4 (40%). \
        Higher values represent more efficient heat pumps.
    :type carnot_efficiency: float
    :param outdoor_temperature_forecast: Array of outdoor temperature forecasts in degrees Celsius, \
        one value per timestep in the prediction horizon.
    :type outdoor_temperature_forecast: np.ndarray or pd.Series
    :param mode: Operating mode, either ``"heat"`` or ``"cool"``. In cooling mode,
        the Carnot lift uses ``T_outdoor - T_supply`` so warm weather no longer
        collapses to COP=1.0 as in heating-only validation.
    :type mode: str
    :return: Array of COP values for each timestep, same length as outdoor_temperature_forecast. \
        Typical COP range: 2-6 for normal operating conditions.
    :rtype: np.ndarray

    Example:
        >>> supply_temp = 35.0  # °C, underfloor heating
        >>> carnot_eff = 0.4  # 40% of ideal Carnot efficiency
        >>> outdoor_temps = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        >>> cops = calculate_cop_heatpump(supply_temp, carnot_eff, outdoor_temps)
        >>> cops
        array([3.521..., 4.108..., 4.926..., 6.163..., 8.217...])
        >>> # At 5°C outdoor: COP = 0.4 × 308.15K / 30K = 4.11

    """
    # Convert to numpy array if pandas Series
    if isinstance(outdoor_temperature_forecast, pd.Series):
        outdoor_temps = outdoor_temperature_forecast.values
    else:
        outdoor_temps = np.asarray(outdoor_temperature_forecast)

    # Supply temp can be scalar (constant) or array (heating curve). Broadcast either way.
    if isinstance(supply_temperature, pd.Series):
        supply_temps = supply_temperature.values
    elif np.ndim(supply_temperature) > 0:
        supply_temps = np.asarray(supply_temperature, dtype=float)
    else:
        supply_temps = float(supply_temperature)

    # Convert temperatures from Celsius to Kelvin for Carnot formula
    supply_temperature_kelvin = supply_temps + 273.15
    outdoor_temperature_kelvin = outdoor_temps + 273.15

    mode_norm = normalize_heat_cool_mode(mode, field_name="mode", context="COP calculation")

    # Calculate temperature lift depending on mode.
    if mode_norm == "heat":
        # Heating: source is outdoor, sink is supply side.
        temperature_diff = supply_temperature_kelvin - outdoor_temperature_kelvin
        invalid_relation = "outdoor temperature >= supply temperature"
    else:
        # Cooling: source is indoor/chilled supply side, sink is outdoor.
        temperature_diff = outdoor_temperature_kelvin - supply_temperature_kelvin
        invalid_relation = "outdoor temperature <= supply temperature"

    # Check for non-physical scenarios where Carnot lift is non-positive.
    if np.any(temperature_diff <= 0):
        logger = logging.getLogger(__name__)
        num_invalid = int(np.sum(temperature_diff <= 0))
        invalid_indices = np.nonzero(temperature_diff <= 0)[0]
        if np.ndim(supply_temps) > 0:
            supply_range = (
                f"supply range [{np.min(supply_temps):.1f}, {np.max(supply_temps):.1f}]°C"
            )
        else:
            supply_range = f"supply {float(supply_temps):.1f}°C"
        logger.warning(
            f"COP calculation: {num_invalid} timestep(s) have {invalid_relation}. "
            f"This is non-physical for {mode_norm} mode. "
            f"Indices: {invalid_indices.tolist()[:5]}{'...' if len(invalid_indices) > 5 else ''}. "
            f"{supply_range}. Setting COP to 1.0 for these periods."
        )

    # Vectorized Carnot-based COP calculation.
    # For non-physical cases (lift <= 0), use a neutral COP of 1.0.

    # Avoid division by zero: use a mask to only calculate for valid cases
    cop_values = np.ones_like(outdoor_temperature_kelvin)  # Default to 1.0 everywhere
    valid_mask = temperature_diff > 0
    if np.any(valid_mask):
        # supply_temperature_kelvin may be scalar or array - index when array.
        if np.ndim(supply_temperature_kelvin) > 0:
            supply_valid = supply_temperature_kelvin[valid_mask]
        else:
            supply_valid = supply_temperature_kelvin
        cop_values[valid_mask] = carnot_efficiency * supply_valid / temperature_diff[valid_mask]

    # Apply realistic bounds: minimum 1.0, maximum 8.0
    # - Lower bound: 1.0 means direct electric heating (no efficiency gain)
    # - Upper bound: 8.0 is an optimistic but reasonable maximum for modern heat pumps
    #   (prevents numerical instability from very small temperature differences)
    cop_values = np.clip(cop_values, 1.0, 8.0)

    return cop_values


def apply_heating_curve(
    heating_curve: dict,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute per-slot supply temperature from a weather-compensated heating curve.

    A heating curve specifies how the heat source modulates its supply temperature in
    response to outdoor temperature, the way every modern boiler / heat pump controller
    does. The linear form supported here:

        T_supply(t) = clip(offset - slope * T_outdoor(t), min_supply, max_supply)

    :param heating_curve: dict with required ``slope`` and ``offset`` (both °C), plus
        optional ``min_supply`` (default 25°C) and ``max_supply`` (default 70°C).
    :param outdoor_temperature_forecast: Per-slot outdoor temperature in °C.
    :return: Per-slot supply temperature array in °C.
    """
    slope = float(heating_curve["slope"])
    offset = float(heating_curve["offset"])
    min_supply = float(heating_curve.get("min_supply", 25.0))
    max_supply = float(heating_curve.get("max_supply", 70.0))
    if min_supply >= max_supply:
        raise ValueError(
            f"heating_curve: min_supply ({min_supply}) must be < max_supply ({max_supply})"
        )
    if isinstance(outdoor_temperature_forecast, pd.Series):
        outdoor = outdoor_temperature_forecast.values
    else:
        outdoor = np.asarray(outdoor_temperature_forecast, dtype=float)
    supply = offset - slope * outdoor
    return np.clip(supply, min_supply, max_supply)


def resolve_min_temperatures(
    config: dict,
    outdoor_temperature_forecast: np.ndarray | pd.Series | list | None,
    length: int,
) -> list[float | None]:
    """Compute the effective per-slot lower temperature bound for a storage tank.

    Combines two sources, taking the element-wise max so the more conservative
    floor always wins:

    1. Static ``min_temperatures`` (or ``min_temperature``) list - an absolute
       weather-independent floor for safety / comfort. Always respected.
    2. ``min_temperature_curve`` dict - a weather-compensated floor matching
       the radiator emission law. Same shape as ``heating_curve``:
       ``T = clip(offset - slope * T_outdoor, min_supply, max_supply)``.

    When only one is set, that one is used. When neither is set, returns an
    empty list (caller decides how to react).

    :param config: Tank or thermal_battery dict potentially carrying
        ``min_temperatures`` and/or ``min_temperature_curve``.
    :param outdoor_temperature_forecast: Per-slot outdoor temperature. May be
        ``None`` when only the static floor is configured.
    :param length: Optimization horizon length.
    :return: List of per-slot minimum temperatures (length = ``length``).
    """
    # Accept either the plural list form or a single scalar/list under the
    # singular key. Normalize a scalar to a one-element list so the slicing
    # / padding below works uniformly.
    static = config.get("min_temperatures")
    if static is None:
        static = config.get("min_temperature")
    if static is None:
        static = []
    elif isinstance(static, int | float):
        static = [float(static)]
    curve = config.get("min_temperature_curve")

    if not static and not curve:
        return []

    if curve is not None:
        if outdoor_temperature_forecast is None:
            raise ValueError("min_temperature_curve requires outdoor_temperature_forecast")
        curve_temps = apply_heating_curve(curve, outdoor_temperature_forecast)[:length]
    else:
        curve_temps = None

    if static:
        static_arr = np.asarray(list(static)[:length], dtype=float)
        if len(static_arr) < length:
            pad_value = static_arr[-1] if len(static_arr) else 20.0
            static_arr = np.concatenate([static_arr, np.full(length - len(static_arr), pad_value)])
    else:
        static_arr = None

    if curve_temps is not None and static_arr is not None:
        # fmax, not maximum: a null static entry (nan after the float cast) means
        # "no static floor at this step - defer to the curve". np.maximum would
        # propagate the nan and silently discard the curve floor for that slot.
        effective = np.fmax(static_arr, curve_temps)
    elif curve_temps is not None:
        effective = curve_temps
    else:
        effective = static_arr

    # A null/None entry in the static list survives np.asarray(dtype=float) as nan.
    # Map it back to None so the downstream "v is not None" guards (the hard-bound and
    # comfort-penalty builders, the recovery ramp) correctly read it as "no floor at
    # this step" rather than letting a nan propagate into the LP.
    return [None if np.isnan(v) else float(v) for v in effective]


def cop_from_tank_temperature(
    tank_temperature: Sequence[float] | np.ndarray,
    carnot_efficiency: float,
    outdoor_temperature_forecast: Sequence[float] | np.ndarray | pd.Series,
    approach: float = 5.0,
    cop_bounds: tuple[float, float] = (1.0, 8.0),
    mode: str = "heat",
) -> np.ndarray:
    """COP for a heat pump charging a tank to its *own* achieved temperature.

    The condenser (heat) runs at the tank temperature PLUS a small heat-exchanger
    approach; the evaporator (cool) runs at the tank temperature MINUS the approach.
    Either way the COP tracks the temperature the tank actually reaches rather than a
    fixed weather curve. Used by the post-solve DP COP refinement to test consistency
    and to set the COP at the globally-optimal temperature. Returns a per-timestep COP
    array clipped to ``cop_bounds``.
    """
    tank = np.asarray(tank_temperature, dtype=float)
    supply = tank - float(approach) if mode == "cool" else tank + float(approach)
    cops = calculate_cop_heatpump(
        supply_temperature=supply,
        carnot_efficiency=carnot_efficiency,
        outdoor_temperature_forecast=outdoor_temperature_forecast,
        mode=mode,
    )
    return np.clip(np.asarray(cops, dtype=float), cop_bounds[0], cop_bounds[1])


def _supply_temperature_profile(values, length, outdoor_temperature_forecast) -> np.ndarray:
    """`supply_temperature` given as a per-timestep list, as a float array.

    A single value is broadcast (so ``[30]`` and ``30`` mean the same thing).
    A longer list is truncated to the horizon, matching how the COP array is
    truncated; a shorter one raises instead of inventing a supply temperature
    for the remaining steps.
    """
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(
                f"thermal_battery 'supply_temperature' must be a number or a flat list, "
                f"got shape {values.shape}"
            )
        values = values.tolist()
    # _numeric_series rejects strings ("30"), booleans (True reads as 1.0), non-finite
    # values and nested entries, so a typo cannot silently price a timestep.
    arr = np.asarray(_numeric_series(values, "thermal_battery 'supply_temperature'"), dtype=float)
    target = length if length is not None else len(np.asarray(outdoor_temperature_forecast))
    if arr.size == 1:
        return np.full(target, float(arr[0]))
    if arr.size < target:
        raise ValueError(
            f"thermal_battery 'supply_temperature' has {arr.size} values but the horizon "
            f"has {target} timesteps"
        )
    return arr[:target]


def resolve_thermal_battery_cop(
    hc: dict,
    outdoor_temperature_forecast: Sequence[float] | np.ndarray | pd.Series | None,
    length: int | None = None,
) -> np.ndarray:
    """
    Resolve the per-timestep energy-conversion factor for a thermal_battery heat source.

    The thermal_battery model treats the conversion factor uniformly as a COP-like
    multiplier on input power: `Q_thermal = COP * P_in / 1000 * dt`. Two source types
    are supported:

        - Heat pump (default): COP is computed via the Carnot formula and varies with
            the outdoor temperature. Requires `supply_temperature` in `hc`; uses
            `carnot_efficiency` (default 0.4). If `sense` is set to `"cool"`, cooling
            Carnot lift is used (`T_outdoor - T_supply`).
    - Constant-efficiency source (gas boiler, oil burner, district heating, etc.):
      `efficiency` in `hc` is used as a flat conversion factor for every timestep.
      `supply_temperature` is not required and outdoor temperature is ignored. The
      constant value passes through Carnot bounds intentionally (typical 0.85-0.95
      for combustion sources).

    :param hc: The thermal_battery sub-config dict from def_load_config.
    :param outdoor_temperature_forecast: Outdoor temperature forecast in degrees
        Celsius. Required in heat-pump mode. In constant-efficiency mode it is
        ignored and may be ``None`` provided ``length`` is given explicitly.
    :param length: Number of timesteps in the returned array. Mandatory when
        ``outdoor_temperature_forecast`` is ``None``; otherwise truncates or
        passes through the forecast length when set, or returns the full forecast
        length when unset.
    :return: Numpy array of conversion factors, one per timestep.
    """
    if "efficiency" in hc and hc["efficiency"] is not None:
        efficiency = float(hc["efficiency"])
        if efficiency <= 0:
            raise ValueError(f"thermal_battery 'efficiency' must be positive, got {efficiency}")
        if length is None:
            if outdoor_temperature_forecast is None:
                raise ValueError(
                    "resolve_thermal_battery_cop in constant-efficiency mode "
                    "requires 'length' when outdoor_temperature_forecast is None"
                )
            length = len(outdoor_temperature_forecast)
        return np.full(length, efficiency)

    if outdoor_temperature_forecast is None:
        raise ValueError(
            "resolve_thermal_battery_cop in heat-pump mode requires outdoor_temperature_forecast"
        )
    # Cut the driving forecast to the requested horizon: a per-step supply profile is
    # truncated to `length`, and the Carnot lift subtracts the two, so a longer
    # forecast would broadcast (length,) against (len,) and raise.
    if length is not None and np.ndim(outdoor_temperature_forecast) > 0:
        outdoor_temperature_forecast = np.asarray(outdoor_temperature_forecast)[:length]
    sense = normalize_heat_cool_mode(
        hc.get("sense") or "heat", field_name="sense", context="thermal_battery"
    )
    # Curve mode: per-slot supply T from outdoor T (weather compensation). Heat sources
    # read `heating_curve`, cool sources read `cooling_curve` (and, as before that key
    # existed, a `heating_curve` when no `cooling_curve` is set); both fall back to a
    # constant `supply_temperature` when no curve is configured. A curve makes the supply
    # (and hence the COP) vary over the horizon, which is what makes the source
    # DP-refinable.
    if sense == "cool":
        curve = hc.get("cooling_curve") or hc.get("heating_curve")
    else:
        curve = hc.get("heating_curve")
    if curve:
        supply_temperature = apply_heating_curve(curve, outdoor_temperature_forecast)
    else:
        supply_temperature = hc.get("supply_temperature")
        if supply_temperature is None:
            raise ValueError(
                "thermal_battery requires either 'efficiency' (constant-efficiency mode), "
                "'supply_temperature' (constant heat-pump mode), or a "
                "'heating_curve'/'cooling_curve' (weather-compensated heat-pump mode)"
            )
        # A per-timestep supply profile (issue #1327): the temperature the executor
        # will actually run at each step, which no curve can express. The Carnot
        # helper already broadcasts an array; only the length has to be pinned to
        # the horizon here.
        if np.ndim(supply_temperature) > 0:
            supply_temperature = _supply_temperature_profile(
                supply_temperature, length, outdoor_temperature_forecast
            )
    cops = calculate_cop_heatpump(
        supply_temperature=supply_temperature,
        carnot_efficiency=hc.get("carnot_efficiency", 0.4),
        outdoor_temperature_forecast=outdoor_temperature_forecast,
        mode=sense,
    )
    return cops if length is None else cops[:length]


def calculate_surface_solar_gain(
    hc: dict,
    ghi_forecast: np.ndarray | None,
    optimization_time_step_minutes: float,
    length: int | None = None,
) -> np.ndarray | None:
    """
    Compute per-timestep solar energy absorbed by an exposed thermal mass surface.

    Intended for thermal_battery configs that model a thermal store exposed
    directly to sunlight (a pool, an outdoor tank, a solar-thermal collector
    routed into a buffer). The gain is independent of the heater and acts
    as a negative term on `heating_demand` (i.e. the optimizer needs less
    pumped heat to maintain temperature when there is solar gain).

    Reuses the existing GHI forecast that EMHASS already fetches for PV. No
    second weather API call is required.

    :param hc: The thermal_battery sub-config dict. Reads two keys:
        - `solar_absorption_area`: effective horizontal absorption surface (m²).
        - `solar_absorption_factor`: fraction of GHI absorbed by the thermal
          mass (typical pool with no cover: 0.7-0.9; covered pool: 0.2-0.4).
          Defaults to 0.7 if absent.
    :param ghi_forecast: Global horizontal irradiance forecast in W/m² per
        timestep. Pass None or zero-length array to skip gain.
    :param optimization_time_step_minutes: Timestep duration in minutes.
    :param length: Truncate / pad the returned array to this length.
    :return: Solar gain in kWh per timestep, or None if not applicable.
    """
    absorption_area = hc.get("solar_absorption_area")
    if absorption_area is None or float(absorption_area) <= 0:
        return None
    if ghi_forecast is None:
        return None

    ghi_arr = np.asarray(ghi_forecast, dtype=float)
    if length is not None:
        if len(ghi_arr) < length:
            ghi_arr = np.concatenate((ghi_arr, np.zeros(length - len(ghi_arr))))
        else:
            ghi_arr = ghi_arr[:length]

    absorption_factor = float(hc.get("solar_absorption_factor", 0.7))
    if absorption_factor < 0:
        raise ValueError(
            f"thermal_battery solar_absorption_factor must be >= 0, got {absorption_factor}"
        )
    dt_hours = optimization_time_step_minutes / 60.0
    # W/m² * m² * factor / 1000 (kW per W) * hours = kWh
    return ghi_arr * float(absorption_area) * absorption_factor / 1000.0 * dt_hours


def _numeric_series(value, path: str) -> list[float]:
    """A list of finite numbers (not booleans or strings), as floats, or a
    ValueError naming ``path``. A string is rejected rather than read as a list
    of characters. NumPy real scalars are numbers like their Python equivalents:
    the same values in an array already pass through ``tolist()``."""
    if not isinstance(value, list | tuple):
        raise ValueError(f"{path} must be a list of numbers, got {type(value).__name__}")
    out = []
    for i, v in enumerate(value):
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValueError(f"{path}[{i}] must be a number, got {v!r}")
        try:
            f = float(v)
        except OverflowError:
            raise ValueError(f"{path}[{i}] is too large") from None
        if not math.isfinite(f):
            raise ValueError(f"{path}[{i}] must be finite, got {v!r}")
        out.append(f)
    return out


# The top-level keys compile_heat_topology reads. The compiler ignores any other
# key; the config page rejects them on save, where an unknown key is a typo.
HEAT_TOPOLOGY_KEYS = frozenset(
    {
        "sources",
        "storage",
        "consumers",
        "flows",
        "actuator_groups",
        "cost_tracks",
        "extend_deferrable_loads",
    }
)


def compile_heat_topology(topology: dict) -> dict:
    """Compile a heat-topology graph descriptor into flat optim_conf fields.

    The graph model lets users declare a small directed graph of
    `sources`, `storage`, `consumers`, `flows`, and `actuator_groups`. This
    function translates that high-level descriptor into the primitives the
    optimizer already understands:

    - Each `flow (source, storage)` becomes one deferrable load with a
      `thermal_source` block in def_load_config.
    - Each `storage` becomes one entry in `shared_thermal_tanks` with
      `load_ids` pointing at the deferrable loads feeding it.
    - Each `consumer` is folded into its target storage (its profile or
      building model populates the storage's draw_off_demand / u_value etc.).
    - Each `actuator_group` becomes one entry in `deferrable_load_groups`.
    - Each source's `cost_track` reference resolves to an entry in
      `cost_forecast_per_deferrable_load`.

    Returns a dict of optim_conf fields to merge into the live optim_conf:
        number_of_deferrable_loads, nominal_power_of_deferrable_loads,
        minimum_power_of_deferrable_loads, treat_deferrable_load_as_semi_cont,
        operating_hours_of_each_deferrable_load, def_load_config,
        shared_thermal_tanks, deferrable_load_groups,
        cost_forecast_per_deferrable_load.

    Validation: missing source/storage references, duplicated ids, and
    consumers targeting unknown storage all raise ValueError with the
    offending field path.
    """
    if not isinstance(topology, dict) or not topology:
        return {}
    # A missing or null field means "none"; any other value must have the right
    # type, so a falsey value such as "" or 0 is not silently read as empty.
    for key in ("sources", "storage", "consumers", "flows", "actuator_groups"):
        if topology.get(key) is not None and not isinstance(topology[key], list):
            raise ValueError(f"heat_topology.{key} must be a list")
    if topology.get("cost_tracks") is not None and not isinstance(topology["cost_tracks"], dict):
        raise ValueError("heat_topology.cost_tracks must be an object")
    if not isinstance(topology.get("extend_deferrable_loads", False), bool):
        raise ValueError("heat_topology.extend_deferrable_loads must be true or false")
    sources = topology.get("sources", []) or []
    storage = topology.get("storage", []) or []
    consumers = topology.get("consumers", []) or []
    flows = topology.get("flows", []) or []
    groups = topology.get("actuator_groups", []) or []
    cost_tracks = topology.get("cost_tracks", {}) or {}

    # Validate ids up front so a malformed entry raises the documented
    # ValueError naming the offending field, not an internal KeyError (the
    # documented contract: an invalid topology raises ValueError).
    for kind, entries in (("sources", sources), ("storage", storage)):
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or entry.get("id") in (None, ""):
                raise ValueError(f"heat_topology.{kind}[{i}] is missing the required 'id' field")

    src_by_id = {s["id"]: s for s in sources}
    src_index_by_id = {s["id"]: i for i, s in enumerate(sources)}
    sto_by_id = {s["id"]: s for s in storage}
    if len(src_by_id) != len(sources):
        raise ValueError("heat_topology.sources contains duplicate ids")
    if len(sto_by_id) != len(storage):
        raise ValueError("heat_topology.storage contains duplicate ids")
    # Flows resolve `from` against sources first, so an id shared by a source and a
    # storage would silently turn a tank->tank transfer into a source load.
    shared_ids = sorted(set(src_by_id) & set(sto_by_id))
    if shared_ids:
        raise ValueError(
            f"heat_topology: ids used by both a source and a storage: {shared_ids}; "
            "source and storage ids must be distinct"
        )

    # Split flows: source->storage flows become deferrable loads; storage->storage
    # flows are tank->tank heat transfers (e.g. a buffer feeding a room through the
    # emitters). Both must point at a real storage; the source side differs.
    source_flows = []
    transfer_flows = []
    transfer_pairs = set()
    # Each transfer publishes its schedule as P_transfer_<from>_<to>.
    transfer_columns = {}
    for i, f in enumerate(flows):
        if f.get("to") not in sto_by_id:
            raise ValueError(
                f"heat_topology.flows[{i}].to='{f.get('to')}' does not match any storage.id"
            )
        if f.get("from") in src_by_id:
            source_flows.append(f)
        elif f.get("from") in sto_by_id:
            if f["from"] == f["to"]:
                raise ValueError(
                    f"heat_topology.flows[{i}] is a self-transfer ('{f['from']}' -> itself)"
                )
            if (f["from"], f["to"]) in transfer_pairs:
                raise ValueError(
                    f"heat_topology.flows[{i}] repeats the transfer {f['from']}->{f['to']}; "
                    "one flow per pair of storages"
                )
            column = f"P_transfer_{f['from']}_{f['to']}"
            if column in transfer_columns:
                other = transfer_columns[column]
                raise ValueError(
                    f"heat_topology.flows[{i}] ({f['from']}->{f['to']}) and the transfer "
                    f"{other[0]}->{other[1]} would both publish as {column}; rename a "
                    "storage so the two schedules stay apart"
                )
            transfer_pairs.add((f["from"], f["to"]))
            transfer_columns[column] = (f["from"], f["to"])
            transfer_flows.append(f)
        else:
            raise ValueError(
                f"heat_topology.flows[{i}].from='{f.get('from')}' does not match any "
                "source.id or storage.id"
            )

    # Validate consumers target real storage
    for i, c in enumerate(consumers):
        if c.get("target") not in sto_by_id:
            raise ValueError(
                f"heat_topology.consumers[{i}].target='{c.get('target')}' does not match any storage.id"
            )

    # Build deferrable loads from the source->storage flows only
    num_loads = len(source_flows)
    nominal_power = []
    min_power = []
    treat_semi_cont = []
    operating_hours = []
    startup_penalty = []
    max_startups = []
    def_load_config = []
    cost_per_load: list = []
    is_electric_load: list[bool] = []
    flow_to_load_idx: dict[tuple[str, str], int] = {}
    cap_by_src_id: dict[str, float | list | None] = {}

    for i, f in enumerate(source_flows):
        src = src_by_id[f["from"]]
        src_nominal_power = float(src.get("nominal_power", 0))
        src_min_power = float(src.get("min_power", 0))
        # p >= min_power and p <= nominal_power cannot both hold, so this bound is
        # unsatisfiable; reject it here rather than let it surface as a generic
        # Infeasible. Equality is legal (off, or pinned at the nominal power).
        if src_min_power > src_nominal_power:
            raise ValueError(
                f"heat_topology.sources[{src['id']}].min_power must be <= nominal_power, "
                f"got min_power={src_min_power}, nominal_power={src_nominal_power}"
            )
        nominal_power.append(src_nominal_power)
        min_power.append(src_min_power)
        treat_semi_cont.append(bool(src.get("treat_as_semi_cont", True)))
        operating_hours.append(int(src.get("operating_hours", 4)))
        # Anti-short-cycle controls (optional, per source). The startup penalty
        # prices each off->on switch (softly discouraging short-cycling); the
        # max_startups cap is a hard limit on the number of starts per horizon.
        # Both default to 0 (disabled) so existing topologies are unchanged.
        penalty = float(src.get("startup_penalty", 0.0))
        if penalty < 0:
            raise ValueError(
                f"heat_topology.sources[{src['id']}].startup_penalty must be >= 0, got {penalty}"
            )
        startup_penalty.append(penalty)
        raw_cap = src.get("max_startups", 0)
        if (
            isinstance(raw_cap, bool)
            or not isinstance(raw_cap, int | float)
            or raw_cap < 0
            or raw_cap != int(raw_cap)
        ):
            raise ValueError(
                f"heat_topology.sources[{src['id']}].max_startups must be a whole "
                f"number >= 0, got {raw_cap!r}"
            )
        starts_cap = int(raw_cap)
        max_startups.append(starts_cap)
        # Source-side fields - shape expected by resolve_thermal_battery_cop
        source_block: dict = {}
        src_type = src.get("type", "").lower()
        # Type -> default electric bus membership. Explicit `electric` flag wins.
        type_is_electric = {
            "heatpump": True,
            "heat_pump": True,
            "electric": True,
            "gas": False,
            "oil": False,
            "district": False,
            "constant_efficiency": True,  # ambiguous - default electric unless overridden
        }
        if src_type in {"heatpump", "heat_pump"}:
            # A weather-compensated curve takes precedence (heating_curve for heat sources,
            # cooling_curve for chillers); a constant supply_temperature is the fallback.
            if "heating_curve" in src and src["heating_curve"]:
                hc_block = dict(src["heating_curve"])
                # Validate required fields up-front so users get clear errors
                for required in ("slope", "offset"):
                    if required not in hc_block:
                        raise ValueError(
                            f"heat_topology.sources[{src['id']}].heating_curve missing "
                            f"required field '{required}'"
                        )
                source_block["heating_curve"] = {
                    "slope": float(hc_block["slope"]),
                    "offset": float(hc_block["offset"]),
                    "min_supply": float(hc_block.get("min_supply", 25.0)),
                    "max_supply": float(hc_block.get("max_supply", 70.0)),
                }
            elif "cooling_curve" in src and src["cooling_curve"]:
                cc_block = dict(src["cooling_curve"])
                for required in ("slope", "offset"):
                    if required not in cc_block:
                        raise ValueError(
                            f"heat_topology.sources[{src['id']}].cooling_curve missing "
                            f"required field '{required}'"
                        )
                source_block["cooling_curve"] = {
                    "slope": float(cc_block["slope"]),
                    "offset": float(cc_block["offset"]),
                    "min_supply": float(cc_block.get("min_supply", 5.0)),
                    "max_supply": float(cc_block.get("max_supply", 18.0)),
                }
            elif "supply_temperature" in src:
                source_block["supply_temperature"] = float(src["supply_temperature"])
            else:
                raise ValueError(
                    f"heat_topology.sources[{src['id']}] (type=heatpump) requires "
                    "'supply_temperature', 'heating_curve', or 'cooling_curve'"
                )
            source_block["carnot_efficiency"] = float(src.get("carnot_efficiency", 0.4))
        elif src_type in {"gas", "oil", "district", "constant_efficiency", "electric"}:
            if src.get("efficiency") is None:
                raise ValueError(
                    f"heat_topology.sources[{src['id']}] (type={src_type}) requires 'efficiency'"
                )
            source_block["efficiency"] = float(src["efficiency"])
        else:
            raise ValueError(
                f"heat_topology.sources[{src_index_by_id[src['id']]}] "
                f"(id='{src['id']}'): type='{src.get('type', '')}' is not "
                "recognised. Allowed types: heatpump, heat_pump, gas, oil, "
                "district, electric, constant_efficiency."
            )
        # Propagate the target storage's comfort_sense onto the source block so the
        # COP resolver computes the correct (heating vs cooling) Carnot lift. Without
        # this, resolve_thermal_battery_cop defaults to "heat" and clamps the cooling
        # COP to 1.0 on a warm day (a heat pump can then never cool the zone).
        target_storage = sto_by_id.get(f["to"], {})
        if "comfort_sense" in target_storage:
            source_block["sense"] = str(target_storage["comfort_sense"]).lower()
        is_electric_load.append(bool(src.get("electric", type_is_electric.get(src_type, True))))
        # Optional per-source temperature ceiling: this source may only add heat
        # while the shared tank is at or below this temperature (e.g. a heat pump
        # limited to its supply/condenser temperature). A source without it (e.g.
        # an electric booster) can drive the tank up to the tank's max_temperatures.
        _cap = src.get("max_supply_temperature")
        if _cap is not None:
            if isinstance(_cap, list | tuple | np.ndarray):
                cap_value = [float(x) for x in _cap]
                if not cap_value:
                    raise ValueError(
                        f"heat_topology.sources[{src['id']}].max_supply_temperature "
                        "is an empty list; omit the key to leave the source uncapped."
                    )
            else:
                cap_value = float(_cap)
            if any(x <= 0 for x in (cap_value if isinstance(cap_value, list) else [cap_value])):
                raise ValueError(
                    f"heat_topology.sources[{src['id']}].max_supply_temperature "
                    "must be > 0 degC; a non-positive ceiling permanently disables "
                    "the source. Omit the key to leave it uncapped."
                )
            source_block["max_supply_temperature"] = cap_value
            cap_by_src_id[src["id"]] = cap_value
        else:
            cap_by_src_id[src["id"]] = None
            # A constant-supply heat pump (supply_temperature, no heating curve)
            # cannot physically heat water above its supply temperature, but
            # supply_temperature drives the COP only - it is NOT enforced as a
            # ceiling. Without max_supply_temperature the optimiser may plan to
            # heat the tank above it (a physically unreachable setpoint). Warn so
            # the user can add max_supply_temperature to cap it. (Left opt-in:
            # auto-capping at supply_temperature makes a tank whose min_temperature
            # sits at the supply temperature infeasible.)
            # Not needed when the storage's own ceiling already sits at or below
            # the supply temperature, and not meaningful for a cooling source.
            storage_max = target_storage.get(
                "max_temperatures", target_storage.get("max_temperature")
            )
            storage_max_values = [
                float(v)
                for v in (storage_max if isinstance(storage_max, list) else [storage_max])
                if isinstance(v, int | float)
            ]
            already_capped = bool(storage_max_values) and max(storage_max_values) <= float(
                source_block.get("supply_temperature", float("inf"))
            )
            if (
                "supply_temperature" in source_block
                and source_block.get("sense", "heat") != "cool"
                and not already_capped
            ):
                logging.getLogger(__name__).warning(
                    "heat_topology.sources[%s] is a fixed-supply heat pump "
                    "(supply_temperature=%.0f C) without max_supply_temperature: "
                    "supply_temperature sets the COP only and is not enforced as a "
                    "physical ceiling, so the optimiser may heat the tank above it. "
                    "Set max_supply_temperature to cap the tank at the supply "
                    "temperature.",
                    src["id"],
                    float(source_block["supply_temperature"]),
                )
        # Optional per-source thermal-output ceiling (W): caps cop * p_deferrable
        # so a high COP (e.g. a small mild-weather Carnot lift) cannot make the
        # model deliver more heat than the physical unit's rated thermal output.
        # Unlike max_supply_temperature this is a straight power bound, not a
        # temperature-gated switch, and is scalar-only: a unit's rated thermal
        # output is a fixed hardware spec, not weather-dependent like condenser
        # temperature.
        _thermal_cap = src.get("max_thermal_power")
        if _thermal_cap is not None:
            if isinstance(_thermal_cap, bool) or not isinstance(_thermal_cap, int | float):
                raise ValueError(
                    f"heat_topology.sources[{src['id']}].max_thermal_power must be a "
                    f"number of W, got {_thermal_cap!r}"
                )
            thermal_cap_value = float(_thermal_cap)
            if not np.isfinite(thermal_cap_value) or thermal_cap_value <= 0:
                raise ValueError(
                    f"heat_topology.sources[{src['id']}].max_thermal_power "
                    f"must be a finite number > 0 W, got {_thermal_cap!r}; omit the "
                    "key to leave the source uncapped."
                )
            source_block["max_thermal_power"] = thermal_cap_value
            # A source feeding several storages compiles into one load per flow;
            # the optimizer caps those loads together by this id.
            source_block["source_id"] = src["id"]
        # Optional per-source threshold: with the storage's desired_temperatures
        # set, this source stops beyond it (see the optimizer for the gate per
        # source mode). Sources without it (or with null) inherit the
        # storage-level overshoot_temperature at solve time.
        if src.get("overshoot_temperature") is not None:
            source_block["overshoot_temperature"] = float(src["overshoot_temperature"])
        def_load_config.append({"thermal_source": source_block})
        # Cost track resolution
        cost_track_id = src.get("cost_track")
        if cost_track_id is None:
            cost_per_load.append(None)
        else:
            if cost_track_id not in cost_tracks:
                raise ValueError(
                    f"heat_topology.sources[{src['id']}].cost_track='{cost_track_id}' "
                    "not found in cost_tracks"
                )
            cost_per_load.append(
                _numeric_series(
                    cost_tracks[cost_track_id], f"heat_topology.cost_tracks.{cost_track_id}"
                )
            )
        flow_to_load_idx[(f["from"], f["to"])] = i

    # Aggregate consumer demand onto storage
    storage_demand: dict[str, dict] = {}
    for ci, c in enumerate(consumers):
        target = c["target"]
        if target not in storage_demand:
            storage_demand[target] = {"profile": None, "building": None, "pool": None}
        ctype = (c.get("type") or "").lower()
        if ctype == "profile":
            if c.get("profile") is None:
                raise ValueError(f"heat_topology.consumers[{ci}] (type=profile) requires 'profile'")
            prof = _numeric_series(c["profile"], f"heat_topology.consumers[{ci}].profile")
            existing = storage_demand[target]["profile"]
            if existing is None:
                storage_demand[target]["profile"] = prof
            else:
                # Pad both lists to the common max length so a later profile
                # that is longer than `existing` is not silently truncated by
                # zip(). Demand after the shorter horizon stays at 0.
                max_len = max(len(existing), len(prof))
                existing_padded = existing + [0.0] * (max_len - len(existing))
                prof_padded = prof + [0.0] * (max_len - len(prof))
                storage_demand[target]["profile"] = [
                    a + b for a, b in zip(existing_padded, prof_padded)
                ]
        elif ctype == "building_demand":
            if storage_demand[target]["building"] is not None:
                raise ValueError(
                    f"heat_topology.consumers: storage '{target}' already has a "
                    "building_demand consumer; only one is permitted per storage"
                )
            storage_demand[target]["building"] = {
                k: c[k]
                for k in (
                    "u_value",
                    "envelope_area",
                    "ventilation_rate",
                    "heated_volume",
                    "indoor_target_temperature",
                    "window_area",
                    "shgc",
                    "internal_gains_factor",
                    "specific_heating_demand",
                    "area",
                    "base_temperature",
                    "annual_reference_hdd",
                )
                if k in c
            }
        elif ctype == "pool_comfort":
            storage_demand[target]["pool"] = {
                k: c[k] for k in ("solar_absorption_area", "solar_absorption_factor") if k in c
            }
        else:
            raise ValueError(
                f"heat_topology.consumers[{c.get('id')}].type='{ctype}' must be one of "
                "profile, building_demand, pool_comfort"
            )

    def _positive(value, field):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"heat_topology.{field} must be a positive number, got {value!r}")
        return number

    # Build shared_thermal_tanks from storage + aggregated demand + flows
    shared_tanks = []
    for s in storage:
        sid = s["id"]
        # Heat capacity: a water volume (with density and heat capacity) or a
        # thermal_mass. Checked here so an unusable storage is rejected when the
        # configuration is saved, not at the next unattended run.
        if s.get("thermal_mass") is not None:
            _positive(s["thermal_mass"], f"storage[{sid}].thermal_mass")
        elif s.get("volume") is not None:
            _positive(s["volume"], f"storage[{sid}].volume")
            for _cap_key in ("density", "heat_capacity"):
                if s.get(_cap_key) is not None:
                    _positive(s[_cap_key], f"storage[{sid}].{_cap_key}")
        else:
            raise ValueError(
                f"heat_topology.storage[{sid}] needs a 'volume' (m3) or a 'thermal_mass' (kWh/K)"
            )
        load_ids = [flow_to_load_idx[(f["from"], f["to"])] for f in source_flows if f["to"] == sid]
        tank: dict = {
            "id": sid,
            "load_ids": load_ids,
            "volume": float(s["volume"]) if s.get("volume") is not None else None,
            "density": float(s.get("density", 1000)),
            "heat_capacity": float(s.get("heat_capacity", 4.186)),
            "start_temperature": float(s.get("start_temperature", 20.0)),
            "thermal_loss": float(s.get("thermal_loss", 0.045)),
            "min_temperatures": list(s.get("min_temperature", []))
            or list(s.get("min_temperatures", [])),
            "max_temperatures": list(s.get("max_temperature", []))
            or list(s.get("max_temperatures", [])),
        }
        # Warn when the static minimum band is probably unreachable: if EVERY
        # source feeding this tank has a max_supply_temperature and some
        # min_temperatures[t] exceeds the highest ceiling at t, no source can
        # heat the tank to it. It is a warning, not an error: a tank that starts
        # hot enough can still hold the minimum, and the solver decides. Only the
        # static list is checked here; a min_temperature_curve resolves against
        # weather at solve time.
        feeding_caps = [cap_by_src_id[f["from"]] for f in source_flows if f["to"] == sid]
        is_cooling = str(s.get("comfort_sense") or "heat").strip().lower() == "cool"
        if not is_cooling and feeding_caps and all(c is not None for c in feeding_caps):
            for t, min_val in enumerate(tank["min_temperatures"]):
                if min_val is None:
                    continue
                ceiling = max(
                    (c[t] if t < len(c) else c[-1]) if isinstance(c, list) else c
                    for c in feeding_caps
                )
                if float(min_val) > ceiling:
                    logging.getLogger(__name__).warning(
                        "heat_topology.storage[%s].min_temperatures[%s]=%s exceeds the "
                        "highest max_supply_temperature (%s) of the sources feeding "
                        "this storage; no source can heat it that far, so only a tank "
                        "that is already hot enough can hold it. Raise a source "
                        "ceiling, lower the minimum, or add an uncapped source (e.g. "
                        "an electric booster).",
                        sid,
                        t,
                        min_val,
                        ceiling,
                    )
                    break
        # The same for overshoot_temperature: with a desired temperature, a
        # continuous source does not heat in a step that would end beyond it. A
        # semi-continuous source is gated at the start of a step, so one step can
        # still cross it, and a source without a threshold (none of its own, none
        # on the storage) can pass it: the check needs every feeding source to be
        # continuous and to have one.
        desired = s.get("desired_temperature", s.get("desired_temperatures"))
        feeding_srcs = [src_by_id[f["from"]] for f in source_flows if f["to"] == sid]
        feeding_overshoots = [
            src.get("overshoot_temperature")
            if src.get("overshoot_temperature") is not None
            else s.get("overshoot_temperature")
            for src in feeding_srcs
        ]
        if (
            desired not in (None, [])
            and feeding_overshoots
            and all(isinstance(o, int | float) for o in feeding_overshoots)
            and not any(bool(src.get("treat_as_semi_cont", True)) for src in feeding_srcs)
        ):
            is_cool = str(s.get("comfort_sense") or "heat").strip().lower() == "cool"
            limit = min(feeding_overshoots) if is_cool else max(feeding_overshoots)
            band_key = "max_temperatures" if is_cool else "min_temperatures"
            for t, val in enumerate(tank[band_key]):
                if val is None:
                    continue
                if (float(val) < limit) if is_cool else (float(val) > limit):
                    logging.getLogger(__name__).warning(
                        "heat_topology.storage[%s].%s[%s]=%s lies beyond the "
                        "overshoot_temperature (%s) of every source feeding this "
                        "storage; overshoot_temperature stops a source in any step "
                        "that would end beyond it, so only a storage that already "
                        "starts there can hold that bound. Move the threshold past "
                        "the bound, or add a source without one.",
                        sid,
                        band_key,
                        t,
                        val,
                        limit,
                    )
                    break
        # Weather-compensated minimum temperature: when the radiator needs a higher
        # supply T to keep up with building heat loss on a cold day, the buffer min
        # should track. Same linear law as the source's heating_curve.
        if "min_temperature_curve" in s and s["min_temperature_curve"]:
            mc = dict(s["min_temperature_curve"])
            for required in ("slope", "offset"):
                if required not in mc:
                    raise ValueError(
                        f"heat_topology.storage[{sid}].min_temperature_curve "
                        f"missing required field '{required}'"
                    )
            tank["min_temperature_curve"] = {
                "slope": float(mc["slope"]),
                "offset": float(mc["offset"]),
                "min_supply": float(mc.get("min_supply", 25.0)),
                "max_supply": float(mc.get("max_supply", 70.0)),
            }
        # Soft comfort constraints: penalize deviations from a target band rather than
        # hard min/max. Accept either scalar `desired_temperature` (broadcast to horizon
        # at solve time) or per-slot `desired_temperatures`. The optimizer's existing
        # thermal_battery soft-constraint code reads these fields directly.
        desired = s.get("desired_temperatures")
        if desired is None and "desired_temperature" in s:
            desired = s["desired_temperature"]
        if desired is not None:
            tank["desired_temperatures"] = (
                list(desired) if isinstance(desired, list | tuple) else float(desired)
            )
        if s.get("overshoot_temperature") is not None:
            tank["overshoot_temperature"] = float(s["overshoot_temperature"])
        if "penalty_factor" in s:
            tank["penalty_factor"] = float(s["penalty_factor"])
        if "comfort_sense" in s:
            sense = str(s["comfort_sense"]).lower()
            if sense not in ("heat", "cool"):
                raise ValueError(
                    f"heat_topology.storage[{sid}].comfort_sense='{sense}' must be 'heat' or 'cool'"
                )
            tank["sense"] = sense
        # Unified thermal-tank options (consolidation, issue #539): a storage MAY
        # opt into building-zone physics. All optional; omitting every one of them
        # preserves the existing water-tank behaviour byte-for-byte.
        #   thermal_mass     (kWh/K) - heat capacity directly, instead of volume
        #   loss_coefficient (kW/K)  - state-dependent loss UA*(T-outdoor)
        #   thermal_inertia  (h)     - lag between heat input and temperature
        #   window_area (m2) + shgc - solar gain through glazing (offsets heating)
        # Each field must be a finite number in its physical range: a bad value
        # would otherwise pass the save and fail every optimization run.
        for _zone_key, _is_valid, _range in (
            ("thermal_mass", lambda v: v > 0, "> 0 kWh/K"),
            ("loss_coefficient", lambda v: v >= 0, ">= 0 kW/K"),
            ("thermal_inertia", lambda v: v >= 0, ">= 0 h"),
            ("window_area", lambda v: v >= 0, ">= 0 m2"),
            ("shgc", lambda v: 0 <= v <= 1, "between 0 and 1"),
        ):
            raw = s.get(_zone_key)
            if raw is None:
                continue
            try:
                value = None if isinstance(raw, bool) else float(raw)
            except (TypeError, ValueError, OverflowError):
                value = None
            if value is None or not math.isfinite(value) or not _is_valid(value):
                raise ValueError(
                    f"heat_topology.storage[{sid}].{_zone_key} must be a number "
                    f"{_range}, got {raw!r}"
                )
            tank[_zone_key] = value
        # prior_heat is the lag model's initial condition (thermal kWh per step
        # already in flight from previous rolling-MPC runs), so unlike the scalars
        # above it is a per-step list and is passed through unconverted. Usually
        # supplied per run via runtimeparams rather than written into the topology.
        if s.get("prior_heat") is not None:
            tank["prior_heat"] = s["prior_heat"]
        demand = storage_demand.get(sid, {})
        if demand.get("profile") is not None:
            tank["draw_off_demand"] = demand["profile"]
        if demand.get("building"):
            tank.update(demand["building"])
        if demand.get("pool"):
            tank.update(demand["pool"])
        # loss_coefficient (state-dependent UA loss) and a building_demand consumer
        # both model heat loss to outdoor - reject both on one storage to avoid
        # double counting the space-heating demand.
        if tank.get("loss_coefficient") is not None and demand.get("building"):
            raise ValueError(
                f"heat_topology.storage[{sid}]: loss_coefficient and a building_demand "
                "consumer both model heat loss to outdoor; configure only one."
            )
        shared_tanks.append(tank)

    # Build deferrable_load_groups from actuator_groups
    def_groups = []
    for gi, g in enumerate(groups):
        names = []
        group_flows = g.get("flows")
        if not isinstance(group_flows, list) or not group_flows:
            raise ValueError(
                f"heat_topology.actuator_groups[{gi}].flows must list at least one "
                "[source_id, storage_id] pair"
            )
        for flow_pair in group_flows:
            if not isinstance(flow_pair, list | tuple) or len(flow_pair) != 2:
                raise ValueError(
                    f"heat_topology.actuator_groups[{gi}].flows entries must be "
                    f"[source_id, storage_id] pairs, got {flow_pair!r}"
                )
            key = (flow_pair[0], flow_pair[1])
            if key not in flow_to_load_idx:
                raise ValueError(
                    f"heat_topology.actuator_groups[{gi}].flows references unknown flow {flow_pair}"
                )
            names.append(f"deferrable{flow_to_load_idx[key]}")
        def_group = {
            "names": names,
            "mutual_exclusion": bool(g.get("mutual_exclusion", False)),
        }
        if "max_combined_power" in g:
            limit = g["max_combined_power"]
            if isinstance(limit, bool) or not isinstance(limit, int | float) or not limit > 0:
                raise ValueError(
                    f"heat_topology.actuator_groups[{gi}].max_combined_power must be a "
                    f"positive number (W), got {limit!r}"
                )
            def_group["max_power"] = float(limit)
        def_groups.append(def_group)

    # Tank->tank heat transfers (e.g. a buffer feeding a room through the
    # emitters). Each carries an emitter conductance (transfer_coefficient, kW/K)
    # and a maximum delivered power (max_transfer_power, W).
    tank_ids = {s["id"] for s in storage}
    tank_transfers = []
    for f in transfer_flows:
        if f["from"] not in tank_ids or f["to"] not in tank_ids:
            raise ValueError(
                f"heat_topology transfer flow {f.get('from')}->{f.get('to')} must connect two storage ids"
            )
        tank_transfers.append(
            {
                "from": f["from"],
                "to": f["to"],
                "transfer_coefficient": _positive(
                    f.get("transfer_coefficient", 1.0),
                    f"flows[{f['from']}->{f['to']}].transfer_coefficient",
                ),
                "max_transfer_power": _positive(
                    f.get("max_transfer_power", 1e6),
                    f"flows[{f['from']}->{f['to']}].max_transfer_power",
                ),
            }
        )

    return {
        "number_of_deferrable_loads": num_loads,
        "nominal_power_of_deferrable_loads": nominal_power,
        "minimum_power_of_deferrable_loads": min_power,
        "treat_deferrable_load_as_semi_cont": treat_semi_cont,
        "operating_hours_of_each_deferrable_load": operating_hours,
        "set_deferrable_load_single_constant": [False] * num_loads,
        "set_deferrable_startup_penalty": startup_penalty,
        "deferrable_load_max_cost": [0.0] * num_loads,
        "set_deferrable_max_startups": max_startups,
        "start_timesteps_of_each_deferrable_load": [0] * num_loads,
        "end_timesteps_of_each_deferrable_load": [0] * num_loads,
        "def_load_config": def_load_config,
        "shared_thermal_tanks": shared_tanks,
        "tank_transfers": tank_transfers,
        "deferrable_load_groups": def_groups,
        "cost_forecast_per_deferrable_load": cost_per_load,
        "is_electric_load": is_electric_load,
    }


def _runtime_shared_tanks_problem(tanks, n_loads: int) -> str | None:
    """Why a runtime shared_thermal_tanks list cannot be used, or None.

    Each tank's load_ids must be a list of whole numbers (not booleans or
    strings) within the configured loads, a load may feed only one tank and
    only once (otherwise its heat is counted twice and its power once), and
    tank ids must be unique (a start-temperature override would update every
    tank with that id). A non-list is reported by the caller.
    """
    if not isinstance(tanks, list) or not all(isinstance(t, dict) for t in tanks):
        return None
    members, ids = [], []
    for t in tanks:
        load_ids = t.get("load_ids")
        if not isinstance(load_ids, list):
            return f"tank {t.get('id')!r} needs load_ids as a list, got {load_ids!r}"
        for i in load_ids:
            # Check ints without converting them to float, which overflows for a
            # huge integer.
            whole = (isinstance(i, int) and not isinstance(i, bool)) or (
                isinstance(i, float) and i.is_integer()
            )
            if not whole or not 0 <= int(i) < n_loads:
                return (
                    f"tank {t.get('id')!r} has load_ids entry {i!r}; it must be a "
                    f"whole number from 0 to {n_loads - 1}"
                )
            members.append(int(i))
        ids.append(str(t.get("id")))
    repeated = sorted({i for i in members if members.count(i) > 1})
    if repeated:
        return f"load_ids {repeated} appear more than once; a load can feed only one tank, once"
    repeated_ids = sorted({i for i in ids if ids.count(i) > 1})
    if repeated_ids:
        return f"tank ids {repeated_ids} are used more than once"
    return None


def _per_user_load_list(value, offset: int) -> list:
    """A per-load value as a list for the user's ``offset`` loads: a list is
    copied, a scalar (e.g. a runtime override) applies to each of them, and a
    missing value is empty."""
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value] * offset


# The fields a heat_topology in replace mode always overwrites with its compiled
# values (otherwise the compiled def_load_config would not match
# number_of_deferrable_loads, etc.). All per-load arrays must match
# number_of_deferrable_loads, which the compiler sets.
_TOPOLOGY_REPLACED_KEYS = frozenset(
    {
        "number_of_deferrable_loads",
        "def_load_config",
        "shared_thermal_tanks",
        "tank_transfers",
        "deferrable_load_groups",
        "nominal_power_of_deferrable_loads",
        "minimum_power_of_deferrable_loads",
        "treat_deferrable_load_as_semi_cont",
        "cost_forecast_per_deferrable_load",
        "set_deferrable_load_single_constant",
        "set_deferrable_startup_penalty",
        "deferrable_load_max_cost",
        "set_deferrable_max_startups",
        "operating_hours_of_each_deferrable_load",
        "start_timesteps_of_each_deferrable_load",
        "end_timesteps_of_each_deferrable_load",
        "is_electric_load",
    }
)


def _prior_heat_values(series):
    """(values, None) for a usable shared_tank_prior_heat list, else (None, why)."""
    if isinstance(series, str) or not isinstance(series, list | tuple):
        return None, f"must be a list of numbers, got {type(series).__name__}"
    try:
        # A boolean is not heat (float(True) would be 1.0).
        if any(isinstance(v, bool) for v in series):
            raise TypeError("boolean entry")
        values = [float(v) for v in series]
    except (TypeError, ValueError, OverflowError):
        return None, f"={series!r} is not all numeric"
    if any(v < 0 or not np.isfinite(v) for v in values):
        return None, "must be finite and >= 0"
    return values, None


def _restore_configured_shared_tanks(params: dict, configured) -> None:
    """Undo the unchecked association copy of a rejected runtime tank list."""
    if configured is None:
        params["optim_conf"].pop("shared_thermal_tanks", None)
    else:
        params["optim_conf"]["shared_thermal_tanks"] = configured


def _warn_replaced_deferrable_loads(
    optim_conf: dict, compiled: dict, emhass_conf: dict, logger: logging.Logger
) -> None:
    """Warn when a heat_topology in replace mode drops configured deferrable loads.

    Without extend_deferrable_loads the compiled flows replace the whole load
    set, so loads configured beyond them (a dishwasher, a pool pump) are not
    optimized. The shipped defaults configure two example loads; a load set whose
    every overwritten field still equals the shipped value is taken as untouched
    and replaced without a word.
    """
    try:
        configured = int(optim_conf.get("number_of_deferrable_loads") or 0)
    except (TypeError, ValueError):
        return
    compiled_count = int(compiled.get("number_of_deferrable_loads") or 0)
    if configured <= compiled_count:
        return
    defaults_path = (emhass_conf or {}).get("defaults_path")
    try:
        defaults = orjson.loads(pathlib.Path(defaults_path).read_bytes()) if defaults_path else {}
    except (OSError, orjson.JSONDecodeError):
        defaults = {}

    def _shipped(key):
        value = optim_conf.get(key)
        if key in defaults:
            return value == defaults[key]
        # Not in the defaults (def_load_config, the tank lists): only an empty
        # value, or a list of empty entries, is untouched.
        return not value or (isinstance(value, list) and not any(value))

    if defaults and all(_shipped(key) for key in _TOPOLOGY_REPLACED_KEYS):
        return
    logger.warning(
        "heat_topology replaces the deferrable loads: %d are configured but its flows "
        "compile to %d, so loads %d..%d are not optimized. Set extend_deferrable_loads: "
        "true in heat_topology to keep the configured loads and add the topology's "
        "after them.",
        configured,
        compiled_count,
        compiled_count,
        configured - 1,
    )


def _extend_optim_conf_with_compiled_topology(
    optim_conf: dict, compiled: dict, logger: logging.Logger
) -> None:
    """Append compiled heat_topology loads after the user's configured loads.

    Used when the topology sets ``extend_deferrable_loads: true``. The user's
    first N deferrable loads stay untouched (short per-load arrays are padded
    to N with the usual defaults first, longer ones truncated to N so the
    compiled loads land at deterministic indices), the compiled loads take
    indices N..N+M-1, and every compiled load reference (shared-tank
    ``load_ids``, actuator-group ``deferrable<i>`` names) is shifted by N.
    Existing manual ``shared_thermal_tanks`` / ``deferrable_load_groups``
    entries are kept; the compiled ones are appended after them.
    """
    offset = int(optim_conf.get("number_of_deferrable_loads") or 0)
    num_compiled = compiled["number_of_deferrable_loads"]

    # Per-load arrays: user's values (normalized to `offset` entries) followed
    # by the compiled values. Pad defaults match the established per-load
    # defaults used elsewhere (check_def_loads call sites).
    pad_defaults = {
        "nominal_power_of_deferrable_loads": 0.0,
        "minimum_power_of_deferrable_loads": 0.0,
        "treat_deferrable_load_as_semi_cont": True,
        "operating_hours_of_each_deferrable_load": 0,
        "set_deferrable_load_single_constant": False,
        "set_deferrable_startup_penalty": 0.0,
        "deferrable_load_max_cost": 0.0,
        "set_deferrable_max_startups": 0,
        "start_timesteps_of_each_deferrable_load": 0,
        "end_timesteps_of_each_deferrable_load": 0,
        "cost_forecast_per_deferrable_load": None,
        "is_electric_load": True,
    }
    for key, default in pad_defaults.items():
        existing = _per_user_load_list(optim_conf.get(key), offset)
        existing += [default] * (offset - len(existing))
        optim_conf[key] = existing[:offset] + list(compiled[key])
    # Minimum on/off times and MPC operating timesteps are not compiled; keep the
    # user's entries for their own loads and give the topology loads none (a
    # longer user array must not leak onto the appended loads).
    n_compiled = len(compiled["def_load_config"])
    for key in (
        "def_minimum_on_time",
        "def_minimum_off_time",
        "operating_timesteps_of_each_deferrable_load",
    ):
        existing = optim_conf.get(key)
        if existing is not None:
            existing = _per_user_load_list(existing, offset)[:offset]
            optim_conf[key] = existing + [0] * (offset - len(existing)) + [0] * n_compiled
    # Runtime state for the user's loads: the appended topology loads get the
    # defaults the optimizer would assume (off, no elapsed time, no power), so
    # every array matches the final load count. An array that already covers
    # every load is kept, and one of any other length is left to the
    # optimizer's own length handling.
    for key, default in (
        ("def_current_state", False),
        ("def_current_on_timesteps", 0),
        ("def_current_off_timesteps", 0),
        ("def_current_power", 0.0),
        ("def_current_operating_timesteps", 0),
    ):
        existing = optim_conf.get(key)
        if isinstance(existing, list) and len(existing) == offset:
            optim_conf[key] = list(existing) + [default] * n_compiled
    # def_load_config pads with fresh dicts (no shared instance between slots).
    def_cfgs = optim_conf.get("def_load_config")
    def_cfgs = list(def_cfgs) if isinstance(def_cfgs, list) else []
    def_cfgs += [{} for _ in range(offset - len(def_cfgs))]
    optim_conf["def_load_config"] = def_cfgs[:offset] + list(compiled["def_load_config"])

    # Shift compiled load references by the user's load count.
    shifted_tanks = []
    for tank in compiled["shared_thermal_tanks"]:
        tank = dict(tank)
        tank["load_ids"] = [int(i) + offset for i in tank.get("load_ids", [])]
        shifted_tanks.append(tank)
    shifted_groups = []
    for group in compiled["deferrable_load_groups"]:
        group = dict(group)
        group["names"] = [
            f"deferrable{int(name.removeprefix('deferrable')) + offset}"
            for name in group.get("names", [])
        ]
        shifted_groups.append(group)
    kept_tanks = list(optim_conf.get("shared_thermal_tanks") or [])
    kept_ids = {str(t.get("id")) for t in kept_tanks if isinstance(t, dict)}
    clashes = sorted(kept_ids & {str(t.get("id")) for t in shifted_tanks})
    if clashes:
        raise ValueError(
            f"heat_topology storage ids {clashes} are also used by the configured "
            "shared_thermal_tanks; with extend_deferrable_loads both are kept, so "
            "each tank needs its own id"
        )
    optim_conf["shared_thermal_tanks"] = kept_tanks + shifted_tanks
    optim_conf["deferrable_load_groups"] = (
        list(optim_conf.get("deferrable_load_groups") or []) + shifted_groups
    )
    # Tank->tank transfers are compiler output only (no user-facing setting), so any
    # existing value is a previous compile's: replace it rather than append.
    optim_conf["tank_transfers"] = list(compiled.get("tank_transfers") or [])

    optim_conf["number_of_deferrable_loads"] = offset + num_compiled
    logger.info(
        "heat_topology extend_deferrable_loads: %d configured loads kept, "
        "%d topology loads appended at indices %d..%d",
        offset,
        num_compiled,
        offset,
        offset + num_compiled - 1,
    )


def calculate_thermal_loss_signed(
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    indoor_temperature: float,
    base_loss: float,
) -> np.ndarray:
    r"""
    Calculate signed thermal loss factor based on indoor/outdoor temperature difference.

    **SIGN CONVENTION:**
    - **Positive** (+loss): outdoor < indoor → heat loss, building cools, heating required
    - **Negative** (-loss): outdoor ≥ indoor → heat gain, building warms passively

    Formula: loss * (1 - 2 * Hot(h)), where Hot(h) = 1 if outdoor ≥ indoor, else 0.
    Based on Langer & Volling (2020) Equation B.13.

    :param outdoor_temperature_forecast: Outdoor temperature forecast (°C)
    :type outdoor_temperature_forecast: np.ndarray or pd.Series
    :param indoor_temperature: Indoor/target temperature threshold (°C)
    :type indoor_temperature: float
    :param base_loss: Base thermal loss coefficient in kW
    :type base_loss: float
    :return: Signed loss array (positive = heat loss, negative = heat gain)
    :rtype: np.ndarray

    """
    # Convert to numpy array if pandas Series
    if isinstance(outdoor_temperature_forecast, pd.Series):
        outdoor_temps = outdoor_temperature_forecast.values
    else:
        outdoor_temps = np.asarray(outdoor_temperature_forecast)

    # Create binary hot indicator: 1 if outdoor temp >= indoor temp, 0 otherwise
    hot_indicator = (outdoor_temps >= indoor_temperature).astype(float)

    return base_loss * (1.0 - 2.0 * hot_indicator)


def calculate_heating_demand(
    specific_heating_demand: float,
    floor_area: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    base_temperature: float = 18.0,
    annual_reference_hdd: float = 3000.0,
    optimization_time_step: int | None = None,
) -> np.ndarray:
    """
    Calculate heating demand per timestep based on heating degree days method.

    Uses heating degree days (HDD) to calculate heating demand based on outdoor temperature
    forecast, specific heating demand, and floor area. The specific heating demand should be
    calibrated to the annual reference HDD value.

    :param specific_heating_demand: Specific heating demand in kWh/m²/year (calibrated to annual_reference_hdd)
    :type specific_heating_demand: float
    :param floor_area: Floor area in m²
    :type floor_area: float
    :param outdoor_temperature_forecast: Outdoor temperature forecast in °C for each timestep
    :type outdoor_temperature_forecast: np.ndarray | pd.Series
    :param base_temperature: Base temperature for HDD calculation in °C, defaults to 18.0 (European standard)
    :type base_temperature: float, optional
    :param annual_reference_hdd: Annual reference HDD value for normalization, defaults to 3000.0 (Central Europe)
    :type annual_reference_hdd: float, optional
    :param optimization_time_step: Optimization time step in minutes. If None, automatically infers from
        pandas Series DatetimeIndex frequency. Falls back to 30 minutes if not inferrable.
    :type optimization_time_step: int | None, optional
    :return: Array of heating demand values (kWh) per timestep
    :rtype: np.ndarray

    """

    # Convert outdoor temperature forecast to numpy array if pandas Series
    outdoor_temps = (
        outdoor_temperature_forecast.values
        if isinstance(outdoor_temperature_forecast, pd.Series)
        else np.asarray(outdoor_temperature_forecast)
    )

    # Calculate heating degree days per timestep
    # HDD = max(base_temperature - outdoor_temperature, 0)
    hdd_per_timestep = np.maximum(base_temperature - outdoor_temps, 0.0)

    # Determine timestep duration in hours
    if optimization_time_step is None:
        # Try to infer from pandas Series DatetimeIndex
        if isinstance(outdoor_temperature_forecast, pd.Series) and isinstance(
            outdoor_temperature_forecast.index, pd.DatetimeIndex
        ):
            if len(outdoor_temperature_forecast.index) > 1:
                freq_minutes = (
                    outdoor_temperature_forecast.index[1] - outdoor_temperature_forecast.index[0]
                ).total_seconds() / 60.0
                hours_per_timestep = freq_minutes / 60.0
            else:
                # Single datapoint, fallback to default 30 min
                hours_per_timestep = 0.5
        else:
            # Cannot infer, use default 30 minutes
            hours_per_timestep = 0.5
    else:
        # Convert minutes to hours
        hours_per_timestep = optimization_time_step / 60.0

    # Scale HDD to timestep duration (standard HDD is per 24 hours)
    hdd_per_timestep_scaled = hdd_per_timestep * (hours_per_timestep / 24.0)

    return specific_heating_demand * floor_area * (hdd_per_timestep_scaled / annual_reference_hdd)


def calculate_thermal_balance_physics(
    u_value: float,
    envelope_area: float,
    ventilation_rate: float,
    heated_volume: float,
    indoor_target_temperature: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    optimization_time_step: int,
    solar_irradiance_forecast: np.ndarray | pd.Series | None = None,
    window_area: float | None = None,
    shgc: float = 0.6,
    internal_gains_forecast: np.ndarray | pd.Series | None = None,
    internal_gains_factor: float = 0.0,
    sense: str = "heat",
) -> np.ndarray:
    """
    Calculate the signed physics-based thermal balance per timestep.

    Positive values mean the building loses heat and cools down. Negative values mean
    solar/internal gains exceed losses and the building warms up.

    :param u_value: Overall thermal transmittance (U-value) in W/(m²·K). Typical values:
        - 0.2-0.3: Well-insulated modern building
        - 0.4-0.6: Average insulation
        - 0.8-1.2: Poor insulation / old building
    :type u_value: float
    :param envelope_area: Total building envelope area (walls + roof + floor + windows) in m²
    :type envelope_area: float
    :param ventilation_rate: Air changes per hour (ACH). Typical values:
        - 0.3-0.5: Well-sealed modern building with controlled ventilation
        - 0.5-1.0: Average building
        - 1.0-2.0: Leaky old building
    :type ventilation_rate: float
    :param heated_volume: Total heated volume in m³
    :type heated_volume: float
    :param indoor_target_temperature: Target indoor temperature in °C
    :type indoor_target_temperature: float
    :param outdoor_temperature_forecast: Outdoor temperature forecast in °C for each timestep
    :type outdoor_temperature_forecast: np.ndarray | pd.Series
    :param optimization_time_step: Optimization time step in minutes
    :type optimization_time_step: int
    :param solar_irradiance_forecast: Global Horizontal Irradiance (GHI) in W/m² for each timestep.
        If provided along with window_area, solar gains will be subtracted from heating demand.
    :type solar_irradiance_forecast: np.ndarray | pd.Series | None, optional
    :param window_area: Total window area in m². If provided along with solar_irradiance_forecast,
        solar gains will reduce heating demand. Typical values: 15-25% of floor area.
    :type window_area: float | None, optional
    :param shgc: Solar Heat Gain Coefficient (dimensionless, 0-1). Fraction of solar radiation
        that becomes heat inside the building. Typical values:
        - 0.5-0.6: Modern low-e double-glazed windows
        - 0.6-0.7: Standard double-glazed windows
        - 0.7-0.8: Single-glazed windows
        Default: 0.6
    :type shgc: float, optional
    :param internal_gains_forecast: Electrical load power forecast in W for each timestep.
        If provided along with internal_gains_factor > 0, internal gains from electrical
        appliances will be subtracted from heating demand.
    :type internal_gains_forecast: np.ndarray | pd.Series | None, optional
    :param internal_gains_factor: Factor (0-1) representing what fraction of electrical load
        becomes useful internal heat gains. Typical values:
        - 0.0: No internal gains considered (default, backwards compatible)
        - 0.5-0.7: Conservative estimate (some heat lost to ventilation/drains)
        - 0.8-0.9: Most electrical energy becomes heat (well-insulated building)
        - 1.0: All electrical energy becomes internal heat (theoretical maximum)
        Default: 0.0
    :type internal_gains_factor: float, optional
    :param sense: Thermal mode, ``"heat"`` (default) or ``"cool"``. In heating mode the
        result is the heat the building loses (and must be replaced) when it is
        colder outside than the target, with solar and internal gains subtracted. In
        cooling mode the result is the heat the building gains when it is hotter outside
        than the target, with solar and internal gains added, returned as a SIGNED
        negative value (heat gain) to match the sign convention of
        :func:`calculate_thermal_loss_signed`.
    :type sense: str, optional
    :return: Array of signed thermal balance values (kWh) per timestep. Positive values
        are heat losses; negative values are net heat gains.
    :rtype: np.ndarray

    Example:
        >>> outdoor_temps = np.array([5, 8, 12, 15])
        >>> ghi = np.array([0, 100, 400, 600])  # W/m²
        >>> balance = calculate_thermal_balance_physics(
        ...     u_value=0.3,
        ...     envelope_area=400,
        ...     ventilation_rate=0.5,
        ...     heated_volume=250,
        ...     indoor_target_temperature=20,
        ...     outdoor_temperature_forecast=outdoor_temps,
        ...     optimization_time_step=30,
        ...     solar_irradiance_forecast=ghi,
        ...     window_area=50,
        ...     shgc=0.6
        ... )
    """

    # Convert outdoor temperature forecast to numpy array if pandas Series
    outdoor_temps = (
        outdoor_temperature_forecast.values
        if isinstance(outdoor_temperature_forecast, pd.Series)
        else np.asarray(outdoor_temperature_forecast)
    )

    # Normalize the thermal mode. Heating models the heat the building loses when
    # it is colder outside than the target; cooling models the heat it gains when it
    # is hotter outside. gain_sign folds the direction of solar/internal gains: in
    # heating they REDUCE the load (free heat), in cooling they ADD to it.
    sense = normalize_heat_cool_mode(sense, field_name="sense", context="thermal demand")
    gain_sign = -1.0 if sense == "cool" else 1.0

    # Temperature difference driving the envelope load (magnitude, per timestep).
    # Heating: load when outdoor < indoor. Cooling: load when outdoor > indoor.
    if sense == "cool":
        temp_diff = outdoor_temps - indoor_target_temperature
    else:
        temp_diff = indoor_target_temperature - outdoor_temps
    temp_diff = np.maximum(temp_diff, 0.0)

    # Transmission losses: Q_trans = U * A * ΔT (W to kW)
    transmission_loss_kw = u_value * envelope_area * temp_diff / 1000.0

    # Ventilation losses: Q_vent = V * ρ * c * n * ΔT / 3600
    # ρ = air density (kg/m³), c = specific heat capacity (kJ/(kg·K)), n = ACH
    air_density = 1.2  # kg/m³ at 20°C
    air_heat_capacity = 1.005  # kJ/(kg·K)
    ventilation_loss_kw = (
        ventilation_rate * heated_volume * air_density * air_heat_capacity * temp_diff / 3600.0
    )

    # Total heat loss in kW
    total_loss_kw = transmission_loss_kw + ventilation_loss_kw

    # Calculate solar gains if irradiance and window area are provided
    if solar_irradiance_forecast is not None and window_area is not None:
        # Convert solar irradiance to numpy array if pandas Series
        solar_irradiance = (
            solar_irradiance_forecast.values
            if isinstance(solar_irradiance_forecast, pd.Series)
            else np.asarray(solar_irradiance_forecast)
        )

        # Solar gains: Q_solar = window_area * SHGC * GHI (W to kW)
        # GHI is in W/m², so multiply by window_area (m²) gives W, then divide by 1000 for kW
        solar_gains_kw = window_area * shgc * solar_irradiance / 1000.0

        # Heating: subtract solar gains from heat loss.
        # Cooling: gain_sign flips this so solar gains ADD to the cooling load, which
        # is left unfloored because gains only ever deepen it.
        total_loss_kw = total_loss_kw - gain_sign * solar_gains_kw

    # Validate internal_gains_factor is in expected range [0, 1]
    if internal_gains_factor < 0 or internal_gains_factor > 1:
        raise ValueError(
            f"internal_gains_factor must be between 0 and 1, got {internal_gains_factor}"
        )

    # Calculate internal gains from electrical load if provided and applicable
    if internal_gains_forecast is not None and internal_gains_factor > 0:
        # Convert internal gains forecast to numpy array and normalize to 1D
        # to align with other forecast inputs and avoid broadcast surprises
        internal_gains = (
            internal_gains_forecast.values
            if isinstance(internal_gains_forecast, pd.Series)
            else internal_gains_forecast
        )
        internal_gains = np.asarray(internal_gains).reshape(-1)

        # Validate that internal gains forecast length matches outdoor temperature forecast
        if len(internal_gains) != len(outdoor_temps):
            raise ValueError(
                f"internal_gains_forecast length ({len(internal_gains)}) must match "
                f"outdoor_temperature_forecast length ({len(outdoor_temps)})"
            )

        # Warn if values seem like they might be in kW instead of W
        # Typical household load is 100-10000W; values below 10 suggest kW was passed
        max_load = np.max(internal_gains)
        if max_load > 0 and max_load < 10:
            import warnings

            warnings.warn(
                f"internal_gains_forecast max value ({max_load:.2f}) is very low. "
                "Expected values in W (e.g., 500-5000), but received values that "
                "look like kW. Please ensure you're passing Watts, not kilowatts.",
                UserWarning,
                stacklevel=2,
            )

        # Internal gains: Q_internal = load_power * factor
        # load_power is in W, convert to kW; factor is dimensionless (0-1)
        internal_gains_kw = internal_gains * internal_gains_factor / W_TO_KW

        # Heating: subtract internal gains from heat loss.
        # Cooling: gain_sign flips this so internal gains ADD to the cooling load.
        total_loss_kw = total_loss_kw - gain_sign * internal_gains_kw

    # Convert to kWh for the timestep. Cooling balance is returned as a signed heat
    # gain (<= 0) so the shared thermal state equation (which subtracts the demand
    # term) raises indoor temperature when it is hot, matching calculate_thermal_loss_signed.
    hours_per_timestep = optimization_time_step / 60.0
    balance = total_loss_kw * hours_per_timestep
    if sense == "cool":
        balance = -balance
    return balance


def calculate_heating_demand_physics(
    u_value: float,
    envelope_area: float,
    ventilation_rate: float,
    heated_volume: float,
    indoor_target_temperature: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    optimization_time_step: int,
    solar_irradiance_forecast: np.ndarray | pd.Series | None = None,
    window_area: float | None = None,
    shgc: float = 0.6,
    internal_gains_forecast: np.ndarray | pd.Series | None = None,
    internal_gains_factor: float = 0.0,
    sense: str = "heat",
) -> np.ndarray:
    """
    Calculate heating or cooling demand per timestep based on building physics heat loss model.

    This public demand helper preserves the historical reporting contract: heating
    demand is never negative. Use :func:`calculate_thermal_balance_physics` when the
    signed net balance is needed for indoor-temperature dynamics.
    """
    balance = calculate_thermal_balance_physics(
        u_value=u_value,
        envelope_area=envelope_area,
        ventilation_rate=ventilation_rate,
        heated_volume=heated_volume,
        indoor_target_temperature=indoor_target_temperature,
        outdoor_temperature_forecast=outdoor_temperature_forecast,
        optimization_time_step=optimization_time_step,
        solar_irradiance_forecast=solar_irradiance_forecast,
        window_area=window_area,
        shgc=shgc,
        internal_gains_forecast=internal_gains_forecast,
        internal_gains_factor=internal_gains_factor,
        sense=sense,
    )
    sense = normalize_heat_cool_mode(sense, field_name="sense", context="thermal demand")
    if sense == "cool":
        return balance
    return np.maximum(balance, 0.0)


def update_params_with_ha_config(
    params: str,
    ha_config: dict,
) -> dict:
    """
    Update the params with the Home Assistant configuration.

    Parameters
    ----------
    params : str
        The serialized params.
    ha_config : dict
        The Home Assistant configuration.

    Returns
    -------
    dict
        The updated params.
    """
    # Load serialized params
    params = orjson.loads(params)
    # Update params
    currency_to_symbol = {
        "EUR": "€",
        "USD": "$",
        "GBP": "£",
        "YEN": "¥",
        "JPY": "¥",
        "AUD": "A$",
        "CAD": "C$",
        "CHF": "CHF",  # Swiss Franc has no special symbol
        "CNY": "¥",
        "INR": "₹",
        "CZK": "Kč",
        "BGN": "лв",
        "DKK": "kr",
        "HUF": "Ft",
        "PLN": "zł",
        "RON": "Leu",
        "SEK": "kr",
        "TRY": "Lira",
        "VEF": "Bolivar",
        "VND": "Dong",
        "THB": "Baht",
        "SGD": "S$",
        "IDR": "Roepia",
        "ZAR": "Rand",
        # Add more as needed
    }
    if "currency" in ha_config.keys():
        ha_config["currency"] = currency_to_symbol.get(ha_config["currency"], "Unknown")
    else:
        ha_config["currency"] = "€"

    updated_passed_dict = {
        "custom_cost_fun_id": {
            "unit_of_measurement": ha_config["currency"],
        },
        "custom_unit_load_cost_id": {
            "unit_of_measurement": f"{ha_config['currency']}/kWh",
        },
        "custom_unit_prod_price_id": {
            "unit_of_measurement": f"{ha_config['currency']}/kWh",
        },
    }
    for key, value in updated_passed_dict.items():
        params["passed_data"][key]["unit_of_measurement"] = value["unit_of_measurement"]
    # Serialize the final params
    params = orjson.dumps(params, default=str).decode("utf-8")
    return params


def describe_invalid_forecast_value(forecast_key: str, items) -> str | None:
    """Describe the first forecast value that is not a finite real number (#1135).

    :param forecast_key: Name used in the diagnostic, e.g. ``load_power_forecast``.
    :param items: ``(location, value)`` pairs: ``enumerate(values)`` for a list \
        (reported as a position) or ``mapping.items()``/``series.items()`` \
        (reported as the timestamp).
    :return: One diagnostic for the first invalid value, or None if all are valid. \
        NaN, +/-Inf, booleans (``bool`` subclasses ``int``) and non-numeric \
        values are invalid; the sign of a value is not checked here.
    """
    for location, value in items:
        if isinstance(value, bool | np.bool_):
            reason = "boolean value"
        elif not isinstance(value, int | float | np.integer | np.floating):
            reason = "non-numeric value"
        elif isinstance(value, float | np.floating) and not math.isfinite(value):
            reason = "non-finite value"
        else:
            continue
        where = f"position {location}" if isinstance(location, int) else location
        return (
            f"{forecast_key} contains {reason} {value!r} at {where}; "
            "forecast values must be finite real numbers"
        )
    return None


def _align_runtime_forecast_mapping(
    forecast_input: dict,
    forecast_dates: list[str],
    optimization_time_step: int,
    time_zone,
) -> list[float]:
    """Align a timestamped runtime forecast onto the EMHASS forecast grid.

    This is the existing runtime mapping path extracted into one helper so the
    optional external PV P10 companion can use exactly the same resampling and
    hold-last semantics as ``pv_power_forecast``.
    """
    forecast_data_df = pd.DataFrame.from_dict(forecast_input, orient="index").reset_index()
    forecast_data_df.columns = ["time", "value"]
    forecast_data_df["time"] = pd.to_datetime(
        forecast_data_df["time"], format="ISO8601", utc=True
    ).dt.tz_convert(time_zone)

    target_dates = pd.to_datetime(forecast_dates, utc=True)

    # Aggregate any sub-step points to the optimization time step. The buckets are
    # anchored on the first grid instant, so each bucket is exactly one grid step wide
    # and coincides with a grid point whatever the zone offset (sub-hour offsets,
    # 30-minute DST changes).
    forecast_data_df = forecast_data_df.resample(
        pd.to_timedelta(optimization_time_step, "minutes"),
        on="time",
        origin=target_dates[0].tz_convert(time_zone) if len(target_dates) else "start_day",
    ).aggregate({"value": "mean"})

    # Align by instant in UTC across DST boundaries.
    forecast_data_df.index = forecast_data_df.index.tz_convert("UTC")

    # Existing runtime mapping semantics: each supplied point holds until the
    # next point. A point before the horizon anchors the first target step.
    combined_index = forecast_data_df.index.union(target_dates)
    forecast_data_df = forecast_data_df.reindex(combined_index)
    forecast_data_df["value"] = forecast_data_df["value"].ffill().bfill()
    forecast_data_df = forecast_data_df.reindex(target_dates)
    return forecast_data_df["value"].tolist()


def _validate_and_align_external_pv_pair(
    p50_input,
    p10_input,
    forecast_dates: list[str],
    optimization_time_step: int,
    time_zone,
) -> tuple[list | None, list | None, str | None]:
    """Validate and align an external PV P50/P10 pair on one source timeline."""
    if p50_input is None:
        return None, None, "pv_power_forecast_p10 requires pv_power_forecast"

    both_lists = isinstance(p50_input, list) and isinstance(p10_input, list)
    both_mappings = isinstance(p50_input, dict) and isinstance(p10_input, dict)
    if not (both_lists or both_mappings):
        return (
            None,
            None,
            "pv_power_forecast and pv_power_forecast_p10 must both be lists "
            "or both be timestamped mappings",
        )

    if both_lists:
        if len(p50_input) != len(p10_input):
            return (
                None,
                None,
                "pv_power_forecast/pv_power_forecast_p10 length mismatch: "
                f"{len(p50_input)}/{len(p10_input)}",
            )
        if len(p50_input) < len(forecast_dates):
            return (
                None,
                None,
                "pv_power_forecast/pv_power_forecast_p10 do not cover the full "
                f"forecast horizon: {len(p50_input)} values for "
                f"{len(forecast_dates)} required steps",
            )
        numeric_inputs = [p50_input, p10_input]
    else:
        if not p50_input or not p10_input:
            return None, None, "timestamped external PV P50/P10 mappings must not be empty"
        try:
            p50_index = pd.DatetimeIndex(
                pd.to_datetime(list(p50_input), format="ISO8601", utc=True)
            )
            p10_index = pd.DatetimeIndex(
                pd.to_datetime(list(p10_input), format="ISO8601", utc=True)
            )
        except (TypeError, ValueError) as exc:
            return None, None, f"invalid timestamp in external PV P50/P10 mapping: {exc}"
        if p50_index.has_duplicates or p10_index.has_duplicates:
            return (
                None,
                None,
                "external PV P50/P10 mappings contain duplicate timestamps after normalising to UTC",
            )
        if not p50_index.sort_values().equals(p10_index.sort_values()):
            return (
                None,
                None,
                "pv_power_forecast and pv_power_forecast_p10 timestamp mismatch; "
                "both mappings must describe the same forecast timeline",
            )
        numeric_inputs = [list(p50_input.values()), list(p10_input.values())]

    # #1135: validate the original values before NumPy/pandas can coerce
    # booleans or numeric-looking strings into floats. This is the same
    # finite-real contract used by the ordinary runtime forecast path.
    for key, source in (("pv_power_forecast", p50_input), ("pv_power_forecast_p10", p10_input)):
        items = source.items() if both_mappings else enumerate(source)
        invalid = describe_invalid_forecast_value(key, items)
        if invalid is not None:
            return None, None, invalid

    try:
        numeric = np.asarray(numeric_inputs, dtype=float)
    except (TypeError, ValueError):
        return None, None, "external PV P50/P10 must contain only numeric values"
    if numeric.ndim != 2 or numeric.shape[0] != 2:
        return None, None, "external PV P50/P10 must be one-dimensional series"
    if not np.isfinite(numeric).all():
        return None, None, "external PV P50/P10 contains non-finite values"

    if both_lists:
        horizon = len(forecast_dates)
        return p50_input[:horizon], p10_input[:horizon], None

    aligned_p50 = _align_runtime_forecast_mapping(
        dict(zip(p50_input, numeric[0])),
        forecast_dates,
        optimization_time_step,
        time_zone,
    )
    aligned_p10 = _align_runtime_forecast_mapping(
        dict(zip(p10_input, numeric[1])),
        forecast_dates,
        optimization_time_step,
        time_zone,
    )
    for key, values in (
        ("pv_power_forecast", aligned_p50),
        ("pv_power_forecast_p10", aligned_p10),
    ):
        invalid = describe_invalid_forecast_value(key, enumerate(values))
        if invalid is not None:
            return None, None, f"{invalid} after timestamp alignment"
    return aligned_p50, aligned_p10, None


# Runtime PV forecasts per inverter (#1150). One inverter takes one flat list (or one
# timestamped mapping). Above one inverter the payload is a list with one entry per
# inverter, an empty one for an inverter with no PV plant. Every check compares lengths
# and walks the payload or the plant index list; nothing is built from the inverter count.

# Longest diagnostic returned for a per-inverter payload.
_PV_PER_INVERTER_MESSAGE_LIMIT = 400


def _bounded_fault(message: str) -> str:
    """Cap a diagnostic that quotes payload text, so a huge value cannot make it huge."""
    if len(message) <= _PV_PER_INVERTER_MESSAGE_LIMIT:
        return message
    return message[: _PV_PER_INVERTER_MESSAGE_LIMIT - 3] + "..."


def describe_nested_pv_payload_at_one_inverter(forecast_key: str, forecast_input) -> str | None:
    """Describe a nested PV forecast payload sent for a plant with one inverter (#1150).

    With ``number_of_inverters`` at 1 the forecast is one flat list, or one timestamped
    mapping. A list that holds a list or a mapping is the per-inverter shape. It is only
    read with more than one inverter, so with one inverter it is refused instead of being
    planned with another forecast method.

    :param forecast_key: Name used in the diagnostic, e.g. ``pv_power_forecast``.
    :param forecast_input: The runtime value, after a stringified list was parsed.
    :return: A bounded diagnostic, or None when the payload is not a nested list.
    """
    if not isinstance(forecast_input, list):
        return None
    if not any(isinstance(element, list | dict) for element in forecast_input):
        return None
    return (
        f"{forecast_key}: send one flat list of numbers (or one timestamped mapping). It holds "
        "lists or mappings, one per inverter, but number_of_inverters is 1. A list per "
        "inverter is only accepted when number_of_inverters is above 1."
    )


def inverter_curtailment_flags(plant_conf: dict) -> list[bool]:
    """The curtailment switch of each inverter, read after validation (#1150).

    A scalar or a missing value gives a one-element list holding its truthiness, which is
    what the single ``compute_curtailment`` flag has always meant. A list gives one flag
    per entry, coerced as the validation does (a bool, null as false, "true" or "false" in
    any case). An entry that cannot be coerced counts as false instead of raising, since
    this is read after the configuration was checked.

    :param plant_conf: The plant_conf dict
    :return: One flag per inverter for a list, otherwise a single flag
    """
    value = plant_conf.get("compute_curtailment", False)
    if not isinstance(value, list):
        return [bool(value)]
    flags: list[bool] = []
    for element in value:
        try:
            flags.append(_coerce_inverter_flag(element, "compute_curtailment"))
        except InverterConfigError:
            flags.append(False)
    return flags


def any_inverter_curtails(plant_conf: dict) -> bool:
    """Whether any inverter has curtailment switched on (#1150).

    For a scalar or missing ``compute_curtailment`` this is its plain truthiness, as
    before. A list is not tested for truthiness, since ``[False, False]`` is truthy.

    :param plant_conf: The plant_conf dict
    :return: True when at least one inverter curtails
    """
    return any(inverter_curtailment_flags(plant_conf))


def pv_carrying_inverters(plant_conf: dict) -> list[int]:
    """The inverters that carry a PV plant, in ascending order (#1150).

    These are the distinct values of ``pv_plant_inverter_index``, so an index that lists an
    inverter more than once (two pvlib plants on one inverter, ``[0, 0, 1]``) gives
    ``[0, 1]``. Only meaningful above one inverter: with one inverter the index is not read
    and the result is empty.

    :param plant_conf: The plant_conf dict
    :return: Sorted distinct inverter numbers
    :raises InverterConfigError: If the index is not a list of inverter numbers in range.
        The message names the position and never echoes the value, so it stays short.
    """
    num_inverters = validate_num_inverters(plant_conf)
    if num_inverters <= 1:
        return []
    index = plant_conf.get("pv_plant_inverter_index")
    if not isinstance(index, list):
        raise InverterConfigError(
            "pv_plant_inverter_index must be a list with one inverter number per PV plant "
            f"when number_of_inverters is above 1, got {type(index).__name__}"
        )
    found: set[int] = set()
    for position, value in enumerate(index):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float | np.integer)
            or (isinstance(value, float) and not value.is_integer())
        ):
            raise InverterConfigError(
                f"pv_plant_inverter_index[{position}] must be a whole inverter number, "
                f"got a {type(value).__name__}"
            )
        number = int(value)
        if not 0 <= number < num_inverters:
            raise InverterConfigError(
                f"pv_plant_inverter_index[{position}] is not an inverter number from 0 to "
                f"{num_inverters - 1}"
            )
        found.add(number)
    return sorted(found)


def _per_inverter_shape_fault(
    forecast_key: str, payload, n_inverters: int, pv_inverters: Sequence[int]
) -> str | None:
    """The first way a per-inverter PV payload does not fit the configured topology.

    The outer length is compared before anything else is read. Each entry is a list or a
    mapping; it is empty for an inverter with no PV plant and not empty for one with a PV
    plant; the non-empty entries are all lists or all mappings.
    """
    if not isinstance(payload, list):
        return (
            f"{forecast_key} must be a list with one entry per inverter when "
            f"number_of_inverters is {n_inverters}, got {type(payload).__name__}: a flat "
            "list or a single mapping is not accepted"
        )
    if len(payload) != n_inverters:
        return (
            f"{forecast_key} has {len(payload)} entries but number_of_inverters is "
            f"{n_inverters}: send one list (or one timestamped mapping) per inverter, "
            "empty for an inverter with no PV plant"
        )
    pv_set = set(pv_inverters)
    kind = None
    for inverter, entry in enumerate(payload):
        if isinstance(entry, list):
            entry_kind = "list"
        elif isinstance(entry, dict):
            entry_kind = "mapping"
        else:
            return (
                f"{forecast_key}[{inverter}] is a {type(entry).__name__}: with "
                f"number_of_inverters at {n_inverters} every entry must be a list or a "
                "timestamped mapping (a flat list of numbers is not accepted)"
            )
        if inverter in pv_set and not entry:
            return f"{forecast_key}[{inverter}] is empty but inverter {inverter} has a PV plant"
        if inverter not in pv_set and entry:
            return (
                f"{forecast_key}[{inverter}] has {len(entry)} values but inverter {inverter} "
                "has no PV plant: send an empty list for it"
            )
        if entry:
            kind = kind or entry_kind
            if kind != entry_kind:
                return f"{forecast_key} mixes lists and timestamped mappings across inverters"
    return None


def _align_one_inverter_pv_series(
    label: str,
    entry,
    forecast_dates: list[str],
    optimization_time_step: int,
    time_zone,
) -> tuple[list | None, str | None]:
    """Check one inverter's list or mapping and return it on the forecast grid."""
    horizon = len(forecast_dates)
    if isinstance(entry, dict):
        invalid = describe_invalid_forecast_value(label, entry.items())
        if invalid is not None:
            return None, invalid
        try:
            values = _align_runtime_forecast_mapping(
                entry, forecast_dates, optimization_time_step, time_zone
            )
        except (TypeError, ValueError) as exc:
            return None, f"{label} has an invalid timestamp: {exc}"
        invalid = describe_invalid_forecast_value(label, enumerate(values))
        if invalid is not None:
            return None, f"{invalid} after timestamp alignment"
        return values, None
    if len(entry) < horizon:
        return None, f"{label} has {len(entry)} values for {horizon} required steps"
    invalid = describe_invalid_forecast_value(label, enumerate(entry))
    if invalid is not None:
        return None, invalid
    return entry[:horizon], None


def validate_and_align_per_inverter_pv_forecast(
    payload,
    n_inverters: int,
    pv_inverters: Sequence[int],
    forecast_dates: list[str],
    optimization_time_step: int,
    time_zone,
    forecast_key: str = "pv_power_forecast",
) -> tuple[list[list] | None, str | None]:
    """Validate and align a runtime PV forecast with one entry per inverter (#1150).

    The payload is a list of exactly ``n_inverters`` entries, each a list of values or a
    timestamped mapping (not a mix). An inverter with no PV plant gets an empty entry; an
    inverter in ``pv_inverters`` gets a non-empty one with at least one value per forecast
    step. Every value goes through the finite-number check of the flat path.

    :param payload: The runtime value
    :param n_inverters: ``number_of_inverters``, only compared with the payload length
    :param pv_inverters: The PV-carrying inverters, see :func:`pv_carrying_inverters`
    :param forecast_dates: The forecast grid
    :param optimization_time_step: The step in minutes
    :param time_zone: The time zone of the grid
    :param forecast_key: Name used in the diagnostics
    :return: ``(aligned, None)`` with one list per inverter (empty for an inverter with no
        PV plant), or ``(None, fault)`` with a bounded diagnostic
    """
    fault = _per_inverter_shape_fault(forecast_key, payload, n_inverters, pv_inverters)
    if fault is not None:
        return None, fault
    pv_set = set(pv_inverters)
    aligned: list[list] = []
    for inverter, entry in enumerate(payload):
        if inverter not in pv_set:
            aligned.append([])
            continue
        values, fault = _align_one_inverter_pv_series(
            f"{forecast_key}[{inverter}]",
            entry,
            forecast_dates,
            optimization_time_step,
            time_zone,
        )
        if fault is not None:
            return None, _bounded_fault(fault)
        aligned.append(values)
    return aligned, None


def validate_and_align_per_inverter_pv_pair(
    p50_payload,
    p10_payload,
    n_inverters: int,
    pv_inverters: Sequence[int],
    forecast_dates: list[str],
    optimization_time_step: int,
    time_zone,
) -> tuple[list[list] | None, list[list] | None, str | None]:
    """Validate and align a per-inverter PV P50/P10 pair (#1150).

    Both payloads have the shape of :func:`validate_and_align_per_inverter_pv_forecast`
    with the same empty and non-empty entries, and each inverter's P50 and P10 are checked
    as a pair by :func:`_validate_and_align_external_pv_pair`.

    :return: ``(p50, p10, None)`` with one list per inverter, or ``(None, None, fault)`` with a
        bounded diagnostic
    """
    if p50_payload is None:
        return None, None, "pv_power_forecast_p10 requires pv_power_forecast"
    for key, payload in (
        ("pv_power_forecast", p50_payload),
        ("pv_power_forecast_p10", p10_payload),
    ):
        fault = _per_inverter_shape_fault(key, payload, n_inverters, pv_inverters)
        if fault is not None:
            return None, None, fault
    pv_set = set(pv_inverters)
    aligned_p50: list[list] = []
    aligned_p10: list[list] = []
    for inverter, (p50_entry, p10_entry) in enumerate(zip(p50_payload, p10_payload)):
        if inverter not in pv_set:
            aligned_p50.append([])
            aligned_p10.append([])
            continue
        p50_values, p10_values, fault = _validate_and_align_external_pv_pair(
            p50_entry, p10_entry, forecast_dates, optimization_time_step, time_zone
        )
        if fault is not None:
            return None, None, _bounded_fault(f"inverter {inverter}: {fault}")
        aligned_p50.append(p50_values)
        aligned_p10.append(p10_values)
    return aligned_p50, aligned_p10, None


_LOAD_PUBLISH_ID_KEYS = (
    "custom_deferrable_forecast_id",
    "custom_deferrable_state_id",
    "custom_predicted_temperature_id",
    "custom_heating_demand_id",
)


def _default_load_publish_ids(k: int, temperature_unit: str = "°C") -> dict[str, dict]:
    """Default Home Assistant entities for deferrable load ``k``, keyed by the
    passed_data id list each belongs to."""
    return {
        "custom_deferrable_forecast_id": {
            "entity_id": f"sensor.p_deferrable{k}",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": f"Deferrable Load {k}",
        },
        "custom_deferrable_state_id": {
            "entity_id": f"sensor.p_deferrable{k}_state",
            "device_class": "enum",
            "unit_of_measurement": "",
            "friendly_name": f"Deferrable Load {k} Command",
        },
        "custom_predicted_temperature_id": {
            "entity_id": f"sensor.temp_predicted{k}",
            "device_class": "temperature",
            "unit_of_measurement": temperature_unit,
            "friendly_name": f"Predicted temperature {k}",
        },
        "custom_heating_demand_id": {
            "entity_id": f"sensor.heating_demand{k}",
            "device_class": "energy",
            "unit_of_measurement": "kWh",
            "friendly_name": f"Heating demand {k}",
        },
    }


async def treat_runtimeparams(
    runtimeparams: str,
    params: dict[str, dict],
    retrieve_hass_conf: dict[str, str],
    optim_conf: dict[str, str],
    plant_conf: dict[str, str],
    set_type: str,
    logger: logging.Logger,
    emhass_conf: dict[str, pathlib.Path],
) -> tuple[str, dict[str, dict]]:
    """
    Treat the passed optimization runtime parameters.

    :param runtimeparams: Json string containing the runtime parameters dict.
    :type runtimeparams: str
    :param params: Built configuration parameters
    :type params: str
    :param retrieve_hass_conf: Config dictionary for data retrieving parameters.
    :type retrieve_hass_conf: dict
    :param optim_conf: Config dictionary for optimization parameters.
    :type optim_conf: dict
    :param plant_conf: Config dictionary for technical plant parameters.
    :type plant_conf: dict
    :param set_type: The type of action to be performed.
    :type set_type: str
    :param logger: The logger object.
    :type logger: logging.Logger
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :return: Returning the params and optimization parameter container.
    :rtype: Tuple[str, dict]

    """
    # Check if passed params is a dict
    if (params is not None) and (params != "null"):
        if type(params) is str:
            params = orjson.loads(params)
    else:
        params = {}

    # Merge current config categories to params
    params["retrieve_hass_conf"].update(retrieve_hass_conf)
    params["optim_conf"].update(optim_conf)
    params["plant_conf"].update(plant_conf)

    # Check defaults on HA retrieved config
    default_currency_unit = "€"
    default_temperature_unit = "°C"

    # Some default data needed
    load_ids = [
        _default_load_publish_ids(k, default_temperature_unit)
        for k in range(params["optim_conf"]["number_of_deferrable_loads"])
    ]
    custom_deferrable_forecast_id = [ids["custom_deferrable_forecast_id"] for ids in load_ids]
    custom_deferrable_state_id = [ids["custom_deferrable_state_id"] for ids in load_ids]
    custom_predicted_temperature_id = [ids["custom_predicted_temperature_id"] for ids in load_ids]
    custom_heating_demand_id = [ids["custom_heating_demand_id"] for ids in load_ids]
    default_passed_dict = {
        "custom_pv_forecast_id": {
            "entity_id": "sensor.p_pv_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "PV Power Forecast",
        },
        "custom_load_forecast_id": {
            "entity_id": "sensor.p_load_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Load Power Forecast",
        },
        "custom_pv_curtailment_id": {
            "entity_id": "sensor.p_pv_curtailment",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "PV Power Curtailment",
        },
        "custom_hybrid_inverter_id": {
            "entity_id": "sensor.p_hybrid_inverter",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "PV Hybrid Inverter",
        },
        "custom_batt_forecast_id": {
            "entity_id": "sensor.p_batt_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Battery Power Forecast",
        },
        "custom_batt_soc_forecast_id": {
            "entity_id": "sensor.soc_batt_forecast",
            "device_class": "battery",
            "unit_of_measurement": "%",
            "friendly_name": "Battery SOC Forecast",
        },
        "custom_grid_forecast_id": {
            "entity_id": "sensor.p_grid_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Grid Power Forecast",
        },
        "custom_cost_fun_id": {
            "entity_id": "sensor.total_cost_fun_value",
            "device_class": "monetary",
            "unit_of_measurement": default_currency_unit,
            "friendly_name": "Total cost function value",
        },
        "custom_optim_status_id": {
            "entity_id": "sensor.optim_status",
            "device_class": "",
            "unit_of_measurement": "",
            "friendly_name": "EMHASS optimization status",
        },
        "custom_unit_load_cost_id": {
            "entity_id": "sensor.unit_load_cost",
            "device_class": "monetary",
            "unit_of_measurement": f"{default_currency_unit}/kWh",
            "friendly_name": "Unit Load Cost",
        },
        "custom_unit_prod_price_id": {
            "entity_id": "sensor.unit_prod_price",
            "device_class": "monetary",
            "unit_of_measurement": f"{default_currency_unit}/kWh",
            "friendly_name": "Unit Prod Price",
        },
        "custom_deferrable_forecast_id": custom_deferrable_forecast_id,
        "custom_deferrable_state_id": custom_deferrable_state_id,
        "custom_predicted_temperature_id": custom_predicted_temperature_id,
        "custom_heating_demand_id": custom_heating_demand_id,
        "publish_prefix": "",
    }
    if "passed_data" in params.keys():
        for key, value in default_passed_dict.items():
            params["passed_data"][key] = value
    else:
        params["passed_data"] = default_passed_dict
    # Only set_input_data_dict's dry_run argument, which only the dry-run URL
    # passes, makes a dry run. A dry_run left in saved or passed params must
    # not turn a live run into one that writes nothing.
    params["passed_data"].pop("dry_run", None)

    # Capture defaults for power limits before association loop
    power_limit_defaults = {
        "maximum_power_from_grid": params["plant_conf"].get("maximum_power_from_grid"),
        "maximum_power_to_grid": params["plant_conf"].get("maximum_power_to_grid"),
    }

    # Set once the runtime block below has validated the inverter parameters (#1150).
    inverters_checked = False

    # If any runtime parameters where passed in action call
    if runtimeparams is not None:
        if type(runtimeparams) is str:
            runtimeparams = orjson.loads(runtimeparams)

        # Remember which per-battery array params were configured with genuinely
        # distinct per-battery entries: the association loop below overwrites the
        # configured value with the raw runtime one, and the re-normalisation
        # step further down warns when a runtime scalar masks such a list
        # (its broadcast silently flattens the per-battery values, #1032).
        batt_distinct_config_lists: dict[str, list] = {}
        for batt_conf_key, batt_param_names in (
            ("plant_conf", BATT_ARRAY_PARAMS_PLANT_CONF),
            ("optim_conf", BATT_ARRAY_PARAMS_OPTIM_CONF),
        ):
            for batt_param_name in batt_param_names:
                configured = params[batt_conf_key].get(batt_param_name)
                if isinstance(configured, list) and any(
                    element != configured[0] for element in configured[1:]
                ):
                    batt_distinct_config_lists[batt_param_name] = list(configured)

        # The same for the per-inverter broadcast params (#1150).
        inverter_distinct_config_lists: dict[str, list] = {}
        for inverter_param_name in INVERTER_BROADCAST_PARAMS:
            configured = params["plant_conf"].get(inverter_param_name)
            if isinstance(configured, list) and any(
                element != configured[0] for element in configured[1:]
            ):
                inverter_distinct_config_lists[inverter_param_name] = list(configured)

        # The configured shared tanks, before the association loop below copies a
        # runtime list over them unchecked: a runtime list that the shared-tank
        # validation further down rejects falls back to these.
        configured_shared_tanks = params["optim_conf"].get("shared_thermal_tanks")
        # Loop though parameters stored in association file, Check to see if any stored in runtime
        # If true, set runtime parameter to params
        if emhass_conf["associations_path"].exists():
            async with aiofiles.open(emhass_conf["associations_path"]) as data:
                content = await data.read()
                associations = list(csv.reader(content.splitlines(), delimiter=","))
                # Association file key reference
                # association[0] = config categories
                # association[1] = legacy parameter name
                # association[2] = parameter (config.json/config_defaults.json)
                # association[3] = parameter list name if exists (not used, from legacy options.json)
                # A request's site is checked by shape before anything reads it,
                # and may not name a remote solver: no request makes EMHASS call
                # a host of its choosing.
                if runtimeparams.get("site") is not None:
                    problem = site_problem(runtimeparams["site"], from_request=True)
                    if problem:
                        logger.error(
                            "site (in the request): %s; the configured site is kept", problem
                        )
                        runtimeparams = {k: v for k, v in runtimeparams.items() if k != "site"}
                for association in associations:
                    # Connection settings and credentials are configuration-only
                    if (
                        association[2] in RUNTIME_DENIED_KEYS
                        or association[1] in RUNTIME_DENIED_KEYS
                    ):
                        if (
                            runtimeparams.get(association[2], None) is not None
                            or runtimeparams.get(association[1], None) is not None
                        ):
                            logger.warning(
                                "Runtime parameter %s ignored: it can only be set in the configuration",
                                association[2],
                            )
                        continue
                    # Check parameter name exists in runtime
                    if runtimeparams.get(association[2], None) is not None:
                        params[association[0]][association[2]] = runtimeparams[association[2]]
                    # Check Legacy parameter name runtime
                    elif runtimeparams.get(association[1], None) is not None:
                        params[association[0]][association[2]] = runtimeparams[association[1]]
        else:
            logger.warning(
                "Cant find associations file (associations.csv) in: "
                + str(emhass_conf["associations_path"])
            )

        # Special handling for power limit parameters - they can be vectors (Tier 1a)
        def _parse_power_limit(key: str) -> None:
            """Helper to parse list/scalar power limits safely."""
            if key in runtimeparams:
                value = runtimeparams[key]
                try:
                    # If it's a string representation of a list, parse it
                    if isinstance(value, str):
                        parsed = ast.literal_eval(value)
                        params["plant_conf"][key] = parsed
                    # If already a list/array, use it directly
                    # Ruff preferred bitwise OR '|' for union types
                    elif isinstance(value, list | tuple):
                        params["plant_conf"][key] = list(value)
                    # If scalar, use as-is
                    else:
                        params["plant_conf"][key] = value
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Could not parse {key}: {e}. Using default.")
                    if power_limit_defaults.get(key) is not None:
                        params["plant_conf"][key] = power_limit_defaults[key]

        # Apply the helper
        _parse_power_limit("maximum_power_from_grid")
        _parse_power_limit("maximum_power_to_grid")

        # Re-normalise per-battery array params after the generic association
        # loop above may have overwritten them with a raw runtime value (#610).
        # N=1 is a no-op (see check_batt_params); this only does real work once
        # number_of_batteries > 1.
        num_batteries = require_valid_battery_count(params["plant_conf"], logger)
        for batt_param_name, batt_default in BATT_ARRAY_PARAMS_PLANT_CONF.items():
            _warn_if_runtime_scalar_masks_batt_list(
                num_batteries,
                params["plant_conf"],
                batt_param_name,
                batt_distinct_config_lists,
                logger,
            )
            check_batt_params(
                num_batteries, params["plant_conf"], batt_default, batt_param_name, logger
            )
        for batt_param_name, batt_default in BATT_ARRAY_PARAMS_OPTIM_CONF.items():
            _warn_if_runtime_scalar_masks_batt_list(
                num_batteries,
                params["optim_conf"],
                batt_param_name,
                batt_distinct_config_lists,
                logger,
            )
            check_batt_params(
                num_batteries, params["optim_conf"], batt_default, batt_param_name, logger
            )
        for batt_param_name in BATT_WEIGHT_PARAMS:
            check_batt_weight_params(num_batteries, params["optim_conf"], batt_param_name, logger)
        check_batt_charge_derating(
            num_batteries, params["plant_conf"], "battery_charge_power_derating", logger
        )
        # Per-inverter parameters (#1150): the same re-normalisation after the runtime
        # merge, then every specific fault is reported together. Before the curve
        # check, which reads the normalised caps.
        try:
            runtime_num_inverters = validate_num_inverters(params["plant_conf"])
        except InverterConfigError:
            runtime_num_inverters = 0  # require_usable_inverter_config reports the count
        if runtime_num_inverters > 1:
            for inverter_param_name in INVERTER_BROADCAST_PARAMS:
                _warn_if_runtime_scalar_masks_inverter_list(
                    runtime_num_inverters,
                    params["plant_conf"],
                    inverter_param_name,
                    inverter_distinct_config_lists,
                    logger,
                )
        # A request (or saved) site that turns the plant hybrid must fill
        # plant_conf before the curve check, which otherwise refuses a
        # supplied curve while inverter_is_hybrid is still off.
        site = params["optim_conf"].get("site")
        if site:
            problem = site_problem(site)
            if problem:
                logger.error("site: %s; not used", problem)
            else:
                compile_site(
                    site,
                    params["plant_conf"],
                    logger,
                    bool(params["optim_conf"].get("set_use_battery")),
                )
        require_usable_inverter_config(params["plant_conf"], logger)
        inverters_checked = True
        # Strict: a curve in the request must be valid and usable (no recovery).
        check_inverter_power_curves(
            params["plant_conf"],
            logger,
            supplied=[name for name, *_ in _INVERTER_CURVE_PARAMS if name in runtimeparams],
        )

        # Generate forecast_dates
        # Force update optimization_time_step if present in runtimeparams
        if "optimization_time_step" in runtimeparams:
            optimization_time_step = int(runtimeparams["optimization_time_step"])
            params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
                optimization_time_step, "minutes"
            )
        elif "freq" in runtimeparams:
            optimization_time_step = int(runtimeparams["freq"])
            params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
                optimization_time_step, "minutes"
            )
        else:
            optimization_time_step = int(
                params["retrieve_hass_conf"]["optimization_time_step"].seconds / 60.0
            )

        if (
            runtimeparams.get("delta_forecast_daily", None) is not None
            or runtimeparams.get("delta_forecast", None) is not None
        ):
            # Use old param name delta_forecast (if provided) for backwards compatibility
            delta_forecast = runtimeparams.get("delta_forecast", None)
            # Prefer new param name delta_forecast_daily
            delta_forecast = runtimeparams.get("delta_forecast_daily", delta_forecast)
            # Ensure delta_forecast is numeric and at least 1 day
            if delta_forecast is None:
                logger.warning("delta_forecast_daily is missing so defaulting to 1 day")
                delta_forecast = 1
            else:
                try:
                    delta_forecast = int(delta_forecast)
                except ValueError:
                    logger.warning(
                        "Invalid delta_forecast_daily value (%s) so defaulting to 1 day",
                        delta_forecast,
                    )
                    delta_forecast = 1
            if delta_forecast <= 0:
                logger.warning(
                    "delta_forecast_daily is too low (%s) so defaulting to 1 day",
                    delta_forecast,
                )
                delta_forecast = 1
            params["optim_conf"]["delta_forecast_daily"] = pd.Timedelta(days=delta_forecast)
        else:
            delta_forecast = int(params["optim_conf"]["delta_forecast_daily"].days)
        if runtimeparams.get("time_zone", None) is not None:
            time_zone = pytz.timezone(params["retrieve_hass_conf"]["time_zone"])
            params["retrieve_hass_conf"]["time_zone"] = time_zone
        else:
            time_zone = params["retrieve_hass_conf"]["time_zone"]

        # Same rounding as Forecast.__init__ so runtime forecasts are aligned to the grid
        # the optimizer will actually use.
        method_ts_round = params["retrieve_hass_conf"].get("method_ts_round", "nearest")
        if method_ts_round not in ("nearest", "first", "last"):
            logger.error("Wrong method_ts_round passed parameter: %r", method_ts_round)
            method_ts_round = "nearest"  # actions that build a Forecast still fail on it
        # The request clock is read here, once, and used for every grid of this request.
        # Forecast reuses the captured start instead of reading the clock again, so the
        # runtime forecasts and the optimizer share one grid even when the request crosses a
        # time-step boundary.
        request_now = _get_now()
        forecast_dates = get_forecast_dates(
            optimization_time_step,
            delta_forecast,
            time_zone,
            method_ts_round=method_ts_round,
            now=request_now,
        )
        params["passed_data"]["forecast_start"] = forecast_dates[0]

        # Add runtime exclusive (not in config) parameters to params
        # regressor-model-fit
        if set_type == "regressor-model-fit":
            if "csv_file" in runtimeparams:
                csv_file = validate_runtime_file_name("csv_file", runtimeparams["csv_file"])
                params["passed_data"]["csv_file"] = csv_file
            if "features" in runtimeparams:
                features = runtimeparams["features"]
                params["passed_data"]["features"] = features
            if "target" in runtimeparams:
                target = runtimeparams["target"]
                params["passed_data"]["target"] = target
            if "timestamp" not in runtimeparams:
                params["passed_data"]["timestamp"] = None
            else:
                timestamp = runtimeparams["timestamp"]
                params["passed_data"]["timestamp"] = timestamp
            if "date_features" not in runtimeparams:
                params["passed_data"]["date_features"] = []
            else:
                date_features = runtimeparams["date_features"]
                params["passed_data"]["date_features"] = date_features

        # regressor-model-predict
        if set_type == "regressor-model-predict":
            if "new_values" in runtimeparams:
                new_values = runtimeparams["new_values"]
                params["passed_data"]["new_values"] = new_values
            if "csv_file" in runtimeparams:
                csv_file = validate_runtime_file_name("csv_file", runtimeparams["csv_file"])
                params["passed_data"]["csv_file"] = csv_file
            if "features" in runtimeparams:
                features = runtimeparams["features"]
                params["passed_data"]["features"] = features
            if "target" in runtimeparams:
                target = runtimeparams["target"]
                params["passed_data"]["target"] = target

        # export-influxdb-to-csv
        if set_type == "export-influxdb-to-csv":
            # Use dictionary comprehension to simplify parameter assignment
            export_keys = {
                k: runtimeparams[k]
                for k in (
                    "sensor_list",
                    "csv_filename",
                    "start_time",
                    "end_time",
                    "resample_freq",
                    "timestamp_col_name",
                    "decimal_places",
                    "handle_nan",
                )
                if k in runtimeparams
            }
            if "csv_filename" in export_keys:
                validate_runtime_file_name("csv_filename", export_keys["csv_filename"], ".csv")
            if "timestamp_col_name" in export_keys:
                validate_timestamp_col_name(export_keys["timestamp_col_name"])
            params["passed_data"].update(export_keys)

        # MPC control case
        if set_type == "naive-mpc-optim":
            if (
                "prediction_horizon" not in runtimeparams.keys()
                or runtimeparams["prediction_horizon"] is None
            ):
                prediction_horizon = 10  # 10 time steps by default
            else:
                prediction_horizon = runtimeparams["prediction_horizon"]
            params["passed_data"]["prediction_horizon"] = prediction_horizon
            # Auto-extend the forecast window to cover the requested MPC horizon.
            # Forecast days are local calendar days, so their timestep count is not
            # always 24h / optimization_time_step: a DST spring-forward day is shorter
            # and a fall-back day is longer. Compare the requested timestep horizon
            # against the actual timezone-aware grid instead of assuming a fixed number
            # of steps per day. Grow by whole local calendar days until the grid covers
            # the request; downstream slicing still trims it to prediction_horizon.
            required_delta_forecast = delta_forecast
            extended_forecast_dates = forecast_dates
            while len(extended_forecast_dates) < int(prediction_horizon):
                required_delta_forecast += 1
                extended_forecast_dates = get_forecast_dates(
                    optimization_time_step,
                    required_delta_forecast,
                    time_zone,
                    method_ts_round=method_ts_round,
                    now=request_now,
                )
            if required_delta_forecast > delta_forecast:
                logger.info(
                    "naive-mpc prediction_horizon=%s exceeds the actual %s-day forecast "
                    "grid (%s timestep(s)); extending delta_forecast_daily to %s day(s) "
                    "to cover the full horizon and its weather forecast.",
                    prediction_horizon,
                    delta_forecast,
                    len(forecast_dates),
                    required_delta_forecast,
                )
                delta_forecast = required_delta_forecast
                params["optim_conf"]["delta_forecast_daily"] = pd.Timedelta(days=delta_forecast)
                forecast_dates = extended_forecast_dates
            reject_configured_soc_fractions(params["plant_conf"], params["optim_conf"], logger)
            num_batteries = require_valid_battery_count(params["plant_conf"], logger)
            if num_batteries == 1:
                # Unchanged from before #610: soc_init/soc_final stay plain
                # floats, not [x] lists, so downstream consumers written for
                # the single-battery model (and existing tests pinning this
                # exact shape) are untouched.
                if "soc_init" not in runtimeparams.keys():
                    soc_init = params["plant_conf"]["battery_target_state_of_charge"]
                else:
                    soc_init = runtimeparams["soc_init"]
                    if isinstance(soc_init, list):
                        # A naive-mpc runtime list is handled symmetrically
                        # with the num_batteries>1 branch below
                        # (_resolve_soc_runtime_list) and with the dayahead
                        # passthrough (_passthrough_soc_runtime): a length-1
                        # list unwraps to its scalar, any other length is the
                        # same clear ValueError the N>1 path raises, not an
                        # opaque TypeError from comparing a list to a float.
                        if len(soc_init) != 1:
                            raise ValueError(
                                f"soc_init has {len(soc_init)} entries but "
                                f"number_of_batteries=1; provide a scalar (broadcast "
                                f"to every battery) or a list of exactly 1 entry"
                            )
                        soc_init = soc_init[0]
                reject_soc_fraction_above_one("soc_init", soc_init, logger)
                if soc_init < params["plant_conf"]["battery_minimum_state_of_charge"]:
                    logger.warning(
                        f"Passed soc_init={soc_init} is lower than soc_min={params['plant_conf']['battery_minimum_state_of_charge']}, keeping real initial SOC for optimization recovery"
                    )
                if soc_init > params["plant_conf"]["battery_maximum_state_of_charge"]:
                    logger.warning(
                        f"Passed soc_init={soc_init} is greater than soc_max={params['plant_conf']['battery_maximum_state_of_charge']}, keeping real initial SOC for optimization recovery"
                    )
                params["passed_data"]["soc_init"] = soc_init
                if "soc_final" not in runtimeparams.keys():
                    soc_final = params["plant_conf"]["battery_target_state_of_charge"]
                else:
                    soc_final = runtimeparams["soc_final"]
                    if isinstance(soc_final, list):
                        # Same symmetric length-1-unwrap / clear-ValueError
                        # handling as soc_init above.
                        if len(soc_final) != 1:
                            raise ValueError(
                                f"soc_final has {len(soc_final)} entries but "
                                f"number_of_batteries=1; provide a scalar (broadcast "
                                f"to every battery) or a list of exactly 1 entry"
                            )
                        soc_final = soc_final[0]
                reject_soc_fraction_above_one("soc_final", soc_final, logger)
                if soc_final < params["plant_conf"]["battery_minimum_state_of_charge"]:
                    logger.warning(
                        f"Passed soc_final={soc_final} is lower than soc_min={params['plant_conf']['battery_minimum_state_of_charge']}, setting soc_final=soc_min"
                    )
                    soc_final = params["plant_conf"]["battery_minimum_state_of_charge"]
                if soc_final > params["plant_conf"]["battery_maximum_state_of_charge"]:
                    logger.warning(
                        f"Passed soc_final={soc_final} is greater than soc_max={params['plant_conf']['battery_maximum_state_of_charge']}, setting soc_final=soc_max"
                    )
                    soc_final = params["plant_conf"]["battery_maximum_state_of_charge"]
                params["passed_data"]["soc_final"] = soc_final
            else:
                # #610: battery_target_state_of_charge/min/max are now
                # per-battery lists (check_batt_params already normalised them
                # above). soc_init/soc_final resolve PER BATTERY k against
                # target[k]; a runtime scalar broadcasts, a runtime list must
                # be exactly num_batteries long (hard error otherwise).
                target_list = params["plant_conf"]["battery_target_state_of_charge"]
                min_list = params["plant_conf"]["battery_minimum_state_of_charge"]
                max_list = params["plant_conf"]["battery_maximum_state_of_charge"]

                def _resolve_soc_runtime_list(key: str, fallback_list: list) -> list:
                    if key not in runtimeparams.keys():
                        return list(fallback_list)
                    value = runtimeparams[key]
                    if isinstance(value, list):
                        if len(value) != num_batteries:
                            raise ValueError(
                                f"{key} has {len(value)} entries but "
                                f"number_of_batteries={num_batteries}; provide a scalar "
                                f"(broadcast to every battery) or a list of exactly "
                                f"{num_batteries} entries"
                            )
                        return list(value)
                    return [value] * num_batteries

                soc_init_list = _resolve_soc_runtime_list("soc_init", target_list)
                reject_soc_fraction_above_one("soc_init", soc_init_list, logger)
                for k in range(num_batteries):
                    if soc_init_list[k] < min_list[k]:
                        logger.warning(
                            f"Passed soc_init[{k}]={soc_init_list[k]} is lower than "
                            f"soc_min[{k}]={min_list[k]}, keeping real initial SOC for "
                            "optimization recovery"
                        )
                    if soc_init_list[k] > max_list[k]:
                        logger.warning(
                            f"Passed soc_init[{k}]={soc_init_list[k]} is greater than "
                            f"soc_max[{k}]={max_list[k]}, keeping real initial SOC for "
                            "optimization recovery"
                        )
                params["passed_data"]["soc_init"] = soc_init_list

                soc_final_list = _resolve_soc_runtime_list("soc_final", target_list)
                reject_soc_fraction_above_one("soc_final", soc_final_list, logger)
                for k in range(num_batteries):
                    if soc_final_list[k] < min_list[k]:
                        logger.warning(
                            f"Passed soc_final[{k}]={soc_final_list[k]} is lower than "
                            f"soc_min[{k}]={min_list[k]}, setting soc_final[{k}]=soc_min[{k}]"
                        )
                        soc_final_list[k] = min_list[k]
                    if soc_final_list[k] > max_list[k]:
                        logger.warning(
                            f"Passed soc_final[{k}]={soc_final_list[k]} is greater than "
                            f"soc_max[{k}]={max_list[k]}, setting soc_final[{k}]=soc_max[{k}]"
                        )
                        soc_final_list[k] = max_list[k]
                params["passed_data"]["soc_final"] = soc_final_list
            # Optional intermediate SOC target (issue #553). Both are runtime-only
            # and default to None (no intermediate target); clamping/validation of
            # the value and the timestep is handled in Optimization.perform_optimization.
            params["passed_data"]["soc_target"] = runtimeparams.get("soc_target", None)
            reject_soc_fraction_above_one("soc_target", params["passed_data"]["soc_target"], logger)
            params["passed_data"]["soc_target_timestep"] = runtimeparams.get(
                "soc_target_timestep", None
            )
            # Per-battery availability window (EV / V2G) and discharge energy
            # budget. Runtime-only, default None (always available, no budget);
            # shape validation happens in Optimization.perform_optimization.
            params["passed_data"]["battery_availability"] = runtimeparams.get(
                "battery_availability", None
            )
            params["passed_data"]["battery_discharge_energy_max"] = runtimeparams.get(
                "battery_discharge_energy_max", None
            )
            # Per-deferrable-load availability, same shape and validation as
            # battery_availability (done in Optimization.perform_optimization).
            params["passed_data"]["def_availability"] = runtimeparams.get("def_availability", None)
            # Peak grid import already incurred this billing period (issue #623,
            # Phase 2). Runtime-only, in Watts; defaults to None (no floor).
            # Coercion/validation happens in Optimization.perform_optimization.
            params["passed_data"]["current_period_peak"] = runtimeparams.get(
                "current_period_peak", None
            )
            # Demand-window mask for the capacity charge (issue #623, Phase 3).
            # Runtime-only list of [0, 1] weights, length = prediction_horizon;
            # defaults to None (all timesteps priced). Validation happens in
            # Optimization.perform_optimization. Deliberately NOT mirrored into
            # optim_conf / associations.csv: it changes every call, so putting
            # it in the structural config would defeat the OptimizationCache.
            params["passed_data"]["capacity_charge_window"] = runtimeparams.get(
                "capacity_charge_window", None
            )
            # MPC capacity consideration (issue #540 follow-up). Runtime-only
            # list of [0, 1] weights, length = prediction_horizon; SEPARATE
            # from capacity_charge_window above - this controls whether an
            # otherwise tariff-eligible prospective timestep participates in
            # THIS solve's prospective capacity peak, not tariff eligibility
            # itself. Defaults to None (every eligible timestep considered,
            # i.e. today's behaviour). Validation happens in
            # Optimization.perform_optimization. Deliberately NOT mirrored
            # into optim_conf / associations.csv, for the same
            # OptimizationCache reason as capacity_charge_window.
            params["passed_data"]["capacity_charge_consideration"] = runtimeparams.get(
                "capacity_charge_consideration", None
            )
            # Realised import history for the currently open tariff interval
            # (issue #540). Runtime-only list, in Watts; defaults to None (empty
            # history, i.e. horizon start assumed to sit on an interval
            # boundary). Coercion/validation happens in
            # Optimization.perform_optimization.
            params["passed_data"]["capacity_charge_current_interval_history"] = runtimeparams.get(
                "capacity_charge_current_interval_history", None
            )
            if "operating_timesteps_of_each_deferrable_load" in runtimeparams.keys():
                params["passed_data"]["operating_timesteps_of_each_deferrable_load"] = (
                    runtimeparams["operating_timesteps_of_each_deferrable_load"]
                )
                params["optim_conf"]["operating_timesteps_of_each_deferrable_load"] = runtimeparams[
                    "operating_timesteps_of_each_deferrable_load"
                ]
            if "operating_hours_of_each_deferrable_load" in params["optim_conf"].keys():
                params["passed_data"]["operating_hours_of_each_deferrable_load"] = params[
                    "optim_conf"
                ]["operating_hours_of_each_deferrable_load"]
            params["passed_data"]["start_timesteps_of_each_deferrable_load"] = params[
                "optim_conf"
            ].get("start_timesteps_of_each_deferrable_load", None)
            params["passed_data"]["end_timesteps_of_each_deferrable_load"] = params[
                "optim_conf"
            ].get("end_timesteps_of_each_deferrable_load", None)

            forecast_dates = copy.deepcopy(forecast_dates)[0:prediction_horizon]
        else:
            reject_configured_soc_fractions(params["plant_conf"], params["optim_conf"], logger)
            params["passed_data"]["prediction_horizon"] = None

            def _passthrough_soc_runtime(key: str):
                # The dayahead/perfect branch (this else) never had the
                # naive-mpc branch's target-fallback/clamp logic - it is (and
                # stays) a bare passthrough. The only pre-#610 behaviour was
                # float(scalar); a runtime list must not crash here. Length
                # validation and the scalar-broadcast/target-fallback for a
                # multi-battery plant both happen once, downstream, in
                # Optimization._normalize_soc_arg / perform_optimization - this
                # layer must not duplicate that logic, only stop crashing on it.
                if key not in runtimeparams:
                    return None
                value = runtimeparams[key]
                return list(value) if isinstance(value, list) else float(value)

            params["passed_data"]["soc_init"] = _passthrough_soc_runtime("soc_init")
            params["passed_data"]["soc_final"] = _passthrough_soc_runtime("soc_final")
            params["passed_data"]["soc_target"] = _passthrough_soc_runtime("soc_target")
            for soc_name in ("soc_init", "soc_final", "soc_target"):
                reject_soc_fraction_above_one(soc_name, params["passed_data"][soc_name], logger)
            # A per-battery list (naive-mpc-optim) passes through unchanged so a
            # runtimeparams blob shared with naive-mpc-optim does not crash here.
            soc_target_timestep = runtimeparams.get("soc_target_timestep")
            params["passed_data"]["soc_target_timestep"] = (
                soc_target_timestep
                if soc_target_timestep is None or isinstance(soc_target_timestep, list)
                else int(soc_target_timestep)
            )
            # Capacity-tariff peak floor: read it here too so dayahead-optim honours
            # the already-incurred billing-period peak instead of minimising the
            # absolute peak. (perfect-optim covers past data and does not use it.)
            params["passed_data"]["current_period_peak"] = runtimeparams.get(
                "current_period_peak", None
            )
            # The demand-window mask stays naive-mpc-only: dayahead/perfect
            # optimizations price the full horizon peak.
            params["passed_data"]["capacity_charge_window"] = None
            # Like the above, MPC capacity consideration is naive-mpc-only
            # (issue #540 follow-up).
            params["passed_data"]["capacity_charge_consideration"] = None
            # Like the two above, the interval-aggregation history is
            # naive-mpc-only (issue #540).
            params["passed_data"]["capacity_charge_current_interval_history"] = None

        # Parsing the thermal model parameters
        # Load the default config
        if "def_load_config" in runtimeparams:
            params["optim_conf"]["def_load_config"] = runtimeparams["def_load_config"]
            params["optim_conf"]["number_of_deferrable_loads"] = len(
                runtimeparams["def_load_config"]
            )
        if "def_load_config" in params["optim_conf"]:
            for k in range(len(params["optim_conf"]["def_load_config"])):
                if "thermal_config" in params["optim_conf"]["def_load_config"][k]:
                    if (
                        "heater_desired_temperatures" in runtimeparams
                        and len(runtimeparams["heater_desired_temperatures"]) > k
                    ):
                        params["optim_conf"]["def_load_config"][k]["thermal_config"][
                            "desired_temperatures"
                        ] = runtimeparams["heater_desired_temperatures"][k]
                    if (
                        "heater_start_temperatures" in runtimeparams
                        and len(runtimeparams["heater_start_temperatures"]) > k
                    ):
                        params["optim_conf"]["def_load_config"][k]["thermal_config"][
                            "start_temperature"
                        ] = runtimeparams["heater_start_temperatures"][k]

        # Shared thermal tanks (multi-source storage) passed at runtime: the
        # manual flat alternative to heat_topology (issue #539). Same surface
        # as def_load_config above - runtime-only, no config-file counterpart.
        if "shared_thermal_tanks" in runtimeparams:
            tanks = runtimeparams["shared_thermal_tanks"]
            problem = _runtime_shared_tanks_problem(
                tanks, int(params["optim_conf"].get("number_of_deferrable_loads") or 0)
            )
            if problem is not None:
                logger.warning("shared_thermal_tanks: %s. Ignoring the runtime tanks.", problem)
                _restore_configured_shared_tanks(params, configured_shared_tanks)
            elif isinstance(tanks, list) and all(isinstance(t, dict) for t in tanks):
                params["optim_conf"]["shared_thermal_tanks"] = tanks
                topology = params["optim_conf"].get("heat_topology")
                if isinstance(topology, dict) and topology:
                    logger.warning(
                        "Both heat_topology and runtime shared_thermal_tanks are "
                        "set; the compiled topology replaces the runtime tank "
                        "list (with extend_deferrable_loads: true the runtime "
                        "tanks are kept and the topology's tanks appended). "
                        "Pass heat_topology itself at runtime to change tanks "
                        "per run."
                    )
            else:
                logger.warning(
                    "shared_thermal_tanks must be a list of tank objects, got %s; ignoring.",
                    type(tanks).__name__,
                )
                _restore_configured_shared_tanks(params, configured_shared_tanks)

        # Treat passed forecast data lists. When an external P10
        # companion is supplied, P50/P10 are validated and aligned together below
        # so they cannot silently drift onto different timelines.
        list_forecast_key = [
            "pv_power_forecast",
            "load_power_forecast",
            "load_cost_forecast",
            "prod_price_forecast",
            "outdoor_temperature_forecast",
        ]
        forecast_methods = [
            "weather_forecast_method",
            "load_forecast_method",
            "load_cost_forecast_method",
            "production_price_forecast_method",
            "outdoor_temperature_forecast_method",
        ]
        paired_external_pv = "pv_power_forecast_p10" in runtimeparams
        # A PV forecast per inverter is only read with more than one inverter. With one
        # inverter a nested payload is refused below instead of falling back to the
        # configured method.
        n_inverters = validate_num_inverters(params["plant_conf"])
        pv_inverters: list[int] = []
        pv_inverters_fault = None
        if n_inverters > 1:
            try:
                pv_inverters = pv_carrying_inverters(params["plant_conf"])
            except ValueError as exc:
                pv_inverters_fault = str(exc)
        # Internal provenance marker used only to distinguish an omitted
        # outdoor-temperature runtime forecast from one supplied and rejected
        # by the #1135 numerical-validity contract.
        params["passed_data"]["_outdoor_temperature_forecast_rejected"] = False

        # Loop forecasts, check if value is a list and greater than or equal to forecast_dates
        for method, forecast_key in enumerate(list_forecast_key):
            # The paired external-PV path owns both P50 and P10 so the P50
            # mapping is not independently normalised first.
            if forecast_key == "pv_power_forecast" and paired_external_pv:
                continue
            if forecast_key in runtimeparams.keys():
                forecast_input = runtimeparams[forecast_key]
                invalid = None

                # Preserve the existing legacy stringified-list compatibility,
                # but normalize it before length/value validation so the parsed
                # values are subject to the same #1135 finite-real contract.
                if isinstance(forecast_input, str):
                    try:
                        parsed_forecast = ast.literal_eval(forecast_input)
                    except (SyntaxError, ValueError):
                        parsed_forecast = None
                    if isinstance(parsed_forecast, list):
                        forecast_input = parsed_forecast
                        runtimeparams[forecast_key] = parsed_forecast

                if forecast_key == "pv_power_forecast" and n_inverters > 1:
                    nested_pv, invalid = (
                        (None, pv_inverters_fault)
                        if pv_inverters_fault is not None
                        else validate_and_align_per_inverter_pv_forecast(
                            forecast_input,
                            n_inverters,
                            pv_inverters,
                            forecast_dates,
                            optimization_time_step,
                            time_zone,
                        )
                    )
                    if invalid is None:
                        params["passed_data"][forecast_key] = nested_pv
                    else:
                        logger.error("ERROR: %s", invalid)
                        params["passed_data"][forecast_key] = None
                    params["optim_conf"][forecast_methods[method]] = "list"
                    continue
                if forecast_key == "pv_power_forecast":
                    # One inverter takes one flat list or one mapping.
                    invalid = describe_nested_pv_payload_at_one_inverter(
                        forecast_key, forecast_input
                    )

                if isinstance(forecast_input, dict):
                    # Check the supplied values before pandas aggregation can
                    # coerce (bool -> 1.0) or fail on them, so the diagnostic
                    # names the source timestamp and value (#1135).
                    invalid = describe_invalid_forecast_value(forecast_key, forecast_input.items())
                    if invalid is None:
                        forecast_input = _align_runtime_forecast_mapping(
                            forecast_input,
                            forecast_dates,
                            optimization_time_step,
                            time_zone,
                        )
                if (
                    invalid is None
                    and isinstance(forecast_input, list)
                    and len(forecast_input) >= len(forecast_dates)
                ):
                    invalid = describe_invalid_forecast_value(
                        forecast_key, enumerate(forecast_input)
                    )
                    if invalid is None:
                        params["passed_data"][forecast_key] = forecast_input
                        params["optim_conf"][forecast_methods[method]] = "list"
                elif invalid is None:
                    logger.error(
                        f"ERROR: The passed data is either the wrong type or the length is not correct, length should be {str(len(forecast_dates))}"
                    )
                    logger.error(
                        f"Passed type is {str(type(runtimeparams[forecast_key]))} and length is {str(len(runtimeparams[forecast_key]))}"
                    )
                if invalid is not None:
                    # Fail closed (#1135): select the list method without data
                    # so the forecast/optimization cycle stops instead of
                    # falling back to the configured forecast method.
                    logger.error("ERROR: %s", invalid)
                    params["passed_data"][forecast_key] = None
                    params["optim_conf"][forecast_methods[method]] = "list"
                    if forecast_key == "outdoor_temperature_forecast":
                        params["passed_data"]["_outdoor_temperature_forecast_rejected"] = True
            else:
                params["passed_data"][forecast_key] = None

        # Optional external P10 companion (#1128). This is deliberately
        # pair-oriented rather than a sixth independent forecast: both series
        # must use the same representation/timeline and are aligned through the
        # exact same mapping helper. Supplying P10 explicitly selects the list
        # weather path; an invalid companion leaves a visible invalid sentinel so
        # Forecast._get_weather_list() fails the action instead of falling back.
        if paired_external_pv:
            params["optim_conf"]["weather_forecast_method"] = "list"
            if n_inverters > 1:
                p50_values, p10_values, pair_error = (
                    (None, None, pv_inverters_fault)
                    if pv_inverters_fault is not None
                    else validate_and_align_per_inverter_pv_pair(
                        runtimeparams.get("pv_power_forecast"),
                        runtimeparams.get("pv_power_forecast_p10"),
                        n_inverters,
                        pv_inverters,
                        forecast_dates,
                        optimization_time_step,
                        time_zone,
                    )
                )
            else:
                p50_values, p10_values, pair_error = _validate_and_align_external_pv_pair(
                    runtimeparams.get("pv_power_forecast"),
                    runtimeparams.get("pv_power_forecast_p10"),
                    forecast_dates,
                    optimization_time_step,
                    time_zone,
                )
            if pair_error is not None:
                logger.error("ERROR: external PV P10 companion rejected: %s", pair_error)
                params["passed_data"]["pv_power_forecast"] = None
                params["passed_data"]["pv_power_forecast_p10"] = []
            else:
                params["passed_data"]["pv_power_forecast"] = p50_values
                params["passed_data"]["pv_power_forecast_p10"] = p10_values
                if isinstance(runtimeparams["pv_power_forecast"], dict):
                    logger.info(
                        "Aligned paired external PV P50/P10 forecast from %d timestamped "
                        "point(s) onto %d optimization timestep(s).",
                        len(runtimeparams["pv_power_forecast"]),
                        len(forecast_dates),
                    )
        else:
            params["passed_data"]["pv_power_forecast_p10"] = None

        # Explicitly handle historic_days_to_retrieve from runtimeparams BEFORE validation
        if "historic_days_to_retrieve" in runtimeparams:
            params["retrieve_hass_conf"]["historic_days_to_retrieve"] = int(
                runtimeparams["historic_days_to_retrieve"]
            )

        # Treat passed data for forecast model fit/predict/tune at runtime
        if (
            params["passed_data"].get("historic_days_to_retrieve", None) is not None
            and params["passed_data"]["historic_days_to_retrieve"] < 9
        ):
            logger.warning(
                "warning `days_to_retrieve` is set to a value less than 9, this could cause an error with the fit"
            )
            logger.warning("setting`passed_data:days_to_retrieve` to 9 for fit/predict/tune")
            params["passed_data"]["historic_days_to_retrieve"] = 9
        else:
            if params["retrieve_hass_conf"].get("historic_days_to_retrieve", 0) < 9:
                logger.debug("setting`passed_data:days_to_retrieve` to 9 for fit/predict/tune")
                params["passed_data"]["historic_days_to_retrieve"] = 9
            else:
                params["passed_data"]["historic_days_to_retrieve"] = params["retrieve_hass_conf"][
                    "historic_days_to_retrieve"
                ]

        # UPDATED ML PARAMETER HANDLING
        # Define Helper Functions
        def _cast_bool(value):
            """Helper to cast string inputs to boolean safely."""
            # ast.literal_eval('None') returns None without raising — explicit guard needed.
            if value is None:
                return False
            try:
                return ast.literal_eval(str(value).capitalize())
            except (ValueError, SyntaxError):
                return False

        def _get_ml_param(name, params, runtimeparams, default=None, cast=None):
            """
            Prioritize Runtime Params -> Config Params (optim_conf) -> Default.
            """
            if name in runtimeparams:
                value = runtimeparams[name]
            else:
                value = params["optim_conf"].get(name, default)

            if cast is not None and value is not None:
                try:
                    value = cast(value)
                except Exception:
                    pass
            return value

        # Compute dynamic defaults
        # Default for var_model falls back to the configured load sensor
        default_var_model = params["retrieve_hass_conf"].get(
            "sensor_power_load_no_var_loads", "sensor.power_load_no_var_loads"
        )

        # Define Configuration Table
        # Format: (parameter_name, default_value, cast_function)
        ml_param_defs = [
            ("model_type", "long_train_data", None),
            ("var_model", default_var_model, None),
            ("sklearn_model", "KNeighborsRegressor", None),
            ("regression_model", "AdaBoostRegressor", None),
            ("num_lags", 48, None),
            ("split_date_delta", "48h", None),
            ("n_trials", 10, int),
            ("perform_backtest", False, _cast_bool),
            ("mlforecaster_weather_features", [], None),
        ]

        # Apply Configuration
        for name, default, caster in ml_param_defs:
            params["passed_data"][name] = _get_ml_param(
                name=name,
                params=params,
                runtimeparams=runtimeparams,
                default=default,
                cast=caster,
            )
        # model_type becomes part of the model's file name, whether the request
        # or the saved configuration supplied it
        if params["passed_data"]["model_type"] is not None:
            validate_runtime_file_name("model_type", params["passed_data"]["model_type"])
        # config_defaults.json always supplies var_model, and a GUI save writes that stock
        # value back to config.json. An unset or stock var_model follows the configured
        # load sensor, so a custom sensor_power_load_no_var_loads is the forecaster input (#1208).
        if (
            "var_model" not in runtimeparams
            and default_var_model
            and params["passed_data"]["var_model"] in (None, "sensor.power_load_no_var_loads")
        ):
            params["passed_data"]["var_model"] = default_var_model

        # Other non-dynamic options
        if "model_predict_publish" not in runtimeparams.keys():
            model_predict_publish = False
        else:
            try:
                model_predict_publish = ast.literal_eval(
                    str(runtimeparams["model_predict_publish"]).capitalize()
                )
            except (ValueError, SyntaxError):
                # The parser's error quotes the capitalized text, which the log
                # mask cannot match if it is a secret sent here by mistake.
                raise RuntimeParamError("model_predict_publish must be true or false") from None
        params["passed_data"]["model_predict_publish"] = model_predict_publish
        if "model_predict_entity_id" not in runtimeparams.keys():
            model_predict_entity_id = "sensor.p_load_forecast_custom_model"
        else:
            model_predict_entity_id = runtimeparams["model_predict_entity_id"]
        params["passed_data"]["model_predict_entity_id"] = model_predict_entity_id
        if "model_predict_device_class" not in runtimeparams.keys():
            model_predict_device_class = "power"
        else:
            model_predict_device_class = runtimeparams["model_predict_device_class"]
        params["passed_data"]["model_predict_device_class"] = model_predict_device_class
        if "model_predict_unit_of_measurement" not in runtimeparams.keys():
            model_predict_unit_of_measurement = "W"
        else:
            model_predict_unit_of_measurement = runtimeparams["model_predict_unit_of_measurement"]
        params["passed_data"]["model_predict_unit_of_measurement"] = (
            model_predict_unit_of_measurement
        )
        if "model_predict_friendly_name" not in runtimeparams.keys():
            model_predict_friendly_name = "Load Power Forecast custom ML model"
        else:
            model_predict_friendly_name = runtimeparams["model_predict_friendly_name"]
        params["passed_data"]["model_predict_friendly_name"] = model_predict_friendly_name
        if "mlr_predict_entity_id" not in runtimeparams.keys():
            mlr_predict_entity_id = "sensor.mlr_predict"
        else:
            mlr_predict_entity_id = runtimeparams["mlr_predict_entity_id"]
        params["passed_data"]["mlr_predict_entity_id"] = mlr_predict_entity_id
        if "mlr_predict_device_class" not in runtimeparams.keys():
            mlr_predict_device_class = "power"
        else:
            mlr_predict_device_class = runtimeparams["mlr_predict_device_class"]
        params["passed_data"]["mlr_predict_device_class"] = mlr_predict_device_class
        if "mlr_predict_unit_of_measurement" not in runtimeparams.keys():
            mlr_predict_unit_of_measurement = None
        else:
            mlr_predict_unit_of_measurement = runtimeparams["mlr_predict_unit_of_measurement"]
        params["passed_data"]["mlr_predict_unit_of_measurement"] = mlr_predict_unit_of_measurement
        if "mlr_predict_friendly_name" not in runtimeparams.keys():
            mlr_predict_friendly_name = "mlr predictor"
        else:
            mlr_predict_friendly_name = runtimeparams["mlr_predict_friendly_name"]
        params["passed_data"]["mlr_predict_friendly_name"] = mlr_predict_friendly_name

        # Treat passed data for other parameters
        if "alpha" not in runtimeparams.keys():
            alpha = 0.5
        else:
            alpha = runtimeparams["alpha"]
        params["passed_data"]["alpha"] = alpha
        if "beta" not in runtimeparams.keys():
            beta = 0.5
        else:
            beta = runtimeparams["beta"]
        params["passed_data"]["beta"] = beta

        # Param to save forecast cache (i.e. Solcast)
        if "weather_forecast_cache" not in runtimeparams.keys():
            weather_forecast_cache = False
        else:
            weather_forecast_cache = runtimeparams["weather_forecast_cache"]
        params["passed_data"]["weather_forecast_cache"] = weather_forecast_cache

        # Param to make sure optimization only uses cached data. (else produce error)
        if "weather_forecast_cache_only" not in runtimeparams.keys():
            weather_forecast_cache_only = False
        else:
            weather_forecast_cache_only = runtimeparams["weather_forecast_cache_only"]
        params["passed_data"]["weather_forecast_cache_only"] = weather_forecast_cache_only

        # Param to bypass PV-feedback mixing during curtailment events (#818)
        if "ignore_pv_feedback_during_curtailment" not in runtimeparams.keys():
            ignore_pv_feedback_during_curtailment = False
        else:
            ignore_pv_feedback_during_curtailment = bool(
                runtimeparams["ignore_pv_feedback_during_curtailment"]
            )
        params["passed_data"]["ignore_pv_feedback_during_curtailment"] = (
            ignore_pv_feedback_during_curtailment
        )

        # A condition to manually save entity data under data_path/entities after optimization
        if "entity_save" not in runtimeparams.keys():
            entity_save = ""
        else:
            entity_save = runtimeparams["entity_save"]
        params["passed_data"]["entity_save"] = entity_save

        # A condition to put a prefix on all published data, or check for saved data under prefix name
        if "publish_prefix" not in runtimeparams.keys():
            publish_prefix = ""
        else:
            publish_prefix = runtimeparams["publish_prefix"]
        params["passed_data"]["publish_prefix"] = publish_prefix

        # Treat optimization (optim_conf) configuration parameters passed at runtime
        if "def_current_state" in runtimeparams.keys():
            dcs = runtimeparams["def_current_state"]
            # If passed as a string (e.g. '[false, false]'), parse it to a list
            if isinstance(dcs, str):
                try:
                    dcs = orjson.loads(dcs)
                except Exception:
                    logger.warning(
                        f"Could not parse def_current_state string ({len(dcs)} characters)"
                    )
            # Use _cast_bool (not bool()) so string 'False' → False, not True.
            # bool('False') = True because non-empty strings are truthy in Python.
            if isinstance(dcs, list):
                params["optim_conf"]["def_current_state"] = [_cast_bool(s) for s in dcs]
            else:
                n_loads = len(params["optim_conf"]["nominal_power_of_deferrable_loads"])
                params["optim_conf"]["def_current_state"] = [_cast_bool(dcs)] * n_loads

        def _per_load_numbers(name, value, cast):
            """A per-load runtime list: a JSON list string, a list, or one value for
            every load. Neither the warning nor the error quotes the value."""
            if isinstance(value, str):
                try:
                    value = orjson.loads(value)
                except Exception:
                    logger.warning(f"Could not parse {name} string ({len(value)} characters)")
            try:
                if isinstance(value, list):
                    return [cast(v) for v in value]
                one = cast(value)
            except (TypeError, ValueError, OverflowError):
                raise RuntimeParamError(f"{name} must be a number or a list of numbers") from None
            # Only one value for every load needs the load count; a list has its own.
            return [one] * len(params["optim_conf"]["nominal_power_of_deferrable_loads"])

        # def_current_on_timesteps: per-load elapsed ON timesteps for min-on remainder
        # (issue #952). Mirrors def_current_state: absent key -> no initial force (NOT
        # assumed zero). Validates that each entry is a non-negative integer.
        if "def_current_on_timesteps" in runtimeparams.keys():
            params["optim_conf"]["def_current_on_timesteps"] = _per_load_numbers(
                "def_current_on_timesteps", runtimeparams["def_current_on_timesteps"], int
            )

        # def_current_off_timesteps: per-load elapsed OFF timesteps for min-off remainder
        # (#952 follow-on). Mirrors def_current_on_timesteps: absent key -> no initial
        # force (NOT assumed zero). Validates that each entry is a non-negative integer.
        if "def_current_off_timesteps" in runtimeparams:
            params["optim_conf"]["def_current_off_timesteps"] = _per_load_numbers(
                "def_current_off_timesteps", runtimeparams["def_current_off_timesteps"], int
            )

        # def_current_power: per-load current power in watts (issue #605).
        # Absent key -> no pin, no force-on (NOT assumed zero). Mirrors def_current_on_timesteps.
        if "def_current_power" in runtimeparams:
            params["optim_conf"]["def_current_power"] = _per_load_numbers(
                "def_current_power", runtimeparams["def_current_power"], float
            )

        # def_current_operating_timesteps: per-load completed operating timesteps today (issue #983).
        # Absent key -> no decrement (NOT assumed zero). Mirrors def_current_on_timesteps.
        if "def_current_operating_timesteps" in runtimeparams:
            params["optim_conf"]["def_current_operating_timesteps"] = _per_load_numbers(
                "def_current_operating_timesteps",
                runtimeparams["def_current_operating_timesteps"],
                int,
            )

        # set_deferrable_load_single_constant arrives via the generic associations.csv
        # path as-is (may be a list of strings from runtimeparams JSON).  Apply the
        # same _cast_bool coercion so 'False' → False, not True (#873, mirrors #876).
        if "set_deferrable_load_single_constant" in runtimeparams.keys():
            sdlsc = runtimeparams["set_deferrable_load_single_constant"]
            if isinstance(sdlsc, list):
                params["optim_conf"]["set_deferrable_load_single_constant"] = [
                    _cast_bool(s) for s in sdlsc
                ]
            else:
                n_loads = len(params["optim_conf"]["nominal_power_of_deferrable_loads"])
                params["optim_conf"]["set_deferrable_load_single_constant"] = [
                    _cast_bool(sdlsc)
                ] * n_loads

        # Treat retrieve data from Home Assistant (retrieve_hass_conf) configuration parameters passed at runtime
        # Secrets passed at runtime
        if "solcast_api_key" in runtimeparams.keys():
            params["retrieve_hass_conf"]["solcast_api_key"] = runtimeparams["solcast_api_key"]
        if "solcast_rooftop_id" in runtimeparams.keys():
            params["retrieve_hass_conf"]["solcast_rooftop_id"] = runtimeparams["solcast_rooftop_id"]
        if "solar_forecast_kwp" in runtimeparams.keys():
            params["retrieve_hass_conf"]["solar_forecast_kwp"] = runtimeparams["solar_forecast_kwp"]
        # Treat custom entities id's and friendly names for variables
        # Runtime-only day windows for the forecast-calibration report. These are
        # report knobs (not config settings), so they live in passed_data next to
        # the custom_* ids and never touch retrieve_hass_conf / optim_conf. Each is
        # optional; the calibration action falls back to its default when unset.
        for calibration_key in (
            "calibration_days_to_retrieve",
            "calibration_test_days",
            "calibration_val_days",
        ):
            if calibration_key in runtimeparams.keys():
                try:
                    params["passed_data"][calibration_key] = int(runtimeparams[calibration_key])
                except (TypeError, ValueError):
                    logger.warning(
                        f"Ignoring non-integer runtime value for {calibration_key}: "
                        f"{runtimeparams[calibration_key]}"
                    )
        if "custom_pv_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_pv_forecast_id"] = runtimeparams["custom_pv_forecast_id"]
        if "custom_load_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_load_forecast_id"] = runtimeparams[
                "custom_load_forecast_id"
            ]
        if "custom_pv_curtailment_id" in runtimeparams.keys():
            params["passed_data"]["custom_pv_curtailment_id"] = runtimeparams[
                "custom_pv_curtailment_id"
            ]
        if "custom_hybrid_inverter_id" in runtimeparams.keys():
            params["passed_data"]["custom_hybrid_inverter_id"] = runtimeparams[
                "custom_hybrid_inverter_id"
            ]
        if "custom_batt_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_batt_forecast_id"] = runtimeparams[
                "custom_batt_forecast_id"
            ]
        if "custom_batt_soc_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_batt_soc_forecast_id"] = runtimeparams[
                "custom_batt_soc_forecast_id"
            ]
        if "custom_grid_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_grid_forecast_id"] = runtimeparams[
                "custom_grid_forecast_id"
            ]
        if "custom_cost_fun_id" in runtimeparams.keys():
            params["passed_data"]["custom_cost_fun_id"] = runtimeparams["custom_cost_fun_id"]
        if "custom_optim_status_id" in runtimeparams.keys():
            params["passed_data"]["custom_optim_status_id"] = runtimeparams[
                "custom_optim_status_id"
            ]
        if "custom_unit_load_cost_id" in runtimeparams.keys():
            params["passed_data"]["custom_unit_load_cost_id"] = runtimeparams[
                "custom_unit_load_cost_id"
            ]
        if "custom_unit_prod_price_id" in runtimeparams.keys():
            params["passed_data"]["custom_unit_prod_price_id"] = runtimeparams[
                "custom_unit_prod_price_id"
            ]
        if "custom_deferrable_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_deferrable_forecast_id"] = runtimeparams[
                "custom_deferrable_forecast_id"
            ]
        if "custom_deferrable_state_id" in runtimeparams.keys():
            params["passed_data"]["custom_deferrable_state_id"] = runtimeparams[
                "custom_deferrable_state_id"
            ]
        if "custom_predicted_temperature_id" in runtimeparams.keys():
            params["passed_data"]["custom_predicted_temperature_id"] = runtimeparams[
                "custom_predicted_temperature_id"
            ]
        if "custom_heating_demand_id" in runtimeparams.keys():
            params["passed_data"]["custom_heating_demand_id"] = runtimeparams[
                "custom_heating_demand_id"
            ]

        # Entity ids end up in a Home Assistant URL path and in a saved file name
        validate_runtime_entity_ids(runtimeparams)

    # split config categories from params
    retrieve_hass_conf = params["retrieve_hass_conf"]
    optim_conf = params["optim_conf"]
    plant_conf = params["plant_conf"]
    # costfun becomes part of a results file name (perfect-optim). Checked on the merged
    # value, so a saved setting gets the same check as a runtime one.
    if optim_conf.get("costfun") is not None and optim_conf["costfun"] not in COSTFUN_VALUES:
        raise RuntimeParamError(f"costfun must be one of {', '.join(COSTFUN_VALUES)}")
    # After the runtime merge when a request was supplied, and on the saved
    # config when it was not. A percentage in either place must fail here.
    reject_configured_soc_fractions(plant_conf, optim_conf, logger)
    # The same for the inverter parameters (#1150): the CLI passes no runtime parameters,
    # so the saved configuration is validated and refused here when the block above
    # did not already do so.
    if not inverters_checked:
        # site first: its meter and hybrid inverter fill EMHASS's own keys, which
        # the check below then validates, as the request path does (above)
        site = optim_conf.get("site")
        if site:
            problem = site_problem(site)
            if problem:
                logger.error("site: %s; not used", problem)
            else:
                compile_site(site, plant_conf, logger, bool(optim_conf.get("set_use_battery")))
        require_usable_inverter_config(plant_conf, logger)
    # The same for the battery count: the CLI merges --params into plant_conf, and
    # without --runtimeparams the runtime block above, which checks it, does not run.
    require_valid_battery_count(plant_conf, logger)

    # Shared tanks from the configuration file get the same check as runtime
    # ones (above), so a bad entry is reported and ignored instead of failing
    # inside the optimizer. A runtime list that passed is checked again, which
    # is harmless.
    configured_tanks = optim_conf.get("shared_thermal_tanks")
    if configured_tanks:
        if not isinstance(configured_tanks, list) or not all(
            isinstance(t, dict) for t in configured_tanks
        ):
            problem = "must be a list of tank objects"
        else:
            try:
                n_loads = int(optim_conf.get("number_of_deferrable_loads") or 0)
            except (TypeError, ValueError):
                n_loads = 0
            problem = _runtime_shared_tanks_problem(configured_tanks, n_loads)
        if problem is not None:
            logger.warning("shared_thermal_tanks in the configuration: %s. Ignoring them.", problem)
            optim_conf.pop("shared_thermal_tanks", None)

    # If heat_topology is present (static config or runtime override), compile
    # it down to flat optim_conf primitives. Runtime override wins over static
    # config because runtimeparams have already been merged above.
    heat_topology = optim_conf.get("heat_topology")
    extend_mode = bool(
        isinstance(heat_topology, dict)
        and heat_topology
        and heat_topology.get("extend_deferrable_loads", False)
    )
    if isinstance(heat_topology, dict) and heat_topology:
        try:
            compiled = compile_heat_topology(heat_topology)
        except ValueError as e:
            logger.error("heat_topology compile failed: %s", e)
            raise
        if heat_topology.get("extend_deferrable_loads", False):
            # Opt-in append mode: keep the user's configured deferrable loads
            # and add the compiled topology loads AFTER them. Setting the flag
            # is the user asserting their flat per-load config describes real
            # loads (the compiler cannot tell configured loads apart from
            # defaults, which is why replace is the default).
            _extend_optim_conf_with_compiled_topology(optim_conf, compiled, logger)
            # naive-mpc-optim reads the operating timesteps from passed_data.
            passed = params.get("passed_data") or {}
            if passed.get("operating_timesteps_of_each_deferrable_load") is not None:
                passed["operating_timesteps_of_each_deferrable_load"] = optim_conf[
                    "operating_timesteps_of_each_deferrable_load"
                ]
        else:
            # Default replace mode: merge compiled fields into optim_conf,
            # allowing user-set fields to win for things the compiler always
            # populates (e.g. operating_hours).
            _warn_replaced_deferrable_loads(optim_conf, compiled, emhass_conf, logger)
            for key, val in compiled.items():
                if key not in optim_conf or optim_conf[key] in (None, [], {}):
                    optim_conf[key] = val
                else:
                    # For the structural fields we ALWAYS want compiled values
                    # (otherwise the compiled def_load_config doesn't match
                    # number_of_deferrable_loads, etc.)
                    if key in _TOPOLOGY_REPLACED_KEYS:
                        optim_conf[key] = val
        params["optim_conf"] = optim_conf
        logger.info(
            "heat_topology compiled: %d sources, %d storage, %d flows, %d groups",
            len(heat_topology.get("sources", [])),
            len(heat_topology.get("storage", [])),
            len(heat_topology.get("flows", [])),
            len(heat_topology.get("actuator_groups", [])),
        )
    elif heat_topology:
        logger.warning(
            "heat_topology is set but is %s, not a dict (value: %r). "
            "Treating as 'no topology'. Use JSON null (not the string \"null\") "
            "or omit the key to disable.",
            type(heat_topology).__name__,
            heat_topology,
        )

    # Per-tank start-temperature override (issue #539), keyed by tank id.
    # Applied AFTER the heat_topology compile so it patches manual and
    # compiled tanks alike - the per-run analog of soc_init and
    # heater_start_temperatures for MPC loops feeding live sensor readings.
    if runtimeparams and "shared_tank_start_temperatures" in runtimeparams:
        overrides = runtimeparams["shared_tank_start_temperatures"]
        if not isinstance(overrides, dict):
            logger.warning(
                "shared_tank_start_temperatures must be an object mapping tank "
                "id to a temperature, got %s; ignoring.",
                type(overrides).__name__,
            )
        else:
            tanks = optim_conf.get("shared_thermal_tanks") or []
            # JSON object keys are always strings: match tank ids as text.
            tank_ids = {str(t.get("id")) for t in tanks if isinstance(t, dict)}
            for tank_id, temp in overrides.items():
                tank_id = str(tank_id)
                if tank_id not in tank_ids:
                    logger.warning(
                        "shared_tank_start_temperatures: unknown tank id '%s' "
                        "(known ids: %s); ignoring.",
                        tank_id,
                        sorted(i for i in tank_ids if i is not None),
                    )
                    continue
                try:
                    # A boolean is not a temperature (float(True) would be 1.0).
                    temp_value = None if isinstance(temp, bool) else float(temp)
                except (TypeError, ValueError, OverflowError):
                    temp_value = None
                if temp_value is None or not math.isfinite(temp_value):
                    logger.warning(
                        "shared_tank_start_temperatures['%s']=%r is not numeric; ignoring.",
                        tank_id,
                        temp,
                    )
                    continue
                for tank in tanks:
                    if isinstance(tank, dict) and str(tank.get("id")) == tank_id:
                        tank["start_temperature"] = temp_value
    # Re-normalise per-load deferrable array params against the FINAL
    # number_of_deferrable_loads (#1040). This has to run last: the
    # association loop above, the def_load_config handling, and the
    # heat_topology compile step can each change number_of_deferrable_loads
    # or one of these arrays, and heat_topology runs even when runtimeparams
    # is None. Mirrors the per-battery re-normalisation above (#610), but
    # pads short arrays rather than truncating or erroring, matching
    # check_def_loads's existing pad-to-fit semantics.
    raw_num_def_loads = optim_conf.get("number_of_deferrable_loads", None)
    if raw_num_def_loads is not None:
        try:
            final_num_def_loads = int(raw_num_def_loads)
        except (TypeError, ValueError):
            logger.warning(
                "number_of_deferrable_loads is not a valid integer "
                f"({raw_num_def_loads!r}); skipping deferrable load array "
                "re-normalisation"
            )
            final_num_def_loads = None
        if final_num_def_loads is not None:
            # Write the cast int back: downstream readers of
            # optim_conf["number_of_deferrable_loads"] need an int, not
            # whatever raw value (e.g. a numeric string) the association
            # loop copied in.
            optim_conf["number_of_deferrable_loads"] = final_num_def_loads
            runtime_source = runtimeparams or {}
            for def_array_name, def_array_default in DEF_LOAD_ARRAY_PARAMS.items():
                legacy_name = DEF_LOAD_ARRAY_LEGACY_NAMES.get(def_array_name)
                runtime_value = runtime_source.get(def_array_name)
                if runtime_value is None and legacy_name is not None:
                    runtime_value = runtime_source.get(legacy_name)
                # A JSON null is treated as "not provided", matching the
                # association loop's own null-skipping - a padded config
                # array should never be reported as a runtime override.
                was_provided = runtime_value is not None

                if (
                    was_provided
                    and not isinstance(runtime_value, list)
                    and extend_mode
                    and isinstance(optim_conf.get(def_array_name), list)
                    and len(optim_conf[def_array_name]) == final_num_def_loads
                ):
                    # Extend mode already applied the scalar to the user's own
                    # loads and kept the compiled values of the topology loads.
                    continue
                if was_provided and not isinstance(runtime_value, list):
                    # A runtime scalar means every load, not "pad with the
                    # default" - mirrors check_batt_params's silent
                    # broadcast, and overrides any earlier partial handling
                    # (e.g. set_deferrable_load_single_constant's own scalar
                    # broadcast above, which doesn't know the final count).
                    optim_conf[def_array_name] = [runtime_value] * final_num_def_loads
                    continue

                current_value = optim_conf.get(def_array_name)
                if isinstance(current_value, list):
                    # Copy before padding: check_def_loads mutates in place,
                    # and this list may be the same object as
                    # runtimeparams[name] or a passed_data alias set earlier
                    # in this function - only optim_conf's own value should
                    # change.
                    before_value = list(current_value)
                    optim_conf[def_array_name] = list(current_value)
                else:
                    before_value = current_value
                optim_conf[def_array_name] = check_def_loads(
                    final_num_def_loads,
                    optim_conf,
                    def_array_default,
                    def_array_name,
                    logger,
                )
                _warn_if_runtime_def_array_too_short(
                    was_provided,
                    def_array_name,
                    before_value,
                    optim_conf[def_array_name],
                    final_num_def_loads,
                    logger,
                )
            params["optim_conf"] = optim_conf
            # The default publish ids were built from the load count before the
            # steps above could change it (a heat_topology compile, a runtime
            # def_load_config). Pad each per-load id list with the defaults for
            # the extra loads, or publish-data indexes past its end. A list the
            # caller passed is padded too (it would otherwise fail the same way),
            # but that is logged: those loads publish under their default names.
            passed_data = params.get("passed_data") or {}
            user_keys = set(runtimeparams or {})
            for ids_key in _LOAD_PUBLISH_ID_KEYS:
                ids = passed_data.get(ids_key)
                if isinstance(ids, list) and len(ids) < final_num_def_loads:
                    if ids_key in user_keys:
                        logger.warning(
                            "%s has %d entries for %d deferrable loads; loads %d..%d "
                            "publish under their default entity names",
                            ids_key,
                            len(ids),
                            final_num_def_loads,
                            len(ids),
                            final_num_def_loads - 1,
                        )
                    passed_data[ids_key] = list(ids) + [
                        _default_load_publish_ids(k)[ids_key]
                        for k in range(len(ids), final_num_def_loads)
                    ]

    # Per-tank in-flight heat for a lagged tank (thermal_inertia), keyed by tank id:
    # the thermal kWh already delivered per step by previous rolling-MPC runs but not
    # yet arrived. Sibling of shared_tank_start_temperatures - both are per-run state,
    # not configuration - and likewise tolerant: bad entries warn and are skipped so a
    # flaky feed degrades to the previous cold-start behaviour instead of failing.
    if runtimeparams and "shared_tank_prior_heat" in runtimeparams:
        overrides = runtimeparams["shared_tank_prior_heat"]
        if not isinstance(overrides, dict):
            logger.warning(
                "shared_tank_prior_heat must be an object mapping tank id to a list "
                "of per-step thermal kWh, got %s; ignoring.",
                type(overrides).__name__,
            )
        else:
            tanks = optim_conf.get("shared_thermal_tanks") or []
            # JSON object keys are always strings: match tank ids as text.
            tank_ids = {str(t.get("id")) for t in tanks if isinstance(t, dict)}
            for tank_id, series in overrides.items():
                tank_id = str(tank_id)
                if tank_id not in tank_ids:
                    logger.warning(
                        "shared_tank_prior_heat: unknown tank id '%s' (known ids: %s); ignoring.",
                        tank_id,
                        sorted(i for i in tank_ids if i is not None),
                    )
                    continue
                matching = [t for t in tanks if isinstance(t, dict) and str(t.get("id")) == tank_id]
                if len(matching) > 1:
                    # Two tanks whose ids read the same as text (0 and "0"): the
                    # key cannot say which one is meant.
                    logger.warning(
                        "shared_tank_prior_heat['%s'] matches %d tanks whose ids read the "
                        "same; ignoring.",
                        tank_id,
                        len(matching),
                    )
                    continue
                values, problem = _prior_heat_values(series)
                if problem is not None:
                    # A bad runtime value starts that tank cold for the run, as
                    # documented: a configured prior_heat would be stale here.
                    logger.warning(
                        "shared_tank_prior_heat['%s'] %s; ignoring, the tank starts cold.",
                        tank_id,
                        problem,
                    )
                    matching[0].pop("prior_heat", None)
                    continue
                matching[0]["prior_heat"] = values
    # Canonicalise the structural multi-component capacity-charge params (#540
    # Part B): a config-UI singleton list, an empty list or a stringified list
    # is normalised, and a wrong-length interval list is rejected here - once,
    # before the value reaches the OptimizationCache key or the model build.
    optim_conf = canonicalize_capacity_charge_config(optim_conf, logger)
    params["optim_conf"] = optim_conf

    # Serialize the final params
    params = orjson.dumps(params, default=str).decode()
    return params, retrieve_hass_conf, optim_conf, plant_conf


def get_yaml_parse(params: str | dict, logger: logging.Logger) -> tuple[dict, dict, dict]:
    """
    Perform parsing of the params into the configuration catagories

    :param params: Built configuration parameters
    :type params: str or dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: A tuple with the dictionaries containing the parsed data
    :rtype: tuple(dict)

    """
    if params:
        if type(params) is str:
            input_conf = orjson.loads(params)
        else:
            input_conf = params
    else:
        input_conf = {}
        logger.error("No params have been detected for get_yaml_parse")
        return False, False, False

    optim_conf = input_conf.get("optim_conf", {})
    retrieve_hass_conf = input_conf.get("retrieve_hass_conf", {})
    plant_conf = input_conf.get("plant_conf", {})

    # Format time parameters
    if optim_conf.get("delta_forecast_daily", None) is not None:
        optim_conf["delta_forecast_daily"] = pd.Timedelta(days=optim_conf["delta_forecast_daily"])
    if retrieve_hass_conf.get("optimization_time_step", None) is not None:
        retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(
            retrieve_hass_conf["optimization_time_step"], "minutes"
        )
    if retrieve_hass_conf.get("time_zone", None) is not None:
        retrieve_hass_conf["time_zone"] = pytz.timezone(retrieve_hass_conf["time_zone"])

    return retrieve_hass_conf, optim_conf, plant_conf


_UI_DROP_PLOT_POWER_COLS = frozenset({"P_grid_pos", "P_grid_neg"})
_UI_DROP_TABLE_COLS = frozenset(
    {"P_grid_pos", "P_grid_neg", "maximum_power_from_grid", "maximum_power_to_grid"}
)


def _fig_to_html(fig: go.Figure, *, include_plotlyjs: bool = False) -> str:
    """Serialize a Plotly figure into responsive HTML suitable for the WebUI."""
    layout_kwargs: dict = {
        "autosize": True,
        "font": {"size": 12},
        "title": {"x": 0.01, "xanchor": "left", "y": 0.96, "yanchor": "top", "font": {"size": 15}},
        "margin": {"l": 85, "r": 28, "t": 72, "b": 100},
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
            "title": {"text": ""},
        },
    }
    if fig.layout.height is None:
        layout_kwargs["height"] = 460
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(automargin=True, title_standoff=14)
    fig.update_yaxes(automargin=True, title_standoff=14)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        default_width="100%",
        config={"responsive": True, "displaylogo": False},
    )


def _infer_timestep_hours(index: pd.Index, ui_meta: dict | None = None) -> float:
    """Infer timestep length in hours from index diffs, index freq, or UI metadata (default 0.5h)."""
    if len(index) >= 2:
        try:
            dt_index = pd.DatetimeIndex(index)
            diffs = dt_index.to_series().diff().dropna().dt.total_seconds()
            positive_diffs = diffs[diffs > 0]
            if not positive_diffs.empty:
                return float(positive_diffs.median() / 3600.0)
        except (TypeError, ValueError):
            pass
    if isinstance(index, pd.DatetimeIndex) and index.freq is not None:
        try:
            freq_td = pd.Timedelta(index.freq)
            if freq_td.total_seconds() > 0:
                return float(freq_td.total_seconds() / 3600.0)
        except (TypeError, ValueError):
            pass
    if isinstance(ui_meta, dict):
        meta_dt = ui_meta.get("optimization_time_step_hours")
        if (
            isinstance(meta_dt, int | float)
            and math.isfinite(float(meta_dt))
            and float(meta_dt) > 0
        ):
            return float(meta_dt)
    return 0.5


def _electric_deferrable_cols(df: pd.DataFrame, ui_meta: dict | None = None) -> list[str]:
    """Return P_deferrable* columns that represent electric loads (respecting is_electric_load)."""
    optim_conf = (ui_meta.get("optim_conf") or {}) if isinstance(ui_meta, dict) else {}
    is_electric = optim_conf.get("is_electric_load")
    cols: list[str] = []
    for col in sorted(c for c in df.columns if c.startswith("P_deferrable")):
        suffix = col[len("P_deferrable") :]
        if suffix.isdigit() and isinstance(is_electric, list):
            idx = int(suffix)
            if 0 <= idx < len(is_electric) and is_electric[idx] is False:
                continue
        cols.append(col)
    return cols


def _battery_power_total(df: pd.DataFrame) -> pd.Series | None:
    """Return total battery power series (P_batt or sum of P_batt_<k>), or None."""
    if "P_batt" in df.columns:
        return pd.to_numeric(df["P_batt"], errors="coerce").astype(float)
    multi_cols = [
        c for c in df.columns if c.startswith("P_batt_") and c[len("P_batt_") :].isdigit()
    ]
    if not multi_cols:
        return None
    return (
        df[multi_cols].apply(pd.to_numeric, errors="coerce").astype(float).sum(axis=1, min_count=1)
    )


def _battery_charge_discharge(df: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """Return (charge_w, discharge_w) using indexed P_batt_<k> columns when present, else P_batt.

    Separating charge and discharge per battery before summing preserves simultaneous
    opposing flows across multi-battery banks.
    """
    multi_cols = [
        c for c in df.columns if c.startswith("P_batt_") and c[len("P_batt_") :].isdigit()
    ]
    if multi_cols:
        mat = df[multi_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        if not mat.notna().any().any():
            return None
        charge_w = (-mat).clip(lower=0.0).fillna(0.0).sum(axis=1)
        discharge_w = mat.clip(lower=0.0).fillna(0.0).sum(axis=1)
        return charge_w, discharge_w
    if "P_batt" in df.columns:
        p_batt = pd.to_numeric(df["P_batt"], errors="coerce").astype(float)
        if not p_batt.notna().any():
            return None
        batt = p_batt.fillna(0.0)
        return (-batt).clip(lower=0.0), batt.clip(lower=0.0)
    return None


_PV_P50_LABEL = "PV P50 (median)"
_PV_P10_LABEL = "PV P10 (conservative)"
_PLOTLY_LINES_MARKERS = "lines+markers"


def _to_utc_index(idx_like: object) -> pd.DatetimeIndex | None:
    """Parse a datetime-like index or string/Timestamp sequence into a UTC DatetimeIndex."""
    try:
        if isinstance(idx_like, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(idx_like):
            parsed = pd.DatetimeIndex(pd.to_datetime(idx_like, utc=True, errors="coerce"))
            return parsed if parsed.notna().all() else None
        # Only parse string/Timestamp sequences; never convert numeric RangeIndex to 1970 epoch.
        sample = list(idx_like) if not isinstance(idx_like, pd.Index) else idx_like.tolist()
        if sample and all(isinstance(v, str | datetime | pd.Timestamp) for v in sample):
            parsed = pd.DatetimeIndex(pd.to_datetime(sample, utc=True, errors="coerce"))
            return parsed if parsed.notna().all() else None
    except Exception:
        return None
    return None


def _align_pv_quantile_series(
    series_like: object,
    x_idx: pd.Index,
    x_utc: pd.DatetimeIndex | None,
    q_timestamps: object,
) -> pd.Series | None:
    """Align a PV quantile array/Series onto the forecast DataFrame index."""
    if series_like is None:
        return None
    try:
        s = pd.to_numeric(pd.Series(series_like), errors="coerce").astype(float)
        src_utc = None
        if isinstance(series_like, pd.Series):
            if series_like.index.equals(x_idx):
                return s
            src_utc = _to_utc_index(series_like.index)
        elif isinstance(q_timestamps, list | tuple | pd.Index) and len(q_timestamps) == len(s):
            src_utc = _to_utc_index(q_timestamps)

        if src_utc is not None and x_utc is not None:
            timed = pd.Series(s.to_numpy(dtype=float), index=src_utc)
            return pd.Series(timed.reindex(x_utc).to_numpy(dtype=float), index=x_idx)
        if len(s) >= len(x_idx):
            return pd.Series(s.iloc[: len(x_idx)].to_numpy(dtype=float), index=x_idx)
        if len(s) > 0:
            padded = np.full(len(x_idx), np.nan, dtype=float)
            padded[: len(s)] = s.to_numpy(dtype=float)
            return pd.Series(padded, index=x_idx)
    except (TypeError, ValueError):
        return None
    return None


def _add_pv_quantile_traces(
    fig: go.Figure,
    x_idx: pd.Index,
    p10: pd.Series | None,
    p50: pd.Series | None,
    p90: pd.Series | None,
) -> None:
    """Add P90, P10, and P50 traces with shaded uncertainty bands to a Plotly figure."""
    if p90 is not None:
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=p90,
                name="PV P90 (optimistic)",
                line={"color": "rgba(245, 158, 11, 0.55)", "width": 1.5, "dash": "dot"},
                mode="lines",
            )
        )
    if p10 is not None and p90 is not None:
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=p10,
                name=_PV_P10_LABEL,
                line={"color": "rgba(217, 119, 6, 0.7)", "width": 1.5, "dash": "dot"},
                fill="tonexty",
                fillcolor="rgba(245, 158, 11, 0.16)",
                mode="lines",
            )
        )
        if p50 is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=p50,
                    name=_PV_P50_LABEL,
                    line={"color": "#f59e0b", "width": 2, "dash": "dash"},
                    mode="lines",
                )
            )
        return
    if p50 is not None and p10 is not None:
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=p50,
                name=_PV_P50_LABEL,
                line={"color": "rgba(245, 158, 11, 0.85)", "width": 2, "dash": "dash"},
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=p10,
                name=_PV_P10_LABEL,
                line={"color": "rgba(180, 83, 9, 0.75)", "width": 1.5, "dash": "dot"},
                fill="tonexty",
                fillcolor="rgba(245, 158, 11, 0.18)",
                mode="lines",
            )
        )
        return
    if p50 is not None:
        trace_kwargs: dict = {
            "x": x_idx,
            "y": p50,
            "name": _PV_P50_LABEL,
            "line": {"color": "#f59e0b", "width": 2, "dash": "dash"},
            "mode": "lines",
        }
        if p90 is not None:
            trace_kwargs["fill"] = "tonexty"
            trace_kwargs["fillcolor"] = "rgba(245, 158, 11, 0.14)"
        fig.add_trace(go.Scatter(**trace_kwargs))
        return
    if p10 is not None:
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=p10,
                name=_PV_P10_LABEL,
                line={"color": "rgba(180, 83, 9, 0.75)", "width": 1.5, "dash": "dot"},
                mode="lines",
            )
        )


def _build_forecast_method_badges(
    has_pv: bool,
    has_load: bool,
    pv_quantiles: dict | None,
    optim_conf: dict | None,
) -> str:
    """Build HTML method-badge strip displayed above the PV & Load forecast chart."""
    conf = optim_conf if isinstance(optim_conf, dict) else {}
    q_dict = pv_quantiles if isinstance(pv_quantiles, dict) else {}
    badge_items: list[str] = []
    if has_pv or conf.get("set_use_pv"):
        pv_method = conf.get("weather_forecast_method") or q_dict.get("source")
        if pv_method:
            badge_items.append(
                f'<span class="method-badge method-pv">PV Forecast Method: '
                f"<strong>{html.escape(str(pv_method))}</strong></span>"
            )
    if q_dict.get("bias") is not None:
        badge_items.append(
            f'<span class="method-badge method-bias">PV Quantile Bias (P10/P50): '
            f"<strong>{html.escape(str(q_dict['bias']))}</strong></span>"
        )
    elif conf.get("weather_forecast_pv_quantile_bias"):
        badge_items.append(
            f'<span class="method-badge method-bias">PV Quantile Bias: '
            f"<strong>{html.escape(str(conf['weather_forecast_pv_quantile_bias']))}</strong></span>"
        )
    if has_load and conf.get("load_forecast_method"):
        badge_items.append(
            f'<span class="method-badge method-load">Load Forecast Method: '
            f"<strong>{html.escape(str(conf['load_forecast_method']))}</strong></span>"
        )
    if conf.get("costfun"):
        badge_items.append(
            f'<span class="method-badge method-costfun">Objective: '
            f"<strong>{html.escape(str(conf['costfun']))}</strong></span>"
        )
    return f'<div class="plot-method-badges">{"".join(badge_items)}</div>' if badge_items else ""


def _format_pv_forecast_title(has_q: bool, q_dict: dict) -> str:
    """Format the PV & Load forecast chart title."""
    if not has_q:
        return "PV & Load Forecast Overview"
    src = str(q_dict.get("source") or "quantiles").capitalize()
    bias = q_dict.get("bias")
    bias_txt = f", bias={bias}" if bias is not None else ""
    return f"PV & Load Forecast with Uncertainty Quantiles ({src}{bias_txt})"


def _build_pv_forecast_quantiles_figure(
    df: pd.DataFrame,
    pv_quantiles: dict | None,
    optim_conf: dict | None = None,
) -> str | None:
    """Build PV & Load forecast chart with optional P10/P50/P90 uncertainty bands and method badges."""
    has_pv = "P_PV" in df.columns
    has_load = "P_Load" in df.columns
    q_dict = pv_quantiles if isinstance(pv_quantiles, dict) else {}
    has_q = any(q_dict.get(k) is not None for k in ("p50", "p10", "p90"))
    if not (has_pv or has_load or has_q):
        return None

    fig = go.Figure()
    x_idx = df.index
    x_utc = _to_utc_index(x_idx)
    q_timestamps = q_dict.get("timestamps")

    p10 = _align_pv_quantile_series(q_dict.get("p10"), x_idx, x_utc, q_timestamps)
    p50 = _align_pv_quantile_series(q_dict.get("p50"), x_idx, x_utc, q_timestamps)
    p90 = _align_pv_quantile_series(q_dict.get("p90"), x_idx, x_utc, q_timestamps)
    _add_pv_quantile_traces(fig, x_idx, p10, p50, p90)

    if has_pv:
        pv_series = pd.to_numeric(df["P_PV"], errors="coerce").astype(float)
        pv_name = "PV Forecast Used (P_PV)" if has_q else "PV Forecast (P_PV)"
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=pv_series,
                name=pv_name,
                line={"color": "#d97706", "width": 2.5, "shape": "hv"},
                mode="lines",
            )
        )

    if has_load:
        load_series = pd.to_numeric(df["P_Load"], errors="coerce").astype(float)
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=load_series,
                name="Base Load Forecast (P_Load)",
                line={"color": "#2563eb", "width": 2, "shape": "hv"},
                mode="lines",
            )
        )

    fig.update_layout(
        title=_format_pv_forecast_title(has_q, q_dict),
        template="presentation",
        xaxis_title="Timestamp",
        yaxis_title="Power (W)",
    )
    chart_html = _fig_to_html(fig, include_plotlyjs=False)
    badges_html = _build_forecast_method_badges(has_pv, has_load, pv_quantiles, optim_conf)
    return badges_html + chart_html


def _attribute_battery_charging(
    df: pd.DataFrame,
    charge_w: pd.Series,
    pv_w: pd.Series,
    total_load_w: pd.Series,
    grid_import_w: pd.Series,
    is_hybrid: bool,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Attribute battery charging power to PV, grid, or unknown/other bus flows."""
    if "P_hybrid_inverter" in df.columns:
        # Hybrid topology: P_hybrid_inverter == ac_out - ac_in (negative means AC->DC charging).
        p_inv = pd.to_numeric(df["P_hybrid_inverter"], errors="coerce").astype(float).fillna(0.0)
        ac_to_dc = (-p_inv).clip(lower=0.0)
        dc_to_ac = p_inv.clip(lower=0.0)
        pv_for_batt = (pv_w - dc_to_ac).clip(lower=0.0)
        charge_from_pv = np.minimum(charge_w, pv_for_batt)
        rem_charge = (charge_w - charge_from_pv).clip(lower=0.0)
        charge_from_grid = np.minimum(rem_charge, np.minimum(ac_to_dc, grid_import_w))
        charge_from_other = (rem_charge - charge_from_grid).clip(lower=0.0)
        return charge_from_pv, charge_from_grid, charge_from_other

    if is_hybrid:
        # Hybrid inverter configured without P_hybrid_inverter flow column: cannot prove AC->DC source.
        charge_from_pv = np.minimum(charge_w, pv_w)
        charge_from_grid = pd.Series(0.0, index=df.index)
        charge_from_other = (charge_w - charge_from_pv).clip(lower=0.0)
        return charge_from_pv, charge_from_grid, charge_from_other

    pv_excess = (pv_w - total_load_w).clip(lower=0.0)
    charge_from_pv = np.minimum(charge_w, pv_excess)
    rem_charge = (charge_w - charge_from_pv).clip(lower=0.0)
    charge_from_grid = np.minimum(rem_charge, grid_import_w)
    charge_from_other = (rem_charge - charge_from_grid).clip(lower=0.0)
    return charge_from_pv, charge_from_grid, charge_from_other


def _build_battery_sources_figure(df: pd.DataFrame, ui_meta: dict | None = None) -> str | None:
    """Build colored bar chart attributing battery charging sources and discharge destinations."""
    batt_flows = _battery_charge_discharge(df)
    if batt_flows is None:
        return None
    charge_w, discharge_w = batt_flows

    optim_conf = (ui_meta.get("optim_conf") or {}) if isinstance(ui_meta, dict) else {}
    plant_conf = (ui_meta.get("plant_conf") or {}) if isinstance(ui_meta, dict) else {}
    no_discharge_to_grid = bool(optim_conf.get("set_nodischarge_to_grid", False))
    is_hybrid = bool(
        plant_conf.get("inverter_is_hybrid", optim_conf.get("inverter_is_hybrid", False))
    )

    pv_w = (
        pd.to_numeric(df["P_PV"], errors="coerce").astype(float).fillna(0.0).clip(lower=0.0)
        if "P_PV" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    if "P_PV_curtailment" in df.columns:
        curt_w = (
            pd.to_numeric(df["P_PV_curtailment"], errors="coerce")
            .astype(float)
            .fillna(0.0)
            .clip(lower=0.0)
        )
        pv_w = (pv_w - curt_w).clip(lower=0.0)
    elec_def_cols = _electric_deferrable_cols(df, ui_meta)
    load_cols = (["P_Load"] if "P_Load" in df.columns else []) + elec_def_cols
    total_load_w = (
        df[load_cols].apply(pd.to_numeric, errors="coerce").astype(float).fillna(0.0).sum(axis=1)
        if load_cols
        else pd.Series(0.0, index=df.index)
    )
    grid_w = (
        pd.to_numeric(df["P_grid"], errors="coerce").astype(float).fillna(0.0)
        if "P_grid" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    grid_import_w = grid_w.clip(lower=0.0)
    grid_export_w = (-grid_w).clip(lower=0.0)

    charge_from_pv, charge_from_grid, charge_from_other = _attribute_battery_charging(
        df, charge_w, pv_w, total_load_w, grid_import_w, is_hybrid
    )

    if no_discharge_to_grid:
        discharge_to_grid = pd.Series(0.0, index=df.index)
        discharge_to_loads = discharge_w
    else:
        net_load_deficit = (total_load_w - pv_w).clip(lower=0.0)
        discharge_to_loads = np.minimum(discharge_w, net_load_deficit)
        rem_discharge = (discharge_w - discharge_to_loads).clip(lower=0.0)
        discharge_to_grid = np.minimum(rem_discharge, grid_export_w)
        # Any residual above grid_export_w (e.g. inverter/conversion losses) stays with loads
        discharge_to_loads = (discharge_w - discharge_to_grid).clip(lower=0.0)

    dt_h = _infer_timestep_hours(df.index, ui_meta)

    def _kwh(series: pd.Series) -> float:
        return float(series.sum() * dt_h / W_TO_KW)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=charge_from_pv,
            name=f"Charged from Excess PV ({_kwh(charge_from_pv):.2f} kWh)",
            marker_color="#16a34a",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=charge_from_grid,
            name=f"Charged from Grid ({_kwh(charge_from_grid):.2f} kWh)",
            marker_color="#ea580c",
        )
    )
    if float(charge_from_other.sum()) > 1e-6:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=charge_from_other,
                name=f"Charged from Other / DC Bus ({_kwh(charge_from_other):.2f} kWh)",
                marker_color="#8b5cf6",
            )
        )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=-discharge_to_loads,
            name=f"Discharged to Home Loads ({_kwh(discharge_to_loads):.2f} kWh)",
            marker_color="#2563eb",
        )
    )
    if float(discharge_to_grid.sum()) > 1e-6:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=-discharge_to_grid,
                name=f"Discharged to Grid Export (est., {_kwh(discharge_to_grid):.2f} kWh)",
                marker_color="#0d9488",
            )
        )

    fig.update_layout(
        title="Battery Charging Sources & Discharging Destinations (Estimated)",
        template="presentation",
        barmode="relative",
        xaxis_title="Timestamp",
        yaxis_title="Battery Power (+ Charge / - Discharge, W)",
    )
    return _fig_to_html(fig, include_plotlyjs=False)


def _append_positive_kwh(
    labels: list[str], values: list[float], label: str, series: pd.Series, dt_h: float
) -> None:
    """Append rounded positive kWh energy total to label/value lists if > 1e-4 kWh."""
    kwh = float(series.fillna(0.0).sum() * dt_h / W_TO_KW)
    if kwh > 1e-4:
        labels.append(label)
        values.append(round(kwh, 3))


def _collect_energy_supply_and_demand(
    df: pd.DataFrame, dt_h: float, ui_meta: dict | None
) -> tuple[list[str], list[float], list[str], list[float]]:
    """Aggregate energy supply and allocation kWh buckets from an optimization result DataFrame."""
    supply_labels: list[str] = []
    supply_values: list[float] = []
    demand_labels: list[str] = []
    demand_values: list[float] = []

    if "P_PV" in df.columns:
        pv_pos = pd.to_numeric(df["P_PV"], errors="coerce").astype(float).clip(lower=0.0)
        _append_positive_kwh(supply_labels, supply_values, "PV Generation", pv_pos, dt_h)
    if "P_PV_curtailment" in df.columns:
        curt = pd.to_numeric(df["P_PV_curtailment"], errors="coerce").astype(float).clip(lower=0.0)
        _append_positive_kwh(demand_labels, demand_values, "PV Curtailment", curt, dt_h)
    if "P_grid" in df.columns:
        grid = pd.to_numeric(df["P_grid"], errors="coerce").astype(float).fillna(0.0)
        _append_positive_kwh(
            supply_labels, supply_values, "Grid Import", grid.clip(lower=0.0), dt_h
        )
        _append_positive_kwh(
            demand_labels, demand_values, "Grid Export", (-grid).clip(lower=0.0), dt_h
        )

    batt_flows = _battery_charge_discharge(df)
    if batt_flows is not None:
        charge_w, discharge_w = batt_flows
        _append_positive_kwh(supply_labels, supply_values, "Battery Discharge", discharge_w, dt_h)
        _append_positive_kwh(demand_labels, demand_values, "Battery Charge", charge_w, dt_h)

    if "P_Load" in df.columns:
        base = pd.to_numeric(df["P_Load"], errors="coerce").astype(float).clip(lower=0.0)
        _append_positive_kwh(demand_labels, demand_values, "Base Home Load", base, dt_h)

    for col in _electric_deferrable_cols(df, ui_meta):
        def_s = pd.to_numeric(df[col], errors="coerce").astype(float).clip(lower=0.0)
        _append_positive_kwh(
            demand_labels,
            demand_values,
            col.replace("P_deferrable", "Deferrable Load "),
            def_s,
            dt_h,
        )

    # Account for inverter / DC-AC conversion losses when supply exceeds allocated AC/DC demand.
    unallocated_kwh = round(sum(supply_values) - sum(demand_values), 3)
    if ("P_hybrid_inverter" in df.columns or batt_flows is not None) and unallocated_kwh > 1e-3:
        demand_labels.append("Inverter / Conversion Loss")
        demand_values.append(unallocated_kwh)

    return supply_labels, supply_values, demand_labels, demand_values


def _build_energy_pie_figure(df: pd.DataFrame, ui_meta: dict | None = None) -> str | None:
    """Build side-by-side kWh supply vs demand donut charts."""
    dt_h = _infer_timestep_hours(df.index, ui_meta)
    present_core = [c for c in ("P_PV", "P_Load", "P_grid") if c in df.columns]
    if len(present_core) < 2:
        return None

    supply_labels, supply_values, demand_labels, demand_values = _collect_energy_supply_and_demand(
        df, dt_h, ui_meta
    )
    if not supply_values and not demand_values:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("Energy Supply Mix (kWh)", "Energy Allocation Mix (kWh)"),
    )
    if supply_values:
        fig.add_trace(
            go.Pie(
                labels=supply_labels,
                values=supply_values,
                hole=0.42,
                textinfo="percent",
                textposition="inside",
                hoverinfo="label+value+percent",
                name="Supply",
            ),
            1,
            1,
        )
    if demand_values:
        fig.add_trace(
            go.Pie(
                labels=demand_labels,
                values=demand_values,
                hole=0.42,
                textinfo="percent",
                textposition="inside",
                hoverinfo="label+value+percent",
                name="Allocation",
            ),
            1,
            2,
        )
    fig.update_annotations(font_size=13, yshift=10)
    fig.update_layout(
        title="Energy Balance & Allocation Breakdown (kWh)",
        template="presentation",
        height=480,
    )
    return _fig_to_html(fig, include_plotlyjs=False)


def _build_histograms_figure(df: pd.DataFrame) -> str | None:
    """Build overlaid distribution histograms for core system power series."""
    candidate_cols = [
        c
        for c in df.columns
        if c not in _UI_DROP_PLOT_POWER_COLS
        and not c.startswith("fed_node_power_")
        and (
            c in ("P_PV", "P_Load", "P_grid", "P_batt")
            or (c.startswith("P_batt_") and c[len("P_batt_") :].isdigit())
            or c.startswith("P_deferrable")
        )
    ]
    if not candidate_cols:
        return None

    fig = go.Figure()
    plotted = 0
    for col in candidate_cols[:6]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=vals,
                name=col,
                opacity=0.6,
                nbinsx=min(24, max(8, len(vals) // 2)),
            )
        )
        plotted += 1

    if plotted == 0:
        return None

    fig.update_layout(
        title="Power Distribution Histograms",
        template="presentation",
        barmode="overlay",
        xaxis_title="Power (W)",
        yaxis_title="Timestep Count",
    )
    return _fig_to_html(fig, include_plotlyjs=False)


def _build_heatmap_axis(index: pd.Index) -> tuple[list, list[str]]:
    """Build unique UTC-safe x-coordinates and local-with-offset display labels for heatmaps."""
    x_utc = _to_utc_index(index)
    display_labels: list[str] = []
    for ts in index:
        if hasattr(ts, "strftime"):
            fmt = "%m-%d %H:%M %z" if getattr(ts, "tzinfo", None) is not None else "%m-%d %H:%M"
            display_labels.append(ts.strftime(fmt))
        else:
            display_labels.append(str(ts))
    if x_utc is not None:
        return [ts.isoformat() for ts in x_utc], display_labels
    return display_labels, display_labels


def _build_heatmap_figure(df: pd.DataFrame, cols_p: list[str]) -> str | None:
    """Build 2D schedule heatmap of system powers across timesteps."""
    hm_cols = [c for c in cols_p if c in df.columns][:10]
    if len(df) < 2 or len(hm_cols) < 2:
        return None
    matrix_df = df[hm_cols].apply(pd.to_numeric, errors="coerce")
    if not matrix_df.notna().any().any():
        return None

    x_coords, x_labels = _build_heatmap_axis(df.index)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.T.to_numpy(dtype=float),
            x=x_coords,
            y=hm_cols,
            customdata=[x_labels for _ in hm_cols],
            hovertemplate="Time: %{customdata}<br>Channel: %{y}<br>Power: %{z:.1f} W<extra></extra>",
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "W"},
        )
    )
    tick_step = max(1, len(x_coords) // 12)
    fig.update_layout(
        title="System Power Matrix Heatmap",
        template="presentation",
        xaxis_title="Timestamp",
        yaxis_title="Channel",
        xaxis={
            "type": "category",
            "tickmode": "array",
            "tickvals": x_coords[::tick_step],
            "ticktext": x_labels[::tick_step],
        },
    )
    return _fig_to_html(fig, include_plotlyjs=False)


def _build_load_effectiveness_figure(df: pd.DataFrame, ui_meta: dict) -> str | None:
    """Build load forecast effectiveness & diurnal/statistical profile plot."""
    if "P_Load" not in df.columns:
        return None
    pred = pd.to_numeric(df["P_Load"], errors="coerce").astype(float).dropna()
    if pred.empty:
        return None

    optim_conf = ui_meta.get("optim_conf") or {}
    method = str(optim_conf.get("load_forecast_method") or "configured")
    hist_raw = ui_meta.get("load_history_series")

    fig = go.Figure()
    x_idx = df.index
    pred_full = pd.to_numeric(df["P_Load"], errors="coerce").astype(float)
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=pred_full,
            name=f"Forecast P_Load ({method})",
            line={"color": "#2563eb", "width": 2.5, "shape": "hv"},
            mode="lines",
        )
    )

    subtitle = f"Method: {method} | Mean: {pred.mean():.0f} W | Peak: {pred.max():.0f} W"
    matched_baseline = False
    if isinstance(hist_raw, dict) and "values" in hist_raw and "timestamps" in hist_raw:
        try:
            hist_ts = pd.to_datetime(hist_raw["timestamps"], errors="coerce", utc=True)
            hist_raw = pd.Series(hist_raw["values"], index=hist_ts)
        except Exception:
            hist_raw = None
    if isinstance(hist_raw, pd.Series) and len(hist_raw.dropna()) >= 4:
        hist_clean = pd.to_numeric(hist_raw, errors="coerce").dropna()
        try:
            hist_idx = pd.DatetimeIndex(hist_clean.index)
            fcst_idx = pd.DatetimeIndex(x_idx)
            if fcst_idx.tz is not None:
                if hist_idx.tz is None:
                    hist_idx = hist_idx.tz_localize("UTC")
                hist_idx = hist_idx.tz_convert(fcst_idx.tz)
            elif hist_idx.tz is not None:
                hist_idx = hist_idx.tz_convert("UTC")
                fcst_idx = fcst_idx.tz_localize("UTC")
            # Align by (hour, minute) diurnal median + IQR band as benchmark
            slot_df = pd.DataFrame(
                {
                    "val": hist_clean.to_numpy(dtype=float),
                    "hm": list(zip(hist_idx.hour, hist_idx.minute, strict=True)),
                }
            )
            grouped = slot_df.groupby("hm")["val"]
            med_map = grouped.median().to_dict()
            q25_map = grouped.quantile(0.25).to_dict()
            q75_map = grouped.quantile(0.75).to_dict()

            fcst_hm = list(zip(fcst_idx.hour, fcst_idx.minute, strict=True))
            baseline = pd.Series([med_map.get(k, np.nan) for k in fcst_hm], index=x_idx)
            q25 = pd.Series([q25_map.get(k, np.nan) for k in fcst_hm], index=x_idx)
            q75 = pd.Series([q75_map.get(k, np.nan) for k in fcst_hm], index=x_idx)

            if baseline.notna().sum() >= 2:
                matched_baseline = True
                fig.add_trace(
                    go.Scatter(
                        x=x_idx,
                        y=q75,
                        name="Historical IQR (75%)",
                        line={"color": "rgba(100, 116, 139, 0.35)", "width": 1},
                        mode="lines",
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_idx,
                        y=q25,
                        name="Historical IQR (25%)",
                        line={"color": "rgba(100, 116, 139, 0.35)", "width": 1},
                        fill="tonexty",
                        fillcolor="rgba(148, 163, 184, 0.22)",
                        mode="lines",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_idx,
                        y=baseline,
                        name="Historical Diurnal Median",
                        line={"color": "#16a34a", "width": 2, "dash": "dash"},
                        mode="lines",
                    )
                )
                metrics = compute_forecast_metrics(baseline, pred_full)
                mape_str = f"{metrics['mape']:.1f}%" if np.isfinite(metrics["mape"]) else "n/a"
                r2_str = f"{metrics['r2']:.2f}" if np.isfinite(metrics["r2"]) else "n/a"
                subtitle = (
                    f"Method: {method} | vs Diurnal Baseline (n={metrics['n_samples']}): "
                    f"MAE={metrics['mae']:.0f} W, RMSE={metrics['rmse']:.0f} W, "
                    f"MAPE={mape_str}, R2={r2_str}"
                )
        except Exception:
            pass
    if not matched_baseline:
        win = min(6, max(2, len(pred_full)))
        rolling_mean = pred_full.rolling(window=win, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=rolling_mean,
                name="Rolling Mean",
                line={"color": "#0d9488", "width": 2, "dash": "dash"},
                mode="lines",
            )
        )

    fig.update_layout(
        title=f"Load Forecast Effectiveness<br><sup>{subtitle}</sup>",
        template="presentation",
        xaxis_title="Timestamp",
        yaxis_title="Load Power (W)",
    )
    return _fig_to_html(fig, include_plotlyjs=False)


def _build_federated_figures(df: pd.DataFrame) -> dict[str, str]:
    """Build dedicated Federated Coordinator figures when fed_* columns exist."""
    fed_figs: dict[str, str] = {}

    price_cols = [
        c
        for c in df.columns
        if c == "fed_meter_price"
        or c.startswith("fed_local_price_")
        or c.startswith("fed_limit_price_")
    ]
    if price_cols:
        price_df = df[price_cols].apply(pd.to_numeric, errors="coerce")
        if price_df.notna().any().any():
            fig_prices = px.line(
                price_df,
                title="Federated Coordinator Dual & Shadow Prices",
                template="presentation",
                line_shape="hv",
                render_mode="svg",
            )
            fig_prices.update_layout(
                xaxis_title="Timestamp",
                yaxis_title="Shadow Price (currency/kWh)",
            )
            fed_figs["figure_fed_prices"] = _fig_to_html(fig_prices, include_plotlyjs=False)

    share_cols = [c for c in df.columns if c.startswith("fed_share_")]
    if share_cols:
        shares = {
            c.replace("fed_share_", ""): float(pd.to_numeric(df[c], errors="coerce").iloc[0])
            for c in share_cols
            if pd.to_numeric(df[c], errors="coerce").notna().any()
        }
        if shares:
            meta_parts = []
            for scalar_col, label in (
                ("fed_iterations", "iters"),
                ("fed_gap", "gap"),
                ("fed_lower_bound", "LB"),
            ):
                if scalar_col in df.columns:
                    val = pd.to_numeric(df[scalar_col], errors="coerce").iloc[0]
                    if pd.notna(val):
                        fmt = (
                            f"{int(val)}" if scalar_col == "fed_iterations" else f"{float(val):.4f}"
                        )
                        meta_parts.append(f"{label}={fmt}")
            meta_txt = f" ({', '.join(meta_parts)})" if meta_parts else ""
            fig_shares = go.Figure(
                data=go.Bar(
                    x=list(shares.keys()),
                    y=list(shares.values()),
                    text=[f"{v:.4f}" for v in shares.values()],
                    textposition="auto",
                    marker_color="#0ea5e9",
                )
            )
            fig_shares.update_layout(
                title=f"Federated Device Cost / Value Shares{meta_txt}",
                template="presentation",
                xaxis_title="Device / Group",
                yaxis_title="Allocated Cost / Share",
            )
            fed_figs["figure_fed_shares"] = _fig_to_html(fig_shares, include_plotlyjs=False)

    node_cols = [c for c in df.columns if c.startswith("fed_node_power_")]
    if node_cols:
        node_df = df[node_cols].apply(pd.to_numeric, errors="coerce")
        if node_df.notna().any().any():
            fig_nodes = px.line(
                node_df,
                title="Federated Sub-Meter / Node Power Flows",
                template="presentation",
                line_shape="hv",
                render_mode="svg",
            )
            fig_nodes.update_layout(xaxis_title="Timestamp", yaxis_title="Node Power (W)")
            fed_figs["figure_fed_nodes"] = _fig_to_html(fig_nodes, include_plotlyjs=False)

    return fed_figs


def _build_stage_times_figure(ui_meta: dict) -> str | None:
    """Build horizontal bar chart of pipeline stage execution timings from last_run metadata."""
    last_run_snap = ui_meta.get("last_run")
    if not isinstance(last_run_snap, dict):
        return None
    stage_times = last_run_snap.get("stage_times")
    if not isinstance(stage_times, dict) or not stage_times:
        return None

    ordered_keys = [
        "pv_forecast",
        "load_forecast",
        "price_prep",
        "optim_solve.build",
        "optim_solve.solve",
        "optim_solve.extract",
        "optim_solve",
        "publish",
    ]
    labels: list[str] = []
    values: list[float] = []
    seen = set()
    for key in ordered_keys + sorted(stage_times.keys()):
        if key in seen or key not in stage_times:
            continue
        seen.add(key)
        raw_val = stage_times[key]
        if isinstance(raw_val, int | float) and math.isfinite(float(raw_val)):
            labels.append(key)
            values.append(float(raw_val))

    if not labels:
        return None

    total_sec = last_run_snap.get("duration_total_seconds")
    total_txt = (
        f" — Total: {float(total_sec):.3f} s"
        if isinstance(total_sec, int | float) and math.isfinite(float(total_sec))
        else ""
    )
    colors = ["#0284c7" if "." in lbl else "#0f766e" for lbl in labels]
    fig = go.Figure(
        data=go.Bar(
            x=values[::-1],
            y=labels[::-1],
            orientation="h",
            text=[f"{v:.3f} s ({v * 1000:.0f} ms)" for v in values[::-1]],
            textposition="auto",
            marker_color=colors[::-1],
        )
    )
    fig.update_layout(
        title=f"Pipeline Stage Execution Times{total_txt}",
        template="presentation",
        xaxis_title="Duration (seconds)",
        yaxis_title="Pipeline Stage",
    )
    return _fig_to_html(fig, include_plotlyjs=False)


def get_injection_dict(df: pd.DataFrame, plot_size: int | None = 1366) -> dict:
    """
    Build a dictionary with graphs and tables for the webui.

    :param df: The optimization result DataFrame
    :type df: pd.DataFrame
    :param plot_size: Size of the plot figure in pixels, defaults to 1366
    :type plot_size: Optional[int], optional
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict

    """
    ui_meta = dict(df.attrs.get("emhass_ui_meta") or {})
    df = df.copy()
    # Clear attrs on the local copy before slicing/astype so pandas __finalize__
    # never compares embedded pd.Series objects inside attrs (#attrs-bool-error).
    df.attrs = {}
    cols_p = [
        i
        for i in df.columns.to_list()
        if "P_" in i and i not in _UI_DROP_PLOT_POWER_COLS and not i.startswith("fed_node_power_")
    ]
    # Let's round the data in the DF
    if "optim_status" in df.columns:
        optim_status = df["optim_status"].iloc[0]
        df.drop("optim_status", axis=1, inplace=True)
    else:
        optim_status = "Status not available"
    # Other text columns (with a coordinator backend: backend_used,
    # backend_fallback_reason, fed_stop_reason) hold one value for the whole
    # plan, so they go in the summary table instead of the plots.
    text_cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and (pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()).any()
    ]
    summary_text = {c: df[c].iloc[0] for c in text_cols}
    df = df.drop(columns=text_cols)
    cols_p = [i for i in cols_p if i not in text_cols]
    cols_else = [i for i in df.columns.to_list() if "P_" not in i]
    df = df.apply(pd.to_numeric)
    # Keep unrounded float powers for analytical figures while rounding integer Watts for main display/table.
    df_num = df.copy()
    df_num[cols_else] = df_num[cols_else].round(3)
    all_power_cols = [i for i in df.columns.to_list() if "P_" in i]
    df[all_power_cols] = df[all_power_cols].round().astype("Int64")
    df[cols_else] = df[cols_else].round(3)

    # Create plots
    # Figure 0: Systems Powers (loads Plotly.js once for all figures in the template)
    n_colors = len(cols_p)
    colors = px.colors.sample_colorscale(
        "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
    )
    fig_0 = px.line(
        df[cols_p],
        title="Systems powers schedule after optimization results",
        template="presentation",
        line_shape="hv",
        color_discrete_sequence=colors,
        render_mode="svg",
    )
    fig_0.update_layout(xaxis_title="Timestamp", yaxis_title="System powers (W)")
    image_path_0 = _fig_to_html(fig_0, include_plotlyjs=True)

    # Figure PV Forecast Quantiles / Overview
    image_path_pv_forecast = _build_pv_forecast_quantiles_figure(
        df_num, ui_meta.get("pv_quantiles"), ui_meta.get("optim_conf")
    )

    # Figure 1: Battery SOC (Optional)
    # One trace per SOC column: the bare "SOC_opt" with one battery, or "SOC_opt_<k>" per
    # battery when number_of_batteries > 1 (#610), where no bare "SOC_opt" exists.
    image_path_1 = None
    cols_soc = [
        i
        for i in df.columns.to_list()
        if i == "SOC_opt" or (i.startswith("SOC_opt_") and i[len("SOC_opt_") :].isdigit())
    ]
    if cols_soc:
        # Size the palette to the SOC traces, not the power columns above: with one
        # battery this is the same first colour as before, with several it spreads them.
        n_colors = len(cols_soc)
        colors_soc = px.colors.sample_colorscale(
            "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
        )
        fig_1 = px.line(
            df[cols_soc],
            title="Battery state of charge schedule after optimization results",
            template="presentation",
            line_shape="hv",
            color_discrete_sequence=colors_soc,
            render_mode="svg",
        )
        fig_1.update_layout(xaxis_title="Timestamp", yaxis_title="Battery SOC (%)")
        image_path_1 = _fig_to_html(fig_1, include_plotlyjs=False)

    # Figure Battery Sources (Optional, when battery power is present)
    image_path_battery_sources = _build_battery_sources_figure(df_num, ui_meta)

    # Figure Thermal: Temperatures (Optional)
    cols_temp = [
        i
        for i in df.columns.to_list()
        if "predicted_temp_heater" in i
        or "target_temp_heater" in i
        or "min_temp_heater" in i
        or "max_temp_heater" in i
    ]
    image_path_temp = None
    if len(cols_temp) > 0:
        n_colors = len(cols_temp)
        colors = px.colors.sample_colorscale(
            "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
        )
        fig_temp = px.line(
            df[cols_temp],
            title="Thermal loads temperature schedule",
            template="presentation",
            line_shape="hv",
            color_discrete_sequence=colors,
            render_mode="svg",
        )
        fig_temp.update_layout(xaxis_title="Timestamp", yaxis_title="Temperature (&deg;C)")
        image_path_temp = _fig_to_html(fig_temp, include_plotlyjs=False)

    # Figure 2: Costs
    cols_cost = [i for i in df.columns.to_list() if "cost_" in i or "unit_" in i]
    n_colors = len(cols_cost)
    colors = px.colors.sample_colorscale(
        "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
    )
    fig_2 = px.line(
        df[cols_cost],
        title="Systems costs obtained from optimization results",
        template="presentation",
        line_shape="hv",
        color_discrete_sequence=colors,
        render_mode="svg",
    )
    fig_2.update_layout(xaxis_title="Timestamp", yaxis_title="System costs (currency)")
    image_path_2 = _fig_to_html(fig_2, include_plotlyjs=False)

    # Advanced analytical figures
    image_path_energy_pie = _build_energy_pie_figure(df_num, ui_meta)
    image_path_histograms = _build_histograms_figure(df_num)
    image_path_heatmap = _build_heatmap_figure(df_num, cols_p)
    image_path_load_eff = _build_load_effectiveness_figure(df_num, ui_meta)
    fed_figures = _build_federated_figures(df_num)
    image_path_stage_times = _build_stage_times_figure(ui_meta)

    # Scalar federated metrics go into Summary Table 2 instead of repeating across rows in Table 1
    fed_scalar_cols = [
        c
        for c in df.columns
        if c in ("fed_lower_bound", "fed_gap", "fed_iterations") or c.startswith("fed_share_")
    ]
    summary_scalars = {c: df[c].iloc[0] for c in fed_scalar_cols if not df.empty}
    drop_from_table1 = [c for c in df.columns if c in _UI_DROP_TABLE_COLS or c in fed_scalar_cols]
    df_table1 = df.drop(columns=drop_from_table1) if drop_from_table1 else df

    # Tables
    table1 = df_table1.reset_index().to_html(classes="mystyle", index=False)
    cost_cols = [i for i in df.columns if "cost_" in i]
    table2 = df[cost_cols].reset_index().sum(numeric_only=True)
    table2["optim_status"] = optim_status
    for col, value in summary_text.items():
        table2[col] = value
    for col, value in summary_scalars.items():
        table2[col] = value
    table2 = (
        table2.to_frame(name="Value")
        .reset_index(names="Variable")
        .to_html(classes="mystyle", index=False)
    )

    # Construct Injection Dict
    injection_dict = {}
    injection_dict["title"] = "<h2>EMHASS optimization results</h2>"
    injection_dict["subsubtitle0"] = "<h4>Plotting latest optimization results</h4>"
    injection_dict["figure_0"] = image_path_0
    if image_path_pv_forecast is not None:
        injection_dict["figure_pv_forecast"] = image_path_pv_forecast
    if image_path_temp is not None:
        injection_dict["figure_thermal"] = image_path_temp
    if image_path_1 is not None:
        injection_dict["figure_1"] = image_path_1
    if image_path_battery_sources is not None:
        injection_dict["figure_battery_sources"] = image_path_battery_sources
    injection_dict["figure_2"] = image_path_2
    if any(
        x is not None
        for x in (
            image_path_energy_pie,
            image_path_histograms,
            image_path_heatmap,
            image_path_load_eff,
        )
    ):
        injection_dict["analytics_intro"] = (
            '<div class="analytics-explainer-card">'
            "<h4>How to Read Advanced Analytics</h4>"
            "<ul>"
            "<li><strong>Energy Balance Donuts (kWh):</strong> Compares where energy came from "
            "(PV generation, grid import, battery discharge) against where it was consumed or stored "
            "(base home load, each deferrable load, battery charge, grid export) over the horizon.</li>"
            "<li><strong>Power Distribution Histograms:</strong> Shows how frequently each asset operates "
            "at given wattage levels—useful to spot standby base loads vs peak appliance plateaus.</li>"
            "<li><strong>System Power Matrix Heatmap:</strong> Displays the 2D time-by-channel flow intensity "
            "(red/positive = supply/import/discharge, blue/negative = export/battery charging).</li>"
            "<li><strong>Load Forecast Effectiveness:</strong> Compares predicted base load against your "
            "historical diurnal median and IQR band (with MAE, RMSE, MAPE, and R² baseline-deviation metrics "
            "relative to that historical baseline).</li>"
            "</ul></div>"
        )
    if image_path_energy_pie is not None:
        injection_dict["figure_energy_pie"] = image_path_energy_pie
    if image_path_histograms is not None:
        injection_dict["figure_histograms"] = image_path_histograms
    if image_path_heatmap is not None:
        injection_dict["figure_heatmap"] = image_path_heatmap
    if image_path_load_eff is not None:
        injection_dict["figure_load_effectiveness"] = image_path_load_eff
    for fed_key, fed_html in fed_figures.items():
        injection_dict[fed_key] = fed_html
    if image_path_stage_times is not None:
        injection_dict["figure_stage_times"] = image_path_stage_times
    injection_dict["subsubtitle1"] = "<h4>Last run optimization results table</h4>"
    injection_dict["table1"] = table1
    injection_dict["subsubtitle2"] = "<h4>Summary table for latest optimization results</h4>"
    injection_dict["table2"] = table2
    return injection_dict


def compute_forecast_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Compute goodness-of-fit metrics for a forecast against realised values.

    Only rows where both the actual and the predicted value are present are
    used; any leading warm-up NaNs are excluded. The guards mirror the degenerate
    cases the metrics can hit: r2_score requires at least 2 samples, MAE/RMSE
    require at least 1, and MAPE excludes zero actuals to avoid division by zero.

    :param actual: The realised values.
    :type actual: pd.Series | np.ndarray
    :param predicted: The forecasted values (same index/length as ``actual``).
    :type predicted: pd.Series | np.ndarray
    :param logger: Optional logger for the degenerate-case warning.
    :type logger: logging.Logger, optional
    :return: A dict with keys mae, rmse, r2, mape, n_samples.
    :rtype: dict
    """
    actual_values = pd.Series(np.asarray(actual, dtype=float))
    predicted_values = pd.Series(np.asarray(predicted, dtype=float))
    valid_mask = predicted_values.notna() & actual_values.notna()
    n_valid_samples = int(valid_mask.sum())
    actual_valid = actual_values[valid_mask]
    predicted_valid = predicted_values[valid_mask]

    # Guard against degenerate cases (all-NaN predictions or single sample).
    # r2_score requires at least 2 samples; MAE/RMSE require at least 1.
    if n_valid_samples == 0:
        if logger is not None:
            logger.warning("Forecast metrics: no valid predictions - metrics set to NaN")
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "mape": float("nan"),
            "n_samples": 0,
        }
    mae = float(mean_absolute_error(actual_valid, predicted_valid))
    rmse = float(np.sqrt(mean_squared_error(actual_valid, predicted_valid)))
    # r2_score is undefined for a single sample (variance == 0)
    r2 = float(r2_score(actual_valid, predicted_valid)) if n_valid_samples > 1 else float("nan")
    # MAPE: exclude zero actuals to avoid division by zero
    nonzero_mask = actual_valid != 0
    if nonzero_mask.sum() > 0:
        mape = float(
            np.mean(
                np.abs(
                    (actual_valid[nonzero_mask] - predicted_valid[nonzero_mask])
                    / actual_valid[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "n_samples": n_valid_samples,
    }


def get_injection_dict_forecast_model_fit(df_fit_pred: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF fit case.

    :param df_fit_pred: The fit result DataFrame
    :type df_fit_pred: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
    fig = df_fit_pred.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=mlf.model_type)
    fig.update_xaxes(title_text="Time")
    image_path_0 = _fig_to_html(fig, include_plotlyjs=True)
    # The dict of plots
    injection_dict = {}
    injection_dict["title"] = "<h2>Custom machine learning forecast model fit</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Plotting train/test forecast model results for "
        + html.escape(str(mlf.model_type))
        + "<br>"
        + "Forecasting variable "
        + html.escape(str(mlf.var_model))
        + "</h4>"
    )
    injection_dict["figure_0"] = image_path_0
    return injection_dict


def _build_forecast_calibration_metrics_figure(table_df: pd.DataFrame) -> str | None:
    """Build grouped bar chart comparing forecast calibration MAE & RMSE across methods."""
    if not isinstance(table_df, pd.DataFrame) or table_df.empty or "method" not in table_df.columns:
        return None
    metric_fig = go.Figure()
    methods = table_df["method"].astype(str).tolist()
    for col, label, color in (
        ("test_mae", "Test MAE (W)", "#0284c7"),
        ("val_mae", "Validation MAE (W)", "#0f766e"),
        ("test_rmse", "Test RMSE (W)", "#f59e0b"),
        ("val_rmse", "Validation RMSE (W)", "#dc2626"),
    ):
        if col not in table_df.columns:
            continue
        vals = pd.to_numeric(table_df[col], errors="coerce")
        if vals.notna().any():
            metric_fig.add_trace(
                go.Bar(
                    x=methods,
                    y=vals,
                    name=label,
                    marker_color=color,
                    text=[f"{v:.1f}" if pd.notna(v) else "N/A" for v in vals],
                    textposition="auto",
                )
            )
    if len(metric_fig.data) == 0:
        return None
    metric_fig.update_layout(
        title="Forecast Error Comparison Across Methods (MAE & RMSE)",
        template="presentation",
        barmode="group",
        xaxis_title="Load Forecast Method",
        yaxis_title="Error (W)",
    )
    return _fig_to_html(metric_fig, include_plotlyjs=False)


def get_injection_dict_forecast_calibration(result: dict) -> dict:
    """
    Build the webui graphs + metrics table for the forecast-calibration action.

    :param result: The dict returned by ``compute_forecast_calibration``
        (keys: ``table``, ``plot``, ``val_window``, ``caveats``).
    :type result: dict
    :return: A dictionary containing the graphs and table in html format
    :rtype: dict
    """
    plot_df = result["plot"]
    fig = plot_df.plot()
    fig.update_layout(
        title="Load Forecast Calibration — Actual vs Forecast Methods",
        template="presentation",
        xaxis_title="Timestamp",
        yaxis_title="Load Power (W)",
    )
    figure_0 = _fig_to_html(fig, include_plotlyjs=True)

    table_df = result["table"]
    metrics_fig_html = _build_forecast_calibration_metrics_figure(table_df)

    table1 = table_df.to_html(classes="mystyle", index=False)
    val_start, val_end = result["val_window"]
    injection_dict = {}
    injection_dict["title"] = "<h2>Load forecast calibration</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Actual vs forecast methods over the validation window "
        + f"({val_start} to {val_end})</h4>"
    )
    injection_dict["figure_0"] = figure_0
    if metrics_fig_html is not None:
        injection_dict["figure_calibration_metrics"] = metrics_fig_html
    injection_dict["subsubtitle1"] = (
        "<h4>Accuracy metrics by method and split (train / test / val)</h4>"
    )
    injection_dict["table1"] = table1
    injection_dict["subsubtitle2"] = "<h5>" + html.escape(str(result["caveats"])) + "</h5>"
    return injection_dict


MIN_THERMAL_ROWS = 6
_THERMAL_ENTITY_ID_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?$")


def _emhass_thermal_step_delta(
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    alpha_h: float,
    gamma_c: float,
    dt_hours: float,
    nominal_power: float,
    sense: str = "heat",
) -> np.ndarray:
    """First-order EMHASS thermal balance step delta."""
    p_def, t_in, t_out = inputs
    sign = 1.0 if sense == "heat" else -1.0
    return sign * alpha_h * dt_hours * (p_def / nominal_power) - gamma_c * dt_hours * (t_in - t_out)


def _normalize_thermal_arrays(
    t_init_or_actual: float | np.ndarray,
    n: int,
    observed_mask: np.ndarray | None,
    step_contiguous: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize initial temperature, observation mask, and contiguity arrays."""
    t_actual = (
        np.full(n, float(t_init_or_actual), dtype=float)
        if np.ndim(t_init_or_actual) == 0
        else np.asarray(t_init_or_actual, dtype=float)
    )
    obs = np.ones(n, dtype=bool) if observed_mask is None else np.asarray(observed_mask, dtype=bool)
    if step_contiguous is None:
        contig = np.ones(n, dtype=bool)
        if n > 0:
            contig[-1] = False
    else:
        contig = np.asarray(step_contiguous, dtype=bool)
    return t_actual, obs, contig


def _simulate_thermal_trajectory(
    t_init_or_actual: float | np.ndarray,
    p_shifted: np.ndarray,
    t_out: np.ndarray,
    alpha_h: float,
    gamma_c: float,
    dt_hours: float,
    nominal_power: float,
    observed_mask: np.ndarray | None = None,
    step_contiguous: np.ndarray | None = None,
    sense: str = "heat",
) -> np.ndarray:
    """Simulate closed-loop multi-step indoor temperature trajectory over contiguous segments."""
    n = len(p_shifted)
    sign = 1.0 if sense == "heat" else -1.0
    t_actual, obs, contig = _normalize_thermal_arrays(
        t_init_or_actual, n, observed_mask, step_contiguous
    )
    t_sim = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if not obs[i]:
            continue
        is_anchor = i == 0 or not obs[i - 1] or not contig[i - 1] or not np.isfinite(t_sim[i])
        if is_anchor:
            t_sim[i] = t_actual[i]
        if contig[i] and i + 1 < n and obs[i + 1]:
            delta = sign * alpha_h * dt_hours * (
                p_shifted[i] / nominal_power
            ) - gamma_c * dt_hours * (t_sim[i] - t_out[i])
            t_sim[i + 1] = t_sim[i] + delta
    return t_sim


def _valid_thermal_entity_id(eid: str, logger: logging.Logger) -> str | None:
    """Validate and strip an entity_id before interpolating into an InfluxDB query."""
    cleaned = str(eid or "").strip()
    if not cleaned:
        return None
    if not _THERMAL_ENTITY_ID_RE.match(cleaned):
        logger.warning("Ignoring invalid thermal entity_id for InfluxDB query: %r", cleaned)
        return None
    return cleaned


async def _run_influx_query(
    session: aiohttp.ClientSession,
    conn: dict,
    query: str,
) -> pd.DataFrame:
    """Execute one InfluxDB HTTP query and concatenate its series into a time-indexed DataFrame."""
    async with session.get(
        conn["base_url"],
        params={"db": conn["db"], "q": query},
        auth=conn["auth"],
        ssl=conn["verify_ssl"],
    ) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    if not isinstance(payload, dict) or payload.get("error"):
        raise RuntimeError("InfluxDB query returned a top-level error")
    results = payload.get("results") or [{}]
    first_result = results[0] if isinstance(results[0], dict) else {}
    if first_result.get("error"):
        raise RuntimeError("InfluxDB query returned a statement error")
    series_list = first_result.get("series", [])
    if not series_list:
        return pd.DataFrame()
    frames = []
    for s in series_list:
        frame = pd.DataFrame(s.get("values", []), columns=s.get("columns", []))
        if "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"], utc=True)
            frame = frame.set_index("time")
        frames.append(frame)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


async def _find_influx_measurement(
    session: aiohttp.ClientSession,
    conn: dict,
    eid: str,
    logger: logging.Logger,
) -> str | None:
    """Resolve the InfluxDB measurement name for a Home Assistant entity_id."""
    valid = _valid_thermal_entity_id(eid, logger)
    if not valid:
        return None
    bare = valid.split(".", 1)[-1]
    for target_id in (valid, bare):
        df_s = await _run_influx_query(
            session, conn, f"SHOW SERIES WHERE \"entity_id\" = '{target_id}'"
        )
        if not df_s.empty:
            return str(df_s.iloc[0, 0]).split(",")[0]
    return None


async def _fetch_influx_numeric_series(
    session: aiohttp.ClientSession,
    conn: dict,
    eid: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    dt_minutes: int,
    logger: logging.Logger,
) -> pd.Series:
    """Fetch a resampled numeric series from InfluxDB for the given entity_id."""
    valid = _valid_thermal_entity_id(eid, logger)
    if not valid:
        return pd.Series(dtype=float)
    meas = await _find_influx_measurement(session, conn, valid, logger)
    if not meas:
        return pd.Series(dtype=float)
    bare = valid.split(".", 1)[-1]
    meas_esc = meas.replace("\\", "\\\\").replace('"', '\\"')
    q = (
        f'SELECT mean("value") AS "val" FROM "{meas_esc}" '
        f"WHERE (\"entity_id\" = '{valid}' OR \"entity_id\" = '{bare}') "
        f"AND time >= '{start_utc.strftime('%Y-%m-%d %H:%M:%S')}' "
        f"AND time <= '{end_utc.strftime('%Y-%m-%d %H:%M:%S')}' "
        f"GROUP BY time({dt_minutes}m) fill(none)"
    )
    res = await _run_influx_query(session, conn, q)
    return pd.to_numeric(res["val"], errors="coerce") if not res.empty else pd.Series(dtype=float)


def _bucket_on_fraction(ev_ns: np.ndarray, ev_vals: np.ndarray, b0: int, dt_ns: int) -> float:
    """Compute time-weighted ON fraction in [b0, b0 + dt_ns) from sorted event arrays."""
    b1 = b0 + dt_ns
    idx_before = int(np.searchsorted(ev_ns, b0, side="right")) - 1
    idx_end = int(np.searchsorted(ev_ns, b1, side="left"))
    if idx_before < 0 and idx_end <= 0:
        return float("nan")
    state = float(ev_vals[idx_before]) if idx_before >= 0 else 0.0
    cursor = b0
    on_ns = 0.0
    for k in range(max(0, idx_before + 1), idx_end):
        t_ev = int(ev_ns[k])
        if state > 0.0:
            on_ns += t_ev - cursor
        cursor = t_ev
        state = float(ev_vals[k])
    if state > 0.0:
        on_ns += b1 - cursor
    return on_ns / dt_ns


def _integrate_switch_duty(
    duty: pd.Series,
    target_index: pd.DatetimeIndex,
    dt_minutes: int,
) -> pd.Series:
    """Compute time-weighted on-duty fraction per bucket by carrying each switch state forward."""
    clean = duty.dropna().sort_index()
    if clean.empty or len(target_index) == 0:
        return pd.Series(dtype=float)
    ev_ns = pd.DatetimeIndex(pd.to_datetime(clean.index, utc=True)).asi8
    ev_vals = clean.to_numpy(dtype=float)
    tgt_ns = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True)).asi8
    dt_ns = int(pd.Timedelta(minutes=dt_minutes).value)
    out = np.array(
        [_bucket_on_fraction(ev_ns, ev_vals, int(b0), dt_ns) for b0 in tgt_ns], dtype=float
    )
    return pd.Series(out, index=target_index, dtype=float)


def _bucket_on_fraction(ev_ns: np.ndarray, ev_vals: np.ndarray, b0: int, dt_ns: int) -> float:
    """Compute time-weighted ON fraction in [b0, b0 + dt_ns) from sorted event arrays."""
    b1 = b0 + dt_ns
    idx_before = int(np.searchsorted(ev_ns, b0, side="right")) - 1
    idx_end = int(np.searchsorted(ev_ns, b1, side="left"))
    if idx_before < 0 and idx_end <= 0:
        return float("nan")
    state = float(ev_vals[idx_before]) if idx_before >= 0 else 0.0
    cursor = b0
    on_ns = 0.0
    for k in range(max(0, idx_before + 1), idx_end):
        t_ev = int(ev_ns[k])
        if state > 0.0:
            on_ns += t_ev - cursor
        cursor = t_ev
        state = float(ev_vals[k])
    if state > 0.0:
        on_ns += b1 - cursor
    return on_ns / dt_ns


def _integrate_switch_duty(
    duty: pd.Series,
    target_index: pd.DatetimeIndex,
    dt_minutes: int,
) -> pd.Series:
    """Compute time-weighted on-duty fraction per bucket by carrying each switch state forward."""
    clean = duty.dropna().sort_index()
    if clean.empty or len(target_index) == 0:
        return pd.Series(dtype=float)
    ev_ns = pd.DatetimeIndex(pd.to_datetime(clean.index, utc=True)).asi8
    ev_vals = clean.to_numpy(dtype=float)
    tgt_ns = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True)).asi8
    dt_ns = int(pd.Timedelta(minutes=dt_minutes).value)
    out = np.array(
        [_bucket_on_fraction(ev_ns, ev_vals, int(b0), dt_ns) for b0 in tgt_ns], dtype=float
    )
    return pd.Series(out, index=target_index, dtype=float)


async def _fetch_influx_switch_duty(
    session: aiohttp.ClientSession,
    conn: dict,
    eid: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    dt_minutes: int,
    target_index: pd.DatetimeIndex,
    logger: logging.Logger,
) -> pd.Series:
    """Fetch time-weighted heater switch on/off duty cycle from InfluxDB, or empty if absent."""
    valid = _valid_thermal_entity_id(eid, logger)
    if not valid:
        return pd.Series(dtype=float)
    meas = await _find_influx_measurement(session, conn, valid, logger)
    if not meas:
        return pd.Series(dtype=float)
    bare = valid.split(".", 1)[-1]
    meas_esc = meas.replace("\\", "\\\\").replace('"', '\\"')
    lookback = (start_utc - pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    q = (
        f'SELECT * FROM "{meas_esc}" '
        f"WHERE (\"entity_id\" = '{valid}' OR \"entity_id\" = '{bare}') "
        f"AND time >= '{lookback}' AND time <= '{end_utc.strftime('%Y-%m-%d %H:%M:%S')}'"
    )
    res = await _run_influx_query(session, conn, q)
    if res.empty:
        return pd.Series(dtype=float)
    if "state" in res.columns and res["state"].notna().any():
        duty = (res["state"].astype(str).str.lower() == "on").astype(float)
    elif "value" in res.columns:
        duty = (pd.to_numeric(res["value"], errors="coerce").fillna(0.0) > 0).astype(float)
    else:
        return pd.Series(dtype=float)
    return _integrate_switch_duty(duty, target_index, dt_minutes)


def _assemble_influx_thermal_frame(
    t_in: pd.Series,
    t_out: pd.Series,
    p_direct: pd.Series,
    t_set: pd.Series,
    sw_duty: pd.Series,
    nominal_power: float,
    sense: str,
) -> pd.DataFrame:
    """Combine InfluxDB thermal series into a clean calibration DataFrame without fabricating power."""
    frame = pd.DataFrame({"T_in": t_in, "T_out": t_out})
    if not t_set.empty:
        frame["T_set"] = t_set
    has_sw = not sw_duty.empty and sw_duty.notna().any()
    if not p_direct.empty and p_direct.notna().sum() >= 2:
        frame["P_def"] = p_direct.reindex(frame.index).fillna(0.0).clip(lower=0.0)
    elif "T_set" in frame.columns:
        clim_duty = (
            (frame["T_in"] > frame["T_set"]).astype(float)
            if sense == "cool"
            else (frame["T_set"] > frame["T_in"]).astype(float)
        )
        sw_aligned = sw_duty.reindex(frame.index) if has_sw else pd.Series(1.0, index=frame.index)
        frame["P_def"] = sw_aligned * clim_duty * nominal_power
    elif has_sw:
        frame["P_def"] = sw_duty.reindex(frame.index) * nominal_power
    else:
        return pd.DataFrame()
    return frame.dropna(subset=["T_in", "T_out", "P_def"])


async def _fetch_influx_thermal_window(
    rh_conf: dict,
    entity_t_in: str,
    entity_t_out: str,
    entity_power: str,
    entity_t_pred: str,
    entity_switch: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    dt_minutes: int,
    nominal_power: float,
    logger: logging.Logger,
    sense: str = "heat",
) -> pd.DataFrame:
    """Query historical thermal series directly from InfluxDB using user entities."""
    host = str(rh_conf.get("influxdb_host") or "localhost")
    raw_port = rh_conf.get("influxdb_port")
    port_val = 8086 if raw_port in (None, "") else raw_port
    if not is_valid_db_host(host) or not is_valid_db_port(port_val):
        logger.warning("Refusing InfluxDB thermal query due to invalid host/port configuration")
        return pd.DataFrame()
    port = int(port_val)
    use_ssl = bool(rh_conf.get("influxdb_use_ssl", False))
    user = os.getenv("INFLUXDB_USERNAME") or str(rh_conf.get("influxdb_username") or "")
    password = os.getenv("INFLUXDB_PASSWORD") or str(rh_conf.get("influxdb_password") or "")
    conn = {
        "base_url": f"{'https' if use_ssl else 'http'}://{host}:{port}/query",
        "db": str(rh_conf.get("influxdb_database") or "homeassistant"),
        "auth": aiohttp.BasicAuth(user, password) if user else None,
        "verify_ssl": bool(rh_conf.get("influxdb_verify_ssl", False)),
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        t_in = await _fetch_influx_numeric_series(
            session, conn, entity_t_in, start_utc, end_utc, dt_minutes, logger
        )
        if t_in.empty:
            return pd.DataFrame()
        t_out = await _fetch_influx_numeric_series(
            session, conn, entity_t_out, start_utc, end_utc, dt_minutes, logger
        )
        if t_out.empty:
            return pd.DataFrame()
        p_direct = (
            await _fetch_influx_numeric_series(
                session, conn, entity_power, start_utc, end_utc, dt_minutes, logger
            )
            if entity_power
            else pd.Series(dtype=float)
        )
        t_set = (
            await _fetch_influx_numeric_series(
                session, conn, entity_t_pred, start_utc, end_utc, dt_minutes, logger
            )
            if entity_t_pred
            else pd.Series(dtype=float)
        )
        sw_duty = await _fetch_influx_switch_duty(
            session, conn, entity_switch, start_utc, end_utc, dt_minutes, t_in.index, logger
        )

    clean_frame = _assemble_influx_thermal_frame(
        t_in, t_out, p_direct, t_set, sw_duty, nominal_power, sense
    )
    logger.info("Loaded %d InfluxDB rows for thermal model tuning", len(clean_frame))
    return clean_frame


def _build_offline_demo_thermal_frame(
    dt_hours: float, nominal_power: float, lookback_hours: int, sense: str = "heat"
) -> pd.DataFrame:
    """Build a realistic synthetic thermal validation dataset for offline testing."""
    n_steps = max(16, int(round(max(lookback_hours, 12) / dt_hours)))
    end_ts = pd.Timestamp.now(tz="UTC").floor(f"{int(round(dt_hours * 60))}min")
    idx = pd.date_range(end=end_ts, periods=n_steps, freq=f"{int(round(dt_hours * 60))}min")
    is_cool = sense == "cool"
    t_out_base = 28.5 if is_cool else 8.5
    t_out = np.array(
        [
            t_out_base + 4.5 * math.sin((i / max(n_steps, 1) - 0.25) * 2.0 * math.pi)
            for i in range(n_steps)
        ],
        dtype=float,
    )
    t_set = np.array(
        [
            (22.0 if 0.25 <= (i / max(n_steps, 1)) <= 0.85 else 25.0)
            if is_cool
            else (21.0 if 0.25 <= (i / max(n_steps, 1)) <= 0.85 else 18.5)
            for i in range(n_steps)
        ],
        dtype=float,
    )
    # Two active blocks with true plant alpha=0.85 C/h, gamma=0.035 /h, lag=2 steps (1.0h at 0.5h step)
    true_alpha, true_gamma, true_lag = 0.85, 0.035, 2
    sign = -1.0 if is_cool else 1.0
    p_def = np.zeros(n_steps, dtype=float)
    for i in range(n_steps):
        frac = i / max(n_steps, 1)
        if (0.20 <= frac <= 0.42) or (0.60 <= frac <= 0.78):
            p_def[i] = nominal_power

    rng = np.random.default_rng(42)
    t_in = np.empty(n_steps, dtype=float)
    t_in[0] = 24.8 if is_cool else 19.2
    for i in range(n_steps - 1):
        p_eff = p_def[i - true_lag] if i >= true_lag else 0.0
        dt_val = sign * true_alpha * dt_hours * (p_eff / nominal_power) - true_gamma * dt_hours * (
            t_in[i] - t_out[i]
        )
        t_in[i + 1] = t_in[i] + dt_val + float(rng.normal(0.0, 0.02))

    return pd.DataFrame(
        {"T_in": t_in, "T_set": t_set, "T_out": t_out, "P_def": p_def},
        index=idx,
    )


def _parse_json_dict(
    raw: dict | str | None,
    logger: logging.Logger | None = None,
    warn_msg: str | None = None,
) -> dict:
    """Parse an optional JSON string or dict into a plain dict."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = orjson.loads(raw)
            if isinstance(loaded, dict):
                return loaded
        except orjson.JSONDecodeError:
            if logger is not None and warn_msg:
                logger.warning(warn_msg)
    return {}


def _resolve_thermal_dt_hours(rh_conf: dict, rt: dict) -> tuple[float, int]:
    """Resolve optimization timestep in hours and integer minutes for thermal calibration."""
    raw_step = rh_conf.get("optimization_time_step", 30)
    fallback_dt = float(rt.get("dt_hours", 0.5) or 0.5)
    if hasattr(raw_step, "total_seconds"):
        dt_hours = float(raw_step.total_seconds() / 3600.0)
    elif isinstance(raw_step, int | float):
        dt_hours = float(raw_step) / 60.0
    elif isinstance(raw_step, str) and raw_step.strip():
        try:
            dt_hours = float(pd.Timedelta(raw_step).total_seconds() / 3600.0)
        except Exception:
            dt_hours = fallback_dt
    else:
        dt_hours = fallback_dt
    dt_hours = max(dt_hours, 1.0 / 60.0)
    return dt_hours, max(1, int(round(dt_hours * 60.0)))


def _extract_thermal_load_defaults(optim_conf: dict, load_index: int) -> tuple[float, dict]:
    """Return default nominal power and thermal_config dict for a deferrable load index."""
    nom_list = optim_conf.get("nominal_power_of_deferrable_loads")
    default_p_nom = 3000.0
    if isinstance(nom_list, list) and 0 <= load_index < len(nom_list):
        default_p_nom = float(nom_list[load_index])
    def_configs = optim_conf.get("def_load_config")
    tc: dict = {}
    if isinstance(def_configs, list) and 0 <= load_index < len(def_configs):
        entry = def_configs[load_index]
        if isinstance(entry, dict) and isinstance(entry.get("thermal_config"), dict):
            tc = entry["thermal_config"]
    return default_p_nom, tc


def _parse_thermal_time_window(
    rt: dict, lookback_hours: int
) -> tuple[bool, pd.Timestamp, pd.Timestamp]:
    """Parse explicit date_ranges or default lookback window in UTC."""
    date_ranges = rt.get("date_ranges")
    if (
        isinstance(date_ranges, list)
        and date_ranges
        and isinstance(date_ranges[0], list | tuple)
        and len(date_ranges[0]) >= 2
    ):
        return (
            True,
            pd.to_datetime(date_ranges[0][0], utc=True),
            pd.to_datetime(date_ranges[0][1], utc=True),
        )
    end_utc = pd.Timestamp.now(tz="UTC")
    return False, end_utc - pd.Timedelta(hours=lookback_hours), end_utc


def _parse_thermal_entities(rt: dict, load_index: int) -> dict[str, str]:
    """Resolve Home Assistant / InfluxDB entity IDs for thermal calibration."""
    return {
        "entity_t_in": str(rt.get("entity_t_in") or "sensor.temp_air_in").strip(),
        "entity_t_out": str(rt.get("entity_t_out") or "sensor.outdoor_temperature").strip(),
        "entity_power": str(rt.get("entity_power") or f"sensor.p_deferrable{load_index}").strip(),
        "entity_t_pred": str(
            rt.get("entity_t_pred") or f"sensor.temp_predicted_{load_index}_proxy"
        ).strip(),
        "entity_switch": str(rt.get("entity_switch") or "switch.floor_heater_switch").strip(),
    }


def _parse_thermal_calibration_config(
    params_dict: dict,
    rt: dict,
) -> dict:
    """Extract normalized thermal calibration configuration, entities, and time window."""
    optim_conf = params_dict.get("optim_conf") or {}
    rh_conf = params_dict.get("retrieve_hass_conf") or {}
    load_index = int(rt.get("load_index", 0) or 0)
    dt_hours, dt_minutes = _resolve_thermal_dt_hours(rh_conf, rt)
    default_p_nom, tc = _extract_thermal_load_defaults(optim_conf, load_index)

    raw_curr_inertia = rt.get("current_thermal_inertia")
    if raw_curr_inertia is None:
        raw_curr_inertia = tc.get("thermal_inertia")
    sense = str(rt.get("sense") or tc.get("sense") or "heat").strip().lower()
    if sense not in ("heat", "cool"):
        raise ValueError(f"Unsupported thermal sense: {sense!r} (expected 'heat' or 'cool')")

    lookback_hours = max(4, int(rt.get("lookback_hours", 24) or 24))
    has_explicit_window, start_utc, end_utc = _parse_thermal_time_window(rt, lookback_hours)

    return {
        "load_index": load_index,
        "dt_hours": dt_hours,
        "dt_minutes": dt_minutes,
        "nominal_power": float(rt.get("nominal_power") or default_p_nom or 3000.0),
        "curr_alpha": float(rt.get("current_heating_rate") or tc.get("heating_rate") or 5.0),
        "curr_gamma": float(
            rt.get("current_cooling_constant") or tc.get("cooling_constant") or 0.1
        ),
        "curr_inertia": float(0.0 if raw_curr_inertia is None else raw_curr_inertia),
        "sense": sense,
        "lookback_hours": lookback_hours,
        "has_explicit_window": has_explicit_window,
        "start_utc": start_utc,
        "end_utc": end_utc,
        **_parse_thermal_entities(rt, load_index),
    }


def _load_thermal_from_runtime_payload(rt: dict, dt_minutes: int) -> pd.DataFrame:
    """Build thermal calibration DataFrame from explicit runtimeparams arrays if valid."""
    if not (
        isinstance(rt.get("indoor_temperature"), list)
        and isinstance(rt.get("outdoor_temperature"), list)
    ):
        return pd.DataFrame()
    t_in_arr = pd.to_numeric(pd.Series(rt["indoor_temperature"]), errors="coerce")
    t_out_arr = pd.to_numeric(pd.Series(rt["outdoor_temperature"]), errors="coerce")
    n = min(len(t_in_arr), len(t_out_arr))
    if n < MIN_THERMAL_ROWS:
        return pd.DataFrame()
    end_ts = pd.Timestamp.now(tz="UTC").floor(f"{dt_minutes}min")
    idx = pd.date_range(end=end_ts, periods=n, freq=f"{dt_minutes}min")
    if isinstance(rt.get("heater_power"), list) and len(rt["heater_power"]) >= n:
        p_arr = pd.to_numeric(pd.Series(rt["heater_power"][:n]), errors="coerce").fillna(0.0)
    else:
        p_arr = pd.Series(0.0, index=idx)
    candidate_df = pd.DataFrame(
        {
            "T_in": t_in_arr.iloc[:n].to_numpy(dtype=float),
            "T_out": t_out_arr.iloc[:n].to_numpy(dtype=float),
            "P_def": np.clip(p_arr.to_numpy(dtype=float), 0.0, None),
        },
        index=idx,
    )
    if isinstance(rt.get("predicted_temperature"), list) and len(rt["predicted_temperature"]) >= n:
        candidate_df["T_set"] = pd.to_numeric(
            pd.Series(rt["predicted_temperature"][:n]), errors="coerce"
        ).to_numpy(dtype=float)
    candidate_df = candidate_df.dropna(subset=["T_in", "T_out", "P_def"])
    return candidate_df if len(candidate_df) >= MIN_THERMAL_ROWS else pd.DataFrame()


async def _load_thermal_from_ha(
    emhass_conf: dict,
    params_dict: dict,
    rh_conf: dict,
    cfg: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Fetch thermal calibration history from Home Assistant REST/WS."""
    from emhass.retrieve_hass import RetrieveHass

    ha_params = dict(params_dict)
    ha_rh_conf = dict(rh_conf)
    # Bypass InfluxDB/VictoriaMetrics on this fallback so RetrieveHass uses HA REST/WS directly.
    ha_rh_conf["use_influxdb"] = False
    ha_rh_conf["use_victoriametrics"] = False
    ha_params["retrieve_hass_conf"] = ha_rh_conf

    rh = RetrieveHass(
        rh_conf["hass_url"],
        rh_conf.get("long_lived_token", ""),
        pd.Timedelta(minutes=cfg["dt_minutes"]),
        rh_conf.get("time_zone", pytz.UTC),
        ha_params,
        emhass_conf,
        logger,
    )
    try:
        if cfg["has_explicit_window"]:
            days_list = pd.date_range(
                start=cfg["start_utc"].floor("D"),
                end=cfg["end_utc"].floor("D"),
                freq="D",
                tz="UTC",
            )
        else:
            days_list = get_days_list(max(1, math.ceil(cfg["lookback_hours"] / 24)))
        var_list = [v for v in (cfg["entity_t_in"], cfg["entity_t_out"], cfg["entity_power"]) if v]
        ok = await rh.get_data(days_list, var_list, minimal_response=False, keep_partial_days=True)
        if not ok or rh.df_final is None:
            return pd.DataFrame()
        df_ha = rh.df_final
        if cfg["entity_t_in"] not in df_ha.columns or cfg["entity_t_out"] not in df_ha.columns:
            return pd.DataFrame()
        p_series = (
            pd.to_numeric(df_ha[cfg["entity_power"]], errors="coerce").fillna(0.0)
            if cfg["entity_power"] in df_ha.columns
            else pd.Series(0.0, index=df_ha.index)
        )
        candidate_df = pd.DataFrame(
            {
                "T_in": pd.to_numeric(df_ha[cfg["entity_t_in"]], errors="coerce"),
                "T_out": pd.to_numeric(df_ha[cfg["entity_t_out"]], errors="coerce"),
                "P_def": p_series.clip(lower=0.0),
            },
            index=df_ha.index,
        ).dropna()
        if cfg["has_explicit_window"]:
            idx_utc = pd.to_datetime(candidate_df.index, utc=True)
            candidate_df = candidate_df.loc[
                (idx_utc >= cfg["start_utc"]) & (idx_utc <= cfg["end_utc"])
            ]
        elif len(candidate_df) >= MIN_THERMAL_ROWS:
            keep = max(MIN_THERMAL_ROWS, int(round(cfg["lookback_hours"] / cfg["dt_hours"])))
            candidate_df = candidate_df.iloc[-keep:]
        return candidate_df if len(candidate_df) >= MIN_THERMAL_ROWS else pd.DataFrame()
    finally:
        await rh.close()


async def _load_thermal_calibration_frame(
    emhass_conf: dict,
    params_dict: dict,
    rh_conf: dict,
    rt: dict,
    cfg: dict,
    get_data_from_file: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str]:
    """Load thermal calibration series from runtime payload, InfluxDB, HA, or offline demo."""
    df_rt = _load_thermal_from_runtime_payload(rt, cfg["dt_minutes"])
    if not df_rt.empty:
        return df_rt, "runtime_payload"

    if not get_data_from_file and rh_conf.get("use_influxdb"):
        try:
            df_inf = await _fetch_influx_thermal_window(
                rh_conf,
                cfg["entity_t_in"],
                cfg["entity_t_out"],
                cfg["entity_power"],
                cfg["entity_t_pred"],
                cfg["entity_switch"],
                cfg["start_utc"],
                cfg["end_utc"],
                cfg["dt_minutes"],
                cfg["nominal_power"],
                logger,
                sense=cfg["sense"],
            )
            if len(df_inf) >= MIN_THERMAL_ROWS:
                return df_inf, "influxdb"
        except Exception as exc:
            logger.warning("InfluxDB thermal history retrieval failed (%s); falling back", exc)

    if not get_data_from_file and rh_conf.get("hass_url") not in (None, "", "empty"):
        try:
            df_ha = await _load_thermal_from_ha(emhass_conf, params_dict, rh_conf, cfg, logger)
            if not df_ha.empty:
                return df_ha, "home_assistant"
        except Exception as exc:
            logger.warning("HA thermal history retrieval failed (%s); using offline demo", exc)

    return (
        _build_offline_demo_thermal_frame(
            cfg["dt_hours"], cfg["nominal_power"], cfg["lookback_hours"], sense=cfg["sense"]
        ),
        "offline_demo",
    )


def _prepare_thermal_grid(
    df_win: pd.DataFrame, dt_minutes: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reindex thermal history onto a regular step grid and compute contiguity/observation masks."""
    df_sorted = df_win.sort_index().copy()
    step_td = pd.Timedelta(minutes=dt_minutes)
    if isinstance(df_sorted.index, pd.DatetimeIndex) and len(df_sorted) >= 2:
        max_grid = max(len(df_sorted) * 4, 2000)
        span_steps = int((df_sorted.index.max() - df_sorted.index.min()) // step_td) + 1
        if 0 < span_steps <= max_grid:
            full_grid = pd.date_range(
                start=df_sorted.index.min(), end=df_sorted.index.max(), freq=f"{dt_minutes}min"
            )
            df_sorted = df_sorted.reindex(full_grid)
        next_diff = df_sorted.index.to_series().shift(-1) - df_sorted.index.to_series()
        step_contiguous = (next_diff == step_td).fillna(False).to_numpy(dtype=bool)
    else:
        step_contiguous = np.ones(len(df_sorted), dtype=bool)
        if len(step_contiguous) > 0:
            step_contiguous[-1] = False

    observed_mask = (
        df_sorted["T_in"].notna() & df_sorted["T_out"].notna() & df_sorted["P_def"].notna()
    ).to_numpy(dtype=bool)
    df_sorted["dT_actual"] = df_sorted["T_in"].shift(-1) - df_sorted["T_in"]
    df_sorted.loc[~step_contiguous | ~observed_mask, "dT_actual"] = np.nan
    return df_sorted, step_contiguous, observed_mask


def _build_lagged_power_series(
    p_raw_vals: np.ndarray,
    observed_mask: np.ndarray,
    index: pd.Index,
    lag: int,
    step_td: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray]:
    """Build gap-aware lagged power vector and validity mask for the given lag."""
    n_rows = len(p_raw_vals)
    p_arr = np.full(n_rows, np.nan, dtype=float)
    p_ok = np.zeros(n_rows, dtype=bool)
    is_dt_index = isinstance(index, pd.DatetimeIndex)
    for i in range(n_rows):
        if not observed_mask[i]:
            continue
        if lag == 0:
            p_arr[i] = p_raw_vals[i]
            p_ok[i] = True
        elif i >= lag:
            expected_span = (index[i] - index[i - lag] == lag * step_td) if is_dt_index else True
            if expected_span and np.all(observed_mask[i - lag : i + 1]):
                p_arr[i] = p_raw_vals[i - lag]
                p_ok[i] = True
    return p_arr, p_ok


def _simulate_and_score_thermal(
    grid: dict,
    p_arr: np.ndarray,
    p_ok: np.ndarray,
    alpha_val: float,
    gamma_val: float,
    cfg: dict,
    common_p_ok: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Simulate contiguous transitions with known lagged power and score non-anchor steps."""
    observed_mask = grid["observed_mask"]
    effective_p_ok = p_ok if common_p_ok is None else (p_ok & common_p_ok)
    trans_ok = grid["step_contiguous"] & effective_p_ok
    t_in_vals = grid["t_in_vals"]
    t_sim = _simulate_thermal_trajectory(
        t_in_vals,
        np.nan_to_num(p_arr, nan=0.0),
        grid["t_out_vals"],
        alpha_val,
        gamma_val,
        cfg["dt_hours"],
        cfg["nominal_power"],
        observed_mask=observed_mask,
        step_contiguous=trans_ok,
        sense=cfg["sense"],
    )
    n_rows = len(t_in_vals)
    reached = np.zeros(n_rows, dtype=bool)
    if n_rows > 1:
        reached[1:] = trans_ok[:-1] & observed_mask[:-1] & observed_mask[1:]
    scored = reached & np.isfinite(t_sim)
    rmse_val = (
        float(np.sqrt(np.mean((t_in_vals[scored] - t_sim[scored]) ** 2)))
        if scored.any()
        else float("nan")
    )
    return t_sim, rmse_val


def _is_thermal_fit_identifiable(
    p_vals: np.ndarray,
    dt_diff: np.ndarray,
    has_heating_excitation: bool,
    min_fit_transitions: int,
) -> bool:
    """Return True only when usable transition rows can identify the fitted thermal parameters."""
    if len(dt_diff) < min_fit_transitions:
        return False
    has_temp_grad = bool(np.any(np.abs(dt_diff) > 1e-6))
    if not has_heating_excitation:
        return has_temp_grad
    return has_temp_grad and bool(np.any(p_vals > 0.0))


def _fit_thermal_lag_candidate(
    lag_steps: int,
    p_shifted: np.ndarray,
    p_valid: np.ndarray,
    common_p_ok: np.ndarray,
    grid: dict,
    cfg: dict,
    curve_fit_fn,
) -> tuple[dict, np.ndarray]:
    """Fit heating/cooling constants for one candidate thermal inertia lag."""
    valid = grid["dt_valid_mask"] & p_valid & grid["observed_mask"]
    p_use = p_shifted[valid]
    dt_diff = grid["t_in_vals"][valid] - grid["t_out_vals"][valid]
    x_data = (p_use, grid["t_in_vals"][valid], grid["t_out_vals"][valid])
    y_data = grid["dt_actual_vals"][valid]
    fit_ok = False
    alpha_opt, gamma_opt = cfg["curr_alpha"], cfg["curr_gamma"]
    dt_hours = cfg["dt_hours"]
    nominal_power = cfg["nominal_power"]
    sense = cfg["sense"]

    if _is_thermal_fit_identifiable(
        p_use, dt_diff, grid["has_heating_excitation"], grid["min_fit_transitions"]
    ):
        try:
            if not grid["has_heating_excitation"]:
                popt_g, _ = curve_fit_fn(
                    lambda delta_t, g: -g * dt_hours * delta_t,
                    dt_diff,
                    y_data,
                    p0=[max(0.005, min(gamma_opt, 0.5))],
                    bounds=(0.0005, 1.5),
                    maxfev=5000,
                )
                gamma_opt = float(popt_g[0])
            else:
                popt, _ = curve_fit_fn(
                    lambda x, a, g: _emhass_thermal_step_delta(
                        x, a, g, dt_hours, nominal_power, sense=sense
                    ),
                    x_data,
                    y_data,
                    p0=[max(0.05, min(alpha_opt, 5.0)), max(0.005, min(gamma_opt, 0.5))],
                    bounds=(0.0005, [15.0, 1.5]),
                    maxfev=5000,
                )
                alpha_opt, gamma_opt = float(popt[0]), float(popt[1])
            fit_ok = True
        except Exception:
            fit_ok = False

    if fit_ok:
        y_pred_step = _emhass_thermal_step_delta(
            x_data, alpha_opt, gamma_opt, dt_hours, nominal_power, sense=sense
        )
        rmse_step = float(np.sqrt(np.mean((y_data - y_pred_step) ** 2)))
        t_sim, rmse_traj = _simulate_and_score_thermal(
            grid, p_shifted, p_valid, alpha_opt, gamma_opt, cfg, common_p_ok=common_p_ok
        )
    else:
        rmse_step = float("nan")
        t_sim = np.full(len(grid["t_in_vals"]), np.nan, dtype=float)
        rmse_traj = float("nan")

    row = {
        "lag_steps": lag_steps,
        "thermal_inertia_h": round(lag_steps * dt_hours, 4),
        "heating_rate": round(alpha_opt, 4),
        "cooling_constant": round(gamma_opt, 4),
        "rmse_step_c": round(rmse_step, 4) if np.isfinite(rmse_step) else float("nan"),
        "rmse_trajectory_c": round(rmse_traj, 4) if np.isfinite(rmse_traj) else float("nan"),
        "fit_ok": fit_ok,
    }
    return row, t_sim


def _select_best_thermal_lag(
    lag_rows: list[dict],
    curr_lag: int,
    curr_inertia: float,
    dt_hours: float,
    has_heating_excitation: bool,
) -> tuple[dict, int, float, bool]:
    """Select the best lag row and recommended thermal inertia from candidate fits."""
    valid_lag_rows = [r for r in lag_rows if r["fit_ok"] and np.isfinite(r["rmse_trajectory_c"])]
    fit_succeeded = bool(valid_lag_rows)
    if not has_heating_excitation:
        best_row = next((r for r in valid_lag_rows if int(r["lag_steps"]) == 0), None) or (
            valid_lag_rows[0] if valid_lag_rows else lag_rows[0]
        )
        return best_row, curr_lag, float(curr_inertia), fit_succeeded
    if valid_lag_rows:
        best_row = min(valid_lag_rows, key=lambda r: r["rmse_trajectory_c"])
        best_lag = int(best_row["lag_steps"])
        return best_row, best_lag, float(best_lag * dt_hours), fit_succeeded
    return lag_rows[0], curr_lag, float(curr_inertia), fit_succeeded


async def compute_thermal_model_calibration(
    emhass_conf: dict,
    params: dict | str | None,
    runtimeparams: str | dict | None,
    logger: logging.Logger,
    get_data_from_file: bool = False,
) -> dict:
    """Validate predicted vs actual thermal trajectories and fit heating/cooling/inertia constants."""
    from scipy.optimize import curve_fit

    rt = _parse_json_dict(
        runtimeparams,
        logger=logger,
        warn_msg="Ignoring malformed JSON runtimeparams in thermal-model-calibration",
    )
    params_dict = _parse_json_dict(params)
    rh_conf = params_dict.get("retrieve_hass_conf") or {}
    cfg = _parse_thermal_calibration_config(params_dict, rt)

    df_raw, data_source = await _load_thermal_calibration_frame(
        emhass_conf, params_dict, rh_conf, rt, cfg, get_data_from_file, logger
    )
    df_win, step_contiguous, observed_mask = _prepare_thermal_grid(df_raw, cfg["dt_minutes"])
    t_in_vals = df_win["T_in"].to_numpy(dtype=float)
    t_out_vals = df_win["T_out"].to_numpy(dtype=float)
    p_raw_vals = df_win["P_def"].to_numpy(dtype=float)
    observed_p = p_raw_vals[observed_mask]
    has_heating_excitation = bool(observed_p.size > 0 and np.any(observed_p > 0.0))

    grid = {
        "index": df_win.index,
        "step_td": pd.Timedelta(minutes=cfg["dt_minutes"]),
        "step_contiguous": step_contiguous,
        "observed_mask": observed_mask,
        "t_in_vals": t_in_vals,
        "t_out_vals": t_out_vals,
        "p_raw_vals": p_raw_vals,
        "dt_actual_vals": df_win["dT_actual"].to_numpy(dtype=float),
        "dt_valid_mask": df_win["dT_actual"].notna().to_numpy(dtype=bool),
        "has_heating_excitation": has_heating_excitation,
        "min_fit_transitions": 2 if not has_heating_excitation else 3,
    }

    n_rows = len(df_win)
    # Match optimization.py truncation & horizon bound: L = max(0, min(int(inertia / dt), n - 1))
    curr_lag = max(0, min(int(cfg["curr_inertia"] / cfg["dt_hours"]), max(0, n_rows - 1)))
    candidate_lags = (0, 1, 2, 3, 4)
    lagged_by_step = {
        lag: _build_lagged_power_series(
            p_raw_vals, observed_mask, df_win.index, lag, grid["step_td"]
        )
        for lag in (*candidate_lags, curr_lag)
    }
    # Score all candidate lags and the current model on the exact same shared transition set.
    common_p_ok = observed_mask.copy()
    for lag in candidate_lags:
        common_p_ok &= lagged_by_step[lag][1]
    if curr_lag not in candidate_lags:
        common_p_ok &= lagged_by_step[curr_lag][1]

    lag_rows: list[dict] = []
    trajectories_by_lag: dict[int, np.ndarray] = {}
    for lag_steps in candidate_lags:
        p_shifted, p_valid = lagged_by_step[lag_steps]
        row, t_sim = _fit_thermal_lag_candidate(
            lag_steps, p_shifted, p_valid, common_p_ok, grid, cfg, curve_fit
        )
        trajectories_by_lag[lag_steps] = t_sim
        lag_rows.append(row)

    best_row, best_lag, recommended_inertia, fit_succeeded = _select_best_thermal_lag(
        lag_rows, curr_lag, cfg["curr_inertia"], cfg["dt_hours"], has_heating_excitation
    )

    p_curr_shifted, p_curr_valid = lagged_by_step[curr_lag]
    t_sim_current, rmse_traj_current = _simulate_and_score_thermal(
        grid,
        p_curr_shifted,
        p_curr_valid,
        cfg["curr_alpha"],
        cfg["curr_gamma"],
        cfg,
        common_p_ok=common_p_ok,
    )

    p_best_shifted, p_best_valid = lagged_by_step[best_lag]
    if best_lag in trajectories_by_lag and has_heating_excitation and fit_succeeded:
        t_sim_tuned = trajectories_by_lag[best_lag]
        rmse_traj_tuned = float(best_row["rmse_trajectory_c"])
    elif fit_succeeded:
        t_sim_tuned, rmse_traj_tuned = _simulate_and_score_thermal(
            grid,
            p_best_shifted,
            p_best_valid,
            best_row["heating_rate"],
            best_row["cooling_constant"],
            cfg,
            common_p_ok=common_p_ok,
        )
    else:
        t_sim_tuned = np.full(n_rows, np.nan, dtype=float)
        rmse_traj_tuned = float("nan")

    df_win["T_pred_current"] = np.round(t_sim_current, 3)
    df_win["T_pred_tuned"] = np.round(t_sim_tuned, 3)
    dt_pred_tuned = np.full(n_rows, np.nan, dtype=float)
    step_eval_mask = observed_mask & p_best_valid
    if fit_succeeded and step_eval_mask.any():
        dt_pred_tuned[step_eval_mask] = _emhass_thermal_step_delta(
            (
                p_best_shifted[step_eval_mask],
                t_in_vals[step_eval_mask],
                t_out_vals[step_eval_mask],
            ),
            best_row["heating_rate"],
            best_row["cooling_constant"],
            cfg["dt_hours"],
            cfg["nominal_power"],
            sense=cfg["sense"],
        )
    df_win["dT_pred_tuned"] = np.round(dt_pred_tuned, 4)

    return {
        "data_source": data_source,
        "load_index": cfg["load_index"],
        "dt_hours": cfg["dt_hours"],
        "nominal_power": cfg["nominal_power"],
        "sense": cfg["sense"],
        "cooling_only": not has_heating_excitation,
        "fit_succeeded": fit_succeeded,
        "current_params": {
            "heating_rate": cfg["curr_alpha"],
            "cooling_constant": cfg["curr_gamma"],
            "thermal_inertia": cfg["curr_inertia"],
            "rmse_trajectory_c": (
                round(rmse_traj_current, 4) if np.isfinite(rmse_traj_current) else float("nan")
            ),
        },
        "best_params": {
            "heating_rate": best_row["heating_rate"],
            "cooling_constant": best_row["cooling_constant"],
            "thermal_inertia": recommended_inertia,
            "lag_steps": best_lag,
            "rmse_trajectory_c": (
                round(rmse_traj_tuned, 4) if np.isfinite(rmse_traj_tuned) else float("nan")
            ),
        },
        "lag_table": pd.DataFrame(lag_rows).drop(columns=["fit_ok"]),
        "series": df_win,
    }


def get_injection_dict_thermal_calibration(result: dict) -> dict:
    """Build WebUI plots and parameter recommendations for thermal model validation & tuning."""
    series_df: pd.DataFrame = result["series"]
    lag_df: pd.DataFrame = result["lag_table"]
    curr = result["current_params"]
    best = result["best_params"]
    source_label = {
        "influxdb": "InfluxDB History",
        "home_assistant": "Home Assistant History",
        "runtime_payload": "Runtime Payload",
        "offline_demo": "Offline Synthetic Validation Demo (no live HA/InfluxDB connected)",
    }.get(result["data_source"], result["data_source"])

    # Figure 0: Predicted vs Actual Temperature Trajectory
    fig_traj = make_subplots(specs=[[{"secondary_y": True}]])
    fig_traj.add_trace(
        go.Scatter(
            x=series_df.index,
            y=series_df["T_in"],
            name="Actual Indoor Temp (T_in)",
            line={"color": "#0f172a", "width": 3},
            mode=_PLOTLY_LINES_MARKERS,
        ),
        secondary_y=False,
    )
    best_inertia_display = round(float(best["thermal_inertia"]), 4)
    fig_traj.add_trace(
        go.Scatter(
            x=series_df.index,
            y=series_df["T_pred_tuned"],
            name=f"Predicted (Tuned: α={best['heating_rate']}, γ={best['cooling_constant']}, lag={best_inertia_display}h)",
            line={"color": "#16a34a", "width": 2.5},
            mode="lines",
        ),
        secondary_y=False,
    )
    fig_traj.add_trace(
        go.Scatter(
            x=series_df.index,
            y=series_df["T_pred_current"],
            name=f"Predicted (Current: α={curr['heating_rate']}, γ={curr['cooling_constant']}, lag={curr['thermal_inertia']}h)",
            line={"color": "#ea580c", "width": 2, "dash": "dash"},
            mode="lines",
        ),
        secondary_y=False,
    )
    if "T_set" in series_df.columns:
        fig_traj.add_trace(
            go.Scatter(
                x=series_df.index,
                y=series_df["T_set"],
                name="Setpoint / Plan Temp Proxy (T_set)",
                line={"color": "#8b5cf6", "width": 1.5, "dash": "dot"},
                mode="lines",
            ),
            secondary_y=False,
        )
    fig_traj.add_trace(
        go.Scatter(
            x=series_df.index,
            y=series_df["T_out"],
            name="Outdoor Temp (T_out)",
            line={"color": "#0284c7", "width": 1.5, "dash": "dashdot"},
            mode="lines",
        ),
        secondary_y=False,
    )
    fig_traj.add_trace(
        go.Bar(
            x=series_df.index,
            y=series_df["P_def"],
            name="Heater Power (P_def, W)",
            marker_color="rgba(234, 179, 8, 0.28)",
        ),
        secondary_y=True,
    )
    curr_rmse_txt = (
        f"{curr['rmse_trajectory_c']:.3f}°C" if np.isfinite(curr["rmse_trajectory_c"]) else "n/a"
    )
    best_rmse_txt = (
        f"{best['rmse_trajectory_c']:.3f}°C" if np.isfinite(best["rmse_trajectory_c"]) else "n/a"
    )
    fig_traj.update_layout(
        title=(
            f"Predicted vs Actual Indoor Temperature Trajectory"
            f"<br><sup>Source: {html.escape(source_label)} | Trajectory RMSE: Current={curr_rmse_txt} → Tuned={best_rmse_txt}</sup>"
        ),
        template="presentation",
        height=500,
    )
    fig_traj.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig_traj.update_yaxes(title_text="Heater Power (W)", secondary_y=True)
    fig_traj.update_xaxes(title_text="Timestamp")

    # Figure 1: RMSE & Fitted Constants Across Inertia Lags
    fig_lags = make_subplots(specs=[[{"secondary_y": True}]])
    x_lags = [
        f"{row['thermal_inertia_h']}h ({row['lag_steps']} steps)" for _, row in lag_df.iterrows()
    ]
    fig_lags.add_trace(
        go.Bar(
            x=x_lags,
            y=lag_df["rmse_trajectory_c"],
            name="Trajectory RMSE (°C)",
            marker_color=[
                "#16a34a" if int(r["lag_steps"]) == best["lag_steps"] else "#0284c7"
                for _, r in lag_df.iterrows()
            ],
            text=[f"{v:.4f}" for v in lag_df["rmse_trajectory_c"]],
            textposition="auto",
        ),
        secondary_y=False,
    )
    fig_lags.add_trace(
        go.Scatter(
            x=x_lags,
            y=lag_df["heating_rate"],
            name="Fitted heating_rate (°C/h)",
            line={"color": "#ea580c", "width": 2.5},
            mode=_PLOTLY_LINES_MARKERS,
        ),
        secondary_y=True,
    )
    fig_lags.update_layout(
        title="Thermal Inertia Lag Sweep (0..4 steps)",
        template="presentation",
        xaxis_title="Thermal Inertia Lag",
    )
    fig_lags.update_yaxes(title_text="Trajectory RMSE (°C)", secondary_y=False)
    fig_lags.update_yaxes(title_text="Fitted Heating Rate (°C/h)", secondary_y=True)

    # Figure 2: Step-by-Step Temperature Delta (dT actual vs tuned model)
    valid_delta = series_df.dropna(subset=["dT_actual"])
    fig_step = go.Figure()
    fig_step.add_trace(
        go.Bar(
            x=valid_delta.index,
            y=valid_delta["dT_actual"],
            name="Actual Step ΔT (°C)",
            marker_color="rgba(37, 99, 235, 0.55)",
        )
    )
    fig_step.add_trace(
        go.Scatter(
            x=valid_delta.index,
            y=valid_delta["dT_pred_tuned"],
            name="Tuned Model Step ΔT (°C)",
            line={"color": "#16a34a", "width": 2.5},
            mode=_PLOTLY_LINES_MARKERS,
        )
    )
    fig_step.update_layout(
        title="Per-Step Indoor Temperature Change (Actual ΔT vs Tuned Model ΔT)",
        template="presentation",
        xaxis_title="Timestamp",
        yaxis_title="ΔT per Timestep (°C)",
    )

    best_inertia_json = round(float(best["thermal_inertia"]), 4)
    fit_succeeded = result.get("fit_succeeded", True)
    if result.get("cooling_only"):
        recommended_patch: dict[str, object] = {
            "cooling_constant": best["cooling_constant"],
        }
    else:
        recommended_patch = {
            "heating_rate": best["heating_rate"],
            "cooling_constant": best["cooling_constant"],
            "thermal_inertia": best_inertia_json,
        }
    recommended_json = html.escape(
        orjson.dumps(recommended_patch, option=orjson.OPT_INDENT_2).decode("utf-8")
    )
    fit_warning_note = (
        '<p class="muted-hint"><strong>Insufficient contiguous transitions:</strong> '
        "Too few uninterrupted adjacent temperature steps were available to fit the model; "
        "current coefficients are shown unchanged without a valid prediction score.</p>"
        if not fit_succeeded
        else ""
    )
    cooling_only_note = (
        '<p class="muted-hint"><strong>Note (cooling-only window):</strong> '
        "Heater power remained 0 W across the selected history, so only "
        "<code>cooling_constant</code> was fitted; <code>heating_rate</code> and "
        "<code>thermal_inertia</code> remain at your current values.</p>"
        if result.get("cooling_only") and fit_succeeded
        else ""
    )
    curr_rmse_str = (
        f"{curr['rmse_trajectory_c']:.3f}°C"
        if isinstance(curr.get("rmse_trajectory_c"), int | float)
        and math.isfinite(float(curr["rmse_trajectory_c"]))
        else "n/a"
    )
    best_rmse_str = (
        f"{best['rmse_trajectory_c']:.3f}°C"
        if isinstance(best.get("rmse_trajectory_c"), int | float)
        and math.isfinite(float(best["rmse_trajectory_c"]))
        else "n/a"
    )
    summary_banner = (
        '<div class="analytics-explainer-card">'
        f"<h4>Recommended <code>thermal_config</code> for Deferrable Load {result['load_index']} "
        f"(Source: {html.escape(source_label)})</h4>"
        '<div class="plot-method-badges">'
        f'<span class="method-badge method-load">heating_rate: <strong>{best["heating_rate"]} °C/h</strong></span>'
        f'<span class="method-badge method-pv">cooling_constant: <strong>{best["cooling_constant"]} 1/h</strong></span>'
        f'<span class="method-badge method-bias">thermal_inertia: <strong>{best_inertia_json} h ({best["lag_steps"]} steps)</strong></span>'
        f'<span class="method-badge method-costfun">RMSE Improvement: <strong>{curr_rmse_str} → {best_rmse_str}</strong></span>'
        "</div>"
        f"{fit_warning_note}{cooling_only_note}"
        f"<pre>{recommended_json}</pre>"
        "</div>"
    )

    preview_cols = [
        c
        for c in ("T_in", "T_pred_current", "T_pred_tuned", "T_out", "P_def", "dT_actual")
        if c in series_df.columns
    ]
    table1 = lag_df.to_html(classes="mystyle", index=False)
    table2 = (
        series_df[preview_cols]
        .round(3)
        .reset_index(names="Timestamp")
        .to_html(classes="mystyle", index=False)
    )

    return {
        "title": "<h2>Thermal Model Validation &amp; Parameter Auto-Tuning</h2>",
        "subsubtitle0": summary_banner,
        "figure_0": _fig_to_html(fig_traj, include_plotlyjs=True),
        "figure_thermal_lag_comparison": _fig_to_html(fig_lags, include_plotlyjs=False),
        "figure_thermal_scatter": _fig_to_html(fig_step, include_plotlyjs=False),
        "subsubtitle1": "<h4>Thermal Inertia Lag Fitting Results (curve_fit)</h4>",
        "table1": table1,
        "subsubtitle2": "<h4>Predicted vs Actual Time-Series Sample</h4>",
        "table2": table2,
    }


def get_injection_dict_forecast_model_tune(df_pred_optim: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF tune case.

    :param df_pred_optim: The tune result DataFrame
    :type df_pred_optim: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
    fig = df_pred_optim.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=mlf.model_type)
    fig.update_xaxes(title_text="Time")
    image_path_0 = _fig_to_html(fig, include_plotlyjs=True)
    # The dict of plots
    injection_dict = {}
    injection_dict["title"] = "<h2>Custom machine learning forecast model tune</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Performed a tuning routine using bayesian optimization for "
        + html.escape(str(mlf.model_type))
        + "<br>"
        + "Forecasting variable "
        + html.escape(str(mlf.var_model))
        + "</h4>"
    )
    injection_dict["figure_0"] = image_path_0
    return injection_dict


async def build_config(
    emhass_conf: dict,
    logger: logging.Logger,
    defaults_path: str,
    config_path: str | None = None,
    legacy_config_path: str | None = None,
) -> dict:
    """
    Retrieve parameters from configuration files.
    priority order (low - high) = defaults_path, config_path legacy_config_path

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :param defaults_path: path to config file for parameter defaults (config_defaults.json)
    :type defaults_path: str
    :param config_path: path to the main configuration file (config.json)
    :type config_path: str
    :param legacy_config_path: path to legacy config file (config_emhass.yaml)
    :type legacy_config_path: str
    :return: The built config dictionary
    :rtype: dict
    """

    # Read default parameters (default root_path/data/config_defaults.json)
    if defaults_path and pathlib.Path(defaults_path).is_file():
        async with aiofiles.open(defaults_path) as data:
            content = await data.read()
            config = orjson.loads(content)
    else:
        logger.error("config_defaults.json. does not exist ")
        return False

    # Read user config parameters if provided (default /share/config.json)
    if config_path and pathlib.Path(config_path).is_file():
        async with aiofiles.open(config_path) as data:
            content = await data.read()
            # Set override default parameters (config_defaults) with user given parameters (config.json)
            logger.info("Obtaining parameters from config.json:")
            config.update(orjson.loads(content))
    else:
        logger.info(
            "config.json does not exist, or has not been passed. config parameters may default to config_defaults.json"
        )
        logger.info("you may like to generate the config.json file on the configuration page")

    # Check to see if legacy config_emhass.yaml was provided (default /app/config_emhass.yaml)
    # Convert legacy parameter definitions/format to match config.json
    if legacy_config_path and pathlib.Path(legacy_config_path).is_file():
        async with aiofiles.open(legacy_config_path) as data:
            content = await data.read()
            legacy_config = yaml.safe_load(content)
            legacy_config_parameters = await build_legacy_config_params(
                emhass_conf, legacy_config, logger
            )
            if type(legacy_config_parameters) is not bool:
                logger.info(
                    "Obtaining parameters from config_emhass.yaml: (will overwrite config parameters)"
                )
                config.update(legacy_config_parameters)

    return config


async def build_legacy_config_params(
    emhass_conf: dict[str, pathlib.Path],
    legacy_config: dict[str, str],
    logger: logging.Logger,
) -> dict[str, str]:
    """
    Build a config dictionary with legacy config_emhass.yaml file.
    Uses the associations file to convert parameter naming conventions (to config.json/config_defaults.json).
    Extracts the parameter values and formats to match config.json.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param legacy_config: The legacy config dictionary
    :type legacy_config: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built config dictionary
    :rtype: dict
    """

    # Association file key reference
    # association[0] = config catagories
    # association[1] = legacy parameter name
    # association[2] = parameter (config.json/config_defaults.json)
    # association[3] = parameter list name if exists (not used, from legacy options.json)

    # Check each config catagories exists, else create blank dict for categories (avoid errors)
    legacy_config["retrieve_hass_conf"] = legacy_config.get("retrieve_hass_conf", {})
    legacy_config["optim_conf"] = legacy_config.get("optim_conf", {})
    legacy_config["plant_conf"] = legacy_config.get("plant_conf", {})
    config = {}

    # Use associations list to map legacy parameter name with config.json parameter name
    if emhass_conf["associations_path"].exists():
        async with aiofiles.open(emhass_conf["associations_path"]) as data:
            content = await data.read()
            associations = list(csv.reader(content.splitlines(), delimiter=","))
    else:
        logger.error(
            "Cant find associations file (associations.csv) in: "
            + str(emhass_conf["associations_path"])
        )
        return False

    # Loop through all parameters in association file
    # Append config with existing legacy config parameters (converting alternative parameter naming conventions with associations list)
    for association in associations:
        # if legacy config catagories exists and if legacy parameter exists in config catagories
        if (
            legacy_config.get(association[0]) is not None
            and legacy_config[association[0]].get(association[1], None) is not None
        ):
            config[association[2]] = legacy_config[association[0]][association[1]]

            # If config now has load_peak_hour_periods, extract from list of dict
            if association[2] == "load_peak_hour_periods" and type(config[association[2]]) is list:
                config[association[2]] = {key: d[key] for d in config[association[2]] for key in d}

    return config


def get_keys_to_mask() -> list[str]:
    """
    Return a list of sensitive configuration keys that should be masked in logs
    or treated specially in the UI (e.g., secrets).
    """
    return [
        "influxdb_username",
        "influxdb_password",
        "victoriametrics_username",
        "victoriametrics_password",
        "solcast_api_key",
        "solcast_rooftop_id",
        "long_lived_token",
        "remote_tokens",
        "time_zone",
        "Latitude",
        "Longitude",
        "Altitude",
        "hass_url",  # Ensure this is included if you want it masked everywhere
        "solar_forecast_kwp",  # Ensure this is included if you want it masked everywhere
    ]


def param_to_config(param: dict[str, dict], logger: logging.Logger) -> dict[str, str]:
    """
    A function that extracts the parameters from param back to the config.json format.
    Extracts parameters from config catagories.
    Attempts to exclude secrets hosed in retrieve_hass_conf.

    :param params: Built configuration parameters
    :type param: dict[str, dict]
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built config dictionary
    :rtype: dict[str, str]
    """
    logger.debug("Converting param to config")

    return_config = {}

    config_categories = ["retrieve_hass_conf", "optim_conf", "plant_conf"]
    secret_params = get_keys_to_mask()

    # Loop through config catagories that contain config params, and extract
    for config in config_categories:
        for parameter in param[config]:
            # If parameter is not a secret, append to return_config
            if parameter not in secret_params:
                return_config[str(parameter)] = param[config][parameter]

    return return_config


async def build_secrets(
    emhass_conf: dict[str, pathlib.Path],
    logger: logging.Logger,
    argument: dict[str, str] | None = None,
    options_path: str | None = None,
    secrets_path: str | None = None,
    no_response: bool = False,
) -> tuple[dict[str, pathlib.Path], dict[str, str | float]]:
    """
    Retrieve and build parameters from secrets locations (ENV, ARG, Secrets file (secrets_emhass.yaml/options.json) and/or Home Assistant (via API))
    priority order (lwo to high) = Defaults (written in function), ENV, Options json file, Home Assistant API,  Secrets yaml file, Arguments

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :param argument: dictionary of secrets arguments passed (url,key)
    :type argument: dict
    :param options_path: path to the options file (options.json) (usually provided by EMHASS-Add-on)
    :type options_path: str
    :param secrets_path: path to secrets file (secrets_emhass.yaml)
    :type secrets_path: str
    :param no_response: bypass get request to Home Assistant (json response errors)
    :type no_response: bool
    :return: Updated emhass_conf, the built secrets dictionary
    :rtype: Tuple[dict, dict]:
    """
    # Set defaults to be overwritten
    if argument is None:
        argument = {}
    params_secrets = {
        "hass_url": "https://myhass.duckdns.org/",
        "long_lived_token": "thatverylongtokenhere",
        "time_zone": "Europe/Paris",
        "Latitude": 45.83,
        "Longitude": 6.86,
        "Altitude": 4807.8,
        "solcast_api_key": "yoursecretsolcastapikey",
        "solcast_rooftop_id": "yourrooftopid",
        "solar_forecast_kwp": 5,
        "influxdb_username": "yourinfluxdbusername",
        "influxdb_password": "yourinfluxdbpassword",
        "victoriametrics_username": "",
        "victoriametrics_password": "",
    }

    # Obtain Secrets from ENV?
    params_secrets["hass_url"] = os.getenv("EMHASS_URL", params_secrets["hass_url"])
    params_secrets["long_lived_token"] = os.getenv(
        "SUPERVISOR_TOKEN", params_secrets["long_lived_token"]
    )
    params_secrets["time_zone"] = os.getenv("TIME_ZONE", params_secrets["time_zone"])
    params_secrets["Latitude"] = float(os.getenv("LAT", params_secrets["Latitude"]))
    params_secrets["Longitude"] = float(os.getenv("LON", params_secrets["Longitude"]))
    params_secrets["Altitude"] = float(os.getenv("ALT", params_secrets["Altitude"]))

    # Obtain secrets from options.json (Generated from EMHASS-Add-on, Home Assistant addon Configuration page) or Home Assistant API (from local Supervisor API)?
    # Use local supervisor API to obtain secrets from Home Assistant if hass_url in options.json is empty and SUPERVISOR_TOKEN ENV exists (provided by Home Assistant when running the container as addon)
    options = {}
    if options_path and pathlib.Path(options_path).is_file():
        async with aiofiles.open(options_path) as data:
            content = await data.read()
            options = orjson.loads(content)

            # Obtain secrets from Home Assistant?
            url_from_options = options.get("hass_url", "empty")
            key_from_options = options.get("long_lived_token", "empty")

            # If data path specified by options.json, overwrite emhass_conf['data_path']
            data_path_value = options.get("data_path", None)
            if (
                data_path_value is not None
                and data_path_value != ""
                and data_path_value != "default"
            ):
                # Try to create directory if it doesn't exist. if successful set data_path in emhass_conf
                try:
                    data_path = pathlib.Path(data_path_value)
                    # Use parents=True to create nested directories
                    data_path.mkdir(parents=True, exist_ok=True)
                    emhass_conf["data_path"] = data_path
                    logger.info(f"Using custom data_path: {data_path}")
                except Exception as e:
                    logger.warning(
                        f"Cannot create data_path directory '{data_path_value}' provided via options. Keeping default. Error: {e}"
                    )

            # If config path specified by options.json, overwrite emhass_conf['config_path']
            config_path_value = options.get("config_path", None)
            if (
                config_path_value is not None
                and config_path_value != ""
                and config_path_value != "default"
            ):
                try:
                    config_path = pathlib.Path(config_path_value)
                    # Validate that the config file or its parent directory path is valid
                    if config_path.exists():
                        # File exists - use it
                        emhass_conf["config_path"] = config_path
                        logger.info(f"Using custom config_path from addon settings: {config_path}")
                    elif config_path.parent.exists():
                        # Parent directory exists but file doesn't - set path anyway (file may be created later)
                        emhass_conf["config_path"] = config_path
                        logger.warning(
                            f"Config file does not exist yet: {config_path} (will use defaults until created)"
                        )
                    else:
                        # Neither file nor parent directory exists - this is likely an error
                        logger.error(
                            f"Invalid config_path '{config_path_value}': parent directory does not exist. Keeping default config_path."
                        )
                except Exception as e:
                    logger.warning(
                        f"Cannot set config_path '{config_path_value}' provided via options. Keeping default. Error: {e}"
                    )
            else:
                # No config path provided via options.json, check default and legacy paths
                # This will move the config file to the addon_config_path if it is found in the legacy location.
                logger.info(
                    "No config_path provided via options.json, checking default (/config/config.json) and legacy path (/share/config.json)."
                )
                default_config_path = pathlib.Path("/config/config.json")
                default_config_dir_exists = default_config_path.parent.exists()
                legacy_config_path = pathlib.Path("/share/config.json")

                if default_config_path.is_file():
                    logger.info("Found config.json in /config, using this path for config_path.")
                    emhass_conf["config_path"] = default_config_path
                elif legacy_config_path.is_file():
                    # found legacy config path, move the file to the default addon-mode config path and use it for config_path
                    if default_config_dir_exists:
                        try:
                            shutil.move(str(legacy_config_path), str(default_config_path))
                            logger.info(
                                f"Moved legacy config from {legacy_config_path} to {default_config_path} and using it for config_path."
                            )
                            emhass_conf["config_path"] = default_config_path
                        except Exception as e:
                            logger.warning(
                                f"Failed to move legacy config from {legacy_config_path} to {default_config_path}: {e}"
                            )
                            emhass_conf["config_path"] = legacy_config_path
                    else:
                        logger.warning(
                            f"Directory {default_config_path.parent} does not exist, keeping legacy config_path: {legacy_config_path}."
                        )
                        emhass_conf["config_path"] = legacy_config_path
                elif default_config_dir_exists:
                    logger.info(
                        "No legacy config.json found in /share, using addon-mode default /config/config.json for config_path."
                    )
                    emhass_conf["config_path"] = default_config_path
                else:
                    logger.warning(
                        f"Directory {default_config_path.parent} does not exist, keeping current legacy default"
                    )

            # Check to use Home Assistant local API
            if not no_response and os.getenv("SUPERVISOR_TOKEN", None) is not None:
                params_secrets["long_lived_token"] = os.getenv("SUPERVISOR_TOKEN", None)
                # Use hass_url from options.json if available, otherwise use supervisor API for addon
                if url_from_options != "empty" and url_from_options != "":
                    params_secrets["hass_url"] = url_from_options
                else:
                    # For addons, use supervisor API for both REST and WebSocket access
                    params_secrets["hass_url"] = "http://supervisor/core/api"
                headers = {
                    "Authorization": "Bearer " + params_secrets["long_lived_token"],
                    "content-type": "application/json",
                }
                # Obtain secrets from Home Assistant via API
                logger.debug("Obtaining secrets from Home Assistant Supervisor API")
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        params_secrets["hass_url"] + "/config", headers=headers
                    ) as response:
                        if response.status < 400:
                            config_hass = await response.json()
                            params_secrets.update(
                                {
                                    "hass_url": params_secrets["hass_url"],
                                    "long_lived_token": params_secrets["long_lived_token"],
                                    "time_zone": config_hass["time_zone"],
                                    "Latitude": config_hass["latitude"],
                                    "Longitude": config_hass["longitude"],
                                    "Altitude": config_hass["elevation"],
                                    # If defined in HA config, use them, otherwise keep defaults
                                    "solcast_api_key": config_hass.get(
                                        "solcast_api_key", params_secrets["solcast_api_key"]
                                    ),
                                    "solcast_rooftop_id": config_hass.get(
                                        "solcast_rooftop_id", params_secrets["solcast_rooftop_id"]
                                    ),
                                    "solar_forecast_kwp": config_hass.get(
                                        "solar_forecast_kwp", params_secrets["solar_forecast_kwp"]
                                    ),
                                    "influxdb_username": config_hass.get(
                                        "influxdb_username", params_secrets.get("influxdb_username")
                                    ),
                                    "influxdb_password": config_hass.get(
                                        "influxdb_password", params_secrets.get("influxdb_password")
                                    ),
                                    "victoriametrics_username": config_hass.get(
                                        "victoriametrics_username",
                                        params_secrets.get("victoriametrics_username"),
                                    ),
                                    "victoriametrics_password": config_hass.get(
                                        "victoriametrics_password",
                                        params_secrets.get("victoriametrics_password"),
                                    ),
                                }
                            )
                        else:
                            # Obtain the url and key secrets if any from options.json (default /app/options.json)
                            logger.warning(
                                "Error obtaining secrets from Home Assistant Supervisor API"
                            )
                            logger.debug("Obtaining url and key secrets from options.json")
                            if url_from_options != "empty" and url_from_options != "":
                                params_secrets["hass_url"] = url_from_options
                            if key_from_options != "empty" and key_from_options != "":
                                params_secrets["long_lived_token"] = key_from_options
                            if (
                                options.get("time_zone", "empty") != "empty"
                                and options["time_zone"] != ""
                            ):
                                params_secrets["time_zone"] = options["time_zone"]
                            if options.get("Latitude", None) is not None and bool(
                                options["Latitude"]
                            ):
                                params_secrets["Latitude"] = options["Latitude"]
                            if options.get("Longitude", None) is not None and bool(
                                options["Longitude"]
                            ):
                                params_secrets["Longitude"] = options["Longitude"]
                            if options.get("Altitude", None) is not None and bool(
                                options["Altitude"]
                            ):
                                params_secrets["Altitude"] = options["Altitude"]

            # Obtain the forecast secrets (if any) from options.json (default /app/options.json)
            # This logic runs regardless of whether HA API call above succeeded or failed,
            # so we removed the duplicate logic from the 'else' block above.
            forecast_secrets = [
                "solcast_api_key",
                "solcast_rooftop_id",
                "solar_forecast_kwp",
            ]
            if any(x in forecast_secrets for x in list(options.keys())):
                logger.debug("Obtaining forecast secrets from options.json")
                if (
                    options.get("solcast_api_key", "empty") != "empty"
                    and options["solcast_api_key"] != ""
                ):
                    params_secrets["solcast_api_key"] = options["solcast_api_key"]
                if (
                    options.get("solcast_rooftop_id", "empty") != "empty"
                    and options["solcast_rooftop_id"] != ""
                ):
                    params_secrets["solcast_rooftop_id"] = options["solcast_rooftop_id"]
                if options.get("solar_forecast_kwp", None) and bool(options["solar_forecast_kwp"]):
                    params_secrets["solar_forecast_kwp"] = options["solar_forecast_kwp"]

            # Obtain InfluxDB secrets from options.json
            influx_secrets = ["influxdb_username", "influxdb_password"]
            if any(x in influx_secrets for x in list(options.keys())):
                logger.debug("Obtaining InfluxDB secrets from options.json")
                if (
                    options.get("influxdb_username", "empty") != "empty"
                    and options["influxdb_username"] != ""
                ):
                    params_secrets["influxdb_username"] = options["influxdb_username"]
                if (
                    options.get("influxdb_password", "empty") != "empty"
                    and options["influxdb_password"] != ""
                ):
                    params_secrets["influxdb_password"] = options["influxdb_password"]

            # Obtain VictoriaMetrics secrets from options.json (Basic auth, optional)
            vm_secrets = ["victoriametrics_username", "victoriametrics_password"]
            if any(x in vm_secrets for x in list(options.keys())):
                logger.debug("Obtaining VictoriaMetrics secrets from options.json")
                for key in vm_secrets:
                    if options.get(key, "empty") != "empty" and options[key] != "":
                        params_secrets[key] = options[key]

    # Obtain secrets from secrets_emhass.yaml? (default /app/secrets_emhass.yaml)
    if secrets_path and pathlib.Path(secrets_path).is_file():
        logger.debug("Obtaining secrets from secrets file")
        async with aiofiles.open(pathlib.Path(secrets_path)) as file:
            content = await file.read()
        # The parser's own error text can quote the secret: a YAML error quotes
        # the line, and a !!bool, !!int or !!float tag that does not fit raises
        # an error with the value lowercased or without its "_", which the log
        # mask cannot match. Only the error type and the line number are kept.
        try:
            loaded = yaml.safe_load(content)
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            where = f" near line {mark.line + 1}" if mark is not None else ""
            raise yaml.YAMLError(
                f"secrets file is not valid YAML ({type(e).__name__}{where})"
            ) from None
        except Exception as e:
            raise ValueError(f"Could not read the secrets file ({type(e).__name__})") from None
        params_secrets.update(loaded)
    # The web server's API token is read by api_auth on its own. Keep it out of the
    # secrets that go on into params, logs and the configuration routes.
    params_secrets.pop("emhass_api_token", None)

    # Receive key and url from ARG/arguments?
    if argument.get("url") is not None:
        params_secrets["hass_url"] = argument["url"]
        logger.debug("Obtaining url from passed argument")
    if argument.get("key") is not None:
        params_secrets["long_lived_token"] = argument["key"]
        logger.debug("Obtaining long_lived_token from passed argument")

    return emhass_conf, params_secrets


async def build_params(
    emhass_conf: dict[str, pathlib.Path],
    params_secrets: dict[str, str | float],
    config: dict[str, str],
    logger: logging.Logger,
) -> dict[str, dict]:
    """
    Build the main params dictionary from the config and secrets
    Appends configuration catagories used by emhass to the parameters. (with use of the associations file as a reference)

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict[str, pathlib.Path]
    :param params_secrets: The dictionary containing the built secret variables
    :type params_secrets: dict[str, str | float]
    :param config: The dictionary of built config parameters
    :type config: dict[str, str]
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built param dictionary
    :rtype: dict[str, dict]
    """
    if not isinstance(params_secrets, dict):
        params_secrets = {}

    params = {}
    # Start with blank config catagories
    params["retrieve_hass_conf"] = {}
    params["params_secrets"] = {}
    params["optim_conf"] = {}
    params["plant_conf"] = {}

    # Obtain associations to categorize parameters to their corresponding config catagories
    if emhass_conf.get(
        "associations_path", get_root(__file__, num_parent=2) / "data/associations.csv"
    ).exists():
        async with aiofiles.open(emhass_conf["associations_path"]) as data:
            content = await data.read()
            associations = list(csv.reader(content.splitlines(), delimiter=","))
    else:
        logger.error(
            "Unable to obtain the associations file (associations.csv) in: "
            + str(emhass_conf["associations_path"])
        )
        return False

    # Association file key reference
    # association[0] = config catagories
    # association[1] = legacy parameter name
    # association[2] = parameter (config.json/config_defaults.json)
    # association[3] = parameter list name if exists (not used, from legacy options.json)
    # Use association list to append parameters from config into params (with corresponding config catagories)
    for association in associations:
        # If parameter has list_ name and parameter in config is presented with its list name
        # (ie, config parameter is in legacy options.json format)
        if len(association) == 4 and config.get(association[3]) is not None:
            # Extract lists of dictionaries
            if config[association[3]] and type(config[association[3]][0]) is dict:
                params[association[0]][association[2]] = [
                    i[association[2]] for i in config[association[3]]
                ]
            else:
                params[association[0]][association[2]] = config[association[3]]
        # Else, directly set value of config parameter to param
        elif config.get(association[2]) is not None:
            params[association[0]][association[2]] = config[association[2]]

    # Check if we need to create `list_hp_periods` from config (ie. legacy options.json format)
    if (
        params.get("optim_conf") is not None
        and config.get("list_peak_hours_periods_start_hours") is not None
        and config.get("list_peak_hours_periods_end_hours") is not None
    ):
        start_hours_list = [
            i["peak_hours_periods_start_hours"]
            for i in config["list_peak_hours_periods_start_hours"]
        ]
        end_hours_list = [
            i["peak_hours_periods_end_hours"] for i in config["list_peak_hours_periods_end_hours"]
        ]
        num_peak_hours = len(start_hours_list)
        list_hp_periods_list = {
            "period_hp_" + str(i + 1): [
                {"start": start_hours_list[i]},
                {"end": end_hours_list[i]},
            ]
            for i in range(num_peak_hours)
        }
        params["optim_conf"]["load_peak_hour_periods"] = list_hp_periods_list
    else:
        # Else, check param already contains load_peak_hour_periods from config
        if params["optim_conf"].get("load_peak_hour_periods", None) is None:
            logger.warning("Unable to detect or create load_peak_hour_periods parameter")

    # Format load_peak_hour_periods list to dict if necessary
    if params["optim_conf"].get("load_peak_hour_periods", None) is not None and isinstance(
        params["optim_conf"]["load_peak_hour_periods"], list
    ):
        params["optim_conf"]["load_peak_hour_periods"] = {
            key: d[key] for d in params["optim_conf"]["load_peak_hour_periods"] for key in d
        }

    # Call function to check parameter lists that require the same length as deferrable loads
    # If not, set defaults it fill in gaps
    if params["optim_conf"].get("number_of_deferrable_loads", None) is not None:
        num_def_loads = params["optim_conf"]["number_of_deferrable_loads"]
        # Looped over DEF_LOAD_ARRAY_PARAMS (name -> default) instead of one
        # call per array (#1040) - same defaults and call signature per entry.
        for def_array_name, def_array_default in DEF_LOAD_ARRAY_PARAMS.items():
            params["optim_conf"][def_array_name] = check_def_loads(
                num_def_loads,
                params["optim_conf"],
                def_array_default,
                def_array_name,
                logger,
            )
        # Validate deferrable_load_groups
        groups = params["optim_conf"].get("deferrable_load_groups", [])
        if groups:
            seen_indices = set()
            for gi, group in enumerate(groups):
                # Validate names
                names = group.get("names", [])
                if not names:
                    raise ValueError(
                        f"deferrable_load_groups[{gi}]: 'names' must contain at least 1 deferrable load reference"
                    )
                indices = []
                group_indices = set()
                for name in names:
                    try:
                        idx = int(name.replace("deferrable", ""))
                    except ValueError as err:
                        raise ValueError(
                            f"deferrable_load_groups[{gi}]: could not parse index from name '{name}'"
                        ) from err
                    if idx < 0 or idx >= num_def_loads:
                        raise ValueError(
                            f"deferrable_load_groups[{gi}]: '{name}' references index {idx}, "
                            f"but only {num_def_loads} deferrable loads are configured"
                        )
                    if idx in group_indices:
                        raise ValueError(
                            f"deferrable_load_groups[{gi}]: '{name}' is duplicated within the group"
                        )
                    if idx in seen_indices:
                        raise ValueError(
                            f"deferrable_load_groups[{gi}]: '{name}' is already in another group. "
                            f"A deferrable load cannot belong to multiple groups"
                        )
                    indices.append(idx)
                    group_indices.add(idx)
                seen_indices.update(indices)

                # Validate max_power (optional when mutual_exclusion is true)
                max_power = group.get("max_power")
                mutual_exclusion = group.get("mutual_exclusion", False)
                if max_power is not None and max_power <= 0:
                    raise ValueError(
                        f"deferrable_load_groups[{gi}]: 'max_power' must be a positive number"
                    )
                if not isinstance(mutual_exclusion, bool):
                    raise ValueError(
                        f"deferrable_load_groups[{gi}]: 'mutual_exclusion' must be a boolean"
                    )
                if max_power is None and not mutual_exclusion:
                    raise ValueError(
                        f"deferrable_load_groups[{gi}]: 'max_power' is required when 'mutual_exclusion' is false"
                    )
    else:
        logger.warning("unable to obtain parameter: number_of_deferrable_loads")

    # Normalise per-battery array params against number_of_batteries (#610).
    # Missing key defaults to 1 (single-battery, the only shape supported
    # before this parameter existed); N=1 is a true no-op (see
    # check_batt_params docstring).
    num_batteries = validate_num_batteries(params["plant_conf"])
    for batt_param_name, batt_default in BATT_ARRAY_PARAMS_PLANT_CONF.items():
        check_batt_params(
            num_batteries, params["plant_conf"], batt_default, batt_param_name, logger
        )
    for batt_param_name, batt_default in BATT_ARRAY_PARAMS_OPTIM_CONF.items():
        check_batt_params(
            num_batteries, params["optim_conf"], batt_default, batt_param_name, logger
        )
    for batt_param_name in BATT_WEIGHT_PARAMS:
        check_batt_weight_params(num_batteries, params["optim_conf"], batt_param_name, logger)
    check_batt_charge_derating(
        num_batteries, params["plant_conf"], "battery_charge_power_derating", logger
    )
    # Per-inverter parameters (#1150). Never raises here: an unusable value is logged and
    # left for treat_runtimeparams to refuse, so /get-config and startup stay up and the
    # configuration can be corrected.
    inverter_problems = _inverter_config_problems(params["plant_conf"], logger)
    for inverter_problem in inverter_problems:
        logger.error("%s", inverter_problem)
    if inverter_problems:
        logger.error("Actions are refused until the inverter configuration is corrected.")
    # Persisted configuration recovers (an unusable curve is cleared and logged) so the
    # service and configuration page stay reachable; /set-config validates strictly first.
    check_inverter_power_curves(params["plant_conf"], logger, recover=True)

    # historic_days_to_retrieve should be no less then 2
    if params["retrieve_hass_conf"].get("historic_days_to_retrieve", None) is not None:
        if params["retrieve_hass_conf"]["historic_days_to_retrieve"] < 2:
            params["retrieve_hass_conf"]["historic_days_to_retrieve"] = 2
            logger.warning(
                "days_to_retrieve should not be lower then 2, setting days_to_retrieve to 2. Make sure your sensors also have at least 2 days of history"
            )
    else:
        logger.warning("unable to obtain parameter: historic_days_to_retrieve")

    # Configure secrets, set params to correct config categorie
    # retrieve_hass_conf
    params["retrieve_hass_conf"]["hass_url"] = params_secrets.get("hass_url")
    params["retrieve_hass_conf"]["long_lived_token"] = params_secrets.get("long_lived_token")
    params["retrieve_hass_conf"]["time_zone"] = params_secrets.get("time_zone")
    params["retrieve_hass_conf"]["Latitude"] = params_secrets.get("Latitude")
    params["retrieve_hass_conf"]["Longitude"] = params_secrets.get("Longitude")
    params["retrieve_hass_conf"]["Altitude"] = params_secrets.get("Altitude")
    if params_secrets.get("influxdb_username") is not None:
        params["retrieve_hass_conf"]["influxdb_username"] = params_secrets.get("influxdb_username")
        params["params_secrets"]["influxdb_username"] = params_secrets.get("influxdb_username")
    if params_secrets.get("influxdb_password") is not None:
        params["retrieve_hass_conf"]["influxdb_password"] = params_secrets.get("influxdb_password")
        params["params_secrets"]["influxdb_password"] = params_secrets.get("influxdb_password")
    for vm_secret in ("victoriametrics_username", "victoriametrics_password"):
        if params_secrets.get(vm_secret) is not None:
            params["retrieve_hass_conf"][vm_secret] = params_secrets.get(vm_secret)
            params["params_secrets"][vm_secret] = params_secrets.get(vm_secret)
    # The tokens site's remote solvers are reached with: only the secrets a
    # remote names (its token_secret), for the coordinator
    configured_site = params["optim_conf"].get("site")
    for element in configured_site if isinstance(configured_site, list) else []:
        solver = element.get("solver") if isinstance(element, dict) else None
        name = solver.get("token_secret") if isinstance(solver, dict) else None
        if site_token_secret_usable(name) and params_secrets.get(name) is not None:
            params["retrieve_hass_conf"].setdefault("remote_tokens", {})[name] = params_secrets[
                name
            ]
    # Update optional param secrets
    if params["optim_conf"].get("weather_forecast_method", None) is not None:
        if params["optim_conf"]["weather_forecast_method"] == "solcast":
            params["retrieve_hass_conf"]["solcast_api_key"] = params_secrets.get(
                "solcast_api_key", "123456"
            )
            params["params_secrets"]["solcast_api_key"] = params_secrets.get(
                "solcast_api_key", "123456"
            )
            params["retrieve_hass_conf"]["solcast_rooftop_id"] = params_secrets.get(
                "solcast_rooftop_id", "123456"
            )
            params["params_secrets"]["solcast_rooftop_id"] = params_secrets.get(
                "solcast_rooftop_id", "123456"
            )
        elif params["optim_conf"]["weather_forecast_method"] == "solar.forecast":
            params["retrieve_hass_conf"]["solar_forecast_kwp"] = params_secrets.get(
                "solar_forecast_kwp", 5
            )
            params["params_secrets"]["solar_forecast_kwp"] = params_secrets.get(
                "solar_forecast_kwp", 5
            )
    else:
        logger.warning("Unable to detect weather_forecast_method parameter")
    #  Check if secrets parameters still defaults values
    secret_params = [
        "https://myhass.duckdns.org/",
        "thatverylongtokenhere",
        45.83,
        6.86,
        4807.8,
    ]
    if any(x in secret_params for x in params["retrieve_hass_conf"].values()):
        logger.warning("Some secret parameters values are still matching their defaults")

    # Set empty dict objects for params passed_data
    # To be latter populated with runtime parameters (treat_runtimeparams)
    params["passed_data"] = {
        "pv_power_forecast": None,
        "load_power_forecast": None,
        "load_cost_forecast": None,
        "prod_price_forecast": None,
        "prediction_horizon": None,
        "soc_init": None,
        "soc_final": None,
        "soc_target": None,
        "soc_target_timestep": None,
        "battery_availability": None,
        "battery_discharge_energy_max": None,
        "def_availability": None,
        "current_period_peak": None,
        "capacity_charge_window": None,
        "capacity_charge_consideration": None,
        "capacity_charge_current_interval_history": None,
        "operating_hours_of_each_deferrable_load": None,
        "start_timesteps_of_each_deferrable_load": None,
        "end_timesteps_of_each_deferrable_load": None,
        "alpha": None,
        "beta": None,
    }

    # Canonicalise the structural multi-component capacity-charge params (#540
    # Part B) so a config-UI-serialised singleton list / empty list is
    # normalised here too - /get-config and /get-json then return the canonical
    # scalar/list form, and a wrong-length interval list is rejected at
    # config-build time. treat_runtimeparams re-runs the same idempotent helper
    # once runtime overrides are merged.
    params["optim_conf"] = canonicalize_capacity_charge_config(params["optim_conf"], logger)

    return params


# Per-deferrable-load ARRAY parameters normalised against
# number_of_deferrable_loads via check_def_loads (#929, #1040). build_params
# loops over this table to pad each one from config; treat_runtimeparams's
# re-normalisation pass reuses it to re-apply the same padding after a
# runtime override may have replaced one of these arrays with a raw,
# potentially short, value (the per-battery arrays already get an equivalent
# pass via check_batt_params, #610).
DEF_LOAD_ARRAY_PARAMS: dict[str, bool | int | float] = {
    "start_timesteps_of_each_deferrable_load": 0,
    "end_timesteps_of_each_deferrable_load": 0,
    "set_deferrable_load_single_constant": False,
    "treat_deferrable_load_as_semi_cont": True,
    "set_deferrable_startup_penalty": 0.0,
    "deferrable_load_max_cost": 0.0,
    "set_deferrable_max_startups": 0,
    "operating_hours_of_each_deferrable_load": 0,
    "nominal_power_of_deferrable_loads": 0,
    "def_minimum_on_time": 0,
    "def_minimum_off_time": 0,
}
# Legacy (pre-#342) names for the same arrays, from
# src/emhass/data/associations.csv column 2. The association loop accepts
# either the modern name (column 3) or this legacy one, so the
# runtime-provided check in treat_runtimeparams has to match both.
DEF_LOAD_ARRAY_LEGACY_NAMES: dict[str, str] = {
    "start_timesteps_of_each_deferrable_load": "def_start_timestep",
    "end_timesteps_of_each_deferrable_load": "def_end_timestep",
    "set_deferrable_load_single_constant": "set_def_constant",
    "treat_deferrable_load_as_semi_cont": "treat_def_as_semi_cont",
    "set_deferrable_startup_penalty": "def_start_penalty",
    "deferrable_load_max_cost": "deferrable_load_max_cost",
    "set_deferrable_max_startups": "set_deferrable_max_startups",
    "operating_hours_of_each_deferrable_load": "def_total_hours",
    "nominal_power_of_deferrable_loads": "P_deferrable_nom",
}


def _as_capacity_sequence(value):
    """Return ``value`` as a plain list when it is a list / tuple / 1-D
    ``np.ndarray`` / a stringified list (``"[3.0, 7.0]"`` from an HA template),
    else ``None``. A bare scalar or an unparseable string returns ``None`` so
    the caller leaves it untouched for the legacy scalar path.
    """
    if isinstance(value, list | tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        return list(parsed) if isinstance(parsed, list | tuple) else None
    return None


def canonicalize_capacity_charge_config(
    optim_conf: dict, logger: logging.Logger | None = None
) -> dict:
    """Normalise the two structural multi-component capacity-charge params to
    their canonical form BEFORE any structural decision (issue #540 Part B).

    Must run before ``OptimizationCacheKey`` construction and before the model
    build so a config-UI-serialised singleton list, an empty list or a
    stringified list all map to the same canonical configuration, and a
    wrong-length interval list is rejected rather than silently reinterpreting
    the tariff measurement basis.

    ``capacity_cost_per_kw``: ``[]`` becomes ``0.0`` (disabled K=1); ``[x]`` /
    ``(x,)`` / ``"[x]"`` becomes the scalar ``x`` (legacy K=1 - the #610
    singleton-list-is-a-single-instance rule); a list of length >= 2 stays a
    list (``K = len``); a bare scalar or an absent key is left unchanged.

    ``capacity_charge_interval_timesteps`` is canonicalised the same way (the
    per-component list form is only meaningful once the rate is a K>=2 list):
    ``[n]`` / ``"[n]"`` becomes the scalar ``n`` (shared measurement basis);
    ``[]`` becomes ``1``; a list of exactly K entries stays a list
    (component-specific bases); any other list length raises ``ValueError``; a
    bare scalar or an absent key is left unchanged.

    Mutates and returns ``optim_conf``. Idempotent.
    """
    if "capacity_cost_per_kw" in optim_conf:
        seq = _as_capacity_sequence(optim_conf["capacity_cost_per_kw"])
        if seq is not None:
            if len(seq) == 0:
                optim_conf["capacity_cost_per_kw"] = 0.0
            elif len(seq) == 1:
                optim_conf["capacity_cost_per_kw"] = seq[0]
            else:
                optim_conf["capacity_cost_per_kw"] = seq

    # K derived from the now-canonical rate: a bare scalar is K=1, a list is K=len.
    canonical_rate = optim_conf.get("capacity_cost_per_kw", 0.0)
    k_components = len(canonical_rate) if isinstance(canonical_rate, list | tuple) else 1

    if "capacity_charge_interval_timesteps" in optim_conf:
        seq = _as_capacity_sequence(optim_conf["capacity_charge_interval_timesteps"])
        if seq is not None:
            if len(seq) == 0:
                optim_conf["capacity_charge_interval_timesteps"] = 1
            elif len(seq) == 1:
                optim_conf["capacity_charge_interval_timesteps"] = seq[0]
            elif len(seq) == k_components:
                optim_conf["capacity_charge_interval_timesteps"] = seq
            else:
                raise ValueError(
                    f"capacity_charge_interval_timesteps has {len(seq)} entries but "
                    f"capacity_cost_per_kw defines {k_components} capacity component(s); "
                    "provide a single value (shared measurement basis) or a list of "
                    f"exactly {k_components} entries (component-specific measurement "
                    "bases). Refusing to silently reinterpret the tariff measurement "
                    "basis."
                )
    return optim_conf


def check_def_loads(
    num_def_loads: int,
    parameter: list[dict],
    default: str | float,
    parameter_name: str,
    logger: logging.Logger,
) -> list[dict]:
    """
    Check parameter lists with deferrable loads number, if they do not match, enlarge to fit.

    A missing key or ``None`` value is filled with ``default`` for every load. ``parameter``
    is updated in place (and the same list returned), matching how every call site reassigns
    ``params["optim_conf"][name] = check_def_loads(...)``.

    :param num_def_loads: Total number deferrable loads
    :type num_def_loads: int
    :param parameter: parameter config dict containing paramater
    :type parameter: list[dict]
    :param default: default value for parameter to pad missing
    :type default: str | int | float
    :param parameter_name: name of parameter
    :type parameter_name: str
    :param logger: The logger object
    :type logger: logging.Logger
    :return: parameter list
    :rtype: list[dict]
    """
    current = parameter.get(parameter_name, None)
    # Missing key or explicit JSON null: apply the default for every load. This is
    # expected when a per-load array is absent from the user config (e.g. a parameter
    # added in a later release), so do it silently rather than raising KeyError.
    if current is None:
        parameter[parameter_name] = [default] * num_def_loads
        return parameter[parameter_name]
    if isinstance(current, list) and num_def_loads > len(current):
        # Enlarging a short list to match number_of_deferrable_loads is this function's
        # documented job, not an error: the shipped defaults are sized for the default
        # load count, so any user that raises the count without restating every per-load
        # array lands here. Log at debug to avoid spurious startup warnings (#929).
        logger.debug(
            parameter_name
            + " has fewer entries than number_of_deferrable_loads, padding with default ("
            + str(default)
            + ")"
        )
        for _x in range(len(current), num_def_loads):
            current.append(default)
    result = parameter[parameter_name]
    # Replace any None elements with the default (can occur when set-config
    # receives a partial config, e.g. [null, 0] instead of [0, 0]).
    if isinstance(result, list):
        result = [v if v is not None else default for v in result]
        parameter[parameter_name] = result
    return result


def _warn_if_runtime_def_array_too_short(
    was_provided: bool,
    def_array_name: str,
    before_value: object,
    after_value: object,
    final_num_def_loads: int,
    logger: logging.Logger,
) -> None:
    """
    Warn when the re-normalisation pass actually changed a runtime-provided
    per-load deferrable array (#1040).

    check_def_loads pads a short array (and heals a None element) silently at
    debug level (#929) - correct for a config-sourced array, e.g. a config
    predating a load-count bump. A runtime-provided array is different: a
    stale/short one is the foot-gun #1040 reports (an MPC caller resending an
    old, shorter array after the load count changed elsewhere), so it gets a
    visible warning instead. The values are unchanged either way - this only
    makes a runtime-sourced change visible.

    Fires only when the caller has established the value was genuinely
    provided at runtime (`was_provided`) AND check_def_loads actually changed
    it. The comparison is by value, not identity, since check_def_loads
    always returns a freshly built list even when nothing changed - so an
    identity check would report "changed" even when heat_topology's own
    compiled value replaced the runtime one and nothing was really padded.

    Describes the outcome rather than the input shape: a length increase says
    how many entries were padded; a same-length change (a None element
    healed) says that instead of overclaiming padding.

    :param was_provided: whether the modern or legacy name carried a
        non-null value in runtimeparams
    :type was_provided: bool
    :param def_array_name: the modern per-load array parameter name
    :type def_array_name: str
    :param before_value: optim_conf[def_array_name] immediately before this
        call's check_def_loads (an independent copy, never mutated by it)
    :type before_value: object
    :param after_value: optim_conf[def_array_name] immediately after
    :type after_value: object
    :param final_num_def_loads: number_of_deferrable_loads after every reset
        inside treat_runtimeparams has been applied
    :type final_num_def_loads: int
    :param logger: The logger object
    :type logger: logging.Logger
    """
    if not was_provided or before_value == after_value:
        return
    if (
        isinstance(before_value, list)
        and isinstance(after_value, list)
        and len(before_value) < len(after_value)
    ):
        logger.warning(
            f"{def_array_name}: padded from {len(before_value)} to "
            f"{len(after_value)} entries to match "
            f"number_of_deferrable_loads={final_num_def_loads}. Pass a "
            "full-length array to avoid this warning."
        )
    else:
        logger.warning(
            f"{def_array_name}: a None/null entry was replaced with the "
            f"parameter default (number_of_deferrable_loads="
            f"{final_num_def_loads}). Pass a fully populated array to avoid "
            "this warning."
        )


# Per-battery ARRAY parameters (#610): scalar accepted and broadcast to every
# battery, exact-length-N list accepted as-is. Unlike check_def_loads (deferrable
# loads), a wrong-length list is a hard error, not padded - silently enlarging a
# physical-plant array risks masking a genuine per-battery mis-configuration.
# weight_battery_charge/discharge are intentionally excluded: they nest instead
# (see check_batt_weight_params) because a flat list there is already a time
# series today, not a per-battery array.
BATT_ARRAY_PARAMS_PLANT_CONF: dict[str, bool | int | float] = {
    "battery_discharge_power_max": 1000,
    "battery_charge_power_max": 1000,
    "battery_discharge_efficiency": 0.95,
    "battery_charge_efficiency": 0.95,
    "battery_nominal_energy_capacity": 5000,
    "battery_minimum_state_of_charge": 0.3,
    "battery_maximum_state_of_charge": 0.9,
    "battery_target_state_of_charge": 0.6,
    "battery_stress_cost": 0.0,
}
# Plant SOC bounds are fractions in [0, 1]. Checked on their own because the
# other entries in BATT_ARRAY_PARAMS_PLANT_CONF are watts, efficiencies, or
# costs, which are valid above 1 (#935).
SOC_FRACTION_PLANT_PARAMS: tuple[str, ...] = (
    "battery_minimum_state_of_charge",
    "battery_maximum_state_of_charge",
    "battery_target_state_of_charge",
)
# Penalty thresholds are fractions of capacity too. Their costs are currency
# and stay out of this list (#935).
SOC_FRACTION_OPTIM_PARAMS: tuple[str, ...] = (
    "battery_soc_deficit_threshold",
    "battery_soc_surplus_threshold",
)
BATT_ARRAY_PARAMS_OPTIM_CONF: dict[str, bool | int | float] = {
    "battery_soc_deficit_threshold": 0.4,
    "battery_soc_deficit_cost": 0.0,
    "battery_soc_surplus_threshold": 0.9,
    "battery_soc_surplus_cost": 0.0,
}
BATT_WEIGHT_PARAMS: tuple[str, ...] = ("weight_battery_charge", "weight_battery_discharge")


def _coerce_batt_element(
    value: bool | float | str | None, default: bool | float, parameter_name: str
) -> bool | int | float:
    """Coerce a single per-battery array element (#610).

    None or a stringly-typed "null" resolve to the per-slot default; a numeric
    string coerces to float; anything else passes through unchanged. A genuinely
    non-numeric string raises a clear ValueError rather than silently becoming 0.
    """
    if value is None:
        return default
    if isinstance(value, str):
        if value.strip().lower() == "null":
            return default
        try:
            return float(value)
        except ValueError as err:
            raise ValueError(f"{parameter_name}: non-numeric string element {value!r}") from err
    return value


# Every per-battery setting is expanded to one entry per battery, so the count has to
# be bounded: an unbounded one lets a single request allocate memory in proportion to
# the number it sends. 100 is far above any real plant.
MAX_BATTERIES = 100


def validate_num_batteries(plant_conf: dict) -> int:
    """
    Read and validate plant_conf["number_of_batteries"] (#610).

    Accepts a positive integer up to MAX_BATTERIES (an integral float or numeric
    string is coerced); zero, negative, fractional, boolean, non-numeric or
    larger values raise BatteryConfigError (a ValueError) here, at config time,
    rather than surfacing as an empty or oversized per-battery list deep inside
    the optimization build.

    :param plant_conf: the plant configuration dict
    :type plant_conf: dict
    :return: the validated battery count (missing key -> 1)
    :rtype: int
    """
    raw = plant_conf.get("number_of_batteries", 1)
    if isinstance(raw, int) and raw.bit_length() > 128:
        # repr of an int past sys.get_int_max_str_digits() raises ValueError.
        shown = f"an integer of {raw.bit_length()} bits"
    else:
        shown = repr(raw)
        if len(shown) > 40:
            # Not a cut copy: the log mask matches a secret pasted here by
            # mistake (the API token, say) only when it is whole.
            shown = f"a value of {len(shown)} characters"
    invalid = BatteryConfigError(f"number_of_batteries must be a positive integer, got {shown}")
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        raise invalid
    if isinstance(raw, int):
        value = raw  # compared as is: an int can be too large for a float
    else:
        try:
            value = float(raw)
        except ValueError:
            raise invalid from None
        if value == math.inf:  # a count too large for a float, such as "1e309"
            raise BatteryConfigError(
                f"number_of_batteries must be at most {MAX_BATTERIES}, got {shown}"
            )
        if not value.is_integer():  # also false for -inf and nan
            raise invalid
    if value < 1:
        raise invalid
    if value > MAX_BATTERIES:
        raise BatteryConfigError(
            f"number_of_batteries must be at most {MAX_BATTERIES}, got {shown}"
        )
    return int(value)


def require_valid_battery_count(plant_conf: dict, logger: logging.Logger) -> int:
    """validate_num_batteries for a run: the fault is logged at error level before the
    BatteryConfigError is raised, which is what lets the web server return the action
    log with its 400, as for SocFractionError and InverterConfigError."""
    try:
        return validate_num_batteries(plant_conf)
    except BatteryConfigError as err:
        logger.error("%s", err)
        raise


def battery_count_faults(config: dict) -> list[str]:
    """The fault with number_of_batteries in a configuration about to be saved, if any.

    A null value is not a fault: the web UI saves an emptied number input as null and
    build_params drops it, so the default of 1 applies. build_params drops only a real
    null, so the string "null" is checked like any other value.
    """
    if config.get("number_of_batteries") is None:
        return []
    try:
        validate_num_batteries(config)
    except BatteryConfigError as err:
        return [str(err)]
    return []


def check_batt_params(
    num_batteries: int,
    parameter: dict,
    default: bool | float,
    parameter_name: str,
    logger: logging.Logger,
) -> bool | int | float | list:
    """
    Normalise a per-battery plant_conf/optim_conf array parameter (#610).

    N == 1 (the default/off state) is a true no-op: a scalar stays a scalar,
    byte-identical to master, since optimization.py's per-battery model
    reduces to reading that same scalar at index 0. For N > 1: a missing key
    or scalar value broadcasts to ``[value] * num_batteries``; a list must be
    exactly ``num_batteries`` long (wrong length raises ValueError naming the
    parameter and the expected length - no silent padding, unlike deferrable
    loads). List elements are coerced per _coerce_batt_element (None/"null" ->
    default, numeric string -> float).

    :param num_batteries: plant_conf["number_of_batteries"]
    :type num_batteries: int
    :param parameter: parameter config dict containing parameter_name
    :type parameter: dict
    :param default: default value used to fill missing/None slots
    :type default: bool | int | float
    :param parameter_name: name of parameter
    :type parameter_name: str
    :param logger: The logger object
    :type logger: logging.Logger
    :return: the normalised value: a scalar at N=1, a length-num_batteries list at N>1
    :rtype: bool | int | float | list
    """
    current = parameter.get(parameter_name, None)
    if current is None:
        if num_batteries == 1:
            return current
        parameter[parameter_name] = [default] * num_batteries
        return parameter[parameter_name]
    if isinstance(current, list):
        if len(current) != num_batteries:
            raise ValueError(
                f"{parameter_name} has {len(current)} entries but "
                f"number_of_batteries={num_batteries}; provide a scalar (broadcast "
                f"to every battery) or a list of exactly {num_batteries} entries"
            )
        normalized = [_coerce_batt_element(v, default, parameter_name) for v in current]
        parameter[parameter_name] = normalized[0] if num_batteries == 1 else normalized
        return parameter[parameter_name]
    # Scalar (or stringly-typed scalar, e.g. a "null"/"0.0" string from HA).
    coerced = _coerce_batt_element(current, default, parameter_name)
    if num_batteries == 1:
        parameter[parameter_name] = coerced
        return coerced
    parameter[parameter_name] = [coerced] * num_batteries
    return parameter[parameter_name]


REPLACE_FILE_RETRIES = 20


def _copy_owner_and_mode(path: str, tmp: str) -> None:
    """Give the new file the owner and permissions of the one it replaces, as writing
    into the old file would have kept them. Best effort: where the process or the
    filesystem will not take one of them (not root, a user namespace, FUSE), the new
    file keeps what it was created with. Not on Windows, where the only mode bit is
    read-only, and a read-only new file could not be moved into place."""
    if os.name == "nt":
        return
    try:
        st = os.stat(path)
    except OSError:
        return
    try:
        os.chown(tmp, st.st_uid, st.st_gid)
    except OSError:
        pass
    try:
        os.chmod(tmp, stat.S_IMODE(st.st_mode) & 0o777)
    except OSError:
        pass


async def replace_file(path, content: str | bytes) -> None:
    """Write ``content`` to ``path`` through a temporary file and os.replace, so a
    reader running at the same time sees the old file or the new one, never a
    partly written one."""
    # Replace what a symlink points to, not the link itself.
    path = os.path.realpath(path)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        async with aiofiles.open(tmp, "wb" if isinstance(content, bytes) else "w") as f:
            await f.write(content)
        _copy_owner_and_mode(path, tmp)
        # On Windows a file that another handle has open cannot be replaced until it is
        # closed. Readers hold it for milliseconds, so retry briefly there.
        for attempt in range(REPLACE_FILE_RETRIES + 1):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == REPLACE_FILE_RETRIES:
                    raise
                await asyncio.sleep(0.05)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# A lock that one process holds and every other one - another Gunicorn worker
# included - fails to take at once, rather than waits for.
try:
    import fcntl as _fcntl

    def try_lock_file(f) -> bool:
        """Lock the open file ``f`` if no other open file holds it. True if locked."""
        try:
            _fcntl.flock(f, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def unlock_file(f) -> None:
        _fcntl.flock(f, _fcntl.LOCK_UN)

except ImportError:
    try:
        import msvcrt as _msvcrt

        def try_lock_file(f) -> bool:
            """Lock the open file ``f`` if no other open file holds it. True if locked."""
            f.seek(0)
            try:
                _msvcrt.locking(f.fileno(), _msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
            return True

        def unlock_file(f) -> None:
            f.seek(0)
            _msvcrt.locking(f.fileno(), _msvcrt.LK_UNLCK, 1)

    except ImportError:  # no locking primitive: nothing to hold

        def try_lock_file(f) -> bool:
            return True

        def unlock_file(f) -> None:
            pass


# A dry run solves as a live run does, returns the plan, and writes nothing.
# It is a URL of its own, not a runtime parameter: an EMHASS that predates it
# answers 404 there, where it would ignore an unknown parameter and run live.
DRY_RUN_URL = "/api/v1/dry-run/<action>"
DRY_RUN_ACTIONS = ("perfect-optim", "dayahead-optim", "naive-mpc-optim")
# Runtime parameters that ask for a write. A dry run refuses them rather than
# ignore them, so a caller never believes something was saved.
DRY_RUN_REFUSED_KEYS = ("entity_save", "weather_forecast_cache")
# The on/off parameters a dry run reads strictly, then passes on as booleans:
# the forecast reads them by truthiness, and the string "false" is truthy.
DRY_RUN_FLAGS = (*DRY_RUN_REFUSED_KEYS, "weather_forecast_cache_only")

_FLAG_TRUE = ("true", "1", "yes", "on")
_FLAG_FALSE = ("false", "0", "no", "off", "")


def read_flag(value: object, name: str) -> bool:
    """Read a runtime on/off parameter strictly.

    ``bool("false")`` is True, and a templated request often sends the string.
    Booleans, 0 and 1, None, and the strings true/false, 1/0, yes/no, on/off
    (any case) are read; anything else raises.

    :param value: The value sent for the parameter.
    :param name: The parameter's name, for the error message.
    :return: The parameter's value.
    :raises ValueError: For any other value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _FLAG_TRUE:
            return True
        if text in _FLAG_FALSE:
            return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def dry_run_request_problem(runtimeparams: dict, dry_run: bool) -> str | None:
    """Why a request must be refused before anything runs, or None.

    On an action URL a ``dry_run`` parameter that is not false is refused:
    that URL never honours it, and running live would replace the plan the
    caller meant to keep. On the dry-run URL, ``dry_run`` may be left out or
    true, and the parameters that ask for a write are refused.

    :param runtimeparams: The request's runtime parameters.
    :param dry_run: True for the dry-run URL, False for an action URL.
    :return: The reason to refuse the request, or None.
    """
    try:
        asked = read_flag(runtimeparams.get("dry_run"), "dry_run")
        if asked and not dry_run:
            return (
                f"dry_run is not a runtime parameter of this URL; send the same "
                f"body to {DRY_RUN_URL} for a dry run. Nothing was run"
            )
        if dry_run and "dry_run" in runtimeparams and not asked:
            return "dry_run is false in the body of a dry run. Nothing was run"
        if dry_run:
            for key in DRY_RUN_FLAGS:
                if read_flag(runtimeparams.get(key), key) and key in DRY_RUN_REFUSED_KEYS:
                    return f"{key} asks for a write, and a dry run writes nothing. Nothing was run"
    except ValueError as exc:
        return f"{exc}. Nothing was run"
    return None


class SocFractionError(ValueError):
    """A state of charge was given above 1, where a fraction in [0, 1] is required (#935)."""


class BatteryConfigError(ValueError):
    """number_of_batteries is not a whole number from 1 to MAX_BATTERIES (#610)."""


class InverterConfigError(ValueError):
    """An inverter setting is unusable: the count or a per-inverter parameter is invalid (#1150).

    The optimizer also raises it above one inverter for a plan the settings rule out
    (perfect-optim, or a PV forecast without the per-inverter split they need).

    ``problems`` lists every fault behind the error: just the error itself for a single
    fault, the individual faults for one that joins several (see _join_inverter_faults).
    """

    def __init__(self, *args, problems: Sequence[InverterConfigError] | None = None):
        super().__init__(*args)
        self.problems: list[InverterConfigError] = list(problems) if problems else [self]


class RuntimeParamError(ValueError):
    """A runtime parameter has an unusable value (a file or column name, or a
    per-load number list). The message is fixed text and never quotes the value."""


# Connection settings that only the configuration may set. A runtime request
# must not redirect the data source or carry its credentials elsewhere.
# use_influxdb is not here: the web UI's export passes it, and it only picks the
# configured InfluxDB server, which a request cannot change.
RUNTIME_DENIED_KEYS: frozenset[str] = frozenset(
    {
        "influxdb_host",
        "influxdb_port",
        "influxdb_database",
        "influxdb_measurement",
        "influxdb_retention_policy",
        "influxdb_use_ssl",
        "influxdb_verify_ssl",
        "influxdb_username",
        "influxdb_password",
        "use_victoriametrics",
        "victoriametrics_host",
        "victoriametrics_port",
        "victoriametrics_database",
        "victoriametrics_metric_regex",
        "victoriametrics_entity_label",
        "victoriametrics_use_ssl",
        "victoriametrics_verify_ssl",
        "victoriametrics_username",
        "victoriametrics_password",
        "use_websocket",
        # Not read back from a runtime request today, but listed so that it stays that way
        "hass_url",
        "long_lived_token",
        "ssl_no_verify",
        "emhass_api_token",
    }
)

_RUNTIME_FILE_NAME_MAX = 128
_TIMESTAMP_COL_NAME_RE = re.compile(r"[A-Za-z0-9_. -]{1,64}")
# A Home Assistant entity id: domain.object_id. Upper case is allowed because
# existing setups use ids like sensor.P_PV.
_ENTITY_ID_RE = re.compile(r"[A-Za-z0-9_]{1,64}\.[A-Za-z0-9_]{1,190}")
_PUBLISH_PREFIX_RE = re.compile(r"[A-Za-z0-9_]{0,64}")
# Runtime parameters that hold one entity id
_RUNTIME_ENTITY_ID_KEYS = ("model_predict_entity_id", "mlr_predict_entity_id")
# A runtime request may only publish to sensor entities (every EMHASS default and
# documented id is one); other domains could overwrite e.g. a door sensor's state.
_RUNTIME_ENTITY_DOMAINS = ("sensor.",)
# The cost functions the optimizer knows. costfun is also part of a results file name.
COSTFUN_VALUES = ("profit", "cost", "self-consumption")
# A database host setting: a host name or an IP address, nothing that could add a
# scheme, credentials, a path or a query to the URL built from it.
_DB_HOST_RE = re.compile(r"[A-Za-z0-9_.:\[\]-]{1,253}")
# (use flag, host key, port key) for each database integration
_DB_KEYS = (
    ("use_influxdb", "influxdb_host", "influxdb_port"),
    ("use_victoriametrics", "victoriametrics_host", "victoriametrics_port"),
)


def is_valid_db_host(value) -> bool:
    """True when ``value`` is a plain host name or IP address (no scheme, path or query)."""
    return isinstance(value, str) and _DB_HOST_RE.fullmatch(value) is not None


def is_valid_db_port(value) -> bool:
    """True when ``value`` is a port number from 1 to 65535, as an int or a string of digits."""
    if isinstance(value, bool):
        return False
    if isinstance(value, str) and value.isascii() and value.isdigit():
        value = int(value)
    return isinstance(value, int) and 1 <= value <= 65535


def config_value_faults(config: dict) -> list[str]:
    """Faults in the settings that name files, hosts or the cost function.

    Used when a configuration is saved, so a value that the runtime checks would
    refuse cannot be stored instead.
    """
    faults = []
    costfun = config.get("costfun")
    if costfun is not None and costfun not in COSTFUN_VALUES:
        faults.append(f"costfun must be one of {', '.join(COSTFUN_VALUES)}")
    for use_key, host_key, port_key in _DB_KEYS:
        enabled = bool(config.get(use_key))
        # The web UI sends "" for a cleared field: fine while the integration is off
        # A key that is not sent keeps its default
        host = config.get(host_key)
        if host is not None and (enabled or host != "") and not is_valid_db_host(host):
            faults.append(f"{host_key} must be a host name or IP address, without a scheme or path")
        port = config.get(port_key)
        if port is not None and (enabled or port != "") and not is_valid_db_port(port):
            faults.append(f"{port_key} must be a port number from 1 to 65535")
    model_type = config.get("model_type")
    if model_type is not None:
        try:
            validate_runtime_file_name("model_type", model_type)
        except RuntimeParamError as exc:
            faults.append(str(exc))
    return faults


def data_file_path(data_path, name) -> pathlib.Path:
    """The file ``name`` inside the data folder, refused if it would land outside it.

    A second line of defence behind the runtime checks: every file whose name a
    request or a saved setting can influence is opened through this.

    :param data_path: The data folder.
    :param name: A file name, or a relative path such as ``entities/sensor.x.json``.
    :raises RuntimeParamError: When ``name`` resolves outside ``data_path``.
    """
    # Only the folder the file goes in is resolved and checked. The file itself may be
    # a symlink to somewhere else on purpose (atomic_write and replace_file replace
    # what a link points to), so following it here would refuse a working setup.
    refused = RuntimeParamError(f"{name!r} is not a file name inside the data folder")
    try:
        base = os.path.realpath(str(data_path))
        folder, file_name = os.path.split(os.path.join(base, str(name)))
        # "", ".", ".." and, on Windows, names such as ". " or "..." that it reads as
        # one of those are not file names
        if file_name.rstrip(". ") == "" or "\x00" in str(name):
            raise refused
        folder = os.path.realpath(folder)
        inside = folder == base or os.path.commonpath([base, folder]) == base
    except ValueError:  # another drive on Windows, or a NUL byte in the name
        inside = False
    if not inside:
        raise refused
    path = pathlib.Path(folder) / file_name
    # A folder (or a link to one) is not a file a caller may write or load
    if path.is_dir():
        raise refused
    return path


def is_valid_entity_id(value) -> bool:
    """True when ``value`` is a plain ``domain.object_id`` entity id.

    Entity ids become part of a Home Assistant URL path and of a saved file
    name, so anything else (a slash, ``..``, a query) is refused.
    """
    return isinstance(value, str) and _ENTITY_ID_RE.fullmatch(value) is not None


def validate_runtime_entity_ids(runtimeparams: dict) -> None:
    """Check every entity id and the publish prefix a runtime request sets.

    Covers ``model_predict_entity_id``, ``mlr_predict_entity_id``,
    ``publish_prefix`` and the ``entity_id`` of every ``custom_*_id`` object
    (or list of objects, for the per-load ones).

    :raises RuntimeParamError: Naming the parameter whose value is not usable.
    """
    prefix = runtimeparams.get("publish_prefix", "")
    if not isinstance(prefix, str) or _PUBLISH_PREFIX_RE.fullmatch(prefix) is None:
        raise RuntimeParamError("publish_prefix must be at most 64 letters, digits or underscores")

    def check(label: str, value) -> None:
        if not is_valid_entity_id(value):
            raise RuntimeParamError(f"{label} must be an entity id like sensor.my_name")
        if not value.startswith(_RUNTIME_ENTITY_DOMAINS):
            raise RuntimeParamError(f"{label} must be a sensor entity id, like sensor.my_name")
        # post_data adds the prefix after "sensor."; the result must still be an entity id
        if not is_valid_entity_id(value.replace("sensor.", "sensor." + prefix)):
            raise RuntimeParamError(f"{label} is too long once publish_prefix is added")

    for key in _RUNTIME_ENTITY_ID_KEYS:
        if key in runtimeparams:
            check(key, runtimeparams[key])
    for key, value in runtimeparams.items():
        if not (key.startswith("custom_") and key.endswith("_id")):
            continue
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict) and "entity_id" in item:
                check(f"{key} entity_id", item["entity_id"])


def validate_runtime_file_name(name: str, value, suffix: str | None = None) -> str:
    """Check a caller-supplied plain file name and return it.

    It must be a non-empty string of at most 128 characters, with no path
    separator or control character, not starting with a dot and without ``..``.
    ``suffix`` (for example ``".csv"``) is matched case-insensitively.

    :param name: The runtime parameter name, used in the error message.
    :param value: The value the caller sent.
    :param suffix: Required file name ending, if any.
    :raises RuntimeParamError: When the value is not a plain file name.
    """
    if not isinstance(value, str) or not value:
        raise RuntimeParamError(f"{name} must be a non-empty string")
    if len(value) > _RUNTIME_FILE_NAME_MAX:
        raise RuntimeParamError(f"{name} is longer than {_RUNTIME_FILE_NAME_MAX} characters")
    if "/" in value or "\\" in value:
        raise RuntimeParamError(f"{name} must be a plain file name, without a path")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise RuntimeParamError(f"{name} must not contain control characters")
    if value.startswith(".") or ".." in value:
        raise RuntimeParamError(f"{name} must not start with a dot or contain '..'")
    if suffix is not None and not value.lower().endswith(suffix.lower()):
        raise RuntimeParamError(f"{name} must end with {suffix}")
    return value


def validate_timestamp_col_name(value) -> str:
    """Check the ``timestamp_col_name`` runtime parameter and return it.

    :raises RuntimeParamError: When it is not 1 to 64 letters, digits, ``_``,
        ``.``, ``-`` or spaces.
    """
    if not isinstance(value, str) or _TIMESTAMP_COL_NAME_RE.fullmatch(value) is None:
        raise RuntimeParamError(
            "timestamp_col_name must be 1 to 64 letters, digits, '_', '.', '-' or spaces"
        )
    return value


def reject_configured_soc_fractions(
    plant_conf: dict, optim_conf: dict, logger: logging.Logger
) -> None:
    """Reject a saved or overridden SOC fraction above 1 (#935).

    Plant bounds and the SOC penalty thresholds are fractions of battery
    capacity. Their costs are currency and are not checked here.
    """
    for soc_name in SOC_FRACTION_PLANT_PARAMS:
        reject_soc_fraction_above_one(soc_name, plant_conf.get(soc_name), logger)
    for soc_name in SOC_FRACTION_OPTIM_PARAMS:
        reject_soc_fraction_above_one(soc_name, optim_conf.get(soc_name), logger)


def reject_soc_fraction_above_one(name: str, value, logger: logging.Logger) -> None:
    """Raise SocFractionError when a SOC value is above 1 (#935).

    ``1`` is a full battery. ``None`` means the value was omitted. A numeric
    string is read as a number, so ``"80"`` is rejected the same way as ``80``.
    Anything else that is not a number is left untouched. An out-of-band
    fraction such as ``0.05`` stays valid for the existing recovery and clamp
    logic.
    """
    if value is None:
        return
    items = value if isinstance(value, list) else [value]
    for index, item in enumerate(items):
        number = item
        if isinstance(item, str):
            try:
                number = float(item)
            except ValueError:
                continue
        elif isinstance(item, bool) or not isinstance(item, int | float):
            continue
        if number <= 1:
            continue
        label = f"{name}[{index}]" if isinstance(value, list) else name
        message = (
            f"{label}={number} is above 1. State of charge is a fraction in "
            f"[0, 1] (0.65 means 65%), not a percentage."
        )
        logger.error("%s", message)
        raise SocFractionError(message)


def _warn_if_runtime_scalar_masks_batt_list(
    num_batteries: int,
    parameter: dict,
    parameter_name: str,
    distinct_config_lists: dict[str, list],
    logger: logging.Logger,
) -> None:
    """
    Warn when a runtime scalar masks a configured per-battery list (#1032).

    Runtime values override configured ones by design, and a scalar broadcasts
    to every battery. When the configured value was a list with genuinely
    distinct per-battery entries, that broadcast silently flattens them - a
    documented but easy-to-miss interaction (a real two-battery deployment
    lost its per-unit power limits to an aggregate runtime scalar this way).
    The resulting values are unchanged; this only makes the override visible.

    Uniform configured lists stay silent on purpose: build_params normalises a
    configured scalar to ``[value] * N`` before this runs, so a warning keyed
    on "configured value is a list" alone would fire on every runtime scalar
    a broadcast-config user sends, every MPC cycle.

    :param num_batteries: plant_conf["number_of_batteries"]
    :type num_batteries: int
    :param parameter: the plant_conf/optim_conf dict AFTER the runtime
        association loop has applied any override
    :type parameter: dict
    :param parameter_name: name of the per-battery array parameter
    :type parameter_name: str
    :param distinct_config_lists: parameter name -> configured list, captured
        BEFORE the association loop, only for lists with distinct entries
    :type distinct_config_lists: dict[str, list]
    :param logger: The logger object
    :type logger: logging.Logger
    """
    if num_batteries <= 1 or parameter_name not in distinct_config_lists:
        return
    current = parameter.get(parameter_name)
    if current is None or isinstance(current, list):
        return
    logger.warning(
        f"{parameter_name}: runtime scalar {current} overrides the configured "
        f"per-battery list {distinct_config_lists[parameter_name]} and will "
        f"broadcast to all {num_batteries} batteries; pass a list of "
        f"{num_batteries} entries to keep per-battery values"
    )


# How many entries of a masked per-inverter list the runtime warning shows.
_MAX_MASKED_LIST_SHOWN = 10


def _warn_if_runtime_scalar_masks_inverter_list(
    num_inverters: int,
    plant_conf: dict,
    parameter_name: str,
    distinct_config_lists: dict[str, list],
    logger: logging.Logger,
) -> None:
    """
    Warn when a runtime scalar masks a configured per-inverter list (#1150).

    The same rule as for batteries (#1032): a runtime scalar broadcasts to every
    inverter and flattens a configured list whose entries differ. The values are
    unchanged; this only makes the override visible. Uniform lists stay silent, and so
    does a single inverter. The configured list is shown truncated, so the message
    stays short however long the list is.

    :param num_inverters: the validated plant_conf["number_of_inverters"]
    :type num_inverters: int
    :param plant_conf: the plant_conf dict AFTER the runtime association loop
    :type plant_conf: dict
    :param parameter_name: name of the per-inverter broadcast parameter
    :type parameter_name: str
    :param distinct_config_lists: parameter name -> configured list, captured BEFORE
        the association loop, only for lists with distinct entries
    :type distinct_config_lists: dict[str, list]
    :param logger: The logger object
    :type logger: logging.Logger
    """
    if num_inverters <= 1 or parameter_name not in distinct_config_lists:
        return
    current = plant_conf.get(parameter_name)
    if current is None or isinstance(current, list):
        return
    configured = distinct_config_lists[parameter_name]
    shown = repr(configured[:_MAX_MASKED_LIST_SHOWN])
    if len(configured) > _MAX_MASKED_LIST_SHOWN:
        shown += f" and {len(configured) - _MAX_MASKED_LIST_SHOWN} more entries"
    logger.warning(
        f"{parameter_name}: runtime scalar {current!r} overrides the configured "
        f"per-inverter list {shown} and will broadcast to all {num_inverters} "
        f"inverters; pass a list of {num_inverters} entries to keep per-inverter values"
    )


# Per-inverter parameters (#1150). The broadcast ones follow the battery rule: a scalar
# is the same value for every inverter, an exact-length list is per inverter, and a
# list of the wrong length is a hard error. The caps are never broadcast, since one
# number cannot be both a per-unit cap and a site total. Values are the defaults that
# fill a missing or null slot.
INVERTER_BROADCAST_PARAMS: dict[str, bool | float] = {
    "inverter_efficiency_dc_ac": 1.0,
    "inverter_efficiency_ac_dc": 1.0,
    "inverter_stress_cost": 0.0,
    "compute_curtailment": False,
}
INVERTER_CAP_PARAMS: tuple[str, ...] = ("inverter_ac_output_max", "inverter_ac_input_max")
INVERTER_INDEX_PARAMS: tuple[str, ...] = ("battery_inverter_index", "pv_plant_inverter_index")


def _is_null_like(value) -> bool:
    """True for None and for a stringly-typed "null", the two ways an empty value arrives."""
    return value is None or (isinstance(value, str) and value.strip().lower() == "null")


def validate_num_inverters(conf: dict) -> int:
    """
    Read and validate conf["number_of_inverters"] (#1150).

    Accepts a positive integer (an integral float or numeric string is coerced);
    zero, negative, fractional, boolean or non-numeric values raise
    InverterConfigError. A missing key gives 1, the only shape before this
    parameter existed. So does a null value (None, "null" or an empty string): the
    web UI saves an emptied number input as null, and build_params skips null
    values, which is how a null number_of_batteries becomes the default today.

    :param conf: the plant_conf dict, or the flat configuration of a save request
    :type conf: dict
    :return: the validated inverter count (missing or null -> 1)
    :rtype: int
    """
    raw = conf.get("number_of_inverters", 1)
    if _is_null_like(raw) or (isinstance(raw, str) and not raw.strip()):
        return 1
    invalid = InverterConfigError(f"number_of_inverters must be a positive integer, got {raw!r}")
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        raise invalid
    try:
        value = float(raw)
    except (ValueError, OverflowError):  # OverflowError: an int too large for a float
        raise invalid from None
    if not value.is_integer() or value < 1:
        raise invalid
    return int(value)


def _inverter_label(name: str, index: int | None) -> str:
    return name if index is None else f"{name}[{index}]"


def _coerce_inverter_number(value, default: float | None, name: str, index: int | None = None):
    """Coerce one numeric per-inverter element (#1150).

    None or a stringly-typed "null" resolves to ``default``, a numeric string
    becomes a float, and an int or float passes through. A bool, a nested list,
    a non-numeric string or a non-finite number raises InverterConfigError naming
    the parameter and the slot. With no default, a null element is an error.
    """
    label = _inverter_label(name, index)
    if _is_null_like(value):
        if default is None:
            raise InverterConfigError(f"{label} has no value; a number is required")
        return default
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise InverterConfigError(f"{label}: non-numeric string element {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InverterConfigError(f"{label} must be a number, got {value!r}")
    try:
        finite = math.isfinite(value)
    except OverflowError:  # an int too large for a float
        finite = False
    if not finite:
        raise InverterConfigError(f"{label} must be a finite number, got {value!r}")
    return value


def _coerce_inverter_flag(value, name: str, index: int | None = None) -> bool:
    """Coerce one compute_curtailment element: a bool, None (false) or "true"/"false"."""
    if isinstance(value, bool):
        return value
    if _is_null_like(value):
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    raise InverterConfigError(
        f"{_inverter_label(name, index)} must be true or false, got {value!r}"
    )


def _coerce_inverter_element(value, default: bool | float, name: str, index: int | None = None):
    if isinstance(default, bool):
        return _coerce_inverter_flag(value, name, index)
    return _coerce_inverter_number(value, default, name, index)


def _inverter_index_is_unset(value) -> bool:
    """True for an index list that points at nothing: absent, null, empty or all-blank.

    The web UI saves an emptied array input as ``[null]``, so that counts as unset.
    """
    if not isinstance(value, list):
        return _is_null_like(value) or (isinstance(value, str) and not value.strip())
    return all(
        _is_null_like(element) or (isinstance(element, str) and not element.strip())
        for element in value
    )


def _inverter_index_values(conf: dict, key: str, num_inverters: int) -> list[int]:
    """Read and validate one per-inverter index list for number_of_inverters > 1 (#1150).

    Elements are 0-based inverter numbers: an int, an integral float or a numeric
    string (a bool is rejected), each in ``[0, num_inverters)``. Returns plain ints
    and raises InverterConfigError naming the key. A battery index must also have
    one entry per battery.
    """
    value = conf.get(key)
    owner = "battery" if key == "battery_inverter_index" else "PV plant"
    if _inverter_index_is_unset(value):
        raise InverterConfigError(
            f"{key} is required when number_of_inverters={num_inverters}: list the 0-based "
            f"inverter number of each {owner}"
        )
    if not isinstance(value, list):
        raise InverterConfigError(
            f"{key} must be a list with one 0-based inverter number per {owner}, got {value!r}"
        )
    numbers: list[int] = []
    for position, element in enumerate(value):
        label = f"{key}[{position}]"
        number = element
        if isinstance(element, str):
            try:
                number = float(element)
            except ValueError:
                raise InverterConfigError(
                    f"{label} must be an inverter number, got {element!r}"
                ) from None
        if isinstance(number, bool) or not isinstance(number, int | float):
            raise InverterConfigError(f"{label} must be an inverter number, got {element!r}")
        if isinstance(number, float) and not number.is_integer():
            raise InverterConfigError(f"{label} must be a whole number, got {element!r}")
        number = int(number)
        if not 0 <= number < num_inverters:
            raise InverterConfigError(
                f"{label}={number} is not an inverter: with number_of_inverters={num_inverters} "
                f"the numbers run from 0 to {num_inverters - 1}"
            )
        numbers.append(number)
    if key == "battery_inverter_index":
        try:
            num_batteries = validate_num_batteries(conf)
        except ValueError as err:
            raise InverterConfigError(str(err)) from err
        if len(numbers) != num_batteries:
            raise InverterConfigError(
                f"{key} has {_entry_count(len(numbers))} but number_of_batteries={num_batteries}: "
                "give the inverter of each battery"
            )
    return numbers


class _Broadcast(NamedTuple):
    """A scalar every inverter shares, repeated only once the configuration is usable (#1150)."""

    value: bool | float


def _join_inverter_faults(faults: Sequence[InverterConfigError]) -> InverterConfigError:
    """One error for a list of faults: the fault itself, or all of them joined with ". "."""
    if len(faults) == 1:
        return faults[0]
    return InverterConfigError(". ".join(str(fault) for fault in faults), problems=faults)


def check_inverter_params(conf: dict, logger: logging.Logger) -> None:
    """
    Validate and normalise the per-inverter parameters of a configuration (#1150).

    Works in place on the plant_conf and on the flat configuration of a save
    request, and is idempotent: a second call changes nothing. Every parameter is
    checked, so the user sees all faults at once: each unusable parameter is reported
    once (its first bad element), and when there are several they come in one
    InverterConfigError whose message joins them with ". " and whose ``problems``
    attribute lists them. A configuration with any fault is left exactly as it was.
    Nothing is refused for being above one inverter: a count above 1 is validated like any other.

    One inverter (the default, and a missing ``number_of_inverters``) is a true
    no-op for every existing configuration: a scalar is left exactly as written. The
    web UI saves array inputs as lists, so a one-element list collapses to its
    element (null and "null" become the default, a numeric string a float), any
    other length is an error, and an index list that points at nothing (absent,
    null, empty, all blank) is silent while a set one warns that it is ignored.

    Above one inverter:

    - ``inverter_efficiency_dc_ac``, ``inverter_efficiency_ac_dc``,
      ``inverter_stress_cost`` and ``compute_curtailment`` take a scalar (broadcast to
      every inverter) or a list of exactly ``number_of_inverters`` entries; a missing
      key broadcasts the default. Both efficiencies must be above 0 and at most 1, as a
      scalar or in every entry (the first bad entry is named).
    - ``inverter_ac_output_max`` and ``inverter_ac_input_max`` must be a list of
      exactly ``number_of_inverters`` finite entries of at least 0. A scalar is
      refused, since it is either the 5000 W default or a one-inverter site total.
    - ``battery_inverter_index`` (one entry per battery) and ``pv_plant_inverter_index``
      are required lists of inverter numbers, and every inverter must carry a battery
      or a PV plant. The PV list length depends on the forecast method and is checked
      when the forecast is built.
    - ``inverter_is_hybrid`` is ignored, with a warning, because each inverter's shape
      comes from what the two index lists attach to it.

    The work done is proportional to the size of the configuration, never to the value of
    ``number_of_inverters``: a scalar broadcast is written out only once there is no fault,
    when both caps are lists of exactly that length.

    :param conf: the plant_conf dict, or the flat configuration of a save request
    :type conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :raises InverterConfigError: when a setting is unusable
    """
    num_inverters = validate_num_inverters(conf)
    updates, removals, faults = _inverter_param_faults(conf, num_inverters, logger)
    if faults:
        raise _join_inverter_faults(faults)
    for name, value in updates.items():
        conf[name] = [value.value] * num_inverters if isinstance(value, _Broadcast) else value
    for name in removals:
        del conf[name]


def _inverter_param_faults(
    conf: dict, num_inverters: int, logger: logging.Logger
) -> tuple[dict, list[str], list[InverterConfigError]]:
    """Run every per-inverter check and collect what to change and what is wrong (#1150).

    Returns ``(updates, removals, faults)``: the keys to set, the keys to drop, and one
    fault per unusable parameter in check order (the broadcast parameters, the caps, the
    index lists, then the coverage of the inverters). ``conf`` is not touched. Checks that
    depend on another setting are skipped when that setting is unusable, so a bad
    ``number_of_batteries`` is reported once. ``num_inverters`` is the validated count.
    """
    updates: dict = {}
    removals: list[str] = []
    faults: list[InverterConfigError] = []
    if num_inverters == 1:
        _normalise_single_inverter(conf, updates, removals, faults, logger)
    else:
        _normalise_multiple_inverters(conf, num_inverters, updates, faults, logger)
    return updates, removals, faults


def _inverter_wrong_length(name: str, length: int, num_inverters: int) -> InverterConfigError:
    if num_inverters == 1:
        advice = "provide a single value or a list of exactly 1 entry"
    else:
        advice = (
            f"provide a scalar (broadcast to every inverter) or a list of exactly "
            f"{num_inverters} entries"
        )
    return InverterConfigError(
        f"{name} has {_entry_count(length)} but number_of_inverters={num_inverters}; {advice}"
    )


def _entry_count(length: int) -> str:
    return f"{length} entry" if length == 1 else f"{length} entries"


def _normalise_single_inverter(
    conf: dict,
    updates: dict,
    removals: list[str],
    faults: list[InverterConfigError],
    logger: logging.Logger,
) -> None:
    for name, default in INVERTER_BROADCAST_PARAMS.items():
        value = conf.get(name)
        if isinstance(value, list):
            try:
                if len(value) != 1:
                    raise _inverter_wrong_length(name, len(value), 1)
                updates[name] = _coerce_inverter_element(value[0], default, name, 0)
            except InverterConfigError as err:
                faults.append(err)
    for name in INVERTER_CAP_PARAMS:
        value = conf.get(name)
        if isinstance(value, list):
            try:
                if len(value) != 1:
                    raise _inverter_wrong_length(name, len(value), 1)
                if _is_null_like(value[0]):
                    # An emptied number input: the cap is simply not set, as for a null scalar.
                    removals.append(name)
                else:
                    updates[name] = _coerce_inverter_number(value[0], None, name, 0)
            except InverterConfigError as err:
                faults.append(err)
    for key in INVERTER_INDEX_PARAMS:
        if _inverter_index_is_unset(conf.get(key)):
            # What the web UI saves for an empty array input ([null]) points at nothing:
            # drop it, as for a null cap, so a one-inverter save gains no new keys.
            if key in conf:
                removals.append(key)
        else:
            logger.warning(
                f"{key} is ignored: number_of_inverters is 1, so there is only one "
                "inverter to point at."
            )


def _broadcast_param_update(
    conf: dict, name: str, default: bool | float, num_inverters: int
) -> list | _Broadcast:
    """The normalised value of one broadcast parameter above one inverter (#1150).

    An exact-length list is coerced element by element. A scalar (or a missing value) is
    coerced once and returned as a _Broadcast, so no per-inverter list exists until the
    whole configuration is known to be usable.
    """
    value = conf.get(name)
    if isinstance(value, list):
        if len(value) != num_inverters:
            raise _inverter_wrong_length(name, len(value), num_inverters)
        return [
            _coerce_inverter_element(element, default, name, position)
            for position, element in enumerate(value)
        ]
    return _Broadcast(_coerce_inverter_element(value, default, name))


def _cap_param_update(conf: dict, name: str, num_inverters: int) -> list:
    """The normalised value of one cap parameter above one inverter (#1150)."""
    value = conf.get(name)
    if not isinstance(value, list):
        found = "it is missing" if value is None else f"got {value!r}"
        raise InverterConfigError(
            f"{name} must be a list of {num_inverters} entries, one cap in watts per "
            f"inverter, when number_of_inverters={num_inverters}, but {found}. A single "
            "value is not broadcast: it is either the 5000 W default or a one-inverter "
            "site total, not a per-inverter cap"
        )
    if len(value) != num_inverters:
        raise InverterConfigError(
            f"{name} has {_entry_count(len(value))} but number_of_inverters={num_inverters}; "
            f"provide a list of exactly {num_inverters} caps, one per inverter"
        )
    caps = [
        _coerce_inverter_number(element, None, name, position)
        for position, element in enumerate(value)
    ]
    for position, cap in enumerate(caps):
        if cap < 0:
            raise InverterConfigError(f"{name}[{position}]={cap} must not be negative")
    return caps


# How many inverters with nothing attached are named before the rest are only counted.
_MAX_UNATTACHED_LISTED = 10


def _inverter_coverage_fault(
    indexes: dict[str, list[int]], num_inverters: int
) -> InverterConfigError | None:
    """The fault for inverters that no battery and no PV plant is attached to (#1150).

    Walks the inverter numbers only until ``_MAX_UNATTACHED_LISTED`` unattached ones are
    found, so the work and the message stay bounded however large ``number_of_inverters``
    is. The rest are counted from the total, which holds because every number in the
    index lists was validated to be in range.
    """
    attached = set().union(*indexes.values())
    unattached: list[int] = []
    for number in range(num_inverters):
        if number not in attached:
            unattached.append(number)
            if len(unattached) == _MAX_UNATTACHED_LISTED:
                break
    if not unattached:
        return None
    sentences = [
        f"inverter {number} has no battery and no PV plant: add {number} to "
        "battery_inverter_index or pv_plant_inverter_index"
        for number in unattached
    ]
    more = num_inverters - len(attached) - len(unattached)
    if more:
        verb = "inverters have" if more > 1 else "inverter has"
        sentences.append(f"and {more} more {verb} no battery and no PV plant")
    return InverterConfigError("; ".join(sentences))


_INVERTER_EFFICIENCY_PARAMS: tuple[str, ...] = (
    "inverter_efficiency_dc_ac",
    "inverter_efficiency_ac_dc",
)


def _efficiency_range_fault(name: str, update: list | _Broadcast) -> InverterConfigError | None:
    """The fault for an efficiency outside ``0 < eff <= 1`` above one inverter (#1150).

    Names the first bad slot only, so the message stays short for any inverter count.
    """
    if isinstance(update, _Broadcast):
        values: list = [update.value]
    else:
        values = update
    for position, value in enumerate(values):
        if not 0 < value <= 1:
            index = None if isinstance(update, _Broadcast) else position
            return InverterConfigError(
                f"{_inverter_label(name, index)}={value} must be above 0 and at most 1"
            )
    return None


def _normalise_multiple_inverters(
    conf: dict,
    num_inverters: int,
    updates: dict,
    faults: list[InverterConfigError],
    logger: logging.Logger,
) -> None:
    for name, default in INVERTER_BROADCAST_PARAMS.items():
        try:
            update = _broadcast_param_update(conf, name, default, num_inverters)
        except InverterConfigError as err:
            faults.append(err)
            continue
        if name in _INVERTER_EFFICIENCY_PARAMS:
            range_fault = _efficiency_range_fault(name, update)
            if range_fault:
                faults.append(range_fault)
                continue
        updates[name] = update
    for name in INVERTER_CAP_PARAMS:
        try:
            updates[name] = _cap_param_update(conf, name, num_inverters)
        except InverterConfigError as err:
            faults.append(err)
    indexes: dict[str, list[int]] = {}
    for key in INVERTER_INDEX_PARAMS:
        try:
            indexes[key] = _inverter_index_values(conf, key, num_inverters)
        except InverterConfigError as err:
            faults.append(err)
    if len(indexes) == len(INVERTER_INDEX_PARAMS):
        # Only when both lists are usable: otherwise "no battery" is a guess.
        coverage_fault = _inverter_coverage_fault(indexes, num_inverters)
        if coverage_fault:
            faults.append(coverage_fault)
        updates.update(indexes)
    if conf.get("inverter_is_hybrid"):
        logger.warning(
            "inverter_is_hybrid is ignored above one inverter: each inverter's shape comes "
            "from what battery_inverter_index and pv_plant_inverter_index attach to it."
        )


# Who calls what (#1150): build_params logs every problem and carries on, so the
# configuration page and startup stay up; treat_runtimeparams raises through
# require_usable_inverter_config, which logs first; /set-config collects the messages on
# a copy with inverter_config_faults. All three share _inverter_config_problems, which
# runs validate_num_inverters and check_inverter_params.
def _inverter_config_problems(conf: dict, logger: logging.Logger) -> list[InverterConfigError]:
    """Everything wrong with the inverter settings of a configuration (#1150), in order.

    An unusable count is the only problem reported for it. Otherwise every fault from
    check_inverter_params is reported, one per unusable parameter. A count above 1 is
    not a fault in itself.
    """
    try:
        validate_num_inverters(conf)
    except InverterConfigError as err:
        return [err]
    problems: list[InverterConfigError] = []
    try:
        check_inverter_params(conf, logger)
    except InverterConfigError as err:
        problems.extend(err.problems)
    return problems


def require_usable_inverter_config(conf: dict, logger: logging.Logger) -> None:
    """Validate the inverter parameters, raising when one is unusable (#1150).

    Every problem is logged at error level here before the error is raised, which is what
    lets the web server return the action log with its 400, as for SocFractionError. When
    there are several, the raised error carries all of them, joined with ". ".

    :param conf: the plant_conf dict
    :type conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :raises InverterConfigError: when the count or a per-inverter setting is unusable
    """
    problems = _inverter_config_problems(conf, logger)
    for problem in problems:
        logger.error("%s", problem)
    if len(problems) == 1:
        # Raised as is: "from problems[0]" would make the error its own cause.
        raise problems[0]
    if problems:
        raise _join_inverter_faults(problems) from problems[0]


# Receives the warnings of the trial run in inverter_config_faults: a save request is
# checked on a copy and its warnings are logged by the build that follows. Configured
# once here, since getLogger returns the same logger on every call.
_QUIET_INVERTER_LOGGER = logging.getLogger(f"{__name__}.inverter_config_faults")
_QUIET_INVERTER_LOGGER.addHandler(logging.NullHandler())
_QUIET_INVERTER_LOGGER.propagate = False


def inverter_config_faults(config: dict) -> list[str]:
    """
    Return the reasons a save request's inverter settings would be refused (#1150).

    Used by ``/set-config`` before anything is written. Runs the same checks as
    require_usable_inverter_config on a copy, so the request is left alone and logs
    nothing. The list holds every fault, one per unusable parameter (its first bad
    entry), in check order. A count above 1 is accepted when the settings are usable.

    :param config: the submitted flat configuration
    :type config: dict
    :return: the messages (empty when the settings are usable)
    :rtype: list[str]
    """
    problems = _inverter_config_problems(copy.deepcopy(config), _QUIET_INVERTER_LOGGER)
    return [str(problem) for problem in problems]


def _coerce_batt_weight_value(
    value: bool | float | str | list | None,
    default: bool | float,
    parameter_name: str,
) -> bool | int | float | list:
    """Recursively coerce a weight_battery_* value/element (#610).

    Applies the same leaf-level rule as _coerce_batt_element (None/"null" ->
    default, numeric string -> float, non-numeric string -> clear ValueError)
    at ANY nesting depth: a bare scalar, a flat time series, or a nested
    per-battery list-of-series. Only leaf elements are coerced - the outer
    SHAPE is always preserved (a list stays a list of the same length/nesting,
    a scalar stays a scalar), so this is safe to run unconditionally,
    including at num_batteries == 1 where check_batt_weight_params must
    otherwise stay a byte-identical no-op.
    """
    if isinstance(value, list):
        return [_coerce_batt_weight_value(v, default, parameter_name) for v in value]
    return _coerce_batt_element(value, default, parameter_name)


def check_batt_weight_params(
    num_batteries: int,
    parameter: dict,
    parameter_name: str,
    logger: logging.Logger,
    default: bool | float = 0.0,
) -> None:
    """
    Normalise weight_battery_charge/weight_battery_discharge into the nested
    per-battery form for N > 1 (#610). Mutates parameter[parameter_name] in
    place; returns nothing (mirrors the value being read back via ``parameter``).

    A stringly-typed guard runs FIRST, at every num_batteries (including 1):
    every leaf element is coerced via _coerce_batt_weight_value (None/"null"
    -> default, numeric string -> float, non-numeric string -> clear
    ValueError naming the parameter). Before #610 these two params had zero
    validation of any kind at N=1 (the default and only value every
    non-adopting user has), unlike every other per-battery array param. Only
    leaf VALUES are coerced; the shape is untouched, so the N=1 no-op still
    holds for any already-well-typed value.

    Disambiguation, in priority order:
    1. num_batteries == 1: no-op (beyond the stringly-typed guard above),
       exactly today's behaviour (scalar or series, no nesting) -
       optimization.py's existing np.array(...) broadcast handles both shapes
       unchanged.
    2. Scalar value: broadcast, every battery gets the same scalar -> [v] * N.
    3. List of length num_batteries whose elements are themselves scalars or
       lists: already per-battery (entry k -> battery k); passed through as-is.
    4. Any other list (a flat time series not of length N): shared by every
       battery, today's semantics preserved -> [v] * N (each battery gets an
       identical copy of the series).
    5. Ambiguous corner: a flat NUMERIC list whose length happens to equal
       num_batteries is caught by rule 3 (per-battery), not rule 4 (shared
       series). Documented in docs/config.md: nest explicitly ([series] * N,
       or distinct per-battery entries) if a shared series of that exact
       length was intended.

    :param num_batteries: plant_conf["number_of_batteries"]
    :type num_batteries: int
    :param parameter: the optim_conf dict containing parameter_name
    :type parameter: dict
    :param parameter_name: "weight_battery_charge" or "weight_battery_discharge"
    :type parameter_name: str
    :param logger: The logger object
    :type logger: logging.Logger
    :param default: fallback value for a None/"null" leaf (config_defaults.json
        value for both weight params is 0.0)
    :type default: bool | int | float
    """
    current = parameter.get(parameter_name, None)
    if current is None:
        return
    current = _coerce_batt_weight_value(current, default, parameter_name)
    parameter[parameter_name] = current
    if num_batteries == 1:
        return
    if not isinstance(current, list):
        parameter[parameter_name] = [current] * num_batteries
        return
    if len(current) == num_batteries and all(
        isinstance(v, list) or (isinstance(v, int | float) and not isinstance(v, bool))
        for v in current
    ):
        # Already per-battery nested (rule 3, including the ambiguous rule-5
        # corner), pass through unchanged.
        return
    # Flat list not of length num_batteries -> shared time series (rule 4). Each
    # battery gets its OWN copy: [current] * N would alias one list object
    # across every battery, so a later in-place mutation of one battery's
    # series would silently rewrite them all.
    parameter[parameter_name] = [list(current) for _ in range(num_batteries)]


def _charge_derating_fault(table: list[list[float]]) -> str | None:
    """Name the first fault in one battery_charge_power_derating table (#807).

    Returns the offending row and value, or None when the table is usable.
    """
    previous_soc = 0.0
    previous_max = 1.0
    for position, row in enumerate(table, start=1):
        if not isinstance(row, list | tuple) or len(row) != 2:
            return f"row {position} is {row!r}, expected a [soc_threshold, power_max] pair"
        soc_threshold, power_max = row
        for name, value in (("soc_threshold", soc_threshold), ("power_max", power_max)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                return f"row {position} has {name}={value!r}, expected a number"
            # Explicit, so the guard survives a rewrite of the range check below:
            # nan is caught there only because every comparison against it is
            # false, which a reordering into `value < 0 or value > 1` would undo.
            if not math.isfinite(value):
                return f"row {position} has {name}={value!r}, expected a finite number"
            if not 0 <= value <= 1:
                return (
                    f"row {position} has {name}={value}, expected a value between 0 and 1 "
                    f"(percentage/100, so 84% is 0.84)"
                )
        if soc_threshold <= previous_soc:
            return (
                f"row {position} has soc_threshold={soc_threshold}, which does not exceed "
                f"the {previous_soc} before it: rows must ascend by state of charge"
            )
        if power_max > previous_max:
            return (
                f"row {position} has power_max={power_max}, above the {previous_max} that "
                f"applies below it: the charge limit cannot rise as the battery fills"
            )
        previous_soc, previous_max = soc_threshold, power_max
    return None


# Hard cap on the points of one inverter curve. The MILP carries P-1 timestep-sized
# binaries per curved direction, so the solve time grows with the point count; a
# datasheet efficiency table has 5-12 points, so this leaves ample headroom while
# stopping an oversized (accidental or hostile) curve before any variable exists.
INVERTER_CURVE_MAX_POINTS = 20


def _finite_curve_number(value, parameter_name: str, position: int, name: str) -> float:
    """Return value as a finite float, or raise ValueError (never OverflowError)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"{parameter_name}: point {position} has {name}={value!r}, expected a number"
        )
    try:
        number = float(value)
    except OverflowError:  # an int too large for a float
        number = math.inf
    if not math.isfinite(number):
        raise ValueError(
            f"{parameter_name}: point {position} has {name}={value!r}, expected a finite number"
        )
    return number


def inverter_power_curve_is_unset(curve) -> bool:
    """True only for the explicit empty list, the default that keeps the scalar efficiency.

    Every other value (None, False, 0, "", () ...) is a supplied curve and must be
    validated: treating falsey values as "unset" would silently swap the physical model.
    """
    return isinstance(curve, list) and not curve


def validate_inverter_power_curve(
    curve: list, parameter_name: str, direction: str
) -> list[tuple[float, float]]:
    """
    Validate one configured inverter efficiency curve and return internal
    (ac_w, dc_w) transfer points (#746).

    Public configuration uses [ac_power_w, efficiency] points, indexed on the AC
    side, which is where inverters and AC-coupled batteries are commanded and
    measured. Efficiency is a fraction, so 97% is 0.97. The zero-power transfer
    origin is inserted internally; users do not provide an efficiency at 0 W.

    - ``dc_ac`` (discharge): ``[ac_output_power_w, efficiency]``, with
      ``dc_input = ac_output / efficiency``. Delivering positive AC power needs
      DC power, so efficiency must be > 0.
    - ``ac_dc`` (charge): ``[ac_input_power_w, efficiency]``, with
      ``dc_output = ac_input * efficiency``. Fixed losses can consume all of a
      small AC input, so efficiency may be 0 (for example 50 W AC in -> 0 W DC out).

    The DC-side transfer must be non-decreasing. Flat segments are valid.

    :param curve: configured [ac_power_w, efficiency] points
    :type curve: list
    :param parameter_name: name used in error messages
    :type parameter_name: str
    :param direction: dc_ac or ac_dc
    :type direction: str
    :raises ValueError: when the curve is unusable
    :return: internal transfer points as (ac_w, dc_w) tuples
    :rtype: list[tuple[float, float]]
    """
    ac_name = "ac_output_power_w" if direction == "dc_ac" else "ac_input_power_w"
    pair = f"[{ac_name}, efficiency]"
    if not isinstance(curve, list | tuple) or len(curve) < 2:
        raise ValueError(
            f"{parameter_name}: must be a list of at least 2 {pair} points, got {curve!r}"
        )
    if len(curve) > INVERTER_CURVE_MAX_POINTS:
        raise ValueError(
            f"{parameter_name}: has {len(curve)} points, at most {INVERTER_CURVE_MAX_POINTS} "
            f"are supported (each adds binary variables to every time step)"
        )

    configured: list[tuple[float, float]] = []
    for position, row in enumerate(curve, start=1):
        if not isinstance(row, list | tuple) or len(row) != 2:
            raise ValueError(
                f"{parameter_name}: point {position} is {row!r}, expected a {pair} pair"
            )
        power = _finite_curve_number(row[0], parameter_name, position, ac_name)
        efficiency = _finite_curve_number(row[1], parameter_name, position, "efficiency")
        if power <= 0:
            raise ValueError(
                f"{parameter_name}: point {position} has {ac_name}={power}, expected a strictly "
                f"positive AC-side power in watts; the zero-power origin is added internally"
            )
        if efficiency < 0 or efficiency > 1:
            raise ValueError(
                f"{parameter_name}: point {position} has efficiency={efficiency}, expected "
                f"0 <= efficiency <= 1 (percentage/100; use 0.97 for 97%)"
            )
        if direction == "dc_ac" and efficiency == 0:
            raise ValueError(
                f"{parameter_name}: point {position} has efficiency=0 at positive {ac_name}; "
                f"DC-to-AC delivers AC output, so zero efficiency would require infinite DC "
                f"input and cannot be represented by this curve"
            )
        configured.append((power, efficiency))

    for position in range(2, len(configured) + 1):
        previous_power = configured[position - 2][0]
        power = configured[position - 1][0]
        if power <= previous_power:
            raise ValueError(
                f"{parameter_name}: point {position} has {ac_name}={power}, which does not "
                f"exceed the {previous_power} before it: points must ascend strictly by {ac_name}"
            )

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for power, efficiency in configured:
        dc_power = power / efficiency if direction == "dc_ac" else power * efficiency
        if not math.isfinite(dc_power):
            raise ValueError(
                f"{parameter_name}: {ac_name}={power} with efficiency={efficiency} converts to a "
                f"non-finite DC-side power; use a larger efficiency"
            )
        points.append((power, dc_power))

    for position in range(2, len(points)):
        dc_prev = points[position - 1][1]
        dc_now = points[position][1]
        if dc_now < dc_prev - 1e-9 * max(1.0, dc_prev):
            raise ValueError(
                f"{parameter_name}: point {position} converts to DC-side power {dc_now}, which "
                f"is below {dc_prev} from the previous point: the DC-side transfer power must "
                f"be non-decreasing"
            )
    for (ac_a, dc_a), (ac_b, dc_b) in itertools.pairwise(points):
        if not math.isfinite((dc_b - dc_a) / (ac_b - ac_a)):
            raise ValueError(
                f"{parameter_name}: points around {ac_name}={ac_b} are too close together "
                f"to form a finite segment slope"
            )
    return points


_INVERTER_CURVE_PARAMS = (
    ("inverter_power_curve_dc_ac", "dc_ac", "inverter_ac_output_max", "inverter_efficiency_dc_ac"),
    ("inverter_power_curve_ac_dc", "ac_dc", "inverter_ac_input_max", "inverter_efficiency_ac_dc"),
)


def _inverter_curve_shape(curve) -> str:
    """
    Classify an inverter power curve value by shape (#1150).

    - ``unset``: the explicit empty list, the default that keeps the scalar efficiency.
    - ``single``: one curve, a list of ``[ac_power_w, efficiency]`` points. Its first
      point starts with a number.
    - ``per_inverter``: a list with one entry per inverter, each entry one curve or
      ``[]``. Its first entry is ``[]`` or starts with a point.
    - ``malformed``: anything else. The single-curve validator reports why.

    Shape, not length, tells a per-inverter list from a curve: two inverters sharing a
    two-point curve would otherwise be split into one point each (the same trap as
    battery_charge_power_derating).
    """
    if inverter_power_curve_is_unset(curve):
        return "unset"
    if not isinstance(curve, list) or not isinstance(curve[0], list | tuple):
        return "malformed"
    first = curve[0]
    if not first or isinstance(first[0], list | tuple):
        return "per_inverter"
    if isinstance(first[0], int | float) and not isinstance(first[0], bool):
        return "single"
    return "malformed"


def _inverter_curve_shape_fault(
    parameter_name: str, shape: str, curve, num_inverters: int
) -> str | None:
    """The reason a well-formed curve value does not fit number_of_inverters, or None."""
    if shape == "per_inverter" and len(curve) != num_inverters:
        if num_inverters == 1:
            advice = "give one curve, or [] to keep the scalar efficiency"
        else:
            advice = f"give a list of exactly {num_inverters} entries, each a curve or []"
        return (
            f"{parameter_name} has {len(curve)} {'curve' if len(curve) == 1 else 'curves'} but "
            f"number_of_inverters={num_inverters}; {advice}"
        )
    if shape == "single" and num_inverters > 1:
        return (
            f"{parameter_name} is a single curve, which is not broadcast above one inverter: "
            f"its points are absolute AC watts for one inverter. With "
            f"number_of_inverters={num_inverters}, give a list of exactly {num_inverters} "
            f"entries, each a curve or []"
        )
    return None


def inverter_power_curve_faults(config: dict) -> list[str]:
    """
    Return one message per unusable inverter power curve in an explicit save request (#746).

    Used by ``/set-config`` before anything is written, so a curve the user typed is
    rejected with its reason instead of being silently dropped. An absent parameter or
    an explicit empty list is fine (scalar efficiency); any other value is validated.
    The value is one curve, or a list with one curve (or ``[]``) per inverter (#1150):
    the list length must equal ``number_of_inverters`` and a single curve is refused
    above one inverter. The hybrid flag is not checked, since a valid curve may be kept
    while ``inverter_is_hybrid`` is off (see :func:`check_inverter_power_curves`).

    :param config: the submitted flat configuration
    :type config: dict
    :return: validation messages (empty when every curve is usable)
    :rtype: list[str]
    """
    try:
        num_inverters = validate_num_inverters(config)
    except InverterConfigError:
        # Reported by inverter_config_faults; each curve is still checked on its own.
        num_inverters = None
    faults = []
    for parameter_name, direction, *_ in _INVERTER_CURVE_PARAMS:
        curve = config.get(parameter_name, [])
        shape = _inverter_curve_shape(curve)
        if shape == "unset":
            continue
        if shape == "malformed" or num_inverters is None:
            fault = None
        else:
            fault = _inverter_curve_shape_fault(parameter_name, shape, curve, num_inverters)
        if fault:
            faults.append(fault)
            continue
        if shape == "per_inverter":
            for position, entry in enumerate(curve):
                if inverter_power_curve_is_unset(entry):
                    continue
                label = parameter_name if num_inverters == 1 else f"{parameter_name}[{position}]"
                try:
                    validate_inverter_power_curve(entry, label, direction)
                except ValueError as err:
                    faults.append(str(err))
            continue
        try:
            validate_inverter_power_curve(curve, parameter_name, direction)
        except ValueError as err:
            faults.append(str(err))
    return faults


def check_inverter_power_curves(
    plant_conf: dict,
    logger: logging.Logger,
    *,
    recover: bool = False,
    supplied: tuple[str, ...] | list[str] = (),
) -> None:
    """
    Validate the optional inverter power-dependent efficiency curves of a plant_conf (#746).

    An absent parameter or an explicit empty list is the default and leaves the scalar
    ``inverter_efficiency_dc_ac`` / ``inverter_efficiency_ac_dc`` path untouched.
    Any other value is a configured [ac_power_w, efficiency] curve: it is validated and
    converted to internal transfer points (see :func:`validate_inverter_power_curve`).

    Two contracts, chosen by the caller:

    - strict (default, runtime parameters): a fault is logged at error level and raises
      InverterConfigError (a ValueError), because a request that carries a curve expects
      it to be used and silently falling back to the scalar would change the physics.
      The web server turns it into a 400 that carries the logged reason.
    - ``recover=True`` (persisted configuration in ``build_params``): the unusable
      curve is cleared and logged at error level, so the scalar efficiency of that
      direction applies and the configuration page and service stay reachable to
      repair it. Same trade-off as ``battery_charge_power_derating``.

    A valid curve with ``inverter_is_hybrid`` false is dormant: it is kept, ignored
    by the optimizer, and warned about, so switching hybrid off in the UI never
    discards or rejects a stored curve. A curve named in ``supplied`` (set by the
    current request) with hybrid off is a fault instead, as it would be ignored
    without the requester knowing.

    The value is one curve, or (#1150) a list with one curve or ``[]`` per inverter,
    told apart by shape (see :func:`_inverter_curve_shape`). With one inverter a
    one-entry list unwraps to its curve and everything else behaves as above. Above
    one inverter a single curve is refused, the list length must equal
    ``number_of_inverters``, and ``inverter_is_hybrid`` is not consulted: each entry is
    validated on its own, and the below-the-cap warning compares it with that
    inverter's cap. An AC-to-DC entry for an inverter that has no battery in
    ``battery_inverter_index`` gets the hybrid-off treatment (kept and warned about
    when saved, a fault when supplied). In recover mode a faulty entry becomes
    ``[]``, and a shape fault clears the whole parameter.

    :param plant_conf: the plant_conf dict
    :type plant_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :param recover: clear and log an unusable curve instead of raising
    :type recover: bool
    :param supplied: curve parameter names provided by the current request
    :type supplied: tuple[str, ...] | list[str]
    :raises InverterConfigError: when a curve is unusable and ``recover`` is False (a
        ValueError, so existing handlers still catch it); the curve's reason is logged at
        error level first, so the action log returned with the 400 names it. A count or
        index fault is raised without a log line here: treat_runtimeparams has already
        logged it through require_usable_inverter_config, and logging it again would
        duplicate the line
    """
    if recover:
        _check_inverter_power_curves(plant_conf, logger, recover=True, supplied=supplied)
        return
    try:
        _check_inverter_power_curves(plant_conf, logger, recover=False, supplied=supplied)
    except InverterConfigError:
        # A count or index fault: already logged by require_usable_inverter_config.
        raise
    except ValueError as err:
        # What is left are the curve checks' own faults (the curve validator, the shape
        # and topology faults), which nothing has logged yet.
        logger.error("%s", err)
        raise InverterConfigError(str(err)) from err


def _check_inverter_power_curves(
    plant_conf: dict,
    logger: logging.Logger,
    *,
    recover: bool,
    supplied: tuple[str, ...] | list[str],
) -> None:
    """The checks of :func:`check_inverter_power_curves`; strict faults raise plain ValueError."""
    try:
        num_inverters = validate_num_inverters(plant_conf)
    except InverterConfigError:
        if not recover:
            raise
        # A saved configuration with an unusable count is reported by
        # check_inverter_params; its curves wait for a usable count instead of
        # being cleared for an unrelated fault.
        return
    if num_inverters > 1:
        _check_per_inverter_curves(
            plant_conf, logger, num_inverters, recover=recover, supplied=supplied
        )
        return
    hybrid = plant_conf.get("inverter_is_hybrid", False)
    for parameter_name, direction, limit_name, scalar_name in _INVERTER_CURVE_PARAMS:
        curve = plant_conf.get(parameter_name, [])
        try:
            if _inverter_curve_shape(curve) == "per_inverter":
                fault = _inverter_curve_shape_fault(parameter_name, "per_inverter", curve, 1)
                if fault:
                    raise ValueError(fault)
                curve = plant_conf[parameter_name] = curve[0]
            if inverter_power_curve_is_unset(curve):
                continue
            points = validate_inverter_power_curve(curve, parameter_name, direction)
            if not hybrid and parameter_name in supplied:
                raise ValueError(
                    f"{parameter_name}: requires inverter_is_hybrid=true; the curves model the "
                    f"hybrid inverter's DC bus <-> AC conversion"
                )
        except ValueError as err:
            if not recover:
                raise
            # At error level: a dropped curve is otherwise indistinguishable from a
            # working one, as the scalar efficiency applies either way.
            logger.error(_curve_recovery_message(err, parameter_name, plant_conf, scalar_name))
            plant_conf[parameter_name] = []
            continue
        if not hybrid:
            logger.warning(
                f"{parameter_name} is set but inverter_is_hybrid is false: the curve is "
                f"kept and ignored until inverter_is_hybrid is enabled."
            )
            continue
        limit = plant_conf.get(limit_name)
        if isinstance(limit, int | float) and points[-1][0] < limit:
            logger.warning(_curve_below_cap_message(parameter_name, points, limit_name, limit))


def _curve_recovery_message(
    problem, label: str, plant_conf: dict, scalar_name: str, applies_to: str = ""
) -> str:
    """The error logged when a saved curve is cleared and the scalar efficiency applies."""
    return (
        f"{problem}. Ignoring {label}: the scalar {scalar_name}="
        f"{plant_conf.get(scalar_name, 1.0)} applies {applies_to}until the curve is corrected."
    )


def _curve_below_cap_message(label: str, points: list, limit_name: str, limit) -> str:
    """The warning for a curve whose last point is below the AC cap it sits under."""
    return (
        f"{label}: the curve ends at {points[-1][0]} W on the AC side, below "
        f"{limit_name}={limit} W. The curve's last point is the supported power "
        f"domain, so the optimizer will not plan beyond it."
    )


def _check_inverter_curve_entry(
    plant_conf: dict,
    logger: logging.Logger,
    entry: list,
    position: int,
    *,
    parameter_name: str,
    direction: str,
    limit_name: str,
    scalar_name: str,
    has_battery: bool | None,
    recover: bool,
    supplied: tuple[str, ...] | list[str],
) -> list:
    """Validate one entry of a per-inverter curve list and warn about it (#1150).

    Returns the entry to keep: the entry itself, or ``[]`` when it is unusable and
    ``recover`` is set (an unusable entry raises ValueError otherwise). ``has_battery``
    is None when the battery index is not usable, so no topology is applied.
    """
    if inverter_power_curve_is_unset(entry):
        return entry
    label = f"{parameter_name}[{position}]"
    without_battery = direction == "ac_dc" and has_battery is False
    try:
        points = validate_inverter_power_curve(entry, label, direction)
        if without_battery and parameter_name in supplied:
            raise ValueError(
                f"{label}: inverter {position} has no battery in "
                f"battery_inverter_index, and an AC-to-DC curve models charging a battery"
            )
    except ValueError as err:
        if not recover:
            raise
        logger.error(
            _curve_recovery_message(err, label, plant_conf, scalar_name, f"to inverter {position} ")
        )
        return []
    if without_battery:
        logger.warning(
            f"{label} is set but inverter {position} has no battery in "
            f"battery_inverter_index: the curve is kept and ignored until a battery "
            f"is attached to that inverter."
        )
        return entry
    limits = plant_conf.get(limit_name)
    limit = limits[position] if isinstance(limits, list) and position < len(limits) else None
    if isinstance(limit, int | float) and not isinstance(limit, bool) and points[-1][0] < limit:
        logger.warning(_curve_below_cap_message(label, points, f"{limit_name}[{position}]", limit))
    return entry


def _check_per_inverter_curves(
    plant_conf: dict,
    logger: logging.Logger,
    num_inverters: int,
    *,
    recover: bool,
    supplied: tuple[str, ...] | list[str],
) -> None:
    """The number_of_inverters > 1 half of :func:`check_inverter_power_curves` (#1150)."""
    try:
        battery_index = _inverter_index_values(plant_conf, "battery_inverter_index", num_inverters)
    except InverterConfigError:
        # Topology is only used when the index list is usable; its fault is reported
        # by check_inverter_params.
        battery_index = None
    for parameter_name, direction, limit_name, scalar_name in _INVERTER_CURVE_PARAMS:
        curve = plant_conf.get(parameter_name, [])
        shape = _inverter_curve_shape(curve)
        if shape == "unset":
            continue
        if shape == "malformed":
            try:
                validate_inverter_power_curve(curve, parameter_name, direction)
                problem = f"{parameter_name} is neither a curve nor a list of curves: {curve!r}"
            except ValueError as err:
                problem = str(err)
        else:
            problem = _inverter_curve_shape_fault(parameter_name, shape, curve, num_inverters)
        if problem:
            if not recover:
                raise ValueError(problem)
            logger.error(
                _curve_recovery_message(
                    problem, parameter_name, plant_conf, scalar_name, "to every inverter "
                )
            )
            plant_conf[parameter_name] = []
            continue
        plant_conf[parameter_name] = [
            _check_inverter_curve_entry(
                plant_conf,
                logger,
                entry,
                position,
                parameter_name=parameter_name,
                direction=direction,
                limit_name=limit_name,
                scalar_name=scalar_name,
                has_battery=None if battery_index is None else position in battery_index,
                recover=recover,
                supplied=supplied,
            )
            for position, entry in enumerate(curve)
        ]


def check_batt_charge_derating(
    num_batteries: int,
    parameter: dict,
    parameter_name: str,
    logger: logging.Logger,
) -> None:
    """
    Normalise battery_charge_power_derating into the nested per-battery form for
    N > 1 (#807). Mutates parameter[parameter_name] in place; returns nothing.

    The value is a table: [soc_threshold, power_max] pairs, both percentage/100,
    ascending by SOC. Above soc_threshold the charge power is capped at that much
    of the maximum; below the first threshold the flat maximum applies.

    Shared and per-battery are told apart by shape, not by length: two batteries
    sharing a two-row table would be split into one row each. A shared table's
    first element is a pair of numbers, a per-battery value's is a table.

    1. Missing, None or empty: left alone; the flat battery_charge_power_max
       applies, i.e. the behaviour before this parameter.
    2. First element is a table: already per-battery, and its length must equal
       num_batteries (never padded or truncated).
    3. First element is a pair: one shared table. At num_batteries == 1 it is
       left as written; above that every battery gets its own copy, so an
       in-place edit of one cannot rewrite the others.
    4. Anything else, content included: logged with the offending row and value,
       then cleared.

    :param num_batteries: plant_conf["number_of_batteries"]
    :type num_batteries: int
    :param parameter: the plant_conf dict containing parameter_name
    :type parameter: dict
    :param parameter_name: "battery_charge_power_derating"
    :type parameter_name: str
    :param logger: The logger object
    :type logger: logging.Logger
    """
    current = parameter.get(parameter_name, None)
    if not current:
        return

    # Cleared and logged rather than raised: this table refines a ceiling that
    # already exists, so dropping it is a valid way to run. At error level
    # because a dropped table is otherwise indistinguishable from a working one -
    # the flat battery_charge_power_max still applies either way.
    def _reject(problem: str) -> None:
        logger.error(
            f"{parameter_name}: {problem}. Ignoring it: charging is limited by the "
            f"flat battery_charge_power_max instead."
        )
        parameter[parameter_name] = []

    if not isinstance(current, list) or not isinstance(current[0], list | tuple) or not current[0]:
        _reject(f"must be a list of [soc_threshold, power_max] pairs, got {current!r}")
        return
    if isinstance(current[0][0], list | tuple):
        if len(current) != num_batteries:
            _reject(f"has {len(current)} tables but number_of_batteries={num_batteries}")
            return
        for table in current:
            fault = _charge_derating_fault(table)
            if fault:
                _reject(fault)
                return
        return
    fault = _charge_derating_fault(current)
    if fault:
        _reject(fault)
        return
    if num_batteries == 1:
        return
    parameter[parameter_name] = [[list(row) for row in current] for _ in range(num_batteries)]


def get_days_list(days_to_retrieve: int) -> pd.DatetimeIndex:
    """
    Get list of past days from today to days_to_retrieve.

    :param days_to_retrieve: Total number of days to retrieve from the past
    :type days_to_retrieve: int
    :return: The list of days
    :rtype: pd.DatetimeIndex

    """
    today = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    d = (today - timedelta(days=days_to_retrieve)).isoformat()
    days_list = pd.date_range(start=d, end=today.isoformat(), freq="D").normalize()
    return days_list


def add_date_features(
    data: pd.DataFrame,
    timestamp: str | None = None,
    date_features: list[str] | None = None,
) -> pd.DataFrame:
    """Add date-related features from a DateTimeIndex or a timestamp column.

    :param data: The input DataFrame.
    :type data: pd.DataFrame
    :param timestamp: The column containing the timestamp (optional if DataFrame has a DateTimeIndex).
    :type timestamp: Optional[str]
    :param date_features: List of date features to extract (default: all).
    :type date_features: Optional[List[str]]
    :return: The DataFrame with added date features.
    :rtype: pd.DataFrame
    """

    df = copy.deepcopy(data)  # Avoid modifying the original DataFrame

    # If no specific features are requested, extract all by default
    default_features = ["year", "month", "day_of_week", "day_of_year", "day", "hour"]
    date_features = date_features or default_features

    # Determine whether to use index or a timestamp column
    if timestamp:
        df[timestamp] = pd.to_datetime(df[timestamp], utc=True)
        source = df[timestamp].dt
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DateTimeIndex or a valid timestamp column.")
        source = df.index

    # Extract date features
    if "year" in date_features:
        df["year"] = source.year
    if "month" in date_features:
        df["month"] = source.month
    if "day_of_week" in date_features:
        df["day_of_week"] = source.dayofweek
    if "day_of_year" in date_features:
        df["day_of_year"] = source.dayofyear
    if "day" in date_features:
        df["day"] = source.day
    if "hour" in date_features:
        df["hour"] = source.hour

    return df


def set_df_index_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the freq of a DataFrame DateTimeIndex.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Input DataFrame with freq defined
    :rtype: pd.DataFrame

    """
    idx_diff = np.diff(df.index)
    # Sometimes there are zero values in this list.
    idx_diff = idx_diff[np.nonzero(idx_diff)]
    sampling = pd.to_timedelta(np.median(idx_diff))
    df = df[~df.index.duplicated()]
    return df.asfreq(sampling)


def parse_export_time_range(
    start_time: str,
    end_time: str | None,
    time_zone: pd.Timestamp.tz,
    logger: logging.Logger,
) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[bool, bool]:
    """
    Parse and validate start_time and end_time for export operations.

    :param start_time: Start time string in ISO format
    :type start_time: str
    :param end_time: End time string in ISO format (optional)
    :type end_time: str | None
    :param time_zone: Timezone for localization
    :type time_zone: pd.Timestamp.tz
    :param logger: Logger object
    :type logger: logging.Logger
    :return: Tuple of (start_dt, end_dt) or (False, False) on error
    :rtype: tuple[pd.Timestamp, pd.Timestamp] | tuple[bool, bool]
    """
    try:
        start_dt = pd.to_datetime(start_time)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize(time_zone)
    except Exception as e:
        logger.error(f"Invalid start_time format: {start_time}. Error: {e}")
        logger.error("Use format like '2024-01-01' or '2024-01-01 00:00:00'")
        return False, False

    if end_time:
        try:
            end_dt = pd.to_datetime(end_time)
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize(time_zone)
        except Exception as e:
            logger.error(f"Invalid end_time format: {end_time}. Error: {e}")
            return False, False
    else:
        end_dt = pd.Timestamp.now(tz=time_zone)
        logger.info(f"No end_time specified, using current time: {end_dt}")

    return start_dt, end_dt


def clean_sensor_column_names(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Clean sensor column names by removing 'sensor.' prefix.

    :param df: Input DataFrame with sensor columns
    :type df: pd.DataFrame
    :param timestamp_col: Name of timestamp column to preserve
    :type timestamp_col: str
    :return: DataFrame with cleaned column names
    :rtype: pd.DataFrame
    """
    column_mapping = {}
    for col in df.columns:
        if col != timestamp_col and col.startswith("sensor."):
            column_mapping[col] = col.replace("sensor.", "")
    return df.rename(columns=column_mapping)


def handle_nan_values(
    df: pd.DataFrame,
    handle_nan: str,
    timestamp_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Handle NaN values in DataFrame according to specified strategy.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :param handle_nan: Strategy for handling NaN values
    :type handle_nan: str
    :param timestamp_col: Name of timestamp column to exclude from processing
    :type timestamp_col: str
    :param logger: Logger object
    :type logger: logging.Logger
    :return: DataFrame with NaN values handled
    :rtype: pd.DataFrame
    """
    nan_count_before = df.isna().sum().sum()
    if nan_count_before == 0:
        return df

    logger.info(f"Found {nan_count_before} NaN values, applying handle_nan method: {handle_nan}")

    if handle_nan == "drop":
        df = df.dropna()
        logger.info(f"Dropped rows with NaN. Remaining rows: {len(df)}")
    elif handle_nan == "fill_zero":
        # Exclude timestamp_col from fillna to avoid unintended changes
        fill_cols = [col for col in df.columns if col != timestamp_col]
        df[fill_cols] = df[fill_cols].fillna(0)
        logger.info("Filled NaN values with 0 (excluding timestamp)")
    elif handle_nan == "interpolate":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude timestamp_col from interpolation
        interp_cols = [col for col in numeric_cols if col != timestamp_col]
        df[interp_cols] = df[interp_cols].interpolate(method="linear", limit_direction="both")
        df[interp_cols] = df[interp_cols].ffill().bfill()
        logger.info("Interpolated NaN values (excluding timestamp)")
    elif handle_nan == "forward_fill":
        # Exclude timestamp_col from forward fill
        fill_cols = [col for col in df.columns if col != timestamp_col]
        df[fill_cols] = df[fill_cols].ffill()
        logger.info("Forward filled NaN values (excluding timestamp)")
    elif handle_nan == "backward_fill":
        # Exclude timestamp_col from backward fill
        fill_cols = [col for col in df.columns if col != timestamp_col]
        df[fill_cols] = df[fill_cols].bfill()
        logger.info("Backward filled NaN values (excluding timestamp)")
    elif handle_nan == "keep":
        logger.info("Keeping NaN values as-is")
    else:
        logger.warning(f"Unknown handle_nan option '{handle_nan}', keeping NaN values")

    return df


def resample_and_filter_data(
    df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    resample_freq: str,
    logger: logging.Logger,
) -> pd.DataFrame | bool:
    """
    Filter DataFrame to time range and resample to specified frequency.

    :param df: Input DataFrame with datetime index
    :type df: pd.DataFrame
    :param start_dt: Start datetime for filtering
    :type start_dt: pd.Timestamp
    :param end_dt: End datetime for filtering
    :type end_dt: pd.Timestamp
    :param resample_freq: Resampling frequency string (e.g., '1h', '30min')
    :type resample_freq: str
    :param logger: Logger object
    :type logger: logging.Logger
    :return: Resampled DataFrame or False on error
    :rtype: pd.DataFrame | bool
    """
    # Validate that DataFrame index is datetime and properly localized
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error(f"DataFrame index must be DatetimeIndex, got {type(df.index).__name__}")
        return False

    # Check if timezone aware and matches expected timezone
    if df.index.tz is None:
        logger.warning("DataFrame index is timezone-naive, localizing to match start/end times")
        df = df.copy()
        df.index = df.index.tz_localize(start_dt.tz)
    elif df.index.tz != start_dt.tz:
        logger.warning(
            f"DataFrame timezone ({df.index.tz}) differs from filter timezone ({start_dt.tz}), converting"
        )
        df = df.copy()
        df.index = df.index.tz_convert(start_dt.tz)

    # Filter to exact time range
    df_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]

    if df_filtered.empty:
        logger.error("No data in the specified time range after filtering")
        return False

    logger.info(f"Retrieved {len(df_filtered)} data points")

    # Resample to specified frequency
    logger.info(f"Resampling data to frequency: {resample_freq}")
    try:
        df_resampled = df_filtered.resample(resample_freq).mean()
        df_resampled = df_resampled.dropna(how="all")

        if df_resampled.empty:
            logger.error("No data after resampling. Check frequency and data availability.")
            return False

        logger.info(f"After resampling: {len(df_resampled)} data points")
        return df_resampled

    except Exception as e:
        logger.error(f"Error during resampling: {e}")
        return False


@contextmanager
def stage_timer(stage_times: dict, name: str, logger: logging.Logger | None = None):
    """Record wall-clock elapsed time of a block in ``stage_times[name]``.

    Emits one DEBUG line per stage when ``logger`` is provided. Works across
    ``await`` because ``time.perf_counter()`` is a plain monotonic read.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        stage_times[name] = dt
        if logger is not None:
            logger.debug(f"Stage [{name}] completed in {dt:.3f}s")


def log_runtime_banner(logger, optim_conf: dict | None = None):
    """Log a single INFO line with EMHASS/Python/CVXPY/platform info for bug-report reproducibility.

    When ``optim_conf`` is provided, the configured solver (or the EMHASS
    default ``Highs`` when the key is absent) is shown — this matches the
    solver the LP actually uses (see ``optimization.py`` constructor).
    When ``optim_conf`` is missing entirely (early-fail paths), falls back
    to ``cvxpy.installed_solvers()[0]``.
    """
    try:
        import platform as _plat
        from importlib.metadata import version as _pkg_version

        import cvxpy as _cvx

        _ver = _pkg_version("emhass")
        if isinstance(optim_conf, dict):
            solver = str(optim_conf.get("lp_solver", "Highs"))
        else:
            solvers = _cvx.installed_solvers()
            solver = solvers[0] if solvers else "none"
        logger.info(
            f"EMHASS {_ver} | Python {_plat.python_version()} | "
            f"CVXPY {_cvx.__version__} ({solver}) | "
            f"{_plat.system()}-{_plat.machine()}"
        )
    except Exception as err:
        try:
            from importlib.metadata import version as _pkg_version

            _ver = _pkg_version("emhass")
            logger.info(f"EMHASS {_ver} (runtime info unavailable: {err})")
        except Exception:
            logger.info("EMHASS (runtime info unavailable)")
