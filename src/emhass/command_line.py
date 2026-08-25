#!/usr/bin/env python3

import argparse
import asyncio
import copy
import json
import logging
import os
import pathlib
import pickle
import threading
import time as _time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import version

import aiofiles
import numpy as np
import orjson
import pandas as pd

from emhass import last_run, plan_store, utils
from emhass.battery_identification import SCHEMA_VERSION as _BATTERY_ID_SCHEMA_VERSION_V1
from emhass.battery_identification import BatteryIdentification
from emhass.forecast import Forecast
from emhass.forecast_calibration import (
    CALIBRATION_DEFAULT_DAYS,
    CALIBRATION_TEST_DAYS,
    CALIBRATION_VAL_DAYS,
    compute_forecast_calibration,
)
from emhass.machine_learning_forecaster import MLForecaster
from emhass.machine_learning_regressor import MLRegressor
from emhass.optimization import Optimization
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import log_runtime_banner, stage_timer

default_csv_filename = "opt_res_latest.csv"
default_pkl_suffix = "_mlf.pkl"
default_metadata_json = "metadata.json"
test_df_literal = "test_df_final.pkl"
EMHASS_SCHEMA_VERSION = "1.0"


def _record_optim_snapshot(
    input_data_dict: dict,
    action: str,
    opt_res,
    t0_monotonic: float,
    logger: logging.Logger,
) -> None:
    """Persist a last_run snapshot after an optim wrapper completes.

    Best-effort: any failure is logged with a traceback but does not
    propagate so the wrapper's return path stays intact.
    """
    try:
        optim_status = (
            opt_res["optim_status"].iloc[0]
            if isinstance(opt_res, pd.DataFrame) and "optim_status" in opt_res
            else "Unknown"
        )
        ts = last_run.record(
            input_data_dict["emhass_conf"]["data_path"],
            action=action,
            stage_times=input_data_dict["stage_times"],
            optim_status=optim_status,
            infeasible=(optim_status == "Infeasible"),
            duration_total_seconds=_time.monotonic() - t0_monotonic,
            schema_version=EMHASS_SCHEMA_VERSION,
        )
        # Publish the structured plan ONLY for a successful (Optimal) run, reusing
        # the SAME timestamp last_run stamped so /api/v1/plan's generated_at matches
        # /api/v1/last-run for that run. Only the timestamp is shared, not the
        # verdict: a failed/infeasible run is still recorded by last_run (status
        # error/infeasible) but must not surface on /api/v1/plan as status "ok" —
        # the plan endpoint keeps serving the last VALID plan (or no-run). Gating on
        # optim_status == "Optimal" mirrors last_run's own "ok" criterion, so the
        # two endpoints stay consistent (plan published iff last-run is "ok").
        if optim_status == "Optimal":
            plan_store.record(
                input_data_dict["emhass_conf"]["data_path"],
                plan=plan_store.serialize(opt_res),
                generated_at=ts,
                schema_version=EMHASS_SCHEMA_VERSION,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("last_run: failed to record %s snapshot", action, exc_info=exc)


@dataclass
class SetupContext:
    """
    A dataclass that serves as a context container for optimization preparation helpers.
    This context object encapsulates all necessary configuration and utility objects
    required for setting up and preparing optimization tasks.

    Attributes:
        retrieve_hass_conf (dict): Configuration dictionary for Home Assistant data retrieval.
        optim_conf (dict): Configuration dictionary for optimization parameters.
        plant_conf (dict): Configuration dictionary for plant/system parameters.
        emhass_conf (dict): Configuration dictionary for EMHASS settings.
        params (dict): Additional parameters dictionary.
        logger (logging.Logger): Logger instance for logging messages.
        get_data_from_file (bool): Flag indicating whether to retrieve data from file instead of live source.
        rh (RetrieveHass): RetrieveHass instance for retrieving Home Assistant data.
        fcst (Forecast | None): Optional Forecast object for weather or energy forecasting. Defaults to None.
    """

    retrieve_hass_conf: dict
    optim_conf: dict
    plant_conf: dict
    emhass_conf: dict
    params: dict
    logger: logging.Logger
    get_data_from_file: bool
    rh: RetrieveHass
    fcst: Forecast | None = None


@dataclass
class PublishContext:
    """
    Context object for data publishing helpers.

    Attributes:
        input_data_dict (dict): Dictionary containing input data with keys 'rh' (RetrieveHass),
            'opt' (Optimization - may be None for publish-data), and 'fcst' (Forecast) objects.
        params (dict): Parameters dictionary for publishing configuration.
        idx (int): Index identifier for the current publishing operation.
        common_kwargs (dict): Common keyword arguments shared across publishing helpers.
        logger (logging.Logger): Logger instance for recording publishing operations.
    """

    input_data_dict: dict
    params: dict
    idx: int
    common_kwargs: dict
    logger: logging.Logger

    @property
    def rh(self) -> RetrieveHass:
        return self.input_data_dict["rh"]

    @property
    def opt(self) -> Optimization:
        return self.input_data_dict["opt"]

    @property
    def optim_conf(self) -> dict:
        """Access optim_conf directly from input_data_dict (works even when opt is None)."""
        return self.input_data_dict["optim_conf"]

    @property
    def plant_conf(self) -> dict:
        """Access plant_conf directly from input_data_dict (works even when opt is None)."""
        return self.input_data_dict["plant_conf"]

    @property
    def fcst(self) -> Forecast:
        return self.input_data_dict["fcst"]


@dataclass(frozen=True)
class OptimizationCacheKey:
    """
    Frozen dataclass representing configuration fields that affect optimization structure.

    Changes to any of these fields require rebuilding the optimization problem.
    Using a frozen dataclass makes the cache key explicit, hashable, and easy to extend.
    """

    number_of_deferrable_loads: int
    set_use_battery: bool
    set_use_pv: bool
    treat_deferrable_load_as_semi_cont: tuple
    set_deferrable_load_single_constant: tuple
    set_deferrable_startup_penalty: tuple
    deferrable_load_max_cost: tuple
    set_deferrable_max_startups: tuple
    set_deferrable_load_as_timeseries: tuple
    nominal_power_of_deferrable_loads: tuple
    def_load_config_structure: tuple  # (index, type) tuples for each load
    deferrable_load_groups: tuple
    shared_thermal_tanks: tuple  # shared-tank multi-source topology structure
    is_electric_load: tuple  # per-load electric-bus membership flag
    inverter_is_hybrid: bool
    compute_curtailment: bool
    optimization_time_step_s: float | None
    delta_forecast_daily_s: float | None
    num_timesteps: int | None
    costfun: str
    plant_conf_hash: str
    optim_conf_structural_hash: str  # Hash of optim_conf keys that affect problem structure


class OptimizationCache:
    """
    In-memory cache for Optimization objects to enable warm-starting.

    Warm-starting reuses the previous solution as a starting point for the solver,
    which can significantly speed up repeated MPC optimizations where consecutive
    problems are similar.

    The cache is invalidated when configuration changes that affect the optimization
    structure (number of variables, constraints, etc.).

    Thread-safe: Uses a lock to prevent race conditions when multiple optimizations
    run concurrently in async code.
    """

    _instance: "Optimization | None" = None
    _cache_key: OptimizationCacheKey | None = None
    _last_used: datetime | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def _compute_cache_key(
        cls,
        optim_conf: dict,
        plant_conf: dict,
        costfun: str,
        retrieve_hass_conf: dict,
        num_timesteps: int | None = None,
    ) -> OptimizationCacheKey:
        """
        Compute a cache key from configuration that affects optimization structure.

        Returns a frozen dataclass that can be directly compared for equality.
        Changes to any field require rebuilding the optimization problem.
        """

        def to_seconds(val):
            """Convert Timedelta/timedelta to seconds."""
            if val is None:
                return None
            return val.total_seconds() if hasattr(val, "total_seconds") else float(val)

        def to_tuple(val):
            """Convert lists/arrays to tuples for hashability."""
            if val is None:
                return ()
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, list | tuple):
                # Handle nested lists (e.g., nominal_power with sequences)
                return tuple(tuple(v) if isinstance(v, list | tuple) else v for v in val)
            return (val,)

        def config_hash(cfg: dict, exclude_keys: set | None = None) -> str:
            """Create a stable hash of config dict for cache key comparison.

            Args:
                cfg: The config dict to hash
                exclude_keys: Keys to exclude from hash (runtime params)
            """
            import hashlib

            if exclude_keys is None:
                exclude_keys = set()
            # Sort keys for deterministic ordering, exclude runtime parameters
            sorted_items = sorted(
                ((k, v) for k, v in cfg.items() if k not in exclude_keys),
                key=lambda x: str(x[0]),
            )
            config_str = str(sorted_items)
            return hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Runtime parameters that should NOT affect cache key
        # These change between MPC iterations but don't affect problem structure
        thermal_runtime_keys = {
            "start_temperature",
            "desired_temperatures",
            "indoor_target_temperature",  # thermal_battery runtime param
            "q_input_initial",  # thermal inertia warm-start override
            "draw_off_demand",  # hot water tank daily profile (updates heating_demand param)
        }
        # Plant parameters that are updated dynamically (no rebuild needed)
        plant_runtime_keys = {
            "soc_init",
            "battery_target_state_of_charge",
            "battery_charge_power_max",
            "battery_discharge_power_max",
        }
        # Optim conf parameters that don't affect problem structure
        # (parameterized via CVXPY Parameters, solver options, or forecast method selection)
        optim_conf_runtime_keys = {
            # Parameterized via CVXPY Parameters
            "operating_hours_of_each_deferrable_load",
            "operating_timesteps_of_each_deferrable_load",
            "start_timesteps_of_each_deferrable_load",
            "end_timesteps_of_each_deferrable_load",
            "def_current_state",
            # Per-call elapsed on-time for min-on remainder (issue #952); value
            # is read via cp.Parameter so no rebuild on cache hit.
            "def_current_on_timesteps",
            # Per-call elapsed off-time for min-off remainder (#952 follow-on); value
            # is read via cp.Parameter so no rebuild on cache hit.
            "def_current_off_timesteps",
            # Per-call current power in watts (issue #605); pin value is a cp.Parameter.
            "def_current_power",
            # Per-call completed operating timesteps today (issue #983); decrements
            # required_timesteps + target_energy via cp.Parameter (no rebuild on cache hit).
            "def_current_operating_timesteps",
            "minimum_power_of_deferrable_loads",
            "cost_forecast_per_deferrable_load",
            # shared_thermal_tanks has its own structural hash field above
            "shared_thermal_tanks",
            # heat_topology is compiled down to the structural fields which ARE
            # part of the cache key (def_load_config_structure,
            # deferrable_load_groups, shared_thermal_tanks). The raw
            # heat_topology itself is excluded to avoid double-counting.
            "heat_topology",
            # Solver options (updated on cache hit)
            "lp_solver_timeout",
            "lp_solver_mip_rel_gap",
            "num_threads",
            "lp_solver",
            # Forecast method selection (not optimization structure)
            "weather_forecast_method",
            "load_cost_forecast_method",
            "production_price_forecast_method",
            "load_forecast_method",
            # Already handled by explicit fields or separate hash
            "def_load_config",
            "delta_forecast_daily",
        }
        # Extract def_load_config structure (which loads are thermal/thermal_battery/standard)
        # Include hash of thermal config contents to detect parameter changes
        def_load_config = optim_conf.get("def_load_config", []) or []
        def_structure = []
        for i, cfg in enumerate(def_load_config):
            cfg = cfg or {}
            if "thermal_config" in cfg:
                # Include hash of thermal_config contents to detect parameter changes
                # Exclude runtime parameters (start_temperature, desired_temperatures) from hash
                thermal_hash = config_hash(cfg["thermal_config"], thermal_runtime_keys)
                load_type = f"thermal_config:{thermal_hash}"
            elif "thermal_battery" in cfg:
                # Include hash of thermal_battery contents to detect parameter changes
                # Exclude runtime parameters (start_temperature, desired_temperatures) from hash
                thermal_hash = config_hash(cfg["thermal_battery"], thermal_runtime_keys)
                load_type = f"thermal_battery:{thermal_hash}"
            else:
                load_type = "standard"
            def_structure.append((i, load_type))

        return OptimizationCacheKey(
            number_of_deferrable_loads=optim_conf.get("number_of_deferrable_loads", 0),
            set_use_battery=optim_conf.get("set_use_battery", False),
            set_use_pv=optim_conf.get("set_use_pv", True),
            treat_deferrable_load_as_semi_cont=to_tuple(
                optim_conf.get("treat_deferrable_load_as_semi_cont", [])
            ),
            set_deferrable_load_single_constant=to_tuple(
                optim_conf.get("set_deferrable_load_single_constant", [])
            ),
            set_deferrable_startup_penalty=to_tuple(
                optim_conf.get("set_deferrable_startup_penalty", [])
            ),
            deferrable_load_max_cost=to_tuple(optim_conf.get("deferrable_load_max_cost", [])),
            set_deferrable_max_startups=to_tuple(optim_conf.get("set_deferrable_max_startups", [])),
            set_deferrable_load_as_timeseries=to_tuple(
                optim_conf.get("set_deferrable_load_as_timeseries", [])
            ),
            # Note: The following are parameterized and don't require rebuild:
            # - start_timesteps and end_timesteps (via window masks)
            # - operating_hours_of_each_deferrable_load (via Big-M energy constraints)
            nominal_power_of_deferrable_loads=to_tuple(
                optim_conf.get("nominal_power_of_deferrable_loads", [])
            ),
            def_load_config_structure=tuple(def_structure),
            deferrable_load_groups=tuple(
                (tuple(g.get("names", [])), g.get("max_power"), g.get("mutual_exclusion", False))
                for g in optim_conf.get("deferrable_load_groups", [])
            ),
            # shared_thermal_tanks change problem structure (new tank state
            # variable + dynamics constraints), so include a structure hash.
            shared_thermal_tanks=tuple(
                (
                    t.get("id", ""),
                    tuple(int(k) for k in t.get("load_ids", [])),
                    config_hash(
                        {
                            k: v
                            for k, v in t.items()
                            if k not in {"start_temperature", "draw_off_demand"}
                        },
                    ),
                )
                for t in optim_conf.get("shared_thermal_tanks", []) or []
            ),
            # is_electric_load changes p_def_sum membership, hence the electric
            # power balance shape, hence structural.
            is_electric_load=to_tuple(optim_conf.get("is_electric_load", [])),
            inverter_is_hybrid=plant_conf.get("inverter_is_hybrid", False),
            compute_curtailment=plant_conf.get("compute_curtailment", False),
            optimization_time_step_s=to_seconds(retrieve_hass_conf.get("optimization_time_step")),
            delta_forecast_daily_s=to_seconds(optim_conf.get("delta_forecast_daily")),
            num_timesteps=num_timesteps,
            costfun=costfun,
            plant_conf_hash=config_hash(plant_conf, plant_runtime_keys),
            optim_conf_structural_hash=config_hash(optim_conf, optim_conf_runtime_keys),
        )

    @classmethod
    def get(
        cls,
        optim_conf: dict,
        plant_conf: dict,
        costfun: str,
        retrieve_hass_conf: dict,
        logger: logging.Logger,
        num_timesteps: int | None = None,
    ) -> "Optimization | None":
        """
        Get cached Optimization object if configuration matches.

        Returns None if cache is empty or configuration has changed.
        Thread-safe via internal locking.
        """
        cache_key = cls._compute_cache_key(
            optim_conf, plant_conf, costfun, retrieve_hass_conf, num_timesteps
        )

        with cls._lock:
            if cls._instance is not None and cls._cache_key == cache_key:
                age = datetime.now() - cls._last_used if cls._last_used else timedelta(0)
                logger.debug(
                    f"OptimizationCache HIT: Reusing cached optimization object "
                    f"(age={age.total_seconds():.1f}s) - warm-start enabled"
                )
                cls._last_used = datetime.now()
                return cls._instance

            if cls._instance is not None:
                # Log which fields changed for debugging
                if cls._cache_key is not None:
                    changed_fields = []
                    for field in cache_key.__dataclass_fields__:
                        old_val = getattr(cls._cache_key, field)
                        new_val = getattr(cache_key, field)
                        if old_val != new_val:
                            changed_fields.append(f"{field}: {old_val!r} -> {new_val!r}")
                    if changed_fields:
                        logger.debug(
                            f"OptimizationCache MISS: Config changed - {', '.join(changed_fields)}"
                        )
                    else:
                        logger.debug("OptimizationCache MISS: Config changed (unknown diff)")
                else:
                    logger.debug("OptimizationCache MISS: Config changed, rebuilding optimization")
            else:
                logger.debug("OptimizationCache MISS: Empty cache, building new optimization")

            return None

    @classmethod
    def put(
        cls,
        opt: "Optimization",
        optim_conf: dict,
        plant_conf: dict,
        costfun: str,
        retrieve_hass_conf: dict,
        logger: logging.Logger,
        num_timesteps: int | None = None,
    ) -> None:
        """Store Optimization object in cache. Thread-safe via internal locking."""
        cache_key = cls._compute_cache_key(
            optim_conf, plant_conf, costfun, retrieve_hass_conf, num_timesteps
        )
        with cls._lock:
            cls._instance = opt
            cls._cache_key = cache_key
            cls._last_used = datetime.now()
            logger.debug(
                f"OptimizationCache: Stored optimization object "
                f"(loads={cache_key.number_of_deferrable_loads}, battery={cache_key.set_use_battery})"
            )

    @classmethod
    def clear(cls, logger: logging.Logger | None = None) -> None:
        """Clear the cache (e.g., for testing or explicit invalidation). Thread-safe."""
        with cls._lock:
            cls._instance = None
            cls._cache_key = None
            cls._last_used = None
            if logger:
                logger.debug("OptimizationCache: CLEARED")

    @classmethod
    def get_stats(cls) -> dict:
        """Get cache statistics for debugging. Thread-safe."""
        with cls._lock:
            return {
                "has_instance": cls._instance is not None,
                "cache_key": cls._cache_key,
                "last_used": cls._last_used.isoformat() if cls._last_used else None,
            }


async def _retrieve_from_file(
    emhass_conf: dict,
    test_df_literal: str,
    rh: RetrieveHass,
    retrieve_hass_conf: dict,
    optim_conf: dict,
) -> tuple[bool, object]:
    """Helper to retrieve data from a pickle file and configure variables."""
    async with aiofiles.open(emhass_conf["data_path"] / test_df_literal, "rb") as inp:
        content = await inp.read()
        rh.df_final, days_list, var_list, rh.ha_config = pickle.loads(content)
        rh.var_list = var_list
    # Assign variables based on set_type
    retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(var_list[0])
    if optim_conf.get("set_use_pv", True):
        retrieve_hass_conf["sensor_power_photovoltaics"] = str(var_list[1])
        retrieve_hass_conf["sensor_linear_interp"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"],
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
        ]
        retrieve_hass_conf["sensor_replace_zero"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"],
            var_list[2],
        ]
    else:
        retrieve_hass_conf["sensor_linear_interp"] = [
            retrieve_hass_conf["sensor_power_load_no_var_loads"]
        ]
        retrieve_hass_conf["sensor_replace_zero"] = []
    return True, days_list


def _append_entity_ids(var_list: list, value) -> None:
    """
    Append a battery_id sensor config value to ``var_list``.

    A list (N>1, already resolved and duplicate-checked by
    :func:`_resolve_battery_sensor_lists`) is appended element-wise, deduped
    against ``var_list``. The resolver only rejects a within-list or
    cross-list duplicate, not a battery sensor equal to the load sensor
    already at ``var_list[0]``, so this guard is genuinely reachable there,
    not redundant with it. A bare string (N=1) is appended unconditionally.
    """
    if isinstance(value, list):
        for entity_id in value:
            if entity_id not in var_list:
                var_list.append(entity_id)
    else:
        var_list.append(value)


def _append_battery_id_sensors(var_list: list, retrieve_hass_conf: dict) -> None:
    """Append the battery power and SoC sensor config values to ``var_list``."""
    _append_entity_ids(var_list, retrieve_hass_conf["sensor_power_battery"])
    _append_entity_ids(var_list, retrieve_hass_conf["sensor_battery_state_of_charge"])


async def _retrieve_from_hass(
    set_type: str,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    logger: logging.Logger | None,
) -> tuple[bool, object]:
    """Helper to retrieve live data from Home Assistant."""
    # Determine days_list based on set_type
    if set_type in ("perfect-optim", "adjust_pv", "battery_id"):
        days_list = utils.get_days_list(retrieve_hass_conf["historic_days_to_retrieve"])
    elif set_type == "naive-mpc-optim":
        days_list = utils.get_days_list(1)
    else:
        days_list = None  # Not needed for dayahead
    var_list = [retrieve_hass_conf["sensor_power_load_no_var_loads"]]
    if set_type == "battery_id":
        # Battery identification needs signed battery power and measured SoC,
        # one pair per battery. The load sensor stays at var_list[0] so
        # prepare_data's load handling is unchanged; the battery columns are
        # passed to prepare_data as protected_columns so its set_zero_min clip
        # cannot destroy the discharge direction or a measured 0% SoC (#1041).
        # A list value (N>1, already resolved by _identify_battery_impl) is
        # appended per-id, deduped against var_list; a bare string (N=1) is
        # the plain single-sensor case.
        _append_battery_id_sensors(var_list, retrieve_hass_conf)
        if logger:
            logger.debug(f"Variable list for battery_id retrieval: {var_list}")
    elif optim_conf.get("set_use_pv", True):
        var_list.append(retrieve_hass_conf["sensor_power_photovoltaics"])
        if optim_conf.get("set_use_adjusted_pv", True):
            var_list.append(retrieve_hass_conf["sensor_power_photovoltaics_forecast"])
            if logger:
                logger.debug(f"Variable list for data retrieval: {var_list}")
    success = await rh.get_data(
        days_list,
        var_list,
        minimal_response=False,
        significant_changes_only=False,
        # A stale battery's sensor shouldn't cost a healthy battery days it did
        # have data for; each battery's fit only reads its own two columns (#1061).
        keep_partial_days=(set_type == "battery_id"),
    )
    return success, days_list


async def retrieve_home_assistant_data(
    set_type: str,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: str,
    logger: logging.Logger | None = None,
) -> tuple[bool, pd.DataFrame | None, list | None]:
    """Retrieve data from Home Assistant or file and prepare it for optimization."""

    if get_data_from_file:
        success, days_list = await _retrieve_from_file(
            emhass_conf, test_df_literal, rh, retrieve_hass_conf, optim_conf
        )
    else:
        success, days_list = await _retrieve_from_hass(
            set_type, retrieve_hass_conf, optim_conf, rh, logger
        )
    if not success:
        return False, None, days_list
    protected_columns = None
    if set_type == "battery_id":
        # The identifier needs both flow directions of the signed battery
        # power sensor and any legitimately measured 0% SoC sample, so these
        # columns are exempt from the set_zero_min clip (#1041). Columns not
        # present in the retrieved frame are ignored by prepare_data.
        protected_columns = []
        _append_battery_id_sensors(protected_columns, retrieve_hass_conf)
    rh.prepare_data(
        retrieve_hass_conf["sensor_power_load_no_var_loads"],
        load_negative=retrieve_hass_conf["load_negative"],
        set_zero_min=retrieve_hass_conf["set_zero_min"],
        var_replace_zero=retrieve_hass_conf["sensor_replace_zero"],
        var_interp=retrieve_hass_conf["sensor_linear_interp"],
        protected_columns=protected_columns,
    )
    return True, rh.df_final.copy(), days_list


def is_model_outdated(
    model_path: pathlib.Path,
    max_age_hours: int,
    logger: logging.Logger,
    label: str = "Adjusted PV model",
) -> bool:
    """
    Check if the saved model file is outdated based on its modification time.

    Format-agnostic: only the file mtime is inspected, so this serves both the
    adjusted-PV regressor pickle and the battery-identification JSON.

    :param model_path: Path to the saved model file.
    :type model_path: pathlib.Path
    :param max_age_hours: Maximum age in hours before model is considered outdated.
    :type max_age_hours: int
    :param logger: Logger object for logging information.
    :type logger: logging.Logger
    :param label: Human-readable name of the artifact, used in the log lines so
        the message matches the caller (e.g. "Battery identification model").
    :type label: str
    :return: True if model is outdated or doesn't exist, False otherwise.
    :rtype: bool
    """
    if not model_path.exists():
        logger.info(f"{label} file does not exist, will train new model")
        return True

    if max_age_hours <= 0:
        logger.info(f"{label} max age is set to 0, forcing model re-fit")
        return True

    model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
    model_age = datetime.now() - model_mtime
    max_age = timedelta(hours=max_age_hours)

    if model_age > max_age:
        logger.info(
            f"{label} is outdated (age: {model_age.total_seconds() / 3600:.1f}h, "
            f"max: {max_age_hours}h), will train new model"
        )
        return True
    else:
        logger.info(
            f"Using existing {label} (age: {model_age.total_seconds() / 3600:.1f}h, "
            f"max: {max_age_hours}h)"
        )
        return False


async def _retrieve_and_fit_pv_model(
    fcst: Forecast,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: pd.DataFrame,
) -> bool:
    """
    Helper function to retrieve data and fit the PV adjustment model.

    :param fcst: Forecast object used for PV forecast adjustment.
    :type fcst: Forecast
    :param get_data_from_file: Whether to retrieve data from a file instead of Home Assistant.
    :type get_data_from_file: bool
    :param retrieve_hass_conf: Configuration dictionary for retrieving data from Home Assistant.
    :type retrieve_hass_conf: dict
    :param optim_conf: Configuration dictionary for optimization settings.
    :type optim_conf: dict
    :param rh: RetrieveHass object for interacting with Home Assistant.
    :type rh: RetrieveHass
    :param emhass_conf: Configuration dictionary for emhass paths and settings.
    :type emhass_conf: dict
    :param test_df_literal: DataFrame containing test data for debugging purposes.
    :type test_df_literal: pd.DataFrame
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    # Retrieve data from Home Assistant
    success, df_input_data, days_list = await retrieve_home_assistant_data(
        "adjust_pv",
        get_data_from_file,
        retrieve_hass_conf,
        optim_conf,
        rh,
        emhass_conf,
        test_df_literal,
    )
    if not success:
        return False
    # Best-effort retrieval of the curtailment history: timesteps where PV was
    # curtailed must not train the adjustment model (issue #1026). Any failure
    # here (no history, entity missing) falls back to unfiltered training.
    curtailment_series = None
    plant_conf = getattr(fcst, "plant_conf", None) or {}
    if plant_conf.get("compute_curtailment", False) and not get_data_from_file:
        params = getattr(fcst, "params", None) or {}
        curtailment_entity = (
            params.get("passed_data", {})
            .get("custom_pv_curtailment_id", {})
            .get("entity_id", "sensor.p_pv_curtailment")
        )
        try:
            success_curtailment = await rh.get_data(days_list, [curtailment_entity])
            if success_curtailment is not False and curtailment_entity in rh.df_final.columns:
                curtailment_series = rh.df_final[curtailment_entity].copy()
            else:
                fcst.logger.info(
                    f"No history for curtailment entity {curtailment_entity}, "
                    "training the PV adjustment on unfiltered data."
                )
        except Exception as e:
            fcst.logger.info(
                f"Could not retrieve curtailment history ({type(e).__name__}: {e}), "
                "training the PV adjustment on unfiltered data."
            )
    # Call data preparation method
    fcst.adjust_pv_forecast_data_prep(df_input_data, curtailment_series=curtailment_series)
    n_splits = 5
    x_adjust_pv = getattr(fcst, "x_adjust_pv", None)
    if x_adjust_pv is not None and len(x_adjust_pv) <= n_splits:
        fcst.logger.warning(
            f"Not enough data to fit the PV model (found {len(x_adjust_pv)} samples, "
            f"require > {n_splits}). Falling back to unadjusted PV forecast."
        )
        return False
    # Call the fit method
    await fcst.adjust_pv_forecast_fit(
        n_splits=n_splits,
        regression_model=optim_conf["adjusted_pv_regression_model"],
    )
    return True


async def adjust_pv_forecast(
    logger: logging.Logger,
    fcst: Forecast,
    p_pv_forecast: pd.Series,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: pd.DataFrame,
) -> pd.Series:
    """
    Adjust the photovoltaic (PV) forecast using historical data and a regression model.

    This method retrieves historical data, prepares it for model fitting, trains a regression
    model, and adjusts the provided PV forecast based on the trained model.

    :param logger: Logger object for logging information and errors.
    :type logger: logging.Logger
    :param fcst: Forecast object used for PV forecast adjustment.
    :type fcst: Forecast
    :param p_pv_forecast: The initial PV forecast to be adjusted.
    :type p_pv_forecast: pd.Series
    :param get_data_from_file: Whether to retrieve data from a file instead of Home Assistant.
    :type get_data_from_file: bool
    :param retrieve_hass_conf: Configuration dictionary for retrieving data from Home Assistant.
    :type retrieve_hass_conf: dict
    :param optim_conf: Configuration dictionary for optimization settings.
    :type optim_conf: dict
    :param rh: RetrieveHass object for interacting with Home Assistant.
    :type rh: RetrieveHass
    :param emhass_conf: Configuration dictionary for emhass paths and settings.
    :type emhass_conf: dict
    :param test_df_literal: DataFrame containing test data for debugging purposes.
    :type test_df_literal: pd.DataFrame
    :return: The adjusted PV forecast as a pandas Series.
    :rtype: pd.Series
    """
    # Normalize data_path to Path object for safety (handles both str and Path types)
    data_path = pathlib.Path(emhass_conf["data_path"])
    model_filename = "adjust_pv_regressor.pkl"
    model_path = data_path / model_filename
    max_age_hours = optim_conf.get("adjusted_pv_model_max_age", 24)
    # Check if model needs to be re-fitted
    if is_model_outdated(model_path, max_age_hours, logger):
        logger.info("Adjusting PV forecast, retrieving history data for model fit")
        success = await _retrieve_and_fit_pv_model(
            fcst,
            get_data_from_file,
            retrieve_hass_conf,
            optim_conf,
            rh,
            emhass_conf,
            test_df_literal,
        )
        if not success:
            logger.warning(
                "Could not train adjusted PV model, falling back to unadjusted PV forecast."
            )
            return p_pv_forecast
    else:
        # Load existing model
        logger.info("Loading existing adjusted PV model from file")
        try:
            async with aiofiles.open(model_path, "rb") as inp:
                content = await inp.read()
                fcst.model_adjust_pv = pickle.loads(content)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
            logger.error(f"Failed to load existing adjusted PV model: {type(e).__name__}: {str(e)}")
            logger.warning(
                "Model file may be corrupted or incompatible. Falling back to re-fitting the model."
            )
            # Use helper function to retrieve data and re-fit model
            success = await _retrieve_and_fit_pv_model(
                fcst,
                get_data_from_file,
                retrieve_hass_conf,
                optim_conf,
                rh,
                emhass_conf,
                test_df_literal,
            )
            if not success:
                logger.error(
                    "Failed to retrieve data for model re-fit after load error. Falling back to unadjusted forecast."
                )
                return p_pv_forecast
            logger.info("Successfully re-fitted model after load failure")
        except Exception as e:
            logger.error(
                f"Unexpected error loading adjusted PV model: {type(e).__name__}: {str(e)}"
            )
            logger.error("Cannot recover from this error")
            return False
    # Call the predict method
    p_pv_forecast_in = p_pv_forecast.rename("forecast").to_frame()
    try:
        p_pv_forecast_out = fcst.adjust_pv_forecast_predict(forecasted_pv=p_pv_forecast_in)
    except ValueError as e:
        # A model persisted by an older version may have been trained on a
        # different feature set (e.g. the raw integer "hour" feature that was
        # replaced by the cyclic hour encoding). scikit-learn then raises a
        # ValueError on predict (feature-name mismatch). Re-fit once with the
        # current feature set and retry; other exception types propagate.
        logger.warning(
            f"Adjusted PV model prediction failed ({type(e).__name__}: {e}). "
            "The saved model may predate a feature-set change. Re-fitting."
        )
        success = await _retrieve_and_fit_pv_model(
            fcst,
            get_data_from_file,
            retrieve_hass_conf,
            optim_conf,
            rh,
            emhass_conf,
            test_df_literal,
        )
        if not success:
            logger.warning(
                "Could not re-fit the adjusted PV model, falling back to unadjusted PV forecast."
            )
            return p_pv_forecast
        p_pv_forecast_out = fcst.adjust_pv_forecast_predict(forecasted_pv=p_pv_forecast_in)
    # Update the PV forecast
    return p_pv_forecast_out["adjusted_forecast"].rename(None)


# Suggest-tier HA sensor entity ids (fixed; not user-configurable).
BATTERY_ID_CAPACITY_SENSOR = "sensor.battery_identified_capacity"
BATTERY_ID_RTE_SENSOR = "sensor.battery_identified_round_trip_efficiency"


def _batt_conf_val(value, k: int | None):
    """
    Scalar-or-list read for a plant_conf battery value (#1032 array-ifies 9
    plant_conf battery params at number_of_batteries > 1).

    k=None (N=1) always returns ``value`` unchanged, so every call site's N=1
    output is identical to master regardless of this helper's existence. This
    is unrelated to the sensor-key list/bare-string ambiguity (CONTRACT.md's
    SCOPE NOTE on invariant 1): this helper only ever reads plant_conf battery
    params, which keep #1032's own scalar-at-N=1 normalisation. At N>1, k
    selects index k of a per-battery list; a value that is still a bare
    scalar despite k being given is returned as-is (defensive: normal configs
    are already array-ified by check_batt_params before plant_conf reaches
    here, but a hand-built plant_conf, e.g. in a test fixture, may not be).
    """
    if k is None or not isinstance(value, list):
        return value
    return value[k]


def _resolve_battery_sensor_lists(
    retrieve_hass_conf: dict, num_batteries: int, logger: logging.Logger
) -> tuple[list, list] | None:
    """
    Resolve sensor_power_battery / sensor_battery_state_of_charge into exact-
    length per-battery lists, index-matched to the battery config lists.

    Deliberately NO scalar broadcast at num_batteries > 1: one HA sensor
    cannot identify two independent batteries, unlike the numeric plant_conf
    params check_batt_params fans out. At N=1 a bare scalar (today's only
    supported shape) and a length-1 list both resolve to a single-element
    list. Anything else at N>1 (a scalar, or a list of the wrong length) is a
    misconfiguration. So is any list entry that is not a non-empty entity-id
    string, a duplicate id within one list (two batteries sharing a meter), or
    an id shared between the power and SOC lists (one entity cannot be both
    signals). Every rejection returns None after logging one precise warning
    naming the offending key and what's wrong, so the caller can skip cleanly
    before ever touching retrieval.
    """
    resolved: dict[str, list] = {}
    for key in ("sensor_power_battery", "sensor_battery_state_of_charge"):
        raw = retrieve_hass_conf.get(key)
        if isinstance(raw, list):
            if len(raw) != num_batteries:
                logger.warning(
                    "Battery identification: '%s' is a list of length %d but "
                    "number_of_batteries=%d requires exactly %d entity ids "
                    "(one per battery); skipping.",
                    key,
                    len(raw),
                    num_batteries,
                    num_batteries,
                )
                return None
            for idx, entry in enumerate(raw):
                if not isinstance(entry, str) or not entry:
                    logger.warning(
                        "Battery identification: '%s'[%d] is %r, expected a "
                        "non-empty entity-id string; skipping.",
                        key,
                        idx,
                        entry,
                    )
                    return None
            if num_batteries > 1:
                seen: set[str] = set()
                for entry in raw:
                    if entry in seen:
                        logger.warning(
                            "Battery identification: '%s' has duplicate entity id "
                            "%r; one sensor cannot identify two batteries; skipping.",
                            key,
                            entry,
                        )
                        return None
                    seen.add(entry)
            resolved[key] = list(raw)
        else:
            if num_batteries > 1:
                logger.warning(
                    "Battery identification: '%s' is a single value (%r) but "
                    "number_of_batteries=%d requires a list of %d entity ids "
                    "(one per battery, no broadcast for per-battery sensors); "
                    "skipping.",
                    key,
                    raw,
                    num_batteries,
                    num_batteries,
                )
                return None
            resolved[key] = [raw]
    power_list = resolved["sensor_power_battery"]
    soc_list = resolved["sensor_battery_state_of_charge"]
    if num_batteries > 1:
        overlap = set(power_list) & set(soc_list)
        if overlap:
            logger.warning(
                "Battery identification: entity id(s) %s used as both a power "
                "and a state-of-charge sensor; one entity cannot be both "
                "signals; skipping.",
                sorted(overlap),
            )
            return None
    return power_list, soc_list


# On-disk persistence: flat v1 payload at N=1 (same JSON shape as master for
# both a bare-string and a length-1-list sensor config - see CONTRACT.md's
# SCOPE NOTE on invariant 1); a schema_version=2 container of per-battery
# v1-style payloads at N>1, keyed by str(k), so one battery's freshness/
# failure never touches another's entry.
_BATTERY_ID_SCHEMA_VERSION_V2 = 2
# _BATTERY_ID_SCHEMA_VERSION_V1 is imported above (aliased from
# emhass.battery_identification.SCHEMA_VERSION): written into a failure-only
# N=1 payload or N>1 slot (see _record_battery_identification_failed_attempt)
# when there is no usable prior fit to retain. Importing it (rather than
# redeclaring the literal here) keeps the flat-payload schema_version and
# BatteryIdentificationResult.to_dict()'s own schema_version in sync by
# construction.


def _load_battery_identification_container(json_path: pathlib.Path, logger: logging.Logger) -> dict:
    """
    Read the persisted battery-identification JSON as a v2 per-battery container.

    Tolerant of every "not a usable v2 container" shape: missing file,
    unreadable JSON, a flat v1 payload (has "status" at top level, no
    "batteries" dict), and a foreign/future schema_version all normalise to an
    empty container, so every battery in the per-k loop is treated as
    absent/stale and re-fits - exactly like a missing file does today. A v1
    (or v3+) file is never partially parsed as v2.
    """
    empty = {"schema_version": _BATTERY_ID_SCHEMA_VERSION_V2, "batteries": {}}
    if not json_path.exists():
        return empty
    try:
        payload = json.loads(json_path.read_bytes())
    except (KeyError, ValueError, OSError) as e:
        logger.warning(
            f"Battery identification result unreadable ({type(e).__name__}); will re-fit."
        )
        return empty
    if not isinstance(payload, dict) or not isinstance(payload.get("batteries"), dict):
        return empty
    if payload.get("schema_version") != _BATTERY_ID_SCHEMA_VERSION_V2:
        return empty
    return {
        "schema_version": _BATTERY_ID_SCHEMA_VERSION_V2,
        "batteries": dict(payload["batteries"]),
    }


def _atomic_json_write(json_path: pathlib.Path, data: dict) -> None:
    """
    Write ``data`` as JSON to ``json_path`` atomically (tmp + os.replace).

    Unique temp name (pid + uuid) so overlapping identify_battery calls in
    the same process never share a tmp file before the replace.
    """
    tmp_path = json_path.with_name(
        f"battery_identification.json.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    with open(tmp_path, "w", encoding="utf-8") as outp:
        json.dump(data, outp, indent=2)
    os.replace(tmp_path, json_path)


def _read_raw_battery_identification_json(json_path: pathlib.Path):
    """
    Read the raw persisted JSON with no shape normalisation, or ``None`` if
    the file is missing or unreadable.

    Used by the failure-recording paths, which must distinguish v1/v2/foreign
    shapes themselves rather than have
    :func:`_load_battery_identification_container` silently normalise a
    foreign shape away before they get a chance to look at it.
    """
    try:
        with open(json_path, "rb") as inp:
            return json.loads(inp.read())
    except (KeyError, ValueError, OSError):
        return None


def _write_battery_identification_container(
    json_path: pathlib.Path, logger: logging.Logger, k: int, entry: dict
) -> None:
    """
    Read-modify-write a single battery's entry into the on-disk v2 container.

    Re-reads the container from disk immediately before merging, rather than
    writing back an in-memory copy loaded at the start of the run: two
    concurrent identify_battery runs (e.g. a dayahead call racing an MPC cron)
    both loading before either writes would otherwise let the second writer's
    whole-container overwrite silently drop the first writer's entry. Re-
    reading here shrinks that lost-update window to the time between this
    read and this write, not the whole per-k loop. The write itself is still
    atomic (tmp + os.replace), so a crash mid-write can never corrupt the file
    or lose any OTHER battery's already-persisted entry.
    """
    container = _load_battery_identification_container(json_path, logger)
    container["batteries"][str(k)] = entry
    _atomic_json_write(json_path, container)


def _battery_id_sensors_match(sensors, current_power: str, current_soc: str) -> bool:
    """
    True iff ``sensors`` (an entry's or a last_attempt's own recorded
    ``{"power": ..., "soc": ...}`` pair) matches the CURRENTLY resolved
    ``(current_power, current_soc)`` pair.

    Missing (e.g. an entry written before this field existed) or malformed
    ``sensors`` both count as not-matching. Shared by
    :func:`_battery_fit_is_stale` (both its ok-record and last_attempt
    freshness branches) and the N>1 not-stale publish gate, so "does this
    record's sensors match the current pair" has one definition, not two.
    """
    return (
        isinstance(sensors, dict)
        and sensors.get("power") == current_power
        and sensors.get("soc") == current_soc
    )


def _battery_identification_last_attempt(entry) -> dict | None:
    """
    Return ``entry["last_attempt"]`` if it is well-formed enough to trust as a
    recorded failure - a dict with a truthy ``status`` and a parsable
    ``attempted_at`` - else ``None``.

    Both call sites (the N=1 cache-hit branch and :func:`_battery_fit_is_stale`)
    fall through to a re-fit when this returns ``None``, so a malformed
    last_attempt (missing, not a dict, no status, unparsable attempted_at -
    e.g. a hand-edited or truncated file) can never become a permanent no-fit
    trap.
    """
    if not isinstance(entry, dict):
        return None
    attempt = entry.get("last_attempt")
    if not isinstance(attempt, dict) or not attempt.get("status"):
        return None
    try:
        datetime.fromisoformat(attempt.get("attempted_at"))
    except (TypeError, ValueError):
        return None
    return attempt


def _record_battery_identification_failed_attempt(
    existing: dict | None, status: str, messages: list[str], sensors: dict | None = None
) -> dict:
    """
    Build the payload/slot to persist for a failed identification attempt.

    If ``existing`` is a retained ok record (a dict with ``status == "ok"``),
    every field is kept and ``last_attempt`` is added/replaced - a failed fit
    never destroys a retained ok record. Otherwise (missing, unreadable, or
    already non-ok) the result is a failure-only payload: ``schema_version``
    and ``last_attempt`` only, deliberately with NO top-level/slot-level
    ``status``, so it can never be mistaken for a fit result by any of the
    several places that gate on ``status == "ok"``.

    ``sensors`` is recorded inside ``last_attempt`` only when given. N>1
    callers pass the attempted ``(power, soc)`` pair - required so a later
    sensor-list edit can void the backoff (see _battery_fit_is_stale). The
    N=1 caller does not pass it: the N=1 cache-hit branch is deliberately
    sensor-indifferent (it never consults sensor_power_battery/
    sensor_battery_state_of_charge on a cache hit), so recording sensors
    there would be dead data.

    ``existing`` must already be known to be the SAME schema shape this
    caller expects (flat v1 at N=1, a v2 container slot at N>1): this
    function only distinguishes "ok" from "not ok" within that shape, it
    does not detect a foreign shape (e.g. a v2 container passed in at N=1).
    A foreign shape would fail the ``status == "ok"`` check and take the
    "no usable prior payload" branch, silently discarding it - so both call
    sites check for a foreign shape and skip the write entirely BEFORE
    calling this, rather than relying on this function to notice.
    """
    attempt = {
        "status": status,
        "attempted_at": datetime.now(UTC).isoformat(),
        "messages": messages,
    }
    if sensors is not None:
        attempt["sensors"] = sensors
    if isinstance(existing, dict) and existing.get("status") == "ok":
        new_payload = dict(existing)
    else:
        new_payload = {"schema_version": _BATTERY_ID_SCHEMA_VERSION_V1}
    new_payload["last_attempt"] = attempt
    return new_payload


def _battery_fit_is_stale(
    logger: logging.Logger,
    entry,
    current_power: str,
    current_soc: str,
    max_age_hours: float,
    k: int,
) -> bool:
    """
    Per-battery counterpart to :func:`is_model_outdated`: freshness comes from
    the stored entry's own timestamp fields, not the shared file's mtime, so
    one battery's fresh fit (or fresh recorded failure) can never suppress
    another battery's retry.

    ``entry`` is whatever the persisted container has under ``batteries[str(k)]``
    - possibly ``None`` (missing), possibly corrupt (hand-edited or from a
    future incompatible writer). This function must never raise, since it
    runs inside an eager per-k scan (`stale_ks`) computed before the main
    loop, and one corrupt entry must not abort every other battery's
    freshness read for the whole cycle.

    Not stale (a re-fit is skipped) iff, for the CURRENT ``(power, soc)`` pair:

    * the entry has a retained ``status == "ok"`` record fitted against this
      exact pair, within ``max_age_hours`` of its ``fitted_at``; OR
    * the entry has a well-formed ``last_attempt`` (a recorded failure, see
      :func:`_battery_identification_last_attempt`) made against this exact pair, within
      ``max_age_hours`` of its ``attempted_at`` - this is the backoff #1052
      adds: a battery whose fits keep failing retries at ``max_age_hours``
      cadence instead of re-fitting every run.

    Either branch requires the record's own ``sensors`` to match the
    currently resolved pair: missing (e.g. an entry written before this field
    existed) or mismatched (the lists were edited or reordered since the fit
    or attempt) both count as not-matching for that branch, so a cached ok
    result or a backed-off failure is never served for a different sensor
    pair than it was fitted/attempted against.

    ``max_age_hours <= 0`` is checked BEFORE either branch, so "0 means
    attempt every run" can never be defeated by a fresh last_attempt.

    A genuinely unusable entry (not a dict, or a dict with neither a usable
    ok record nor a well-formed last_attempt at all) logs a WARNING and is
    treated as stale. An entry that DOES have a usable ok record or
    last_attempt, just not a fresh-and-matching one right now (a sensor-pair
    change, ordinary expiry, or an unparsable/missing timestamp), logs at
    INFO instead - this includes a well-formed recorded failure simply still
    inside its backoff window, which is normal, expected operation, not a
    problem (see #1052: a WARNING firing every cycle for a backoff that is
    working as designed was the complaint this feature exists to fix).
    """
    label = f"Battery {k} identification model"
    if not isinstance(entry, dict):
        if entry is not None:
            logger.warning(
                f"{label} persisted entry is not a usable object "
                f"({type(entry).__name__}); will re-fit."
            )
        else:
            logger.info(f"{label} has no recorded fit, will train new model")
        return True

    if max_age_hours <= 0:
        logger.info(f"{label} max age is set to 0, forcing model re-fit")
        return True

    def _age_of(timestamp) -> timedelta | None:
        if not timestamp:
            return None
        try:
            dt = datetime.fromisoformat(timestamp)
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return datetime.now(UTC) - dt

    max_age = timedelta(hours=max_age_hours)

    if entry.get("status") == "ok" and _battery_id_sensors_match(
        entry.get("sensors"), current_power, current_soc
    ):
        age = _age_of(entry.get("fitted_at"))
        if age is not None and age <= max_age:
            logger.info(
                f"Using existing {label} (age: {age.total_seconds() / 3600:.1f}h, "
                f"max: {max_age_hours}h)"
            )
            return False

    attempt = _battery_identification_last_attempt(entry)
    if attempt is not None and _battery_id_sensors_match(
        attempt.get("sensors"), current_power, current_soc
    ):
        age = _age_of(attempt.get("attempted_at"))
        if age is not None and age <= max_age:
            logger.info(
                f"{label} recorded a failed attempt (status={attempt.get('status')}, "
                f"age: {age.total_seconds() / 3600:.1f}h, max: {max_age_hours}h); "
                "keeping configured battery values, backing off."
            )
            return False

    # Neither branch is fresh-and-matching: either a genuinely unusable entry
    # (warn) or a normal expiry / sensor-pair change / expired attempt (info).
    # The INFO wording deliberately doesn't claim WHICH of those it is - it
    # also covers a missing/unparsable fitted_at or attempted_at (_age_of
    # returned None above), which is neither "outdated" nor "re-bound".
    if entry.get("status") == "ok" or attempt is not None:
        logger.info(
            f"{label} has no fresh, matching record for the current sensor pair "
            "(outdated, re-bound, or an unreadable timestamp); will re-fit."
        )
    else:
        logger.warning(
            f"{label} persisted entry has status {entry.get('status')!r}, not 'ok', "
            "and no usable recorded attempt; will re-fit."
        )
    return True


def _log_battery_identification_summary(
    logger: logging.Logger, payload: dict, plant_conf: dict, k: int | None = None
) -> None:
    """
    Log the identified values and, in reported units, how they compare to config.

    k=None: N=1, wording byte-identical to master for a bare-string sensor
    config; a length-1-list config (e.g. the config UI's saved shape, see
    CONTRACT.md's SCOPE NOTE) reaches this with the same payload and produces
    the same wording either way. k=<int>: N>1, per-battery plant_conf reads
    via :func:`_batt_conf_val` and a "Battery {k} " infix, matching every
    other per-battery log line in this module (e.g. the guardrail-failure
    warning in the N>1 loop).
    """
    cap = payload.get("capacity_kwh", {})
    rte = payload.get("round_trip_efficiency", {})
    configured_cap_kwh = (
        float(_batt_conf_val(plant_conf.get("battery_nominal_energy_capacity", 0), k)) / 1000.0
    )
    configured_eta = float(_batt_conf_val(plant_conf.get("battery_charge_efficiency", 0), k))
    battery_tag = "" if k is None else f" {k}"
    logger.info(
        "Battery%s identification: capacity %.2f kWh (CI [%s, %s]) vs configured %.2f kWh; "
        "round-trip efficiency %.3f (one-way %.3f) vs configured one-way %.3f.",
        battery_tag,
        cap.get("value") or float("nan"),
        cap.get("ci_low"),
        cap.get("ci_high"),
        configured_cap_kwh,
        rte.get("value") or float("nan"),
        payload.get("eta_charge_symmetric") or float("nan"),
        configured_eta,
    )


def _log_battery_identification_recommendation(
    logger: logging.Logger, payload: dict, plant_conf: dict, k: int | None = None
) -> None:
    """
    Log a plain 'consider updating X from A to B' recommendation (suggest tier).

    k=None: N=1, log wording unchanged from master (independent of whether the
    sensor config is a bare string or a length-1 list). k=<int>: N>1,
    per-battery plant_conf reads via :func:`_batt_conf_val` and the same
    "Battery {k} " infix as :func:`_log_battery_identification_summary`.
    """
    cap = payload.get("capacity_kwh", {})
    identified_cap_kwh = cap.get("value")
    configured_cap_kwh = (
        float(_batt_conf_val(plant_conf.get("battery_nominal_energy_capacity", 0), k) or 0) / 1000.0
    )
    identified_eta = payload.get("eta_charge_symmetric")
    configured_eta = _batt_conf_val(plant_conf.get("battery_charge_efficiency"), k)
    battery_tag = "" if k is None else f" {k}"
    logger.info(
        "Battery%s identification recommendation: consider updating "
        "battery_nominal_energy_capacity from %.2f kWh to %.2f kWh, and "
        "battery_charge_efficiency / battery_discharge_efficiency from %s to %s "
        "(symmetric sqrt of the identified round-trip efficiency).",
        battery_tag,
        configured_cap_kwh,
        identified_cap_kwh if identified_cap_kwh is not None else float("nan"),
        configured_eta,
        identified_eta,
    )


async def _publish_battery_identification(
    rh: RetrieveHass, payload: dict, logger: logging.Logger, k: int | None = None
) -> None:
    """
    Publish the two read-only advisory sensors (suggest tier only).

    Attributes carry the confidence interval, sample counts, the last successful
    fit time, and the assumptions, so a user can judge trust from the sensor
    itself. ``fitted_at`` reflects the last SUCCESSFUL fit, never the publish time.

    k=None: N=1, exactly today's two fixed entity ids (zero new entities) -
    true for both a bare-string and a length-1-list sensor config, since
    publish only depends on ``k``, never on the sensor config itself.
    k=<int>: N>1, entity ids suffixed ``_battery<k>`` and friendly names
    suffixed ``Battery {k}``, mirroring the #1032 ``_publish_battery_data``
    per-battery convention (no separator before the digit).
    """
    cap = payload.get("capacity_kwh", {})
    rte = payload.get("round_trip_efficiency", {})
    common = {
        "fitted_at": payload.get("fitted_at"),
        "assumptions": payload.get("assumptions"),
        "n_charge_segments": payload.get("n_charge_segments"),
        "n_discharge_segments": payload.get("n_discharge_segments"),
    }
    cap_entity = (
        BATTERY_ID_CAPACITY_SENSOR if k is None else f"{BATTERY_ID_CAPACITY_SENSOR}_battery{k}"
    )
    rte_entity = BATTERY_ID_RTE_SENSOR if k is None else f"{BATTERY_ID_RTE_SENSOR}_battery{k}"
    name_suffix = "" if k is None else f" Battery {k}"
    await rh.post_scalar_sensor(
        cap_entity,
        cap.get("value"),
        {
            "friendly_name": f"Battery identified capacity{name_suffix}",
            "unit_of_measurement": "kWh",
            "device_class": "energy_storage",
            "ci_low": cap.get("ci_low"),
            "ci_high": cap.get("ci_high"),
            "method": cap.get("method"),
            "crosscheck_theil_sen_kwh": cap.get("crosscheck_theil_sen_kwh"),
            **common,
        },
    )
    await rh.post_scalar_sensor(
        rte_entity,
        rte.get("value"),
        {
            "friendly_name": f"Battery identified round-trip efficiency{name_suffix}",
            "one_way_efficiency_sqrt": payload.get("eta_charge_symmetric"),
            "ci_low": rte.get("ci_low"),
            "ci_high": rte.get("ci_high"),
            "crosscheck_energy_balance": rte.get("crosscheck_energy_balance"),
            **common,
        },
    )


async def identify_battery(
    logger: logging.Logger,
    optim_conf: dict,
    plant_conf: dict,
    retrieve_hass_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    get_data_from_file: bool,
    test_df_literal: str,
) -> None:
    """
    Opt-in battery self-identification (observe/suggest). Structural twin of
    :func:`adjust_pv_forecast`: cadence-gated on a persisted artifact, retrieves
    HA history only when the estimate is stale, and NEVER raises - any failure
    logs a warning and returns, leaving the configured battery values in force.

    v1 never touches ``plant_conf``. In the ``observe`` tier it writes the
    estimate to a JSON under ``data_path`` and logs it; in the ``suggest``
    tier it additionally publishes two read-only HA sensors at N=1, or 2N of
    them (one capacity + one round-trip-efficiency sensor per battery) at
    N>1.

    At ``number_of_batteries`` > 1 each battery is identified independently
    (own config reads, own retrieval columns, own persisted entry, own
    publish): one pack can fit and publish while another still lacks enough
    cycles. See :func:`_identify_battery_impl` for the per-battery loop.

    :param logger: Logger.
    :type logger: logging.Logger
    :param optim_conf: Optimization config (holds the three feature params).
    :type optim_conf: dict
    :param plant_conf: Plant config; read-only here, used for the sanity bound
        and the "configured vs identified" comparison.
    :type plant_conf: dict
    :param retrieve_hass_conf: Retrieve config (holds sensor_power_battery and
        sensor_battery_state_of_charge - a bare entity-id string at N=1, or a
        list of ``number_of_batteries`` entity ids at N>1). At N>1,
        ``_identify_battery_impl`` temporarily mutates these two keys in place
        for the duration of the retrieval ``await`` and restores the original
        values in a ``finally``, so this dict is briefly not what the caller
        put in it if this coroutine is inspected concurrently.
    :type retrieve_hass_conf: dict
    :param rh: RetrieveHass instance.
    :type rh: RetrieveHass
    :param emhass_conf: emhass paths.
    :type emhass_conf: dict
    :param get_data_from_file: Whether history comes from a file instead of HA.
    :type get_data_from_file: bool
    :param test_df_literal: Test data filename for file mode.
    :type test_df_literal: str
    :rtype: None
    """
    # Never-raise boundary: this is an advisory side-feature and must never be
    # able to break an optimization run, so ANY unexpected error is swallowed
    # with a warning, leaving the configured battery values in force.
    try:
        await _identify_battery_impl(
            logger,
            optim_conf,
            plant_conf,
            retrieve_hass_conf,
            rh,
            emhass_conf,
            get_data_from_file,
            test_df_literal,
        )
    except Exception as e:
        logger.warning(
            f"Battery identification failed unexpectedly ({type(e).__name__}: {e}); "
            "keeping configured battery values.",
            exc_info=True,
        )


async def _identify_battery_impl(
    logger: logging.Logger,
    optim_conf: dict,
    plant_conf: dict,
    retrieve_hass_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    get_data_from_file: bool,
    test_df_literal: str,
) -> None:
    """Implementation of :func:`identify_battery`; wrapped for the never-raise guarantee."""
    num_batteries = utils.validate_num_batteries(plant_conf)

    data_path = pathlib.Path(emhass_conf["data_path"])
    json_path = data_path / "battery_identification.json"
    max_age_hours = optim_conf.get("battery_identification_model_max_age", 24)
    tier = optim_conf.get("battery_identification_trust_tier", "observe")

    if num_batteries == 1:
        # Flat v1 shape, same as master for a bare-string sensor config (a
        # length-1-list config, e.g. the config UI's saved shape, takes the
        # equivalent list-handling path further below - see CONTRACT.md's
        # SCOPE NOTE on invariant 1). In particular: the freshness gate and
        # cached publish come FIRST, exactly like base, WITHOUT ever
        # consulting sensor_power_battery/sensor_battery_state_of_charge -
        # base only read those after a successful retrieval, on the re-fit
        # path, so a cache hit must stay indifferent to whatever (even
        # malformed) shape those two keys happen to be in.
        if not is_model_outdated(
            json_path, max_age_hours, logger, label="Battery identification model"
        ):
            try:
                with open(json_path, "rb") as inp:
                    payload = json.loads(inp.read())
            except (KeyError, ValueError, OSError) as e:
                logger.warning(
                    f"Battery identification result unreadable ({type(e).__name__}); will re-fit."
                )
                payload = None
            if payload is not None and payload.get("status") == "ok":
                _log_battery_identification_summary(logger, payload, plant_conf)
                if tier == "suggest":
                    _log_battery_identification_recommendation(logger, payload, plant_conf)
                    await _publish_battery_identification(rh, payload, logger)
                return
            attempt = _battery_identification_last_attempt(payload)
            if attempt is not None:
                # A fresh, well-formed recorded failure: back off (retry at
                # max_age cadence, per #1052) instead of re-fitting every run.
                # Logged at INFO - this is expected operation, not a problem.
                logger.info(
                    f"Battery identification: recorded attempt at {attempt.get('attempted_at')} "
                    f"failed guardrails (status={attempt.get('status')}); keeping configured "
                    "battery values, backing off until max_age."
                )
                return
            # Fall through to a re-fit on an unreadable, non-ok payload with
            # no (or malformed) last_attempt (also covers a v2 container left
            # over from a reverted N>1 run: it has no top-level "status", so
            # it reads as not-ok here). A malformed last_attempt must fall
            # through too, so a hand-edited or truncated file can never
            # become a permanent no-fit trap.

        # Refit path only: resolve the sensor keys now, matching where base
        # first read power_col (after a successful retrieval was decided on,
        # never on the cache-hit path above).
        resolved = _resolve_battery_sensor_lists(retrieve_hass_conf, num_batteries, logger)
        if resolved is None:
            return
        power_list, soc_list = resolved

        logger.info("Battery identification: retrieving history for a fresh fit")
        success, df, _ = await retrieve_home_assistant_data(
            "battery_id",
            get_data_from_file,
            retrieve_hass_conf,
            optim_conf,
            rh,
            emhass_conf,
            test_df_literal,
            logger,
        )
        if not success or df is None:
            logger.warning(
                "Battery identification: could not retrieve history; keeping configured battery values."
            )
            return
        power_col = power_list[0]
        soc_col = soc_list[0]
        if power_col not in df.columns or soc_col not in df.columns:
            logger.warning(
                f"Battery identification: sensors '{power_col}'/'{soc_col}' missing from retrieved "
                "history; keeping configured battery values."
            )
            return
        configured_capacity_wh = float(plant_conf.get("battery_nominal_energy_capacity", 0) or 0)
        result = BatteryIdentification(
            logger, time_step=retrieve_hass_conf.get("optimization_time_step")
        ).identify(df, power_col, soc_col, configured_capacity_wh)
        for msg in result.messages:
            logger.info("Battery identification: %s", msg)
        if not result.is_ok:
            # Record the failed attempt via a read-modify-write of the file: a
            # retained ok payload keeps every field and gains last_attempt; a
            # missing/unreadable/already-non-ok payload becomes a
            # failure-only payload (see
            # _record_battery_identification_failed_attempt). This DOES bump
            # the file mtime - that is now the intended mechanism: it is what
            # makes is_model_outdated (and the cache-hit branch above) back
            # off retrying for max_age_hours instead of re-fitting every run
            # (#1052).
            #
            # Read-then-write, not a single atomic transaction: a concurrent
            # writer (e.g. a dayahead call racing an MPC cron, both fitting
            # this same battery) could complete its own write in the
            # millisecond window between this read and _atomic_json_write
            # below, and get silently overwritten. Same lost-update class the
            # container writer's docstring already accepts for N>1; self-
            # heals within max_age_hours. Not worth file locking for.
            existing = _read_raw_battery_identification_json(json_path)
            existing_batteries = existing.get("batteries") if isinstance(existing, dict) else None
            if isinstance(existing_batteries, dict) and any(
                isinstance(slot, dict) and slot.get("status") == "ok"
                for slot in existing_batteries.values()
            ):
                # Foreign shape WORTH PROTECTING: a v2 (N>1) container left on
                # disk (e.g. after reverting number_of_batteries from >1 back
                # to 1) that still holds at least one retained ok slot.
                # _record_battery_identification_failed_attempt only knows
                # how to retain a flat v1 "ok" record; handing it this
                # container would fail its status=="ok" check and take the
                # "no usable prior payload" branch, silently discarding every
                # retained per-battery slot (invariant 2 violation across an
                # N-transition - a failed fit must never destroy a retained
                # ok record, not even one in the "wrong" shape). Skip the
                # write entirely and retry every run instead, exactly like
                # master's failed-fit-writes-nothing behaviour, until either a
                # successful N=1 fit overwrites the file or the leftover file
                # is deleted.
                #
                # Deliberately narrower than "any v2 container": a container
                # whose slots are ALL failure-only has nothing to protect, so
                # skipping here would just re-create the every-run re-pull
                # #1052 exists to fix, with no self-clearing path (only a
                # successful N=1 fit removes the leftover shape, and that
                # can't happen while every N=1 fit keeps failing). Proceed to
                # the normal failure write below in that case - it discards
                # the empty container and starts a fresh N=1 backoff.
                logger.info(
                    "Battery identification: existing file has the multi-battery "
                    "layout with a retained estimate; not recording the attempt, "
                    "will retry next run."
                )
                return
            new_payload = _record_battery_identification_failed_attempt(
                existing, result.status, result.messages
            )
            _atomic_json_write(json_path, new_payload)
            logger.warning(
                f"Battery identification did not pass guardrails (status={result.status}); "
                "keeping configured battery values and recording the failed attempt. Any "
                "previously stored estimate is kept."
            )
            return

        # Persist a successful estimate, atomically; this replaces the whole
        # file (including any previously recorded last_attempt), which is how
        # a success after a run of failures clears the backoff record.
        payload = result.to_dict()
        payload["fitted_at"] = datetime.now(UTC).isoformat()
        payload["trust_tier"] = tier
        payload["configured_at_fit_time"] = {
            "battery_nominal_energy_capacity": plant_conf.get("battery_nominal_energy_capacity"),
            "battery_charge_efficiency": plant_conf.get("battery_charge_efficiency"),
            "battery_discharge_efficiency": plant_conf.get("battery_discharge_efficiency"),
        }
        _atomic_json_write(json_path, payload)

        _log_battery_identification_summary(logger, payload, plant_conf)
        if tier == "suggest":
            _log_battery_identification_recommendation(logger, payload, plant_conf)
            await _publish_battery_identification(rh, payload, logger)
        return

    # N > 1: schema_version=2 container, one entry per battery. Each battery is
    # independent end to end (own freshness, own fit, own persisted entry, own
    # publish) under the one global trust tier; a v1 flat file left over from a
    # reverted N=1 run is treated as absent (re-fit every battery), never
    # partially parsed. Resolver-first here (unlike N=1): the mutation below
    # needs the resolved lists before any retrieval, and the sensor pair is
    # also an input to the per-battery freshness check.
    resolved = _resolve_battery_sensor_lists(retrieve_hass_conf, num_batteries, logger)
    if resolved is None:
        # Warning already logged by the resolver, naming the offending key.
        return
    power_list, soc_list = resolved

    container = _load_battery_identification_container(json_path, logger)
    batteries = container["batteries"]
    stale_ks = [
        k
        for k in range(num_batteries)
        if _battery_fit_is_stale(
            logger, batteries.get(str(k)), power_list[k], soc_list[k], max_age_hours, k
        )
    ]

    df = None
    if stale_ks:
        # One batched retrieval covering only the currently-stale batteries'
        # sensors, not the full lists: the per-k loop below only ever reads
        # power_list[k]/soc_list[k] for a stale k, so an already-fresh
        # battery's columns would just be fetched and never read. Restricting
        # the mutation to the stale subset means a fresh battery's sensor
        # going away (or lacking history) can never affect a stale sibling's
        # re-fit, and an unreachable sensor among the stale batteries defers
        # only those batteries, not the fresh ones (which are never part of
        # this batch). _retrieve_from_hass's append site reads
        # sensor_power_battery/sensor_battery_state_of_charge straight off
        # retrieve_hass_conf; presenting the resolved subset here - restored
        # immediately after - is the narrowest way to get the stale
        # batteries' entity ids into one var_list without threading a new
        # argument through retrieve_home_assistant_data/_retrieve_from_hass,
        # which are shared with every other set_type.
        logger.info("Battery identification: retrieving history for a fresh fit")
        stale_power = [power_list[k] for k in stale_ks]
        stale_soc = [soc_list[k] for k in stale_ks]
        original_power = retrieve_hass_conf.get("sensor_power_battery")
        original_soc = retrieve_hass_conf.get("sensor_battery_state_of_charge")
        retrieve_hass_conf["sensor_power_battery"] = stale_power
        retrieve_hass_conf["sensor_battery_state_of_charge"] = stale_soc
        try:
            success, df, _ = await retrieve_home_assistant_data(
                "battery_id",
                get_data_from_file,
                retrieve_hass_conf,
                optim_conf,
                rh,
                emhass_conf,
                test_df_literal,
                logger,
            )
        finally:
            retrieve_hass_conf["sensor_power_battery"] = original_power
            retrieve_hass_conf["sensor_battery_state_of_charge"] = original_soc
        if not success:
            df = None

    for k in range(num_batteries):
        if k not in stale_ks:
            # Not stale, per _battery_fit_is_stale - but that can now mean
            # either a fresh ok record OR a fresh, well-formed backed-off
            # last_attempt: a slot can hold a retained ok record fitted
            # against an OLD sensor pair plus a fresh last_attempt against
            # the CURRENTLY resolved pair (e.g. after a sensor-list edit
            # followed by a failed re-fit), which _battery_fit_is_stale
            # correctly treats as not-stale. Only an actual ok record for
            # THIS battery's currently resolved sensor pair is ever published
            # or summarised; a backed-off failure-only entry (or an ok record
            # whose own sensors no longer match the current pair) is logged
            # at info and skipped, never mistaken for a publishable result -
            # this is what stops the old pair's capacity ever being served as
            # the new pair's.
            entry = batteries[str(k)]
            entry_matches_current_pair = _battery_id_sensors_match(
                entry.get("sensors") if isinstance(entry, dict) else None,
                power_list[k],
                soc_list[k],
            )
            if (
                isinstance(entry, dict)
                and entry.get("status") == "ok"
                and entry_matches_current_pair
            ):
                _log_battery_identification_summary(logger, entry, plant_conf, k=k)
                if tier == "suggest":
                    _log_battery_identification_recommendation(logger, entry, plant_conf, k=k)
                    await _publish_battery_identification(rh, entry, logger, k=k)
            else:
                logger.info(
                    f"Battery {k} identification: recorded attempt is backed off; "
                    "keeping configured battery values, nothing to publish."
                )
            continue
        if df is None:
            logger.warning(
                f"Battery {k} identification: could not retrieve history; "
                "keeping configured battery values."
            )
            continue
        power_col = power_list[k]
        soc_col = soc_list[k]
        if power_col not in df.columns or soc_col not in df.columns:
            logger.warning(
                f"Battery {k} identification: sensors '{power_col}'/'{soc_col}' missing from "
                "retrieved history; keeping configured battery values."
            )
            continue
        configured_capacity_wh = float(
            _batt_conf_val(plant_conf.get("battery_nominal_energy_capacity", 0), k) or 0
        )
        result = BatteryIdentification(
            logger, time_step=retrieve_hass_conf.get("optimization_time_step")
        ).identify(df, power_col, soc_col, configured_capacity_wh)
        for msg in result.messages:
            logger.info("Battery %d identification: %s", k, msg)
        if not result.is_ok:
            # Record the failed attempt in battery k's slot only, via the
            # same read-modify-write container helper the success path uses
            # below (_write_battery_identification_container): an entry with
            # status == "ok" keeps every field and gains last_attempt (with
            # "sensors" recording the pair this attempt was against - unlike
            # N=1, see _record_battery_identification_failed_attempt);
            # otherwise the slot becomes a failure-only payload with no
            # slot-level "status", mirroring the N=1 shape. No other
            # battery's entry is touched.
            #
            # Read-then-write, not a single atomic transaction: same narrow,
            # self-healing lost-update window as the N=1 failure write and
            # the container writer's own documented concurrency note (a
            # concurrent successful write for this exact battery landing
            # inside this window could be overwritten; corrects itself within
            # max_age_hours). Not worth file locking for.
            #
            # Read the RAW file first (bypassing the v2-container loader,
            # which would silently normalise a foreign shape to an empty
            # container): a flat v1 payload left on disk (e.g. after raising
            # number_of_batteries from 1 to >1) must not be discarded by a
            # FAILURE the way the success path's existing treat-v1-as-absent
            # normalisation already discards it on success (invariant 1,
            # unchanged) - a failed fit destroying a retained ok record
            # would violate invariant 2, even across an N-transition.
            raw_existing = _read_raw_battery_identification_json(json_path)
            if (
                isinstance(raw_existing, dict)
                and "status" in raw_existing
                and "batteries" not in raw_existing
            ):
                logger.info(
                    f"Battery {k} identification: existing file has the single-battery "
                    "layout; not recording the attempt, will retry next run."
                )
                continue
            prior = _load_battery_identification_container(json_path, logger)["batteries"].get(
                str(k)
            )
            failed_entry = _record_battery_identification_failed_attempt(
                prior, result.status, result.messages, sensors={"power": power_col, "soc": soc_col}
            )
            _write_battery_identification_container(json_path, logger, k, failed_entry)
            logger.warning(
                f"Battery {k} identification did not pass guardrails (status={result.status}); "
                "keeping configured battery values and recording the failed attempt. Any "
                "previously stored estimate is kept."
            )
            continue

        # Persist a successful estimate for battery k, read-modify-write,
        # atomic; this replaces battery k's whole slot (including any
        # previously recorded last_attempt), which is how a success after a
        # run of failures clears the backoff record. A failed fit for battery
        # k (above) never touches battery k's (or any other battery's)
        # previously stored entry. "sensors" binds this entry to the exact
        # pair it was fitted from, so a later run that edits or reorders the
        # lists can never serve it as a stale-free cache hit for the wrong
        # sensor pair (see _battery_fit_is_stale).
        new_entry = result.to_dict()
        new_entry["fitted_at"] = datetime.now(UTC).isoformat()
        new_entry["trust_tier"] = tier
        new_entry["sensors"] = {"power": power_col, "soc": soc_col}
        new_entry["configured_at_fit_time"] = {
            "battery_nominal_energy_capacity": _batt_conf_val(
                plant_conf.get("battery_nominal_energy_capacity"), k
            ),
            "battery_charge_efficiency": _batt_conf_val(
                plant_conf.get("battery_charge_efficiency"), k
            ),
            "battery_discharge_efficiency": _batt_conf_val(
                plant_conf.get("battery_discharge_efficiency"), k
            ),
        }
        _write_battery_identification_container(json_path, logger, k, new_entry)

        _log_battery_identification_summary(logger, new_entry, plant_conf, k=k)
        if tier == "suggest":
            _log_battery_identification_recommendation(logger, new_entry, plant_conf, k=k)
            await _publish_battery_identification(rh, new_entry, logger, k=k)


async def _prepare_perfect_optim(ctx: SetupContext):
    """Helper to prepare data for perfect optimization."""
    success, df_input_data, days_list = await retrieve_home_assistant_data(
        "perfect-optim",
        ctx.get_data_from_file,
        ctx.retrieve_hass_conf,
        ctx.optim_conf,
        ctx.rh,
        ctx.emhass_conf,
        test_df_literal,
        ctx.logger,
    )
    if not success:
        return None
    return {
        "df_input_data": df_input_data,
        "days_list": days_list,
    }


async def _get_dayahead_pv_forecast(ctx: SetupContext):
    """Helper to retrieve and optionally adjust PV forecast."""
    # Check if we should calculate PV forecast
    if not (
        ctx.optim_conf["set_use_pv"]
        or ctx.optim_conf.get("weather_forecast_method", None) == "list"
    ):
        return pd.Series(0, index=ctx.fcst.forecast_dates), None
    # Get weather forecast
    df_weather = await ctx.fcst.get_weather_forecast(
        method=ctx.optim_conf["weather_forecast_method"]
    )
    if isinstance(df_weather, bool) and not df_weather:
        return None, None
    p_pv_forecast = ctx.fcst.get_power_from_weather(df_weather)
    # Adjust PV forecast if needed
    if ctx.optim_conf["set_use_adjusted_pv"]:
        p_pv_forecast = await adjust_pv_forecast(
            ctx.logger,
            ctx.fcst,
            p_pv_forecast,
            ctx.get_data_from_file,
            ctx.retrieve_hass_conf,
            ctx.optim_conf,
            ctx.rh,
            ctx.emhass_conf,
            test_df_literal,
        )
    return p_pv_forecast, df_weather


def _apply_df_freq_horizon(
    df: pd.DataFrame, retrieve_hass_conf: dict, prediction_horizon: int | None
) -> pd.DataFrame:
    """Helper to apply frequency adjustment and prediction horizon slicing."""
    # Handle Frequency
    if retrieve_hass_conf.get("optimization_time_step"):
        step = retrieve_hass_conf["optimization_time_step"]
        if not isinstance(step, pd._libs.tslibs.timedeltas.Timedelta):
            step = pd.to_timedelta(step, "minute")
        df = df[~df.index.duplicated(keep="last")]
        df = df.asfreq(step)
    else:
        df = utils.set_df_index_freq(df)
    # Handle Prediction Horizon
    if prediction_horizon:
        # Slice the dataframe up to the horizon
        df = copy.deepcopy(df)[df.index[0] : df.index[min(prediction_horizon, len(df)) - 1]]
    return df


async def _prepare_dayahead_optim(ctx: SetupContext, stage_times: dict | None = None):
    """Helper to prepare data for day-ahead optimization.

    :param stage_times: Optional dict to record per-stage elapsed times (seconds).
    :type stage_times: dict, optional
    """
    if stage_times is None:
        stage_times = {}
    # Get PV Forecast
    with stage_timer(stage_times, "pv_forecast", ctx.logger):
        p_pv_forecast, df_weather = await _get_dayahead_pv_forecast(ctx)
    if p_pv_forecast is None:
        return None
    # Get Load Forecast
    with stage_timer(stage_times, "load_forecast", ctx.logger):
        p_load_forecast = await ctx.fcst.get_load_forecast(
            days_min_load_forecast=ctx.optim_conf["delta_forecast_daily"].days,
            method=ctx.optim_conf["load_forecast_method"],
        )
    if isinstance(p_load_forecast, bool) and not p_load_forecast:
        ctx.logger.error("Unable to get load forecast.")
        return None
    # Build Input DataFrame
    df_input_data_dayahead = pd.DataFrame(
        np.transpose(np.vstack([p_pv_forecast.values, p_load_forecast.values])),
        index=p_pv_forecast.index,
        columns=["p_pv_forecast", "p_load_forecast"],
    )
    # Apply Frequency and Prediction Horizon
    # Use explicitly passed horizon, avoiding JSON re-parsing
    prediction_horizon = ctx.params["passed_data"].get("prediction_horizon")
    df_input_data_dayahead = _apply_df_freq_horizon(
        df_input_data_dayahead, ctx.retrieve_hass_conf, prediction_horizon
    )
    return {
        "df_input_data_dayahead": df_input_data_dayahead,
        "df_weather": df_weather,
        "p_pv_forecast": p_pv_forecast,
        "p_load_forecast": p_load_forecast,
    }


async def _get_naive_mpc_history(ctx: SetupContext):
    """Helper to retrieve historical data for Naive MPC."""
    # Check if we need to skip historical data retrieval
    is_list_forecast = ctx.optim_conf.get("load_forecast_method") == "list"
    is_list_weather = ctx.optim_conf.get("weather_forecast_method") == "list"
    no_pv = not ctx.optim_conf["set_use_pv"]

    if (is_list_forecast and is_list_weather) or (is_list_forecast and no_pv):
        return True, None, None, False  # success, df, days_list, set_mix_forecast
    # Retrieve data from Home Assistant
    success, df_input_data, days_list = await retrieve_home_assistant_data(
        "naive-mpc-optim",
        ctx.get_data_from_file,
        ctx.retrieve_hass_conf,
        ctx.optim_conf,
        ctx.rh,
        ctx.emhass_conf,
        test_df_literal,
        ctx.logger,
    )
    return success, df_input_data, days_list, True


async def _get_naive_mpc_pv_forecast(ctx: SetupContext, set_mix_forecast, df_input_data):
    """Helper to generate PV forecast for Naive MPC."""
    # If PV is disabled and no weather list, return zero series
    if not (
        ctx.optim_conf["set_use_pv"] or ctx.optim_conf.get("weather_forecast_method") == "list"
    ):
        return pd.Series(0, index=ctx.fcst.forecast_dates), None
    # Get weather forecast
    df_weather = await ctx.fcst.get_weather_forecast(
        method=ctx.optim_conf["weather_forecast_method"]
    )
    if isinstance(df_weather, bool) and not df_weather:
        return None, None
    # Calculate PV power
    p_pv_forecast = ctx.fcst.get_power_from_weather(
        df_weather, set_mix_forecast=set_mix_forecast, df_now=df_input_data
    )
    # Adjust PV forecast if needed
    if ctx.optim_conf["set_use_adjusted_pv"]:
        p_pv_forecast = await adjust_pv_forecast(
            ctx.logger,
            ctx.fcst,
            p_pv_forecast,
            ctx.get_data_from_file,
            ctx.retrieve_hass_conf,
            ctx.optim_conf,
            ctx.rh,
            ctx.emhass_conf,
            test_df_literal,
        )
    return p_pv_forecast, df_weather


async def _prepare_naive_mpc_optim(ctx: SetupContext, stage_times: dict | None = None):
    """Helper to prepare data for Naive MPC optimization.

    :param stage_times: Optional dict to record per-stage elapsed times (seconds).
    :type stage_times: dict, optional
    """
    if stage_times is None:
        stage_times = {}
    # Retrieve Historical Data
    success, df_input_data, days_list, set_mix_forecast = await _get_naive_mpc_history(ctx)
    if not success:
        return None
    # Get PV Forecast
    with stage_timer(stage_times, "pv_forecast", ctx.logger):
        p_pv_forecast, df_weather = await _get_naive_mpc_pv_forecast(
            ctx, set_mix_forecast, df_input_data
        )
    if p_pv_forecast is None:
        return None
    # Get Load Forecast
    with stage_timer(stage_times, "load_forecast", ctx.logger):
        p_load_forecast = await ctx.fcst.get_load_forecast(
            days_min_load_forecast=ctx.optim_conf["delta_forecast_daily"].days,
            method=ctx.optim_conf["load_forecast_method"],
            set_mix_forecast=set_mix_forecast,
            df_now=df_input_data,
        )
    if isinstance(p_load_forecast, bool) and not p_load_forecast:
        return None
    # Build and Format Input DataFrame
    df_input_data_dayahead = pd.concat([p_pv_forecast, p_load_forecast], axis=1)
    df_input_data_dayahead.columns = ["p_pv_forecast", "p_load_forecast"]
    # Reuse freq/horizon helper
    prediction_horizon = ctx.params["passed_data"].get("prediction_horizon")
    df_input_data_dayahead = _apply_df_freq_horizon(
        df_input_data_dayahead, ctx.retrieve_hass_conf, prediction_horizon
    )
    return {
        "df_input_data": df_input_data,
        "days_list": days_list,
        "df_input_data_dayahead": df_input_data_dayahead,
        "df_weather": df_weather,
        "p_pv_forecast": p_pv_forecast,
        "p_load_forecast": p_load_forecast,
    }


async def _prepare_ml_fit_predict(ctx: SetupContext):
    """Helper to prepare data for ML fit/predict/tune."""
    days_to_retrieve = ctx.params["passed_data"]["historic_days_to_retrieve"]
    model_type = ctx.params["passed_data"]["model_type"]
    var_model = ctx.params["passed_data"]["var_model"]
    if ctx.get_data_from_file:
        filename = model_type + ".pkl"
        filename_path = ctx.emhass_conf["data_path"] / filename
        async with aiofiles.open(filename_path, "rb") as inp:
            content = await inp.read()
            df_input_data, _, _, _ = pickle.loads(content)
        df_input_data = df_input_data[df_input_data.index[-1] - pd.offsets.Day(days_to_retrieve) :]
        return {"df_input_data": df_input_data}
    else:
        days_list = utils.get_days_list(days_to_retrieve)
        var_list = [var_model]
        if not await ctx.rh.get_data(days_list, var_list):
            return None
        ctx.rh.prepare_data(
            var_model,
            load_negative=ctx.retrieve_hass_conf.get("load_negative", False),
            set_zero_min=ctx.retrieve_hass_conf.get("set_zero_min", True),
            var_replace_zero=ctx.retrieve_hass_conf.get("sensor_replace_zero", []),
            var_interp=ctx.retrieve_hass_conf.get("sensor_linear_interp", []),
            skip_renaming=True,
        )
        return {"df_input_data": ctx.rh.df_final.copy()}


def _prepare_regressor_fit(ctx: SetupContext):
    """Helper to prepare data for Regressor fit/predict."""
    csv_file = ctx.params["passed_data"].get("csv_file", None)
    if not csv_file:
        ctx.logger.error("csv_file is required for regressor actions but was not provided.")
        return None
    if ctx.get_data_from_file:
        base_path = ctx.emhass_conf["data_path"]
        filename_path = pathlib.Path(base_path) / csv_file
    else:
        filename_path = ctx.emhass_conf["data_path"] / csv_file
    if filename_path.is_file():
        df_input_data = pd.read_csv(filename_path, parse_dates=True)
    else:
        ctx.logger.error(
            f"The CSV file {csv_file} was not found in path: {ctx.emhass_conf['data_path']}"
        )
        return None
    # Validate columns
    required_columns = []
    if "features" in ctx.params["passed_data"]:
        required_columns.extend(ctx.params["passed_data"]["features"])
    if "target" in ctx.params["passed_data"]:
        required_columns.append(ctx.params["passed_data"]["target"])
    if "timestamp" in ctx.params["passed_data"]:
        required_columns.append(ctx.params["passed_data"]["timestamp"])
    if not set(required_columns).issubset(df_input_data.columns):
        ctx.logger.error(
            f"The csv file does not contain the required columns: {', '.join(required_columns)}"
        )
        return None
    return {"df_input_data": df_input_data}


async def set_input_data_dict(
    emhass_conf: dict,
    costfun: str,
    params: str,
    runtimeparams: str,
    set_type: str,
    logger: logging.Logger,
    get_data_from_file: bool | None = False,
) -> dict:
    """
    Set up some of the data needed for the different actions.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param costfun: The type of cost function to use for optimization problem
    :type costfun: str
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param runtimeparams: Runtime optimization parameters passed as a dictionary
    :type runtimeparams: str
    :param set_type: Set the type of setup based on following type of optimization
    :type set_type: str
    :param logger: The passed logger object
    :type logger: logging object
    :param get_data_from_file: Use data from saved CSV file (useful for debug)
    :type get_data_from_file: bool, optional
    :return: A dictionnary with multiple data used by the action functions
    :rtype: dict

    """
    stage_times = {}
    logger.info("Setting up needed data")
    # Parse Parameters
    if (params is not None) and (params != "null"):
        if isinstance(params, str):
            params = dict(orjson.loads(params))
    else:
        params = {}
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
    if type(retrieve_hass_conf) is bool:
        return False
    (
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
    ) = await utils.treat_runtimeparams(
        runtimeparams,
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        set_type,
        logger,
        emhass_conf,
    )
    log_runtime_banner(logger, optim_conf=optim_conf)
    if isinstance(params, str):
        params = dict(orjson.loads(params))
    # Initialize Core Objects
    rh = RetrieveHass(
        retrieve_hass_conf["hass_url"],
        retrieve_hass_conf["long_lived_token"],
        retrieve_hass_conf["optimization_time_step"],
        retrieve_hass_conf["time_zone"],
        params,
        emhass_conf,
        logger,
        get_data_from_file=get_data_from_file,
    )
    # Retrieve HA config
    if get_data_from_file:
        async with aiofiles.open(emhass_conf["data_path"] / test_df_literal, "rb") as inp:
            content = await inp.read()
            _, _, _, rh.ha_config = pickle.loads(content)
    elif not await rh.get_ha_config():
        return False
    if isinstance(params, dict):
        params_str = orjson.dumps(params).decode("utf-8")
        params = utils.update_params_with_ha_config(params_str, rh.ha_config)
    else:
        params = utils.update_params_with_ha_config(params, rh.ha_config)
    if isinstance(params, str):
        params = dict(orjson.loads(params))
    costfun = optim_conf.get("costfun", costfun)
    # Two-tier guard:
    #   - actions_without_fcst_or_opt: read saved results only; build neither.
    #   - actions_skip_optim_cache: need a Forecast object but no Optimization.
    #     Keeping these out of the OptimizationCache path stops them poisoning
    #     the cache key with config-default values that a subsequent
    #     naive-mpc-optim call would then miss against.
    actions_without_fcst_or_opt = ["publish-data", "export-influxdb-to-csv"]
    actions_skip_optim_cache = [
        "forecast-model-fit",
        "forecast-model-predict",
        "forecast-model-tune",
        "forecast-calibration",
    ]
    if set_type in actions_without_fcst_or_opt:
        fcst = None
        opt = None
        logger.debug(f"Skipping Optimization creation for action: {set_type}")
    else:
        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params,
            emhass_conf,
            logger,
            get_data_from_file=get_data_from_file,
        )
        if set_type in actions_skip_optim_cache:
            opt = None
            logger.debug(f"Skipping OptimizationCache for action: {set_type}")
        else:
            # Try to get cached Optimization object for warm-starting
            _num_ts = len(fcst.forecast_dates)
            opt = OptimizationCache.get(
                optim_conf, plant_conf, costfun, retrieve_hass_conf, logger, _num_ts
            )
            if opt is None:
                # Cache miss - create new Optimization object
                opt = Optimization(
                    retrieve_hass_conf,
                    optim_conf,
                    plant_conf,
                    fcst.var_load_cost,
                    fcst.var_prod_price,
                    costfun,
                    emhass_conf,
                    logger,
                    num_timesteps=_num_ts,
                )
                # Store in cache for future warm-starts
                OptimizationCache.put(
                    opt, optim_conf, plant_conf, costfun, retrieve_hass_conf, logger, _num_ts
                )
            else:
                # Cache hit - update references that may have changed
                # (logger, var names from forecast, and runtime-configurable optim_conf values)
                opt.logger = logger
                opt.var_load_cost = fcst.var_load_cost
                opt.var_prod_price = fcst.var_prod_price
                # Update internal config dictionaries to prevent stale lookups
                # for runtime parameters (like battery_target_state_of_charge)
                opt.plant_conf = plant_conf
                opt.optim_conf = optim_conf
                # Update CVXPY Parameters for thermal start temperatures
                # This is critical: updating optim_conf alone doesn't change baked-in constraint values
                opt.update_thermal_start_temps(optim_conf)
                # Same idea for battery power limits — they participate in
                # constraints via cp.Parameter and need the runtime value
                # propagated even on a cache hit.
                opt.update_battery_power_limits(plant_conf)
            # Update runtime-configurable solver options from optim_conf
            # These don't affect problem structure, so they're safe to update on cached object
            runtime_solver_opts = [
                "lp_solver_timeout",
                "lp_solver_mip_rel_gap",
                "num_threads",
            ]
            for key in runtime_solver_opts:
                if key in optim_conf:
                    opt.optim_conf[key] = optim_conf[key]
    # Create SetupContext
    ctx = SetupContext(
        retrieve_hass_conf=retrieve_hass_conf,
        optim_conf=optim_conf,
        plant_conf=plant_conf,
        emhass_conf=emhass_conf,
        params=params,
        logger=logger,
        get_data_from_file=get_data_from_file,
        rh=rh,
        fcst=fcst,
    )
    # Initialize Default Return Data
    data_results = {
        "df_input_data": None,
        "df_input_data_dayahead": None,
        "df_weather": None,
        "p_pv_forecast": None,
        "p_load_forecast": None,
        "days_list": None,
    }
    # Delegate to Helpers based on set_type
    result = None
    if set_type == "dayahead-optim":
        # Dayahead uses granular per-stage timing inside _prepare_dayahead_optim;
        # no coarse outer wrap here to avoid double-counting.
        result = await _prepare_dayahead_optim(ctx, stage_times=stage_times)
    elif set_type == "perfect-optim":
        # Perfect uses historical HA data; no input_data stage timing —
        # price_prep / optim_solve / publish are timed inside perfect_forecast_optim.
        result = await _prepare_perfect_optim(ctx)
    elif set_type == "naive-mpc-optim":
        # Naive MPC uses granular per-stage timing inside _prepare_naive_mpc_optim;
        # no coarse outer wrap here to avoid double-counting.
        result = await _prepare_naive_mpc_optim(ctx, stage_times=stage_times)
    elif set_type in ["forecast-model-fit", "forecast-model-predict", "forecast-model-tune"]:
        result = await _prepare_ml_fit_predict(ctx)
    elif set_type == "forecast-calibration":
        # The calibration action retrieves its own (longer) history window inside
        # forecast_calibration(); no ML-prep here.
        result = {}
    elif set_type == "regressor-model-fit":
        result = _prepare_regressor_fit(ctx)
    elif set_type == "regressor-model-predict":
        if get_data_from_file:
            result = _prepare_regressor_fit(ctx)
        else:
            result = {}
    elif set_type == "publish-data" or set_type == "export-influxdb-to-csv":
        result = {}
    else:
        logger.error(f"The passed action set_type parameter '{set_type}' is not valid")
        result = {}
    if result is None:
        return False
    # Opt-in battery self-identification (observe/suggest). Runs alongside the
    # optimization prep for the two live optim actions; never affects the
    # optimizer in v1. Cadence-gated and never raises, so it cannot break an
    # optimization run.
    if (
        set_type in ("dayahead-optim", "naive-mpc-optim")
        and optim_conf.get("set_use_battery", False)
        and optim_conf.get("set_use_battery_identification", False)
    ):
        await identify_battery(
            logger,
            optim_conf,
            plant_conf,
            retrieve_hass_conf,
            rh,
            emhass_conf,
            get_data_from_file,
            test_df_literal,
        )
    data_results.update(result)
    # Build Final Dictionary
    input_data_dict = {
        "emhass_conf": emhass_conf,
        "retrieve_hass_conf": retrieve_hass_conf,
        "optim_conf": optim_conf,
        "plant_conf": plant_conf,
        "rh": rh,
        "opt": opt,
        "fcst": fcst,
        "costfun": costfun,
        "params": params,
        "stage_times": stage_times,
        **data_results,
    }
    return input_data_dict


async def weather_forecast_cache(
    emhass_conf: dict, params: str, runtimeparams: str, logger: logging.Logger
) -> bool:
    """
    Perform a call to get forecast function, intend to save results to cache.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param runtimeparams: Runtime optimization parameters passed as a dictionary
    :type runtimeparams: str
    :param logger: The passed logger object
    :type logger: logging object
    :return: A bool for function completion
    :rtype: bool

    """
    # Parsing yaml
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
    # Treat runtimeparams
    (
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
    ) = await utils.treat_runtimeparams(
        runtimeparams,
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "forecast",
        logger,
        emhass_conf,
    )
    # Make sure weather_forecast_cache is true
    if (params is not None) and (params != "null"):
        params = orjson.loads(params)
    else:
        params = {}
    params["passed_data"]["weather_forecast_cache"] = True
    params = orjson.dumps(params).decode("utf-8")
    # Create Forecast object
    fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf, params, emhass_conf, logger)
    result = await fcst.get_weather_forecast(optim_conf["weather_forecast_method"])
    if isinstance(result, bool) and not result:
        return False

    return True


def _log_optimization_summary(input_data_dict: dict, logger: logging.Logger) -> None:
    """Emit the one-line optimization summary (total elapsed + top stage).

    Reads per-stage timings recorded by the orchestrators in ``input_data_dict["stage_times"]``.
    No-op if no stages were recorded.
    """
    stage_times = input_data_dict.get("stage_times", {})
    if not stage_times:
        return
    total = sum(stage_times.values())
    top_name, top_s = max(stage_times.items(), key=lambda x: x[1])
    pct = int(100 * top_s / total) if total > 0 else 0
    logger.info(f"Optimization completed in {total:.1f}s (top: {top_name}={top_s:.1f}s, {pct}%)")


async def perfect_forecast_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = True,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the perfect forecast optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    _t0 = _time.monotonic()
    logger.info("Performing perfect forecast optimization")
    # Load cost and prod price forecast
    with stage_timer(input_data_dict["stage_times"], "price_prep", logger):
        df_input_data = input_data_dict["fcst"].get_load_cost_forecast(
            input_data_dict["df_input_data"],
            method=input_data_dict["fcst"].optim_conf["load_cost_forecast_method"],
            list_and_perfect=True,
        )
        if isinstance(df_input_data, bool) and not df_input_data:
            return False
        df_input_data = input_data_dict["fcst"].get_prod_price_forecast(
            df_input_data,
            method=input_data_dict["fcst"].optim_conf["production_price_forecast_method"],
            list_and_perfect=True,
        )
    if isinstance(df_input_data, bool) and not df_input_data:
        return False
    with stage_timer(input_data_dict["stage_times"], "optim_solve", logger):
        opt_res = input_data_dict["opt"].perform_perfect_forecast_optim(
            df_input_data, input_data_dict["days_list"]
        )
    # Save CSV file for analysis
    if save_data_to_file:
        filename = "opt_res_perfect_optim_" + input_data_dict["costfun"] + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )
    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save perfect results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        with stage_timer(input_data_dict["stage_times"], "publish", logger):
            # Trigger the publish function, save entity data and not post to HA
            await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    _log_optimization_summary(input_data_dict, logger)
    _record_optim_snapshot(input_data_dict, last_run.ACTION_PERFECT_OPTIM, opt_res, _t0, logger)

    return opt_res


def prepare_forecast_and_weather_data(
    input_data_dict: dict,
    logger: logging.Logger,
    warn_on_resolution: bool = False,
) -> pd.DataFrame | bool:
    """
    Prepare forecast data with load costs, production prices, outdoor temperature, and GHI.

    This helper function eliminates duplication between dayahead_forecast_optim and naive_mpc_optim.

    :param input_data_dict: Dictionary with forecast and input data
    :type input_data_dict: dict
    :param logger: Logger object
    :type logger: logging.Logger
    :param warn_on_resolution: Whether to warn about GHI resolution mismatch
    :type warn_on_resolution: bool
    :return: Prepared DataFrame or False on error
    :rtype: pd.DataFrame | bool
    """
    # Get load cost forecast
    df_input_data_dayahead = input_data_dict["fcst"].get_load_cost_forecast(
        input_data_dict["df_input_data_dayahead"],
        method=input_data_dict["fcst"].optim_conf["load_cost_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False

    # Get production price forecast
    df_input_data_dayahead = input_data_dict["fcst"].get_prod_price_forecast(
        df_input_data_dayahead,
        method=input_data_dict["fcst"].optim_conf["production_price_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False

    # Add outdoor temperature if provided
    passed_outdoor_temp = input_data_dict["params"]["passed_data"].get(
        "outdoor_temperature_forecast"
    )

    if passed_outdoor_temp is not None:
        forecast_len = len(df_input_data_dayahead)

        # If the passed forecast is shorter than the horizon, pad it with the last value to prevent Pandas crashes
        if len(passed_outdoor_temp) < forecast_len:
            logger.warning(
                "Passed outdoor_temperature_forecast length (%s) "
                "is shorter than the prediction horizon (%s). Padding with the last value.",
                len(passed_outdoor_temp),
                forecast_len,
            )
            last_val = passed_outdoor_temp[-1] if len(passed_outdoor_temp) > 0 else 15.0
            passed_outdoor_temp = passed_outdoor_temp + [last_val] * (
                forecast_len - len(passed_outdoor_temp)
            )

        # If it's longer (e.g. 48h data for 13h horizon), slice it securely
        df_input_data_dayahead["outdoor_temperature_forecast"] = passed_outdoor_temp[:forecast_len]

    # Auto-fallback to temp_air from Open-Meteo weather forecast
    elif (
        input_data_dict["df_weather"] is not None
        and "temp_air" in input_data_dict["df_weather"].columns
    ):
        dayahead_index = df_input_data_dayahead.index
        weather_series = input_data_dict["df_weather"]["temp_air"].copy()

        # Handle Timezone Mismatches
        # If optimization is naive but weather is aware -> Strip weather TZ
        if dayahead_index.tz is None and weather_series.index.tz is not None:
            weather_series.index = weather_series.index.tz_localize(None)
        # If optimization is aware but weather is naive -> Localize weather
        elif dayahead_index.tz is not None and weather_series.index.tz is None:
            weather_series.index = weather_series.index.tz_localize(dayahead_index.tz)
        # If both are aware -> Convert weather to optimization TZ
        elif dayahead_index.tz is not None and weather_series.index.tz is not None:
            weather_series.index = weather_series.index.tz_convert(dayahead_index.tz)

        # Robust Reindexing (The fix for "Found 48 NaNs")
        # method='nearest' snaps 10:00:15 to 10:00:00
        # tolerance='1h' prevents filling data from too far away
        df_input_data_dayahead["outdoor_temperature_forecast"] = weather_series.reindex(
            dayahead_index, method="nearest", tolerance=pd.Timedelta("1h")
        )

        # Final safety fill (forward/backward) for any remaining gaps
        if df_input_data_dayahead["outdoor_temperature_forecast"].isnull().any():
            df_input_data_dayahead["outdoor_temperature_forecast"] = (
                df_input_data_dayahead["outdoor_temperature_forecast"]
                .fillna(method="ffill")
                .fillna(method="bfill")
            )

    # Merge GHI (Global Horizontal Irradiance) from weather forecast if available
    if input_data_dict["df_weather"] is not None and "ghi" in input_data_dict["df_weather"].columns:
        dayahead_index = df_input_data_dayahead.index
        ghi_series = input_data_dict["df_weather"]["ghi"].copy()

        # Handle Timezone Mismatches (Same as above)
        if dayahead_index.tz is None and ghi_series.index.tz is not None:
            ghi_series.index = ghi_series.index.tz_localize(None)
        elif dayahead_index.tz is not None and ghi_series.index.tz is None:
            ghi_series.index = ghi_series.index.tz_localize(dayahead_index.tz)
        elif dayahead_index.tz is not None and ghi_series.index.tz is not None:
            ghi_series.index = ghi_series.index.tz_convert(dayahead_index.tz)

        # Check time resolution if requested
        if (
            warn_on_resolution
            and len(input_data_dict["df_weather"].index) > 1
            and len(dayahead_index) > 1
        ):
            weather_index = input_data_dict["df_weather"].index
            weather_freq = (weather_index[1] - weather_index[0]).total_seconds()
            dayahead_freq = (dayahead_index[1] - dayahead_index[0]).total_seconds()
            if weather_freq > 2 * dayahead_freq:
                logger.warning(
                    "Weather data time resolution (%.0fs) is much coarser than dayahead index (%.0fs). "
                    "Step changes in GHI may occur.",
                    weather_freq,
                    dayahead_freq,
                )

        # Robust Reindexing
        df_input_data_dayahead["ghi"] = ghi_series.reindex(
            dayahead_index, method="nearest", tolerance=pd.Timedelta("1h")
        )

        # Final safety fill
        if df_input_data_dayahead["ghi"].isnull().any():
            df_input_data_dayahead["ghi"] = (
                df_input_data_dayahead["ghi"].fillna(method="ffill").fillna(method="bfill")
            )

        logger.debug(
            "Merged GHI data into optimization input: mean=%.1f W/m², max=%.1f W/m²",
            df_input_data_dayahead["ghi"].mean(),
            df_input_data_dayahead["ghi"].max(),
        )

    return df_input_data_dayahead


async def dayahead_forecast_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the day-ahead optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    _t0 = _time.monotonic()
    soc_init = input_data_dict["params"]["passed_data"].get("soc_init")
    soc_final = input_data_dict["params"]["passed_data"].get("soc_final")
    logger.info(
        f"Performing day-ahead forecast optimization with soc_init: {soc_init}, soc_final: {soc_final}"
    )
    # Prepare forecast data with costs, prices, outdoor temp, and GHI
    with stage_timer(input_data_dict["stage_times"], "price_prep", logger):
        df_input_data_dayahead = prepare_forecast_and_weather_data(
            input_data_dict, logger, warn_on_resolution=False
        )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    with stage_timer(input_data_dict["stage_times"], "optim_solve", logger):
        opt_res_dayahead = input_data_dict["opt"].perform_dayahead_forecast_optim(
            df_input_data_dayahead,
            input_data_dict["p_pv_forecast"],
            input_data_dict["p_load_forecast"],
            soc_init=soc_init,
            soc_final=soc_final,
            stage_times=input_data_dict["stage_times"],
        )
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_dayahead_" + today.strftime("%Y_%m_%d") + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res_dayahead.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )

    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save day_ahead results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        with stage_timer(input_data_dict["stage_times"], "publish", logger):
            # Trigger the publish function, save entity data and not post to HA
            await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    _log_optimization_summary(input_data_dict, logger)
    _record_optim_snapshot(
        input_data_dict, last_run.ACTION_DAYAHEAD_OPTIM, opt_res_dayahead, _t0, logger
    )

    return opt_res_dayahead


async def naive_mpc_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the naive Model Predictive Controller optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    _t0 = _time.monotonic()
    logger.info("Performing naive MPC optimization")
    # Prepare forecast data with costs, prices, outdoor temp, and GHI (with resolution warning)
    with stage_timer(input_data_dict["stage_times"], "price_prep", logger):
        df_input_data_dayahead = prepare_forecast_and_weather_data(
            input_data_dict, logger, warn_on_resolution=True
        )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    # The specifics params for the MPC at runtime
    prediction_horizon = min(
        input_data_dict["params"]["passed_data"]["prediction_horizon"],
        len(df_input_data_dayahead),
    )
    soc_init = input_data_dict["params"]["passed_data"]["soc_init"]
    soc_final = input_data_dict["params"]["passed_data"]["soc_final"]
    soc_target = input_data_dict["params"]["passed_data"].get("soc_target", None)
    soc_target_timestep = input_data_dict["params"]["passed_data"].get("soc_target_timestep", None)
    current_period_peak = input_data_dict["params"]["passed_data"].get("current_period_peak", None)
    capacity_charge_window = input_data_dict["params"]["passed_data"].get(
        "capacity_charge_window", None
    )
    capacity_charge_consideration = input_data_dict["params"]["passed_data"].get(
        "capacity_charge_consideration", None
    )
    capacity_charge_current_interval_history = input_data_dict["params"]["passed_data"].get(
        "capacity_charge_current_interval_history", None
    )
    def_total_hours = input_data_dict["params"]["optim_conf"].get(
        "operating_hours_of_each_deferrable_load", None
    )
    def_total_timestep = input_data_dict["params"]["optim_conf"].get(
        "operating_timesteps_of_each_deferrable_load", None
    )
    def_start_timestep = input_data_dict["params"]["optim_conf"].get(
        "start_timesteps_of_each_deferrable_load"
    )
    def_end_timestep = input_data_dict["params"]["optim_conf"].get(
        "end_timesteps_of_each_deferrable_load"
    )
    with stage_timer(input_data_dict["stage_times"], "optim_solve", logger):
        opt_res_naive_mpc = input_data_dict["opt"].perform_naive_mpc_optim(
            df_input_data_dayahead,
            input_data_dict["p_pv_forecast"],
            input_data_dict["p_load_forecast"],
            prediction_horizon,
            soc_init,
            soc_final,
            soc_target=soc_target,
            soc_target_timestep=soc_target_timestep,
            current_period_peak=current_period_peak,
            capacity_charge_window=capacity_charge_window,
            capacity_charge_consideration=capacity_charge_consideration,
            capacity_charge_current_interval_history=capacity_charge_current_interval_history,
            def_total_hours=def_total_hours,
            def_total_timestep=def_total_timestep,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
            stage_times=input_data_dict["stage_times"],
        )
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_naive_mpc_" + today.strftime("%Y_%m_%d") + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res_naive_mpc.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )

    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save mpc results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        with stage_timer(input_data_dict["stage_times"], "publish", logger):
            # Trigger the publish function, save entity data and not post to HA
            await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    _log_optimization_summary(input_data_dict, logger)
    _record_optim_snapshot(
        input_data_dict, last_run.ACTION_NAIVE_MPC_OPTIM, opt_res_naive_mpc, _t0, logger
    )

    return opt_res_naive_mpc


def _get_weather_features(input_data_dict: dict) -> list[str]:
    """Read the configured mlforecaster weather covariate columns (empty when unset)."""
    return list(input_data_dict["params"]["passed_data"].get("mlforecaster_weather_features") or [])


async def _attach_weather_covariates(
    input_data_dict: dict, data: pd.DataFrame, weather_features: list[str], logger: logging.Logger
) -> pd.DataFrame:
    """Attach the configured weather covariate columns onto a load DataFrame (in place, returned).

    Used for the training data (fit/tune) so the columns are aligned to the load history. A no-op
    that returns ``data`` unchanged when no ``weather_features`` are configured.
    """
    if not weather_features:
        return data
    covariates = await input_data_dict["fcst"].get_weather_covariates(data.index, weather_features)
    for column in weather_features:
        data[column] = covariates[column].to_numpy()
    logger.info("Attached %s weather covariate(s) to the load data", len(weather_features))
    return data


async def forecast_model_fit(
    input_data_dict: dict, logger: logging.Logger, debug: bool | None = False
) -> tuple[pd.DataFrame, pd.DataFrame, MLForecaster]:
    """Perform a forecast model fit from training data retrieved from Home Assistant.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :return: The DataFrame containing the forecast data results without and with backtest and the `mlforecaster` object
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, mlforecaster]
    """
    data = copy.deepcopy(input_data_dict["df_input_data"])
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    var_model = input_data_dict["params"]["passed_data"]["var_model"]
    sklearn_model = input_data_dict["params"]["passed_data"]["sklearn_model"]
    num_lags = input_data_dict["params"]["passed_data"]["num_lags"]
    split_date_delta = input_data_dict["params"]["passed_data"]["split_date_delta"]
    perform_backtest = input_data_dict["params"]["passed_data"]["perform_backtest"]
    # Optionally attach weather covariates (aligned to the load history) for the model to use.
    weather_features = _get_weather_features(input_data_dict)
    data = await _attach_weather_covariates(input_data_dict, data, weather_features, logger)
    # The ML forecaster object
    mlf = MLForecaster(
        data,
        model_type,
        var_model,
        sklearn_model,
        num_lags,
        input_data_dict["emhass_conf"],
        logger,
        weather_features=weather_features,
    )
    # Fit the ML model
    df_pred, df_pred_backtest = await mlf.fit(
        split_date_delta=split_date_delta, perform_backtest=perform_backtest
    )
    # Save model
    if not debug:
        filename = model_type + default_pkl_suffix
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlf, pickle.HIGHEST_PROTOCOL))
            logger.debug("saved model to " + str(filename_path))
    return df_pred, df_pred_backtest, mlf


async def forecast_calibration(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Compute an on-demand load forecast calibration report from HA history.

    Retrieves the load history, then compares the built-in load forecast methods
    (naive, typical, mlforecaster) against the realised load over held-out
    test/val windows. Reporting only: no model or artifact is saved and no
    optimization is affected.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: The calibration result dict, or None when there is not enough history
    :rtype: dict | None
    """
    passed_data = input_data_dict["params"]["passed_data"]
    retrieve_hass_conf = input_data_dict["retrieve_hass_conf"]
    rh = input_data_dict["rh"]
    var_model = passed_data.get("var_model") or retrieve_hass_conf.get(
        "sensor_power_load_no_var_loads", "sensor.power_load_no_var_loads"
    )

    # The calibration day windows are runtime-overridable, report-only knobs (never
    # read from the config GUI). Each falls back to its module default, so an empty
    # request reproduces the standard 90 / 14 / 14 day report. A non-positive value
    # is treated as "not set" and falls back to the default; an over-short retrieval
    # window is caught by compute_forecast_calibration's minimum-history gate, which
    # returns a clean error rather than crashing.
    def _positive_or_default(key: str, default: int) -> int:
        value = int(passed_data.get(key) or 0)
        return value if value > 0 else default

    days_to_retrieve = _positive_or_default(
        "calibration_days_to_retrieve", CALIBRATION_DEFAULT_DAYS
    )
    test_days = _positive_or_default("calibration_test_days", CALIBRATION_TEST_DAYS)
    val_days = _positive_or_default("calibration_val_days", CALIBRATION_VAL_DAYS)

    days_list = utils.get_days_list(days_to_retrieve)
    if not await rh.get_data(days_list, [var_model]):
        logger.error("Forecast calibration: failed to retrieve load history from Home Assistant")
        return None
    rh.prepare_data(
        var_model,
        load_negative=retrieve_hass_conf.get("load_negative", False),
        set_zero_min=retrieve_hass_conf.get("set_zero_min", True),
        var_replace_zero=retrieve_hass_conf.get("sensor_replace_zero", []),
        var_interp=retrieve_hass_conf.get("sensor_linear_interp", []),
        skip_renaming=True,
    )
    load = rh.df_final[var_model].copy()
    # The mlforecaster row is fit fresh in memory with a fast, stable
    # LinearRegression (not the user's configured/saved model), so the report is
    # quick and reproducible and never depends on a slow estimator (e.g. SVR/MLP).
    result = await compute_forecast_calibration(
        load,
        rh.freq,
        input_data_dict["emhass_conf"],
        logger,
        sklearn_model="LinearRegression",
        test_days=test_days,
        val_days=val_days,
        var_model=var_model,
    )
    if result.get("error"):
        return None
    return result


async def forecast_model_predict(
    input_data_dict: dict,
    logger: logging.Logger,
    use_last_window: bool | None = True,
    debug: bool | None = False,
    mlf: MLForecaster | None = None,
) -> pd.DataFrame:
    r"""Perform a forecast model predict using a previously trained skforecast model.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param use_last_window: True if the 'last_window' option should be used for the \
        custom machine learning forecast model. The 'last_window=True' means that the data \
        that will be used to generate the new forecast will be freshly retrieved from \
        Home Assistant. This data is needed because the forecast model is an auto-regressive \
        model with lags. If 'False' then the data using during the model train is used. Defaults to True
    :type use_last_window: Optional[bool], optional
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
        and unit testing. In production the actual model will be read from a saved pickle file. Defaults to None
    :type mlf: Optional[mlforecaster], optional
    :return: The DataFrame containing the forecast prediction data
    :rtype: pd.DataFrame
    """
    # Load model
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    filename = model_type + default_pkl_suffix
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlf = pickle.loads(content)
                logger.debug("loaded saved model from " + str(filename_path))
        else:
            logger.error(
                "The ML forecaster file ("
                + str(filename_path)
                + ") was not found, please run a model fit method before this predict method",
            )
            return
    # Make predictions
    if use_last_window:
        data_last_window = copy.deepcopy(input_data_dict["df_input_data"])
    else:
        data_last_window = None
    # When the model was trained with weather covariates, supply the future weather over the
    # forecast horizon so the recursive predict has the exog columns it expects.
    weather_future = await input_data_dict["fcst"]._build_weather_future(data_last_window, mlf)
    predictions = await mlf.predict(data_last_window, weather_future=weather_future)
    # Publish data to a Home Assistant sensor
    model_predict_publish = input_data_dict["params"]["passed_data"]["model_predict_publish"]
    model_predict_entity_id = input_data_dict["params"]["passed_data"]["model_predict_entity_id"]
    model_predict_device_class = input_data_dict["params"]["passed_data"][
        "model_predict_device_class"
    ]
    model_predict_unit_of_measurement = input_data_dict["params"]["passed_data"][
        "model_predict_unit_of_measurement"
    ]
    model_predict_friendly_name = input_data_dict["params"]["passed_data"][
        "model_predict_friendly_name"
    ]
    publish_prefix = input_data_dict["params"]["passed_data"]["publish_prefix"]
    if model_predict_publish is True:
        # Estimate the current index
        now_precise = datetime.now(input_data_dict["retrieve_hass_conf"]["time_zone"]).replace(
            second=0, microsecond=0
        )
        if input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "nearest":
            idx_closest = predictions.index.get_indexer([now_precise], method="nearest")[0]
        elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "first":
            idx_closest = predictions.index.get_indexer([now_precise], method="ffill")[0]
        elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "last":
            idx_closest = predictions.index.get_indexer([now_precise], method="bfill")[0]
        if idx_closest == -1:
            idx_closest = predictions.index.get_indexer([now_precise], method="nearest")[0]
        # Publish Load forecast
        await input_data_dict["rh"].post_data(
            predictions,
            idx_closest,
            model_predict_entity_id,
            model_predict_device_class,
            model_predict_unit_of_measurement,
            model_predict_friendly_name,
            type_var="mlforecaster",
            publish_prefix=publish_prefix,
        )
    return predictions


async def forecast_model_tune(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
    mlf: MLForecaster | None = None,
) -> tuple[pd.DataFrame, MLForecaster]:
    """Tune a forecast model hyperparameters using bayesian optimization.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
        and unit testing. In production the actual model will be read from a saved pickle file. Defaults to None
    :type mlf: Optional[mlforecaster], optional
    :return: The DataFrame containing the forecast data results using the optimized model
    :rtype: pd.DataFrame
    """
    # Load model
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    filename = model_type + default_pkl_suffix
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlf = pickle.loads(content)
                logger.debug("loaded saved model from " + str(filename_path))
        else:
            logger.error(
                "The ML forecaster file ("
                + str(filename_path)
                + ") was not found, please run a model fit method before this tune method",
            )
            return None, None
    # Tune the model
    split_date_delta = input_data_dict["params"]["passed_data"]["split_date_delta"]
    if debug:
        n_trials = 5
    else:
        n_trials = input_data_dict["params"]["passed_data"]["n_trials"]
    df_pred_optim = await mlf.tune(
        split_date_delta=split_date_delta, n_trials=n_trials, debug=debug
    )
    # Save model
    if not debug:
        filename = model_type + default_pkl_suffix
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlf, pickle.HIGHEST_PROTOCOL))
            logger.debug("Saved model to " + str(filename_path))
    return df_pred_optim, mlf


async def regressor_model_fit(
    input_data_dict: dict, logger: logging.Logger, debug: bool | None = False
) -> MLRegressor:
    """Perform a forecast model fit from training data retrieved from Home Assistant.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    """
    data = copy.deepcopy(input_data_dict["df_input_data"])
    if "model_type" in input_data_dict["params"]["passed_data"]:
        model_type = input_data_dict["params"]["passed_data"]["model_type"]
    else:
        logger.error("parameter: 'model_type' not passed")
        return False
    if "regression_model" in input_data_dict["params"]["passed_data"]:
        regression_model = input_data_dict["params"]["passed_data"]["regression_model"]
    else:
        logger.error("parameter: 'regression_model' not passed")
        return False
    if "features" in input_data_dict["params"]["passed_data"]:
        features = input_data_dict["params"]["passed_data"]["features"]
    else:
        logger.error("parameter: 'features' not passed")
        return False
    if "target" in input_data_dict["params"]["passed_data"]:
        target = input_data_dict["params"]["passed_data"]["target"]
    else:
        logger.error("parameter: 'target' not passed")
        return False
    if "timestamp" in input_data_dict["params"]["passed_data"]:
        timestamp = input_data_dict["params"]["passed_data"]["timestamp"]
    else:
        logger.error("parameter: 'timestamp' not passed")
        return False
    if "date_features" in input_data_dict["params"]["passed_data"]:
        date_features = input_data_dict["params"]["passed_data"]["date_features"]
    else:
        logger.error("parameter: 'date_features' not passed")
        return False
    # The MLRegressor object
    mlr = MLRegressor(data, model_type, regression_model, features, target, timestamp, logger)
    # Fit the ML model
    fit = await mlr.fit(date_features=date_features)
    if not fit:
        return False
    # Save model
    if not debug:
        filename = model_type + "_mlr.pkl"
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlr, pickle.HIGHEST_PROTOCOL))
    return mlr


async def regressor_model_predict(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
    mlr: MLRegressor | None = None,
) -> np.ndarray:
    """Perform a prediction from csv file.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    """
    if "model_type" in input_data_dict["params"]["passed_data"]:
        model_type = input_data_dict["params"]["passed_data"]["model_type"]
    else:
        logger.error("parameter: 'model_type' not passed")
        return False
    filename = model_type + "_mlr.pkl"
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlr = pickle.loads(content)
        else:
            logger.error(
                "The ML forecaster file was not found, please run a model fit method before this predict method",
            )
            return False
    if "new_values" in input_data_dict["params"]["passed_data"]:
        new_values = input_data_dict["params"]["passed_data"]["new_values"]
    else:
        logger.error("parameter: 'new_values' not passed")
        return False
    # Predict from csv file
    prediction = await mlr.predict(new_values)
    mlr_predict_entity_id = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_entity_id", "sensor.mlr_predict"
    )
    mlr_predict_device_class = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_device_class", "power"
    )
    mlr_predict_unit_of_measurement = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_unit_of_measurement", "W"
    )
    mlr_predict_friendly_name = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_friendly_name", "mlr predictor"
    )
    # Publish prediction
    idx = 0
    if not debug:
        await input_data_dict["rh"].post_data(
            prediction,
            idx,
            mlr_predict_entity_id,
            mlr_predict_device_class,
            mlr_predict_unit_of_measurement,
            mlr_predict_friendly_name,
            type_var="mlregressor",
        )
    return prediction


async def export_influxdb_to_csv(
    input_data_dict: dict | None,
    logger: logging.Logger,
    emhass_conf: dict | None = None,
    params: str | None = None,
    runtimeparams: str | None = None,
) -> bool:
    """Export data from InfluxDB to CSV file.

    This function can be called in two ways:
    1. With input_data_dict (from web_server via set_input_data_dict)
    2. Without input_data_dict (direct call from command line or web_server before set_input_data_dict)

    :param input_data_dict: Dictionary containing configuration and parameters (optional)
    :type input_data_dict: dict | None
    :param logger: Logger object
    :type logger: logging.Logger
    :param emhass_conf: Dictionary containing EMHASS configuration paths (used when input_data_dict is None)
    :type emhass_conf: dict | None
    :param params: JSON string of params (used when input_data_dict is None)
    :type params: str | None
    :param runtimeparams: JSON string of runtime parameters (used when input_data_dict is None)
    :type runtimeparams: str | None
    :return: Success status
    :rtype: bool
    """
    # Handle two calling modes
    if input_data_dict is None:
        # Direct mode: parse params and create RetrieveHass
        if emhass_conf is None or params is None:
            logger.error("emhass_conf and params are required when input_data_dict is None")
            return False
        # Parse params
        if isinstance(params, str):
            params = orjson.loads(params)
        if isinstance(runtimeparams, str):
            runtimeparams = orjson.loads(runtimeparams)
        # Get configuration
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
        if isinstance(retrieve_hass_conf, bool):
            return False
        # Treat runtime params
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            orjson.dumps(runtimeparams).decode("utf-8") if runtimeparams else "{}",
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "export-influxdb-to-csv",
            logger,
            emhass_conf,
        )
        # Parse params again if it's a string
        if isinstance(params, str):
            params = orjson.loads(params)
        # Create RetrieveHass object
        rh = RetrieveHass(
            retrieve_hass_conf["hass_url"],
            retrieve_hass_conf["long_lived_token"],
            retrieve_hass_conf["optimization_time_step"],
            retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
        )
        time_zone = rh.time_zone
        data_path = emhass_conf["data_path"]
    else:
        # Standard mode: use input_data_dict
        params = input_data_dict["params"]
        if isinstance(params, str):
            params = orjson.loads(params)
        rh = input_data_dict["rh"]
        time_zone = rh.time_zone
        data_path = input_data_dict["emhass_conf"]["data_path"]
    # Extract parameters from passed_data
    if "sensor_list" not in params.get("passed_data", {}):
        logger.error("parameter: 'sensor_list' not passed")
        return False
    sensor_list = params["passed_data"]["sensor_list"]
    if "csv_filename" not in params.get("passed_data", {}):
        logger.error("parameter: 'csv_filename' not passed")
        return False
    csv_filename = params["passed_data"]["csv_filename"]
    if "start_time" not in params.get("passed_data", {}):
        logger.error("parameter: 'start_time' not passed")
        return False
    start_time = params["passed_data"]["start_time"]
    # Optional parameters with defaults
    end_time = params["passed_data"].get("end_time", None)
    resample_freq = params["passed_data"].get("resample_freq", "1h")
    timestamp_col = params["passed_data"].get("timestamp_col_name", "timestamp")
    decimal_places = params["passed_data"].get("decimal_places", 2)
    handle_nan = params["passed_data"].get("handle_nan", "keep")
    # Check if InfluxDB is enabled
    if not rh.use_influxdb:
        logger.error(
            "InfluxDB is not enabled in configuration. Set use_influxdb: true in config.json"
        )
        return False
    # Parse time range
    start_dt, end_dt = utils.parse_export_time_range(start_time, end_time, time_zone, logger)
    if start_dt is False:
        return False
    # Create days list for data retrieval
    days_list = pd.date_range(start=start_dt.date(), end=end_dt.date(), freq="D", tz=time_zone)
    if len(days_list) == 0:
        logger.error("No days to retrieve. Check start_time and end_time.")
        return False
    logger.info(
        f"Retrieving {len(sensor_list)} sensors from {start_dt} to {end_dt} ({len(days_list)} days)"
    )
    logger.info(f"Sensors: {sensor_list}")
    # Retrieve data from InfluxDB
    success = rh.get_data(days_list, sensor_list)
    if not success or rh.df_final is None or rh.df_final.empty:
        logger.error("Failed to retrieve data from InfluxDB")
        return False
    # Filter and resample data
    df_export = utils.resample_and_filter_data(rh.df_final, start_dt, end_dt, resample_freq, logger)
    if df_export is False:
        return False
    # Reset index to make timestamp a column
    # Handle custom index names by renaming the index first
    df_export = df_export.rename_axis(timestamp_col).reset_index()
    # Clean column names
    df_export = utils.clean_sensor_column_names(df_export, timestamp_col)
    # Handle NaN values
    df_export = utils.handle_nan_values(df_export, handle_nan, timestamp_col, logger)
    # Round numeric columns to specified decimal places
    numeric_cols = df_export.select_dtypes(include=[np.number]).columns
    df_export[numeric_cols] = df_export[numeric_cols].round(decimal_places)
    # Save to CSV
    csv_path = pathlib.Path(data_path) / csv_filename
    df_export.to_csv(csv_path, index=False)
    logger.info(f"✓ Successfully exported to {csv_filename}")
    logger.info(f"  Rows: {df_export.shape[0]}")
    logger.info(f"  Columns: {list(df_export.columns)}")
    logger.info(
        f"  Time range: {df_export[timestamp_col].min()} to {df_export[timestamp_col].max()}"
    )
    logger.info(f"  File location: {csv_path}")
    return True


def _get_params(input_data_dict: dict) -> dict:
    """Helper to extract params from input_data_dict."""
    if input_data_dict:
        if not isinstance(input_data_dict.get("params", {}), dict):
            return orjson.loads(input_data_dict["params"])
        return input_data_dict.get("params", {})
    return {}


async def _publish_from_saved_entities(
    input_data_dict: dict, logger: logging.Logger, params: dict
) -> pd.DataFrame | None:
    """
    Helper to publish data from saved entity JSON files if publish_prefix is set.
    Returns DataFrame if successful, None if fallback to CSV is needed.
    """
    publish_prefix = params["passed_data"].get("publish_prefix", "")
    entity_path = input_data_dict["emhass_conf"]["data_path"] / "entities"
    if not entity_path.exists() or not os.listdir(entity_path):
        logger.warning(f"No saved entity json files in path: {entity_path}")
        logger.warning("Falling back to opt_res_latest")
        return None
    entity_path_contents = os.listdir(entity_path)
    matches_prefix = any(publish_prefix in entity for entity in entity_path_contents)
    if not (matches_prefix or publish_prefix == "all"):
        logger.warning(f"No saved entity json files that match prefix: {publish_prefix}")
        logger.warning("Falling back to opt_res_latest")
        return None
    opt_res_list = []
    opt_res_list_names = []
    for entity in entity_path_contents:
        if entity == default_metadata_json:
            continue
        if publish_prefix == "all" or publish_prefix in entity:
            entity_data = await publish_json(entity, input_data_dict, entity_path, logger)
            if isinstance(entity_data, bool):
                return None  # Error occurred
            opt_res_list.append(entity_data)
            opt_res_list_names.append(entity.replace(".json", ""))
    opt_res = pd.concat(opt_res_list, axis=1)
    opt_res.columns = opt_res_list_names
    return opt_res


def _load_opt_res_latest(
    input_data_dict: dict, logger: logging.Logger, save_data_to_file: bool
) -> pd.DataFrame | None:
    """Helper to load the optimization results DataFrame from CSV."""
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_dayahead_" + today.strftime("%Y_%m_%d") + ".csv"
    else:
        filename = default_csv_filename
    file_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not file_path.exists():
        logger.error("File not found error, run an optimization task first.")
        return None
    opt_res_latest = pd.read_csv(file_path, index_col="timestamp")
    opt_res_latest.index = pd.to_datetime(opt_res_latest.index, utc=True).tz_convert(
        input_data_dict["retrieve_hass_conf"]["time_zone"]
    )
    # Infer the index frequency from the saved data itself rather than asserting
    # the current request's optimization_time_step onto it (#976): the CSV may
    # have been written by a run with a different runtime optimization_time_step,
    # and pandas raises on the mismatch. The publish path only needs the
    # timestamps for nearest-index matching. Frames with fewer than 2 rows carry
    # no inferable spacing, so leave their freq unset.
    if len(opt_res_latest.index) > 1:
        opt_res_latest = utils.set_df_index_freq(opt_res_latest)
    return opt_res_latest


def _get_closest_index(retrieve_hass_conf: dict, index: pd.DatetimeIndex) -> int:
    """Helper to find the closest index in the DataFrame to the current time."""
    now_precise = datetime.now(retrieve_hass_conf["time_zone"]).replace(second=0, microsecond=0)
    now_ts = pd.Timestamp(now_precise)
    if index.tz is None and now_ts.tz is not None:
        now_ts = now_ts.tz_localize(None)
    elif index.tz is not None and now_ts.tz is None:
        now_ts = now_ts.tz_localize(index.tz)
    method = retrieve_hass_conf.get("method_ts_round", "nearest")
    if method == "nearest":
        return index.get_indexer([now_ts], method="nearest")[0]
    elif method == "first":
        return index.get_indexer([now_ts], method="ffill")[0]
    elif method == "last":
        return index.get_indexer([now_ts], method="bfill")[0]
    return index.get_indexer([now_ts], method="nearest")[0]


async def _publish_standard_forecasts(
    ctx: PublishContext, opt_res_latest: pd.DataFrame
) -> list[str]:
    """Publish PV, Load, Curtailment, and Hybrid Inverter data."""
    cols = []
    # Load Forecast
    custom_load = ctx.params["passed_data"]["custom_load_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_Load"],
        ctx.idx,
        custom_load["entity_id"],
        "power",
        custom_load["unit_of_measurement"],
        custom_load["friendly_name"],
        type_var="power",
        **ctx.common_kwargs,
    )
    cols.append("P_Load")
    # PV Forecast
    if "P_PV" in opt_res_latest.columns:
        custom_pv = ctx.params["passed_data"]["custom_pv_forecast_id"]
        await ctx.rh.post_data(
            opt_res_latest["P_PV"],
            ctx.idx,
            custom_pv["entity_id"],
            "power",
            custom_pv["unit_of_measurement"],
            custom_pv["friendly_name"],
            type_var="power",
            **ctx.common_kwargs,
        )
        cols.append("P_PV")
    # Curtailment
    if ctx.plant_conf["compute_curtailment"]:
        custom_curt = ctx.params["passed_data"]["custom_pv_curtailment_id"]
        await ctx.rh.post_data(
            opt_res_latest["P_PV_curtailment"],
            ctx.idx,
            custom_curt["entity_id"],
            "power",
            custom_curt["unit_of_measurement"],
            custom_curt["friendly_name"],
            type_var="power",
            **ctx.common_kwargs,
        )
        cols.append("P_PV_curtailment")
    # Hybrid Inverter
    if ctx.plant_conf["inverter_is_hybrid"]:
        custom_inv = ctx.params["passed_data"]["custom_hybrid_inverter_id"]
        await ctx.rh.post_data(
            opt_res_latest["P_hybrid_inverter"],
            ctx.idx,
            custom_inv["entity_id"],
            "power",
            custom_inv["unit_of_measurement"],
            custom_inv["friendly_name"],
            type_var="power",
            **ctx.common_kwargs,
        )
        cols.append("P_hybrid_inverter")
    return cols


async def _publish_deferrable_loads(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish data for all deferrable loads."""
    cols = []
    custom_def = ctx.params["passed_data"]["custom_deferrable_forecast_id"]
    for k in range(ctx.optim_conf["number_of_deferrable_loads"]):
        col_name = f"P_deferrable{k}"
        if col_name not in opt_res_latest.columns:
            ctx.logger.error(f"{col_name} was not found in results DataFrame.")
            continue
        await ctx.rh.post_data(
            opt_res_latest[col_name],
            ctx.idx,
            custom_def[k]["entity_id"],
            "power",
            custom_def[k]["unit_of_measurement"],
            custom_def[k]["friendly_name"],
            type_var="deferrable",
            **ctx.common_kwargs,
        )
        cols.append(col_name)
    return cols


# Fraction of nominal power below which a deferrable load is treated as idle
# ('off') and at/above which it is treated as fully on. Scaling the bands with
# nominal_power keeps the labels meaningful for both small and large loads.
_DEFERRABLE_OFF_FRACTION = 0.01
_DEFERRABLE_ON_FRACTION = 0.99
# Floor for the 'off' band (in W) used when nominal_power is unknown/non-positive,
# so a tiny solver residual still reads as 'off'.
_DEFERRABLE_OFF_FLOOR_W = 1.0


def _deferrable_power_to_state(power: float, nominal_power: float) -> str:
    """Map a scheduled deferrable power to an interpretable command label.

    Generic and convention-free: 'off' when the load is essentially idle, 'on'
    when it runs at (near) its nominal power, and 'variable' for any modulated
    level in between. Users layer their own logic (e.g. mapping 'on' to a switch)
    on top of this label rather than re-deriving it from the raw power forecast.

    The 'off' and 'on' bands are derived from ``nominal_power`` so they scale with
    the load; when ``nominal_power`` is unknown the 'off' band falls back to a
    small fixed floor and only 'off'/'variable' can be distinguished.
    """
    if not np.isfinite(power):
        return "off"
    if nominal_power and nominal_power > 0:
        off_threshold = max(_DEFERRABLE_OFF_FRACTION * nominal_power, _DEFERRABLE_OFF_FLOOR_W)
        if power <= off_threshold:
            return "off"
        if power >= _DEFERRABLE_ON_FRACTION * nominal_power:
            return "on"
        return "variable"
    # Nominal power unknown: only the idle floor is meaningful.
    if power <= _DEFERRABLE_OFF_FLOOR_W:
        return "off"
    return "variable"


async def _publish_deferrable_states(
    ctx: PublishContext, opt_res_latest: pd.DataFrame
) -> list[str]:
    """Publish one interpretable command sensor per deferrable load (opt-in).

    Gated by the ``publish_deferrable_load_states`` option (default off, so the
    zero-config behaviour is unchanged). For each deferrable load this posts a
    string-state sensor ('on'/'off'/'variable') for the current timestep plus the
    full optimized plan as a 'schedule' attribute — a generic, glue-agnostic
    command surface that automations can act on directly.
    """
    cols = []
    if not ctx.optim_conf.get("publish_deferrable_load_states", False):
        return cols
    custom_state = ctx.params["passed_data"].get("custom_deferrable_state_id")
    if not custom_state:
        return cols
    number_of_deferrable_loads = ctx.optim_conf["number_of_deferrable_loads"]
    if len(custom_state) < number_of_deferrable_loads:
        # A short custom_deferrable_state_id means some loads get no command
        # sensor. Warn once rather than silently dropping them, so a mis-sized
        # runtime override is visible instead of failing quietly.
        ctx.logger.warning(
            "custom_deferrable_state_id has %d entries but there are %d deferrable "
            "loads; command sensors will only be published for the first %d load(s).",
            len(custom_state),
            number_of_deferrable_loads,
            len(custom_state),
        )
    nominal = ctx.optim_conf.get("nominal_power_of_deferrable_loads", [])
    for k in range(number_of_deferrable_loads):
        col_name = f"P_deferrable{k}"
        if col_name not in opt_res_latest.columns or k >= len(custom_state):
            continue
        nominal_power = nominal[k] if k < len(nominal) else 0.0
        states = opt_res_latest[col_name].apply(
            lambda power, npow=nominal_power: _deferrable_power_to_state(power, npow)
        )
        await ctx.rh.post_data(
            states,
            ctx.idx,
            custom_state[k]["entity_id"],
            custom_state[k]["device_class"],
            custom_state[k]["unit_of_measurement"],
            custom_state[k]["friendly_name"],
            type_var="categorical",
            **ctx.common_kwargs,
        )
    # The command states are derived (not opt_res columns), so they are
    # published as a side-effect without being added to the returned column set.
    return cols


async def _publish_thermal_variable(
    rh, opt_res_latest, idx, k, custom_ids, col_prefix, type_var, unit_type, kwargs
) -> str | None:
    """Helper to publish a single thermal variable if valid."""
    if custom_ids and k < len(custom_ids):
        col_name = f"{col_prefix}{k}"
        if col_name in opt_res_latest.columns:
            entity_conf = custom_ids[k]
            await rh.post_data(
                opt_res_latest[col_name],
                idx,
                entity_conf["entity_id"],
                unit_type,
                entity_conf["unit_of_measurement"],
                entity_conf["friendly_name"],
                type_var=type_var,
                **kwargs,
            )
            return col_name
    return None


async def _publish_thermal_loads(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish predicted temperature and heating demand for thermal loads."""
    cols = []
    if "custom_predicted_temperature_id" not in ctx.params["passed_data"]:
        return cols
    custom_temp = ctx.params["passed_data"]["custom_predicted_temperature_id"]
    custom_heat = ctx.params["passed_data"].get("custom_heating_demand_id")
    def_load_config = ctx.optim_conf.get("def_load_config", [])
    if not isinstance(def_load_config, list):
        def_load_config = []
    for k in range(ctx.optim_conf["number_of_deferrable_loads"]):
        if k >= len(def_load_config):
            continue
        load_cfg = def_load_config[k]
        if "thermal_config" not in load_cfg and "thermal_battery" not in load_cfg:
            continue
        col_t = await _publish_thermal_variable(
            ctx.rh,
            opt_res_latest,
            ctx.idx,
            k,
            custom_temp,
            "predicted_temp_heater",
            "temperature",
            "temperature",
            ctx.common_kwargs,
        )
        if col_t:
            cols.append(col_t)
        col_h = await _publish_thermal_variable(
            ctx.rh,
            opt_res_latest,
            ctx.idx,
            k,
            custom_heat,
            "heating_demand_heater",
            "energy",
            "energy",
            ctx.common_kwargs,
        )
        if col_h:
            cols.append(col_h)
    return cols


_BATTERY_POWER_ENTITY_TEMPLATE = "sensor.p_batt_forecast_battery{k}"
_BATTERY_SOC_ENTITY_TEMPLATE = "sensor.soc_batt_forecast_battery{k}"
_DEFAULT_SOC_ENTITY_ID = "sensor.soc_batt_forecast"


async def _publish_battery_data(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish Battery Power (fleet total; per-battery at N>1) and SOC
    (bare at N=1, per-battery only at N>1)."""
    cols = []
    if not ctx.optim_conf["set_use_battery"]:
        return cols
    if "P_batt" not in opt_res_latest.columns:
        ctx.logger.error("P_batt was not found in results DataFrame.")
        return cols
    # Power - fleet total. custom_batt_forecast_id overrides this one entity at
    # any N: unchanged code path, so this stays a true no-op at N=1.
    custom_batt = ctx.params["passed_data"]["custom_batt_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_batt"],
        ctx.idx,
        custom_batt["entity_id"],
        "power",
        custom_batt["unit_of_measurement"],
        custom_batt["friendly_name"],
        type_var="batt",
        **ctx.common_kwargs,
    )
    cols.append("P_batt")

    n_batt = utils.validate_num_batteries(ctx.plant_conf)
    if n_batt == 1:
        # N=1: exactly today's entity set, zero new entities. Still
        # overridable via custom_batt_soc_forecast_id.
        # Guard the bare SOC_opt read the same way P_batt is guarded above: a
        # stale N>1 results frame (fleet P_batt + SOC_opt_0/1, no bare
        # SOC_opt) replayed under a config reverted to N=1 must warn+skip
        # SOC, never KeyError.
        if "SOC_opt" not in opt_res_latest.columns:
            ctx.logger.error("SOC_opt was not found in results DataFrame.")
            return cols
        custom_soc = ctx.params["passed_data"]["custom_batt_soc_forecast_id"]
        await ctx.rh.post_data(
            opt_res_latest["SOC_opt"] * 100,
            ctx.idx,
            custom_soc["entity_id"],
            "battery",
            custom_soc["unit_of_measurement"],
            custom_soc["friendly_name"],
            type_var="SOC",
            **ctx.common_kwargs,
        )
        cols.append("SOC_opt")
        return cols

    # N > 1: no bare fleet SOC sensor - SOC has no meaningful fleet aggregate.
    # custom_batt_soc_forecast_id has no natural single target here (it
    # customizes the bare sensor that no longer exists at N>1), so a runtime
    # override is ignored with a one-time warning rather than silently
    # reinterpreted as a prefix. This keeps every per-battery entity id
    # exactly the pinned sensor.*_battery<K> name regardless of a legacy
    # single-battery override, adding no new config surface.
    custom_soc = ctx.params["passed_data"].get("custom_batt_soc_forecast_id") or {}
    if custom_soc.get("entity_id") not in (None, _DEFAULT_SOC_ENTITY_ID):
        ctx.logger.warning(
            "custom_batt_soc_forecast_id override (%s) has no effect when "
            "number_of_batteries=%d: SOC has no meaningful fleet aggregate, so "
            "per-battery SOC always publishes on the fixed "
            "sensor.soc_batt_forecast_battery<K> entity ids.",
            custom_soc.get("entity_id"),
            n_batt,
        )

    for k in range(n_batt):
        p_col = f"P_batt_{k}"
        if p_col in opt_res_latest.columns:
            await ctx.rh.post_data(
                opt_res_latest[p_col],
                ctx.idx,
                _BATTERY_POWER_ENTITY_TEMPLATE.format(k=k),
                "power",
                "W",
                f"Battery Power Forecast Battery {k}",
                type_var="batt",
                **ctx.common_kwargs,
            )
            cols.append(p_col)
        else:
            ctx.logger.error(f"{p_col} was not found in results DataFrame.")

        soc_col = f"SOC_opt_{k}"
        if soc_col in opt_res_latest.columns:
            await ctx.rh.post_data(
                opt_res_latest[soc_col] * 100,
                ctx.idx,
                _BATTERY_SOC_ENTITY_TEMPLATE.format(k=k),
                "battery",
                "%",
                f"Battery SOC Forecast Battery {k}",
                type_var="SOC",
                **ctx.common_kwargs,
            )
            cols.append(soc_col)
        else:
            ctx.logger.error(f"{soc_col} was not found in results DataFrame.")
    return cols


async def _publish_grid_and_costs(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish Grid Power, Costs, and Optimization Status."""
    cols = []
    # Grid
    custom_grid = ctx.params["passed_data"]["custom_grid_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_grid"],
        ctx.idx,
        custom_grid["entity_id"],
        "power",
        custom_grid["unit_of_measurement"],
        custom_grid["friendly_name"],
        type_var="power",
        **ctx.common_kwargs,
    )
    cols.append("P_grid")
    # Cost Function
    custom_cost = ctx.params["passed_data"]["custom_cost_fun_id"]
    col_cost_fun = [i for i in opt_res_latest.columns if "cost_fun_" in i]
    await ctx.rh.post_data(
        opt_res_latest[col_cost_fun],
        ctx.idx,
        custom_cost["entity_id"],
        "monetary",
        custom_cost["unit_of_measurement"],
        custom_cost["friendly_name"],
        type_var="cost_fun",
        **ctx.common_kwargs,
    )
    # Optim Status
    custom_status = ctx.params["passed_data"]["custom_optim_status_id"]
    if "optim_status" not in opt_res_latest:
        opt_res_latest["optim_status"] = "Optimal"
        ctx.logger.warning("no optim_status in opt_res_latest")
    status_val = opt_res_latest["optim_status"]
    await ctx.rh.post_data(
        status_val,
        ctx.idx,
        custom_status["entity_id"],
        "",
        "",
        custom_status["friendly_name"],
        type_var="optim_status",
        **ctx.common_kwargs,
    )
    cols.append("optim_status")
    # Unit Costs
    for key, var_name in [
        ("custom_unit_load_cost_id", "unit_load_cost"),
        ("custom_unit_prod_price_id", "unit_prod_price"),
    ]:
        custom_id = ctx.params["passed_data"][key]
        await ctx.rh.post_data(
            opt_res_latest[var_name],
            ctx.idx,
            custom_id["entity_id"],
            "monetary",
            custom_id["unit_of_measurement"],
            custom_id["friendly_name"],
            type_var=var_name,
            **ctx.common_kwargs,
        )
        cols.append(var_name)
    return cols


async def publish_data(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    opt_res_latest: pd.DataFrame | None = None,
    entity_save: bool | None = False,
    dont_post: bool | None = False,
) -> pd.DataFrame:
    """
    Publish the data obtained from the optimization results.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: If True we will read data from optimization results in dayahead CSV file
    :type save_data_to_file: bool, optional
    :return: The output data of the optimization readed from a CSV file in the data folder
    :rtype: pd.DataFrame
    :param entity_save: Save built entities to data_path/entities
    :type entity_save: bool, optional
    :param dont_post: Do not post to Home Assistant. Works with entity_save
    :type dont_post: bool, optional

    """
    logger.info("Publishing data to HASS instance")
    # Parse Parameters
    params = _get_params(input_data_dict)
    # Check for Entity Publishing (Prefix mode)
    publish_prefix = params["passed_data"].get("publish_prefix", "")
    if not save_data_to_file and publish_prefix != "" and not dont_post:
        opt_res = await _publish_from_saved_entities(input_data_dict, logger, params)
        if opt_res is not None:
            opt_res.attrs["emhass_schema_version"] = EMHASS_SCHEMA_VERSION
            return opt_res
    # Load Optimization Results (if not passed)
    if opt_res_latest is None:
        opt_res_latest = _load_opt_res_latest(input_data_dict, logger, save_data_to_file)
        if opt_res_latest is None:
            return None
    # A failed/infeasible optimization yields a results frame with only the
    # optim_status column (see Optimization.perform_optimization); the forecast
    # and battery columns are absent. Publishing it would crash on the first
    # lookup (e.g. opt_res_latest["P_Load"]). Surface the failure instead.
    if "P_Load" not in opt_res_latest.columns:
        status = "unknown"
        if "optim_status" in opt_res_latest.columns and not opt_res_latest.empty:
            status = opt_res_latest["optim_status"].iloc[0]
        logger.error(
            "Optimization result is incomplete (status: %s); nothing to publish. "
            "Run a successful optimization before publishing.",
            status,
        )
        return None
    # Determine Closest Index
    idx_closest = _get_closest_index(input_data_dict["retrieve_hass_conf"], opt_res_latest.index)
    # Create Context
    common_kwargs = {
        "publish_prefix": publish_prefix,
        "save_entities": entity_save,
        "dont_post": dont_post,
    }
    ctx = PublishContext(
        input_data_dict=input_data_dict,
        params=params,
        idx=idx_closest,
        common_kwargs=common_kwargs,
        logger=logger,
    )
    # Publish Data Components
    cols_published = []
    cols_published.extend(await _publish_standard_forecasts(ctx, opt_res_latest))
    cols_published.extend(await _publish_deferrable_loads(ctx, opt_res_latest))
    cols_published.extend(await _publish_deferrable_states(ctx, opt_res_latest))
    cols_published.extend(await _publish_thermal_loads(ctx, opt_res_latest))
    cols_published.extend(await _publish_battery_data(ctx, opt_res_latest))
    cols_published.extend(await _publish_grid_and_costs(ctx, opt_res_latest))
    # Return Summary DataFrame
    opt_res = opt_res_latest[cols_published].loc[[opt_res_latest.index[idx_closest]]]
    opt_res.attrs["emhass_schema_version"] = EMHASS_SCHEMA_VERSION
    return opt_res


async def continual_publish(
    input_data_dict: dict, entity_path: pathlib.Path, logger: logging.Logger
):
    """
    If continual_publish is true and a entity file saved in /data_path/entities, continually publish sensor on freq rate, updating entity current state value based on timestamp

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param entity_path: Path for entities folder in data_path
    :type entity_path: Path
    :param logger: The passed logger object
    :type logger: logging.Logger
    """
    logger.info("Continual publish thread service started")
    freq = input_data_dict["retrieve_hass_conf"].get(
        "optimization_time_step", pd.to_timedelta(1, "minutes")
    )
    while True:
        # Sleep for x seconds (using current time as a reference for time left)
        time_zone = input_data_dict["retrieve_hass_conf"]["time_zone"]
        timestamp_diff = freq.total_seconds() - (datetime.now(time_zone).timestamp() % 60)
        sleep_seconds = max(0.0, min(timestamp_diff, 60.0))
        await asyncio.sleep(sleep_seconds)
        # Delegate processing to helper function to reduce complexity. A
        # transient failure in a single cycle (e.g. a half-written entity file
        # read mid-publish) must not kill the background task: log it and retry
        # on the next interval, otherwise published sensors freeze until restart.
        try:
            freq = await _publish_and_update_freq(input_data_dict, entity_path, logger, freq)
        except asyncio.CancelledError:
            # Task cancellation (e.g. shutdown) must propagate, never be retried.
            # CancelledError is a BaseException so the broad except below would
            # not catch it anyway; this makes that intent explicit.
            raise
        except Exception:
            logger.exception("continual_publish cycle failed; retrying next interval")
    return False


async def _publish_and_update_freq(input_data_dict, entity_path, logger, current_freq):
    """
    Helper to process entity publishing and frequency updates.
    Returns the (potentially updated) frequency.
    """
    # Guard clause: if path doesn't exist, do nothing and return current freq
    if not os.path.exists(entity_path):
        return current_freq
    entity_path_contents = os.listdir(entity_path)
    # Guard clause: if directory is empty, do nothing
    if not entity_path_contents:
        return current_freq
    # Loop through all saved entity files
    for entity in entity_path_contents:
        # Skip metadata and any in-flight atomic-write temp files: entity and
        # metadata files are committed via a "<name>.<...>.tmp" + os.replace, and
        # that temp file can be momentarily visible to this directory listing.
        if entity != default_metadata_json and not entity.endswith(".tmp"):
            await publish_json(
                entity,
                input_data_dict,
                entity_path,
                logger,
                "continual_publish",
            )
    # Retrieve entity metadata from file
    metadata_file = entity_path / default_metadata_json
    if os.path.isfile(metadata_file):
        async with aiofiles.open(metadata_file) as file:
            content = await file.read()
            metadata = orjson.loads(content)
            # Check if freq should be shorter
            if metadata.get("lowest_time_step") is not None:
                return pd.to_timedelta(metadata["lowest_time_step"], "minutes")
    return current_freq


async def publish_json(
    entity: dict,
    input_data_dict: dict,
    entity_path: pathlib.Path,
    logger: logging.Logger,
    reference: str | None = "",
):
    """
    Extract saved entity data from .json (in data_path/entities), build entity, post results to post_data

    :param entity: json file containing entity data
    :type entity: dict
    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param entity_path: Path for entities folder in data_path
    :type entity_path: Path
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param reference: String for identifying who ran the function
    :type reference: str, optional

    """
    # Retrieve entity metadata from file
    if os.path.isfile(entity_path / default_metadata_json):
        async with aiofiles.open(entity_path / default_metadata_json) as file:
            content = await file.read()
            metadata = orjson.loads(content)
    else:
        logger.error(f"unable to locate metadata.json in: {entity_path}")
        return False
    # Round current timecode (now)
    now_precise = datetime.now(input_data_dict["retrieve_hass_conf"]["time_zone"]).replace(
        second=0, microsecond=0
    )
    # Retrieve entity data from file
    entity_data = pd.read_json(entity_path / entity, orient="index")
    # Remove ".json" from string for entity_id
    entity_id = entity.replace(".json", "")
    # Adjust Dataframe from received entity json file
    entity_data.columns = [metadata[entity_id]["name"]]
    entity_data.index.name = "timestamp"
    entity_data.index = pd.to_datetime(entity_data.index).tz_convert(
        input_data_dict["retrieve_hass_conf"]["time_zone"]
    )
    entity_data.index.freq = pd.to_timedelta(
        int(metadata[entity_id]["optimization_time_step"]), "minutes"
    )
    # Calculate the current state value
    if input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "nearest":
        idx_closest = entity_data.index.get_indexer([now_precise], method="nearest")[0]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "first":
        idx_closest = entity_data.index.get_indexer([now_precise], method="ffill")[0]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "last":
        idx_closest = entity_data.index.get_indexer([now_precise], method="bfill")[0]
    if idx_closest == -1:
        idx_closest = entity_data.index.get_indexer([now_precise], method="nearest")[0]
    # Call post data
    if reference == "continual_publish":
        logger.debug("Auto Published sensor:")
        logger_levels = "DEBUG"
    else:
        logger_levels = "INFO"
    # post/save entity
    await input_data_dict["rh"].post_data(
        data_df=entity_data[metadata[entity_id]["name"]],
        idx=idx_closest,
        entity_id=entity_id,
        device_class=dict.get(metadata[entity_id], "device_class"),
        unit_of_measurement=metadata[entity_id]["unit_of_measurement"],
        friendly_name=metadata[entity_id]["friendly_name"],
        type_var=metadata[entity_id].get("type_var", ""),
        save_entities=False,
        logger_levels=logger_levels,
    )
    return entity_data[metadata[entity_id]["name"]]


async def main():
    r"""Define the main command line entry function.

    This function may take several arguments as inputs. You can type `emhass --help` to see the list of options:

    - action: Set the desired action, options are: perfect-optim, dayahead-optim,
      naive-mpc-optim, publish-data, forecast-model-fit, forecast-model-predict, forecast-model-tune,
      forecast-calibration

    - config: Define path to the config.yaml file

    - costfun: Define the type of cost function, options are: profit, cost, self-consumption

    - log2file: Define if we should log to a file or not

    - params: Configuration parameters passed from data/options.json if using the add-on

    - runtimeparams: Pass runtime optimization parameters as dictionnary

    - debug: Use True for testing purposes

    """
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        help="Set the desired action, options are: perfect-optim, dayahead-optim,\
        naive-mpc-optim, publish-data, forecast-model-fit, forecast-model-predict, forecast-model-tune,\
        forecast-calibration",
    )
    parser.add_argument(
        "--config", type=str, help="Define path to the config.json/defaults.json file"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="String of configuration parameters passed",
    )
    parser.add_argument("--data", type=str, help="Define path to the Data files (.csv & .pkl)")
    parser.add_argument("--root", type=str, help="Define path emhass root")
    parser.add_argument(
        "--costfun",
        type=str,
        default="profit",
        help="Define the type of cost function, options are: profit, cost, self-consumption",
    )
    parser.add_argument(
        "--log2file",
        type=bool,
        default=False,
        help="Define if we should log to a file or not",
    )
    parser.add_argument(
        "--secrets",
        type=str,
        default=None,
        help="Define secret parameter file (secrets_emhass.yaml) path",
    )
    parser.add_argument(
        "--runtimeparams",
        type=str,
        default=None,
        help="Pass runtime optimization parameters as dictionnary",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Use True for testing purposes",
    )
    args = parser.parse_args()

    # The path to the configuration files
    if args.config is not None:
        config_path = pathlib.Path(args.config)
    else:
        config_path = pathlib.Path(str(utils.get_root(__file__, num_parent=3) / "config.json"))
    if args.data is not None:
        data_path = pathlib.Path(args.data)
    else:
        data_path = config_path.parent / "data/"
    if args.root is not None:
        root_path = pathlib.Path(args.root)
    else:
        root_path = utils.get_root(__file__, num_parent=1)
    if args.secrets is not None:
        secrets_path = pathlib.Path(args.secrets)
    else:
        secrets_path = pathlib.Path(config_path.parent / "secrets_emhass.yaml")

    associations_path = root_path / "data/associations.csv"
    defaults_path = root_path / "data/config_defaults.json"

    emhass_conf = {}
    emhass_conf["config_path"] = config_path
    emhass_conf["data_path"] = data_path
    emhass_conf["root_path"] = root_path
    emhass_conf["associations_path"] = associations_path
    emhass_conf["defaults_path"] = defaults_path
    # create logger
    logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=bool(args.log2file))

    # Check paths
    logger.debug("config path: " + str(config_path))
    logger.debug("data path: " + str(data_path))
    logger.debug("root path: " + str(root_path))
    if not associations_path.exists():
        logger.error("Could not find associations.csv file in: " + str(associations_path))
        logger.error("Try setting config file path with --associations")
        return False
    if not config_path.exists():
        logger.warning("Could not find config.json file in: " + str(config_path))
        logger.warning("Try setting config file path with --config")
    if not secrets_path.exists():
        logger.warning("Could not find secrets file in: " + str(secrets_path))
        logger.warning("Try setting secrets file path with --secrets")
    if not os.path.isdir(data_path):
        logger.error("Could not find data folder in: " + str(data_path))
        logger.error("Try setting data path with --data")
        return False
    if not os.path.isdir(root_path):
        logger.error("Could not find emhass/src folder in: " + str(root_path))
        logger.error("Try setting emhass root path with --root")
        return False

    # Additional argument
    try:
        parser.add_argument(
            "--version",
            action="version",
            version="%(prog)s " + version("emhass"),
        )
        args = parser.parse_args()
    except Exception:
        logger.info(
            "Version not found for emhass package. Or importlib exited with PackageNotFoundError.",
        )

    # Setup config
    config = {}
    # Check if passed config file is yaml of json, build config accordingly
    if config_path.exists():
        # Safe: Use pathlib's suffix instead of regex to avoid ReDoS
        file_extension = config_path.suffix.lstrip(".").lower()

        if file_extension:
            match file_extension:
                case "json":
                    config = await utils.build_config(
                        emhass_conf, logger, defaults_path, config_path
                    )
                case "yaml" | "yml":
                    config = await utils.build_config(
                        emhass_conf, logger, defaults_path, config_path=config_path
                    )
                case _:
                    logger.warning(
                        f"Unsupported config file format: .{file_extension}, building parameters with only defaults"
                    )
                    config = await utils.build_config(emhass_conf, logger, defaults_path)
        else:
            logger.warning("Config file has no extension, building parameters with only defaults")
            config = await utils.build_config(emhass_conf, logger, defaults_path)
    else:
        # If unable to find config file, use only defaults_config.json
        logger.warning("Unable to obtain config.json file, building parameters with only defaults")
        config = await utils.build_config(emhass_conf, logger, defaults_path)
    if type(config) is bool and not config:
        raise Exception("Failed to find default config")

    # Obtain secrets from secrets_emhass.yaml?
    params_secrets = {}
    emhass_conf, built_secrets = await utils.build_secrets(
        emhass_conf, logger, secrets_path=secrets_path
    )
    params_secrets.update(built_secrets)

    # Build params
    params = await utils.build_params(emhass_conf, params_secrets, config, logger)
    if type(params) is bool:
        raise Exception("A error has occurred while building parameters")
    # Add any passed params from args to params
    if args.params:
        params.update(orjson.loads(args.params))

    input_data_dict = await set_input_data_dict(
        emhass_conf,
        args.costfun,
        orjson.dumps(params).decode("utf-8"),
        args.runtimeparams,
        args.action,
        logger,
        args.debug,
    )
    if type(input_data_dict) is bool:
        raise Exception("A error has occurred while creating action objects")

    # Perform selected action
    if args.action == "perfect-optim":
        opt_res = await perfect_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "dayahead-optim":
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "naive-mpc-optim":
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "forecast-model-fit":
        df_fit_pred, df_fit_pred_backtest, mlf = await forecast_model_fit(
            input_data_dict, logger, debug=args.debug
        )
        opt_res = None
    elif args.action == "forecast-model-predict":
        if args.debug:
            _, _, mlf = await forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred = await forecast_model_predict(input_data_dict, logger, debug=args.debug, mlf=mlf)
        opt_res = None
    elif args.action == "forecast-model-tune":
        if args.debug:
            _, _, mlf = await forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred_optim, mlf = await forecast_model_tune(
            input_data_dict, logger, debug=args.debug, mlf=mlf
        )
        opt_res = None
    elif args.action == "forecast-calibration":
        await forecast_calibration(input_data_dict, logger)
        opt_res = None
    elif args.action == "regressor-model-fit":
        mlr = await regressor_model_fit(input_data_dict, logger, debug=args.debug)
        opt_res = None
    elif args.action == "regressor-model-predict":
        if args.debug:
            mlr = await regressor_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlr = None
        prediction = await regressor_model_predict(
            input_data_dict, logger, debug=args.debug, mlr=mlr
        )
        opt_res = None
    elif args.action == "export-influxdb-to-csv":
        success = await export_influxdb_to_csv(input_data_dict, logger)
        opt_res = None
    elif args.action == "publish-data":
        opt_res = await publish_data(input_data_dict, logger)
    else:
        logger.error("The passed action argument is not valid")
        logger.error(
            "Try setting --action: perfect-optim, dayahead-optim, naive-mpc-optim, forecast-model-fit, forecast-model-predict, forecast-model-tune, forecast-calibration, export-influxdb-to-csv or publish-data"
        )
        opt_res = None
    logger.info(opt_res)
    # Flush the logger
    ch.close()
    logger.removeHandler(ch)
    if (
        args.action == "perfect-optim"
        or args.action == "dayahead-optim"
        or args.action == "naive-mpc-optim"
        or args.action == "publish-data"
    ):
        return opt_res
    elif args.action == "forecast-model-fit":
        return df_fit_pred, df_fit_pred_backtest, mlf
    elif args.action == "forecast-model-predict":
        return df_pred
    elif args.action == "regressor-model-fit":
        return mlr
    elif args.action == "regressor-model-predict":
        return prediction
    elif args.action == "export-influxdb-to-csv":
        return success
    elif args.action == "forecast-model-tune":
        return df_pred_optim, mlf
    else:
        return opt_res


def main_sync():
    """Sync wrapper for async main function - used as CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
