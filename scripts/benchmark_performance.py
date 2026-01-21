#!/usr/bin/env python3
"""
Benchmark Performance Script
============================

This script compares the performance of the new CVXPY-based optimization
(Define-Once architecture) against the legacy PuLP-based implementation
(Rebuild-Every-Loop architecture).

DEPENDENCIES:
-------------
Since the main EMHASS project has removed PuLP, you must install the legacy
dependencies manually to run this benchmark:

1. Install PuLP (Required for Legacy class):
   $ pip install pulp

2. (Optional) Install HiGHS Solver (Recommended for fair speed comparison):
   $ pip install highspy numpy pandas cvxpy

   Note:
   - If HiGHS is not installed, the benchmark will fall back to:
     - Legacy: CBC (Bundled with PuLP on most systems)
     - New:    OSQP or CLARABEL (Bundled with CVXPY)
   - To use HiGHS with PuLP, ensure the 'highs' executable is in your system PATH.
"""

import asyncio
import copy
import logging
import pathlib
import sys
import time

import cvxpy as cp
import numpy as np
import orjson
import pandas as pd

# --- PATH SETUP & IMPORTS ---
# 1. Determine Project Root (Assuming scripts/benchmark_performance.py)
script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent

# 2. Add 'src' to path so we can import 'emhass'
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))

# 3. Add 'scripts' to path so we can import 'optimization_legacy'
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# 4. Imports
try:
    import pulp  # Check for benchmark dependency
except ImportError:
    print("ERROR: 'pulp' is missing.")
    print("Please run: pip install pulp")
    sys.exit(1)

try:
    # New Class (from src)
    # Legacy Class (from scripts/optimization_legacy.py)
    from optimization_legacy import Optimization as OptimizationLegacy

    from emhass.optimization import Optimization as OptimizationNew

    # Utils
    from emhass.utils import (
        build_config,
        build_params,
        build_secrets,
        get_yaml_parse,
    )
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("Ensure 'optimization_legacy.py' is in the 'scripts/' folder.")
    sys.exit(1)


async def setup_benchmark_config_async():
    """Load configuration exactly like the unit tests."""
    # Adjusted root for scripts/ folder structure
    emhass_conf = {
        "data_path": project_root / "data/",
        "root_path": project_root / "src/emhass/",
        "defaults_path": project_root / "src/emhass/data/config_defaults.json",
        "associations_path": project_root / "src/emhass/data/associations.csv",
    }

    # Quiet logger
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.ERROR)
    if not logger.handlers:
        ch = logging.StreamHandler()
        logger.addHandler(ch)

    if emhass_conf["defaults_path"].exists():
        config = await build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await build_secrets(emhass_conf, logger, no_response=True)
        params = await build_params(emhass_conf, secrets, config, logger)
    else:
        print(f"Warning: config_defaults.json not found at {emhass_conf['defaults_path']}")
        params = {}

    params_json = orjson.dumps(params).decode("utf-8")
    retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(params_json, logger)

    return retrieve_hass_conf, optim_conf, plant_conf, emhass_conf, logger


def generate_input_data(num_timesteps=48):
    """Generate dummy data for the benchmark."""
    dates = pd.date_range(start="2024-01-01", periods=num_timesteps, freq="30min")
    df = pd.DataFrame(index=dates)
    np.random.seed(42)
    df["p_pv_forecast"] = np.clip(
        np.random.normal(2000, 1000, num_timesteps) * np.sin(np.linspace(0, 3.14, num_timesteps)),
        0,
        5000,
    )
    df["p_load_forecast"] = np.clip(np.random.normal(1000, 300, num_timesteps), 500, 3000)
    df["unit_load_cost"] = np.random.uniform(0.10, 0.40, num_timesteps)
    df["unit_prod_price"] = np.random.uniform(0.05, 0.15, num_timesteps)
    return df


def apply_benchmark_overrides(optim_conf, plant_conf):
    """Force specific settings for the benchmark."""
    optim_conf.update(
        {
            "set_use_battery": True,
            "set_nocharge_from_grid": False,
            "set_battery_dynamic": False,
            "set_nodischarge_to_grid": False,
            "set_total_pv_sell": False,
            "set_deferrable_startup_penalty": [0.0],
            "number_of_deferrable_loads": 1,
            "nominal_power_of_deferrable_loads": [3000.0],
            "operating_hours_of_each_deferrable_load": [0],
            "start_timesteps_of_each_deferrable_load": [0],
            "end_timesteps_of_each_deferrable_load": [0],
            "treat_deferrable_load_as_semi_cont": [True],
            "set_deferrable_load_single_constant": [False],
            "def_current_state": [False],
            "def_load_config": [{}],  # Empty dict avoids triggering thermal logic
            "lp_solver_timeout": 30,
            "num_threads": 4,
        }
    )

    plant_conf.update(
        {
            "inverter_is_hybrid": True,
            "compute_curtailment": False,
            "inverter_ac_output_max": 5000,
            "inverter_ac_input_max": 5000,
            "battery_nominal_energy_capacity": 5000,
            "battery_discharge_power_max": 1000,
            "battery_charge_power_max": 1000,
            "maximum_power_from_grid": 9000,
            "maximum_power_to_grid": 9000,
        }
    )


def run_benchmark():
    print(f"Benchmark Script Path: {pathlib.Path(__file__).resolve()}")

    # 1. Setup
    print("Setting up configuration...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    retrieve_conf, optim_conf, plant_conf, emhass_conf, logger = loop.run_until_complete(
        setup_benchmark_config_async()
    )
    apply_benchmark_overrides(optim_conf, plant_conf)

    # 2. Check Solvers
    pulp_highs_available = False
    try:
        if pulp.getSolver("HiGHS").available():
            pulp_highs_available = True
    except Exception:
        pass

    cvxpy_highs_available = "HIGHS" in cp.installed_solvers()

    # 3. Configure
    optim_conf_legacy = copy.deepcopy(optim_conf)
    optim_conf_new = copy.deepcopy(optim_conf)

    # Legacy: Try HiGHS, fallback to CBC
    if pulp_highs_available:
        optim_conf_legacy["lp_solver"] = "HiGHS"
    else:
        optim_conf_legacy["lp_solver"] = "PULP_CBC_CMD"
    optim_conf_legacy["lp_solver_path"] = "empty"

    # New: Try HiGHS, fallback to CBC or OSQP
    if cvxpy_highs_available:
        optim_conf_new["lp_solver"] = "HIGHS"
    elif "CBC" in cp.installed_solvers():
        optim_conf_new["lp_solver"] = "CBC"
    else:
        optim_conf_new["lp_solver"] = "OSQP"

    print(f"Legacy Solver (PuLP): {optim_conf_legacy['lp_solver']}")
    print(f"New Solver (CVXPY):   {optim_conf_new['lp_solver']}")

    # 4. Init
    print("-" * 40)
    print("Initializing Legacy Class...", end="", flush=True)
    try:
        opt_legacy = OptimizationLegacy(
            retrieve_conf,
            optim_conf_legacy,
            plant_conf,
            "unit_load_cost",
            "unit_prod_price",
            "profit",
            emhass_conf,
            logger,
        )
        print(" Done.")
    except Exception as e:
        print(f"\nERROR initializing Legacy: {e}")
        return

    print("Initializing New Class...   ", end="", flush=True)
    try:
        opt_new = OptimizationNew(
            retrieve_conf,
            optim_conf_new,
            plant_conf,
            "unit_load_cost",
            "unit_prod_price",
            "profit",
            emhass_conf,
            logger,
        )
        print(" Done.")
    except Exception as e:
        print(f"\nERROR initializing New: {e}")
        return

    # 5. Benchmark
    ITERATIONS = 50
    df_input = generate_input_data()
    p_pv = df_input["p_pv_forecast"].values
    p_load = df_input["p_load_forecast"].values
    cost = df_input["unit_load_cost"].values
    prod = df_input["unit_prod_price"].values

    print(f"\n--- Running {ITERATIONS} MPC Iterations ---")

    # Legacy Loop
    print("Legacy (PuLP): ", end="", flush=True)
    start_time_legacy = time.time()
    for i in range(ITERATIONS):
        if i % 5 == 0:
            print(".", end="", flush=True)
        perturbation = np.random.uniform(0.9, 1.1, len(p_pv))
        opt_legacy.perform_optimization(
            df_input,
            p_pv * perturbation,
            p_load * perturbation,
            cost,
            prod,
            soc_init=0.5,
            soc_final=0.5,
        )
    end_time_legacy = time.time()
    print(" Done.")

    # New Loop
    print("New (CVXPY):   ", end="", flush=True)
    start_time_new = time.time()
    for i in range(ITERATIONS):
        if i % 5 == 0:
            print(".", end="", flush=True)
        perturbation = np.random.uniform(0.9, 1.1, len(p_pv))
        opt_new.perform_optimization(
            df_input,
            p_pv * perturbation,
            p_load * perturbation,
            cost,
            prod,
            soc_init=0.5,
            soc_final=0.5,
        )
    end_time_new = time.time()
    print(" Done.")

    # 6. Stats
    total_legacy = end_time_legacy - start_time_legacy
    total_new = end_time_new - start_time_new

    print("\n" + "=" * 40)
    print(f"BENCHMARK RESULTS ({ITERATIONS} Runs)")
    print("=" * 40)
    print(f"Legacy (PuLP):  {total_legacy:.4f} s  ({total_legacy / ITERATIONS:.4f} s/iter)")
    print(f"New (CVXPY):    {total_new:.4f} s  ({total_new / ITERATIONS:.4f} s/iter)")
    print("-" * 40)

    if total_new > 0:
        print(f"SPEEDUP: {total_legacy / total_new:.2f}x FASTER")
    else:
        print("Speedup: Infinite")


if __name__ == "__main__":
    run_benchmark()
