#!/usr/bin/env python3
"""Offline local-venv scenario runner and live WebUI launcher for EMHASS.

Runs 9 representative EMHASS documentation / study-case scenarios using the
repository's local offline CSV fixtures (no Home Assistant connection or `uv`
required), builds the full ``injection_dict`` WebUI payloads, writes standalone
HTML previews under ``data/webui_previews/``, and optionally launches the live
Quart WebUI server pre-loaded with any scenario.

Examples (using your activated local virtualenv, no ``uv`` needed):
    python scripts/webui_test_scenarios.py --list
    python scripts/webui_test_scenarios.py --all
    python scripts/webui_test_scenarios.py --scenario pv_battery --serve --port 5000
    python scripts/webui_test_scenarios.py --scenario federated_coordinator --serve
"""

from __future__ import annotations

import argparse
import asyncio
import html
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emhass import command_line, last_run, utils  # noqa: E402

OFFLINE_CSV_FIXTURES = (
    "data_load_forecast.csv",
    "data_load_cost_forecast.csv",
    "data_prod_price_forecast.csv",
    "data_weather_forecast.csv",
    "heating_prediction.csv",
    "test_df_final.pkl",
    "long_train_data.pkl",
)

DEFAULT_PREVIEW_ROOT = REPO_ROOT / "data" / "webui_previews"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    title: str
    description: str
    action: str
    config_overrides: dict[str, Any]
    runtime_params: dict[str, Any]


def _make_daily_curve(
    n_steps: int, base: float, amp: float, peak_frac: float = 0.52
) -> list[float]:
    x = np.linspace(0.0, 1.0, n_steps, endpoint=False)
    bump = np.exp(-((x - peak_frac) ** 2) / 0.025)
    return [round(float(max(0.0, base + amp * val)), 2) for val in bump]


def _make_tou_prices(n_steps: int, offpeak: float = 0.14, peak: float = 0.32) -> list[float]:
    prices: list[float] = []
    for i in range(n_steps):
        frac = (i % 48) / 48.0
        is_peak = (0.29 <= frac <= 0.46) or (0.71 <= frac <= 0.92)
        prices.append(peak if is_peak else offpeak)
    return prices


SCENARIOS: dict[str, ScenarioSpec] = {
    "basic_no_pv": ScenarioSpec(
        name="basic_no_pv",
        title="Basic Consumption (No PV, No Battery)",
        description="Zero-config baseline: load + TOU tariff + 1 deferrable load without solar or battery.",
        action="dayahead-optim",
        config_overrides={
            "costfun": "cost",
            "set_use_pv": False,
            "set_use_battery": False,
            "number_of_deferrable_loads": 1,
            "nominal_power_of_deferrable_loads": [2000.0],
            "operating_hours_of_each_deferrable_load": [3],
            "start_timesteps_of_each_deferrable_load": [0],
            "end_timesteps_of_each_deferrable_load": [0],
            "treat_deferrable_load_as_semi_cont": [True],
            "set_deferrable_load_single_constant": [False],
            "set_deferrable_startup_penalty": [0.0],
            "load_forecast_method": "csv",
            "load_cost_forecast_method": "csv",
            "production_price_forecast_method": "csv",
        },
        runtime_params={
            "load_cost_forecast": _make_tou_prices(48, 0.15, 0.34),
            "prod_price_forecast": [0.0] * 48,
        },
    ),
    "basic_pv": ScenarioSpec(
        name="basic_pv",
        title="Rooftop PV + Deferrable Loads (No Battery)",
        description="Day-ahead scheduling shifting 2 deferrable appliances into daytime solar generation.",
        action="dayahead-optim",
        config_overrides={
            "costfun": "profit",
            "set_use_pv": True,
            "set_use_battery": False,
            "weather_forecast_method": "csv",
            "load_forecast_method": "csv",
            "load_cost_forecast_method": "csv",
            "production_price_forecast_method": "csv",
            "number_of_deferrable_loads": 2,
            "nominal_power_of_deferrable_loads": [3000.0, 750.0],
            "operating_hours_of_each_deferrable_load": [4, 2],
            "start_timesteps_of_each_deferrable_load": [0, 0],
            "end_timesteps_of_each_deferrable_load": [0, 0],
            "treat_deferrable_load_as_semi_cont": [True, True],
            "set_deferrable_load_single_constant": [False, False],
            "set_deferrable_startup_penalty": [0.0, 0.0],
        },
        runtime_params={
            "pv_power_forecast": _make_daily_curve(48, 0.0, 4800.0),
            "load_cost_forecast": _make_tou_prices(48, 0.16, 0.35),
            "prod_price_forecast": [0.08] * 48,
        },
    ),
    "pv_battery": ScenarioSpec(
        name="pv_battery",
        title="PV + Home Battery Storage",
        description="Full solar + battery arbitrage & self-consumption optimization with battery charging source breakdown.",
        action="dayahead-optim",
        config_overrides={
            "costfun": "profit",
            "set_use_pv": True,
            "set_use_battery": True,
            "number_of_batteries": 1,
            "battery_nominal_energy_capacity": [7500.0],
            "battery_discharge_power_max": [2500.0],
            "battery_charge_power_max": [2500.0],
            "battery_minimum_state_of_charge": [0.2],
            "battery_maximum_state_of_charge": [0.95],
            "battery_target_state_of_charge": [0.5],
            "set_nocharge_from_grid": False,
            "set_nodischarge_to_grid": False,
            "weather_forecast_method": "csv",
            "load_forecast_method": "csv",
            "load_cost_forecast_method": "csv",
            "production_price_forecast_method": "csv",
            "number_of_deferrable_loads": 2,
            "nominal_power_of_deferrable_loads": [2400.0, 900.0],
            "operating_hours_of_each_deferrable_load": [3, 2],
            "start_timesteps_of_each_deferrable_load": [0, 0],
            "end_timesteps_of_each_deferrable_load": [0, 0],
            "treat_deferrable_load_as_semi_cont": [True, True],
            "set_deferrable_load_single_constant": [False, False],
            "set_deferrable_startup_penalty": [0.0, 0.0],
        },
        runtime_params={
            "pv_power_forecast": _make_daily_curve(48, 0.0, 5600.0),
            "load_cost_forecast": _make_tou_prices(48, 0.14, 0.38),
            "prod_price_forecast": [0.07] * 48,
        },
    ),
    "mpc_pv_battery": ScenarioSpec(
        name="mpc_pv_battery",
        title="Rolling MPC (Short Horizon + SOC Targets)",
        description="Short 16-step receding-horizon Naive MPC cycle with explicit initial/final battery SOC.",
        action="naive-mpc-optim",
        config_overrides={
            "costfun": "profit",
            "set_use_pv": True,
            "set_use_battery": True,
            "number_of_batteries": 1,
            "battery_nominal_energy_capacity": [6000.0],
            "battery_discharge_power_max": [2000.0],
            "battery_charge_power_max": [2000.0],
            "weather_forecast_method": "csv",
            "load_forecast_method": "csv",
            "load_cost_forecast_method": "csv",
            "production_price_forecast_method": "csv",
            "number_of_deferrable_loads": 1,
            "nominal_power_of_deferrable_loads": [1800.0],
            "operating_hours_of_each_deferrable_load": [2],
            "start_timesteps_of_each_deferrable_load": [0],
            "end_timesteps_of_each_deferrable_load": [0],
            "treat_deferrable_load_as_semi_cont": [True],
            "set_deferrable_load_single_constant": [False],
            "set_deferrable_startup_penalty": [0.0],
        },
        runtime_params={
            "prediction_horizon": 16,
            "soc_init": 0.42,
            "soc_final": 0.65,
            "pv_power_forecast": _make_daily_curve(16, 200.0, 4200.0, peak_frac=0.45),
            "load_power_forecast": _make_daily_curve(16, 450.0, 900.0, peak_frac=0.7),
            "load_cost_forecast": _make_tou_prices(16, 0.16, 0.36),
            "prod_price_forecast": [0.06] * 16,
            "operating_hours_of_each_deferrable_load": [2],
        },
    ),
    "ev_charging": ScenarioSpec(
        name="ev_charging",
        title="EV Variable-Power Charging + Home Battery",
        description="Continuous-power EV charger alongside a semi-continuous home appliance and battery.",
        action="dayahead-optim",
        config_overrides={
            "costfun": "cost",
            "set_use_pv": True,
            "set_use_battery": True,
            "number_of_batteries": 1,
            "battery_nominal_energy_capacity": [10000.0],
            "battery_discharge_power_max": [3000.0],
            "battery_charge_power_max": [3000.0],
            "weather_forecast_method": "csv",
            "load_forecast_method": "csv",
            "load_cost_forecast_method": "csv",
            "production_price_forecast_method": "csv",
            "number_of_deferrable_loads": 2,
            "nominal_power_of_deferrable_loads": [7400.0, 1200.0],
            "minimum_power_of_deferrable_loads": [1400.0, 0.0],
            "operating_hours_of_each_deferrable_load": [4, 2],
            "start_timesteps_of_each_deferrable_load": [0, 4],
            "end_timesteps_of_each_deferrable_load": [48, 30],
            "treat_deferrable_load_as_semi_cont": [False, True],
            "set_deferrable_load_single_constant": [False, False],
            "set_deferrable_startup_penalty": [0.0, 0.05],
        },
        runtime_params={
            "pv_power_forecast": _make_daily_curve(48, 0.0, 6200.0),
            "load_cost_forecast": _make_tou_prices(48, 0.12, 0.36),
            "prod_price_forecast": [0.05] * 48,
        },
    ),
    "simple_thermal": ScenarioSpec(
        name="simple_thermal",
        title="Simple Thermal Zone + Comfort Range",
        description="First-order thermal heater (thermal_config) maintaining Day/Night temperature comfort bounds.",
        action="dayahead-optim",
        config_overrides={
            "costfun": "cost",
            "set_use_pv": True,
            "set_use_battery": False,
            "weather_forecast_method": "csv",
            "load_forecast_method": "csv",
            "load_cost_forecast_method": "csv",
            "production_price_forecast_method": "csv",
            "number_of_deferrable_loads": 2,
            "nominal_power_of_deferrable_loads": [2000.0, 800.0],
            "operating_hours_of_each_deferrable_load": [0, 2],
            "start_timesteps_of_each_deferrable_load": [0, 0],
            "end_timesteps_of_each_deferrable_load": [0, 0],
            "treat_deferrable_load_as_semi_cont": [True, True],
            "set_deferrable_load_single_constant": [False, False],
            "set_deferrable_startup_penalty": [0.0, 0.0],
            "def_load_config": [
                {
                    "thermal_config": {
                        "heating_rate": 5.0,
                        "cooling_constant": 0.12,
                        "thermal_inertia": 0.5,
                        "start_temperature": 20.5,
                        "sense": "heat",
                        "min_temperatures": [
                            19.0 if (i % 48) < 14 or (i % 48) >= 44 else 20.5 for i in range(48)
                        ],
                        "max_temperatures": [24.0] * 48,
                    }
                },
                {},
            ],
        },
        runtime_params={
            "pv_power_forecast": _make_daily_curve(48, 0.0, 3500.0),
            "load_cost_forecast": _make_tou_prices(48, 0.14, 0.34),
            "prod_price_forecast": [0.06] * 48,
            "outdoor_temperature_forecast": [
                round(float(10.0 + 6.0 * np.sin((i / 48.0 - 0.25) * np.pi * 2)), 1)
                for i in range(48)
            ],
        },
    ),
    "heat_pump_thermal_battery": ScenarioSpec(
        name="heat_pump_thermal_battery",
        title="Air-to-Water Heat Pump (thermal_battery)",
        description="Physics-based thermal battery model with Carnot COP and building envelope dynamics.",
        action="dayahead-optim",
        config_overrides={
            "costfun": "cost",
            "set_use_pv": True,
            "set_use_battery": True,
            "number_of_batteries": 1,
            "battery_nominal_energy_capacity": [5000.0],
            "battery_discharge_power_max": [1800.0],
            "battery_charge_power_max": [1800.0],
            "weather_forecast_method": "csv",
            "load_forecast_method": "csv",
            "load_cost_forecast_method": "csv",
            "production_price_forecast_method": "csv",
            "number_of_deferrable_loads": 2,
            "nominal_power_of_deferrable_loads": [3200.0, 1000.0],
            "operating_hours_of_each_deferrable_load": [0, 2],
            "start_timesteps_of_each_deferrable_load": [0, 0],
            "end_timesteps_of_each_deferrable_load": [0, 0],
            "treat_deferrable_load_as_semi_cont": [False, True],
            "set_deferrable_load_single_constant": [False, False],
            "set_deferrable_startup_penalty": [0.0, 0.0],
            "def_load_config": [
                {
                    "thermal_battery": {
                        "supply_temperature": 35.0,
                        "volume": 14.91,
                        "start_temperature": 21.5,
                        "min_temperatures": [20.0] * 48,
                        "max_temperatures": [25.0] * 48,
                        "carnot_efficiency": 0.45,
                        "u_value": 0.326,
                        "envelope_area": 649.4,
                        "ventilation_rate": 0.12,
                        "heated_volume": 573.5,
                    }
                },
                {},
            ],
        },
        runtime_params={
            "pv_power_forecast": _make_daily_curve(48, 0.0, 4400.0),
            "load_cost_forecast": _make_tou_prices(48, 0.14, 0.35),
            "prod_price_forecast": [0.06] * 48,
            "outdoor_temperature_forecast": [
                round(float(7.0 + 5.0 * np.sin((i / 48.0 - 0.25) * np.pi * 2)), 1)
                for i in range(48)
            ],
        },
    ),
    "federated_coordinator": ScenarioSpec(
        name="federated_coordinator",
        title="Federated Dantzig-Wolfe Coordinator",
        description="Decentralized device-grouped decomposition outputting shadow prices, gaps, and device value shares.",
        action="dayahead-optim",
        config_overrides={
            "costfun": "profit",
            "optimization_backend": "dantzig_wolfe",
            "site": [
                {"id": "battery0", "parent": "grid"},
                {"id": "deferrable0", "parent": "grid", "group": "controllable_loads"},
                {"id": "deferrable1", "parent": "grid", "group": "controllable_loads"},
            ],
            "set_use_pv": True,
            "set_use_battery": True,
            "number_of_batteries": 1,
            "battery_nominal_energy_capacity": [6000.0],
            "battery_discharge_power_max": [2000.0],
            "battery_charge_power_max": [2000.0],
            "set_nocharge_from_grid": False,
            "set_nodischarge_to_grid": False,
            "set_battery_dynamic": False,
            "inverter_is_hybrid": False,
            "weather_forecast_method": "csv",
            "load_forecast_method": "csv",
            "load_cost_forecast_method": "csv",
            "production_price_forecast_method": "csv",
            "number_of_deferrable_loads": 2,
            "nominal_power_of_deferrable_loads": [3000.0, 750.0],
            "operating_hours_of_each_deferrable_load": [4, 2],
            "start_timesteps_of_each_deferrable_load": [0, 0],
            "end_timesteps_of_each_deferrable_load": [0, 0],
            "treat_deferrable_load_as_semi_cont": [True, True],
            "set_deferrable_load_single_constant": [False, False],
            "set_deferrable_startup_penalty": [0.0, 0.0],
        },
        runtime_params={
            "pv_power_forecast": _make_daily_curve(48, 0.0, 5200.0),
            "load_cost_forecast": _make_tou_prices(48, 0.15, 0.36),
            "prod_price_forecast": [0.07] * 48,
        },
    ),
    "pv_p10_p50_quantiles": ScenarioSpec(
        name="pv_p10_p50_quantiles",
        title="PV Uncertainty Quantiles (P50 + P10 Blend)",
        description="External P50 and conservative P10 PV forecasts blended via weather_forecast_pv_quantile_bias.",
        action="dayahead-optim",
        config_overrides={
            "costfun": "profit",
            "set_use_pv": True,
            "set_use_battery": True,
            "number_of_batteries": 1,
            "battery_nominal_energy_capacity": [5000.0],
            "battery_discharge_power_max": [2000.0],
            "battery_charge_power_max": [2000.0],
            "weather_forecast_method": "csv",
            "weather_forecast_pv_quantile_bias": 0.35,
            "load_forecast_method": "csv",
            "load_cost_forecast_method": "csv",
            "production_price_forecast_method": "csv",
            "number_of_deferrable_loads": 2,
            "nominal_power_of_deferrable_loads": [2200.0, 900.0],
            "operating_hours_of_each_deferrable_load": [3, 2],
            "start_timesteps_of_each_deferrable_load": [0, 0],
            "end_timesteps_of_each_deferrable_load": [0, 0],
            "treat_deferrable_load_as_semi_cont": [True, True],
            "set_deferrable_load_single_constant": [False, False],
            "set_deferrable_startup_penalty": [0.0, 0.0],
        },
        runtime_params={
            "pv_power_forecast": _make_daily_curve(48, 0.0, 5800.0),
            "pv_power_forecast_p10": _make_daily_curve(48, 0.0, 3600.0),
            "load_cost_forecast": _make_tou_prices(48, 0.14, 0.34),
            "prod_price_forecast": [0.06] * 48,
        },
    ),
    "thermal_model_tuning": ScenarioSpec(
        name="thermal_model_tuning",
        title="Thermal Model Validation & Auto-Tuning",
        description="Predicted vs actual indoor temperature trajectory check and curve_fit auto-tuning across thermal inertia lags.",
        action="thermal-model-calibration",
        config_overrides={
            "costfun": "cost",
            "set_use_pv": False,
            "set_use_battery": False,
            "number_of_deferrable_loads": 1,
            "nominal_power_of_deferrable_loads": [3300.0],
            "def_load_config": [
                {
                    "thermal_config": {
                        "heating_rate": 0.65,
                        "cooling_constant": 0.025,
                        "thermal_inertia": 0.5,
                    }
                }
            ],
        },
        runtime_params={
            "load_index": 0,
            "nominal_power": 3300.0,
            "lookback_hours": 24,
            "current_heating_rate": 0.65,
            "current_cooling_constant": 0.025,
            "current_thermal_inertia": 0.0,
        },
    ),
}


def _copy_offline_fixtures(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    src_data = REPO_ROOT / "data"
    for fname in OFFLINE_CSV_FIXTURES:
        src = src_data / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)


def _build_system_badges_html(config: dict[str, Any]) -> str:
    badges: list[tuple[str, str]] = []
    if config.get("set_use_pv"):
        wf = config.get("weather_forecast_method")
        badges.append(("sys-pv", f"PV Plant ({wf})" if wf else "PV Plant"))
    else:
        badges.append(("sys-muted", "No PV"))
    if config.get("set_use_battery"):
        nb = int(config.get("number_of_batteries") or 1)
        badges.append(("sys-batt", f"{nb} Batteries" if nb > 1 else "1 Battery"))
    else:
        badges.append(("sys-muted", "No Battery"))
    ndef = int(config.get("number_of_deferrable_loads") or 0)
    badges.append(
        (
            "sys-loads" if ndef > 0 else "sys-muted",
            f"{ndef} Deferrable Load{'s' if ndef != 1 else ''}",
        )
    )
    def_cfgs = config.get("def_load_config") or []
    thermal_n = len(
        [
            entry
            for entry in def_cfgs
            if isinstance(entry, dict)
            and (entry.get("thermal_config") or entry.get("thermal_battery"))
        ]
    )
    if thermal_n > 0:
        badges.append(("sys-thermal", f"{thermal_n} Thermal Load{'s' if thermal_n != 1 else ''}"))
    n_inv = int(config.get("number_of_inverters") or 1)
    badges.append(
        (
            "sys-neutral",
            f"{n_inv} Inverters"
            if n_inv > 1
            else ("Hybrid Inverter" if config.get("inverter_is_hybrid") else "1 Inverter"),
        )
    )
    cap_cost = config.get("capacity_cost_per_kw")
    has_cap = (
        any(float(v) > 0 for v in cap_cost)
        if isinstance(cap_cost, list)
        else bool(cap_cost and float(cap_cost) > 0)
    )
    if has_cap:
        badges.append(("sys-tariff", "Capacity Tariff"))
    if config.get("costfun"):
        badges.append(("sys-neutral", f"Objective: {config['costfun']}"))
    if config.get("load_forecast_method"):
        badges.append(("sys-load", f"Load: {config['load_forecast_method']}"))
    if config.get("optimization_backend") == "dantzig_wolfe":
        badges.append(("sys-fed", "Federated Coordinator"))
    return '<span class="system-overview-title">Configured system:</span>' + "".join(
        f'<span class="sys-badge {cls}">{html.escape(text)}</span>' for cls, text in badges
    )


def _render_standalone_preview(
    spec: ScenarioSpec,
    injection_dict: dict[str, Any],
    out_html: Path,
    version_str: str,
    config: dict[str, Any] | None = None,
    optim_status: str = "Optimal",
) -> None:
    """Write a self-contained static HTML page showing the scenario WebUI plots and tables."""
    css_path = SRC_DIR / "emhass" / "static" / "style.css"
    style_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    sections = [
        f'<div class="table_div" data-plot-key="{html.escape(key)}">{val}</div>'
        for key, val in injection_dict.items()
    ]
    body_html = "\n".join(sections)
    sys_badges_html = _build_system_badges_html(config or spec.config_overrides)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>EMHASS WebUI Preview — {html.escape(spec.title)}</title>
  <style>
{style_css}
  </style>
</head>
<body>
  <header class="app-header">
    <div class="brand-group">
      <div class="brand-text">
        <h1 class="app-title">EMHASS Preview — {html.escape(spec.title)}</h1>
        <span class="app-subtitle">{html.escape(spec.description)}</span>
      </div>
    </div>
    <div class="header-badges">
      <span class="badge version-badge">EMHASS v{html.escape(version_str)}</span>
      <span class="badge status-badge status-ok">{html.escape(spec.action)}</span>
      <span class="badge status-badge status-ok">Solver: {html.escape(optim_status)}</span>
    </div>
  </header>
  <main class="page-shell">
    <div class="system-overview-strip">{sys_badges_html}</div>
    <section class="card results-shell">
      <nav id="results-tabs" class="results-tabs" aria-label="Optimization result views">
        <button type="button" class="results-tab-btn active" data-results-tab="schedule">Main Schedule <span class="tab-count"></span></button>
        <button type="button" class="results-tab-btn" data-results-tab="analytics">Advanced Analytics <span class="tab-count"></span></button>
        <button type="button" class="results-tab-btn" data-results-tab="federated">Federated Coordinator <span class="tab-count"></span></button>
        <button type="button" class="results-tab-btn" data-results-tab="performance">Performance &amp; Times <span class="tab-count"></span></button>
        <button type="button" class="results-tab-btn" data-results-tab="tables">Data Tables <span class="tab-count"></span></button>
        <button type="button" class="results-tab-btn" data-results-tab="all">All View <span class="tab-count"></span></button>
      </nav>
      <div id="template" class="template-results">
        {body_html}
      </div>
    </section>
  </main>
  <script>
    (function () {{
      var groups = {{
        schedule: ["title", "subsubtitle0", "figure_0", "figure_pv_forecast", "figure_thermal", "figure_1", "figure_battery_sources", "figure_2", "figure_calibration_metrics", "figure_thermal_lag_comparison", "figure_thermal_scatter"],
        analytics: ["analytics_intro", "figure_energy_pie", "figure_histograms", "figure_heatmap", "figure_load_effectiveness"],
        performance: ["figure_stage_times"],
        tables: ["subsubtitle1", "table1", "subsubtitle2", "table2"]
      }};
      function tabOf(k) {{
        if (k.indexOf("figure_fed_") === 0) return "federated";
        for (var t in groups) {{ if (groups[t].indexOf(k) !== -1) return t; }}
        return "schedule";
      }}
      var items = Array.prototype.slice.call(document.querySelectorAll("#template > .table_div[data-plot-key]"));
      var btns = Array.prototype.slice.call(document.querySelectorAll(".results-tab-btn"));
      var counts = {{ schedule: 0, analytics: 0, federated: 0, performance: 0, tables: 0, all: 0 }};
      items.forEach(function (el) {{
        var k = el.getAttribute("data-plot-key") || "";
        if (k.indexOf("figure_") === 0 || k.indexOf("table") === 0) {{
          counts[tabOf(k)] += 1;
          counts.all += 1;
        }}
      }});
      btns.forEach(function (b) {{
        var t = b.getAttribute("data-results-tab");
        var span = b.querySelector(".tab-count");
        if (span) span.textContent = counts[t] > 0 ? String(counts[t]) : "";
        if ((t === "federated" || t === "analytics" || t === "performance") && !counts[t]) b.style.display = "none";
        b.addEventListener("click", function () {{
          btns.forEach(function (x) {{ x.classList.toggle("active", x === b); }});
          items.forEach(function (el) {{
            var k = el.getAttribute("data-plot-key") || "";
            el.style.display = (t === "all" || tabOf(k) === t) ? "" : "none";
          }});
          if (window.Plotly && window.Plotly.Plots) {{
            document.querySelectorAll(".js-plotly-plot").forEach(function (p) {{
              try {{ window.Plotly.Plots.resize(p); }} catch (e) {{}}
            }});
          }}
        }});
      }});
      var isSpecial = !document.querySelector('[data-plot-key="figure_2"]') && counts.tables > 0 && counts.schedule > 0;
      btns[isSpecial ? btns.length - 1 : 0].click();
    }})();
  </script>
</body>
</html>
"""
    out_html.write_text(page, encoding="utf-8")


def _apply_scenario_env(scen_dir: Path) -> None:
    os.environ.pop("SUPERVISOR_TOKEN", None)
    os.environ["DATA_PATH"] = str(scen_dir)
    os.environ["ROOT_PATH"] = str(SRC_DIR / "emhass")
    os.environ["CONFIG_PATH"] = str(scen_dir / "config.json")
    os.environ["OPTIONS_PATH"] = str(scen_dir / "options.json")
    os.environ["SECRETS_PATH"] = str(scen_dir / "secrets_emhass.yaml")
    os.environ["LEGACY_CONFIG_PATH"] = str(scen_dir / "config_emhass.yaml")
    os.environ["DEFAULTS_PATH"] = str(SRC_DIR / "emhass" / "data" / "config_defaults.json")
    os.environ["ASSOCIATIONS_PATH"] = str(SRC_DIR / "emhass" / "data" / "associations.csv")


async def _prepare_scenario_workspace(
    spec: ScenarioSpec, preview_root: Path, logger: Any
) -> tuple[Path, dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Copy offline fixtures and build configuration/params files for one scenario."""
    scen_dir = preview_root / spec.name
    _copy_offline_fixtures(scen_dir)
    (scen_dir / "options.json").write_text("{}", encoding="utf-8")
    (scen_dir / "secrets_emhass.yaml").write_text(
        "hass_url: empty\nlong_lived_token: empty\ntime_zone: Europe/Paris\nLatitude: 45.83\nLongitude: 6.86\nAltitude: 4807.8\n",
        encoding="utf-8",
    )
    _apply_scenario_env(scen_dir)

    emhass_conf = {
        "data_path": scen_dir,
        "root_path": SRC_DIR / "emhass",
        "defaults_path": SRC_DIR / "emhass" / "data" / "config_defaults.json",
        "associations_path": SRC_DIR / "emhass" / "data" / "associations.csv",
        "config_path": scen_dir / "config.json",
        "options_path": scen_dir / "options.json",
        "secrets_path": scen_dir / "secrets_emhass.yaml",
        "legacy_config_path": scen_dir / "config_emhass.yaml",
    }
    base_config = orjson.loads(emhass_conf["defaults_path"].read_bytes())
    full_config = {**base_config, "time_zone": "Europe/Paris", **spec.config_overrides}
    (scen_dir / "config.json").write_bytes(orjson.dumps(full_config, option=orjson.OPT_INDENT_2))
    params = await utils.build_params(
        emhass_conf,
        {
            "hass_url": "empty",
            "long_lived_token": "empty",
            "server_ip": "127.0.0.1",
            "time_zone": "Europe/Paris",
            "Latitude": 45.83,
            "Longitude": 6.86,
            "Altitude": 4807.8,
        },
        full_config,
        logger,
    )
    (scen_dir / "params.pkl").write_bytes(pickle.dumps((emhass_conf["config_path"], params)))
    return scen_dir, emhass_conf, full_config, params


async def _execute_scenario_action(
    spec: ScenarioSpec,
    scen_dir: Path,
    emhass_conf: dict[str, Path],
    full_config: dict[str, Any],
    params: dict[str, Any],
    logger: Any,
) -> tuple[dict[str, Any], str, float]:
    """Run calibration or optimization for one scenario and return (injection_dict, optim_status, elapsed)."""
    from emhass import api_auth

    params_json = orjson.dumps(params).decode("utf-8")
    runtime_json = orjson.dumps(spec.runtime_params).decode("utf-8")
    costfun = full_config.get("costfun", "profit")
    t0 = time.perf_counter()

    if spec.action == "thermal-model-calibration":
        calib = await utils.compute_thermal_model_calibration(
            emhass_conf, params, spec.runtime_params, logger, get_data_from_file=True
        )
        elapsed = time.perf_counter() - t0
        optim_status = "Optimal"
        last_run.record(
            scen_dir,
            action=spec.action,
            stage_times={"thermal_calibration": elapsed},
            optim_status=optim_status,
            infeasible=False,
            duration_total_seconds=elapsed,
            schema_version="1.1",
        )
        api_auth.provision_token(scen_dir, logger)
        return utils.get_injection_dict_thermal_calibration(calib), optim_status, elapsed

    input_data_dict = await command_line.set_input_data_dict(
        emhass_conf,
        costfun,
        params_json,
        runtime_json,
        spec.action,
        logger,
        get_data_from_file=True,
    )
    if not input_data_dict:
        raise RuntimeError(f"set_input_data_dict failed for scenario '{spec.name}'")

    if spec.action == "naive-mpc-optim":
        opt_res = await command_line.naive_mpc_optim(input_data_dict, logger)
    else:
        opt_res = await command_line.dayahead_forecast_optim(input_data_dict, logger)

    elapsed = time.perf_counter() - t0
    if not isinstance(opt_res, pd.DataFrame) or opt_res.empty:
        raise RuntimeError(f"Optimization returned empty result for '{spec.name}'")

    stage_times = dict(input_data_dict.get("stage_times") or {})
    optim_status = (
        str(opt_res["optim_status"].iloc[0])
        if "optim_status" in opt_res.columns and not opt_res.empty
        else "Optimal"
    )
    last_run.record(
        scen_dir,
        action=spec.action,
        stage_times=stage_times,
        optim_status=optim_status,
        infeasible=optim_status not in last_run.OK_OPTIM_STATUSES,
        duration_total_seconds=elapsed,
        schema_version="1.1",
    )

    from emhass import web_server as _ws

    api_auth.provision_token(scen_dir, logger)
    _ws.emhass_conf["data_path"] = scen_dir
    _ws._attach_ui_metadata(opt_res, input_data_dict, action_name=spec.action)
    opt_res.to_csv(scen_dir / "opt_res_latest.csv")
    return utils.get_injection_dict(opt_res), optim_status, elapsed


async def _run_scenario_async(
    spec: ScenarioSpec, preview_root: Path
) -> tuple[Path, list[str], float]:
    """Execute one offline scenario end-to-end and persist its WebUI state."""
    logger, ch = utils.get_logger(f"emhass_scenario_{spec.name}", {}, save_to_file=False)
    try:
        scen_dir, emhass_conf, full_config, params = await _prepare_scenario_workspace(
            spec, preview_root, logger
        )
        injection_dict, optim_status, elapsed = await _execute_scenario_action(
            spec, scen_dir, emhass_conf, full_config, params, logger
        )
        (scen_dir / "injection_dict.pkl").write_bytes(pickle.dumps(injection_dict))
        preview_html = preview_root / f"{spec.name}.html"
        _render_standalone_preview(
            spec,
            injection_dict,
            preview_html,
            last_run.emhass_version(),
            config=full_config,
            optim_status=optim_status,
        )
        fig_keys = [k for k in injection_dict if k.startswith("figure_")]
        return scen_dir, fig_keys, elapsed
    finally:
        logger.removeHandler(ch)


def _write_index_html(
    preview_root: Path, results: list[tuple[ScenarioSpec, list[str], float]]
) -> Path:
    cards = []
    for spec, fig_keys, elapsed in results:
        badges = " ".join(f"<code>{html.escape(k)}</code>" for k in fig_keys)
        cards.append(
            f"""<li class="card">
              <h3><a href="{html.escape(spec.name)}.html">{html.escape(spec.title)}</a>
                <small>({html.escape(spec.name)} &bull; {elapsed:.2f}s)</small></h3>
              <p>{html.escape(spec.description)}</p>
              <p><strong>Plots ({len(fig_keys)}):</strong> {badges}</p>
            </li>"""
        )
    index_path = preview_root / "index.html"
    index_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>EMHASS WebUI Scenario Previews</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 1.5rem; background: #f1f5f9; color: #0f172a; }}
    ul {{ list-style: none; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1rem; }}
    .card {{ background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 1rem; }}
    code {{ background: #e0f2fe; padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.78rem; }}
  </style>
</head>
<body>
  <h1>EMHASS WebUI Offline Scenario Previews</h1>
  <p>Launch interactive WebUI for any scenario with:
     <code>python scripts/webui_test_scenarios.py --scenario &lt;name&gt; --serve --port 5000</code></p>
  <ul>{"".join(cards)}</ul>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index_path


async def _serve_scenario(scen_dir: Path, host: str, port: int) -> None:
    _apply_scenario_env(scen_dir)
    os.environ["PORT"] = str(port)

    import uvicorn

    from emhass import api_auth, web_server

    await web_server.initialize({"no_response": True})
    api_auth.configure_now(
        str(scen_dir / "secrets_emhass.yaml"),
        str(scen_dir / "options.json"),
        web_server.app.logger,
        data_dir=str(scen_dir),
    )
    token_file = scen_dir / "emhass_api_token"
    print(
        f"\nLive WebUI running at http://{host}:{port}/  (config: http://{host}:{port}/configuration)",
        flush=True,
    )
    if token_file.exists():
        token_value = token_file.read_text(encoding="utf-8").strip()
        print(f"Sign in at http://{host}:{port}/login using token file: {token_file}", flush=True)
        print(f"API token: {token_value}", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)
    config = uvicorn.Config(
        web_server.app, host=host, port=port, log_level="warning", proxy_headers=False
    )
    server = uvicorn.Server(config)
    await server.serve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EMHASS WebUI test scenarios locally using your Python venv."
    )
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default="pv_battery",
        help="Scenario to run (default: pv_battery).",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all scenarios and build index.html."
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the local EMHASS WebUI server pre-loaded with the selected scenario.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for --serve.")
    parser.add_argument("--port", type=int, default=5000, help="Port for --serve (default: 5000).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PREVIEW_ROOT,
        help="Directory for scenario artifacts and previews (default: data/webui_previews).",
    )
    args = parser.parse_args()

    if args.list:
        for name, spec in SCENARIOS.items():
            print(f"{name:28s} [{spec.action:14s}] - {spec.title}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_specs = list(SCENARIOS.values()) if args.all else [SCENARIOS[args.scenario]]

    async def _runner() -> None:
        summaries: list[tuple[ScenarioSpec, list[str], float]] = []
        active_dir: Path | None = None
        for spec in selected_specs:
            scen_dir, fig_keys, elapsed = await _run_scenario_async(spec, args.output_dir)
            print(
                f"[OK] {spec.name:28s} ({elapsed:.2f}s) -> {len(fig_keys)} figures: {', '.join(fig_keys)}"
            )
            summaries.append((spec, fig_keys, elapsed))
            if spec.name == args.scenario or active_dir is None:
                active_dir = scen_dir
        if args.all:
            idx = _write_index_html(args.output_dir, summaries)
            print(f"\nScenario index written to: {idx}")
        if args.serve and active_dir is not None:
            await _serve_scenario(active_dir, args.host, args.port)

    asyncio.run(_runner())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
