#!/usr/bin/env python
"""Tests for the optional hybrid-inverter power-transfer curves (issue #746).

``inverter_power_curve_dc_ac`` / ``inverter_power_curve_ac_dc`` replace the scalar
``inverter_efficiency_dc_ac`` / ``inverter_efficiency_ac_dc`` of one direction with
an exact piecewise-linear relation between the DC-side power (the optimiser's
``p_dc_ac`` / ``p_ac_dc``) and the AC-side power. Layers covered here:

* config layer: ``utils.validate_inverter_power_curve`` / ``check_inverter_power_curves``
  and the real build_params / runtime entry points;
* constraint layer: exact curve equality, composition with native features,
  default-OFF structural parity, and a negative-import-price adversarial case
  that an inequality relaxation would exploit.

The Optimization builder mirrors test_battery_charge_power_derating.py:
synthetic and self-contained.
"""

import asyncio
import json
import logging
import pathlib

import cvxpy as cp
import numpy as np
import orjson
import pandas as pd
import pytest

from emhass import utils
from emhass.optimization import Optimization

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
VALID_OPTIMAL_STATUSES = ["Optimal", "Optimal (Relaxed)"]

EMHASS_CONF = {
    "data_path": TEST_ROOT / "data/",
    "root_path": TEST_ROOT / "src/emhass/",
    "defaults_path": TEST_ROOT / "src/emhass/data/config_defaults.json",
    "associations_path": TEST_ROOT / "src/emhass/data/associations.csv",
}

DC_AC = "inverter_power_curve_dc_ac"
AC_DC = "inverter_power_curve_ac_dc"
logger = logging.getLogger("inverter_curve_test")

# Synthetic, illustrative curves (not any real hardware).
CHARGE_CURVE = [[0, 0], [1000, 1500], [4000, 4500]]  # lossy first kW, then lossless
DISCHARGE_CURVE = [[0, 0], [1000, 700], [5000, 4500]]


def interp(curve, x):
    pts = np.asarray(curve, dtype=float)
    return np.interp(x, pts[:, 0], pts[:, 1])


# --------------------------------------------------------------------------- #
# Config layer
# --------------------------------------------------------------------------- #


def test_valid_curves_are_accepted_as_tuples():
    assert utils.validate_inverter_power_curve(CHARGE_CURVE, AC_DC, "ac_dc") == [
        (0.0, 0.0),
        (1000.0, 1500.0),
        (4000.0, 4500.0),
    ]
    assert utils.validate_inverter_power_curve(DISCHARGE_CURVE, DC_AC, "dc_ac")[-1] == (
        5000.0,
        4500.0,
    )


def test_lossless_curve_is_valid():
    assert utils.validate_inverter_power_curve([[0, 0], [5000, 5000]], DC_AC, "dc_ac")


@pytest.mark.parametrize(
    "curve,direction,expected",
    [
        ("0,0;1,1", "dc_ac", "at least 2"),  # wrong type
        ([[0, 0]], "dc_ac", "at least 2"),  # one point
        ([[0, 0], [1000]], "dc_ac", "pair"),  # wrong cardinality
        ([[0, 0], [1000, 900, 5]], "dc_ac", "pair"),  # wrong cardinality
        ([[0, 0], ["1000", 900]], "dc_ac", "expected a number"),  # wrong type
        ([[0, 0], [True, 900]], "dc_ac", "expected a number"),  # bool is not a power
        ([[0, 0], [float("nan"), 900]], "dc_ac", "finite"),
        ([[0, 0], [1000, -5]], "dc_ac", "non-negative"),
        ([[10, 5], [1000, 900]], "dc_ac", "first point must be [0, 0]"),  # no zero anchor
        ([[0, 0], [1000, 900], [800, 850]], "dc_ac", "ascend strictly"),  # DC not ascending
        ([[0, 0], [1000, 900], [1000, 950]], "dc_ac", "ascend strictly"),  # duplicate DC
        ([[0, 0], [1000, 900], [2000, 900]], "dc_ac", "rise strictly"),  # AC flat
        ([[0, 0], [1000, 900], [2000, 800]], "dc_ac", "rise strictly"),  # non-monotone AC
        ([[0, 0], [1000, 1100]], "dc_ac", "more AC power than the DC"),  # gain on discharge
        ([[0, 0], [1000, 900]], "ac_dc", "more DC power than the AC"),  # gain on charge
    ],
)
def test_malformed_curve_is_rejected_with_a_named_reason(curve, direction, expected):
    with pytest.raises(ValueError) as err:
        utils.validate_inverter_power_curve(curve, "my_param", direction)
    message = str(err.value)
    assert "my_param" in message
    assert expected in message


def test_absent_and_empty_curves_are_the_legacy_default(caplog):
    for conf in ({}, {DC_AC: [], AC_DC: []}, {DC_AC: None}):
        with caplog.at_level(logging.WARNING):
            utils.check_inverter_power_curves(conf, logger)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_curve_requires_a_hybrid_inverter():
    with pytest.raises(ValueError, match="inverter_is_hybrid"):
        utils.check_inverter_power_curves(
            {"inverter_is_hybrid": False, DC_AC: DISCHARGE_CURVE}, logger
        )


def test_curve_ending_below_the_ac_limit_warns(caplog):
    conf = {
        "inverter_is_hybrid": True,
        "inverter_ac_output_max": 15000,
        DC_AC: DISCHARGE_CURVE,
    }
    with caplog.at_level(logging.WARNING):
        utils.check_inverter_power_curves(conf, logger)
    assert any("supported power domain" in r.message for r in caplog.records)


def _build_params(overrides: dict) -> dict:
    config = json.loads(EMHASS_CONF["defaults_path"].read_text(encoding="utf-8"))
    config.update(overrides)

    async def _build():
        _, secrets = await utils.build_secrets(EMHASS_CONF, logger, no_response=True)
        return await utils.build_params(EMHASS_CONF, secrets, config, logger)

    return asyncio.run(_build())


def test_default_config_has_empty_curves_and_unchanged_scalars():
    params = _build_params({})
    plant_conf = params["plant_conf"]
    assert plant_conf[DC_AC] == [] and plant_conf[AC_DC] == []
    assert plant_conf["inverter_efficiency_dc_ac"] == 1.0
    assert plant_conf["inverter_efficiency_ac_dc"] == 1.0


def test_legacy_scalar_only_config_builds_unchanged():
    params = _build_params(
        {
            "inverter_is_hybrid": True,
            "inverter_efficiency_dc_ac": 0.983,
            "inverter_efficiency_ac_dc": 0.97,
        }
    )
    assert params["plant_conf"]["inverter_efficiency_dc_ac"] == 0.983
    assert params["plant_conf"][DC_AC] == []


def test_valid_curves_pass_through_build_params():
    params = _build_params(
        {"inverter_is_hybrid": True, DC_AC: DISCHARGE_CURVE, AC_DC: CHARGE_CURVE}
    )
    assert params["plant_conf"][DC_AC] == DISCHARGE_CURVE
    assert params["plant_conf"][AC_DC] == CHARGE_CURVE


def test_invalid_curve_fails_build_params():
    with pytest.raises(ValueError, match=AC_DC):
        _build_params({"inverter_is_hybrid": True, AC_DC: [[0, 0], [1000, 900]]})


def test_runtime_curve_is_validated_on_the_runtime_path():
    base = _build_params({"inverter_is_hybrid": True})
    params_json = orjson.dumps(base).decode("utf-8")
    rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)

    async def _treat(runtime):
        return await utils.treat_runtimeparams(
            orjson.dumps(runtime).decode("utf-8"),
            params_json,
            rh_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            EMHASS_CONF,
        )

    *_, treated_plant = asyncio.run(_treat({AC_DC: CHARGE_CURVE}))
    assert treated_plant[AC_DC] == CHARGE_CURVE
    with pytest.raises(ValueError, match=AC_DC):
        asyncio.run(_treat({AC_DC: [[0, 0], [1000, 900]]}))


# --------------------------------------------------------------------------- #
# Constraint layer
# --------------------------------------------------------------------------- #

CHARGE_MAX = 5000
CAP = 10000


def build_optimization(
    plant_overrides=None, optim_overrides=None, n_batt=1, cls=Optimization
) -> Optimization:
    """Self-contained hybrid-inverter Optimization builder (no deferrable loads)."""
    build_logger = logging.getLogger("inverter_curve_build")
    build_logger.handlers = []
    build_logger.addHandler(logging.NullHandler())

    retrieve_hass_conf = {
        "optimization_time_step": pd.to_timedelta(30, "minutes"),
        "time_zone": "Europe/Tallinn",
        "sensor_power_photovoltaics": "pv",
        "sensor_power_load_no_var_loads": "load",
    }
    optim_conf = {
        "delta_forecast_daily": pd.Timedelta(hours=5),
        "num_threads": 0,
        "set_use_battery": True,
        "set_use_pv": True,
        "set_total_pv_sell": False,
        "set_nocharge_from_grid": False,
        "set_nodischarge_to_grid": False,
        "set_battery_dynamic": False,
        "set_battery_first_priority": False,
        "battery_dynamic_max": 0.9,
        "battery_dynamic_min": -0.9,
        "weight_battery_discharge": 0.0,
        "weight_battery_charge": 0.0,
        "battery_soc_deficit_threshold": 0.2,
        "battery_soc_deficit_cost": 0.0,
        "battery_soc_surplus_threshold": 0.9,
        "battery_soc_surplus_cost": 0.0,
        "number_of_deferrable_loads": 0,
        "nominal_power_of_deferrable_loads": [],
        "treat_deferrable_load_as_semi_cont": [],
        "set_deferrable_load_single_constant": [],
        "set_deferrable_startup_penalty": [],
        "operating_hours_of_each_deferrable_load": [],
        "start_timesteps_of_each_deferrable_load": [],
        "end_timesteps_of_each_deferrable_load": [],
        "lp_solver_timeout": 45,
        "lp_solver_mip_rel_gap": 0,
    }
    if optim_overrides:
        optim_conf.update(optim_overrides)
    plant_conf = {
        "inverter_is_hybrid": True,
        "inverter_ac_output_max": 5000,
        "inverter_ac_input_max": 5000,
        "inverter_efficiency_dc_ac": 1.0,
        "inverter_efficiency_ac_dc": 1.0,
        "compute_curtailment": False,
        "maximum_power_from_grid": 50000,
        "maximum_power_to_grid": 50000,
        "battery_discharge_power_max": CHARGE_MAX,
        "battery_charge_power_max": CHARGE_MAX,
        "battery_minimum_state_of_charge": 0.05,
        "battery_maximum_state_of_charge": 0.95,
        "battery_target_state_of_charge": 0.5,
        "battery_nominal_energy_capacity": CAP,
        "battery_discharge_efficiency": 1.0,
        "battery_charge_efficiency": 1.0,
        "battery_stress_cost": 0.0,
        "battery_stress_segments": 10,
    }
    if n_batt > 1:
        plant_conf["number_of_batteries"] = n_batt
        for key in (
            "battery_discharge_power_max",
            "battery_charge_power_max",
            "battery_minimum_state_of_charge",
            "battery_maximum_state_of_charge",
            "battery_target_state_of_charge",
            "battery_nominal_energy_capacity",
            "battery_discharge_efficiency",
            "battery_charge_efficiency",
        ):
            plant_conf[key] = [plant_conf[key]] * n_batt
    if plant_overrides:
        plant_conf.update(plant_overrides)
    return cls(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        {"root_path": TEST_ROOT / "src" / "emhass", "data_path": TEST_ROOT / "data"},
        build_logger,
        opt_time_delta=4,
    )


def _frame(import_price, export_price=0.02, pv=0.0, load=300.0):
    n = len(import_price)
    index = pd.date_range("2026-03-04", periods=n, freq="30min", tz="Europe/Tallinn")
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = import_price
    df_input["unit_prod_price"] = [export_price] * n
    return df_input, pd.Series([pv] * n, index=index), pd.Series([load] * n, index=index)


CHEAP_THEN_DEAR = [0.05] * 4 + [0.60] * 4


def _solve(
    plant=None, optim=None, frame=None, soc_init=0.1, soc_final=0.1, n_batt=1, cls=Optimization
):
    opt = build_optimization(plant, optim, n_batt, cls)
    df_input, p_pv, p_load = frame if frame is not None else _frame(CHEAP_THEN_DEAR)
    if n_batt > 1:
        soc_init, soc_final = [soc_init] * n_batt, [soc_final] * n_batt
    res = opt.perform_dayahead_forecast_optim(
        df_input, p_pv, p_load, soc_init=soc_init, soc_final=soc_final
    )
    assert opt.optim_status in VALID_OPTIMAL_STATUSES
    return opt, res


def _assert_curve_equality(res, charge_curve=None, discharge_curve=None, atol=2.0):
    """With no PV the DC bus is the battery: the curve must hold at every step."""
    batt = res["P_batt"].to_numpy()
    hybrid = res["P_hybrid_inverter"].to_numpy()
    charging, discharging = batt < -1.0, batt > 1.0
    if charge_curve is not None:
        np.testing.assert_allclose(
            hybrid[charging], -interp(charge_curve, -batt[charging]), atol=atol
        )
    if discharge_curve is not None:
        np.testing.assert_allclose(
            hybrid[discharging], interp(discharge_curve, batt[discharging]), atol=atol
        )
    return charging.sum(), discharging.sum()


def test_flat_curves_equal_the_scalar_efficiencies():
    scalar = {"inverter_efficiency_dc_ac": 0.95, "inverter_efficiency_ac_dc": 0.95}
    _, res_scalar = _solve(scalar)
    flat = {
        DC_AC: [[0, 0], [10000, 9500]],
        AC_DC: [[0, 0], [9500, 10000]],  # 95 % charge: 10000 W AC -> 9500 W DC
    }
    _, res_curve = _solve(flat)
    for column in ("P_hybrid_inverter", "P_batt", "P_grid", "SOC_opt"):
        np.testing.assert_allclose(
            res_curve[column].to_numpy(), res_scalar[column].to_numpy(), atol=1.0
        )


def test_two_segment_charge_curve_is_obeyed_exactly():
    opt, res = _solve({AC_DC: CHARGE_CURVE})
    n_charge, _ = _assert_curve_equality(res, charge_curve=CHARGE_CURVE)
    assert n_charge > 0, "scenario must actually charge"
    assert opt.vars["inv_curve_ac_dc_u"][0].shape == (8,)


def test_two_segment_discharge_curve_is_obeyed_exactly():
    _, res = _solve({DC_AC: DISCHARGE_CURVE})
    _, n_discharge = _assert_curve_equality(res, discharge_curve=DISCHARGE_CURVE)
    assert n_discharge > 0, "scenario must actually discharge"


def test_both_directions_enabled():
    _, res = _solve({AC_DC: CHARGE_CURVE, DC_AC: DISCHARGE_CURVE})
    n_charge, n_discharge = _assert_curve_equality(res, CHARGE_CURVE, DISCHARGE_CURVE)
    assert n_charge > 0 and n_discharge > 0


@pytest.mark.parametrize("dc_w", [250, 1000, 2500, 4000])  # interior, breakpoint, interior, end
def test_pinned_charge_power_lands_on_the_curve(dc_w):
    """A single step with a fixed SOC change pins the DC power: segment-exact AC input."""
    soc_final = 0.1 + dc_w * 0.5 / CAP  # one 30-minute step, battery efficiency 1
    _, res = _solve({AC_DC: CHARGE_CURVE}, frame=_frame([0.2]), soc_init=0.1, soc_final=soc_final)
    assert res["P_batt"].iloc[0] == pytest.approx(-dc_w, abs=1.0)
    assert res["P_hybrid_inverter"].iloc[0] == pytest.approx(-interp(CHARGE_CURVE, dc_w), abs=1.0)


@pytest.mark.parametrize("dc_w", [250, 1000, 3000, 5000])  # interior, breakpoint, interior, end
def test_pinned_discharge_power_lands_on_the_curve(dc_w):
    soc_final = 0.9 - dc_w * 0.5 / CAP
    _, res = _solve(
        {DC_AC: DISCHARGE_CURVE}, frame=_frame([0.2]), soc_init=0.9, soc_final=soc_final
    )
    assert res["P_batt"].iloc[0] == pytest.approx(dc_w, abs=1.0)
    assert res["P_hybrid_inverter"].iloc[0] == pytest.approx(
        interp(DISCHARGE_CURVE, dc_w), abs=1.0
    )


def test_idle_has_zero_power_both_sides():
    """Flat prices: nothing to do, so zero DC <-> zero AC; no standby term exists."""
    _, res = _solve({AC_DC: CHARGE_CURVE, DC_AC: DISCHARGE_CURVE}, frame=_frame([0.2] * 6))
    np.testing.assert_allclose(res["P_batt"].to_numpy(), 0.0, atol=1.0)
    np.testing.assert_allclose(res["P_hybrid_inverter"].to_numpy(), 0.0, atol=1.0)


def test_curve_domain_caps_dc_power():
    """The last curve point is the supported domain: no extrapolation beyond 2000 W DC."""
    short = [[0, 0], [1000, 1100], [2000, 2300]]
    _, res = _solve({AC_DC: short})
    assert (-res["P_batt"]).max() <= 2000.0 + 1.0
    _assert_curve_equality(res, charge_curve=short)


def test_ac_limits_apply_on_the_ac_side_of_a_curved_direction():
    wide = [[0, 0], [8000, 7600]]
    _, res = _solve(
        {DC_AC: wide, "inverter_ac_output_max": 3000, "battery_discharge_power_max": 8000}
    )
    assert res["P_hybrid_inverter"].max() <= 3000.0 + 1.0
    _, res = _solve(
        {
            AC_DC: [[0, 0], [8000, 8400]],
            "inverter_ac_input_max": 2500,
            "battery_charge_power_max": 8000,
        }
    )
    assert (-res["P_hybrid_inverter"]).max() <= 2500.0 + 1.0


def test_charge_derating_remains_a_power_ceiling_with_a_curve():
    derating = [[0.5, 0.4]]  # above 50 % SOC: 40 % of 5000 W
    _, res = _solve(
        {AC_DC: CHARGE_CURVE, "battery_charge_power_derating": derating},
        soc_init=0.6,
        soc_final=0.6,
    )
    assert (-res["P_batt"]).max() <= 0.4 * CHARGE_MAX + 1.0
    _assert_curve_equality(res, charge_curve=CHARGE_CURVE)


def test_soc_and_battery_power_limits_hold_with_curves():
    plant = {AC_DC: CHARGE_CURVE, DC_AC: DISCHARGE_CURVE}
    _, res = _solve(plant)
    assert res["SOC_opt"].max() <= 0.95 + 1e-6 and res["SOC_opt"].min() >= 0.05 - 1e-6
    assert res["P_batt"].max() <= CHARGE_MAX + 1.0 and (-res["P_batt"]).max() <= CHARGE_MAX + 1.0


def test_nodischarge_to_grid_still_binds_with_a_discharge_curve():
    frame = _frame([0.5] * 6, export_price=0.4, pv=3000.0, load=300.0)
    _, res = _solve(
        {DC_AC: DISCHARGE_CURVE},
        {"set_nodischarge_to_grid": True},
        frame=frame,
        soc_init=0.9,
        soc_final=0.5,
    )
    exporting = res["P_grid"].to_numpy() < -1.0
    assert (res["P_batt"].to_numpy()[exporting] <= 1.0).all()


def test_two_batteries_share_one_inverter_curve():
    """The curve belongs to the single hybrid inverter and acts on the aggregate DC bus."""
    opt, res = _solve({AC_DC: CHARGE_CURVE}, n_batt=2)
    n_charge, _ = _assert_curve_equality(res, charge_curve=CHARGE_CURVE)
    assert n_charge > 0
    assert len([k for k in opt.vars if k.startswith("inv_curve")]) == 2  # one set, not per battery


# ---- default-OFF structural parity ------------------------------------------ #


def _structure(opt):
    return (
        sorted(v.name() for v in opt.prob.variables()),
        len(opt.prob.constraints),
    )


def test_default_off_graph_is_identical_for_absent_empty_and_none():
    structures, results = [], []
    for overrides in ({}, {DC_AC: [], AC_DC: []}, {DC_AC: None, AC_DC: None}):
        scalars = {"inverter_efficiency_dc_ac": 0.983, "inverter_efficiency_ac_dc": 0.97}
        opt, res = _solve({**scalars, **overrides})
        structures.append(_structure(opt))
        results.append(res[["P_hybrid_inverter", "P_batt", "P_grid", "SOC_opt"]].to_numpy())
        assert not [k for k in opt.vars if k.startswith("inv_curve")]
    assert structures[0] == structures[1] == structures[2]
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


def test_one_curved_direction_leaves_the_other_scalar():
    opt, res = _solve({AC_DC: CHARGE_CURVE, "inverter_efficiency_dc_ac": 0.9})
    assert "inv_curve_dc_ac_u" not in opt.vars and "inv_curve_ac_dc_u" in opt.vars
    batt = res["P_batt"].to_numpy()
    discharging = batt > 1.0
    np.testing.assert_allclose(
        res["P_hybrid_inverter"].to_numpy()[discharging], 0.9 * batt[discharging], atol=1.0
    )


def test_malformed_curve_fails_the_model_build():
    opt = build_optimization({DC_AC: [[0, 0], [1000, 1200]]})
    df_input, p_pv, p_load = _frame(CHEAP_THEN_DEAR)
    with pytest.raises(ValueError, match=DC_AC):
        opt.perform_dayahead_forecast_optim(df_input, p_pv, p_load, soc_init=0.1, soc_final=0.1)


# ---- negative-price adversarial test ---------------------------------------- #


def _negative_price_frame():
    # Paid to import, battery already full: the only way to "earn" more is to import
    # AC that no physical DC power accounts for.
    return _frame([-0.20] * 6, export_price=0.0, pv=0.0, load=300.0)


# Closed loopholes: with export priced at 0 and a negative import price, cycling the battery
# (discharge, export for free, re-import and be paid) is legitimately profitable in ANY model,
# legacy scalar included. The adversarial case therefore forbids discharge and export, leaving
# one way to earn more: import AC power that no DC power accounts for.
NEGATIVE_PRICE_PLANT = {
    AC_DC: CHARGE_CURVE,
    DC_AC: DISCHARGE_CURVE,
    "inverter_ac_input_max": 4000,
    "battery_discharge_power_max": 0,
    "maximum_power_to_grid": 0,
}


class RelaxedOptimization(Optimization):
    """Control: the real model with the AC->DC equality weakened to ac_in >= curve(dc)."""

    def _add_inverter_pwl_transfer(self, constraints, name, dc_var, curve, gate):
        exact = super()._add_inverter_pwl_transfer(constraints, name, dc_var, curve, gate)
        if name != "inv_curve_ac_dc":
            return exact
        return exact + cp.Variable(self.num_timesteps, nonneg=True, name="relax_slack")


def test_negative_import_price_cannot_exploit_the_curve():
    """PWL_EQUALITY_EXACT / NEGATIVE_PRICE_EXPLOIT = NONE through the real solver."""
    opt, res = _solve(
        NEGATIVE_PRICE_PLANT, frame=_negative_price_frame(), soc_init=0.95, soc_final=0.95
    )
    # NO_ARTIFICIAL_EXTRA_AC_IMPORT: import equals the load, nothing more.
    assert res["P_grid_pos"].max() <= 300.0 + 2.0
    # NO_ARTIFICIAL_DC_POWER: a full battery cannot absorb DC.
    assert (-res["P_batt"]).max() <= 1.0
    p_ac_dc = np.asarray(opt.vars["p_ac_dc"].value).ravel()
    p_dc_ac = np.asarray(opt.vars["p_dc_ac"].value).ravel()
    assert p_ac_dc.max() <= 1.0
    # NO_SIMULTANEOUS_INVALID_CONVERSION: never both directions at once.
    assert np.minimum(p_ac_dc, p_dc_ac).max() <= 1e-6
    # NO_FREE_ENERGY: stored energy never exceeds what the curve lets in.
    assert res["SOC_opt"].max() <= 0.95 + 1e-6


def test_negative_import_price_charging_pays_the_curve_loss():
    """With room in the battery, extra import is only possible through real DC power."""
    plant = {AC_DC: CHARGE_CURVE, "inverter_ac_input_max": 4500}
    _, res = _solve(plant, frame=_negative_price_frame(), soc_init=0.5, soc_final=0.5)
    _assert_curve_equality(res, charge_curve=CHARGE_CURVE)
    assert res["SOC_opt"].iloc[-1] >= 0.5 - 1e-6


def test_control_inequality_relaxation_would_be_exploited():
    """RELAXED_CONTROL_CASE: the same scenario with a one-sided relaxation earns free energy.

    Runs the real EMHASS model with only the AC->DC equality weakened. The relaxed model imports
    up to the inverter AC input limit that no DC power accounts for, so the exact model's
    assertions above would detect the defect.
    """
    _, exact = _solve(
        NEGATIVE_PRICE_PLANT, frame=_negative_price_frame(), soc_init=0.95, soc_final=0.95
    )
    _, relaxed = _solve(
        NEGATIVE_PRICE_PLANT,
        frame=_negative_price_frame(),
        soc_init=0.95,
        soc_final=0.95,
        cls=RelaxedOptimization,
    )
    assert exact["P_grid_pos"].max() <= 300.0 + 2.0
    assert relaxed["P_grid_pos"].max() >= 300.0 + 3000.0
    assert relaxed["cost_profit"].sum() > exact["cost_profit"].sum() + 0.01
