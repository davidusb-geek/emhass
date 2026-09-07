"""Config-layer plumbing tests for multi-component capacity charges (#540 Part B).

Covers only the config/runtime normalisation path - build_params, get_yaml_parse,
treat_runtimeparams and the canonicalize_capacity_charge_config helper - and the
config-UI round trip (a singleton list saved by the form must canonicalise back
to the legacy scalar K=1 form). The optimization.py per-component model is
covered by test_capacity_charge_multi_component.py.
"""

import asyncio
import json
import pathlib

import orjson
import pytest

from emhass import utils

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)


def _default_config() -> dict:
    return json.loads(emhass_conf["defaults_path"].read_text(encoding="utf-8"))


async def _build_params(overrides: dict | None = None) -> dict:
    config = _default_config()
    if overrides:
        config.update(overrides)
    _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
    params = await utils.build_params(emhass_conf, secrets, config, logger)
    assert params is not False
    return params


def build_params(overrides: dict | None = None) -> dict:
    return asyncio.run(_build_params(overrides))


def treat_runtime(runtimeparams: dict, base_params: dict, set_type="naive-mpc-optim"):
    params_json = orjson.dumps(base_params).decode("utf-8")
    rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
    rp_json = orjson.dumps(runtimeparams).decode("utf-8")

    async def _run():
        return await utils.treat_runtimeparams(
            rp_json, params_json, rh_conf, optim_conf, plant_conf, set_type, logger, emhass_conf
        )

    _, rh_conf, optim_conf, plant_conf = asyncio.run(_run())
    return optim_conf


# ─────────────────── canonicalize_capacity_charge_config unit ───────────────


@pytest.mark.parametrize(
    "raw_rate,expected_rate",
    [
        (3.0, 3.0),
        ([3.0], 3.0),
        ((3.0,), 3.0),
        ("[3.0]", 3.0),
        ([], 0.0),
        (0.0, 0.0),
        ([3.0, 7.0], [3.0, 7.0]),
        ("[3.0, 7.0]", [3.0, 7.0]),
        ([3.0, 7.0, 2.0], [3.0, 7.0, 2.0]),
    ],
)
def test_rate_canonical_form(raw_rate, expected_rate):
    oc = {"capacity_cost_per_kw": raw_rate}
    utils.canonicalize_capacity_charge_config(oc, logger)
    assert oc["capacity_cost_per_kw"] == expected_rate


@pytest.mark.parametrize(
    "rate,raw_iv,expected_iv",
    [
        ([3.0, 7.0], 6, 6),
        ([3.0, 7.0], [6], 6),
        ([3.0, 7.0], [6, 3], [6, 3]),
        ([3.0, 7.0], [], 1),
        (3.0, 6, 6),
        (3.0, [6], 6),
        ([3.0], [6], 6),  # rate singleton collapses first -> K=1 -> [6] -> 6
    ],
)
def test_interval_canonical_form(rate, raw_iv, expected_iv):
    oc = {"capacity_cost_per_kw": rate, "capacity_charge_interval_timesteps": raw_iv}
    utils.canonicalize_capacity_charge_config(oc, logger)
    assert oc["capacity_charge_interval_timesteps"] == expected_iv


@pytest.mark.parametrize(
    "rate,bad_iv",
    [
        ([8.0, 3.0], [6, 6, 6]),  # K=2 rate, 3 intervals
        ([8.0, 3.0, 1.0], [6, 6]),  # K=3 rate, 2 intervals
        (3.0, [6, 3]),  # K=1 rate, 2 intervals
    ],
)
def test_wrong_length_interval_list_raises(rate, bad_iv):
    oc = {"capacity_cost_per_kw": rate, "capacity_charge_interval_timesteps": bad_iv}
    with pytest.raises(ValueError, match="capacity_charge_interval_timesteps has"):
        utils.canonicalize_capacity_charge_config(oc, logger)


def test_canonicalisation_is_idempotent():
    oc = {"capacity_cost_per_kw": [3.0], "capacity_charge_interval_timesteps": [6]}
    utils.canonicalize_capacity_charge_config(oc, logger)
    snapshot = dict(oc)
    utils.canonicalize_capacity_charge_config(oc, logger)
    assert oc == snapshot


# ─────────────────────── build_params / config round trip ───────────────────


def test_build_params_canonicalises_singleton_config_to_scalar():
    params = build_params(
        {"capacity_cost_per_kw": [3.0], "capacity_charge_interval_timesteps": [6]}
    )
    assert params["optim_conf"]["capacity_cost_per_kw"] == 3.0
    assert params["optim_conf"]["capacity_charge_interval_timesteps"] == 6


def test_build_params_keeps_k2_list_and_broadcasts_singleton_interval():
    params = build_params(
        {"capacity_cost_per_kw": [3.0, 7.0], "capacity_charge_interval_timesteps": [6]}
    )
    assert params["optim_conf"]["capacity_cost_per_kw"] == [3.0, 7.0]
    assert params["optim_conf"]["capacity_charge_interval_timesteps"] == 6


def test_build_params_rejects_wrong_length_interval_config():
    with pytest.raises(ValueError):
        build_params(
            {
                "capacity_cost_per_kw": [3.0, 7.0],
                "capacity_charge_interval_timesteps": [6, 6, 6],
            }
        )


# ─────────────────────── treat_runtimeparams structural override ────────────


def test_runtime_structural_override_singleton_list_canonicalises():
    base = build_params()
    optim_conf = treat_runtime({"capacity_cost_per_kw": [3.0]}, base)
    assert optim_conf["capacity_cost_per_kw"] == 3.0


def test_runtime_structural_override_stringified_list_canonicalises():
    base = build_params()
    optim_conf = treat_runtime({"capacity_cost_per_kw": "[3.0, 7.0]"}, base)
    assert optim_conf["capacity_cost_per_kw"] == [3.0, 7.0]


def test_runtime_wrong_length_interval_override_raises():
    base = build_params()
    with pytest.raises(ValueError):
        treat_runtime(
            {
                "capacity_cost_per_kw": [3.0, 7.0],
                "capacity_charge_interval_timesteps": [6, 6, 6],
            },
            base,
        )


def test_full_config_to_optimization_round_trip_k1_and_k2():
    """config.json -> build_params -> get_yaml_parse -> treat_runtimeparams ->
    Optimization: a UI singleton builds the K=1 model, a K=2 list the K>N model."""
    from emhass.optimization import Optimization

    def _optim(overrides):
        base = build_params(overrides)
        optim_conf = treat_runtime({}, base)
        rh_conf, _, plant_conf = utils.get_yaml_parse(orjson.dumps(base).decode("utf-8"), logger)
        return Optimization(
            rh_conf,
            optim_conf,
            plant_conf,
            "unit_load_cost",
            "unit_prod_price",
            "profit",
            emhass_conf,
            logger,
        )

    opt_singleton = _optim(
        {"capacity_cost_per_kw": [3.0], "capacity_charge_interval_timesteps": [6]}
    )
    assert opt_singleton._capacity_multi is False
    assert not hasattr(opt_singleton, "param_current_period_peak_k")

    opt_k2 = _optim({"capacity_cost_per_kw": [3.0, 7.0], "capacity_charge_interval_timesteps": [6]})
    assert opt_k2._capacity_multi is True
    assert opt_k2.n_capacity_components == 2
    assert opt_k2._capacity_charge_interval_timesteps_list == [6, 6]


def test_param_to_config_get_config_round_trip():
    """The /get-config and /get-json response path (build_config -> build_params
    -> param_to_config) must return the canonical form: a config-form singleton
    list is shown back as the scalar; a K>=2 list is preserved."""
    for raw, expected in (
        ([3.0], 3.0),
        ([], 0.0),
        ([3.0, 7.0], [3.0, 7.0]),
    ):
        params = build_params({"capacity_cost_per_kw": raw})
        returned = utils.param_to_config(params, logger)
        assert returned["capacity_cost_per_kw"] == expected

    params = build_params(
        {"capacity_cost_per_kw": [3.0, 7.0], "capacity_charge_interval_timesteps": [6]}
    )
    returned = utils.param_to_config(params, logger)
    assert returned["capacity_cost_per_kw"] == [3.0, 7.0]
    assert returned["capacity_charge_interval_timesteps"] == 6
