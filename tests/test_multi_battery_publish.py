"""
Multi-battery publish path tests (#610).

Covers the runtime soc_init/soc_final list pass-through (naive-mpc + dayahead
callers into perform_naive_mpc_optim / perform_dayahead_forecast_optim) and the
publish path: _publish_battery_data's N=1 no-op pin, its N>1 fleet-total +
per-battery entity behaviour, the custom entity-id override decision at N>1,
and a real continual_publish/entity_save round trip for per-battery sensors.
The config plumbing and the optimization.py per-battery model are covered
elsewhere and out of scope here.
"""

import asyncio
import json
import pathlib
import tempfile
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import orjson
import pandas as pd
import pytest

from emhass import utils
from emhass.command_line import (
    PublishContext,
    _publish_battery_data,
    dayahead_forecast_optim,
    naive_mpc_optim,
    publish_json,
    set_input_data_dict,
)
from emhass.retrieve_hass import RetrieveHass

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


async def _build_input_data_dict(config_overrides: dict, runtimeparams: dict, action: str) -> dict:
    config = _default_config()
    config.update(config_overrides)
    _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
    params = await utils.build_params(emhass_conf, secrets, config, logger)
    assert params is not False, "build_params failed (see logged error)"
    params_json = orjson.dumps(params).decode("utf-8")
    rp_json = orjson.dumps(runtimeparams).decode("utf-8")
    return await set_input_data_dict(
        emhass_conf, "profit", params_json, rp_json, action, logger, get_data_from_file=True
    )


def build_input_data_dict(config_overrides: dict, runtimeparams: dict, action: str) -> dict:
    return asyncio.run(_build_input_data_dict(config_overrides, runtimeparams, action))


_FORECAST_LISTS = {
    "pv_power_forecast": [1] * 48,
    "load_power_forecast": [1] * 48,
    "load_cost_forecast": [1] * 48,
    "prod_price_forecast": [1] * 48,
}


# ───────────────────────── soc_init/soc_final pass-through ─────────────────────


def test_naive_mpc_soc_lists_reach_perform_naive_mpc_optim_n2():
    """#610: runtime soc_init/soc_final lists of length N=2 must reach
    perform_naive_mpc_optim unchanged (positional args 5/6 per command_line.py)."""
    runtimeparams = dict(_FORECAST_LISTS)
    runtimeparams.update(
        {"prediction_horizon": 10, "soc_init": [0.3, 0.5], "soc_final": [0.4, 0.6]}
    )
    idd = build_input_data_dict(
        {"number_of_batteries": 2, "set_use_battery": True},
        runtimeparams,
        "naive-mpc-optim",
    )
    idx = idd["df_input_data_dayahead"].index[:10]
    mock_res = pd.DataFrame(
        {"P_PV": 0.0, "P_Load": 0.0, "P_grid": 0.0, "optim_status": "Optimal"}, index=idx
    )
    idd["opt"].perform_naive_mpc_optim = MagicMock(return_value=mock_res)
    asyncio.run(naive_mpc_optim(idd, logger, debug=True))
    call = idd["opt"].perform_naive_mpc_optim.call_args
    assert call.args[4] == [0.3, 0.5]
    assert call.args[5] == [0.4, 0.6]


def test_naive_mpc_soc_scalar_broadcasts_shape_preserved_n1():
    """Regression: N=1 (default) keeps soc_init/soc_final as plain scalars
    reaching perform_naive_mpc_optim, byte-identical to master."""
    runtimeparams = dict(_FORECAST_LISTS)
    runtimeparams.update({"prediction_horizon": 10, "soc_init": 0.5, "soc_final": 0.6})
    idd = build_input_data_dict({"set_use_battery": True}, runtimeparams, "naive-mpc-optim")
    idx = idd["df_input_data_dayahead"].index[:10]
    mock_res = pd.DataFrame(
        {"P_PV": 0.0, "P_Load": 0.0, "P_grid": 0.0, "optim_status": "Optimal"}, index=idx
    )
    idd["opt"].perform_naive_mpc_optim = MagicMock(return_value=mock_res)
    asyncio.run(naive_mpc_optim(idd, logger, debug=True))
    call = idd["opt"].perform_naive_mpc_optim.call_args
    assert call.args[4] == 0.5
    assert call.args[5] == 0.6
    assert not isinstance(call.args[4], list)


def test_dayahead_soc_lists_reach_perform_dayahead_forecast_optim_n2():
    """#610: the dayahead branch of treat_runtimeparams goes through the
    else-branch (_passthrough_soc_runtime in utils.py), which used to crash
    on float() for a list. Must now pass runtime soc_init/soc_final lists
    through unchanged to perform_dayahead_forecast_optim (soc_init/soc_final
    are kwargs there)."""
    runtimeparams = dict(_FORECAST_LISTS)
    runtimeparams.update({"soc_init": [0.3, 0.5], "soc_final": [0.4, 0.6]})
    idd = build_input_data_dict(
        {"number_of_batteries": 2, "set_use_battery": True},
        runtimeparams,
        "dayahead-optim",
    )
    idx = idd["df_input_data_dayahead"].index
    mock_res = pd.DataFrame(
        {"P_PV": 0.0, "P_Load": 0.0, "P_grid": 0.0, "optim_status": "Optimal"}, index=idx
    )
    idd["opt"].perform_dayahead_forecast_optim = MagicMock(return_value=mock_res)
    asyncio.run(dayahead_forecast_optim(idd, logger, debug=True))
    call = idd["opt"].perform_dayahead_forecast_optim.call_args
    assert call.kwargs["soc_init"] == [0.3, 0.5]
    assert call.kwargs["soc_final"] == [0.4, 0.6]


def test_dayahead_soc_scalar_reaches_perform_dayahead_forecast_optim_n1():
    """Regression: N=1 scalar soc_init/soc_final still reaches
    perform_dayahead_forecast_optim as a plain float (today's behaviour,
    unaffected by the else-branch list-safety fix)."""
    runtimeparams = dict(_FORECAST_LISTS)
    runtimeparams.update({"soc_init": 0.5, "soc_final": 0.6})
    idd = build_input_data_dict({"set_use_battery": True}, runtimeparams, "dayahead-optim")
    idx = idd["df_input_data_dayahead"].index
    mock_res = pd.DataFrame(
        {"P_PV": 0.0, "P_Load": 0.0, "P_grid": 0.0, "optim_status": "Optimal"}, index=idx
    )
    idd["opt"].perform_dayahead_forecast_optim = MagicMock(return_value=mock_res)
    asyncio.run(dayahead_forecast_optim(idd, logger, debug=True))
    call = idd["opt"].perform_dayahead_forecast_optim.call_args
    assert call.kwargs["soc_init"] == 0.5
    assert call.kwargs["soc_final"] == 0.6
    assert not isinstance(call.kwargs["soc_init"], list)


def test_dayahead_soc_wrong_length_list_raises_end_to_end():
    """#610: a runtime soc_init list not matching number_of_batteries must
    still raise (validated downstream by Optimization._normalize_soc_arg, not
    duplicated here) - proves the dayahead branch's list-safety fix does not
    silently swallow a genuine mis-configuration. Real (non-mocked) call, so
    this exercises optimization.py at runtime without editing it."""
    runtimeparams = dict(_FORECAST_LISTS)
    runtimeparams.update({"soc_init": [0.3, 0.5, 0.9]})
    idd = build_input_data_dict(
        {"number_of_batteries": 2, "set_use_battery": True},
        runtimeparams,
        "dayahead-optim",
    )
    with pytest.raises(ValueError):
        asyncio.run(dayahead_forecast_optim(idd, logger, debug=True))


# ───────────────────────────── _publish_battery_data ───────────────────────────


def _make_ctx(
    n_batt: int,
    custom_batt_override: dict | None = None,
    custom_soc_override: dict | None = None,
    extra_optim_conf: dict | None = None,
) -> tuple[PublishContext, MagicMock]:
    rh = MagicMock()
    rh.post_data = AsyncMock(return_value=True)
    optim_conf = {"set_use_battery": True}
    if extra_optim_conf:
        optim_conf.update(extra_optim_conf)
    input_data_dict = {
        "rh": rh,
        "optim_conf": optim_conf,
        "plant_conf": {"number_of_batteries": n_batt},
    }
    passed_data = {
        "custom_batt_forecast_id": custom_batt_override
        or {
            "entity_id": "sensor.p_batt_forecast",
            "unit_of_measurement": "W",
            "friendly_name": "Battery Power Forecast",
        },
        "custom_batt_soc_forecast_id": custom_soc_override
        or {
            "entity_id": "sensor.soc_batt_forecast",
            "unit_of_measurement": "%",
            "friendly_name": "Battery SOC Forecast",
        },
    }
    ctx = PublishContext(
        input_data_dict=input_data_dict,
        params={"passed_data": passed_data},
        idx=1,
        common_kwargs={"publish_prefix": "", "save_entities": False, "dont_post": False},
        logger=MagicMock(),
    )
    return ctx, rh


def _make_opt_res(columns: dict) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
    return pd.DataFrame(columns, index=idx)


def _entity_ids_posted(rh: MagicMock) -> list[str]:
    return [
        c.args[2] if len(c.args) > 2 else c.kwargs["entity_id"] for c in rh.post_data.call_args_list
    ]


def test_publish_n1_is_exact_todays_two_entities_no_op_pin():
    """#610 no-op pin: N=1 publishes exactly today's P_batt/SOC_opt entities,
    zero new entities."""
    ctx, rh = _make_ctx(1)
    opt_res = _make_opt_res({"P_batt": [1.0, 2.0, 3.0], "SOC_opt": [0.5, 0.6, 0.7]})
    cols = asyncio.run(_publish_battery_data(ctx, opt_res))
    assert cols == ["P_batt", "SOC_opt"]
    entity_ids = _entity_ids_posted(rh)
    assert entity_ids == ["sensor.p_batt_forecast", "sensor.soc_batt_forecast"]
    assert rh.post_data.call_count == 2


def test_publish_n1_stale_n_gt_1_frame_missing_soc_opt_logs_and_skips_not_crash():
    """A stale N>1 results frame (fleet P_batt + P_batt_0/1 + SOC_opt_0/1, NO
    bare SOC_opt column - no aggregate SOC at N>1) replayed under a config
    reverted to number_of_batteries=1 (continual_publish replay, or a
    publish-data call after a config edit with no re-optimize) must warn+skip
    the SOC publish, never KeyError. Mirrors the guard already on the sibling
    P_batt read just above it in _publish_battery_data (command_line.py);
    pre-fix this crashed with an unguarded ``opt_res_latest["SOC_opt"]``
    read."""
    ctx, rh = _make_ctx(1)
    opt_res = _make_opt_res(
        {
            "P_batt": [3.0, 4.0, 5.0],
            "P_batt_0": [1.0, 2.0, 3.0],
            "P_batt_1": [2.0, 2.0, 2.0],
            "SOC_opt_0": [0.5, 0.6, 0.7],
            "SOC_opt_1": [0.4, 0.5, 0.6],
            # deliberately NO bare "SOC_opt" column - the stale-frame shape.
        }
    )
    cols = asyncio.run(_publish_battery_data(ctx, opt_res))
    # Fleet P_batt still publishes fine; SOC_opt is skipped, not crashed.
    assert cols == ["P_batt"]
    entity_ids = _entity_ids_posted(rh)
    assert entity_ids == ["sensor.p_batt_forecast"]
    assert rh.post_data.call_count == 1


def test_publish_n2_fleet_total_plus_per_battery_entities():
    """#610: N=2 publishes the fleet-total P_batt on the unchanged entity,
    plus per-battery power/SOC on fixed sensor.*_battery<K> ids, and NO bare
    SOC_opt/soc_batt_forecast entity."""
    ctx, rh = _make_ctx(2)
    opt_res = _make_opt_res(
        {
            "P_batt": [3.0, 4.0, 5.0],
            "P_batt_0": [1.0, 2.0, 3.0],
            "P_batt_1": [2.0, 2.0, 2.0],
            "SOC_opt_0": [0.5, 0.6, 0.7],
            "SOC_opt_1": [0.4, 0.5, 0.6],
        }
    )
    cols = asyncio.run(_publish_battery_data(ctx, opt_res))
    assert set(cols) == {"P_batt", "P_batt_0", "P_batt_1", "SOC_opt_0", "SOC_opt_1"}
    entity_ids = _entity_ids_posted(rh)
    assert "sensor.p_batt_forecast" in entity_ids
    assert "sensor.p_batt_forecast_battery0" in entity_ids
    assert "sensor.p_batt_forecast_battery1" in entity_ids
    assert "sensor.soc_batt_forecast_battery0" in entity_ids
    assert "sensor.soc_batt_forecast_battery1" in entity_ids
    # No meaningful fleet SOC aggregate: the bare SOC entity never appears.
    assert "sensor.soc_batt_forecast" not in entity_ids
    assert rh.post_data.call_count == 5


def test_publish_n2_friendly_names_get_battery_k_suffix():
    ctx, rh = _make_ctx(2)
    opt_res = _make_opt_res(
        {
            "P_batt": [3.0],
            "P_batt_0": [1.0],
            "P_batt_1": [2.0],
            "SOC_opt_0": [0.5],
            "SOC_opt_1": [0.4],
        }
    )
    opt_res = opt_res.iloc[[0]]
    ctx.idx = 0
    asyncio.run(_publish_battery_data(ctx, opt_res))
    friendly_names = [
        c.args[5] if len(c.args) > 5 else c.kwargs["friendly_name"]
        for c in rh.post_data.call_args_list
    ]
    assert any("Battery 0" in fn for fn in friendly_names)
    assert any("Battery 1" in fn for fn in friendly_names)


def test_publish_n2_missing_per_battery_column_logged_and_skipped():
    """Robustness: if plant_conf says N=2 but the results DataFrame is missing a
    per-battery column (stale data), log an error and skip just that sensor -
    do not crash, and still publish the ones that are present."""
    ctx, rh = _make_ctx(2)
    opt_res = _make_opt_res(
        {
            "P_batt": [3.0],
            "P_batt_0": [1.0],
            "P_batt_1": [2.0],
            "SOC_opt_0": [0.5],
            # SOC_opt_1 missing
        }
    )
    opt_res = opt_res.iloc[[0]]
    ctx.idx = 0
    cols = asyncio.run(_publish_battery_data(ctx, opt_res))
    assert "SOC_opt_1" not in cols
    entity_ids = _entity_ids_posted(rh)
    assert "sensor.soc_batt_forecast_battery1" not in entity_ids
    assert "sensor.p_batt_forecast_battery1" in entity_ids
    ctx.logger.error.assert_called()


def test_custom_batt_forecast_id_override_affects_fleet_total_at_n_gt_1():
    """#610: custom_batt_forecast_id keeps overriding only the fleet-total
    power entity at N>1; per-battery entities stay on the fixed naming
    regardless."""
    ctx, rh = _make_ctx(
        2,
        custom_batt_override={
            "entity_id": "sensor.my_fleet_batt",
            "unit_of_measurement": "W",
            "friendly_name": "My Fleet Battery",
        },
    )
    opt_res = _make_opt_res(
        {
            "P_batt": [3.0],
            "P_batt_0": [1.0],
            "P_batt_1": [2.0],
            "SOC_opt_0": [0.5],
            "SOC_opt_1": [0.4],
        }
    ).iloc[[0]]
    ctx.idx = 0
    asyncio.run(_publish_battery_data(ctx, opt_res))
    entity_ids = _entity_ids_posted(rh)
    assert "sensor.my_fleet_batt" in entity_ids
    assert "sensor.p_batt_forecast" not in entity_ids
    assert "sensor.p_batt_forecast_battery0" in entity_ids
    assert "sensor.p_batt_forecast_battery1" in entity_ids


def test_custom_batt_soc_forecast_id_override_ignored_with_warning_at_n_gt_1():
    """#610: custom_batt_soc_forecast_id has no natural single target at N>1
    (no fleet SOC aggregate), so a runtime override is ignored (per-battery
    SOC keeps the fixed sensor.soc_batt_forecast_battery<K> naming) and a
    warning is logged - the least-surprising option, adding no new config
    surface."""
    ctx, rh = _make_ctx(
        2,
        custom_soc_override={
            "entity_id": "sensor.my_soc",
            "unit_of_measurement": "%",
            "friendly_name": "My SOC",
        },
    )
    opt_res = _make_opt_res(
        {
            "P_batt": [3.0],
            "P_batt_0": [1.0],
            "P_batt_1": [2.0],
            "SOC_opt_0": [0.5],
            "SOC_opt_1": [0.4],
        }
    ).iloc[[0]]
    ctx.idx = 0
    asyncio.run(_publish_battery_data(ctx, opt_res))
    entity_ids = _entity_ids_posted(rh)
    assert not any("my_soc" in e for e in entity_ids)
    assert "sensor.soc_batt_forecast_battery0" in entity_ids
    assert "sensor.soc_batt_forecast_battery1" in entity_ids
    ctx.logger.warning.assert_called()


def test_custom_batt_soc_forecast_id_override_still_works_at_n1():
    """Regression: the override still works at N=1 exactly as today (no
    warning, no ignoring - only an N>1 concept)."""
    ctx, rh = _make_ctx(
        1,
        custom_soc_override={
            "entity_id": "sensor.my_soc",
            "unit_of_measurement": "%",
            "friendly_name": "My SOC",
        },
    )
    opt_res = _make_opt_res({"P_batt": [1.0], "SOC_opt": [0.5]}).iloc[[0]]
    ctx.idx = 0
    asyncio.run(_publish_battery_data(ctx, opt_res))
    entity_ids = _entity_ids_posted(rh)
    assert "sensor.my_soc" in entity_ids
    ctx.logger.warning.assert_not_called()


# ───────────────────── continual_publish / entity_save round trip ──────────────


def test_entity_save_round_trip_per_battery_sensors_n2():
    """#610: saved entity json files for per-battery sensors (distinct
    entity_ids, same type_var "batt"/"SOC" as today) round-trip through the
    real save (RetrieveHass.post_data, save_entities=True) and real read
    (publish_json) path with no collisions and no retrieve_hass.py changes
    needed."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_conf = {"data_path": pathlib.Path(tmp)}
        rh_writer = RetrieveHass(
            hass_url="http://x",
            long_lived_token="tok",
            freq=pd.Timedelta(minutes=30),
            time_zone="UTC",
            params="{}",
            emhass_conf=tmp_conf,
            logger=logger,
            get_data_from_file=True,
        )
        input_data_dict = {
            "rh": rh_writer,
            "optim_conf": {"set_use_battery": True},
            "plant_conf": {"number_of_batteries": 2},
        }
        passed_data = {
            "custom_batt_forecast_id": {
                "entity_id": "sensor.p_batt_forecast",
                "unit_of_measurement": "W",
                "friendly_name": "Battery Power Forecast",
            },
            "custom_batt_soc_forecast_id": {
                "entity_id": "sensor.soc_batt_forecast",
                "unit_of_measurement": "%",
                "friendly_name": "Battery SOC Forecast",
            },
        }
        ctx = PublishContext(
            input_data_dict=input_data_dict,
            params={"passed_data": passed_data},
            idx=1,
            common_kwargs={"publish_prefix": "", "save_entities": True, "dont_post": False},
            logger=logger,
        )
        opt_res = _make_opt_res(
            {
                "P_batt": [3.0, 4.0, 5.0],
                "P_batt_0": [1.0, 2.0, 3.0],
                "P_batt_1": [2.0, 2.0, 2.0],
                "SOC_opt_0": [0.5, 0.6, 0.7],
                "SOC_opt_1": [0.4, 0.5, 0.6],
            }
        )
        asyncio.run(_publish_battery_data(ctx, opt_res))

        entities_path = pathlib.Path(tmp) / "entities"
        saved = {p.name for p in entities_path.glob("*.json")}
        assert "sensor.p_batt_forecast_battery0.json" in saved
        assert "sensor.p_batt_forecast_battery1.json" in saved
        assert "sensor.soc_batt_forecast_battery0.json" in saved
        assert "sensor.soc_batt_forecast_battery1.json" in saved
        assert "sensor.soc_batt_forecast.json" not in saved

        # Real read-back via publish_json, only rh.post_data mocked to avoid HTTP.
        mock_rh = AsyncMock()
        reader_dict = {
            "rh": mock_rh,
            "retrieve_hass_conf": {
                "time_zone": ZoneInfo("UTC"),
                "method_ts_round": "nearest",
            },
        }
        res = asyncio.run(
            publish_json(
                "sensor.soc_batt_forecast_battery1.json", reader_dict, entities_path, logger
            )
        )
        assert res is not False
        mock_rh.post_data.assert_called_once()
        call_kwargs = mock_rh.post_data.call_args.kwargs
        assert call_kwargs["entity_id"] == "sensor.soc_batt_forecast_battery1"
        assert call_kwargs["friendly_name"] == "Battery SOC Forecast Battery 1"
        assert call_kwargs["unit_of_measurement"] == "%"
