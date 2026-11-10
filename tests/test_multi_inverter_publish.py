"""
Per-inverter publish sensors and the ``inverter`` attribute (#1150, PR 4).

Above one inverter publish_data posts the site totals as before and, next to them, one
sensor per inverter for each per-inverter column the plan carries. The battery power and
SOC sensors get an ``inverter`` attribute that names the inverter the battery sits on. The
attribute is kept in the saved entity metadata so a continual publish posts it again.
One inverter keeps today's entities, POST bodies and metadata exactly.
"""

import asyncio
import json
import logging
import pathlib
import re
import tempfile
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import orjson
import pandas as pd
import pytest

from emhass import command_line, plan_store, utils
from emhass.command_line import PublishContext, publish_json
from emhass.retrieve_hass import RetrieveHass

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)

_PARAMS: dict = {}


def _params() -> dict:
    """The default params with the publish ids, built once."""
    if not _PARAMS:

        async def go():
            """Build and treat the default params the way dayahead-optim does."""
            config = json.loads(emhass_conf["defaults_path"].read_text(encoding="utf-8"))
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            params = await utils.build_params(emhass_conf, secrets, config, logger)
            assert params is not False
            params_json = orjson.dumps(params).decode("utf-8")
            rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
            return await utils.treat_runtimeparams(
                "{}",
                params_json,
                rh_conf,
                optim_conf,
                plant_conf,
                "dayahead-optim",
                logger,
                emhass_conf,
            )

        treated = asyncio.run(go())[0]
        _PARAMS.update(orjson.loads(treated) if isinstance(treated, str) else treated)
    return json.loads(json.dumps(_PARAMS))


# --------------------------------- fixtures ---------------------------------

THREE = {
    "number_of_inverters": 3,
    "inverter_is_hybrid": False,
    "number_of_batteries": 2,
    "battery_inverter_index": [0, 1],
    "pv_plant_inverter_index": [1, 2],
    "compute_curtailment": [False, True, True],
}
TWO = {
    "number_of_inverters": 2,
    "inverter_is_hybrid": False,
    "battery_inverter_index": [0],
    "pv_plant_inverter_index": [1],
    "compute_curtailment": [False, True],
}


def _frame(columns: list[str]) -> pd.DataFrame:
    """Return a six-hour plan frame with the site columns, prices, status and the given columns."""
    index = pd.date_range(pd.Timestamp.now(tz="UTC").floor("h"), periods=6, freq="h")
    frame = pd.DataFrame(1.0, index=index, columns=["P_Load", "P_PV", "P_grid", *columns])
    frame["unit_load_cost"] = 0.3
    frame["unit_prod_price"] = 0.1
    frame["cost_fun_profit"] = 0.0
    frame["optim_status"] = "Optimal"
    return frame


def _run_publish(plant_conf: dict, columns: list[str], optim_conf: dict | None = None):
    """Run the real publish_data on a fabricated frame.

    Returns (posts, log, frame): posts maps entity_id to the call details.
    """
    optim = {"number_of_deferrable_loads": 0, "set_use_battery": False}
    optim.update(optim_conf or {})
    plant = {"number_of_inverters": 1, "inverter_is_hybrid": False, "compute_curtailment": False}
    plant.update(plant_conf)
    posts: dict[str, dict] = {}

    async def post_data(
        data, idx, entity_id, device_class, unit, friendly_name, type_var=None, **kwargs
    ):
        """Stand-in for RetrieveHass.post_data that records each call by entity_id."""
        posts[entity_id] = {
            "device_class": device_class,
            "unit": unit,
            "friendly_name": friendly_name,
            "type_var": type_var,
            "kwargs": kwargs,
        }

    rh = MagicMock()
    rh.post_data = AsyncMock(side_effect=post_data)
    input_data_dict = {
        "rh": rh,
        "params": _params(),
        "optim_conf": optim,
        "plant_conf": plant,
        "retrieve_hass_conf": {"time_zone": ZoneInfo("UTC")},
    }
    log = MagicMock(spec=logging.Logger)
    frame = asyncio.run(
        command_line.publish_data(input_data_dict, log, opt_res_latest=_frame(columns))
    )
    return posts, log, frame


def _per_inverter(posts: dict) -> set[str]:
    """Return the posted entity ids that end in an inverter number."""
    return {entity for entity in posts if re.search(r"inverter\d+$", entity)}


def _warnings(log) -> str:
    """Join every logged warning message into one string."""
    return " ".join(str(call.args[0]) for call in log.warning.call_args_list)


THREE_COLUMNS = [
    "P_PV_curtailment",
    "P_hybrid_inverter",
    "P_PV_1",
    "P_PV_2",
    "P_PV_curtailment_1",
    "P_PV_curtailment_2",
    "P_hybrid_inverter_0",
    "P_hybrid_inverter_1",
    "P_hybrid_inverter_2",
]


# ------------------------- per-inverter sensors: N=3 and N=2 -------------------------


def test_n3_mixed_topology_posts_one_sensor_per_inverter_column():
    """Three mixed inverters post one sensor per per-inverter column and keep the site totals."""
    # inverter 0 battery-only (no PV), 1 hybrid (PV and battery), 2 PV-only
    posts, log, _ = _run_publish(THREE, THREE_COLUMNS)
    assert _per_inverter(posts) == {
        "sensor.p_pv_forecast_inverter1",
        "sensor.p_pv_forecast_inverter2",
        "sensor.p_pv_curtailment_inverter1",
        "sensor.p_pv_curtailment_inverter2",
        "sensor.p_hybrid_inverter_inverter0",
        "sensor.p_hybrid_inverter_inverter1",
        "sensor.p_hybrid_inverter_inverter2",
    }
    assert "sensor.p_pv_forecast_inverter0" not in posts
    assert "sensor.p_pv_curtailment_inverter0" not in posts
    # The site totals are still there.
    assert "sensor.p_pv_forecast" in posts
    assert "sensor.p_pv_curtailment" in posts
    assert "sensor.p_hybrid_inverter" in posts
    assert "P_" not in _warnings(log)


def test_per_inverter_sensor_names_and_units_follow_the_site_totals_style():
    """Per-inverter sensors use the site total's name style, unit and device class."""
    posts, _, _ = _run_publish(THREE, THREE_COLUMNS)
    expected = {
        "sensor.p_pv_forecast_inverter2": ("PV Power Forecast Inverter 2"),
        "sensor.p_pv_curtailment_inverter1": ("PV Power Curtailment Inverter 1"),
        "sensor.p_hybrid_inverter_inverter0": ("PV Hybrid Inverter Inverter 0"),
    }
    for entity, name in expected.items():
        post = posts.get(entity)
        assert post is not None, entity
        assert post["friendly_name"] == name
        assert post["unit"] == "W"
        assert post["device_class"] == "power"
        assert post["type_var"] == "power"
        assert not post["kwargs"].get("extra_attributes")


def test_n2_posts_the_per_inverter_sensors():
    """Two inverters post the per-inverter sensors for the columns the plan carries."""
    columns = [
        "P_PV_curtailment",
        "P_hybrid_inverter",
        "P_PV_1",
        "P_PV_curtailment_1",
        "P_hybrid_inverter_0",
        "P_hybrid_inverter_1",
    ]
    posts, _, _ = _run_publish(TWO, columns)
    assert _per_inverter(posts) == {
        "sensor.p_pv_forecast_inverter1",
        "sensor.p_pv_curtailment_inverter1",
        "sensor.p_hybrid_inverter_inverter0",
        "sensor.p_hybrid_inverter_inverter1",
    }


def test_missing_expected_column_warns_and_skips():
    """A missing per-inverter column is skipped with a warning and the rest still publish."""
    columns = [c for c in THREE_COLUMNS if c not in ("P_hybrid_inverter_2", "P_PV_2")]
    posts, log, _ = _run_publish(THREE, columns)
    assert "sensor.p_hybrid_inverter_inverter2" not in posts
    assert "sensor.p_pv_forecast_inverter2" not in posts
    # the others still publish
    assert "sensor.p_hybrid_inverter_inverter1" in posts
    assert "sensor.p_pv_forecast_inverter1" in posts
    warnings = _warnings(log)
    assert "P_hybrid_inverter_2" in warnings
    assert "P_PV_2" in warnings


def test_missing_curtailment_column_on_a_curtailing_inverter_warns():
    """A curtailing inverter with no curtailment column warns and posts no sensor for it."""
    columns = [c for c in THREE_COLUMNS if c != "P_PV_curtailment_2"]
    posts, log, _ = _run_publish(THREE, columns)
    assert "sensor.p_pv_curtailment_inverter2" not in posts
    assert "sensor.p_pv_curtailment_inverter1" in posts
    assert "P_PV_curtailment_2" in _warnings(log)


def test_stale_frame_columns_beyond_the_inverter_count_are_not_published():
    """Columns for an inverter the config no longer has are not posted or returned."""
    # An N=3 plan replayed under an N=2 config: inverter 2 does not exist any more.
    columns = [
        "P_PV_curtailment",
        "P_hybrid_inverter",
        "P_PV_1",
        "P_PV_2",
        "P_PV_curtailment_1",
        "P_PV_curtailment_2",
        "P_hybrid_inverter_0",
        "P_hybrid_inverter_1",
        "P_hybrid_inverter_2",
    ]
    posts, _, frame = _run_publish(TWO, columns)
    assert _per_inverter(posts) == {
        "sensor.p_pv_forecast_inverter1",
        "sensor.p_pv_curtailment_inverter1",
        "sensor.p_hybrid_inverter_inverter0",
        "sensor.p_hybrid_inverter_inverter1",
    }
    for column in ("P_PV_2", "P_PV_curtailment_2", "P_hybrid_inverter_2"):
        assert column not in frame.columns


def test_spill_is_published_only_when_present_and_never_warns():
    """The spill sensor is posted only when the frame has a spill column, and never warns."""
    posts, log, _ = _run_publish(THREE, THREE_COLUMNS)
    assert not [e for e in posts if "spill" in e]
    assert "spill" not in _warnings(log)
    posts, log, frame = _run_publish(THREE, [*THREE_COLUMNS, "P_PV_spill_1"])
    spill = [e for e in posts if "spill" in e]
    assert spill == ["sensor.p_pv_spill_inverter1"]
    post = posts["sensor.p_pv_spill_inverter1"]
    assert post["friendly_name"] == "PV Power Spill Inverter 1"
    assert (post["unit"], post["device_class"], post["type_var"]) == ("W", "power", "power")
    assert "P_PV_spill_1" in frame.columns
    assert "spill" not in _warnings(log)


def test_curtailment_follows_each_inverters_own_flag():
    """Curtailment sensors and warnings follow each inverter's own curtailment flag."""
    conf = {**TWO, "pv_plant_inverter_index": [0, 1], "compute_curtailment": [True, False]}
    columns = [
        "P_PV_curtailment",
        "P_hybrid_inverter",
        "P_PV_0",
        "P_PV_1",
        "P_PV_curtailment_0",
        "P_PV_curtailment_1",
        "P_hybrid_inverter_0",
        "P_hybrid_inverter_1",
    ]
    posts, log, _ = _run_publish(conf, columns)
    assert "sensor.p_pv_curtailment_inverter0" in posts
    assert "sensor.p_pv_curtailment_inverter1" not in posts
    # With inverter 1's column gone there is still no warning for it, since its flag is off.
    columns.remove("P_PV_curtailment_1")
    posts, log, _ = _run_publish(conf, columns)
    assert "sensor.p_pv_curtailment_inverter0" in posts
    assert "P_PV_curtailment_1" not in _warnings(log)


def test_curtailment_flag_on_an_inverter_without_pv_posts_and_warns_nothing():
    """A curtailment flag on an inverter with no PV posts nothing and warns nothing."""
    # Inverter 0 has the flag on but no PV, so the optimizer makes no curtailment
    # column for it and no site total either: nothing is posted and nothing warns.
    conf = {**TWO, "compute_curtailment": [True, False]}
    columns = ["P_hybrid_inverter", "P_PV_1", "P_hybrid_inverter_0", "P_hybrid_inverter_1"]
    posts, log, _ = _run_publish(conf, columns)
    assert "sensor.p_pv_curtailment" not in posts
    assert not any("curtailment" in entity for entity in posts)
    assert "curtailment" not in _warnings(log)


def test_returned_frame_has_the_published_per_inverter_columns():
    """The returned frame keeps the per-inverter columns next to the site columns."""
    _, _, frame = _run_publish(THREE, [*THREE_COLUMNS, "P_PV_spill_2"])
    for column in [c for c in THREE_COLUMNS if c.endswith(("_0", "_1", "_2"))] + ["P_PV_spill_2"]:
        assert column in frame.columns, column
    for column in ("P_Load", "P_PV", "P_PV_curtailment", "P_hybrid_inverter"):
        assert column in frame.columns


# ------------------------------- one inverter pins -------------------------------


def test_n1_full_frame_saves_metadata_with_no_attributes():
    """One inverter saves no per-inverter entities or attributes key, even from a stale frame."""
    # The real post_data with entity save, through the whole publish_data, on a stale
    # frame that still has per-inverter and per-battery columns.
    columns = [*THREE_COLUMNS, "P_PV_spill_1", "P_batt", "SOC_opt", "P_batt_0", "SOC_opt_0"]
    with tempfile.TemporaryDirectory() as tmp:
        input_data_dict = {
            "rh": _real_rh(tmp),
            "params": _params(),
            "optim_conf": {"number_of_deferrable_loads": 0, "set_use_battery": True},
            "plant_conf": {
                "number_of_inverters": 1,
                "compute_curtailment": True,
                "inverter_is_hybrid": True,
            },
            "retrieve_hass_conf": {"time_zone": ZoneInfo("UTC")},
        }
        asyncio.run(
            command_line.publish_data(
                input_data_dict,
                logger,
                entity_save=True,
                dont_post=True,
                opt_res_latest=_frame(columns),
            )
        )
        metadata = json.loads((pathlib.Path(tmp) / "entities" / "metadata.json").read_text())
        entities = sorted(key for key in metadata if key.startswith("sensor."))
        assert "sensor.p_batt_forecast" in entities
        assert "sensor.p_hybrid_inverter" in entities
        assert not [entity for entity in entities if re.search(r"inverter\d+$", entity)]
        for entity in entities:
            assert "attributes" not in metadata[entity], entity


def test_n1_full_frame_posts_exactly_todays_entities_with_no_attributes():
    """A stale frame at one inverter posts exactly today's entities and no extra attributes."""
    columns = [
        *THREE_COLUMNS,
        "P_PV_spill_1",
        "P_PV_0",
        "P_batt",
        "SOC_opt",
        "P_batt_0",
        "SOC_opt_0",
    ]
    posts, _, frame = _run_publish(
        {"compute_curtailment": True, "inverter_is_hybrid": True},
        columns,
        {"set_use_battery": True},
    )
    assert sorted(posts) == sorted(
        [
            "sensor.p_load_forecast",
            "sensor.p_pv_forecast",
            "sensor.p_pv_curtailment",
            "sensor.p_hybrid_inverter",
            "sensor.p_batt_forecast",
            "sensor.soc_batt_forecast",
            "sensor.p_grid_forecast",
            "sensor.unit_load_cost",
            "sensor.unit_prod_price",
            "sensor.total_cost_fun_value",
            "sensor.optim_status",
        ]
    )
    for entity, post in posts.items():
        assert "extra_attributes" not in post["kwargs"], entity
    assert not [c for c in frame.columns if re.search(r"_\d$", c)]


def _real_rh(tmp: str) -> RetrieveHass:
    """Return a real RetrieveHass that reads from file and saves entities under tmp."""
    return RetrieveHass(
        hass_url="http://x",
        long_lived_token="tok",
        freq=pd.Timedelta(minutes=30),
        time_zone="UTC",
        params="{}",
        emhass_conf={"data_path": pathlib.Path(tmp)},
        logger=logger,
        get_data_from_file=True,
    )


def _batt_ctx(rh, plant_conf: dict, save: bool = False) -> PublishContext:
    """Return a PublishContext for the battery publish with the default battery entity ids."""
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
    return PublishContext(
        input_data_dict={
            "rh": rh,
            "optim_conf": {"set_use_battery": True},
            "plant_conf": plant_conf,
        },
        params={"passed_data": passed_data},
        idx=1,
        common_kwargs={"publish_prefix": "", "save_entities": save, "dont_post": False},
        logger=MagicMock(),
    )


def _batt_frame(n_batt: int) -> pd.DataFrame:
    """Return a three-step battery frame, with per-battery columns when n_batt is above one."""
    index = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
    columns = {"P_batt": [1.0, 2.0, 3.0], "SOC_opt": [0.5, 0.6, 0.7]}
    if n_batt > 1:
        columns = {"P_batt": [3.0, 4.0, 5.0]}
        for k in range(n_batt):
            columns[f"P_batt_{k}"] = [1.0, 2.0, 3.0]
            columns[f"SOC_opt_{k}"] = [0.5, 0.6, 0.7]
    return pd.DataFrame(columns, index=index)


def _batt_posts(plant_conf: dict) -> dict[str, dict]:
    """Run _publish_battery_data with a recording post_data; return the kwargs per entity_id."""
    posts: dict[str, dict] = {}

    async def post_data(data, idx, entity_id, *args, **kwargs):
        """Record the keyword arguments of each post by entity_id."""
        posts[entity_id] = kwargs

    rh = MagicMock()
    rh.post_data = AsyncMock(side_effect=post_data)
    ctx = _batt_ctx(rh, plant_conf)
    asyncio.run(
        command_line._publish_battery_data(ctx, _batt_frame(plant_conf["number_of_batteries"]))
    )
    return posts


# ------------------------------- inverter attribute -------------------------------


def test_inverter_attribute_on_per_battery_sensors_with_several_batteries():
    """Each per-battery sensor names its inverter and the fleet total names none."""
    plant = {"number_of_inverters": 3, "number_of_batteries": 2, "battery_inverter_index": [2, 0]}
    posts = _batt_posts(plant)
    assert posts["sensor.p_batt_forecast_battery0"].get("extra_attributes") == {"inverter": 2}
    assert posts["sensor.soc_batt_forecast_battery0"].get("extra_attributes") == {"inverter": 2}
    assert posts["sensor.p_batt_forecast_battery1"].get("extra_attributes") == {"inverter": 0}
    assert posts["sensor.soc_batt_forecast_battery1"].get("extra_attributes") == {"inverter": 0}
    # the fleet total spans inverters, so it carries none
    assert not posts["sensor.p_batt_forecast"].get("extra_attributes")
    assert "extra_attributes" not in posts["sensor.p_batt_forecast"]


def test_inverter_attribute_on_the_bare_sensors_with_one_battery():
    """One battery on a multi-inverter site puts the attribute on the bare battery sensors."""
    plant = {"number_of_inverters": 2, "number_of_batteries": 1, "battery_inverter_index": [1]}
    posts = _batt_posts(plant)
    assert sorted(posts) == ["sensor.p_batt_forecast", "sensor.soc_batt_forecast"]
    assert posts["sensor.p_batt_forecast"].get("extra_attributes") == {"inverter": 1}
    assert posts["sensor.soc_batt_forecast"].get("extra_attributes") == {"inverter": 1}


def test_no_inverter_attribute_with_one_inverter():
    """One inverter never adds the attribute, even with a battery index left in the config."""
    for n_batt in (1, 2):
        plant = {"number_of_batteries": n_batt}
        posts = _batt_posts(plant)
        assert posts
        for entity, kwargs in posts.items():
            assert "extra_attributes" not in kwargs, (n_batt, entity)
        # A battery index left in the config is not read at one inverter.
        plant = {
            "number_of_inverters": 1,
            "number_of_batteries": n_batt,
            "battery_inverter_index": [0] * n_batt,
        }
        for entity, kwargs in _batt_posts(plant).items():
            assert "extra_attributes" not in kwargs, (n_batt, entity)


# --------------------------- post_data and the saved metadata ---------------------------


def _post(rh, entity_id="sensor.p_batt_forecast_battery0", **kwargs):
    """Post a short battery series through rh.post_data and return its result."""
    series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC"),
        name="P_batt_0",
    )
    return asyncio.run(
        rh.post_data(series, 1, entity_id, "power", "W", "Battery", type_var="batt", **kwargs)
    )


def test_post_data_extra_attributes_reach_the_post_body():
    """extra_attributes land in the POST body next to the usual attributes."""
    with tempfile.TemporaryDirectory() as tmp:
        rh = _real_rh(tmp)
        _, data = _post(rh, extra_attributes={"inverter": 1})
        assert data["attributes"].get("inverter") == 1
        # the usual attributes are still there
        assert data["attributes"]["friendly_name"] == "Battery"
        assert data["attributes"]["unit_of_measurement"] == "W"
        orjson.dumps(data)


def test_post_data_without_extra_attributes_is_unchanged():
    """Leaving out extra_attributes, or passing None or {}, gives the same body as before."""
    with tempfile.TemporaryDirectory() as tmp:
        rh = _real_rh(tmp)
        _, plain = _post(rh)
        _, none = _post(rh, extra_attributes=None)
        _, empty = _post(rh, extra_attributes={})
        assert plain == none == empty
        assert "inverter" not in plain["attributes"]


def test_extra_attributes_are_saved_in_the_metadata_only_when_given():
    """The saved metadata gets an attributes key only when extra_attributes is not empty."""
    with tempfile.TemporaryDirectory() as tmp:
        rh = _real_rh(tmp)
        _post(rh, "sensor.with_attr", save_entities=True, extra_attributes={"inverter": 1})
        _post(rh, "sensor.without_attr", save_entities=True)
        _post(rh, "sensor.empty_attr", save_entities=True, extra_attributes={})
        metadata = json.loads((pathlib.Path(tmp) / "entities" / "metadata.json").read_text())
        assert metadata["sensor.with_attr"].get("attributes") == {"inverter": 1}
        assert "attributes" not in metadata["sensor.without_attr"]
        assert "attributes" not in metadata["sensor.empty_attr"]
        assert metadata["sensor.without_attr"].keys() == {
            "name",
            "device_class",
            "unit_of_measurement",
            "friendly_name",
            "type_var",
            "optimization_time_step",
        }


def _publish_json_capture(tmp: str, entity_file: str):
    """Run publish_json on the saved entities with the real post_data; return what it posted."""
    reader = _real_rh(tmp)
    captured: list = []
    real_post = reader.post_data

    async def wrapper(*args, **kwargs):
        """Call the real post_data and keep the body it built."""
        result = await real_post(*args, **kwargs)
        captured.append(result[1])
        return result

    reader.post_data = wrapper
    reader_dict = {
        "rh": reader,
        "retrieve_hass_conf": {"time_zone": ZoneInfo("UTC"), "method_ts_round": "nearest"},
    }
    entities_path = pathlib.Path(tmp) / "entities"
    res = asyncio.run(publish_json(entity_file, reader_dict, entities_path, logger))
    assert res is not False
    assert len(captured) == 1
    return captured[0]


def test_continual_publish_round_trip_keeps_the_inverter_attribute():
    """A continual publish replays the saved inverter attribute on each per-battery sensor."""
    with tempfile.TemporaryDirectory() as tmp:
        plant = {
            "number_of_inverters": 2,
            "number_of_batteries": 2,
            "battery_inverter_index": [1, 0],
        }
        ctx = _batt_ctx(_real_rh(tmp), plant, save=True)
        asyncio.run(command_line._publish_battery_data(ctx, _batt_frame(2)))
        entities = pathlib.Path(tmp) / "entities"
        saved = json.loads((entities / "metadata.json").read_text())
        assert saved["sensor.p_batt_forecast_battery0"].get("attributes") == {"inverter": 1}
        assert saved["sensor.soc_batt_forecast_battery1"].get("attributes") == {"inverter": 0}
        assert "attributes" not in saved["sensor.p_batt_forecast"]
        replay = _publish_json_capture(tmp, "sensor.p_batt_forecast_battery0.json")
        assert replay["attributes"].get("inverter") == 1
        replay = _publish_json_capture(tmp, "sensor.soc_batt_forecast_battery1.json")
        assert replay["attributes"].get("inverter") == 0
        replay = _publish_json_capture(tmp, "sensor.p_batt_forecast.json")
        assert "inverter" not in replay["attributes"]


def test_continual_publish_round_trip_keeps_the_attribute_of_a_single_battery():
    """A continual publish replays the attribute on the bare sensors of a single battery."""
    with tempfile.TemporaryDirectory() as tmp:
        plant = {"number_of_inverters": 2, "number_of_batteries": 1, "battery_inverter_index": [1]}
        ctx = _batt_ctx(_real_rh(tmp), plant, save=True)
        asyncio.run(command_line._publish_battery_data(ctx, _batt_frame(1)))
        for entity_file in ("sensor.p_batt_forecast.json", "sensor.soc_batt_forecast.json"):
            replay = _publish_json_capture(tmp, entity_file)
            assert replay["attributes"]["inverter"] == 1, entity_file


@pytest.mark.parametrize("bad", ["abc", 5, True, [1, 2]])
def test_publish_json_ignores_saved_attributes_that_are_not_an_object(bad):
    """publish_json skips saved attributes that are not a JSON object and still posts."""
    with tempfile.TemporaryDirectory() as tmp:
        plant = {
            "number_of_inverters": 2,
            "number_of_batteries": 2,
            "battery_inverter_index": [1, 0],
        }
        ctx = _batt_ctx(_real_rh(tmp), plant, save=True)
        asyncio.run(command_line._publish_battery_data(ctx, _batt_frame(2)))
        metadata_path = pathlib.Path(tmp) / "entities" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["sensor.p_batt_forecast_battery0"]["attributes"] = bad
        metadata_path.write_text(json.dumps(metadata))
        replay = _publish_json_capture(tmp, "sensor.p_batt_forecast_battery0.json")
        assert "inverter" not in replay["attributes"]
        assert replay["attributes"]["friendly_name"] == "Battery Power Forecast Battery 0"


def test_continual_publish_reads_old_metadata_without_attributes():
    """Metadata with no attributes key, as older versions saved it, still replays cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _batt_ctx(_real_rh(tmp), {"number_of_batteries": 2}, save=True)
        asyncio.run(command_line._publish_battery_data(ctx, _batt_frame(2)))
        metadata_path = pathlib.Path(tmp) / "entities" / "metadata.json"
        assert "attributes" not in metadata_path.read_text()
        replay = _publish_json_capture(tmp, "sensor.p_batt_forecast_battery1.json")
        assert "inverter" not in replay["attributes"]


def test_publish_json_passes_none_when_the_metadata_has_no_attributes():
    """publish_json passes extra_attributes=None when the saved metadata has none."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _batt_ctx(_real_rh(tmp), {"number_of_batteries": 2}, save=True)
        asyncio.run(command_line._publish_battery_data(ctx, _batt_frame(2)))
        mock_rh = AsyncMock()
        reader_dict = {
            "rh": mock_rh,
            "retrieve_hass_conf": {"time_zone": ZoneInfo("UTC"), "method_ts_round": "nearest"},
        }
        entities = pathlib.Path(tmp) / "entities"
        asyncio.run(
            publish_json("sensor.p_batt_forecast_battery0.json", reader_dict, entities, logger)
        )
        assert mock_rh.post_data.call_args.kwargs["extra_attributes"] is None


def test_n1_metadata_has_no_attributes_key():
    """At one inverter the saved metadata has no attributes or inverter key at all."""
    with tempfile.TemporaryDirectory() as tmp:
        for n_batt in (1, 2):
            ctx = _batt_ctx(_real_rh(tmp), {"number_of_batteries": n_batt}, save=True)
            asyncio.run(command_line._publish_battery_data(ctx, _batt_frame(n_batt)))
        text = (pathlib.Path(tmp) / "entities" / "metadata.json").read_text()
        assert "attributes" not in text
        assert "inverter" not in text


# ------------------------------- plan API and chart -------------------------------


def test_plan_serialize_carries_the_per_inverter_columns():
    """The plan API serializer keeps the per-inverter and spill columns."""
    frame = _frame([*THREE_COLUMNS, "P_PV_spill_1"])
    first = plan_store.serialize(frame)[0]
    for column in [*THREE_COLUMNS, "P_PV_spill_1"]:
        assert column in first, column


def test_chart_figure_0_plots_the_per_inverter_columns():
    """The first web UI chart plots the per-inverter and spill columns."""
    frame = _frame([*THREE_COLUMNS, "P_PV_spill_1"])
    html = utils.get_injection_dict(frame)["figure_0"]
    for column in ("P_PV_1", "P_PV_curtailment_2", "P_hybrid_inverter_0", "P_PV_spill_1"):
        assert column in html, column


@pytest.mark.parametrize("flags", [True, [True, True]])
def test_a_scalar_curtailment_flag_applies_to_every_inverter(flags):
    """A scalar True flag works like one per inverter, so every inverter posts curtailment."""
    conf = {**TWO, "pv_plant_inverter_index": [0, 1], "compute_curtailment": flags}
    columns = [
        "P_PV_curtailment",
        "P_hybrid_inverter",
        "P_PV_0",
        "P_PV_1",
        "P_PV_curtailment_0",
        "P_PV_curtailment_1",
        "P_hybrid_inverter_0",
        "P_hybrid_inverter_1",
    ]
    posts, _, _ = _run_publish(conf, columns)
    assert "sensor.p_pv_curtailment_inverter0" in posts
    assert "sensor.p_pv_curtailment_inverter1" in posts
