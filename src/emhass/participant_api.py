"""This EMHASS's devices, served to a coordinator elsewhere (opt-in).

A coordinator (another EMHASS with `optimization_backend: dantzig_wolfe`, or
any client of the participant API) plans a house of which these devices are
a part. It asks what they would do at given prices, and, once it has chosen,
asks this EMHASS to run its part. The API and its handlers come from the
optional home-energy-optimizer package (remote.ParticipantService); this
module builds this EMHASS's model for the horizon asked about, and publishes
a committed plan as a run would.

Switched on by `participant_api` in the configuration (an object; its
`publish_prefix` prefixes the sensors a committed plan is published to, so
two EMHASS on one Home Assistant do not collide) and a `participant_token`
in the secrets, which every call must present.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import pathlib
import pickle
import ssl
import time
import urllib.parse
import urllib.request

import orjson
import pandas as pd

from emhass import last_run, plan_store, utils
from emhass.command_line import (
    _record_optim_snapshot,
    default_csv_filename,
    prepare_forecast_and_weather_data,
    publish_data,
    set_input_data_dict,
)
from emhass.retrieve_hass import _SUPERVISOR_URLS

PARAMS_FILE = "params.pkl"
# The longest horizon a coordinator may ask about: a week, at any step.
MAX_HORIZON_MINUTES = 7 * 24 * 60


class ConfigUnreadable(Exception):
    """params.pkl is there but cannot be read: not the same as no configuration."""


async def read_params(emhass_conf: dict) -> dict | None:
    """The configuration EMHASS runs with (params.pkl), or None if there is none.

    Raises ConfigUnreadable when the file exists but cannot be read, so a
    corrupt save is not reported as the participant API being off."""
    path = pathlib.Path(emhass_conf["data_path"]) / PARAMS_FILE
    if not path.exists():
        return None
    try:
        _, params = pickle.loads(path.read_bytes())
    except Exception as exc:  # noqa: BLE001 - whatever a corrupt pickle raises
        raise ConfigUnreadable(f"{PARAMS_FILE} could not be read ({type(exc).__name__})") from exc
    return params


def soc_now(retrieve_hass_conf: dict, logger: logging.Logger) -> float | None:
    """The battery's state of charge now (0-1), from its
    `sensor_battery_state_of_charge` in Home Assistant, or None.

    The sensor is in percent, as EMHASS documents it: a reading in % or with
    no unit is divided by 100. Another unit, or a charge outside 0-100 %, is
    None (the configured target stands in)."""
    sensor = retrieve_hass_conf.get("sensor_battery_state_of_charge")
    base, token = retrieve_hass_conf.get("hass_url"), retrieve_hass_conf.get("long_lived_token")
    if isinstance(sensor, list):
        sensor = sensor[0] if sensor else None
    if not (isinstance(sensor, str) and sensor and base and token):
        return None
    # As RetrieveHass builds it: the Supervisor's URL already ends in /api.
    root = str(base).rstrip("/")
    path = "/states/" if root in _SUPERVISOR_URLS else "/api/states/"
    url = root + path + urllib.parse.quote(sensor)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    # ssl_no_verify, as the user set it for Home Assistant's other calls
    context = (
        ssl._create_unverified_context()  # noqa: S323 - the user's own setting
        if retrieve_hass_conf.get("ssl_no_verify")
        else None
    )
    try:
        with urllib.request.urlopen(req, timeout=5, context=context) as r:  # noqa: S310 - Home Assistant's own URL
            state = json.load(r)
        value = float(state["state"])
    except Exception as exc:  # noqa: BLE001 - any failure: the configured target stands in
        logger.warning("participant_api: could not read %s (%s)", sensor, exc)
        return None
    unit = str((state.get("attributes") or {}).get("unit_of_measurement") or "").strip()
    if unit not in ("%", ""):
        logger.warning("participant_api: %s is in %r, not %%", sensor, unit)
        return None
    soc = value / 100.0  # never guessed from the value: 0.5 % is 0.005
    if not 0.0 <= soc <= 1.0:
        logger.warning("participant_api: %s reads %s %%, outside 0-100", sensor, value)
        return None
    return soc


async def build_session(
    emhass_conf: dict,
    params: dict,
    window,
    logger: logging.Logger,
    get_data_from_file: bool = False,
):
    """This EMHASS's devices as one participant (home-energy-optimizer's
    EmhassParticipant) for `window` (remote.PlanWindow): its inputs built
    through the naive-MPC path with no PV and no load (a price response meters
    the devices alone), its battery starting from its sensor's state of
    charge. Raises remote.WindowRefused for slots it cannot plan."""
    from home_energy_optimizer.integrations.emhass import emhass_participant
    from home_energy_optimizer.remote import WindowRefused

    conf = params["optim_conf"].get("participant_api") or {}
    step = float(params["retrieve_hass_conf"]["optimization_time_step"])
    if abs(window.step_minutes - step) > 1e-9:
        raise WindowRefused(f"this EMHASS plans {step:g}-minute slots, not {window.step_minutes:g}")
    n = window.slots
    if n < 1 or n * step > MAX_HORIZON_MINUTES:
        raise WindowRefused(
            f"this EMHASS plans 1 to {int(MAX_HORIZON_MINUTES // step)} slots, not {n}"
        )
    runtime = {
        "prediction_horizon": n,
        "pv_power_forecast": [0.0] * n,
        "load_power_forecast": [0.0] * n,
    }
    if params["optim_conf"].get("load_cost_forecast_method") == "list":
        runtime["load_cost_forecast"] = [0.0] * n
    if params["optim_conf"].get("production_price_forecast_method") == "list":
        runtime["prod_price_forecast"] = [0.0] * n
    use_battery = bool(params["optim_conf"].get("set_use_battery"))
    target = params["plant_conf"].get("battery_target_state_of_charge", 0.5)
    if isinstance(target, list):
        target = target[0]
    soc = soc_now(params["retrieve_hass_conf"], logger) if use_battery else None
    if use_battery:
        if soc is None:
            logger.warning(
                "participant_api: no state of charge; planning from the target, %s", target
            )
        runtime["soc_init"] = target if soc is None else soc
        runtime["soc_final"] = target
    passed = copy.deepcopy(params)
    passed["passed_data"] = runtime
    idd = await set_input_data_dict(
        emhass_conf,
        params["optim_conf"].get("costfun", "profit"),
        orjson.dumps(passed).decode(),
        orjson.dumps(runtime).decode(),
        "naive-mpc-optim",
        logger,
        get_data_from_file=get_data_from_file,
        # its own inputs only: nothing it builds here touches the live problem
        dry_run=True,
    )
    if not idd:
        raise RuntimeError("EMHASS could not build its inputs for this horizon")
    try:
        df = prepare_forecast_and_weather_data(idd, logger)
        if isinstance(df, bool) or len(df) < n:
            raise WindowRefused(f"this EMHASS forecasts fewer than {n} slots")
        df = df.iloc[:n].copy()
        if window.start is not None:
            start = pd.Timestamp(window.start)
            start = (
                start.tz_convert(df.index.tz) if start.tzinfo else start.tz_localize(df.index.tz)
            )
            offset = (start - df.index[0]) / pd.Timedelta(minutes=step)
            if abs(offset) > 1e-6:
                # Relabelling the rows would plan the coordinator's slots with
                # another slot's forecasts; the coordinator asks again next run.
                raise WindowRefused(
                    f"the horizon starts {offset:+.2f} slots from this EMHASS's own; "
                    "the coordinator and this EMHASS must agree on the slot"
                )
        soc_init = runtime.get("soc_init")
        soc_final = runtime.get("soc_final")
        return emhass_participant(
            idd["opt"], df, soc_init, soc_final, key=str(conf.get("key", "emhass"))
        )
    finally:
        if "rh" in idd:
            await idd["rh"].close()


async def publish_commit(
    emhass_conf: dict,
    params: dict,
    res: pd.DataFrame,
    logger: logging.Logger,
    get_data_from_file: bool = False,
) -> tuple[bool, str]:
    """Run a committed plan as a run would: `res`, its own result for the
    answer chosen, published to Home Assistant under
    `participant_api.publish_prefix`, then saved as the latest plan and
    recorded for /api/v1/plan and /api/v1/last-run.

    Refused while continual_publish is on (its loop would put an earlier plan
    back) and for a plan that starts before the latest one (a late or repeated
    commit must not replace a newer plan)."""
    t0 = time.monotonic()
    conf = params["optim_conf"].get("participant_api") or {}
    if params["retrieve_hass_conf"].get("continual_publish"):
        return (
            False,
            "continual_publish is on: its loop would publish an earlier plan over this one",
        )
    latest = plan_store.read(pathlib.Path(emhass_conf["data_path"]))
    if latest and latest.get("plan") and len(res.index):
        latest_start = pd.Timestamp(latest["plan"][0]["timestamp"])
        start = res.index[0]
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        if start < latest_start:
            return False, f"the plan starts at {start}, before the latest plan ({latest_start})"
    runtime = {"publish_prefix": str(conf.get("publish_prefix", ""))}
    passed = copy.deepcopy(params)
    passed["passed_data"] = runtime
    idd = await set_input_data_dict(
        emhass_conf,
        params["optim_conf"].get("costfun", "profit"),
        orjson.dumps(passed).decode(),
        orjson.dumps(runtime).decode(),
        "publish-data",
        logger,
        get_data_from_file=get_data_from_file,
    )
    if not idd:
        return False, "EMHASS could not build its inputs to publish"
    # publish_data logs an entity Home Assistant did not take and goes on; a
    # commit must know, so this publish counts them.
    rh, refused = idd["rh"], []
    post = rh.post_data

    async def post_counted(*args, **kwargs):
        out = await post(*args, **kwargs)
        response = out[0] if isinstance(out, tuple) and out else out
        if response is not None and not getattr(response, "ok", True):
            refused.append(kwargs.get("entity_id", args[2] if len(args) > 2 else "?"))
        return out

    rh.post_data = post_counted
    try:
        # save_data_to_file: publish `res` itself, never the entity files a
        # prefix would otherwise republish.
        out = await publish_data(idd, logger, save_data_to_file=True, opt_res_latest=res)
        if out is None or refused:
            # Home Assistant has no transaction: what was published before the
            # failure stays, until this EMHASS publishes again.
            why = (
                f"Home Assistant did not take {', '.join(refused)}"
                if refused
                else "publishing failed"
            )
            return False, f"{why}; published perhaps part way, see the log"
        # The latest plan, whole, only once it is published.
        csv = pathlib.Path(emhass_conf["data_path"]) / default_csv_filename
        await utils.replace_file(csv, res.to_csv(index_label="timestamp"))
        _record_optim_snapshot(idd, last_run.ACTION_NAIVE_MPC_OPTIM, res, t0, logger)
        return True, ""
    finally:
        if "rh" in idd:
            await idd["rh"].close()


def make_service(emhass_conf: dict, params: dict, token: str, logger: logging.Logger):
    """The participant API for this EMHASS (remote.ParticipantService): its
    handlers run in a worker thread, each building or publishing with its own
    event loop."""
    from home_energy_optimizer.integrations.emhass import emhass_description
    from home_energy_optimizer.remote import ParticipantService, PlanWindow

    def build(window):
        return asyncio.run(build_session(emhass_conf, params, window, logger))

    def describe():
        step = float(params["retrieve_hass_conf"]["optimization_time_step"])
        return emhass_description(build(PlanWindow(step, int(round(24 * 60 / step)))))

    def commit(participant, answer):
        return asyncio.run(publish_commit(emhass_conf, params, answer.detail, logger))

    return ParticipantService(describe, build, commit, token=token)
