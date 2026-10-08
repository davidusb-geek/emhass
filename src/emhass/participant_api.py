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
import urllib.parse
import urllib.request

import orjson
import pandas as pd

from emhass.command_line import (
    default_csv_filename,
    prepare_forecast_and_weather_data,
    publish_data,
    set_input_data_dict,
)

PARAMS_FILE = "params.pkl"


async def read_params(emhass_conf: dict) -> dict | None:
    """The configuration EMHASS runs with (params.pkl), or None."""
    path = pathlib.Path(emhass_conf["data_path"]) / PARAMS_FILE
    if not path.exists():
        return None
    try:
        _, params = pickle.loads(path.read_bytes())
    except (EOFError, pickle.UnpicklingError, UnicodeDecodeError, ValueError):
        return None
    return params


def soc_now(retrieve_hass_conf: dict, logger: logging.Logger) -> float | None:
    """The battery's state of charge now (0-1), from its
    `sensor_battery_state_of_charge` in Home Assistant, or None."""
    sensor = retrieve_hass_conf.get("sensor_battery_state_of_charge")
    base, token = retrieve_hass_conf.get("hass_url"), retrieve_hass_conf.get("long_lived_token")
    if isinstance(sensor, list):
        sensor = sensor[0] if sensor else None
    if not (isinstance(sensor, str) and sensor and base and token):
        return None
    url = str(base).rstrip("/") + "/api/states/" + urllib.parse.quote(sensor)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310 - Home Assistant's own URL
            value = float(json.load(r)["state"])
    except Exception as exc:  # noqa: BLE001 - any failure: the configured target stands in
        logger.warning("participant_api: could not read %s (%s)", sensor, exc)
        return None
    return value / 100.0 if value > 1.0 else value


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
    runtime = {
        "prediction_horizon": n,
        "pv_power_forecast": [0.0] * n,
        "load_power_forecast": [0.0] * n,
        # its own inputs only: nothing it builds here touches the live problem
        "dry_run": True,
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
            if abs(offset) > 1:
                raise WindowRefused(
                    f"the horizon starts {offset:+.1f} slots from this EMHASS's own"
                )
            df.index = pd.date_range(start, periods=n, freq=pd.Timedelta(minutes=step))
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
    answer chosen, saved as the latest plan and published to Home Assistant
    under `participant_api.publish_prefix`."""
    conf = params["optim_conf"].get("participant_api") or {}
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
    try:
        res.to_csv(
            pathlib.Path(emhass_conf["data_path"]) / default_csv_filename, index_label="timestamp"
        )
        out = await publish_data(idd, logger, opt_res_latest=res)
        return out is not None, "" if out is not None else "publishing failed; see the log"
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
