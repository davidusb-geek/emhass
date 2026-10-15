"""Offline harness: drives EMHASS through its public entry points with a frozen clock.

* No Home Assistant: hass_url and token are empty, so RetrieveHass.get_ha_config() returns
  early and publish_data(..., dont_post=True) never posts.
* No network: all forecasts are passed at runtime (lists, timestamp dicts) or come from a
  CSV / a pre-seeded Open-Meteo cache file in a temporary data_path.
* One clock (except Env.set_input(forecast_now=...)): time_machine freezes datetime.now(), pd.Timestamp.now() and time.time()
  together, which covers every clock entry point EMHASS reads (utils._get_now,
  Forecast.__init__, publish's _get_closest_index, cache ages).
* No patches to EMHASS code.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import pathlib
import tempfile
from unittest import mock

import orjson
import pandas as pd
import time_machine

import emhass
from emhass import command_line as cl
from emhass import utils

ROOT = pathlib.Path(emhass.__file__).resolve().parent

LOG = logging.getLogger("emhass-dst-tests")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False

os.environ.pop("SUPERVISOR_TOKEN", None)  # would make get_ha_config() call the supervisor


_LOOP: asyncio.AbstractEventLoop | None = None


def run_async(coro):
    """Run a coroutine on one event loop per process.

    asyncio.run() builds a new loop every call. On Windows a new loop creates its self-pipe
    from a loopback TCP socketpair, whose accept() can block forever when many pytest-xdist
    workers create loops at the same moment; one long-lived loop avoids that and is faster.
    """
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
    return _LOOP.run_until_complete(coro)


def marker(ts) -> float:
    """Value that identifies the instant it belongs to: minutes since 2026-01-01T00:00Z."""
    return (pd.Timestamp(ts).timestamp() - 1767225600) / 60.0


def bucket_mean_marker(ts, step: int, src_step: int = 5) -> float:
    """Expected value at grid point ts if src_step-minute marker points are averaged into
    [ts, ts+step) (the documented resample-mean rule of timestamped runtime forecasts)."""
    return marker(ts) + (step - src_step) / 2.0


def utc_list(index) -> list[pd.Timestamp]:
    return [pd.Timestamp(t).tz_convert("UTC") for t in pd.to_datetime(list(index), utc=True)]


class Env:
    """One EMHASS configuration in its own temporary data_path."""

    def __init__(self, zone: str, step: int, mode: str, delta_days: int = 1, **optim):
        self.zone, self.step, self.mode = zone, step, mode
        self.tmp = tempfile.TemporaryDirectory(prefix="emhass-dst-")
        self.data_path = pathlib.Path(self.tmp.name)
        self.conf = {
            "data_path": self.data_path,
            "root_path": ROOT,
            "defaults_path": ROOT / "data/config_defaults.json",
            "associations_path": ROOT / "data/associations.csv",
            "config_path": self.data_path / "config.json",
        }
        self.params = run_async(self._params(delta_days, optim))

    async def _params(self, delta_days, optim):
        config = await utils.build_config(self.conf, LOG, self.conf["defaults_path"])
        secrets = {
            "hass_url": "",
            "long_lived_token": "",
            "time_zone": self.zone,
            "Latitude": 45.0,
            "Longitude": 7.0,
            "Altitude": 100.0,
            "solcast_api_key": "",
            "solcast_rooftop_id": "",
            "solar_forecast_kwp": 0,
            "influxdb_username": "",
            "influxdb_password": "",
            "victoriametrics_username": "",
            "victoriametrics_password": "",
        }
        params = await utils.build_params(self.conf, secrets, config, LOG)
        params["retrieve_hass_conf"]["method_ts_round"] = self.mode
        params["retrieve_hass_conf"]["optimization_time_step"] = self.step
        params["retrieve_hass_conf"]["continual_publish"] = False
        params["optim_conf"]["delta_forecast_daily"] = delta_days
        params["optim_conf"].update(optim)
        return params

    def close(self):
        self.tmp.cleanup()

    # -- actions ---------------------------------------------------------------------

    def set_input(
        self,
        now_utc: pd.Timestamp,
        runtimeparams: dict,
        action: str,
        forecast_now: pd.Timestamp | None = None,
    ):
        """set_input_data_dict under a frozen clock. Returns (idd, error_repr).

        ``forecast_now`` makes ``pd.Timestamp.now()`` (what Forecast.__init__ reads) report a
        different instant than the frozen ``datetime.now()`` (what treat_runtimeparams reads),
        to emulate a request that crosses a time-step boundary between the two reads.
        """
        params = dict(self.params)
        params["passed_data"] = runtimeparams
        if hasattr(cl, "OptimizationCache"):
            cl.OptimizationCache.clear()
        second_clock = contextlib.nullcontext()
        if forecast_now is not None:
            fixed = pd.Timestamp(forecast_now).tz_convert("UTC")

            def _now(tz=None):
                return fixed.tz_convert(tz) if tz is not None else fixed.tz_localize(None)

            second_clock = mock.patch.object(pd.Timestamp, "now", staticmethod(_now))
        with time_machine.travel(now_utc.to_pydatetime(), tick=False), second_clock:
            try:
                idd = run_async(
                    cl.set_input_data_dict(
                        self.conf,
                        "profit",
                        orjson.dumps(params).decode(),
                        orjson.dumps(runtimeparams).decode(),
                        action,
                        LOG,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - the error is the observation
                return None, f"{type(exc).__name__}: {exc}"
        if not idd:
            return None, "set_input_data_dict returned False"
        return idd, None

    def runtime_grid(self, now_utc: pd.Timestamp, delta_days: int, horizon: int | None):
        """The grid treat_runtimeparams builds (utils.get_forecast_dates), same clock."""
        with time_machine.travel(now_utc.to_pydatetime(), tick=False):
            dates = utils.get_forecast_dates(
                self.step, delta_days, self.params_tz(), method_ts_round=self.mode
            )
        dates = utc_list(dates)
        return dates[:horizon] if horizon else dates

    def params_tz(self):
        import pytz

        return pytz.timezone(self.zone)

    def prepare_optim_frame(self, idd, now_utc):
        with time_machine.travel(now_utc.to_pydatetime(), tick=False):
            return cl.prepare_forecast_and_weather_data(idd, LOG)

    def naive_mpc(self, idd, now_utc):
        """Solve and write opt_res_latest.csv like the add-on does (debug=False)."""
        with time_machine.travel(now_utc.to_pydatetime(), tick=False):
            return run_async(cl.naive_mpc_optim(idd, LOG, save_data_to_file=False, debug=False))

    def publish(self, now_utc: pd.Timestamp, runtimeparams: dict | None = None):
        """publish-data as its own call (as Home Assistant triggers it), nothing posted."""
        idd, err = self.set_input(now_utc, runtimeparams or {}, "publish-data")
        if err:
            return None, err
        with time_machine.travel(now_utc.to_pydatetime(), tick=False):
            try:
                res = run_async(cl.publish_data(idd, LOG, dont_post=True))
            except Exception as exc:  # noqa: BLE001
                return None, f"{type(exc).__name__}: {exc}"
        if res is None:
            return None, "publish_data returned None"
        return res, None


def list_runtimeparams(step: int, horizon: int, n: int | None = None, value=1000.0) -> dict:
    """Plain list forecasts long enough for any extension of the horizon."""
    n = n or (3 * 1440 // step + 12)
    vals = [float(value)] * n
    return {
        "prediction_horizon": horizon,
        "optimization_time_step": step,
        "pv_power_forecast": vals,
        "load_power_forecast": vals,
        "load_cost_forecast": [0.2] * n,
        "prod_price_forecast": [0.1] * n,
    }


def marker_dict(now_utc: pd.Timestamp, days: int = 4, src_step: int = 5) -> dict:
    """Timestamped runtime forecast: one point every src_step minutes (UTC-aligned, so it
    contains every local grid point of every zone with a 15-minute-multiple offset)."""
    idx = pd.date_range(
        now_utc.floor("h") - pd.Timedelta(hours=3),
        now_utc + pd.Timedelta(days=days),
        freq=f"{src_step}min",
    )
    return {t.isoformat(): marker(t) for t in idx}
