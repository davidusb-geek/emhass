"""This EMHASS's devices, served to a coordinator elsewhere (opt-in).

A coordinator (another EMHASS with `optimization_backend: dantzig_wolfe`, or
any client of the participant API) plans a house of which these devices are
a part. It asks what they would do at given prices, and, once it has chosen,
asks this EMHASS to run its part. The API's handlers are
emhass.coordinator.remote.ParticipantService; this module builds this
EMHASS's model for the horizon asked about, and publishes a committed plan
as a run would.

Switched on by `participant_api` in the configuration (an object; its
`publish_prefix` prefixes the sensors a committed plan is published to, so
two EMHASS on one Home Assistant do not collide) and a `participant_token`
in the secrets, which every call must present.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import pathlib
import pickle
import ssl
import threading
import time
import urllib.parse
import urllib.request
from typing import Any

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
from emhass.connection_manager import (
    PLACEHOLDER_URL_REFUSED,
    SUPERVISOR_TOKEN_REFUSED,
    SUPERVISOR_URLS,
    canonical_supervisor_url,
    is_refused_placeholder,
    is_supervisor_token_for_another_host,
)
from emhass.retrieve_hass import _upstream_error

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
    # The example hass_url is not anyone's Home Assistant. Never send it a token.
    if is_refused_placeholder(base, token):
        logger.error(PLACEHOLDER_URL_REFUSED)
        return None
    # The Supervisor token is only for the Supervisor proxy. Never send it to another host.
    if is_supervisor_token_for_another_host(base, token):
        logger.error(SUPERVISOR_TOKEN_REFUSED)
        return None
    # As RetrieveHass builds it: the Supervisor's URL already ends in /api.
    root = str(canonical_supervisor_url(base)).rstrip("/")
    path = "/states/" if root in SUPERVISOR_URLS else "/api/states/"
    url = root + path + urllib.parse.quote(sensor)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    # ssl_no_verify, as the user set it for Home Assistant's other calls
    context = (
        ssl._create_unverified_context()  # noqa: S323 - the user's own setting
        if retrieve_hass_conf.get("ssl_no_verify")
        else None
    )
    from emhass.coordinator.remote import _NoRedirect

    # Not followed, a redirect is an error: it would carry the token to a
    # host the configuration never named
    # and no proxy from the environment, which would receive the token
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect, urllib.request.HTTPSHandler(context=context)
    )
    try:
        with opener.open(req, timeout=5) as r:  # Home Assistant's own URL
            state = json.load(r)
    except Exception as exc:  # noqa: BLE001 - any failure: the configured target stands in
        # Its type and HTTP status, and for a connection that failed the type of the
        # reason (refused, not found, TLS). Never its text: a header error quotes the
        # header, token included.
        why = _upstream_error(exc)
        reason = getattr(exc, "reason", None)
        if isinstance(reason, BaseException):
            why += f", {type(reason).__name__}"
        logger.warning("participant_api: could not read %s (%s)", sensor, why)
        return None
    try:
        value = float(state["state"])
    except Exception as exc:  # noqa: BLE001 - not a number: the configured target stands in
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
    from emhass.coordinator.backend import emhass_participant, unsupported
    from emhass.coordinator.remote import WindowRefused
    from emhass.optimization import _coordinated_plant_problem

    conf = params["optim_conf"].get("participant_api") or {}
    # What the coordinator declines in its own house it declines here too:
    # its answers would not be what it can coordinate (a cost that moves with
    # the query, a model it cannot split). The coordinator records "declined".
    reason = _coordinated_plant_problem(params["plant_conf"]) or unsupported(
        params["optim_conf"],
        params["plant_conf"],
        params["optim_conf"].get("costfun", "profit"),
        {},
    )
    hybrid = params["plant_conf"].get("inverter_is_hybrid")
    if not reason and (any(hybrid) if isinstance(hybrid, list) else hybrid):
        # A participant's own solve has no inverter (the coordinator models
        # one as a node of its site): it would answer past the AC limits
        reason = "a hybrid inverter (its AC limits would not be held in its answers)"
    if reason:
        raise WindowRefused(f"this EMHASS cannot be coordinated: {reason}")
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
            # a naive start is UTC, as everywhere in EMHASS
            start = (start if start.tzinfo else start.tz_localize("UTC")).tz_convert(df.index.tz)
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
    on_recorded=None,
    origin: str | None = None,
) -> tuple[str, str]:
    """Run a committed plan as a run would: `res`, its own result for the
    answer chosen, published to Home Assistant under
    `participant_api.publish_prefix`, then saved as the latest plan and
    recorded for /api/v1/plan and /api/v1/last-run.

    Returns (state, reason): "accepted"; "refused", with nothing published,
    while continual_publish is on (its loop would put an earlier plan back),
    for a plan that starts before the latest one (a late commit must not
    replace a newer plan), or when its inputs cannot be built; "partial" when
    publishing failed after it began, since Home Assistant keeps what it took.
    `origin`: what made the plan (`commit:<id>`), saved with it, so that plan is
    known for that commit's own whatever its records say; `on_recorded`, its
    save stamp, once it is saved."""
    t0 = time.monotonic()
    if params["retrieve_hass_conf"].get("continual_publish"):
        return (
            "refused",
            "continual_publish is on: its loop would publish an earlier plan over this one",
        )
    latest = plan_store.read(pathlib.Path(emhass_conf["data_path"]))
    if latest and latest.get("plan") and len(res.index):
        latest_start = pd.Timestamp(latest["plan"][0]["timestamp"])
        start = res.index[0]
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        if start < latest_start:
            return "refused", f"the plan starts at {start}, before the latest plan ({latest_start})"

    async def record(idd: dict) -> None:
        # The latest plan, whole, only once it is published.
        csv = pathlib.Path(emhass_conf["data_path"]) / default_csv_filename
        await utils.replace_file(csv, res.to_csv(index_label="timestamp"))
        stamp = _record_optim_snapshot(
            idd, last_run.ACTION_NAIVE_MPC_OPTIM, res, t0, logger, origin=origin
        )
        # its own snapshot's stamp, as saved - never read back: a plan saved
        # since (a command-line run, which the live-optimization lock does not
        # hold back) would be taken for it
        if stamp is not None and on_recorded is not None:
            on_recorded(stamp)

    return await _publish(emhass_conf, params, res, logger, get_data_from_file, record)


async def _publish(
    emhass_conf: dict,
    params: dict,
    res: pd.DataFrame,
    logger: logging.Logger,
    get_data_from_file: bool = False,
    published=None,
) -> tuple[str, str]:
    """Publish `res`'s value for the current slot to Home Assistant, under
    `participant_api.publish_prefix`, as publish-data does; then, if given,
    `await published(input_data_dict)`. Returns (state, reason): "accepted";
    "refused", nothing published; "partial", published part way."""
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
        return "refused", "EMHASS could not build its inputs to publish"
    # publish_data logs an entity Home Assistant did not take and goes on; a
    # commit must know, so this publish counts them.
    rh, refused, posted = idd["rh"], [], []
    post = rh.post_data

    async def post_counted(*args, **kwargs):
        posted.append(True)  # from here, Home Assistant may hold part of the plan
        out = await post(*args, **kwargs)
        response = out[0] if isinstance(out, tuple) and out else out
        if response is not None and not getattr(response, "ok", True):
            refused.append(kwargs.get("entity_id", args[2] if len(args) > 2 else "?"))
        return out

    rh.post_data = post_counted
    try:
        # save_data_to_file: publish `res` itself, never the entity files a
        # prefix would otherwise republish.
        try:
            out = await publish_data(idd, logger, save_data_to_file=True, opt_res_latest=res)
        except Exception as exc:  # noqa: BLE001 - reported as the commit's outcome
            out = None
            logger.error(f"participant_api: publishing the committed plan failed: {exc}")
        if out is None or refused:
            # Home Assistant has no transaction: what was published before the
            # failure stays, until this EMHASS publishes again.
            why = (
                f"Home Assistant did not take {', '.join(refused)}"
                if refused
                else "publishing failed"
            )
            if not posted:
                return "refused", f"{why}; nothing was published"
            return "partial", f"{why}; published part way, see the log"
        if published is not None:
            await published(idd)
        return "accepted", ""
    finally:
        if "rh" in idd:
            await idd["rh"].close()


def current_slot_check(params: dict):
    """A check that a horizon starts at this EMHASS's current first slot, as
    it rounds the clock (method_ts_round): it raises remote.WindowRefused for
    one that does not, such as a session's whose first slot has passed."""
    from emhass.coordinator.remote import WindowRefused

    rh = params["retrieve_hass_conf"]
    step = pd.Timedelta(minutes=float(rh["optimization_time_step"]))

    def check(window) -> None:
        if window.start is None:
            return
        near = utils.forecast_starts_near(
            pd.Timestamp.now(tz="UTC"), step, rh["time_zone"], rh.get("method_ts_round", "nearest")
        )
        tz = next(iter(near)).tz
        start = pd.Timestamp(window.start)
        start = (start if start.tzinfo else start.tz_localize("UTC")).tz_convert(tz)  # naive: UTC
        if start not in near:
            raise WindowRefused(
                f"the horizon starts at {start}, not at this EMHASS's current slot "
                f"({', '.join(str(t) for t in sorted(near))}): "
                "the coordinator and this EMHASS must agree on the slot"
            )

    return check


class PlanStepper:
    """Steps this EMHASS's Home Assistant through its committed plan (until a
    newer commit, a plan of its own, continual_publish, or the plan's end).

    A commit, as publish-data, publishes each entity's value for the current
    slot only, and Home Assistant keeps it until something publishes again.
    So after a commit (accepted, or published part way), `follow` publishes
    the plan again at each of its slots - holding the live-optimization slot,
    retrying for a while if a live optimization holds it - until a newer
    commit replaces it, this EMHASS saves a plan of its own (`stamp`, what
    the latest saved plan was when the commit was made, changes), or the plan
    has ended (Home Assistant then keeps its last value). What a coordinator
    holds of a plan it no longer commits to is exactly that ("plan_steps")."""

    RETRY_S = 5.0  # a busy slot is tried again this often, within the slot

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gen = 0
        self._wake = threading.Event()
        self.following: str | None = None  # the commit whose plan it steps through
        self.log = None  # the commit log whose record of it is kept (pinned)
        self.stamp: str | None = None  # the saved plan its commit was made over
        self.superseded = False  # this EMHASS saved a plan of its own since
        self.resumed: bytes | None = None  # the params it last resumed with
        self._warned = -1  # the plan whose missing prefix was logged
        self._threads: list[threading.Thread] = []  # its loops, to wait for
        self._published: tuple[int, int] | None = None  # (generation, slot) last published

    def follow(
        self,
        commit_id: str,
        stamp: str | None,
        emhass_conf: dict,
        params: dict,
        res: pd.DataFrame,
        logger: logging.Logger,
        begin=None,
        publish=None,
        now_first: bool = False,
        expect_gen: int | None = None,
    ) -> None:
        """Step through `res` instead of any plan before (`now_first`: publish
        the current slot at once, as after a restart; `expect_gen`: only if no
        other commit was followed since - a resume yields to a commit)."""
        with self._lock:
            if expect_gen is not None and self._gen != expect_gen:
                return
            self._gen += 1
            gen = self._gen
            self._wake.set()  # a stepper waiting for its next slot ends now
            self._wake = threading.Event()
            wake = self._wake
            self.following, self.stamp, self.superseded = commit_id, stamp, False
        index = _utc_index(res)
        step = _step(params)
        # the slot Home Assistant holds: the commit's own publish (the current
        # slot) when it was accepted; not known when resumed or published part way
        rounding = params["retrieve_hass_conf"].get("method_ts_round", "nearest")
        held = None if now_first else _slot_of(index, pd.Timestamp.now(tz="UTC"), rounding)

        def attempt(slot_end: pd.Timestamp, last: bool = False) -> bool:
            # publish this slot, retrying while the slot is busy; False: stop
            nonlocal held
            delay = self.RETRY_S
            while True:
                try:
                    out = self.tick(
                        gen,
                        stamp,
                        emhass_conf,
                        params,
                        res,
                        logger,
                        begin,
                        publish,
                        pd.Timestamp.now(tz="UTC"),
                        last,
                    )
                except Exception as exc:  # noqa: BLE001 - a slot lost, not the plan
                    logger.warning(f"participant_api: stepping the committed plan failed: {exc}")
                    return True
                with self._lock:
                    if self._published is not None and self._published[0] == gen:
                        held = self._published[1]
                if out != "busy":
                    return out != "stop"
                # busy, or not (all) published: again within the slot, less
                # and less often (Home Assistant refusing an entity for good
                # is tried a few times a slot, not every few seconds)
                left = (slot_end - pd.Timestamp.now(tz="UTC")).total_seconds()
                if left <= delay or wake.wait(delay):
                    return not wake.is_set()
                delay = min(2 * delay, max(self.RETRY_S, step.total_seconds() / 4))

        def ended() -> None:
            # Home Assistant keeps the last value published: recorded when that
            # is not the plan's last (its final slots not published) and
            # cleared when it is, so a coordinator carries what runs
            if held is None or self.log is None:
                return
            with self._lock:
                current = gen == self._gen and not self.superseded  # its plan, still
            if current:
                last_slot = int(held) == len(index) - 1
                self.log.annotate(commit_id, {"held_slot": None if last_slot else int(held)})

        def over() -> bool:
            return not len(index) or pd.Timestamp.now(tz="UTC") >= index[-1] + step

        def loop() -> None:
            now = pd.Timestamp.now(tz="UTC")
            if now_first and not attempt(now + step, last=True):
                if over():
                    ended()  # an ended plan published once (after a restart)
                return
            while True:
                now = pd.Timestamp.now(tz="UTC")
                upcoming = index[index > now]
                if not len(upcoming):
                    ended()
                    return
                at = upcoming[0]
                # just after the plan's own slot boundary
                if wake.wait((at - now).total_seconds() + 1.0):
                    return
                if not attempt(at + step):
                    if over():
                        ended()  # its last slot's tries ran past the plan's end
                    return

        thread = threading.Thread(target=loop, name="participant-plan-steps", daemon=True)
        with self._lock:  # started under the lock: join never sees one unstarted
            thread.start()
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]

    def generation(self) -> int:
        with self._lock:
            return self._gen

    def view(self) -> tuple[int, str | None, str | None, bool]:
        """(generation, following, stamp, superseded), read together."""
        with self._lock:
            return self._gen, self.following, self.stamp, self.superseded

    def mark(
        self,
        commit_id: str,
        stamp: str | None,
        superseded: bool,
        expect_gen: int | None = None,
    ) -> None:
        """Record, under its lock, that `commit_id`'s plan (made over `stamp`)
        is the one it stands for, without stepping it (a resume that found it
        superseded): a newer generation, so a reader's earlier view of
        another plan never marks this one. `expect_gen`: only if no other
        commit was followed since (a resume yields to a commit)."""
        with self._lock:
            if expect_gen is not None and self._gen != expect_gen:
                return
            self._gen += 1
            self._wake.set()
            self.following, self.stamp, self.superseded = commit_id, stamp, superseded

    def supersede(self, gen: int, stamp: str | None) -> bool:
        """Mark the plan it follows superseded, if it is still the one seen
        (`gen`, `stamp`): a commit followed since is not. Whether it was."""
        with self._lock:
            if self._gen != gen or self.stamp != stamp:
                return False
            self.superseded = True
            return True

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
            self._wake.set()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for its loops to end (after stop: a slot being published
        finishes first). Whether they all did within `timeout` seconds."""
        with self._lock:
            threads = list(self._threads)
        end = None if timeout is None else time.monotonic() + timeout
        for t in threads:
            t.join(None if end is None else max(end - time.monotonic(), 0.0))
        return not any(t.is_alive() for t in threads)

    def _own(self, origin: str | None, gen: int) -> bool:
        """Whether a plan saved with `origin` is the one this stepper follows
        (its commit saved it), whatever stamp it was given."""
        with self._lock:
            following = self.following if gen == self._gen else None
        return following is not None and origin == _origin(following)

    def _end(self, gen: int) -> None:
        """Its plan no longer runs (a plan of this EMHASS's own replaced it):
        reported as such, and recorded, so no resume brings it back."""
        with self._lock:
            if gen != self._gen:
                return
            self.superseded = True
            following, log = self.following, self.log
        if log is not None and following is not None:
            log.annotate(following, {"superseded": True})

    def tick(
        self, gen, stamp, emhass_conf, params, res, logger, begin, publish, now, last=False
    ) -> str:
        """One slot's publish of `res`: "published", "busy" (the slot is held:
        try again) or "stop" (replaced, superseded or ended). `last`: a plan
        that ended (while this EMHASS was down) is published once - its last
        value, which Home Assistant then keeps."""
        with self._lock:
            if gen != self._gen:
                return "stop"
        index = _utc_index(res)
        ended = not len(index) or now >= index[-1] + _step(params)
        if ended and not last:
            return "stop"  # ended: Home Assistant keeps its last value
        # its params now, not as they were at the commit: a new publish_prefix
        # is used at once. The participant API switched off changes nothing
        # here (a plan committed runs on whether or not it answers): it is
        # published under the prefix it was committed with. With
        # continual_publish on, its loop publishes this EMHASS's own plan:
        # this one ends, for good (whatever else can or cannot be read).
        params = _with_participant_conf(_current_params(emhass_conf, params), params)
        if params["retrieve_hass_conf"].get("continual_publish"):
            self._end(gen)
            return "stop"
        saved, origin = _latest_saved(emhass_conf)  # one read: stamp and origin agree
        if saved is _UNREADABLE:
            return "busy"  # the plan store cannot be read now: tried again
        if saved != stamp and not self._own(origin, gen):
            self._end(gen)  # this EMHASS saved a plan of its own: it runs that
            return "stop"
        if not isinstance(params["optim_conf"].get("participant_api"), dict):
            # switched off, and the prefix it was committed with not known:
            # never published over this EMHASS's own (unprefixed) entities -
            # tried again, and published once a prefix is known
            if self._warned != gen:
                self._warned = gen
                logger.warning(
                    "participant_api: the committed plan's prefix is not known; not stepped"
                )
            return "busy"
        taken, release = begin() if begin is not None else ("ok", None)
        if taken != "ok":
            return "busy"
        try:
            with self._lock:
                if gen != self._gen:
                    return "stop"
            # again, under the slot: a live optimization it waited for may have
            # saved a plan of this EMHASS's own meanwhile
            saved, origin = _latest_saved(emhass_conf)
            if saved is _UNREADABLE:
                return "busy"
            if saved != stamp and not self._own(origin, gen):
                self._end(gen)
                return "stop"
            state, why = (publish or _publish_now)(emhass_conf, params, res, logger)
            if state != "accepted":
                # not (all) published: this slot is tried again
                logger.warning(f"participant_api: stepping the committed plan: {state} ({why})")
                return "busy"
            with self._lock:
                self._published = (
                    gen,
                    _slot_of(index, now, params["retrieve_hass_conf"].get("method_ts_round")),
                )
            return "stop" if ended else "published"
        finally:
            if release is not None:
                release()


def _slot_of(index: pd.DatetimeIndex, now: pd.Timestamp, rounding: str | None = None) -> int:
    """The slot of `index` publishing picks at `now`, as publish-data does
    (command_line._get_closest_index: the minute, then method_ts_round)."""
    if not len(index):
        return 0
    at = pd.Timestamp(now).floor("min")
    idx = pd.DatetimeIndex(index)
    at = at.tz_convert(idx.tz) if idx.tz is not None else at.tz_localize(None)
    method = {"first": "ffill", "last": "bfill"}.get(rounding or "nearest", "nearest")
    k = int(idx.get_indexer([at], method=method)[0])
    if k == -1:
        k = int(idx.get_indexer([at], method="nearest")[0])
    return min(max(k, 0), len(idx) - 1)


_UNREADABLE = object()


def _origin(commit_id: str) -> str:
    """What a commit's plan is saved as made by (plan_store's origin)."""
    return f"commit:{commit_id}"


def _latest_saved(emhass_conf: dict):
    """(stamp, origin) of the latest plan saved, from one read of it: its
    identity (plan_store.stamp; None: none saved) and what made it - or
    (_UNREADABLE, None) when its file is there but cannot be read now."""
    data = pathlib.Path(emhass_conf["data_path"])
    latest = plan_store.read(data)
    try:
        if latest is None and plan_store._path(data).exists():
            return _UNREADABLE, None
    except OSError:  # not even its presence can be told
        return _UNREADABLE, None
    return plan_store.stamp(latest), (latest or {}).get("origin")


def _saved_stamp(emhass_conf: dict):
    """The save stamp of the latest plan saved (see _latest_saved)."""
    return _latest_saved(emhass_conf)[0]


def _with_participant_conf(params: dict, committed: dict) -> dict:
    """`params`, with the participant_api of `committed` (the params a commit
    was made with) when it has none of its own - switched off since: its
    plan is published under the prefix it was committed with."""
    if isinstance(params["optim_conf"].get("participant_api"), dict):
        return params
    conf = (committed.get("optim_conf") or {}).get("participant_api")
    if not isinstance(conf, dict):
        return params
    out = copy.deepcopy(params)
    out["optim_conf"]["participant_api"] = copy.deepcopy(conf)
    return out


def _current_params(emhass_conf: dict, fallback: dict) -> dict:
    """This EMHASS's params as saved now (params.pkl), or `fallback` when
    they cannot be read."""
    try:
        params = asyncio.run(read_params(emhass_conf))
    except Exception:  # noqa: BLE001 - what it had
        return fallback
    return params if isinstance(params, dict) else fallback


def _step(params: dict) -> pd.Timedelta:
    return pd.Timedelta(minutes=float(params["retrieve_hass_conf"]["optimization_time_step"]))


def _utc_index(res: pd.DataFrame) -> pd.DatetimeIndex:
    """`res`'s index as UTC timestamps, whatever it was read as (a CSV across
    a change of daylight saving time reads back as text)."""
    return pd.DatetimeIndex(pd.to_datetime(res.index, utc=True))


def _publish_now(emhass_conf, params, res, logger) -> tuple[str, str]:
    return asyncio.run(_publish(emhass_conf, params, res, logger))


_steppers: dict[pathlib.Path, PlanStepper] = {}
COMMITTED_PLANS = "participant_plans"  # each commit's plan, for resuming it


def committed_plan_path(emhass_conf: dict, commit_id: str) -> pathlib.Path:
    """Where commit `commit_id`'s plan is kept: a name made from the id (the
    coordinator's, so never used as a path itself)."""
    name = hashlib.sha256(str(commit_id).encode()).hexdigest()[:32]
    return pathlib.Path(emhass_conf["data_path"]) / COMMITTED_PLANS / f"{name}.csv"


def _not_kept(path: pathlib.Path, why: str) -> None:
    """Remove a refused commit's kept plan. One that cannot be removed may be
    stepped by a restart (its record may still read running on disk): then
    it is no refusal but a commit that may run - raised, so it is recorded
    so, which a coordinator keeps counting."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise OSError(
            f"{why}, and its kept plan could not be removed: nothing was published, but a "
            "restart may run it"
        ) from exc


def _keep_plans_of(emhass_conf: dict, commit_ids: set[str]) -> None:
    """Remove the plans kept for commits no longer in the log."""
    keep = {committed_plan_path(emhass_conf, c).name for c in commit_ids}
    folder = pathlib.Path(emhass_conf["data_path"]) / COMMITTED_PLANS
    for f in folder.glob("*.csv") if folder.is_dir() else []:
        if f.name not in keep:
            f.unlink(missing_ok=True)


def plan_stepper(emhass_conf: dict) -> PlanStepper:
    """This EMHASS's plan stepper, one per data folder."""
    path = pathlib.Path(emhass_conf["data_path"])
    with _commit_logs_lock:
        if path not in _steppers:
            _steppers[path] = PlanStepper()
        return _steppers[path]


# The commits' records, one log per data folder for the life of the process:
# a service rebuilt (new params, a new token) keeps the commits it answered.
_commit_logs: dict[pathlib.Path, Any] = {}
_commit_logs_lock = threading.Lock()
COMMIT_LOG_FILE = "participant_commits.json"


def commit_log(emhass_conf: dict):
    """This EMHASS's commit log (remote.CommitLog), in its data folder."""
    from emhass.coordinator.remote import CommitLog

    path = pathlib.Path(emhass_conf["data_path"]) / COMMIT_LOG_FILE
    with _commit_logs_lock:
        if path not in _commit_logs:
            _commit_logs[path] = CommitLog(path)
        return _commit_logs[path]


def _resume(emhass_conf: dict, params: dict, logger, stepper: PlanStepper, begin) -> None:
    """After a restart, or with new params (a service rebuilt): go on stepping
    the plan of the last commit that ran (accepted, or published part way),
    from the current slot at once, if no plan of this EMHASS's own was saved
    since - the plan kept for that very commit."""
    key = orjson.dumps(params, option=orjson.OPT_SORT_KEYS, default=str)
    if stepper.resumed == key:
        return
    stepper.resumed = key
    gen = stepper.generation()  # a commit followed while this resumes wins
    log = commit_log(emhass_conf)
    stepper.log = log  # an end it sees is recorded, served or not
    while True:
        last = log.latest(running=True)
        if last is None or last.get("state") not in ("accepted", "partial"):
            return
        if committed_plan_path(emhass_conf, last["commit_id"]).exists():
            break
        # Cut off before its plan was kept - before anything was published,
        # which comes after: nothing of it ran. The commit before it runs.
        log.end(last["commit_id"], "refused", "EMHASS stopped before it published this plan")
    _, following, _, superseded = stepper.view()
    if following == last["commit_id"] and not superseded:
        return  # stepping it already
    # published under the prefix it was committed with, if switched off since
    params = _with_participant_conf(
        params, {"optim_conf": {"participant_api": last.get("participant_api")}}
    )
    if last.get("superseded"):
        return  # a plan of its own replaced it: it never runs again
    if params["retrieve_hass_conf"].get("continual_publish"):
        # its loop publishes this EMHASS's own plan: this one ends, for good
        stepper.mark(last["commit_id"], last.get("plan_stamp"), True)
        log.annotate(last["commit_id"], {"superseded": True})
        return
    try:
        res = pd.read_csv(
            committed_plan_path(emhass_conf, last["commit_id"]), index_col="timestamp"
        )
        res.index = _utc_index(res)
    except Exception as exc:  # noqa: BLE001 - not now: tried again next time
        logger.warning(f"participant_api: the committed plan cannot be read back ({exc})")
        stepper.resumed = None
        return
    # the stamp and what made it, from the same read: a commit saving its own
    # plan between two reads would be taken for that one under another's stamp
    saved, origin = _latest_saved(emhass_conf)
    if saved is _UNREADABLE:
        stepper.resumed = None  # the plan store cannot be read now: next time
        return
    if "plan_stamp" not in last:
        # cut off by a restart while publishing: made over the plan saved now,
        # if none of this EMHASS's own came since - kept, so a later resume
        # never takes a newer plan of its own for that one
        log.annotate(last["commit_id"], {"plan_stamp": saved})
        last = {**last, "plan_stamp": saved}
    stamp = last["plan_stamp"]
    if saved != stamp and origin == _origin(last["commit_id"]):
        # The plan saved is this commit's own (it says so): saved before its
        # stamp could be noted (a stop, a record that could not be saved)
        log.annotate(last["commit_id"], {"plan_stamp": saved})
        stamp = saved
    if last.get("interrupted"):
        log.annotate(last["commit_id"], {"interrupted": False})  # settled
    if saved != stamp:
        stepper.mark(last["commit_id"], stamp, True, expect_gen=gen)  # a commit since wins
        log.annotate(last["commit_id"], {"superseded": True})
        return  # its own plan runs now
    stepper.follow(
        last["commit_id"],
        stamp,
        emhass_conf,
        params,
        res,
        logger,
        begin=begin,
        now_first=True,
        expect_gen=gen,  # a commit followed meanwhile wins
    )


def resume_steps(emhass_conf: dict, params: dict, logger, begin=None) -> None:
    """Resume stepping the committed plan if it stopped (see _resume)."""
    _resume(emhass_conf, params, logger, plan_stepper(emhass_conf), begin)


def _active_commit(log, stepper: PlanStepper, data_path=None):
    """The record of the commit whose plan this EMHASS runs now: one still
    publishing; else the one its stepper follows, unless a plan of its own
    replaced it (the latest plan saved is no longer the one its commit was
    made over)."""
    latest = log.latest(running=True)
    if latest is not None and latest["state"] == "running":
        return latest
    gen, following, stamp, superseded = stepper.view()
    if following is None or superseded:
        return None
    if data_path is not None:
        saved, origin = _latest_saved({"data_path": data_path})  # one read
        if saved is not _UNREADABLE and saved != stamp and origin != _origin(following):
            if stepper.supersede(gen, stamp):
                return None  # a plan of its own replaced the one it follows
            # a commit followed meanwhile, over what was saved then: that one
            _, following, _, superseded = stepper.view()
            if following is None or superseded:
                return None
    record = log.get(following)
    if record is None or record.get("superseded"):
        return None
    return record if record["state"] in ("accepted", "partial") else None


def make_service(
    emhass_conf: dict,
    params: dict,
    token: str,
    logger: logging.Logger,
    begin=None,
):
    """The participant API for this EMHASS (remote.ParticipantService): its
    handlers run in a worker thread, each building or publishing with its own
    event loop. `begin`: what a commit takes before it runs (the web
    server's live-optimization slot), as ParticipantService takes it."""
    from emhass.coordinator.backend import emhass_description
    from emhass.coordinator.remote import ParticipantService, PlanWindow

    def build(window):
        return asyncio.run(build_session(emhass_conf, params, window, logger))

    described: list[dict] = []  # params are fixed for this service's life

    def describe():
        # Every coordinator run asks: built once (a model, its inputs from
        # Home Assistant), not on every run
        if not described:
            step = float(params["retrieve_hass_conf"]["optimization_time_step"])
            described.append(
                emhass_description(build(PlanWindow(step, int(round(24 * 60 / step)))))
            )
        return dict(described[0])

    stepper = plan_stepper(emhass_conf)
    log = commit_log(emhass_conf)
    stepper.log = log

    def commit(participant, answer, commit_id):
        res = answer.detail
        before = _saved_stamp(emhass_conf)
        if before is _UNREADABLE:
            # what it would be made over is not known: never published
            return "refused", "its plan store cannot be read now; nothing was published"
        # Its plan, kept first and under its own commit's name: a restart part
        # way through publishing still knows what it was publishing, and a
        # refused commit leaves the plan that runs as it was. Not kept, it is
        # not published: a resume would take it for one cut off before
        # anything was published.
        try:
            path = committed_plan_path(emhass_conf, commit_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            asyncio.run(utils.replace_file(path, res.to_csv(index_label="timestamp")))
        except Exception as exc:  # noqa: BLE001 - refused, nothing published
            logger.warning(f"participant_api: the committed plan cannot be kept ({exc})")
            return (
                "refused",
                f"its plan could not be kept ({type(exc).__name__}); nothing was published",
            )
        try:
            kept = {r["commit_id"] for r in log.records()} | {commit_id}
            _keep_plans_of(emhass_conf, kept | ({stepper.following} - {None}))
        except Exception as exc:  # noqa: BLE001 - old plans left behind, nothing more
            logger.warning(f"participant_api: older committed plans not removed ({exc})")
        # what it is made over, and the prefix it is published under, in its
        # record before anything is published: a restart part way through
        # leaves it the right stamp, and a later resume the right prefix
        kept_record = log.annotate(
            commit_id,
            {"plan_stamp": before, "participant_api": params["optim_conf"].get("participant_api")},
        )
        if kept_record is False:
            # A restart would not know what it was made over: never published.
            # Nor kept: its record may still read running on disk, and a
            # restart then finds no plan of it, so knows nothing of it ran.
            _not_kept(path, "its record could not be saved")
            return "refused", "its record could not be saved; nothing was published"

        confirmed: dict = {}

        def recorded(stamp):  # the plan saved as its own: its stamp at once
            confirmed["stamp"] = stamp  # followed and recorded whether or not this save holds
            log.annotate(commit_id, {"plan_stamp": stamp})

        try:
            state, why = asyncio.run(
                publish_commit(
                    emhass_conf,
                    params,
                    res,
                    logger,
                    on_recorded=recorded,
                    origin=_origin(commit_id),
                )
            )
        except Exception as exc:  # noqa: BLE001 - it may have published part of it
            logger.warning(f"participant_api: publishing the committed plan failed ({exc})")
            state, why = "partial", f"publishing failed ({type(exc).__name__}); published part way"
        if state not in ("accepted", "partial"):
            # refused: nothing published, the plan before it runs on - and its
            # own plan not kept, so a restart (its final record not saved,
            # say) never steps it
            _not_kept(path, f"it was refused ({why})")
            return state, why
        # It runs this plan from now on - published part way too: Home Assistant
        # holds part of it, and stepping it brings the rest - so its next slots
        # are published as they come. What it was made over: the plan saved now
        # (its own when accepted, the one before when published part way).
        # (its record's: `before`, unless its plan was saved as its own, which
        # `recorded` noted then - not read back here)
        stamp = confirmed.get("stamp", (log.get(commit_id) or {}).get("plan_stamp", before))
        try:
            # published part way: the rest of this slot at once, not at the next
            stepper.follow(
                commit_id,
                stamp,
                emhass_conf,
                params,
                res,
                logger,
                begin=begin,
                now_first=(state == "partial"),
            )
        except Exception as exc:  # noqa: BLE001 - published all the same
            logger.warning(f"participant_api: the committed plan will not be stepped ({exc})")
        return state, why, {"plan_stamp": stamp}

    _resume(emhass_conf, params, logger, stepper, begin)

    return ParticipantService(
        describe,
        build,
        commit,
        token=token,
        begin=begin,
        log=log,
        current=current_slot_check(params),
        active=lambda: _active_commit(log, stepper, emhass_conf["data_path"]),
    )
