"""Participants over the network: a solver in another process or container,
answering the participant API.

`RemoteParticipant` is the coordinator's side: an interface.Participant whose
answers come over HTTP. `ParticipantService` is the solver's side - the API's
handlers, which EMHASS serves from its own web server at /participant/v1/.

Queries have no effect; only a commit does. A session is the horizon a
coordinator plans - its start, slot length and slot count - so queries carry
no session of their own: the solver builds its model for a horizon once, and
answers every query on it from that model.

Privacy: a remote is asked what it would do at given prices. A best response
also carries the rest of the house's load (its `residual_kw`), so it is sent
only to a remote that lists "best_response" among the kinds it answers;
otherwise the same prices are asked as a price response.
"""

from __future__ import annotations

import datetime
import hmac
import http.client
import json
import os
import pathlib
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from emhass.coordinator.interface import (
    FEATURES,
    KINDS,
    SCHEMA_VERSION,
    Answer,
    Participant,
    Query,
    answer_from_dict,
    answer_to_dict,
    query_from_dict,
    query_to_dict,
)

# method, URL, headers, body -> HTTP status, body
Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, bytes]]


@dataclass(frozen=True)
class PlanWindow:
    """The horizon a coordinator plans: `slots` slots of `step_minutes`
    each, the first starting at `start` (ISO 8601) when known."""

    step_minutes: float
    slots: int
    start: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The API's Horizon object."""
        out: dict[str, Any] = {"step_minutes": float(self.step_minutes), "slots": int(self.slots)}
        if self.start is not None:
            out["start"] = self.start
        return out

    @classmethod
    def from_dict(cls, d: Any) -> PlanWindow:
        """A Horizon object back; ValueError if it is not one."""
        if not isinstance(d, dict) or set(d) - {"step_minutes", "slots", "start"}:
            raise ValueError("horizon must be an object with step_minutes, slots and start")
        step, slots, start = d.get("step_minutes"), d.get("slots"), d.get("start")
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not 0 < step <= 1440:
            raise ValueError("horizon.step_minutes must be in (0, 1440]")
        if isinstance(slots, bool) or not isinstance(slots, int) or not 1 <= slots <= 10_000:
            raise ValueError("horizon.slots must be an integer in [1, 10000]")
        if start is not None and not (isinstance(start, str) and len(start) <= 64):
            raise ValueError("horizon.start must be an ISO 8601 date-time")
        return cls(float(step), slots, start)


class RemoteError(RuntimeError):
    """A remote refused a call, or could not be reached. `status` is the HTTP
    status, or 0 when nothing came back."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class RemoteUnavailable(RemoteError):
    """A remote that has not answered once: there is no plan to stand in."""


class WindowRefused(Exception):
    """Raised by a ParticipantService's `build` for a horizon it cannot plan
    (another slot length, too long): the caller gets 409 and why."""


class PlainHttpRefused(RemoteError):
    """A token would go over plain http:// to another machine, and the remote
    is not marked `trusted_network`: it is not called at all."""


def _loopback(url: str) -> bool:
    """Whether `url` names this machine (its token never leaves it): the
    web server's own test (api_auth.is_loopback)."""
    from emhass.api_auth import is_loopback

    return is_loopback(urllib.parse.urlsplit(url).hostname)


# The most a remote's answer may be: a query or an answer is a few kB.
MAX_RESPONSE_BYTES = 1 << 20


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is answered as it came (an HTTP error), never followed: it
    would carry the bearer token to a host the configuration never named."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        return None


def _read(r, deadline: float) -> bytes:
    """`r`'s body, at most MAX_RESPONSE_BYTES, before `deadline`
    (time.monotonic()): a remote answering slowly or at length is refused."""
    out = bytearray()
    # read1: what has arrived (up to 64 kB), so the deadline is checked as
    # the answer trickles in, not once a whole chunk has come
    read = getattr(r, "read1", r.read)
    while True:
        chunk = read(65536)
        if not chunk:
            return bytes(out)
        out += chunk
        if len(out) > MAX_RESPONSE_BYTES:
            raise http.client.HTTPException(f"an answer over {MAX_RESPONSE_BYTES} bytes")
        if time.monotonic() > deadline:
            raise TimeoutError("the answer took longer than the timeout")


class _Tracked:
    """http.client connections that note their sockets in `socks`, so a call
    past its deadline can be cut off whatever it is waiting for."""

    def __init__(self, socks: list[socket.socket]) -> None:
        self.socks = socks

    def __call__(self, base: type[http.client.HTTPConnection]) -> Any:
        socks = self.socks

        class Connection(base):  # type: ignore[valid-type,misc]
            def connect(self) -> None:
                super().connect()
                socks.append(self.sock)

        return Connection


class _HTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, track: _Tracked) -> None:
        super().__init__()
        self._track = track

    def http_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(self._track(http.client.HTTPConnection), req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, track: _Tracked) -> None:
        super().__init__()
        self._track = track

    def https_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(
            self._track(http.client.HTTPSConnection),
            req,
            context=self._context,  # type: ignore[attr-defined]
        )


def urllib_transport(timeout: float) -> Transport:
    """The standard library's HTTP client as a Transport: `timeout` seconds
    for the whole call - resolving the host, connecting, the headers and the
    body -, redirects not followed, answers of at most MAX_RESPONSE_BYTES."""

    def call(
        method: str, url: str, headers: dict[str, str], body: bytes | None, socks: list[Any]
    ) -> tuple[int, bytes]:
        track = _Tracked(socks)
        opener = urllib.request.build_opener(_NoRedirect, _HTTPHandler(track), _HTTPSHandler(track))
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        deadline = time.monotonic() + timeout
        try:
            with opener.open(req, timeout=timeout) as r:  # http(s) only, checked by the caller
                return int(r.status), _read(r, deadline)
        except urllib.error.HTTPError as exc:
            return int(exc.code), _read(exc, deadline)

    def send(
        method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        # A socket timeout bounds each read, not the call: a remote sending
        # its headers a byte at a time would hold the plan. The call runs in
        # a thread, and past the deadline its socket is shut, which ends it.
        socks: list[socket.socket] = []
        out: dict[str, Any] = {}

        def run() -> None:
            try:
                out["answer"] = call(method, url, headers, body, socks)
            except BaseException as exc:  # noqa: BLE001 - raised again below
                out["error"] = exc

        worker = threading.Thread(target=run, name="emhass-remote-call", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            for s in socks:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            raise TimeoutError(f"no answer within {timeout:g} s")
        if "error" in out:
            raise out["error"]
        return out["answer"]

    return send


class RemoteParticipant:
    """A participant answered over HTTP (interface.Participant).

    `key`: its name in the coordinator. `url`: the API's base, below which
    /v1/describe and the rest live (for EMHASS: http://host:5000/participant).
    `window`: the horizon being planned. `token`: the bearer token it expects,
    if any. `transport`: how calls are sent (urllib by default, `timeout`
    seconds each). `trusted_network`: its token may go over plain http:// to
    another machine. `commit_wait_s`: how long a commit's outcome is asked
    for (`timeout` by default).

    Describes itself on construction (RemoteError if it cannot, or speaks
    another version). An answer's `detail` is {"query_id": ...}, which a
    commit names. If a call fails later, the last answer that solved stands
    in, with status "fallback", so the coordinator proves no bound from it.
    """

    modulating = False  # a blend of its plans is not asked for (/v1/blend)

    def __init__(
        self,
        key: str,
        url: str,
        window: PlanWindow,
        token: str | None = None,
        timeout: float = 30.0,
        transport: Transport | None = None,
        trusted_network: bool = False,
        commit_wait_s: float | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"{key}: the url must be http(s)")
        self.key, self.url, self.window = key, url.rstrip("/"), window
        self._token = token
        # how long a commit's outcome is waited for (one call's timeout by default)
        self._commit_wait_s = timeout if commit_wait_s is None else commit_wait_s
        if token and url.startswith("http://") and not _loopback(url) and not trusted_network:
            # Sent in the clear, anyone on the network between could read the
            # token and replay commits: only where the configuration says that
            # network is trusted (as two EMHASS on port 5000 of a home LAN)
            raise PlainHttpRefused(
                f"{key}: its token would be sent over plain http:// to another machine; "
                "use https://, or set trusted_network: true on its solver"
            )
        self._send = transport or urllib_transport(timeout)
        d = self._call("GET", "/v1/describe")
        if d.get("version") != SCHEMA_VERSION:
            raise RemoteError(
                f"{key}: the remote speaks version {d.get('version')!r} of the API, not {SCHEMA_VERSION}"
            )
        kinds = d.get("kinds") or []
        if not isinstance(kinds, list) or "price_response" not in kinds:
            raise RemoteError(f"{key}: the remote does not answer price responses")
        self.kinds = tuple(k for k in kinds if k in KINDS)
        # whether it may be sent the rest of the house's load (a best response)
        self.takes_residual = "best_response" in self.kinds
        power = d.get("max_power_kw")
        if (
            isinstance(power, bool)
            or not isinstance(power, (int, float))
            or not np.isfinite(power)
            or power < 0
        ):
            # it bounds the coordinator's meter: NaN or Infinity would break it
            raise RemoteError(f"{key}: max_power_kw must be a finite number of at least 0")
        self.max_power_kw = float(power)
        self.onoff = bool(d.get("onoff", not d.get("modulating", False)))
        self.devices = tuple(str(x) for x in d.get("devices") or ())
        self.solver = str(d.get("solver", ""))
        # what it serves beyond the first draft (interface.FEATURES); a field or
        # route a feature adds is used only with a participant that lists it
        features = d.get("features")
        self.features = frozenset(
            str(f) for f in (features if isinstance(features, list) else []) if isinstance(f, str)
        )
        # its last commit and how it ended (the participant's own record)
        active = d.get("active_commit")
        self.active_commit = (
            active if isinstance(active, dict) and active.get("state") in COMMIT_STATES else None
        )
        self.solves, self.solve_s = 0, 0.0
        self._good: Answer | None = None
        self._last: tuple[tuple[Any, ...], Answer] | None = None
        self._baseline: Answer | None = None

    def _call(
        self, method: str, path: str, body: Any = None, query_id: str | None = None
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if query_id:
            headers["Query-Id"] = query_id
        data = None if body is None else json.dumps(body).encode()
        try:
            status, raw = self._send(method, self.url + path, headers, data)
        except (
            OSError,
            http.client.HTTPException,
        ) as exc:  # refused, unreachable, timed out, cut short
            raise RemoteError(f"{self.key}: {exc}") from exc
        try:
            out = json.loads(raw or b"{}")
        except ValueError:
            out = {}
        if status not in (200, 202) or not isinstance(out, dict):
            why = out.get("detail") or out.get("title") if isinstance(out, dict) else ""
            raise RemoteError(f"{self.key}: HTTP {status}{': ' + str(why) if why else ''}", status)
        return out

    def _answer(self, path: str, body: dict[str, Any]) -> Answer:
        """POST `body` to `path` and read the answer; a failure answers with
        the last good plan (status "fallback"), or raises RemoteUnavailable."""
        qid = uuid.uuid4().hex
        t0 = time.perf_counter()
        try:
            a = answer_from_dict(self._call("POST", path, body, qid))
            if a.plan_kw.shape != (self.window.slots,) or a.trajectory.shape != (
                self.window.slots + 1,
            ):
                raise RemoteError(f"{self.key}: an answer of the wrong length")
            if (
                a.status not in ("ok", "infeasible", "fallback")
                or not np.all(np.isfinite(a.plan_kw))
                or not np.all(np.isfinite(a.trajectory))
                or not np.isfinite(a.private_cost)
            ):
                # one such answer would break the coordinator's LP for the house
                raise RemoteError(f"{self.key}: an answer that is not a finite plan")
        except (RemoteError, KeyError, ValueError, TypeError) as exc:
            if self._good is None:
                raise RemoteUnavailable(str(exc), getattr(exc, "status", 0)) from exc
            g = self._good
            return Answer(
                g.plan_kw, g.private_cost, g.trajectory, status="fallback", detail=g.detail
            )
        finally:
            self.solves += 1
            self.solve_s += time.perf_counter() - t0
        a.detail = {"query_id": qid}
        if a.status == "ok":
            self._good = a
        return a

    def respond(self, query: Query) -> Answer:
        """Its answer to `query` (interface.Participant). A best response goes
        out as a price response unless the remote takes best responses; a
        query equal to the last one is not sent again."""
        if query.kind == "best_response" and "best_response" not in self.kinds:
            query = Query("price_response", query.price_draw, query.price_supply)
        if query.kind not in self.kinds:
            raise NotImplementedError(f"{self.key} does not answer {query.kind} queries")
        key = (
            (query.kind,)
            + tuple(
                None if a is None else np.asarray(a, dtype=float).tobytes()
                for a in (query.price_draw, query.price_supply, query.residual_kw, query.target_kw)
            )
            + (query.rho,)
        )
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        a = self._answer(
            "/v1/query", query_to_dict(query, self.window.step_minutes, self.window.start)
        )
        self._last = (key, a)
        return a

    def baseline(self) -> Answer:
        """Its plan with no price given (interface.Participant); asked once."""
        if self._baseline is None or self._baseline.status != "ok":
            self._baseline = self._answer("/v1/baseline", self.window.to_dict())
        return self._baseline

    def commit(self, detail: Any, plan_kw: np.ndarray) -> tuple[bool | None, str]:
        """Ask it to run the plan the coordinator chose: `detail`, the chosen
        answer's (it names the query, which is the commit's id), and that
        plan. Returns (accepted, reason): True if it runs the plan, False if
        it refused it (nothing ran), None when that is not known.

        The remote answers at once (202) and runs the plan after; its
        outcome is then asked for (GET /v1/commit/{id}) until it is known or
        the timeout passes. A commit that got no answer at all is asked about
        the same way; one the remote never received (404) is sent again,
        which is safe: a commit it already has is never run twice. Published
        part way (partial), or no outcome in time: None."""
        body: dict[str, Any] = {
            "version": SCHEMA_VERSION,
            "horizon": self.window.to_dict(),
            "plan_kw": [float(x) for x in np.asarray(plan_kw, dtype=float)],
        }
        cid = detail.get("query_id") if isinstance(detail, dict) else None
        if isinstance(cid, str):
            body["query_id"] = cid
        deadline = time.monotonic() + self._commit_wait_s
        try:
            r: dict[str, Any] | None = self._call("POST", "/v1/commit", body)
            why = ""
        except RemoteError as exc:
            if exc.status != 0 or not isinstance(cid, str):
                return False, str(exc)  # it answered: refused (busy, unknown query, ...)
            r, why = None, str(exc)  # no answer: it may have it
        if isinstance(cid, str) and (r is None or r.get("state") == "running"):
            r, why = self._settle(cid, body, deadline, r is not None, why)
        return _verdict(r, why)

    def _settle(
        self, cid: str, body: dict[str, Any], deadline: float, received: bool, why: str
    ) -> tuple[dict[str, Any] | None, str]:
        """Commit `cid`'s final record, asked for until `deadline`: (record,
        "") or (None, why it is not known). `received`: the remote has
        acknowledged it, so it is not sent again."""
        path = f"/v1/commit/{urllib.parse.quote(cid, safe='')}"
        delay = 0.1
        while True:
            try:
                r = self._call("GET", path)
                if r.get("state") != "running":  # ended, or a state it does not know
                    return r, ""
                why = "its commit was still running"
            except RemoteError as exc:
                if exc.status == 404 and not received:
                    # it never had it: send it again, which cannot run it twice
                    received = True
                    try:
                        r = self._call("POST", "/v1/commit", body)
                        if r.get("state") != "running":
                            return r, ""
                    except RemoteError as again:
                        if again.status != 0:
                            return {"state": "refused", "reason": str(again)}, ""
                        why = str(again)
                else:
                    why = str(exc)
            left = deadline - time.monotonic()
            if left <= 0:
                return None, why or "no outcome in time"
            time.sleep(min(delay, left))
            delay = min(2 * delay, 2.0)


def _verdict(r: dict[str, Any] | None, why: str) -> tuple[bool | None, str]:
    """A commit record as (accepted, reason): True, False, or None (not known)."""
    if r is None:
        return None, why
    if r.get("version", SCHEMA_VERSION) != SCHEMA_VERSION:
        return None, f"a commit record of version {r.get('version')!r}, not {SCHEMA_VERSION}"
    state = r.get("state")
    reason = str(r.get("reason", ""))
    if state is None:  # a participant of the API's first draft: accepted only
        return bool(r.get("accepted")), reason
    if state == "accepted":
        return True, reason
    if state == "refused":
        return False, reason
    if state == "partial":
        return None, f"published part way ({reason})" if reason else "published part way"
    return None, why or f"its commit is {state}"


# A commit's states: running while it publishes; then accepted (it runs the
# plan), refused (nothing was published) or partial (publishing failed part
# way: some of the plan may run).
COMMIT_STATES = ("running", "accepted", "refused", "partial")
COMMIT_FINAL = ("accepted", "refused", "partial")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


class CommitLog:
    """Each commit's record, by its id (the query_id it names): what the
    remote did with it, so a commit can be asked about again
    (GET /v1/commit/{id}) and sent again without running twice.

    Kept `keep_s` seconds, at most `max_entries`, in memory and, given
    `path`, in a JSON file that a restart reads back. A commit still running
    when its EMHASS stopped is read back as partial: what it published is
    not known. `terms` and `signatures` are reserved (empty): what a plan is
    worth to each side, and each side's signature on the record."""

    def __init__(
        self,
        path: pathlib.Path | None = None,
        keep_s: float = 48 * 3600.0,
        max_entries: int = 256,
    ) -> None:
        self._path, self.keep_s, self.max_entries = path, keep_s, max_entries
        self._lock = threading.Lock()
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        if path is not None:
            self._load()

    def _load(self) -> None:
        assert self._path is not None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for r in data if isinstance(data, list) else []:
            if not (
                isinstance(r, dict)
                and isinstance(r.get("commit_id"), str)
                and r.get("state") in COMMIT_STATES
                and isinstance(r.get("received_ts"), (int, float))
            ):
                continue
            if r["state"] == "running":
                r = {
                    **r,
                    "state": "partial",
                    "reason": "EMHASS stopped while it was publishing this plan",
                }
            self._records[r["commit_id"]] = r
        self._prune()

    def _prune(self) -> None:  # under the lock, or before any thread sees the log
        cutoff = time.time() - self.keep_s
        for cid in [c for c, r in self._records.items() if r["received_ts"] < cutoff]:
            del self._records[cid]
        while len(self._records) > self.max_entries:
            self._records.popitem(last=False)

    def _save(self) -> None:  # under the lock
        if self._path is None:
            return
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(list(self._records.values())), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass  # kept in memory: a restart would forget it

    def get(self, commit_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._records.get(commit_id)
            return None if r is None else dict(r)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            self._prune()
            return dict(next(reversed(self._records.values()))) if self._records else None

    def begin(self, commit_id: str, plan_kw: list[float]) -> tuple[dict[str, Any], bool]:
        """Its record as running, and True; or the record it already has
        (a commit sent again), and False."""
        with self._lock:
            known = self._records.get(commit_id)
            if known is not None:
                return dict(known), False
            now = _now_iso()
            r = {
                "version": SCHEMA_VERSION,
                "commit_id": commit_id,
                "state": "running",
                "reason": "",
                "plan_kw": plan_kw,
                "received": now,
                "updated": now,
                "received_ts": time.time(),
                "terms": None,
                "signatures": [],
            }
            self._records[commit_id] = r
            self._prune()
            self._save()
            return dict(r), True

    def end(self, commit_id: str, state: str, reason: str) -> None:
        with self._lock:
            r = self._records.get(commit_id)
            if r is None:
                return
            r.update(state=state, reason=reason, updated=_now_iso())
            self._save()


def commit_view(r: dict[str, Any]) -> dict[str, Any]:
    """A commit record as the API sends it: `accepted` is true or false once
    known (null while running, and for a partial commit)."""
    state = r["state"]
    return {
        "version": r.get("version", SCHEMA_VERSION),
        "commit_id": r["commit_id"],
        "state": state,
        "accepted": True if state == "accepted" else False if state == "refused" else None,
        "reason": r.get("reason", ""),
        "received": r.get("received"),
        "updated": r.get("updated"),
        "terms": r.get("terms"),
        "signatures": r.get("signatures", []),
    }


def _outcome(out: Any) -> tuple[str, str]:
    """A commit callable's result as (state, reason): (bool, reason) or
    (state, reason)."""
    first, reason = out
    if isinstance(first, bool):
        return ("accepted" if first else "refused"), str(reason)
    if first in COMMIT_FINAL:
        return str(first), str(reason)
    raise ValueError(f"a commit outcome must be one of {COMMIT_FINAL}, not {first!r}")


# ---------------------------------------------------------------- the solver's side
@dataclass
class _Session:
    participant: Participant
    created: float
    answers: OrderedDict[str, Answer] = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _anchored(d: Any) -> PlanWindow:
    """The Horizon `d`, which must name its first slot: a session is a model
    built for given slots (its state of charge, its forecasts), so one
    without a start could be answered from slots that have since passed."""
    window = PlanWindow.from_dict(d)
    if window.start is None:
        raise ValueError("horizon.start is required: the first slot planned")
    return window


def _problem(status: int, title: str, detail: str = "") -> tuple[int, dict[str, Any]]:
    """An RFC 9457 problem body."""
    return status, {"type": "about:blank", "title": title, "status": status, "detail": detail}


class ParticipantService:
    """The participant API's handlers, for any HTTP host.

    `describe`: returns the Description (key, devices, max_power_kw,
    modulating, onoff, ...); version and kinds are filled in if absent.
    `build`: the participant for a horizon, built when a horizon is first
    asked about. `commit`: runs the chosen answer - given the session's
    participant and its own Answer, with the `detail` it kept - and returns
    (state, reason), state accepted, refused or partial (or (bool, reason));
    None: commits are refused. It runs after the commit is answered (202),
    in a thread of its own. `begin`: what the host takes before a commit
    runs (the live-optimization slot): ("ok", release), ("busy", None) or
    ("unavailable", why); `release()` is called, from that thread, once the
    commit has run. `log`: the commits' records (CommitLog). `token`: the
    bearer token callers must present; None: open (for tests and hosts that
    authenticate themselves). Sessions live `ttl_s`, at most `max_sessions`
    at once.
    """

    def __init__(
        self,
        describe: Callable[[], dict[str, Any]],
        build: Callable[[PlanWindow], Participant],
        commit: Callable[[Participant, Answer], tuple[Any, str]] | None = None,
        token: str | None = None,
        ttl_s: float = 3600.0,
        max_sessions: int = 4,
        begin: Callable[[], tuple[str, Any]] | None = None,
        log: CommitLog | None = None,
    ) -> None:
        self._describe, self._build, self._commit = describe, build, commit
        self._begin = begin
        self.log = log if log is not None else CommitLog()
        self._token = token
        self.ttl_s, self.max_sessions = ttl_s, max_sessions
        self._sessions: OrderedDict[PlanWindow, _Session] = OrderedDict()
        self._lock = threading.Lock()

    def _session(self, window: PlanWindow, create: bool = True) -> _Session | None:
        now = time.monotonic()
        with self._lock:
            for w in [w for w, s in self._sessions.items() if now - s.created > self.ttl_s]:
                del self._sessions[w]
            s = self._sessions.get(window)
            if s is not None or not create:
                return s
        participant = self._build(window)  # outside the lock: it may take a while
        with self._lock:
            s = self._sessions.setdefault(window, _Session(participant, now))
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return s

    @staticmethod
    def _keep(s: _Session, qid: str, a: Answer) -> None:
        s.answers[qid] = a
        while len(s.answers) > 1024:
            s.answers.popitem(last=False)

    def handle(
        self, method: str, path: str, headers: Mapping[str, str], body: bytes | None
    ) -> tuple[int, dict[str, Any]]:
        """One request: `path` is below the API's base ("/v1/query").
        Returns (HTTP status, JSON body)."""
        h = {k.lower(): v for k, v in headers.items()}
        if self._token is not None and not hmac.compare_digest(
            h.get("authorization", "").encode(), f"Bearer {self._token}".encode()
        ):
            return _problem(401, "Unauthorized", "a bearer token is required")
        try:
            if (method, path) == ("GET", "/v1/describe"):
                d = dict(self._describe())
                d.setdefault("version", SCHEMA_VERSION)
                d.setdefault("kinds", ["price_response"])
                d.setdefault("features", list(FEATURES))
                latest = self.log.latest()
                if latest is not None:
                    # its last commit and how it ended, so a coordinator starts
                    # each run from what this participant actually did
                    d["active_commit"] = commit_view(latest)
                return 200, d
            if method == "GET" and path.startswith("/v1/commit/"):
                cid = urllib.parse.unquote(path[len("/v1/commit/") :])
                r = self.log.get(cid)
                if r is None:
                    return _problem(404, "Not Found", "no commit has that id")
                return (202 if r["state"] == "running" else 200), commit_view(r)
            if method != "POST" or path not in ("/v1/baseline", "/v1/query", "/v1/commit"):
                return _problem(404, "Not Found", f"{method} {path}")
            try:
                req = json.loads(body or b"null")
            except ValueError:
                return _problem(400, "Bad Request", "the body is not JSON")
            qid = h.get("query-id") or uuid.uuid4().hex
            if len(qid) > 64:
                return _problem(400, "Bad Request", "Query-Id is at most 64 characters")
            if path == "/v1/baseline":
                window = _anchored(req)
                s = self._session(window)
                assert s is not None
                with s.lock:
                    a = s.participant.baseline()
                    self._keep(s, qid, a)
                return 200, answer_to_dict(a)
            if path == "/v1/query":
                if not isinstance(req, dict) or req.get("version") != SCHEMA_VERSION:
                    return _problem(
                        400, "Bad Request", f"a device-query version {SCHEMA_VERSION} is expected"
                    )
                window = _anchored(req.get("horizon"))
                q = query_from_dict(req)
                if q.kind not in KINDS:
                    return _problem(400, "Bad Request", f"unknown kind {q.kind!r}")
                for name in ("price_draw", "price_supply", "residual_kw", "target_kw"):
                    v = getattr(q, name)
                    if v is not None and (v.shape != (window.slots,) or not np.all(np.isfinite(v))):
                        return _problem(
                            400, "Bad Request", f"{name} must be {window.slots} finite numbers"
                        )
                s = self._session(window)
                assert s is not None
                with s.lock:
                    a = s.participant.respond(q)
                    self._keep(s, qid, a)
                return 200, answer_to_dict(a)
            # /v1/commit
            if not isinstance(req, dict) or req.get("version") != SCHEMA_VERSION:
                return _problem(
                    400, "Bad Request", f"a commit version {SCHEMA_VERSION} is expected"
                )
            window = _anchored(req.get("horizon"))
            cid = str(req.get("query_id") or "")
            plan = np.asarray(req.get("plan_kw"), dtype=float)
            known = self.log.get(cid) if cid else None
            if known is not None:
                # sent again (its answer was lost): what became of it, never run twice
                if plan.shape != (len(known["plan_kw"]),) or not np.allclose(
                    plan, known["plan_kw"], atol=1e-6
                ):
                    return _problem(409, "Conflict", "that commit was made with another plan_kw")
                return (202 if known["state"] == "running" else 200), commit_view(known)
            s = self._session(window, create=False)
            chosen = None if s is None else s.answers.get(cid)
            if s is None or chosen is None:
                return _problem(409, "Conflict", "no answer of this horizon has that query_id")
            if plan.shape != chosen.plan_kw.shape or not np.allclose(
                plan, chosen.plan_kw, atol=1e-6
            ):
                return _problem(
                    409, "Conflict", "plan_kw is not the plan that query was answered with"
                )
            if self._commit is None:
                record, _ = self.log.begin(cid, [float(x) for x in plan])
                self.log.end(cid, "refused", "this participant does not run plans")
                return 200, commit_view(self.log.get(cid) or record)
            taken, release = self._begin() if self._begin is not None else ("ok", None)
            if taken == "busy":
                # nothing was run: a definite refusal, which may be sent again later
                return _problem(429, "Too Many Requests", "a live optimization is running")
            if taken != "ok":
                return _problem(503, "Service Unavailable", f"the commit cannot run ({release})")
            record, new = self.log.begin(cid, [float(x) for x in plan])
            if not new:  # the same commit, sent twice at once: the first runs it
                if release is not None:
                    release()
                return (202 if record["state"] == "running" else 200), commit_view(record)
            participant, commit = s.participant, self._commit

            def run() -> None:
                try:
                    try:
                        state, reason = _outcome(commit(participant, chosen))
                    except Exception as exc:  # noqa: BLE001 - it may have published part of it
                        state, reason = "partial", f"{type(exc).__name__}: {exc}"
                    self.log.end(cid, state, reason)
                finally:
                    if release is not None:
                        release()

            threading.Thread(target=run, name=f"participant-commit-{cid[:8]}", daemon=True).start()
            return 202, commit_view(record)
        except WindowRefused as exc:
            return _problem(409, "Conflict", str(exc))
        except ValueError as exc:
            return _problem(400, "Bad Request", str(exc))
        except NotImplementedError as exc:
            return _problem(422, "Unprocessable Content", str(exc))
        except Exception as exc:  # noqa: BLE001 - the host's model failed: say so, never a bare 500
            return _problem(503, "Service Unavailable", f"{type(exc).__name__}: {exc}")
