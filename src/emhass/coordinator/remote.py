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
import inspect
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
from typing import Any, Protocol

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
class Transport(Protocol):
    """How a call is sent: (method, url, headers, body) -> (status, body).
    The default one (urllib_transport) also takes `within`, a shorter time
    for that call; others are called without it."""

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, /
    ) -> tuple[int, bytes]: ...


class _BoundedTransport(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        within: float | None = None,
        /,
    ) -> tuple[int, bytes]: ...


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


# The most a participant may say it draws or supplies (kW): a site, not a grid
MAX_POWER_KW = 1e6
# The largest state or private cost an answer may carry (its own units)
MAX_ANSWER_VALUE = 1e9

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


class _Call:
    """One call's sockets and whether its deadline has passed. A call is cut
    off by shutting its sockets; one still connecting (resolving the host,
    opening the connection) when its deadline passed closes its socket as
    soon as it has one, before it can send anything."""

    def __init__(self) -> None:
        self.socks: list[socket.socket] = []
        self.cancelled = False
        self.lock = threading.Lock()

    def register(self, sock: socket.socket) -> None:
        with self.lock:
            if self.cancelled:
                sock.close()
                raise TimeoutError("the call was past its deadline before it connected")
            self.socks.append(sock)

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True
            socks = list(self.socks)
        for s in socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class _Tracked:
    """http.client connections that register their socket with their call
    (_Call) once connected, and before they send."""

    def __init__(self, call: _Call) -> None:
        self.call = call

    def __call__(self, base: type[http.client.HTTPConnection]) -> Any:
        call = self.call

        class Connection(base):  # type: ignore[valid-type,misc]
            def connect(self) -> None:
                super().connect()
                call.register(self.sock)

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


def urllib_transport(timeout: float) -> _BoundedTransport:
    """The standard library's HTTP client as a Transport: `timeout` seconds
    for the whole call - resolving the host, connecting, the headers and the
    body -, redirects not followed, answers of at most MAX_RESPONSE_BYTES."""

    def call(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        this: _Call,
        limit: float,
    ) -> tuple[int, bytes]:
        track = _Tracked(this)
        # No proxy from the environment (HTTP_PROXY): a proxy would receive the
        # bearer token, even for this machine or a trusted network.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect, _HTTPHandler(track), _HTTPSHandler(track)
        )
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        deadline = time.monotonic() + limit
        try:
            with opener.open(req, timeout=limit) as r:  # http(s) only, checked by the caller
                return int(r.status), _read(r, deadline)
        except urllib.error.HTTPError as exc:
            return int(exc.code), _read(exc, deadline)

    def send(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        within: float | None = None,
    ) -> tuple[int, bytes]:
        # A socket timeout bounds each read, not the call: a remote sending
        # its headers a byte at a time would hold the plan. The call runs in
        # a thread, and past the deadline its socket is shut, which ends it.
        # `within`: a shorter time for this call (what is left of a wait).
        limit = timeout if within is None else max(min(timeout, within), 0.001)
        this = _Call()
        out: dict[str, Any] = {}

        def run() -> None:
            try:
                out["answer"] = call(method, url, headers, body, this, limit)
            except BaseException as exc:  # noqa: BLE001 - raised again below
                out["error"] = exc

        worker = threading.Thread(target=run, name="emhass-remote-call", daemon=True)
        worker.start()
        worker.join(limit)
        if worker.is_alive():
            this.cancel()  # past its deadline: it sends nothing more
            raise TimeoutError(f"no answer within {limit:g} s")
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
        # the default transport also takes a shorter time for one call (`within`)
        self._bounded: _BoundedTransport | None = None
        self._send: Transport
        if transport is None:
            self._bounded = self._send = urllib_transport(timeout)
        else:
            self._send = transport
        d = self._call("GET", "/v1/describe")
        if d.get("version") != SCHEMA_VERSION:
            raise RemoteError(
                f"{key}: the remote speaks version {d.get('version')!r} of the API, not {SCHEMA_VERSION}"
            )
        try:
            kinds = d.get("kinds") or []
            if not isinstance(kinds, list) or "price_response" not in kinds:
                raise RemoteError(f"{key}: the remote does not answer price responses")
            self.kinds = tuple(k for k in kinds if k in KINDS)
            # whether it may be sent the rest of the house's load (a best response)
            self.takes_residual = "best_response" in self.kinds
            power = d.get("max_power_kw")
            try:
                if isinstance(power, bool) or not isinstance(power, (int, float)):
                    raise TypeError(type(power).__name__)
                power = float(power)  # an integer too large for a float: OverflowError
            except (TypeError, OverflowError) as exc:
                raise RemoteError(f"{key}: max_power_kw must be a number ({exc})") from exc
            if not 0.0 <= power <= MAX_POWER_KW:  # NaN fails too
                # it bounds the coordinator's meter: NaN, Infinity or an absurd
                # number would break it
                raise RemoteError(
                    f"{key}: max_power_kw must be a number from 0 to {MAX_POWER_KW:g}"
                )
            self.max_power_kw = power
            self.onoff = bool(d.get("onoff", not d.get("modulating", False)))
            self.devices = tuple(str(x) for x in d.get("devices") or ())
            self.solver = str(d.get("solver", ""))
            # what it serves beyond the first draft (interface.FEATURES); a field or
            # route a feature adds is used only with a participant that lists it
            features = d.get("features")
            self.features = frozenset(
                str(f)
                for f in (features if isinstance(features, list) else [])
                if isinstance(f, str)
            )
            # its last commit and how it ended (the participant's own record)
            active = d.get("active_commit")
            self.active_commit = (
                active
                if isinstance(active, dict) and active.get("state") in COMMIT_STATES
                else None
            )
        except RemoteError:
            raise
        except Exception as exc:  # noqa: BLE001 - any description it cannot read
            raise RemoteError(f"{key}: a description it cannot read ({exc})") from exc
        self.solves, self.solve_s = 0, 0.0
        self._good: Answer | None = None
        self._last: tuple[tuple[Any, ...], Answer] | None = None
        self._baseline: Answer | None = None

    def _call(
        self,
        method: str,
        path: str,
        body: Any = None,
        query_id: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """`deadline` (time.monotonic()): the call is not started past it, and
        with the default transport it is cut off there."""
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if query_id:
            headers["Query-Id"] = query_id
        data = None if body is None else json.dumps(body).encode()
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            raise RemoteError(f"{self.key}: no time left to ask")
        unsendable = None
        try:
            if left is not None and self._bounded is not None:
                status, raw = self._bounded(method, self.url + path, headers, data, left)
            else:
                status, raw = self._send(method, self.url + path, headers, data)
        except (
            OSError,
            http.client.HTTPException,
        ) as exc:  # refused, unreachable, timed out, cut short
            raise RemoteError(f"{self.key}: {exc}") from exc
        except ValueError as exc:
            # A header that cannot be sent (a line break in the token, say): the error's
            # text quotes the header, token included, so only the error's type is kept, and
            # the error itself is not kept, as cause or context.
            unsendable = type(exc).__name__
        if unsendable is not None:
            raise RemoteError(f"{self.key}: the request could not be sent ({unsendable})")
        try:
            out = json.loads(raw or b"{}")
        except (ValueError, RecursionError):  # not JSON, or nested past reason
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
            if (
                np.max(np.abs(a.plan_kw), initial=0.0) > self.max_power_kw * (1 + 1e-6) + 1e-9
                or np.max(np.abs(a.trajectory), initial=0.0) > MAX_ANSWER_VALUE
                or abs(a.private_cost) > MAX_ANSWER_VALUE
            ):
                # past what it said it can draw or supply, or absurd
                raise RemoteError(f"{self.key}: an answer past its max_power_kw, or absurd")
        except (RemoteError, KeyError, ValueError, TypeError, ArithmeticError) as exc:
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
        `commit_wait_s` has passed since that answer (or the lack of one),
        no call running past it. A commit that got no answer at all is asked about
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
        try:
            r: dict[str, Any] | None = self._call("POST", "/v1/commit", body)
            why = ""
            if not _a_record(r):  # a 2xx that says nothing usable: it may run
                r, why = None, "an answer to the commit that is not a commit record"
        except RemoteError as exc:
            if exc.status in REFUSED_BEFORE_RUNNING:
                return False, str(exc)  # it answered that nothing ran
            # no answer, a gateway's error (502, 504), its own failure (5xx):
            # it may have taken the commit
            r, why = None, str(exc)
        except Exception as exc:  # noqa: BLE001 - whatever failed, it may have been sent
            r, why = None, f"{type(exc).__name__}: {exc}"
        asks = "commit_status" in self.features
        if isinstance(cid, str) and asks and (r is None or r.get("state") == "running"):
            # the wait for its outcome, from the commit's answer: at most
            # one call's timeout and then commit_wait_s in all
            deadline = time.monotonic() + self._commit_wait_s
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
            if time.monotonic() >= deadline:  # no time for another call
                return None, why or "no outcome in time"
            try:
                r = self._call("GET", path, deadline=deadline)
                if not _a_record(r):
                    why = "an answer about the commit that is not a commit record"
                elif r.get("state") != "running":  # ended, or a state it does not know
                    return r, ""
                else:
                    why = "its commit was still running"
            except RemoteError as exc:
                if exc.status == 404 and not received:
                    # it has no record of it: send it again, which cannot run it
                    # twice. A refusal of that (busy, 409) says nothing of the
                    # first, which may still be on its way: asked about again
                    received = True
                    if time.monotonic() >= deadline:
                        return None, "no outcome in time"
                    try:
                        r = self._call("POST", "/v1/commit", body, deadline=deadline)
                        if _a_record(r) and r.get("state") != "running":
                            return r, ""
                    except RemoteError as again:
                        why = str(again)
                else:
                    why = str(exc)
            left = deadline - time.monotonic()
            if left <= 0:
                return None, why or "no outcome in time"
            time.sleep(min(delay, left))
            delay = min(2 * delay, 2.0)


# HTTP statuses with which a participant says a commit did not run (its body
# was refused, it is busy, it has no such answer...). Any other - a gateway's
# 502 or 504, a 500 - leaves it not known whether the commit was taken.
REFUSED_BEFORE_RUNNING = frozenset({400, 401, 403, 404, 405, 408, 409, 413, 415, 422, 429})


def _a_record(r: Any) -> bool:
    """Whether `r` is a commit record: a state, or (first draft) accepted."""
    return isinstance(r, dict) and (
        isinstance(r.get("state"), str) or isinstance(r.get("accepted"), bool)
    )


def _verdict(r: dict[str, Any] | None, why: str) -> tuple[bool | None, str]:
    """A commit record as (accepted, reason): True, False, or None (not known)."""
    if r is None:
        return None, why
    if r.get("version", SCHEMA_VERSION) != SCHEMA_VERSION:
        return None, f"a commit record of version {r.get('version')!r}, not {SCHEMA_VERSION}"
    state = r.get("state")
    reason = str(r.get("reason", ""))
    if state is None:  # a participant of the API's first draft: accepted only
        accepted = r.get("accepted")
        return (accepted, reason) if isinstance(accepted, bool) else (None, why)
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


class CommitNotRecorded(OSError):
    """A commit's record could not be saved, so it is not run."""


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
        changed = False
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
                    # what it published, if anything, is for its host to tell
                    "interrupted": True,
                }
                changed = True
            self._records[r["commit_id"]] = r
        self._prune()
        if changed:
            self._save()  # read back the same after another stop

    def _prune(self) -> None:  # under the lock, or before any thread sees the log
        # Never the latest commit that ran (accepted or partial), nor one still
        # publishing: its host may run its plan for as long as Home Assistant
        # keeps it, however long ago it was made. Nor, while the latest was
        # cut off by a stop (its host may find nothing of it ran), the latest
        # that ran whole before it.
        def latest(interrupted_too: bool) -> str | None:
            for c, r in reversed(self._records.items()):
                if r["state"] in ("accepted", "partial") and (
                    interrupted_too or not r.get("interrupted")
                ):
                    return c
            return None

        keep = {latest(True), latest(False)} | {
            c for c, r in self._records.items() if r["state"] == "running"
        }
        cutoff = time.time() - self.keep_s
        for cid in [c for c, r in self._records.items() if r["received_ts"] < cutoff]:
            if cid not in keep:
                del self._records[cid]
        extra = len(self._records) - self.max_entries
        for cid in [c for c in self._records if c not in keep][: max(extra, 0)]:
            del self._records[cid]

    def _save(self) -> bool:  # under the lock
        """Whether the records are on disk (always, with no file)."""
        if self._path is None:
            return True
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(list(self._records.values())), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            return False  # kept in memory: a restart would forget the change
        return True

    def get(self, commit_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._records.get(commit_id)
            return None if r is None else dict(r)

    def latest(self, running: bool = False) -> dict[str, Any] | None:
        """The latest record; `running`: the latest of a plan it may run
        (accepted, published part way, or still publishing) - a refused
        commit leaves the plan before it in place."""
        with self._lock:
            self._prune()
            for r in reversed(self._records.values()):
                if not running or r["state"] in ("accepted", "partial", "running"):
                    return dict(r)
            return None

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            self._prune()
            return [dict(r) for r in self._records.values()]

    def annotate(self, commit_id: str, fields: dict[str, Any]) -> bool | None:
        """Add its host's own fields to a record (never the core ones).
        Whether the record has them on disk; None for no such record."""
        core = ("commit_id", "state", "reason", "plan_kw", "horizon", "version")
        with self._lock:
            r = self._records.get(commit_id)
            if r is None:
                return None
            before = dict(r)
            r.update({k: v for k, v in fields.items() if k not in core})
            if self._save():
                return True
            # not on disk: not in memory either, so a later save never writes it
            r.clear()
            r.update(before)
            return False

    def begin(
        self, commit_id: str, plan_kw: list[float], horizon: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], bool]:
        """Its record as running, and True; or the record it already has
        (a commit sent again), and False. CommitNotRecorded, and no record,
        when the record cannot be saved: a commit that ran would be lost to
        a restart, so it must not run."""
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
                "horizon": horizon,
                "received": now,
                "updated": now,
                "received_ts": time.time(),
                "terms": None,
                "signatures": [],
            }
            self._records[commit_id] = r
            self._prune()
            if not self._save():
                del self._records[commit_id]
                raise CommitNotRecorded(f"its record could not be saved in {self._path}")
            return dict(r), True

    def drop(self, commit_id: str) -> None:
        """Forget a commit that never ran (its host could not start it)."""
        with self._lock:
            if self._records.pop(commit_id, None) is not None:
                self._save()

    def end(
        self, commit_id: str, state: str, reason: str, extra: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            r = self._records.get(commit_id)
            if r is None:
                return
            # its host's own fields (what it saved the plan as), never the core ones
            core = ("commit_id", "state", "reason", "plan_kw", "horizon", "version")
            r.update({k: v for k, v in (extra or {}).items() if k not in core})
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
        # the plan it runs, from its horizon's first slot: a coordinator that
        # does not commit to it again still counts what of it is left
        "horizon": r.get("horizon"),
        "plan_kw": r.get("plan_kw"),
        "received": r.get("received"),
        "updated": r.get("updated"),
        "terms": r.get("terms"),
        "signatures": r.get("signatures", []),
        # the plan ended with its final slots not published: the slot whose
        # value Home Assistant keeps (absent: the plan's last)
        **({"held_slot": r["held_slot"]} if isinstance(r.get("held_slot"), int) else {}),
    }


def _takes_commit_id(commit: Callable[..., Any]) -> bool:
    """Whether a host's commit callable takes the commit's id (a parameter
    named commit_id, third)."""
    try:
        return "commit_id" in inspect.signature(commit).parameters
    except (TypeError, ValueError):
        return False


def _outcome(out: Any) -> tuple[str, str, dict[str, Any]]:
    """A commit callable's result as (state, reason, fields for its record):
    (bool, reason), (state, reason) or (state, reason, fields)."""
    first, reason, *rest = out
    extra = rest[0] if rest and isinstance(rest[0], dict) else {}
    if isinstance(first, bool):
        return ("accepted" if first else "refused"), str(reason), extra
    if first in COMMIT_FINAL:
        return str(first), str(reason), extra
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


def _same_horizon(stored: Any, window: PlanWindow) -> bool:
    """Whether a commit's recorded horizon is `window` (the same slots, the
    same first instant however written). A record with none: not known, so
    not refused."""
    if stored is None:
        return True
    try:
        known = PlanWindow.from_dict(stored)
    except ValueError:
        return False
    if (known.step_minutes, known.slots) != (window.step_minutes, window.slots):
        return False
    if known.start is None or window.start is None:
        return known.start == window.start
    try:
        import pandas as pd

        a, b = pd.Timestamp(known.start), pd.Timestamp(window.start)
        a = a if a.tzinfo else a.tz_localize("UTC")
        b = b if b.tzinfo else b.tz_localize("UTC")
        return bool(a == b)
    except (ValueError, TypeError):
        return known.start == window.start


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
    commit has run. `log`: the commits' records (CommitLog). `current`:
    raises WindowRefused for a horizon whose first slot is no longer this
    participant's, checked on every baseline, query and new commit (a
    session built for it is not answered from once its slot has passed).
    `active`: the record of the commit whose plan the host runs now, or None
    (by default: the latest accepted, partial or running). `token`: the
    bearer token callers must present; None: open (for tests and hosts that
    authenticate themselves). Sessions live `ttl_s`, at most `max_sessions`
    at once.
    """

    def __init__(
        self,
        describe: Callable[[], dict[str, Any]],
        build: Callable[[PlanWindow], Participant],
        # (participant, answer) or (participant, answer, commit_id) -> its outcome
        commit: Callable[..., Any] | None = None,
        token: str | None = None,
        ttl_s: float = 3600.0,
        max_sessions: int = 4,
        begin: Callable[[], tuple[str, Any]] | None = None,
        log: CommitLog | None = None,
        current: Callable[[PlanWindow], None] | None = None,
        active: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self._describe, self._build, self._commit = describe, build, commit
        self._begin = begin
        self._current = current
        self._active = active
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

    def _fresh(self, window: PlanWindow) -> PlanWindow:
        if self._current is not None:
            self._current(window)
        return window

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
                latest = self._active() if self._active is not None else self.log.latest(True)
                if latest is not None:
                    # the commit whose plan it runs, so a coordinator starts each
                    # run from what this participant actually does
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
                window = self._fresh(_anchored(req))
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
                window = self._fresh(_anchored(req.get("horizon")))
                if req.get("kind") not in KINDS:
                    return _problem(400, "Bad Request", f"unknown kind {req.get('kind')!r}")
                rho = req.get("rho", 0.0)
                if isinstance(rho, bool) or not isinstance(rho, (int, float)):
                    return _problem(400, "Bad Request", "rho must be a number")
                for name in ("price_draw", "price_supply", "residual_kw", "target_kw"):
                    v = req.get(name)
                    # JSON numbers only: "1", true or null inside would be read as numbers
                    if v is not None and not (
                        isinstance(v, list)
                        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
                    ):
                        return _problem(400, "Bad Request", f"{name} must be a list of numbers")
                q = query_from_dict(req)
                if q.kind in ("price_response", "best_response") and q.price_draw is None:
                    return _problem(400, "Bad Request", f"a {q.kind} needs price_draw")
                if not np.isfinite(q.rho) or q.rho < 0:
                    return _problem(400, "Bad Request", "rho must be a finite number of at least 0")
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
                if not _same_horizon(known.get("horizon"), window):
                    return _problem(409, "Conflict", "that commit was made for another horizon")
                return (202 if known["state"] == "running" else 200), commit_view(known)
            self._fresh(window)  # a new commit of a horizon whose slot has passed: 409
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
                try:
                    record, _ = self.log.begin(cid, [float(x) for x in plan], window.to_dict())
                except CommitNotRecorded as exc:
                    return _problem(409, "Conflict", f"the commit could not start ({exc})")
                self.log.end(cid, "refused", "this participant does not run plans")
                return 200, commit_view(self.log.get(cid) or record)
            # Recorded before the slot is taken: from here on it is asked about
            # as running, never as unknown (404) while it may yet run.
            try:
                record, new = self.log.begin(cid, [float(x) for x in plan], window.to_dict())
            except CommitNotRecorded as exc:  # never run: it would be lost to a restart
                return _problem(409, "Conflict", f"the commit could not start ({exc})")
            if not new and not _same_horizon(record.get("horizon"), window):
                return _problem(409, "Conflict", "that commit was made for another horizon")
            if not new:  # the same commit, sent twice at once: the first runs it
                return (202 if record["state"] == "running" else 200), commit_view(record)
            try:
                taken, release = self._begin() if self._begin is not None else ("ok", None)
            except Exception as exc:  # noqa: BLE001 - it never ran
                taken, release = "unavailable", f"{type(exc).__name__}: {exc}"
            if taken != "ok":
                # Nothing ran, and it says so (409/429): a coordinator reads
                # 503 as "it may have taken it", which it did not.
                self.log.drop(cid)
                if taken == "busy":
                    return _problem(429, "Too Many Requests", "a live optimization is running")
                return _problem(409, "Conflict", f"the commit could not start ({release})")
            try:
                # again, now it holds the slot: waiting for it may have taken
                # this horizon's first slot past
                self._fresh(window)
            except Exception:  # never left running, the slot held: nothing ran
                self.log.drop(cid)
                if release is not None:
                    release()
                raise
            participant, commit = s.participant, self._commit
            takes_id = _takes_commit_id(commit)

            def run() -> None:
                try:
                    try:
                        out = (
                            commit(participant, chosen, cid)
                            if takes_id
                            else commit(participant, chosen)
                        )
                        state, reason, extra = _outcome(out)
                    except Exception as exc:  # noqa: BLE001 - it may have published part of it
                        state, reason, extra = "partial", f"{type(exc).__name__}: {exc}", {}
                    self.log.end(cid, state, reason, extra)
                finally:
                    if release is not None:
                        release()

            # Not a daemon: a worker that is stopped (a recycled web worker)
            # finishes publishing before the process ends, rather than leave
            # Home Assistant part way through the plan.
            worker = threading.Thread(target=run, name=f"participant-commit-{cid[:8]}")
            try:
                worker.start()
            except Exception as exc:  # noqa: BLE001 - it never ran
                self.log.drop(cid)
                if release is not None:
                    release()
                return _problem(409, "Conflict", f"the commit could not start ({exc})")
            return 202, commit_view(record)
        except WindowRefused as exc:
            return _problem(409, "Conflict", str(exc))
        except ValueError as exc:
            return _problem(400, "Bad Request", str(exc))
        except NotImplementedError as exc:
            return _problem(422, "Unprocessable Content", str(exc))
        except Exception as exc:  # noqa: BLE001 - the host's model failed: say so, never a bare 500
            return _problem(503, "Service Unavailable", f"{type(exc).__name__}: {exc}")
