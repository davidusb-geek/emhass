"""Loopback ``aiohttp`` test servers for code that talks HTTP(S) through ``aiohttp``.

The code under test makes real requests (real TLS, timeouts and connection failures)
against a local server whose responses the test declares. Certificates come from a
throwaway ``trustme`` CA, so the client verifies them as in production.

    async with HttpTestServer(https=("api.example.org",)) as server:
        server.get("https://api.example.org/data", payload={"ok": True})
        with server.intercept(forecast_module):  # forecast_module.aiohttp -> this server
            ...
        assert [r.url for r in server.requests] == ["https://api.example.org/data"]

Only GET is served. Isolation: only the modules passed to :meth:`HttpTestServer.intercept`
are touched, and only inside that block; the real ``aiohttp`` package is never patched.
Within a patched module only the declared host names are reachable: any other host (an IP
literal too), a request whose scheme differs from the host's declaration or that uses a
proxy, a session whose connector is not the module's own ``aiohttp.TCPConnector`` or
``aiohttp.request`` raises :class:`NetworkBlocked`, and so does leaving the
``async with`` block afterwards. A module can be intercepted by one block at a time. A request
no declared response matches (any non-GET too) is answered by dropping the connection and
fails the block on exit. Code that imports ``ClientSession`` directly is not intercepted.

Test infrastructure only; nothing under ``src/`` imports it.
"""

import asyncio
import contextlib
import ipaddress
import re
import socket
import ssl
import threading
from dataclasses import dataclass
from typing import NamedTuple
from unittest import mock

import aiohttp
import trustme
from aiohttp import web
from aiohttp.abc import AbstractResolver
from aiohttp.test_utils import TestServer

__all__ = ["HttpTestServer", "NetworkBlocked"]


# Modules (by id) with an active ``intercept``: a second one on the same module would restore
# stale state when the two blocks end out of nesting order, so it is refused instead.
_intercepted: set[int] = set()
_intercepted_lock = threading.Lock()


def _normalize(host: str) -> str:
    """DNS names are case-insensitive; aiohttp (yarl) lowercases the host it requests."""
    return host.lower()


class NetworkBlocked(AssertionError):
    """The code under test tried to leave the test server (a failure, not a request error)."""


class Recorded(NamedTuple):
    method: str
    url: str


@dataclass
class _Route:
    pattern: str | re.Pattern
    status: int
    payload: object
    delay: float
    repeat: bool
    used: bool = False

    def matches(self, url: str) -> bool:
        if self.used and not self.repeat:
            return False
        if isinstance(self.pattern, re.Pattern):
            return self.pattern.match(url) is not None
        return url == self.pattern


class HttpTestServer:
    """A loopback server with declared responses and a request log."""

    def __init__(self, *, http=(), https=(), trusted=True):
        """``http``/``https``: host names served in clear / over TLS.

        ``trusted=False`` makes the client reject the server certificate.
        """
        http, https = tuple(map(_normalize, http)), tuple(map(_normalize, https))
        if overlap := set(http) & set(https):
            raise ValueError(f"served both in clear and over TLS: {sorted(overlap)}")
        for host in (*http, *https):
            try:
                ipaddress.ip_address(host)
            except ValueError:
                continue  # a name, as required
            raise ValueError(f"{host}: IP literals skip name resolution and reach the network")
        self._tls = {**dict.fromkeys(http, False), **dict.fromkeys(https, True)}
        self._trusted = trusted
        self._routes: list[_Route] = []
        self._listeners: dict[bool, TestServer] = {}
        self._client_ssl: ssl.SSLContext | None = None
        self.requests: list[Recorded] = []
        self.unmatched: list[str] = []
        self.blocked: list[str] = []

    def get(self, pattern, *, status=200, payload=None, delay=0.0, repeat=False):
        """Declare a response to GET a full URL (``str`` exact, or ``re.Pattern``).

        Each response is used once, in declaration order, unless ``repeat``. ``payload`` is
        sent as JSON; ``delay`` waits before answering (longer than the client timeout, it
        never answers).
        """
        self._routes.append(_Route(pattern, status, payload, delay, repeat))

    async def __aenter__(self):
        server_ssl = None
        if any(self._tls.values()):
            ca = trustme.CA()
            cert = ca.issue_cert(*(host for host, tls in self._tls.items() if tls))
            server_ssl = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            cert.configure_cert(server_ssl)
            self._client_ssl = ssl.create_default_context()
            (ca if self._trusted else trustme.CA()).configure_trust(self._client_ssl)
        try:
            for tls in set(self._tls.values()):
                app = web.Application()
                app.router.add_route("*", "/{tail:.*}", self._handle)
                listener = TestServer(app)
                await listener.start_server(ssl=server_ssl if tls else None)
                self._listeners[tls] = listener
        except BaseException:
            await self._stop()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._stop()
        if exc_type is None:
            if self.blocked:
                raise NetworkBlocked(f"blocked network access: {self.blocked}")
            if self.unmatched:
                raise AssertionError(f"requests no declared response matched: {self.unmatched}")

    async def _stop(self):
        for listener in self._listeners.values():
            await listener.close()
        self._listeners.clear()

    async def _handle(self, request: web.Request) -> web.Response:
        url = f"{request.scheme}://{request.host}{request.path_qs}"
        self.requests.append(Recorded(request.method, url))
        route = next((r for r in self._routes if request.method == "GET" and r.matches(url)), None)
        if route is None:
            self.unmatched.append(f"{request.method} {url}")
            # No answer: aiohttp repeats an idempotent request once after a dropped connection.
            request.transport.close()
            return web.Response()
        route.used = True
        await asyncio.sleep(route.delay)
        if route.payload is None:
            return web.Response(status=route.status)
        return web.json_response(route.payload, status=route.status)

    @contextlib.contextmanager
    def intercept(self, *modules):
        """Point each module's ``aiohttp`` reference at this server until the block ends."""
        server = self

        class Resolver(AbstractResolver):
            async def resolve(self, host, port=0, family=socket.AF_INET):
                listener = server._listeners[server._tls[host]]
                return [
                    {
                        "hostname": host,
                        "host": "127.0.0.1",
                        "port": listener.port,
                        "family": socket.AF_INET,
                        "proto": 0,
                        "flags": 0,
                    }
                ]

            async def close(self):
                pass

        class Connector(aiohttp.TCPConnector):
            def __init__(self, *args, **kwargs):
                kwargs.update(resolver=Resolver(), ssl=server._client_ssl or True)
                super().__init__(*args, **kwargs)

            async def connect(self, req, traces, timeout):
                # Covers what the resolver cannot: IP literals skip name resolution, and the
                # resolver never sees the scheme.
                tls = server._tls.get(req.url.host or "")  # yarl already lowercased it
                if req.proxy is not None or tls is None or tls != req.is_ssl():
                    server.blocked.append(f"{req.method} {req.url}")
                    raise NetworkBlocked(f"{req.url} is not served by the test server")
                return await super().connect(req, traces, timeout)

        def client_session(*args, connector=None, **kwargs):
            if connector is None:
                connector = Connector()
            elif type(connector) is not Connector:  # a subclass could override connect()
                server.blocked.append("session with its own connector")
                raise NetworkBlocked("a session with its own connector bypasses the test server")
            return aiohttp.ClientSession(*args, connector=connector, **kwargs)

        class StandIn:
            ClientSession = staticmethod(client_session)
            TCPConnector = Connector

            @staticmethod
            def request(*args, **kwargs):
                server.blocked.append("aiohttp.request")
                raise NetworkBlocked("aiohttp.request builds its own connector")

            def __getattr__(self, name):
                return getattr(aiohttp, name)

        ids = [id(module) for module in modules]
        with contextlib.ExitStack() as stack:
            with _intercepted_lock:  # check and register together; nothing is patched on refusal
                if len(set(ids)) != len(ids) or _intercepted.intersection(ids):
                    raise RuntimeError("a module can only be intercepted once at a time")
                _intercepted.update(ids)
            stack.callback(_intercepted.difference_update, ids)
            for module in modules:
                stack.enter_context(mock.patch.object(module, "aiohttp", StandIn()))
            yield self
