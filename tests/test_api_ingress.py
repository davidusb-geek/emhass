"""Home Assistant ingress: a listener of its own, trusted only for the Supervisor's address.

These tests drive the ASGI scope the server fills from the accepted socket (``server`` and ``client``).
They do not run Home Assistant: see docs/api_authentication.md for what is and is not verified against
a real Supervisor.
"""

import os
import pathlib
import runpy
import unittest

from emhass import api_auth, web_server

from .api_auth_support import BEARER, OUTSIDER, SUPERVISOR, TOKEN, AuthSupportBase

CONF = pathlib.Path(web_server.__file__).parents[2] / "gunicorn.conf.py"


class TestIngressTrust(AuthSupportBase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await self.enable()

    async def test_the_supervisor_on_the_ingress_port_is_signed_in(self):
        for method, path, kwargs in [
            ("GET", "/api/v1/last-run", {}),
            ("GET", "/configuration", {}),
            ("POST", "/action/publish-data", {"json": {}}),
            ("POST", "/set-config", {"json": {}}),
        ]:
            with self.subTest(route=f"{method} {path}"):
                response = await self.send(method, path, port=8099, peer=SUPERVISOR, **kwargs)
                self.assertNotIn(response.status_code, (401, 403, 503))

    async def test_a_write_from_ingress_must_be_json(self):
        # what a forged cross-site form can send; the UI always sends JSON
        refused = await self.send(
            "POST",
            "/action/publish-data",
            port=8099,
            peer=SUPERVISOR,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=b"a=1",
        )
        self.assertEqual(refused.status_code, 403)
        self.mock_action.assert_not_awaited()

    async def test_anyone_else_on_the_ingress_port_is_refused_even_with_the_token(self):
        for peer in (OUTSIDER, "127.0.0.1", "172.30.32.1", "172.30.32.3", "10.0.0.1"):
            for headers in (None, BEARER):
                with self.subTest(peer=peer, bearer=bool(headers)):
                    response = await self.send(
                        "POST",
                        "/action/publish-data",
                        port=8099,
                        peer=peer,
                        headers=headers,
                        json={},
                    )
                    self.assertEqual(response.status_code, 403)
        self.mock_action.assert_not_awaited()

    async def test_the_supervisors_address_on_the_api_port_is_not_trusted(self):
        for method, path in [
            ("POST", "/action/publish-data"),
            ("GET", "/configuration"),
            ("GET", "/api/v1/plan"),
        ]:
            with self.subTest(route=f"{method} {path}"):
                response = await self.send(
                    method, path, port=5000, peer=SUPERVISOR, json={} if method == "POST" else None
                )
                self.assertIn(response.status_code, (401, 303))
        self.mock_action.assert_not_awaited()

    async def test_headers_never_make_a_request_trusted(self):
        forged = {
            "X-Ingress-Path": "/api/hassio_ingress/abc",
            "X-Forwarded-For": SUPERVISOR,
            "X-Real-IP": SUPERVISOR,
            "Forwarded": f"for={SUPERVISOR}",
            "X-Remote-User-Id": "1",
            "X-Remote-User-Name": "admin",
            "X-Hass-Source": "core.ingress",
            "Host": "supervisor",
        }
        response = await self.send("POST", "/action/publish-data", headers=forged, json={})
        self.assertEqual(response.status_code, 401)
        # and on the ingress port the same headers from a stranger do not help either
        response = await self.send(
            "POST", "/action/publish-data", port=8099, peer=OUTSIDER, headers=forged, json={}
        )
        self.assertEqual(response.status_code, 403)
        self.mock_action.assert_not_awaited()

    async def test_a_forged_scope_value_of_the_wrong_shape_is_not_trusted(self):
        for scope in (
            {"server": None, "client": None},
            {"server": ("x",), "client": ()},
            {"client": (SUPERVISOR, 1)},
        ):
            with self.subTest(scope=scope):
                response = await self.client.open(
                    "/action/publish-data", method="POST", json={}, scope_base=scope
                )
                self.assertEqual(response.status_code, 401)

    async def test_the_health_check_stays_open_on_both_ports(self):
        for port, peer in ((5000, OUTSIDER), (8099, OUTSIDER), (8099, SUPERVISOR)):
            response = await self.send("GET", "/healthz", port=port, peer=peer)
            self.assertNotIn(response.status_code, (401, 403))

    async def test_the_peer_list_replaces_the_default_and_bad_entries_only_narrow(self):
        os.environ[api_auth.INGRESS_PEERS_ENV] = (
            "198.51.100.0/24, not-an-address, ::ffff:172.30.32.2"
        )
        for peer, expected in (
            ("198.51.100.7", True),
            (SUPERVISOR, False),  # the list replaces the default
            ("198.51.101.7", False),
            ("::ffff:198.51.100.9", True),
        ):
            with self.subTest(peer=peer):
                response = await self.send(
                    "POST", "/action/publish-data", port=8099, peer=peer, json={}
                )
                self.assertEqual(response.status_code == 200, expected)

    async def test_only_the_supervisor_by_default(self):
        self.assertEqual(api_auth.DEFAULT_INGRESS_PEERS, "172.30.32.2")


class TestIngressConfiguration(AuthSupportBase):
    def test_no_ingress_port_unless_configured_or_running_as_an_add_on(self):
        self.assertIsNone(api_auth.ingress_port())
        os.environ["SUPERVISOR_TOKEN"] = "x"
        self.assertEqual(api_auth.ingress_port(), 8099)
        os.environ[api_auth.INGRESS_PORT_ENV] = "9100"
        self.assertEqual(api_auth.ingress_port(), 9100)
        for bad in ("abc", "0", "-1", "70000", "80.5"):
            os.environ[api_auth.INGRESS_PORT_ENV] = bad
            self.assertIsNone(api_auth.ingress_port(), bad)

    async def test_without_an_ingress_port_the_supervisor_is_nobody_special(self):
        await self.enable()
        for port in (5000, 8099):
            response = await self.send(
                "POST", "/action/publish-data", port=port, peer=SUPERVISOR, json={}
            )
            self.assertEqual(response.status_code, 401)

    def test_gunicorn_listens_where_the_app_expects_ingress(self):
        """gunicorn.conf.py and api_auth.ingress_port() must make the same decision."""
        cases = [
            ({}, None),
            ({"SUPERVISOR_TOKEN": "x"}, 8099),
            ({"SUPERVISOR_TOKEN": "x", "EMHASS_INGRESS_PORT": "9100"}, 9100),
            ({"EMHASS_INGRESS_PORT": "9100"}, 9100),
        ]
        for env, port in cases:
            with self.subTest(env=env):
                for name in ("SUPERVISOR_TOKEN", "EMHASS_INGRESS_PORT", "PORT", "IP"):
                    os.environ.pop(name, None)
                os.environ.update(env)
                self.assertEqual(api_auth.ingress_port(), port)
                bind = runpy.run_path(str(CONF))["bind"]
                self.assertEqual(bind[0], "0.0.0.0:5000")
                self.assertEqual(bind[1:], [f"0.0.0.0:{port}"] if port else [])

    def test_the_ingress_port_is_never_the_api_port(self):
        os.environ["PORT"] = "5000"
        os.environ[api_auth.INGRESS_PORT_ENV] = "5000"
        self.assertIsNone(api_auth.ingress_port())
        os.environ["SUPERVISOR_TOKEN"] = "x"
        os.environ.pop(api_auth.INGRESS_PORT_ENV)
        os.environ["PORT"] = "8099"  # the default ingress port taken by the API
        self.assertIsNone(api_auth.ingress_port())
        for name in ("PORT", "IP"):
            os.environ.pop(name, None)
        os.environ[api_auth.INGRESS_PORT_ENV] = "5000"
        self.assertEqual(runpy.run_path(str(CONF))["bind"], ["0.0.0.0:5000"])

    def test_gunicorn_refuses_to_rewrite_the_peer_address(self):
        conf = runpy.run_path(str(CONF))
        allowed = [i.strip() for i in conf["forwarded_allow_ips"].split(",") if i.strip()]
        self.assertEqual(allowed, [])


class TestProxyHeadersCannotForgeTheSupervisor(AuthSupportBase):
    """uvicorn's proxy-header handling would let a loopback process choose its peer address."""

    async def _post(self, trusted_hosts):
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await self.enable()
        wrapped = ProxyHeadersMiddleware(web_server.app, trusted_hosts=trusted_hosts)
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/action/publish-data",
            "raw_path": b"/action/publish-data",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"emhass"),
                (b"x-forwarded-for", SUPERVISOR.encode()),
                (b"content-type", b"application/json"),
                (b"content-length", b"2"),
            ],
            "client": ("127.0.0.1", 5555),
            "server": ("0.0.0.0", 8099),
        }
        sent = []
        messages = [{"type": "http.request", "body": b"{}", "more_body": False}]

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        try:
            await wrapped(scope, receive, send)
        except Exception:  # noqa: BLE001 - a request that passes the gate may fail later on mocks
            pass
        return next((m["status"] for m in sent if m["type"] == "http.response.start"), None)

    async def test_the_shipped_setting_leaves_the_peer_alone(self):
        conf = runpy.run_path(str(CONF))
        allowed = [i.strip() for i in conf["forwarded_allow_ips"].split(",") if i.strip()]
        self.assertEqual(await self._post(allowed), 403)
        self.mock_action.assert_not_awaited()

    async def test_control_the_default_setting_would_have_let_loopback_pose_as_the_supervisor(self):
        status = await self._post("127.0.0.1")
        self.assertNotIn(status, (401, 403, 503))


class TestEveryRouteIsGated(AuthSupportBase):
    """No route may be reachable without credentials unless it is one of the three open endpoints."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await self.enable()

    def routes(self):
        adapter = web_server.app.url_map.bind("localhost")
        for rule in web_server.app.url_map.iter_rules():
            values = dict.fromkeys(rule.arguments, "x")
            for method in sorted(rule.methods):
                yield rule.endpoint, method, adapter.build(rule.endpoint, values, method=method)

    def test_the_open_endpoints_are_exactly_these(self):
        self.assertEqual(
            set(api_auth.OPEN_ENDPOINTS), {"healthz", "api_auth_login", "api_auth_logout"}
        )

    async def test_nothing_else_answers_without_credentials(self):
        seen = 0
        for endpoint, method, path in self.routes():
            if endpoint in api_auth.OPEN_ENDPOINTS:
                continue
            seen += 1
            with self.subTest(route=f"{method} {path}"):
                response = await self.send(method, path, json={} if method == "POST" else None)
                self.assertIn(response.status_code, (401, 303), f"{method} {path} is not gated")
        self.assertGreater(seen, 12)  # the enumeration really ran
        self.mock_action.assert_not_awaited()

    async def test_nothing_else_answers_a_stranger_on_the_ingress_port(self):
        for endpoint, method, path in self.routes():
            if endpoint in api_auth.OPEN_ENDPOINTS:
                continue
            with self.subTest(route=f"{method} {path}"):
                response = await self.send(
                    method,
                    path,
                    port=8099,
                    peer=OUTSIDER,
                    headers=BEARER,
                    json={} if method == "POST" else None,
                )
                self.assertEqual(
                    response.status_code,
                    403,
                    f"{method} {path} answers a stranger on the ingress port",
                )


class TestIngressWebSocket(AuthSupportBase):
    async def handshake(self, port, peer, headers=None):
        from quart.testing import WebsocketResponseError

        try:
            async with self.client.websocket(
                "/api/v1/last-run",
                headers=headers,
                scope_base={"server": ("0.0.0.0", port), "client": (peer, 40000)},
            ) as ws:
                await ws.send("x")
                await ws.receive()
        except WebsocketResponseError as e:
            return e.response.status_code
        return 101

    async def asyncSetUp(self):
        await super().asyncSetUp()
        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await self.enable()

    async def test_a_stranger_on_the_ingress_port_is_refused_even_with_the_token(self):
        self.assertEqual(await self.handshake(8099, OUTSIDER, BEARER), 403)

    async def test_the_api_port_needs_the_bearer(self):
        self.assertEqual(await self.handshake(5000, OUTSIDER), 401)
        self.assertNotIn(await self.handshake(5000, OUTSIDER, BEARER), (401, 403, 503))
        self.assertEqual(await self.handshake(5000, SUPERVISOR), 401)

    async def test_the_supervisor_on_the_ingress_port_is_let_in(self):
        self.assertNotIn(await self.handshake(8099, SUPERVISOR), (401, 403, 503))


class TestPortsAreComparedAsNumbers(AuthSupportBase):
    def test_a_zero_padded_api_port_is_still_the_api_port(self):
        os.environ["PORT"] = "05000"
        os.environ[api_auth.INGRESS_PORT_ENV] = "5000"
        self.assertIsNone(api_auth.ingress_port())
        self.assertEqual(runpy.run_path(str(CONF))["bind"], ["0.0.0.0:05000"])
        os.environ["PORT"] = "not-a-number"
        self.assertIsNone(api_auth.ingress_port())  # an unusable PORT is read as the default, 5000


class TestServerConfigurationIsPinned(AuthSupportBase):
    def test_the_direct_server_does_not_rewrite_the_peer_address(self):
        import ast

        source = pathlib.Path(web_server.__file__).read_text(encoding="utf-8")
        configs = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Config"
            and getattr(node.func.value, "id", "") == "uvicorn"
        ]
        self.assertTrue(configs, "main() no longer builds a uvicorn.Config")
        for call in configs:
            keywords = {k.arg: k.value for k in call.keywords}
            self.assertIn("proxy_headers", keywords)
            self.assertIs(getattr(keywords["proxy_headers"], "value", None), False)

    def test_gunicorn_and_the_uvicorn_worker_end_up_trusting_no_forwarder(self):
        pytest = __import__("pytest")
        gunicorn_config = pytest.importorskip(
            "gunicorn.config"
        )  # gunicorn cannot be imported on Windows
        from uvicorn import Config
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

        conf = runpy.run_path(str(CONF))
        cfg = gunicorn_config.Config()
        cfg.set("forwarded_allow_ips", conf["forwarded_allow_ips"])
        self.assertEqual(cfg.forwarded_allow_ips, [])
        # uvicorn.workers.UvicornWorker passes exactly this to uvicorn.Config
        loaded = Config(web_server.app, forwarded_allow_ips=cfg.forwarded_allow_ips)
        loaded.load()
        self.assertIsInstance(loaded.loaded_app, ProxyHeadersMiddleware)
        self.assertEqual(loaded.loaded_app.trusted_hosts, set())
        self.assertFalse(loaded.loaded_app.always_trust)


class TestIngressRelativeUrls(AuthSupportBase):
    """Behind ingress the app is served under /api/hassio_ingress/<token>/, which it does not see:
    every URL it emits must be relative, or it leaves the app and lands on Home Assistant."""

    ABSOLUTE = None

    def test_no_ui_asset_refers_to_an_absolute_app_path(self):
        import re

        root = pathlib.Path(web_server.__file__).parent
        pattern = re.compile(
            r"""(?:fetch\(|href\s*=|src\s*=|action\s*=|location(?:\.href)?\s*=)\s*["'`]/(?!/)"""
        )
        offenders = []
        for path in [
            *root.glob("static/*.js"),
            *root.glob("static/*.html"),
            *root.glob("templates/*.html"),
        ]:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{number}: {line.strip()[:80]}")
        self.assertEqual(offenders, [])

    async def test_redirects_and_cookies_are_relative_and_unscoped(self):
        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await self.enable()
        # the browser-facing redirect on the API port, from a path with a prefix-like depth
        for path in ("/configuration",):
            response = await self.send("GET", path, headers={"Accept": "text/html"})
            self.assertEqual(response.status_code, 303)
            self.assertFalse(
                response.headers["Location"].startswith(("/", "http")), response.headers["Location"]
            )
        login = await self.send(
            "POST",
            "/login",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=f"token={TOKEN}".encode(),
        )
        self.assertFalse(
            login.headers["Location"].startswith(("/", "http")), login.headers["Location"]
        )
        self.assertNotIn("Path=", login.headers["Set-Cookie"])


class TestRelativeHelperDepth(AuthSupportBase):
    async def test_a_redirect_climbs_out_of_every_path_segment(self):
        for path, expected in (
            ("/configuration", "login"),
            ("/a/configuration", "../login"),
            ("/a/b/configuration", "../../login"),
        ):
            with self.subTest(path=path):
                async with web_server.app.test_request_context(path):
                    self.assertEqual(api_auth._relative("login"), expected)


if __name__ == "__main__":
    unittest.main()
