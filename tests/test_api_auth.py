"""API access control (emhass.api_auth) and its wiring into the web server."""

import io
import json
import logging
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import yaml

from emhass import api_auth, utils, web_server

TOKEN = "correct-horse-battery-staple-0123"


class AuthTestBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.original_conf = web_server.emhass_conf
        web_server.emhass_conf = {"data_path": self.dir}
        api_auth.reset()
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop(api_auth.ENV_NAME, None)
        self.delay = patch.object(api_auth, "FAILED_LOGIN_DELAY_S", 0)
        self.delay.start()
        self.client = web_server.app.test_client()
        self.logger = logging.getLogger("test_api_auth")

    async def asyncTearDown(self):
        self.delay.stop()
        self.env.stop()
        api_auth.reset()
        web_server.emhass_conf = self.original_conf
        self.tmp.cleanup()

    def write_secrets(self, content: str) -> pathlib.Path:
        path = self.dir / "secrets_emhass.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def write_options(self, content: str) -> pathlib.Path:
        path = self.dir / "options.json"
        path.write_text(content, encoding="utf-8")
        return path

    async def enable(self, token: str = TOKEN):
        secrets = self.write_secrets(yaml.safe_dump({api_auth.SETTING_NAME: token}))
        await api_auth.configure(secrets, None, self.logger)
        self.assertEqual(api_auth.current_token(), token)


class TestNoTokenConfigured(AuthTestBase):
    """With no token and nothing to generate one into, the server is closed, not open."""

    async def test_routes_are_refused(self):
        await api_auth.configure(None, None, self.logger)
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 503)

    async def test_before_configure_runs_nothing_is_open(self):
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 503)

    @patch("emhass.web_server._action_call_unlocked", new_callable=AsyncMock)
    async def test_actions_do_not_run(self, mock_action):
        await api_auth.configure(None, None, self.logger)
        for headers in ({}, {"Authorization": "Bearer anything-at-all-0123456789"}):
            response = await self.client.post("/action/publish-data", json={}, headers=headers)
            self.assertEqual(response.status_code, 503)
        mock_action.assert_not_awaited()

    async def test_login_is_unavailable(self):
        await api_auth.configure(None, None, self.logger)
        response = await self.client.get("/login")
        self.assertEqual(response.status_code, 503)

    async def test_health_check_stays_open(self):
        response = await self.client.get("/healthz")
        # /healthz reports readiness itself (200 or 503); it is never the access-control refusal
        self.assertNotIn("API authentication", await response.get_data(as_text=True))
        self.assertNotIn("WWW-Authenticate", response.headers)

    @patch("emhass.web_server._action_call_unlocked", new_callable=AsyncMock)
    async def test_explicit_opt_out_runs_open(self, mock_action):
        mock_action.return_value = "ran"
        with patch.dict(os.environ, {api_auth.DISABLE_ENV: "true"}):
            await api_auth.configure(None, None, self.logger)
            response = await self.client.post("/action/publish-data", json={})
            login = await self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(login.headers["Location"], "./")
        mock_action.assert_awaited_once()

    async def test_only_a_clear_opt_out_value_counts(self):
        for value in ("", "0", "false", "no", "off", "tru", "disabled"):
            with self.subTest(value=value), patch.dict(os.environ, {api_auth.DISABLE_ENV: value}):
                await api_auth.configure(None, None, self.logger)
                response = await self.client.get("/api/v1/last-run")
                self.assertEqual(response.status_code, 503)


class TestBearerToken(AuthTestBase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.enable()

    async def test_missing_header_is_401(self):
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["WWW-Authenticate"], 'Bearer realm="EMHASS"')

    async def test_wrong_token_is_401(self):
        for header in (
            "Bearer wrong",
            "Bearer " + TOKEN[:-1],
            "Bearer " + TOKEN + "x",
            "Basic " + TOKEN,
            TOKEN,
            "Bearer",
            "Bearer ",
        ):
            with self.subTest(header=header):
                response = await self.client.get(
                    "/api/v1/last-run", headers={"Authorization": header}
                )
                self.assertEqual(response.status_code, 401)

    async def test_right_token_passes(self):
        for header in ("Bearer " + TOKEN, "bearer " + TOKEN, "BEARER  " + TOKEN + " "):
            with self.subTest(header=header):
                response = await self.client.get(
                    "/api/v1/last-run", headers={"Authorization": header}
                )
                self.assertEqual(response.status_code, 200)

    @patch("emhass.web_server._action_call_unlocked", new_callable=AsyncMock)
    async def test_action_runs_only_with_token(self, mock_action):
        mock_action.return_value = "ran"
        response = await self.client.post("/action/publish-data", json={})
        self.assertEqual(response.status_code, 401)
        mock_action.assert_not_awaited()
        response = await self.client.post(
            "/action/publish-data", json={}, headers={"Authorization": "Bearer " + TOKEN}
        )
        self.assertEqual(response.status_code, 200)
        mock_action.assert_awaited_once()

    @patch("emhass.web_server.build_params", new_callable=AsyncMock)
    async def test_set_config_refused_without_token(self, mock_build):
        response = await self.client.post("/set-config", json={})
        self.assertEqual(response.status_code, 401)
        mock_build.assert_not_awaited()

    async def test_healthz_stays_open(self):
        response = await self.client.get("/healthz")
        self.assertNotEqual(response.status_code, 401)
        self.assertIn(response.status_code, (200, 503))

    async def test_unknown_path_is_401_not_404(self):
        response = await self.client.get("/no-such-route")
        self.assertEqual(response.status_code, 401)
        response = await self.client.get(
            "/no-such-route", headers={"Authorization": "Bearer " + TOKEN}
        )
        self.assertEqual(response.status_code, 404)

    async def test_static_files_need_credentials(self):
        response = await self.client.get("/static/script.js")
        self.assertEqual(response.status_code, 401)
        response = await self.client.get(
            "/static/script.js", headers={"Authorization": "Bearer " + TOKEN}
        )
        self.assertEqual(response.status_code, 200)

    @patch("emhass.web_server._action_call_unlocked", new_callable=AsyncMock)
    async def test_bearer_requests_are_not_limited_to_json(self, mock_action):
        mock_action.return_value = "ran"
        response = await self.client.post(
            "/action/publish-data", headers={"Authorization": "Bearer " + TOKEN}
        )
        self.assertEqual(response.status_code, 200)

    async def test_options_request_needs_credentials(self):
        response = await self.client.options("/action/publish-data")
        self.assertEqual(response.status_code, 401)

    async def test_ingress_and_forwarded_headers_do_not_authenticate(self):
        headers = {
            "X-Ingress-Path": "/api/hassio_ingress/abc",
            "X-Forwarded-For": "192.0.2.2",
            "X-Real-IP": "127.0.0.1",
            "Forwarded": "for=127.0.0.1",
            "X-EMHASS-Gate": "1",
        }
        response = await self.client.get("/api/v1/last-run", headers=headers)
        self.assertEqual(response.status_code, 401)

    async def test_non_latin1_header_is_refused_not_crashing(self):
        # Bytes above 0x7f in the header must be compared, not crash the hook.
        response = await self.client.get(
            "/api/v1/last-run", headers={"Authorization": "Bearer café"}
        )
        self.assertEqual(response.status_code, 401)

    async def test_browser_page_load_redirects_to_login(self):
        for path, location in (("/", "login"), ("/index", "login"), ("/configuration", "login")):
            with self.subTest(path=path):
                response = await self.client.get(path, headers={"Accept": "text/html"})
                self.assertEqual(response.status_code, 303)
                self.assertEqual(response.headers["Location"], location)

    async def test_api_get_without_html_accept_is_401(self):
        response = await self.client.get("/", headers={"Accept": "application/json"})
        self.assertEqual(response.status_code, 401)


class TestUtf8Token(AuthTestBase):
    async def test_utf8_token_matches_its_bytes(self):
        token = "töken-with-ümlauts-0123456"
        await self.enable(token)
        # The test client sends header values as UTF-8 bytes, as a UTF-8 client
        # would. The app decodes them as latin-1 and must get the bytes back.
        response = await self.client.get(
            "/api/v1/last-run", headers={"Authorization": "Bearer " + token}
        )
        self.assertEqual(response.status_code, 200)
        response = await self.client.get(
            "/api/v1/last-run",
            headers={"Authorization": "Bearer " + token.encode().decode("latin-1")},
        )
        self.assertEqual(response.status_code, 401)


class TestBearerBytes(AuthTestBase):
    async def test_token_ending_in_a_byte_str_strip_removes(self):
        # The UTF-8 bytes of these end in 0xA0 and 0x85, which Python's
        # str.strip() treats as space once the header is decoded as latin-1.
        for token in ("token-ending-in-0123456-à", "token-ending-in-0123456-Ņ"):
            with self.subTest(token=token):
                await self.enable(token)
                response = await self.client.get(
                    "/api/v1/last-run", headers={"Authorization": "Bearer " + token}
                )
                self.assertEqual(response.status_code, 200)

    async def test_token_ending_in_a_character_str_strip_removes(self):
        # U+00A0 and U+0085 are space to str.strip(), so trimming the token
        # with it at load would leave a token no client can send.
        for token in ("token-ending-in-0123456- ", "token-ending-in-0123456-\u0085"):
            # (A form feed or vertical tab is trimmed instead: no header can carry it.)
            with self.subTest(token=repr(token)):
                api_auth.reset()
                await self.enable(token)
                response = await self.client.get(
                    "/api/v1/last-run", headers={"Authorization": "Bearer " + token}
                )
                self.assertEqual(response.status_code, 200)
                response = await self.client.post("/login", form={"token": token})
                self.assertEqual(response.status_code, 303)
                self.assertIn(api_auth.SESSION_COOKIE, response.headers.get("Set-Cookie", ""))


class TestWebSocket(AuthTestBase):
    async def handshake_status(self, headers=None) -> int:
        from quart.testing import WebsocketResponseError

        try:
            async with self.client.websocket("/api/v1/last-run", headers=headers) as ws:
                await ws.send("x")
                await ws.receive()
        except WebsocketResponseError as e:
            return e.response.status_code
        return 101

    async def test_handshake_needs_the_bearer_token(self):
        await self.enable()
        self.assertEqual(await self.handshake_status(), 401)
        cookie = {"Cookie": f"{api_auth.SESSION_COOKIE}={api_auth.make_session_value(TOKEN)}"}
        self.assertEqual(await self.handshake_status(cookie), 401)
        status = await self.handshake_status({"Authorization": "Bearer " + TOKEN})
        self.assertNotEqual(status, 401)

    async def test_handshake_when_configuration_failed(self):
        await api_auth.configure(self.write_secrets("emhass_api_token: true\n"), None, self.logger)
        self.assertEqual(await self.handshake_status(), 503)

    async def test_handshake_without_a_token_is_refused(self):
        await api_auth.configure(None, None, self.logger)
        self.assertEqual(await self.handshake_status(), 503)


class TestTokenLogFilter(AuthTestBase):
    def record(self, msg, args=(), exc_info=None) -> logging.LogRecord:
        return logging.LogRecord("x", logging.INFO, __file__, 1, msg, args, exc_info)

    async def test_masks_the_token_by_value(self):
        await self.enable()
        record = self.record("body: %s", ({"nested": {"token": TOKEN}},))
        self.assertTrue(api_auth.token_log_filter.filter(record))
        self.assertNotIn(TOKEN, record.getMessage())
        self.assertIn("***", record.getMessage())

    async def test_masks_the_token_in_a_traceback(self):
        await self.enable()
        try:
            raise ValueError(f"bad line: {TOKEN}")
        except ValueError:
            record = self.record("failed", exc_info=sys.exc_info())
        api_auth.token_log_filter.filter(record)
        text = logging.Formatter().format(record)
        self.assertIn("ValueError", text)
        self.assertNotIn(TOKEN, text)

    async def test_masks_escaped_forms_of_the_token(self):
        # A dict in a log line is written with repr(), which escapes a
        # backslash, quotes and control characters; JSON escapes them too.
        for token in (
            "tok\\en-0123456789abcdef",
            'it\'s-a-"quoted"-token-0123',
            "tab\there-0123456789abcdef",
            "töken-with-ümlauts-0123456",
        ):
            with self.subTest(token=token):
                await self.enable(token)
                for text in (
                    repr({"x": token}),
                    json.dumps({"x": token}),
                    token,
                    "GET /x?t=" + api_auth.urllib.parse.quote(token, safe=""),
                    "GET /x?t=" + api_auth.urllib.parse.quote_plus(token),
                ):
                    record = self.record("body: %s", (text,))
                    api_auth.token_log_filter.filter(record)
                    self.assertEqual(record.getMessage().count("***"), 1, record.getMessage())

    async def test_leaves_records_alone_without_a_token(self):
        await api_auth.configure(None, None, self.logger)
        record = self.record("value %s", ("x",))
        api_auth.token_log_filter.filter(record)
        self.assertEqual(record.args, ("x",))

    async def test_token_in_a_request_body_stays_out_of_the_action_log(self):
        await self.enable()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(api_auth.token_log_filter)
        app_logger = web_server.app.logger
        saved = (app_logger.level, app_logger.disabled)
        app_logger.setLevel(logging.INFO)
        app_logger.disabled = False
        app_logger.addHandler(handler)
        try:
            body = {"nested": {"token": TOKEN}, "Authorization": "Bearer " + TOKEN}
            web_server.app.logger.warning("Passed runtime parameters: %s", body)
        finally:
            app_logger.removeHandler(handler)
            app_logger.setLevel(saved[0])
            app_logger.disabled = saved[1]
        self.assertNotIn(TOKEN, stream.getvalue())
        self.assertIn("Passed runtime parameters", stream.getvalue())

    def test_battery_count_error_quotes_no_cut_copy_of_a_long_value(self):
        # /get-json and /set-config log this message. A cut copy of a token
        # pasted here by mistake would not be matched by the mask.
        from emhass import utils

        token = "Abcdefghij0123456789KLMNOPQRSTUVWXYZ_-abcde"
        with self.assertRaises(utils.BatteryConfigError) as ctx:
            utils.validate_num_batteries({"number_of_batteries": token})
        self.assertNotIn(token[:12], str(ctx.exception))
        with self.assertRaises(utils.BatteryConfigError) as ctx:
            utils.validate_num_batteries({"number_of_batteries": "abc"})
        self.assertIn("'abc'", str(ctx.exception))

    async def test_yaml_error_in_get_json_logs_no_piece_of_the_token(self):
        # PyYAML's error text quotes the line twice, each excerpt cut to about
        # 75 characters, so a token of 40 to 63 characters would be logged as a
        # head and a tail, which a whole-token mask cannot match.
        token = "Abcdefghij0123456789KLMNOPQRSTUVWXYZ_-abcde"
        await self.enable(token)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        app_logger = web_server.app.logger
        saved = (app_logger.level, app_logger.disabled)
        app_logger.setLevel(logging.DEBUG)
        app_logger.disabled = False
        app_logger.addHandler(handler)
        headers = {"Authorization": "Bearer " + token, "Content-Type": "text/plain"}
        try:
            for body in (
                f'emhass_api_token: "{token}',
                f"x: [{token}}}",
                '{"token": "' + token + '"", "a": 1}',
                # Tags whose errors are not YAMLError and hold a changed copy:
                # lowercased (!!bool) or without underscores (!!int, !!float).
                f"x: !!bool {token}",
                f"x: !!int {token}",
                f"x: !!float {token}",
            ):
                with self.subTest(body=body.replace(token, "<token>")):
                    response = await self.client.post(
                        "/get-json", data=body.encode(), headers=headers
                    )
                    self.assertEqual(response.status_code, 400)
        finally:
            app_logger.removeHandler(handler)
            app_logger.setLevel(saved[0])
            app_logger.disabled = saved[1]
        logged = stream.getvalue()
        self.assertIn("Could not parse the YAML sent to /get-json", logged)

        def plain(text):
            return text.lower().replace("_", "").replace("-", "")

        flat_token, flat_log = plain(token), plain(logged)
        pieces = [flat_token[i : i + 12] for i in range(len(flat_token) - 11)]
        self.assertFalse([p for p in pieces if p in flat_log])


class TestLogin(AuthTestBase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.enable()

    async def test_login_page_is_open(self):
        response = await self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        body = await response.get_data(as_text=True)
        self.assertIn('action="login"', body)
        self.assertNotIn(TOKEN, body)

    async def test_wrong_token_is_401_and_sets_no_cookie(self):
        response = await self.client.post("/login", form={"token": "nope"})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("Set-Cookie", response.headers)

    async def test_right_token_sets_strict_httponly_cookie(self):
        response = await self.client.post("/login", form={"token": TOKEN})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["Location"], "./")
        cookie = response.headers["Set-Cookie"]
        self.assertTrue(cookie.startswith(api_auth.SESSION_COOKIE + "="))
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertNotIn("Path=", cookie)  # the browser scopes it to the login folder
        self.assertNotIn("Secure", cookie)  # plain http in the test client
        self.assertNotIn(TOKEN, cookie)

    async def test_cookie_is_secure_over_https(self):
        response = await self.client.post("/login", form={"token": TOKEN}, scheme="https")
        self.assertIn("Secure", response.headers["Set-Cookie"])

    async def test_cookie_is_secure_behind_a_tls_proxy(self):
        for value in ("https", "HTTPS", "https, http"):
            with self.subTest(value=value):
                response = await self.client.post(
                    "/login", form={"token": TOKEN}, headers={"X-Forwarded-Proto": value}
                )
                self.assertIn("Secure", response.headers["Set-Cookie"])
        response = await self.client.post("/logout", headers={"X-Forwarded-Proto": "https"})
        self.assertIn("Secure", response.headers["Set-Cookie"])
        for value in ("proto=https", 'for=192.0.2.1;proto="https"', "Proto=HTTPS;host=x"):
            with self.subTest(forwarded=value):
                response = await self.client.post(
                    "/login", form={"token": TOKEN}, headers={"Forwarded": value}
                )
                self.assertIn("Secure", response.headers["Set-Cookie"])
        for value in ("proto=http", "for=192.0.2.1, proto=https", "host=https"):
            with self.subTest(forwarded=value):
                response = await self.client.post(
                    "/login", form={"token": TOKEN}, headers={"Forwarded": value}
                )
                self.assertNotIn("Secure", response.headers["Set-Cookie"])

    async def test_large_login_body_is_refused_before_parsing(self):
        with patch.object(api_auth.urllib.parse, "parse_qs") as mock_parse:
            # The test client sends no Content-Length, as a chunked upload would not.
            response = await self.client.post("/login", form={"token": "x" * 5000})
            self.assertEqual(response.status_code, 413)
            response = await self.client.post(
                "/login",
                data=b"token=x",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Content-Length": "999999",
                },
            )
            self.assertEqual(response.status_code, 413)
        mock_parse.assert_not_called()
        self.assertEqual(api_auth._failures["count"], 0)

    async def test_login_must_be_a_plain_form(self):
        for headers, data in (
            ({"Content-Type": "multipart/form-data; boundary=x"}, b"--x--"),
            ({"Content-Type": "application/json"}, b'{"token": "x"}'),
            ({"Content-Type": "text/plain"}, b"token=" + TOKEN.encode()),
        ):
            with self.subTest(content_type=headers["Content-Type"]):
                response = await self.client.post("/login", data=data, headers=headers)
                self.assertEqual(response.status_code, 415)
                self.assertNotIn("Set-Cookie", response.headers)

    async def test_login_with_too_many_fields(self):
        response = await self.client.post(
            "/login",
            data=b"a=1&" * 100 + b"token=" + TOKEN.encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("Set-Cookie", response.headers)

    async def test_head_login_is_not_a_failed_attempt(self):
        response = await self.client.head("/login")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(api_auth._failures["count"], 0)

    async def test_cookie_signed_with_another_key_is_refused(self):
        # As after a restart: the key is made at import.
        with patch.object(api_auth, "_SESSION_KEY", b"\x00" * 32):
            old = api_auth.make_session_value(TOKEN)
        self.assertFalse(api_auth.session_is_valid(TOKEN, old))
        self.assertTrue(api_auth.session_is_valid(TOKEN, api_auth.make_session_value(TOKEN)))

    async def test_session_cookie_authenticates(self):
        value = api_auth.make_session_value(TOKEN)
        response = await self.client.get(
            "/api/v1/last-run", headers={"Cookie": f"{api_auth.SESSION_COOKIE}={value}"}
        )
        self.assertEqual(response.status_code, 200)

    async def test_bad_cookies_are_refused(self):
        expired = api_auth.make_session_value(TOKEN, now=0)
        other_token = api_auth.make_session_value("a-different-token-0123456789")
        good = api_auth.make_session_value(TOKEN)
        expires, _, mac = good.partition(".")
        tampered_expiry = f"{int(expires) + 1}.{mac}"
        for value in (
            expired,
            other_token,
            tampered_expiry,
            "garbage",
            ".",
            "123.",
            TOKEN,
            "9" * 5000 + "." + mac,  # past Python's int-from-string limit
            "¹²³." + mac,  # digits to isdigit(), not to int()
        ):
            with self.subTest(value=value[:40]):
                response = await self.client.get(
                    "/api/v1/last-run", headers={"Cookie": f"{api_auth.SESSION_COOKIE}={value}"}
                )
                self.assertEqual(response.status_code, 401)

    def test_session_parser_never_raises(self):
        for value in ("9" * 5000 + ".ab", "¹².ab", "١٢.ab", "-1.ab"):
            with self.subTest(value=value[:20]):
                self.assertFalse(api_auth.session_is_valid(TOKEN, value))

    @patch("emhass.web_server._action_call_unlocked", new_callable=AsyncMock)
    async def test_cookie_post_must_be_json(self, mock_action):
        mock_action.return_value = "ran"
        cookie = {"Cookie": f"{api_auth.SESSION_COOKIE}={api_auth.make_session_value(TOKEN)}"}
        # What a page on another origin can send without a CORS preflight.
        for name, kwargs in (
            ("form", {"form": {"a": "1"}}),
            ("text/plain", {"data": b"{}", "headers": {"Content-Type": "text/plain"}}),
            ("no body", {}),
        ):
            with self.subTest(name=name):
                headers = {**cookie, **kwargs.pop("headers", {})}
                response = await self.client.post("/action/publish-data", headers=headers, **kwargs)
                self.assertEqual(response.status_code, 403)
        mock_action.assert_not_awaited()
        # What the web UI sends.
        response = await self.client.post("/action/publish-data", headers=cookie, json={})
        self.assertEqual(response.status_code, 200)
        mock_action.assert_awaited_once()

    async def test_changing_the_token_ends_sessions(self):
        value = api_auth.make_session_value(TOKEN)
        await self.enable("a-new-token-after-rotation-0001")
        response = await self.client.get(
            "/api/v1/last-run", headers={"Cookie": f"{api_auth.SESSION_COOKIE}={value}"}
        )
        self.assertEqual(response.status_code, 401)

    async def test_logout_clears_cookie(self):
        response = await self.client.post("/logout")
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["Location"], "login")
        cookie = response.headers["Set-Cookie"]
        self.assertTrue(cookie.startswith(api_auth.SESSION_COOKIE + "=;"))
        self.assertTrue("Max-Age=0" in cookie or "Expires=Thu, 01 Jan 1970" in cookie)

    async def test_logout_by_get_only_shows_a_button(self):
        response = await self.client.get("/logout")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Set-Cookie", response.headers)
        self.assertIn('method="post"', await response.get_data(as_text=True))

    async def test_logout_by_head_does_not_log_out(self):
        response = await self.client.head("/logout")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Set-Cookie", response.headers)

    async def test_cross_site_logout_is_refused(self):
        response = await self.client.post("/logout", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("Set-Cookie", response.headers)

    async def test_a_tossed_cookie_does_not_lock_the_user_out(self):
        # A page on another port can set a junk cookie with a longer path, which
        # the browser sends first.
        good = api_auth.make_session_value(TOKEN)
        cookie = f"{api_auth.SESSION_COOKIE}=junk; {api_auth.SESSION_COOKIE}={good}"
        response = await self.client.get("/api/v1/last-run", headers={"Cookie": cookie})
        self.assertEqual(response.status_code, 200)

    async def test_failed_attempts_are_logged_as_a_count(self):
        app_logger = web_server.app.logger
        with self.assertLogs(app_logger, level="WARNING") as logs:
            for _ in range(3):
                await self.client.get("/api/v1/last-run", headers={"Authorization": "Bearer nope"})
            await self.client.get(
                "/api/v1/last-run", headers={"Cookie": f"{api_auth.SESSION_COOKIE}=junk"}
            )
            await self.client.post("/login", form={"token": "nope"})
        failures = [line for line in logs.output if "failed API authentication" in line]
        # The first attempt is reported at once; the rest wait for the next minute.
        self.assertEqual(len(failures), 1)
        self.assertNotIn("nope", "\n".join(logs.output))
        self.assertEqual(api_auth._failures["count"], 4)

    async def test_no_credentials_is_not_a_failed_attempt(self):
        await self.client.get("/api/v1/last-run")
        await self.client.get("/", headers={"Accept": "text/html"})
        self.assertEqual(api_auth._failures["count"], 0)
        self.assertIsNone(api_auth._failures["since"])


class TestTokenSources(AuthTestBase):
    async def test_env_options_and_secrets_precedence(self):
        options = self.write_options('{"emhass_api_token": "from-options-0123456"}')
        secrets = self.write_secrets("emhass_api_token: from-secrets-0123456\n")
        env = {api_auth.ENV_NAME: "from-env-0123456789"}
        self.assertEqual(await api_auth.load_api_token(None, None, env), "from-env-0123456789")
        self.assertEqual(await api_auth.load_api_token(None, options, env), "from-options-0123456")
        self.assertEqual(
            await api_auth.load_api_token(secrets, options, env), "from-secrets-0123456"
        )

    async def test_empty_values_mean_not_set(self):
        options = self.write_options('{"emhass_api_token": ""}')
        secrets = self.write_secrets("emhass_api_token: '   '\nhass_url: http://x/\n")
        self.assertIsNone(await api_auth.load_api_token(secrets, options, {}))

    async def test_ascii_white_space_is_trimmed(self):
        # Form feed and vertical tab too: an HTTP header cannot carry them.
        token = "trimmed-token-0123456789"
        for space in (" ", "\t", "\r\n", "\f", "\v"):
            with self.subTest(space=repr(space)):
                env = {api_auth.ENV_NAME: space + token + space}
                self.assertEqual(api_auth.read_api_token(None, None, env), token)

    async def test_empty_later_source_keeps_earlier_token(self):
        env = {api_auth.ENV_NAME: "e" * 20}
        for options_text, secrets_text in (
            ('{"emhass_api_token": ""}', "emhass_api_token: ''\n"),
            ('{"emhass_api_token": "   "}', "emhass_api_token: '   '\n"),
            ('{"emhass_api_token": null}', "emhass_api_token: ''\n"),
            ("{}", "hass_url: http://x/\n"),
            ("", "# comments only\n"),
            ("{}", ""),
        ):
            with self.subTest(options=options_text, secrets=secrets_text):
                options = self.write_options(options_text)
                secrets = self.write_secrets(secrets_text)
                self.assertEqual(await api_auth.load_api_token(secrets, options, env), "e" * 20)
        options = self.write_options('{"emhass_api_token": "o' + "o" * 20 + '"}')
        secrets = self.write_secrets("emhass_api_token: '  '\n")
        self.assertEqual(await api_auth.load_api_token(secrets, options, env), "o" * 21)

    async def test_missing_files_mean_not_set(self):
        self.assertIsNone(
            await api_auth.load_api_token(self.dir / "nope.yaml", self.dir / "nope.json", {})
        )

    async def test_quoted_digits_are_kept_as_written(self):
        secrets = self.write_secrets("emhass_api_token: '0000000000001234'\n")
        self.assertEqual(await api_auth.load_api_token(secrets, None, {}), "0000000000001234")

    async def test_env_var_applies_before_configure(self):
        os.environ[api_auth.ENV_NAME] = TOKEN
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 401)
        response = await self.client.get(
            "/api/v1/last-run", headers={"Authorization": "Bearer " + TOKEN}
        )
        self.assertEqual(response.status_code, 200)


class TestFailClosed(AuthTestBase):
    async def check_failed(self, secrets=None, options=None):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            await api_auth.configure(secrets, options, self.logger)
        self.assertNotIn("secret-value", "\n".join(logs.output))
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 503)
        response = await self.client.get(
            "/api/v1/last-run", headers={"Authorization": "Bearer secret-value"}
        )
        self.assertEqual(response.status_code, 503)
        response = await self.client.get("/healthz")
        self.assertIn(response.status_code, (200, 503))
        self.assertIn("status", await response.get_json())
        response = await self.client.get("/login")
        self.assertEqual(response.status_code, 503)

    async def test_unparsable_secrets_file(self):
        await self.check_failed(secrets=self.write_secrets("emhass_api_token: [secret-value\n"))

    async def test_unparsable_options_file(self):
        await self.check_failed(options=self.write_options('{"emhass_api_token": "secret-value'))

    async def test_token_that_is_not_a_string(self):
        await self.check_failed(
            secrets=self.write_secrets("emhass_api_token:\n  nested: secret-value\n")
        )

    async def test_unquoted_number_token(self):
        # YAML would read this as octal 668 and shorten the token to three digits.
        await self.check_failed(secrets=self.write_secrets("emhass_api_token: 0000000000001234\n"))

    async def test_number_in_options(self):
        await self.check_failed(
            options=self.write_options('{"emhass_api_token": 12345678901234567}')
        )

    async def test_boolean_token(self):
        await self.check_failed(secrets=self.write_secrets("emhass_api_token: true\n"))

    async def test_secrets_path_is_a_directory(self):
        # What Docker creates for a bind-mounted file that is missing on the host.
        path = self.dir / "secrets_emhass.yaml"
        path.mkdir()
        await self.check_failed(secrets=path)

    async def test_link_to_a_missing_file(self):
        path = self.dir / "secrets_emhass.yaml"
        try:
            path.symlink_to(self.dir / "not-there.yaml")
        except OSError:
            self.skipTest("symlinks need admin rights here")
        await self.check_failed(secrets=path)

    async def test_secrets_file_that_cannot_be_read(self):
        path = self.write_secrets(yaml.safe_dump({api_auth.SETTING_NAME: TOKEN}))
        with patch.object(pathlib.Path, "read_text", side_effect=PermissionError(13, "denied")):
            await self.check_failed(secrets=path)

    async def test_secrets_file_that_is_not_a_mapping(self):
        for text in (
            'emhass_api_token:"secret-value-0123456"\n',  # no space after the colon
            'emhass_api_token = "secret-value-0123456"\n',
            "- emhass_api_token: secret-value-0123456\n",
            "just some text\n",
        ):
            with self.subTest(text=text):
                api_auth.reset()
                await self.check_failed(secrets=self.write_secrets(text))

    async def test_options_file_that_is_not_an_object(self):
        for text in ('["secret-value"]', '"secret-value"', "1"):
            with self.subTest(text=text):
                api_auth.reset()
                await self.check_failed(options=self.write_options(text))

    async def test_key_with_no_value(self):
        # An unquoted token that starts with "#" is a YAML comment, so the key
        # reads as empty. That is a mistake, not "no token".
        for text in ("emhass_api_token: #secret-value\n", "emhass_api_token:\n"):
            with self.subTest(text=text):
                api_auth.reset()
                await self.check_failed(secrets=self.write_secrets(text))

    async def test_token_set_twice(self):
        # safe_load keeps the last one, so an empty second key would turn auth off.
        for second in ('""', "' '", '"secret-value-other-01"', "\n"):
            text = f'emhass_api_token: "secret-value-0123456"\nemhass_api_token: {second}\n'
            with self.subTest(second=second):
                api_auth.reset()
                path = self.write_secrets(text)
                with self.assertRaises(api_auth.ApiTokenError):
                    api_auth.read_api_token(path, None, {})
                await self.check_failed(secrets=path)

    async def test_token_set_twice_through_merge_keys(self):
        real = 'real: &real {emhass_api_token: "secret-value-0123456"}\n'
        empty = 'empty: &empty {emhass_api_token: ""}\n'
        for text in (
            real + empty + "<<: *real\n<<: *empty\n",
            real + empty + "<<: [*empty, *real]\n",
            real + '<<: *real\nemhass_api_token: ""\n',
        ):
            with self.subTest(text=text):
                api_auth.reset()
                await self.check_failed(secrets=self.write_secrets(text))

    def test_one_merged_token_is_read(self):
        path = self.write_secrets(f'base: &base {{emhass_api_token: "{TOKEN}"}}\n<<: *base\n')
        self.assertEqual(api_auth.read_api_token(path, None, {}), TOKEN)

    def test_value_that_cannot_be_converted_is_not_quoted(self):
        for tag in ("!!bool", "!!int", "!!float"):
            with self.subTest(tag=tag):
                path = self.write_secrets(f"hass_url: {tag} Hidden_Value_0123\n")
                with self.assertRaises(api_auth.ApiTokenError) as ctx:
                    api_auth.read_api_token(path, None, {})
                self.assertNotIn("hidden", str(ctx.exception).lower())
                self.assertIsNone(ctx.exception.__cause__)
                self.assertTrue(ctx.exception.__suppress_context__)

    def test_other_repeated_keys_are_left_alone(self):
        path = self.write_secrets(
            f"hass_url: a\nhass_url: b\nnested:\n  {api_auth.SETTING_NAME}: x\n"
            f'{api_auth.SETTING_NAME}: "{TOKEN}"\n'
        )
        self.assertEqual(api_auth.read_api_token(path, None, {}), TOKEN)

    async def test_token_that_is_not_valid_utf8(self):
        # A lone surrogate cannot be encoded, so it could not be compared or
        # masked; before, it made every log call raise.
        await self.check_failed(
            secrets=self.write_secrets('emhass_api_token: "\\ud800-secret-value-0123456"\n')
        )
        with self.assertRaises(api_auth.ApiTokenError):
            api_auth.read_api_token(None, None, {api_auth.ENV_NAME: "\udcff-secret-value-01"})

    async def test_failure_log_says_to_restart(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            await api_auth.configure(
                self.write_secrets("emhass_api_token: true\n"), None, self.logger
            )
        self.assertIn("restarted", "\n".join(logs.output))

    def test_errors_do_not_carry_the_yaml_text(self):
        path = self.write_secrets("emhass_api_token: [secret-value\n")
        with self.assertRaises(api_auth.ApiTokenError) as ctx:
            api_auth.read_api_token(path, None, {})
        self.assertIsNone(ctx.exception.__cause__)
        self.assertTrue(ctx.exception.__suppress_context__)
        self.assertNotIn("secret-value", str(ctx.exception))


class TestStartupWarning(AuthTestBase):
    async def test_says_on_with_a_configured_token(self):
        await self.enable()
        with self.assertNoLogs(self.logger, level="WARNING"):
            api_auth.log_status(self.logger, "0.0.0.0")
        with self.assertLogs(self.logger, level="INFO") as logs:
            api_auth.log_status(self.logger, "0.0.0.0")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("API access control is on (a configured token)", logs.output[0])
        self.assertNotIn(TOKEN, logs.output[0])

    async def test_errors_when_no_token_exists_and_none_can_be_made(self):
        await api_auth.configure(None, None, self.logger)
        with self.assertLogs(self.logger, level="ERROR") as logs:
            api_auth.log_status(self.logger, "0.0.0.0")
        self.assertIn("no emhass_api_token is available", logs.output[0])

    async def test_warns_loudly_when_switched_off(self):
        with patch.dict(os.environ, {api_auth.DISABLE_ENV: "true"}):
            await api_auth.configure(None, None, self.logger)
            for host in ("0.0.0.0", "127.0.0.1"):
                with self.subTest(host=host), self.assertLogs(self.logger, level="WARNING") as logs:
                    api_auth.log_status(self.logger, host)
                self.assertIn("API access control is OFF", logs.output[0])

    async def test_quiet_when_configuration_failed(self):
        await api_auth.configure(self.write_secrets("emhass_api_token: true\n"), None, self.logger)
        with self.assertNoLogs(self.logger, level="INFO"):
            api_auth.log_status(self.logger, "0.0.0.0")

    async def test_short_token_is_refused_without_printing_it(self):
        secrets = self.write_secrets("emhass_api_token: short-secret\n")
        with self.assertLogs(self.logger, level="ERROR") as logs:
            await api_auth.configure(secrets, None, self.logger)
        text = "\n".join(logs.output)
        self.assertIn("shorter than", text)
        self.assertNotIn("short-secret", text)
        response = await self.client.get(
            "/api/v1/last-run", headers={"Authorization": "Bearer short-secret"}
        )
        self.assertEqual(response.status_code, 503)

    async def test_sixteen_characters_is_enough(self):
        await self.enable("x" * api_auth.MIN_LENGTH)

    async def test_bind_host_under_gunicorn_follows_ip_env(self):
        with patch.dict(sys.modules, {"gunicorn": object()}):
            with patch.dict(os.environ, {"IP": "127.0.0.1"}):
                self.assertEqual(api_auth.bind_host("0.0.0.0"), "127.0.0.1")
            os.environ.pop("IP", None)
            self.assertEqual(api_auth.bind_host("127.0.0.1"), "0.0.0.0")

    async def test_bind_host_run_directly_follows_server_ip(self):
        with patch.dict(sys.modules):
            sys.modules.pop("gunicorn", None)
            self.assertEqual(api_auth.bind_host("127.0.0.1"), "127.0.0.1")
            self.assertEqual(api_auth.bind_host(None), "0.0.0.0")


class TestWebServerWiring(AuthTestBase):
    async def test_token_loaded_even_if_initialisation_fails(self):
        # before_serving() swallows an initialize() failure and keeps serving, so
        # the token has to be loaded before initialize() runs, not inside it.
        secrets = self.write_secrets(yaml.safe_dump({api_auth.SETTING_NAME: TOKEN}))
        env = {"SECRETS_PATH": str(secrets), "OPTIONS_PATH": str(self.dir / "options.json")}
        with (
            patch.dict(os.environ, env),
            patch.object(
                web_server, "initialize", AsyncMock(side_effect=ValueError("bad log level"))
            ),
        ):
            await web_server.before_serving()
        self.assertEqual(api_auth.current_token(), TOKEN)
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 401)

    async def test_closed_error_still_logged_when_secrets_setup_fails(self):
        await api_auth.configure(None, None, self.logger)
        paths = (
            self.dir / "config.json",
            self.dir / "options.json",
            self.dir / "defaults.json",
            self.dir / "associations.csv",
            self.dir / "legacy.yaml",
            self.dir,
        )
        original_secrets = web_server.params_secrets
        web_server.params_secrets = {}
        try:
            with (
                patch.object(web_server, "_setup_paths", AsyncMock(return_value=paths)),
                patch.object(
                    web_server, "_setup_secrets", AsyncMock(side_effect=RuntimeError("HA down"))
                ),
                self.assertLogs(web_server.app.logger, level="WARNING") as logs,
            ):
                with self.assertRaises(RuntimeError):
                    await web_server.initialize()
        finally:
            web_server.params_secrets = original_secrets
        self.assertTrue(any("no emhass_api_token is available" in line for line in logs.output))

    async def test_before_serving_keeps_the_token_loaded_at_import(self):
        # A worker that Gunicorn starts later must not take a new
        # reading of the files: a secrets file missing for a moment would switch
        # auth off there.
        await self.enable()
        (self.dir / "secrets_emhass.yaml").unlink()
        env = {
            "SECRETS_PATH": str(self.dir / "secrets_emhass.yaml"),
            "OPTIONS_PATH": str(self.dir / "options.json"),
        }
        with patch.dict(os.environ, env), patch.object(web_server, "initialize", AsyncMock()):
            await web_server.before_serving()
        self.assertEqual(api_auth.current_token(), TOKEN)
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 401)

    async def test_before_serving_fails_closed_if_a_token_appeared(self):
        # The import found no token, but the file holds one by the time the
        # worker starts: refuse rather than run open.
        await api_auth.configure(None, None, self.logger)
        secrets = self.write_secrets(yaml.safe_dump({api_auth.SETTING_NAME: TOKEN}))
        env = {"SECRETS_PATH": str(secrets), "OPTIONS_PATH": str(self.dir / "options.json")}
        with (
            patch.dict(os.environ, env),
            patch.object(web_server, "initialize", AsyncMock()),
            self.assertLogs(web_server.app.logger, level="ERROR") as logs,
        ):
            await web_server.before_serving()
        self.assertIn("restart EMHASS", "\n".join(logs.output))
        self.assertNotIn(TOKEN, "\n".join(logs.output))
        response = await self.client.get(
            "/api/v1/last-run", headers={"Authorization": "Bearer " + TOKEN}
        )
        self.assertEqual(response.status_code, 503)

    async def test_before_serving_stays_closed_when_still_no_token(self):
        await api_auth.configure(None, None, self.logger)
        env = {
            "SECRETS_PATH": str(self.dir / "missing.yaml"),
            "OPTIONS_PATH": str(self.dir / "missing.json"),
        }
        with patch.dict(os.environ, env), patch.object(web_server, "initialize", AsyncMock()):
            await web_server.before_serving()
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 503)

    def test_importing_the_web_server_installs_log_masking(self):
        self.assertIs(logging.getLogRecordFactory(), api_auth._masking_record_factory)
        api_auth.install_log_masking()  # a second call changes nothing
        self.assertIs(logging.getLogRecordFactory(), api_auth._masking_record_factory)
        self.assertIsNot(api_auth._previous_record_factory, api_auth._masking_record_factory)

    async def test_other_loggers_are_masked_too(self):
        # emhass.utils, retrieve_hass, Gunicorn and Uvicorn do not log through
        # the app logger.
        await self.enable()
        for name in ("emhass.utils", "emhass.retrieve_hass", "uvicorn.access", "x"):
            with self.subTest(logger=name):
                stream = io.StringIO()
                handler = logging.StreamHandler(stream)
                other = logging.getLogger(name)
                saved = (other.level, other.disabled, other.propagate)
                other.setLevel(logging.INFO)
                other.disabled = False
                other.propagate = False
                other.addHandler(handler)
                try:
                    other.info("GET /x?t=%s", TOKEN)
                    try:
                        raise ValueError(TOKEN)
                    except ValueError:
                        other.exception("failed")
                finally:
                    other.removeHandler(handler)
                    other.setLevel(saved[0])
                    other.disabled, other.propagate = saved[1], saved[2]
                self.assertIn("GET /x?t=***", stream.getvalue())
                self.assertIn("ValueError", stream.getvalue())
                self.assertNotIn(TOKEN, stream.getvalue())

    async def test_a_handler_without_the_filter_still_gets_it_masked(self):
        # Masking runs as records are made, so a console handler Quart (or
        # Flask) attached, which nobody gave the filter, writes the masked text.
        await self.enable()
        stream = io.StringIO()
        console = logging.StreamHandler(stream)
        app_logger = web_server.app.logger
        saved = (app_logger.level, app_logger.disabled)
        app_logger.setLevel(logging.INFO)
        app_logger.disabled = False
        app_logger.addHandler(console)
        try:
            app_logger.info("Passed runtime parameters: %s", {"t": {"x": TOKEN}})
        finally:
            app_logger.removeHandler(console)
            app_logger.setLevel(saved[0])
            app_logger.disabled = saved[1]
        self.assertNotIn(api_auth.token_log_filter, console.filters)
        self.assertIn("Passed runtime parameters", stream.getvalue())
        self.assertNotIn(TOKEN, stream.getvalue())

    async def test_template_log_line_has_no_status_marker(self):
        # An action's status is read from the action log after the last " >> "
        # line, so GET /template must not write one.
        await self.enable()
        with self.assertLogs(web_server.app.logger, level="INFO") as logs:
            await self.client.get("/template", headers={"Authorization": "Bearer " + TOKEN})
        lines = [line for line in logs.output if "rendered template" in line]
        self.assertEqual(len(lines), 1)
        self.assertNotIn(" >> ", lines[0])

    async def test_token_sources_follow_the_environment(self):
        env = {"SECRETS_PATH": "/x/secrets.yaml", "OPTIONS_PATH": "/x/options.json"}
        with patch.dict(os.environ, env):
            self.assertEqual(
                web_server._api_token_sources(), ("/x/secrets.yaml", "/x/options.json")
            )

    async def test_yaml_error_text_is_not_logged(self):
        error = yaml.YAMLError("line 3: emhass_api_token: [secret-value")
        with (
            patch.object(web_server, "initialize", AsyncMock(side_effect=error)),
            patch.object(web_server, "_configure_api_auth", AsyncMock()),
            self.assertLogs(web_server.app.logger, level="WARNING") as logs,
        ):
            await web_server.before_serving()
        text = "\n".join(logs.output)
        self.assertIn("YAMLError", text)
        self.assertNotIn("secret-value", text)

    async def test_build_secrets_drops_the_api_token(self):
        secrets = self.write_secrets(
            yaml.safe_dump({api_auth.SETTING_NAME: TOKEN, "hass_url": "http://ha:8123/"})
        )
        _, built = await utils.build_secrets(
            {"data_path": self.dir}, self.logger, secrets_path=secrets, no_response=True
        )
        self.assertNotIn(api_auth.SETTING_NAME, built)
        self.assertEqual(built["hass_url"], "http://ha:8123/")

    async def test_build_secrets_does_not_quote_a_value_it_cannot_convert(self):
        # PyYAML's own errors for these quote the value lowercased or without
        # its "_", so the log mask would not match it.
        for tag in ("!!bool", "!!int", "!!float"):
            with self.subTest(tag=tag):
                secrets = self.write_secrets(f"long_lived_token: {tag} Hidden_Value_0123\n")
                with self.assertRaises(ValueError) as ctx:
                    await utils.build_secrets(
                        {"data_path": self.dir}, self.logger, secrets_path=secrets, no_response=True
                    )
                self.assertNotIn("hidden", str(ctx.exception).lower())
                self.assertIsNone(ctx.exception.__cause__)
                self.assertTrue(ctx.exception.__suppress_context__)

    async def test_build_secrets_does_not_quote_a_yaml_error(self):
        # PyYAML's error quotes the line, which can be a secret. Run directly or
        # through the CLI, nothing masks it.
        secrets = self.write_secrets("hass_url: http://ha\nlong_lived_token: [Hidden_Value_0123\n")
        with self.assertRaises(yaml.YAMLError) as ctx:
            await utils.build_secrets(
                {"data_path": self.dir}, self.logger, secrets_path=secrets, no_response=True
            )
        self.assertNotIn("hidden", str(ctx.exception).lower())
        self.assertIn("line", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_token_is_listed_as_config_only(self):
        # Documents the rule: the runtime loop only checks keys in associations.csv,
        # and the token is never read from a body (see the logging test below).
        self.assertIn(api_auth.SETTING_NAME, utils.RUNTIME_DENIED_KEYS)

    def test_redacted(self):
        self.assertEqual(
            api_auth.redacted({api_auth.SETTING_NAME: TOKEN, "a": 1}),
            {api_auth.SETTING_NAME: "***", "a": 1},
        )
        for value in ({"a": 1}, [1, 2], "text", None):
            self.assertEqual(api_auth.redacted(value), value)

    async def test_token_sent_in_a_body_is_not_logged(self):
        body = {api_auth.SETTING_NAME: TOKEN, "prediction_horizon": 10}
        with patch.object(
            web_server, "_load_params", AsyncMock(return_value=({"optim_conf": {}}, "profit"))
        ):
            async with web_server.app.test_request_context(
                "/action/publish-data", method="POST", json=body
            ):
                with self.assertLogs(self.logger, level="INFO") as logs:
                    await web_server._load_params_and_runtime(
                        web_server.request, web_server.emhass_conf, self.logger
                    )
        text = "\n".join(logs.output)
        self.assertIn("Passed runtime parameters", text)
        self.assertIn("prediction_horizon", text)
        self.assertNotIn(TOKEN, text)


class TestWithHardeningHooks(AuthTestBase):
    """The auth hook next to the cross-site check and response headers."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.enable()

    async def test_auth_runs_before_the_cross_site_check(self):
        headers = {"Sec-Fetch-Site": "cross-site"}
        response = await self.client.post("/action/publish-data", headers=headers, json={})
        self.assertEqual(response.status_code, 401)
        headers["Authorization"] = "Bearer " + TOKEN
        response = await self.client.post("/action/publish-data", headers=headers, json={})
        self.assertEqual(response.status_code, 403)

    async def test_auth_replies_get_the_safe_headers(self):
        for path in ("/action/publish-data", "/api/v1/last-run", "/"):
            with self.subTest(path=path):
                response = await self.client.get(path)
                self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
                self.assertNotEqual(response.mimetype, "text/html")

    async def test_login_page_is_still_html(self):
        response = await self.client.get("/login")
        self.assertEqual(response.mimetype, "text/html")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")


if __name__ == "__main__":
    unittest.main()
