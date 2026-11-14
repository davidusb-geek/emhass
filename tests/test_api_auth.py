"""Opt-in API authentication (emhass.api_auth) and its wiring into the web server."""

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
        os.environ.pop(api_auth.API_TOKEN_ENV, None)
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
        secrets = self.write_secrets(yaml.safe_dump({api_auth.API_TOKEN_KEY: token}))
        await api_auth.configure(secrets, None, self.logger)
        self.assertEqual(api_auth.current_token(), token)


class TestNoTokenConfigured(AuthTestBase):
    async def test_routes_behave_as_before(self):
        await api_auth.configure(None, None, self.logger)
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 200)

    async def test_before_configure_runs_nothing_is_enforced(self):
        response = await self.client.get("/api/v1/last-run")
        self.assertEqual(response.status_code, 200)

    @patch("emhass.web_server._action_call_unlocked", new_callable=AsyncMock)
    async def test_actions_need_no_header(self, mock_action):
        mock_action.return_value = "ran"
        await api_auth.configure(None, None, self.logger)
        response = await self.client.post("/action/publish-data", json={})
        self.assertEqual(response.status_code, 200)
        mock_action.assert_awaited_once()

    async def test_login_page_redirects_home(self):
        await api_auth.configure(None, None, self.logger)
        response = await self.client.get("/login")
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["Location"], "./")


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
        self.assertIn("Path=/", cookie)
        self.assertNotIn("Secure", cookie)  # plain http in the test client
        self.assertNotIn(TOKEN, cookie)

    async def test_cookie_is_secure_over_https(self):
        response = await self.client.post("/login", form={"token": TOKEN}, scheme="https")
        self.assertIn("Secure", response.headers["Set-Cookie"])

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
        env = {api_auth.API_TOKEN_ENV: "from-env-0123456789"}
        self.assertEqual(await api_auth.load_api_token(None, None, env), "from-env-0123456789")
        self.assertEqual(await api_auth.load_api_token(None, options, env), "from-options-0123456")
        self.assertEqual(
            await api_auth.load_api_token(secrets, options, env), "from-secrets-0123456"
        )

    async def test_empty_values_mean_not_set(self):
        options = self.write_options('{"emhass_api_token": ""}')
        secrets = self.write_secrets("emhass_api_token: '   '\nhass_url: http://x/\n")
        self.assertIsNone(await api_auth.load_api_token(secrets, options, {}))

    async def test_empty_later_source_keeps_earlier_token(self):
        env = {api_auth.API_TOKEN_ENV: "e" * 20}
        for options_text, secrets_text in (
            ('{"emhass_api_token": ""}', "emhass_api_token: ''\n"),
            ('{"emhass_api_token": "   "}', "emhass_api_token: '   '\n"),
            ('{"emhass_api_token": null}', "emhass_api_token: null\n"),
            ("{}", "emhass_api_token: ~\n"),
            ("{}", "hass_url: http://x/\n"),
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
        os.environ[api_auth.API_TOKEN_ENV] = TOKEN
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


class TestStartupWarning(AuthTestBase):
    async def test_warns_when_open_on_the_network(self):
        await api_auth.configure(None, None, self.logger)
        for host in ("0.0.0.0", "192.0.2.10", "::", "emhass.lan", ""):
            with self.subTest(host=host):
                with self.assertLogs(self.logger, level="WARNING") as logs:
                    api_auth.log_status(self.logger, host)
                self.assertIn("No API token is set", logs.output[0])

    async def test_quiet_on_loopback(self):
        await api_auth.configure(None, None, self.logger)
        for host in ("127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost", "LOCALHOST"):
            with self.subTest(host=host):
                with self.assertNoLogs(self.logger, level="WARNING"):
                    api_auth.log_status(self.logger, host)

    async def test_says_enabled_when_a_token_is_set(self):
        await self.enable()
        with self.assertNoLogs(self.logger, level="WARNING"):
            api_auth.log_status(self.logger, "0.0.0.0")
        with self.assertLogs(self.logger, level="INFO") as logs:
            api_auth.log_status(self.logger, "0.0.0.0")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("API authentication is enabled", logs.output[0])

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
        await self.enable("x" * api_auth.MIN_TOKEN_LENGTH)

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
        secrets = self.write_secrets(yaml.safe_dump({api_auth.API_TOKEN_KEY: TOKEN}))
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

    async def test_open_warning_still_logged_when_secrets_setup_fails(self):
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
        self.assertTrue(any("No API token is set" in line for line in logs.output))

    async def test_before_serving_keeps_the_token_loaded_at_import(self):
        # A worker started by Gunicorn after max_requests must not re-read the
        # files: a secrets file missing for a moment would switch auth off there.
        await self.enable()
        with (
            patch.object(web_server, "_configure_api_auth", AsyncMock()) as mock_configure,
            patch.object(web_server, "initialize", AsyncMock()),
        ):
            await web_server.before_serving()
        mock_configure.assert_not_awaited()
        self.assertEqual(api_auth.current_token(), TOKEN)

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
            yaml.safe_dump({api_auth.API_TOKEN_KEY: TOKEN, "hass_url": "http://ha:8123/"})
        )
        _, built = await utils.build_secrets(
            {"data_path": self.dir}, self.logger, secrets_path=secrets, no_response=True
        )
        self.assertNotIn(api_auth.API_TOKEN_KEY, built)
        self.assertEqual(built["hass_url"], "http://ha:8123/")

    def test_token_is_listed_as_config_only(self):
        # Documents the rule: the runtime loop only checks keys in associations.csv,
        # and the token is never read from a body (see the logging test below).
        self.assertIn(api_auth.API_TOKEN_KEY, utils.RUNTIME_DENIED_KEYS)

    def test_redacted(self):
        self.assertEqual(
            api_auth.redacted({api_auth.API_TOKEN_KEY: TOKEN, "a": 1}),
            {api_auth.API_TOKEN_KEY: "***", "a": 1},
        )
        for value in ({"a": 1}, [1, 2], "text", None):
            self.assertEqual(api_auth.redacted(value), value)

    async def test_token_sent_in_a_body_is_not_logged(self):
        body = {api_auth.API_TOKEN_KEY: TOKEN, "prediction_horizon": 10}
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


class TestWithPartOneHooks(AuthTestBase):
    """The auth hook next to part 1's cross-site check and response headers."""

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
