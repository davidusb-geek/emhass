"""The example hass_url never receives a Home Assistant token.

The example is the default hass_url and the one in the docs and in
secrets_emhass(example).yaml: https://homeassistant.invalid:8123/ now, and the DuckDNS
name https://myhass.duckdns.org/ before, which existing installs may still have. Every
test here mocks the network: aiohttp through aioresponses (an unmatched
request fails without connecting) and the WebSocket client's connect step through a mock.
A name lookup of an example host is refused in this module (socket.getaddrinfo is
patched), so nothing reaches an example host even if a mock stops matching.
"""

import json
import os
import pathlib
import re
import socket
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import pandas as pd
import yaml

from emhass import connection_manager, utils
from emhass.connection_manager import (
    OLD_PLACEHOLDER_HASS_URL,
    PLACEHOLDER_HASS_URL,
    PLACEHOLDER_TOKEN,
    is_placeholder_hass_url,
    is_refused_placeholder,
    is_token_for_placeholder,
)
from emhass.retrieve_hass import RetrieveHass
from emhass.websocket_client import ConnectionError
from tests.aioresponses_compat import aioresponses

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)

REAL_URL = "http://homeassistant.local:8123/"
SUPERVISOR_URL = "http://supervisor/core/api"
# The current example and the older one; both must be refused the same way.
EXAMPLE_URLS = (PLACEHOLDER_HASS_URL, OLD_PLACEHOLDER_HASS_URL)

_real_getaddrinfo = socket.getaddrinfo
_lookup_guard = None


# Literal names, so the guard still holds if is_placeholder_hass_url itself regresses.
_EXAMPLE_HOSTS = ("homeassistant.invalid", "myhass.duckdns.org")


def _no_example_lookup(host, *args, **kwargs):
    name = host.decode() if isinstance(host, bytes) else host
    if name is not None:
        lowered = str(name).casefold().rstrip(".")
        if any(lowered == h or lowered.endswith("." + h) for h in _EXAMPLE_HOSTS) or (
            is_placeholder_hass_url(str(name))
        ):
            raise OSError("test guard: the example host is never looked up")
    return _real_getaddrinfo(host, *args, **kwargs)


def setUpModule():
    global _lookup_guard
    _lookup_guard = patch("socket.getaddrinfo", _no_example_lookup)
    _lookup_guard.start()


def tearDownModule():
    _lookup_guard.stop()


class TestLookupGuard(unittest.TestCase):
    def test_the_example_host_is_never_looked_up(self):
        # The guard must be in place, and it is called directly, so this test can never
        # make a real lookup even if setUpModule did not run.
        self.assertIs(socket.getaddrinfo, _no_example_lookup)
        for host in ("homeassistant.invalid", "myhass.duckdns.org"):
            with self.subTest(host=host), self.assertRaisesRegex(OSError, "test guard"):
                _no_example_lookup(host, 443)


class TestIsPlaceholderHassUrl(unittest.TestCase):
    def test_the_example_urls(self):
        self.assertEqual(PLACEHOLDER_HASS_URL, "https://homeassistant.invalid:8123/")
        self.assertEqual(OLD_PLACEHOLDER_HASS_URL, "https://myhass.duckdns.org/")

    def test_matches_the_current_example_host_in_any_spelling(self):
        for url in (
            PLACEHOLDER_HASS_URL,
            "https://homeassistant.invalid/",
            "http://homeassistant.invalid:8123/",
            "http://homeassistant.invalid:8123/api",
            " HTTPS://HomeAssistant.INVALID:8123/ ",
            "https://homeassistant.invalid.:8123/",
            "https://user@homeassistant.invalid:8123/",
            "https://sub.homeassistant.invalid/",
            "homeassistant.invalid:8123",
            "//homeassistant.invalid/",
            "https://homeassistant。invalid:8123/",
            "https://homeassistant．invalid/",
            "https://home​assistant.invalid/",
            "https://homeassistant%2Einvalid/",
            "https://x%2F:p@homeassistant.invalid/",
            "https://[homeassistant.invalid/",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_placeholder_hass_url(url))

    def test_matches_the_older_example_host_in_any_spelling(self):
        for url in (
            OLD_PLACEHOLDER_HASS_URL,
            "https://myhass.duckdns.org",
            "http://myhass.duckdns.org/",
            " HTTPS://MyHass.DuckDNS.org/ ",
            "https://myhass.duckdns.org:8123/",
            "https://myhass.duckdns.org./",
            "https://user@myhass.duckdns.org/",
            "https://sub.myhass.duckdns.org/",
            "myhass.duckdns.org",
            "myhass.duckdns.org:8123/",
            "//myhass.duckdns.org/",
            # Spellings the HTTP and WebSocket clients still send to that host
            "https://myhass\u3002duckdns\u3002org/",
            "https://myhass\uff0educkdns\uff0eorg/",
            "https://myhass\uff61duckdns.org/",
            "https://\uff4d\uff59\uff48\uff41\uff53\uff53.duckdns.org/",
            "https://myhass.duckdns.org\u3002/",
            "https://myhass.duckdns\u00ad.org/",
            "https://my\u200bhass.duckdns.org/",
            "https://myhass%2Educkdns.org/",
            # Mapped to nothing or folded by Python's IDNA codec (websockets, urllib)
            "https://myha\u00df.duckdns.org/",
            "https://my\u034fhass.duckdns.org/",
            "https://myhass\ufe0f.duckdns.org/",
            "https://my\u1806hass.duckdns.org/",
            "https://myhass\u180b.duckdns.org/",
            # Percent-encoded characters in the user part do not move the host
            "https://x%2F:p@myhass.duckdns.org/",
            "https://x%3F:p@myhass.duckdns.org/",
            "https://x%23:p@myhass.duckdns.org/",
            # Unparsable, but naming the host
            "https://[myhass.duckdns.org/",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_placeholder_hass_url(url))

    def test_matches_a_spelling_only_the_idna_package_maps_to_the_host(self):
        # aiohttp encodes host names with the idna package (UTS 46), which knows newer
        # letters than Python's NFKC. U+A7F1 maps to "s" in the idna releases that know it.
        idna = connection_manager.idna  # the encoder the check itself uses
        if idna is None:
            self.skipTest("the idna package is not installed")
        host = "myha\ua7f1s.duckdns.org"
        try:
            mapped = idna.encode(host, uts46=True).decode("ascii")
        except idna.IDNAError:
            self.skipTest("this idna release does not map U+A7F1")
        if mapped != "myhass.duckdns.org":
            self.skipTest("this idna release does not map U+A7F1 to s")
        self.assertTrue(is_placeholder_hass_url(f"https://{host}/"))

    def test_other_urls_do_not_match(self):
        for url in (
            None,
            "",
            "empty",
            SUPERVISOR_URL,
            REAL_URL,
            "https://notmyhass.duckdns.org/",
            "https://myhass.duckdns.org.example.com/",
            "https://example.com/myhass.duckdns.org",
            "https://example.com/?next=myhass.duckdns.org",
            "https://x%40myhass.duckdns.org@example.com/",
            # Other .invalid names are not the example (the tests use several of them).
            "https://myhomeassistant.invalid/",
            "https://homeassistant.invalid.example.com/",
            "https://example.com/homeassistant.invalid",
            "http://ha.invalid:8123/",
            "https://example.invalid/",
            "http://127.0.0.1:8123/",
            "http://[::1]:8123/",
            "http://[::1",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_placeholder_hass_url(url))

    def test_only_a_real_token_counts(self):
        for example in EXAMPLE_URLS:
            with self.subTest(url=example):
                self.assertTrue(is_token_for_placeholder(example, "a-real-token"))
                for token in (None, "", "  ", "empty", PLACEHOLDER_TOKEN):
                    self.assertFalse(is_token_for_placeholder(example, token))
        self.assertFalse(is_token_for_placeholder(REAL_URL, "a-real-token"))

    def test_only_the_example_token_may_go_to_the_example_url(self):
        for example in EXAMPLE_URLS:
            with self.subTest(url=example):
                self.assertFalse(is_refused_placeholder(example, PLACEHOLDER_TOKEN))
                for token in (None, "", "empty", "a-real-token"):
                    self.assertTrue(is_refused_placeholder(example, token))
        self.assertFalse(is_refused_placeholder(REAL_URL, ""))


class _SecretsTestCase(unittest.IsolatedAsyncioTestCase):
    """Helpers for the build_secrets tests; no tests of its own."""

    def setUp(self):
        self.environ = self.enterContext(patch.dict(os.environ))
        for name in ("EMHASS_URL", "EMHASS_KEY", "SUPERVISOR_TOKEN", "TIME_ZONE"):
            self.environ.pop(name, None)
        self.tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.conf = dict(emhass_conf)
        self.conf["data_path"] = self.tmp

    def write(self, name, text):
        path = self.tmp / name
        path.write_text(text)
        return str(path)

    def write_options(self, options):
        # A config_path keeps build_secrets away from /config and /share on the test machine.
        return self.write(
            "options.json", json.dumps({"config_path": str(self.tmp / "config.json"), **options})
        )

    async def build(self, supervisor_config=None, **kwargs):
        kwargs.setdefault("secrets_path", "")
        with aioresponses() as mocked:
            if supervisor_config is not None:
                mocked.get(
                    SUPERVISOR_URL + "/config", status=200, payload=supervisor_config, repeat=True
                )
            _, secrets = await utils.build_secrets(self.conf, logger, **kwargs)
        sent = [str(url) for (_, url) in mocked.requests]
        self.assertFalse(any(is_placeholder_hass_url(url) for url in sent), sent)
        return secrets


class TestBuildSecretsPlaceholder(_SecretsTestCase):
    async def assert_token_dropped(self, **kwargs):
        with self.assertLogs(logger, level="ERROR") as logs:
            secrets = await self.build(**kwargs)
        self.assertTrue(is_placeholder_hass_url(secrets["hass_url"]))
        self.assertEqual(secrets["long_lived_token"], "")
        self.assertTrue(any("hass_url is still the example" in line for line in logs.output))

    async def test_secrets_file_token_without_url(self):
        path = self.write("secrets.yaml", "long_lived_token: secrets.token\n")
        await self.assert_token_dropped(secrets_path=path)

    async def test_secrets_file_with_a_misspelled_url_key(self):
        path = self.write("secrets.yaml", f"hass_ur: {REAL_URL}\nlong_lived_token: secrets.token\n")
        await self.assert_token_dropped(secrets_path=path)

    async def test_example_secrets_file_with_only_the_token_changed(self):
        # The secrets file ranks above EMHASS_URL, so its example URL wins over it.
        self.environ["EMHASS_URL"] = REAL_URL
        for example in EXAMPLE_URLS:
            with self.subTest(url=example):
                path = self.write(
                    "secrets.yaml", f"hass_url: {example}\nlong_lived_token: secrets.token\n"
                )
                await self.assert_token_dropped(secrets_path=path)

    async def test_the_shipped_example_file_with_only_the_token_changed(self):
        text = (root / "secrets_emhass(example).yaml").read_text(encoding="utf-8")
        self.assertIn(f"long_lived_token: {PLACEHOLDER_TOKEN}\n", text)
        path = self.write(
            "secrets.yaml",
            text.replace(f"long_lived_token: {PLACEHOLDER_TOKEN}", "long_lived_token: my.token"),
        )
        await self.assert_token_dropped(secrets_path=path)

    def test_the_default_is_the_example_in_the_docs(self):
        # build_secrets, the example secrets file and the docs show the same example URL.
        text = (root / "secrets_emhass(example).yaml").read_text(encoding="utf-8")
        self.assertEqual(yaml.safe_load(text)["hass_url"], PLACEHOLDER_HASS_URL)
        for doc in ("docs/config.md", "docs/installation_methods.md"):
            with self.subTest(doc=doc):
                docs_text = (root / doc).read_text(encoding="utf-8")
                self.assertIn(PLACEHOLDER_HASS_URL, docs_text)
                self.assertNotIn(OLD_PLACEHOLDER_HASS_URL, docs_text)

    async def test_emhass_key_without_emhass_url(self):
        self.environ["EMHASS_KEY"] = "env.key"
        await self.assert_token_dropped(no_response=True)

    async def test_key_argument_without_url_argument(self):
        await self.assert_token_dropped(argument={"key": "arg.key"}, no_response=True)

    async def test_addon_with_emhass_key_and_no_url_uses_the_supervisor(self):
        self.environ["SUPERVISOR_TOKEN"] = "supervisor.token"
        self.environ["EMHASS_KEY"] = "env.key"
        for options in (
            {"hass_url": "", "long_lived_token": "options.token"},
            {"hass_url": "empty", "long_lived_token": "options.token"},
            {},
        ):
            with self.subTest(options=options):
                path = self.write_options(options)
                with self.assertLogs(logger, level="WARNING") as logs:
                    secrets = await self.build(options_path=path, no_response=False)
                self.assertEqual(secrets["hass_url"], SUPERVISOR_URL)
                self.assertEqual(secrets["long_lived_token"], "supervisor.token")
                self.assertTrue(any("Supervisor URL" in line for line in logs.output))

    async def test_addon_option_set_to_the_example_url_uses_the_supervisor(self):
        # Without EMHASS_KEY the add-on asks the Supervisor for the Home Assistant config,
        # which must not go to the example URL with the Supervisor token.
        self.environ["SUPERVISOR_TOKEN"] = "supervisor.token"
        ha_config = {"time_zone": "Asia/Tokyo", "latitude": 1.0, "longitude": 2.0, "elevation": 3.0}
        for options in (
            *({"hass_url": url, "long_lived_token": "options.token"} for url in EXAMPLE_URLS),
            *({"hass_url": url} for url in EXAMPLE_URLS),
        ):
            with self.subTest(options=options):
                path = self.write_options(options)
                with self.assertLogs(logger, level="WARNING") as logs:
                    secrets = await self.build(
                        supervisor_config=ha_config, options_path=path, no_response=False
                    )
                self.assertEqual(secrets["hass_url"], SUPERVISOR_URL)
                self.assertEqual(secrets["long_lived_token"], "supervisor.token")
                self.assertEqual(secrets["time_zone"], "Asia/Tokyo")
                self.assertTrue(any("treated as unset" in line for line in logs.output))

    async def test_supervisor_token_set_by_hand_without_options_is_dropped(self):
        # Outside the add-on (no options file) a SUPERVISOR_TOKEN set by hand is not sent
        # to http://supervisor, with or without EMHASS_KEY: the token is dropped instead.
        self.environ["SUPERVISOR_TOKEN"] = "supervisor.token"
        for key in (None, "env.key"):
            with self.subTest(emhass_key=key):
                if key is None:
                    self.environ.pop("EMHASS_KEY", None)
                else:
                    self.environ["EMHASS_KEY"] = key
                await self.assert_token_dropped(no_response=True)

    async def test_addon_with_no_response_still_uses_the_supervisor(self):
        # no_response skips the options branch, so only the options file being there tells
        # the end of build_secrets that this is the add-on.
        self.environ["SUPERVISOR_TOKEN"] = "supervisor.token"
        path = self.write_options({})
        with self.assertLogs(logger, level="WARNING") as logs:
            secrets = await self.build(options_path=path, no_response=True)
        self.assertEqual(secrets["hass_url"], SUPERVISOR_URL)
        self.assertEqual(secrets["long_lived_token"], "supervisor.token")
        self.assertTrue(any("Supervisor URL" in line for line in logs.output))

    async def test_nothing_set_keeps_the_example_values(self):
        with self.assertNoLogs(logger, level="ERROR"):
            secrets = await self.build(no_response=True)
        self.assertEqual(secrets["hass_url"], PLACEHOLDER_HASS_URL)
        self.assertEqual(secrets["long_lived_token"], PLACEHOLDER_TOKEN)

    async def test_a_real_url_keeps_its_token(self):
        self.environ["EMHASS_URL"] = REAL_URL
        self.environ["EMHASS_KEY"] = "env.key"
        secrets = await self.build(no_response=True)
        self.assertEqual(secrets["hass_url"], REAL_URL)
        self.assertEqual(secrets["long_lived_token"], "env.key")
        secrets = await self.build(argument={"url": REAL_URL, "key": "arg.key"}, no_response=True)
        self.assertEqual(secrets["hass_url"], REAL_URL)
        self.assertEqual(secrets["long_lived_token"], "arg.key")


class TestRequestsNeverCarryATokenToThePlaceholder(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.environ = self.enterContext(patch.dict(os.environ))
        for name in ("EMHASS_URL", "EMHASS_KEY", "SUPERVISOR_TOKEN"):
            self.environ.pop(name, None)

    def make_rh(self, url, token):
        return RetrieveHass(url, token, pd.Timedelta(minutes=30), "UTC", {}, emhass_conf, logger)

    async def test_retrieve_hass_drops_a_real_token(self):
        for example in EXAMPLE_URLS:
            with self.subTest(url=example):
                with self.assertLogs(logger, level="ERROR") as logs:
                    rh = self.make_rh(example, "a-real-token")
                self.assertEqual(rh.long_lived_token, "")
                self.assertTrue(
                    any("hass_url is still the example" in line for line in logs.output)
                )
                await self.assert_nothing_sent(rh)

    async def test_retrieve_hass_sends_nothing_with_an_empty_token(self):
        # What build_secrets hands on after dropping a token set without a URL.
        for example in EXAMPLE_URLS:
            for token in ("", "empty"):
                with self.subTest(url=example, token=token):
                    await self.assert_nothing_sent(self.make_rh(example, token))

    async def assert_nothing_sent(self, rh):
        days_list = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        data_df = pd.Series(
            [1.0], index=pd.date_range(start="2024-01-01", periods=1, freq="30min", tz="UTC")
        )
        with aioresponses() as mocked:
            mocked.get(re.compile(r".*"), status=200, payload=[], repeat=True)
            mocked.post(re.compile(r".*"), status=200, payload={}, repeat=True)
            await rh.get_ha_config()
            self.assertFalse(await rh._get_data_rest_api(days_list, ["sensor.test"]))
            with self.assertLogs(logger, level="ERROR"):
                self.assertFalse(await rh.post_scalar_sensor("sensor.test", 1, {}))
            with self.assertLogs(logger, level="ERROR"):
                response, _ = await rh.post_data(
                    data_df, 0, "sensor.test", "power", "W", "Test", type_var="power"
                )
            self.assertFalse(response.ok)
            self.assertEqual(dict(mocked.requests), {})
        await rh.close()

    async def test_retrieve_hass_keeps_the_example_token(self):
        # The example URL with the example token is an install with nothing set: unchanged.
        rh = self.make_rh(PLACEHOLDER_HASS_URL, PLACEHOLDER_TOKEN)
        self.assertEqual(rh.long_lived_token, PLACEHOLDER_TOKEN)
        with aioresponses() as mocked:
            mocked.post(re.compile(r".*"), status=200, payload={}, repeat=True)
            self.assertTrue(await rh.post_scalar_sensor("sensor.test", 1, {}))
            self.assertEqual(len(mocked.requests), 1)
        await rh.close()

    async def test_websocket_is_not_opened_with_a_real_or_empty_token(self):
        for example, token in (
            (url, token) for url in EXAMPLE_URLS for token in ("a-real-token", "", "empty")
        ):
            with self.subTest(url=example, token=token):
                connect = AsyncMock(side_effect=AssertionError("no connection may be opened"))
                with patch("emhass.websocket_client.AsyncWebSocketClient._connect", connect):
                    with self.assertLogs(logger, level="ERROR") as logs:
                        with self.assertRaises(ConnectionError):
                            await connection_manager._create_and_start_client(
                                example, token, logger
                            )
                connect.assert_not_called()
                # Refused by the example-URL check itself, not by a later one.
                self.assertTrue(
                    any("hass_url is still the example" in line for line in logs.output)
                )

    async def test_websocket_with_the_example_token_is_unchanged(self):
        connect = AsyncMock(side_effect=OSError("mocked, no network"))
        with patch("emhass.websocket_client.AsyncWebSocketClient._connect", connect):
            with self.assertRaises(OSError):
                await connection_manager._create_and_start_client(
                    PLACEHOLDER_HASS_URL, PLACEHOLDER_TOKEN, logger
                )
        connect.assert_called_once()


class TestParticipantSocNow(unittest.TestCase):
    def test_soc_now_sends_nothing_to_the_example_url(self):
        from emhass import participant_api

        for example, token in (
            (url, token) for url in EXAMPLE_URLS for token in ("a-real-token", "empty")
        ):
            conf = {"sensor_battery_state_of_charge": "sensor.soc", "hass_url": example}
            with self.subTest(url=example, token=token):
                with patch(
                    "urllib.request.OpenerDirector.open",
                    side_effect=AssertionError("nothing may be sent"),
                ) as opened:
                    with self.assertLogs(logger, level="ERROR"):
                        self.assertIsNone(
                            participant_api.soc_now({**conf, "long_lived_token": token}, logger)
                        )
                opened.assert_not_called()


class TestBlankHassUrl(_SecretsTestCase):
    """A blank or "empty" hass_url with a token: same handling as the example URL."""

    async def assert_unset_url_token_dropped(self, **kwargs):
        with self.assertLogs(logger, level="ERROR") as logs:
            secrets = await self.build(**kwargs)
        self.assertEqual(secrets["long_lived_token"], "")
        self.assertTrue(any("hass_url is not set" in line for line in logs.output))

    async def test_blank_url_with_a_token_drops_the_token(self):
        for blank in ("", "empty"):
            with self.subTest(source="EMHASS_URL", blank=blank):
                self.environ["EMHASS_URL"] = blank
                self.environ["EMHASS_KEY"] = "env.key"
                await self.assert_unset_url_token_dropped(no_response=True)
            self.environ.pop("EMHASS_URL", None)
            self.environ.pop("EMHASS_KEY", None)
            with self.subTest(source="secrets file", blank=blank):
                path = self.write(
                    "secrets.yaml", f'hass_url: "{blank}"\nlong_lived_token: secrets.token\n'
                )
                await self.assert_unset_url_token_dropped(secrets_path=path)
            with self.subTest(source="arguments", blank=blank):
                await self.assert_unset_url_token_dropped(
                    argument={"url": blank, "key": "arg.key"}, no_response=True
                )

    async def test_addon_with_a_blank_url_uses_the_supervisor(self):
        self.environ["SUPERVISOR_TOKEN"] = "supervisor.token"
        self.environ["EMHASS_KEY"] = "env.key"
        self.environ["EMHASS_URL"] = ""
        path = self.write_options({})
        with self.assertLogs(logger, level="WARNING") as logs:
            secrets = await self.build(options_path=path, no_response=False)
        self.assertEqual(secrets["hass_url"], SUPERVISOR_URL)
        self.assertEqual(secrets["long_lived_token"], "supervisor.token")
        self.assertTrue(any("hass_url is not set" in line for line in logs.output))

    async def test_retrieve_hass_sends_only_the_supervisor_token_without_a_url(self):
        self.environ["SUPERVISOR_TOKEN"] = "supervisor.token"
        for blank in ("", "empty"):
            with self.subTest(blank=blank):
                rh = RetrieveHass(
                    blank, "a-real-token", pd.Timedelta(minutes=30), "UTC", {}, emhass_conf, logger
                )
                with aioresponses() as mocked:
                    with self.assertLogs(logger, level="ERROR") as logs:
                        self.assertTrue(await rh.get_ha_config())
                    self.assertEqual(dict(mocked.requests), {})
                self.assertTrue(any("hass_url is not set" in line for line in logs.output))
                await rh.close()
                rh = RetrieveHass(
                    blank,
                    "supervisor.token",
                    pd.Timedelta(minutes=30),
                    "UTC",
                    {},
                    emhass_conf,
                    logger,
                )
                with aioresponses() as mocked:
                    mocked.get(SUPERVISOR_URL + "/config", status=200, payload={"time_zone": "UTC"})
                    self.assertTrue(await rh.get_ha_config())
                    sent = [
                        call.kwargs["headers"]["Authorization"]
                        for calls in mocked.requests.values()
                        for call in calls
                    ]
                self.assertEqual(sent, ["Bearer supervisor.token"])
                await rh.close()

    def make_rh(self, url, token):
        return RetrieveHass(url, token, pd.Timedelta(minutes=30), "UTC", {}, emhass_conf, logger)

    async def test_supervisor_token_set_by_hand_is_not_sent_without_a_url(self):
        # What build_secrets hands on outside the add-on: no URL and no token. The
        # SUPERVISOR_TOKEN in the environment is only used with the Supervisor URL itself.
        self.environ["SUPERVISOR_TOKEN"] = "supervisor.token"
        days_list = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        for blank in ("", "empty", "  "):
            with self.subTest(blank=blank):
                rh = self.make_rh(blank, "")
                with aioresponses() as mocked:
                    mocked.get(re.compile(r".*"), status=200, payload={}, repeat=True)
                    self.assertTrue(await rh.get_ha_config())
                    with self.assertLogs(logger, level="ERROR"):
                        self.assertFalse(await rh._get_data_rest_api(days_list, ["sensor.test"]))
                    self.assertEqual(dict(mocked.requests), {})
                await rh.close()

    async def test_supervisor_token_is_not_sent_to_a_test_url_without_a_url(self):
        # A test_url on the Supervisor proxy doesn't make up for a missing hass_url: the
        # fallback token still needs hass_url to name the Supervisor.
        self.environ["SUPERVISOR_TOKEN"] = "supervisor.token"
        days_list = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        on_supervisor = "http://supervisor/core/api/history/period/2024-01-01T00:00:00Z"
        for blank in ("", "empty", "  "):
            with self.subTest(blank=blank):
                rh = self.make_rh(blank, "")
                with aioresponses() as mocked:
                    mocked.get(re.compile(r".*"), status=200, payload={}, repeat=True)
                    with self.assertLogs(logger, level="ERROR"):
                        self.assertFalse(
                            await rh._get_data_rest_api(
                                days_list, ["sensor.test"], test_url=on_supervisor
                            )
                        )
                    self.assertEqual(dict(mocked.requests), {})
                await rh.close()

    async def test_blank_url_with_the_example_token_sends_nothing_quietly(self):
        for blank in ("", "empty", "  "):
            with self.subTest(blank=blank):
                rh = self.make_rh(blank, PLACEHOLDER_TOKEN)
                with aioresponses() as mocked:
                    mocked.get(re.compile(r".*"), status=200, payload={}, repeat=True)
                    with self.assertNoLogs(logger, level="ERROR"):
                        self.assertTrue(await rh.get_ha_config())
                    self.assertEqual(dict(mocked.requests), {})
                await rh.close()


if __name__ == "__main__":
    unittest.main()
