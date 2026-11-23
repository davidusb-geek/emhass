"""The example hass_url (https://myhass.duckdns.org/) never receives a Home Assistant token.

It is the default hass_url and the example in the docs and in secrets_emhass(example).yaml,
and anyone can register a DuckDNS name. Every test here mocks the network: aiohttp through
aioresponses (an unmatched request fails without connecting) and the WebSocket client's
connect step through a mock. A name lookup of the example host is refused in this module
(socket.getaddrinfo is patched), so nothing reaches the example host even if a mock stops
matching.
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

from emhass import connection_manager, utils
from emhass.connection_manager import (
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

_real_getaddrinfo = socket.getaddrinfo
_lookup_guard = None


def _no_example_lookup(host, *args, **kwargs):
    name = host.decode() if isinstance(host, bytes) else host
    if name is not None and is_placeholder_hass_url(str(name)):
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
        with self.assertRaisesRegex(OSError, "test guard"):
            _no_example_lookup("myhass.duckdns.org", 443)


class TestIsPlaceholderHassUrl(unittest.TestCase):
    def test_matches_the_example_host_in_any_spelling(self):
        for url in (
            PLACEHOLDER_HASS_URL,
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
            "http://127.0.0.1:8123/",
            "http://[::1]:8123/",
            "http://[::1",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_placeholder_hass_url(url))

    def test_only_a_real_token_counts(self):
        self.assertTrue(is_token_for_placeholder(PLACEHOLDER_HASS_URL, "a-real-token"))
        for token in (None, "", "  ", "empty", PLACEHOLDER_TOKEN):
            with self.subTest(token=token):
                self.assertFalse(is_token_for_placeholder(PLACEHOLDER_HASS_URL, token))
        self.assertFalse(is_token_for_placeholder(REAL_URL, "a-real-token"))

    def test_only_the_example_token_may_go_to_the_example_url(self):
        self.assertFalse(is_refused_placeholder(PLACEHOLDER_HASS_URL, PLACEHOLDER_TOKEN))
        for token in (None, "", "empty", "a-real-token"):
            with self.subTest(token=token):
                self.assertTrue(is_refused_placeholder(PLACEHOLDER_HASS_URL, token))
        self.assertFalse(is_refused_placeholder(REAL_URL, ""))


class TestBuildSecretsPlaceholder(unittest.IsolatedAsyncioTestCase):
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
        path = self.write(
            "secrets.yaml", f"hass_url: {PLACEHOLDER_HASS_URL}\nlong_lived_token: secrets.token\n"
        )
        await self.assert_token_dropped(secrets_path=path)

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
            {"hass_url": PLACEHOLDER_HASS_URL, "long_lived_token": "options.token"},
            {"hass_url": PLACEHOLDER_HASS_URL},
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

    async def test_supervisor_token_without_options_or_url_uses_the_supervisor(self):
        self.environ["SUPERVISOR_TOKEN"] = "supervisor.token"
        secrets = await self.build(no_response=True)
        self.assertEqual(secrets["hass_url"], SUPERVISOR_URL)
        self.assertEqual(secrets["long_lived_token"], "supervisor.token")

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
        with self.assertLogs(logger, level="ERROR") as logs:
            rh = self.make_rh(PLACEHOLDER_HASS_URL, "a-real-token")
        self.assertEqual(rh.long_lived_token, "")
        self.assertTrue(any("hass_url is still the example" in line for line in logs.output))
        await self.assert_nothing_sent(rh)

    async def test_retrieve_hass_sends_nothing_with_an_empty_token(self):
        # What build_secrets hands on after dropping a token set without a URL.
        for token in ("", "empty"):
            with self.subTest(token=token):
                await self.assert_nothing_sent(self.make_rh(PLACEHOLDER_HASS_URL, token))

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
        for token in ("a-real-token", "", "empty"):
            with self.subTest(token=token):
                connect = AsyncMock(side_effect=AssertionError("no connection may be opened"))
                with patch("emhass.websocket_client.AsyncWebSocketClient._connect", connect):
                    with self.assertLogs(logger, level="ERROR") as logs:
                        with self.assertRaises(ConnectionError):
                            await connection_manager._create_and_start_client(
                                PLACEHOLDER_HASS_URL, token, logger
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

        conf = {"sensor_battery_state_of_charge": "sensor.soc", "hass_url": PLACEHOLDER_HASS_URL}
        for token in ("a-real-token", "empty"):
            with self.subTest(token=token):
                with patch(
                    "urllib.request.OpenerDirector.open",
                    side_effect=AssertionError("nothing may be sent"),
                ) as opened:
                    with self.assertLogs(logger, level="ERROR"):
                        self.assertIsNone(
                            participant_api.soc_now({**conf, "long_lived_token": token}, logger)
                        )
                opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
