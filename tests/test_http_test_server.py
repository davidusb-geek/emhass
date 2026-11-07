"""The ``HttpTestServer`` fixture: real HTTP(S), scoped interception, network isolation."""

import asyncio
import re
import socket
import time
import types
import unittest

import aiohttp

from tests.http_test_server import HttpTestServer, NetworkBlocked


def consumer():
    """A stand-in for a production module that does ``import aiohttp``."""
    return types.SimpleNamespace(aiohttp=aiohttp)


async def fetch(module, url, **kwargs):
    """GET through the module's own ``aiohttp`` reference; return (status, text)."""
    timeout = kwargs.pop("timeout", None)
    async with module.aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, **kwargs) as response:
            return response.status, await response.text()


class TestResponses(unittest.IsolatedAsyncioTestCase):
    async def test_requests_are_recorded_and_responses_used_once_in_order(self):
        module = consumer()
        any_url = re.compile(r"http://plain\.test/.*")
        async with HttpTestServer(http=("plain.test",)) as server:
            server.get(any_url, status=503)
            server.get(any_url, payload={"ok": True})
            server.get(any_url, payload="always", repeat=True)
            with server.intercept(module):
                seen = [await fetch(module, "http://plain.test/a?x=1") for _ in range(4)]
        self.assertEqual(
            seen, [(503, ""), (200, '{"ok": true}'), (200, '"always"'), (200, '"always"')]
        )
        self.assertEqual(server.requests[0], ("GET", "http://plain.test/a?x=1"))

    async def test_https_is_verified_against_the_test_ca(self):
        module = consumer()
        async with HttpTestServer(http=("plain.test",), https=("secure.test",)) as server:
            server.get("https://secure.test/v", payload="tls")
            with server.intercept(module):
                self.assertEqual(await fetch(module, "https://secure.test/v"), (200, '"tls"'))

    async def test_an_untrusted_certificate_fails_verification_unless_ssl_is_off(self):
        module = consumer()
        async with HttpTestServer(https=("secure.test",), trusted=False) as server:
            server.get("https://secure.test/v", payload="tls")
            with server.intercept(module):
                with self.assertRaises(aiohttp.ClientConnectorCertificateError):
                    await fetch(module, "https://secure.test/v")
                self.assertEqual(
                    await fetch(module, "https://secure.test/v", ssl=False), (200, '"tls"')
                )
        self.assertEqual(len(server.requests), 1, "only the unverified request got through")

    async def test_a_slow_server_times_out_the_client(self):
        module = consumer()
        async with HttpTestServer(http=("plain.test",)) as server:
            server.get("http://plain.test/slow", delay=30)
            started = time.monotonic()
            with server.intercept(module):
                with self.assertRaises(TimeoutError):
                    await fetch(
                        module, "http://plain.test/slow", timeout=aiohttp.ClientTimeout(total=0.2)
                    )
        self.assertLess(time.monotonic() - started, 5, "teardown must not wait for the delay")

    async def test_an_unmatched_request_drops_the_connection_and_fails_the_block(self):
        module = consumer()
        with self.assertRaisesRegex(AssertionError, "no declared response matched"):
            async with HttpTestServer(http=("plain.test",)) as server:
                with server.intercept(module):
                    with self.assertRaises(aiohttp.ClientConnectionError):
                        await fetch(module, "http://plain.test/undeclared")
        self.assertEqual(set(server.unmatched), {"GET http://plain.test/undeclared"})

    async def test_a_request_other_than_get_is_recorded_and_fails_the_block(self):
        module = consumer()
        with self.assertRaisesRegex(AssertionError, "no declared response matched"):
            async with HttpTestServer(http=("plain.test",)) as server:
                server.get("http://plain.test/", payload="ok", repeat=True)
                with server.intercept(module):
                    async with module.aiohttp.ClientSession() as session:
                        with self.assertRaises(aiohttp.ClientConnectionError):
                            await session.post("http://plain.test/")
        self.assertEqual(server.requests[0], ("POST", "http://plain.test/"))

    def test_a_host_that_cannot_be_isolated_is_refused(self):
        for label, kwargs in {
            "IP literal": {"http": ("127.0.0.1",)},
            "both schemes": {"http": ("a.test",), "https": ("a.test",)},
        }.items():
            with self.subTest(label), self.assertRaises(ValueError):
                HttpTestServer(**kwargs)


class TestIsolation(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def session_with_own_connector(module):
        async with aiohttp.TCPConnector() as connector:
            module.aiohttp.ClientSession(connector=connector)

    @staticmethod
    async def call_aiohttp_request(module):
        await module.aiohttp.request("GET", "http://plain.test/")

    async def test_traffic_that_would_leave_the_test_server_is_refused(self):
        module = consumer()
        attempts = {
            "undeclared host": lambda: fetch(module, "http://elsewhere.invalid/"),
            "IP literal": lambda: fetch(module, "http://127.0.0.1:9/"),
            "own connector": lambda: self.session_with_own_connector(module),
            "aiohttp.request": lambda: self.call_aiohttp_request(module),
        }
        for label, attempt in attempts.items():
            with self.subTest(label):
                with self.assertRaises(NetworkBlocked):
                    async with HttpTestServer(http=("plain.test",)) as server:
                        with server.intercept(module):
                            with self.assertRaises(NetworkBlocked):
                                await attempt()
                self.assertEqual(server.requests, [])

    async def test_a_block_that_swallows_the_refusal_still_fails(self):
        module = consumer()
        with self.assertRaises(NetworkBlocked):
            async with HttpTestServer(http=("plain.test",)) as server:
                with server.intercept(module):
                    try:
                        await fetch(module, "http://elsewhere.invalid/")
                    except NetworkBlocked:
                        pass  # production code that catches broadly must not hide it

    async def test_only_the_named_modules_are_patched_and_only_inside_the_block(self):
        inside, outside = consumer(), consumer()
        original = (aiohttp.ClientSession, aiohttp.TCPConnector)
        async with HttpTestServer(http=("plain.test",)) as server:
            with self.assertRaises(RuntimeError):
                with server.intercept(inside):
                    self.assertIsNot(inside.aiohttp, aiohttp)
                    self.assertIs(outside.aiohttp, aiohttp)
                    self.assertEqual((aiohttp.ClientSession, aiohttp.TCPConnector), original)
                    raise RuntimeError("boom")
            self.assertIs(inside.aiohttp, aiohttp, "restored even after a failure")


class TestCleanupConcurrencyAndLeakage(unittest.IsolatedAsyncioTestCase):
    async def test_the_listeners_are_closed_when_the_block_ends(self):
        async with HttpTestServer(http=("plain.test",), https=("secure.test",)) as server:
            ports = [listener.port for listener in server._listeners.values()]
        self.assertEqual(len(ports), 2)
        for port in ports:
            with socket.socket() as probe:  # a listening port cannot be bound again
                probe.bind(("127.0.0.1", port))

    async def test_concurrent_requests_are_all_answered_and_recorded(self):
        module = consumer()
        async with HttpTestServer(http=("plain.test",), https=("secure.test",)) as server:
            server.get(re.compile(r"http://plain\.test/.*"), payload="p", repeat=True)
            server.get(re.compile(r"https://secure\.test/.*"), payload="s", repeat=True)
            with server.intercept(module):
                results = await asyncio.gather(
                    *(
                        fetch(module, f"{scheme}://{host}/{i}")
                        for i in range(20)
                        for scheme, host in (("http", "plain.test"), ("https", "secure.test"))
                    )
                )
        self.assertEqual(len(server.requests), 40)
        self.assertEqual(sorted(text for _, text in results), ['"p"'] * 20 + ['"s"'] * 20)

    async def test_servers_at_once_and_one_after_another_share_no_state(self):
        first, second = consumer(), consumer()
        async with HttpTestServer(http=("a.test",)) as one, HttpTestServer(http=("b.test",)) as two:
            one.get("http://a.test/", payload="one", repeat=True)
            two.get("http://b.test/", payload="two")
            with one.intercept(first), two.intercept(second):
                self.assertEqual(await fetch(first, "http://a.test/"), (200, '"one"'))
                self.assertEqual(await fetch(second, "http://b.test/"), (200, '"two"'))
                with self.assertRaises(NetworkBlocked):
                    await fetch(first, "http://b.test/")
                one.blocked.clear()  # the expected refusal must not fail the block
        async with HttpTestServer(http=("a.test",)) as later:
            self.assertEqual((later.requests, later.unmatched, later.blocked), ([], [], []))
        self.assertEqual([r.url for r in one.requests], ["http://a.test/"])
        self.assertEqual([r.url for r in two.requests], ["http://b.test/"])


if __name__ == "__main__":
    unittest.main()
