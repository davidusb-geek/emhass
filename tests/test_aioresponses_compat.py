"""Regression tests for tests/aioresponses_compat.py (aioresponses 0.7.9 vs aiohttp >= 3.13)."""

import unittest

import aiohttp
from aioresponses import core

from tests import aioresponses_compat
from tests.aioresponses_compat import aioresponses, make_client_response, requires_stream_writer


class _OldResponse:
    """Stand-in for an aiohttp ClientResponse that has no stream_writer argument."""

    def __init__(self, method, url, *, writer, session):
        self.kwargs = {"writer": writer, "session": session}


class _NewResponse:
    """Stand-in for an aiohttp ClientResponse that requires stream_writer."""

    def __init__(self, method, url, *, writer, session, stream_writer):
        self.kwargs = {"writer": writer, "session": session, "stream_writer": stream_writer}


class _OptionalWriterResponse:
    """A stream_writer argument with a default does not need to be supplied."""

    def __init__(self, method, url, *, writer, stream_writer=None):
        self.kwargs = {"writer": writer, "stream_writer": stream_writer}


class TestResponseClassSelection(unittest.TestCase):
    def test_requirement_is_detected_from_the_signature(self):
        self.assertFalse(requires_stream_writer(_OldResponse))
        self.assertTrue(requires_stream_writer(_NewResponse))
        self.assertFalse(requires_stream_writer(_OptionalWriterResponse))

    def test_older_aiohttp_signature_is_left_untouched(self):
        self.assertIs(make_client_response(_OldResponse), _OldResponse)
        self.assertIs(make_client_response(_OptionalWriterResponse), _OptionalWriterResponse)

    def test_current_aiohttp_signature_gets_a_stream_writer(self):
        compat = make_client_response(_NewResponse)
        self.assertIsNot(compat, _NewResponse)
        self.assertTrue(issubclass(compat, _NewResponse))
        response = compat("GET", "http://x/", writer=None, session=None)
        self.assertEqual(response.kwargs["stream_writer"].output_size, 0)

    def test_an_explicit_stream_writer_is_kept(self):
        compat = make_client_response(_NewResponse)
        sentinel = object()
        response = compat("GET", "http://x/", writer=None, session=None, stream_writer=sentinel)
        self.assertIs(response.kwargs["stream_writer"], sentinel)

    def test_installed_aiohttp_is_supported_whichever_signature_it_has(self):
        installed = aiohttp.ClientResponse
        self.assertTrue(issubclass(core.ClientResponse, installed))
        # the patched class must be constructible exactly as aioresponses constructs it
        self.assertEqual(
            requires_stream_writer(installed),
            core.ClientResponse is not installed,
        )

    def test_importing_the_module_installs_the_adapter(self):
        # nothing here calls install(): the import alone must have patched aioresponses
        self.assertIs(core.stream_reader_factory, aioresponses_compat.stream_reader_factory)
        self.assertTrue(issubclass(core.ClientResponse, aiohttp.ClientResponse))

    def test_install_is_idempotent(self):
        # restore what the import installed, so this test cannot mask a missing install
        self.addCleanup(setattr, core, "ClientResponse", core.ClientResponse)
        self.addCleanup(setattr, core, "stream_reader_factory", core.stream_reader_factory)
        aioresponses_compat.install()
        first = core.ClientResponse
        aioresponses_compat.install()
        self.assertTrue(issubclass(core.ClientResponse, aiohttp.ClientResponse))
        self.assertEqual(core.ClientResponse.__name__, first.__name__)
        self.assertIs(core.stream_reader_factory, aioresponses_compat.stream_reader_factory)


class TestMockedRequests(unittest.IsolatedAsyncioTestCase):
    """Real aiohttp client calls against aioresponses on the installed aiohttp."""

    async def _get(self, **mock_kwargs):
        with aioresponses() as mocked:
            mocked.get("http://compat.test/data", **mock_kwargs)
            async with aiohttp.ClientSession() as session:
                async with session.get("http://compat.test/data") as response:
                    return response.status, await response.read(), response

    async def test_small_json_payload(self):
        with aioresponses() as mocked:
            mocked.get("http://compat.test/data", payload={"a": 1})
            async with aiohttp.ClientSession() as session:
                async with session.get("http://compat.test/data") as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(await response.json(), {"a": 1})

    async def test_error_status_is_returned_not_a_construction_failure(self):
        status, body, response = await self._get(status=401, body="denied")
        self.assertEqual(status, 401)
        self.assertEqual(body, b"denied")
        with self.assertRaises(aiohttp.ClientResponseError):
            response.raise_for_status()

    async def test_bodies_around_and_above_the_flow_control_limit(self):
        # 2**16 limit -> high-water mark 2**17; sizes below, at, above and far above it
        for size in (10, 2**16, 2**17 - 1, 2**17, 2**17 + 1, 300_000, 5_000_000):
            with self.subTest(size=size):
                payload = b"x" * size
                status, body, _ = await self._get(body=payload)
                self.assertEqual(status, 200)
                self.assertEqual(body, payload)

    async def test_large_json_body_is_parsed(self):
        data = {"values": list(range(60_000))}
        with aioresponses() as mocked:
            mocked.get("http://compat.test/data", payload=data)
            async with aiohttp.ClientSession() as session:
                async with session.get("http://compat.test/data") as response:
                    self.assertEqual(await response.json(), data)

    async def test_large_body_can_be_streamed_in_chunks(self):
        payload = bytes(range(256)) * 2000  # 512,000 bytes
        with aioresponses() as mocked:
            mocked.get("http://compat.test/data", body=payload)
            async with aiohttp.ClientSession() as session:
                async with session.get("http://compat.test/data") as response:
                    received = b"".join(
                        [chunk async for chunk in response.content.iter_chunked(4096)]
                    )
        self.assertEqual(received, payload)

    async def test_repeated_responses_and_request_recording(self):
        with aioresponses() as mocked:
            mocked.get("http://compat.test/data", status=429, repeat=True)
            async with aiohttp.ClientSession() as session:
                for _ in range(3):
                    async with session.get("http://compat.test/data") as response:
                        self.assertEqual(response.status, 429)
            self.assertEqual(sum(len(v) for v in mocked.requests.values()), 3)


if __name__ == "__main__":
    unittest.main()
