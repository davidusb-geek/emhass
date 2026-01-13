import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import orjson

from emhass.websocket_client import AsyncWebSocketClient, AuthenticationError, RequestError

# Disable logging for tests
logging.basicConfig(level=logging.CRITICAL)


class TestWebSocketClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hass_url = "http://localhost:8123"
        self.token = "test_long_lived_token"
        self.client = AsyncWebSocketClient(self.hass_url, self.token)
        # Mock logger
        self.client.logger = MagicMock()

        # Create a queue for controlling message flow
        self.response_queue = asyncio.Queue()

    def test_init(self):
        self.assertEqual(self.client.hass_url, self.hass_url)
        self.assertEqual(self.client.token, self.token)
        self.assertEqual(self.client.websocket_url, "ws://localhost:8123/api/websocket")

    def _setup_mock_ws(self, mock_connect):
        """Helper to setup the websocket mock with a Queue-based recv."""
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        # The recv method will simply pull from our queue
        # This forces the client to wait until WE put something in the queue
        mock_ws.recv.side_effect = self.response_queue.get
        mock_connect.return_value = mock_ws
        return mock_ws

    @patch("emhass.websocket_client.websockets.connect", new_callable=AsyncMock)
    async def test_startup_success(self, mock_connect):
        mock_ws = self._setup_mock_ws(mock_connect)
        # Queue the auth handshake messages
        await self.response_queue.put(orjson.dumps({"type": "auth_required"}).decode())
        await self.response_queue.put(
            orjson.dumps({"type": "auth_ok", "ha_version": "2023.1.1"}).decode()
        )
        # Run startup
        await self.client.startup()
        self.assertTrue(self.client.connected)
        self.assertTrue(self.client._authenticated)
        # Verify auth message was sent
        expected_auth = orjson.dumps({"type": "auth", "access_token": self.token}).decode()
        mock_ws.send.assert_called_with(expected_auth)
        await self.client.shutdown()

    @patch("emhass.websocket_client.websockets.connect", new_callable=AsyncMock)
    async def test_startup_auth_fail(self, mock_connect):
        self._setup_mock_ws(mock_connect)
        # Queue the fail sequence
        await self.response_queue.put(orjson.dumps({"type": "auth_required"}).decode())
        await self.response_queue.put(
            orjson.dumps({"type": "auth_invalid", "message": "Bad token"}).decode()
        )
        with self.assertRaises(AuthenticationError):
            await self.client.startup()
        self.assertFalse(self.client.connected)

    @patch("emhass.websocket_client.websockets.connect", new_callable=AsyncMock)
    async def test_send_and_receive(self, mock_connect):
        mock_ws = self._setup_mock_ws(mock_connect)
        # Auth Handshake
        await self.response_queue.put(orjson.dumps({"type": "auth_required"}).decode())
        await self.response_queue.put(orjson.dumps({"type": "auth_ok"}).decode())
        await self.client.startup()
        # Prepare the result for the NEXT command
        # We put this in the queue BEFORE calling send, but because the listener loop
        # is currently waiting on the empty queue, it won't consume it instantly
        # until we yield control.
        await self.response_queue.put(
            orjson.dumps(
                {"id": 1, "type": "result", "success": True, "result": {"some": "data"}}
            ).decode()
        )
        # Send command
        result = await self.client.send("get_config")
        self.assertEqual(result, {"some": "data"})
        # Verify ID 1 was used
        call_args = mock_ws.send.call_args_list
        # call_args[0] is auth, call_args[1] is get_config
        sent_json = orjson.loads(call_args[1][0][0])
        self.assertEqual(sent_json["id"], 1)
        await self.client.shutdown()

    @patch("emhass.websocket_client.websockets.connect", new_callable=AsyncMock)
    async def test_convenience_methods(self, mock_connect):
        self._setup_mock_ws(mock_connect)
        # Auth Phase
        await self.response_queue.put(orjson.dumps({"type": "auth_required"}).decode())
        await self.response_queue.put(orjson.dumps({"type": "auth_ok"}).decode())
        await self.client.startup()
        # Test get_states (ID 1)
        # Queue result
        await self.response_queue.put(
            orjson.dumps(
                {
                    "id": 1,
                    "type": "result",
                    "success": True,
                    "result": [{"entity_id": "sensor.test", "state": "10"}],
                }
            ).decode()
        )
        states = await self.client.get_states()
        self.assertEqual(states[0]["entity_id"], "sensor.test")
        # Test get_state (ID 2)
        # Note: get_state calls get_states internally, so it consumes an ID
        await self.response_queue.put(
            orjson.dumps(
                {
                    "id": 2,
                    "type": "result",
                    "success": True,
                    "result": [{"entity_id": "sensor.test", "state": "10"}],
                }
            ).decode()
        )
        state = await self.client.get_state("sensor.test")
        self.assertEqual(state["state"], "10")
        # Test call_service (ID 3)
        await self.response_queue.put(
            orjson.dumps({"id": 3, "type": "result", "success": True, "result": None}).decode()
        )
        await self.client.call_service("switch", "turn_on")
        await self.client.shutdown()

    @patch("emhass.websocket_client.websockets.connect", new_callable=AsyncMock)
    async def test_send_propagates_error_result(self, mock_connect):
        """Queued response with success: False should surface as error from send()."""
        self._setup_mock_ws(mock_connect)
        # Auth Handshake
        await self.response_queue.put(orjson.dumps({"type": "auth_required"}).decode())
        await self.response_queue.put(
            orjson.dumps({"type": "auth_ok", "ha_version": "2023.1.1"}).decode()
        )
        await self.client.startup()
        # Queue a FAILED result for ID 1
        error_payload = {
            "id": 1,
            "type": "result",
            "success": False,
            "error": {"code": "invalid_format", "message": "Invalid JSON"},
        }
        await self.response_queue.put(orjson.dumps(error_payload).decode())
        # Sending should raise RequestError
        with self.assertRaises(RequestError):
            await self.client.send("test_command")
        await self.client.shutdown()

    @patch("emhass.websocket_client.websockets.connect", new_callable=AsyncMock)
    async def test_send_skips_non_matching_messages(self, mock_connect):
        """Listener should ignore unrelated messages until matching response arrives."""
        self._setup_mock_ws(mock_connect)
        # Auth Handshake
        await self.response_queue.put(orjson.dumps({"type": "auth_required"}).decode())
        await self.response_queue.put(orjson.dumps({"type": "auth_ok"}).decode())
        await self.client.startup()
        # Queue a "noise" message (ID 999) followed by the Real Response (ID 1)
        noise_payload = {"id": 999, "type": "result", "success": True, "result": "noise"}
        real_payload = {"id": 1, "type": "result", "success": True, "result": {"foo": "bar"}}
        await self.response_queue.put(orjson.dumps(noise_payload).decode())
        await self.response_queue.put(orjson.dumps(real_payload).decode())
        # Send command (ID 1)
        # The client should implicitly skip ID 999 and return ID 1
        result = await self.client.send("test_command")
        self.assertEqual(result, {"foo": "bar"})
        await self.client.shutdown()


if __name__ == "__main__":
    unittest.main()
