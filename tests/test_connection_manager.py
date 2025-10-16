#!/usr/bin/env python3

import asyncio
import pathlib
import unittest
from unittest.mock import AsyncMock, patch

from emhass import utils_async as utils
from emhass.connection_manager import (
    close_global_connection,
    get_websocket_client,
    is_connected,
)
from emhass.websocket_client import AsyncWebSocketClient, ConnectionError

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"

# Create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)


class TestConnectionManager(unittest.IsolatedAsyncioTestCase):
    """Test connection manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.hass_url = "http://homeassistant.local:8123/api"
        self.token = "test_token_123"

        # Reset global client state before each test
        import emhass.connection_manager as cm

        cm._global_client = None

    async def asyncTearDown(self):
        """Clean up after each test."""
        # Ensure global connection is closed
        await close_global_connection()

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_get_websocket_client_first_time(self, mock_client_class):
        """Test getting WebSocket client for the first time."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.connected = True
        mock_client.startup = AsyncMock()
        mock_client_class.return_value = mock_client

        # Get the client
        client = await get_websocket_client(self.hass_url, self.token, logger)

        # Verify client was created and started up
        self.assertEqual(client, mock_client)
        mock_client_class.assert_called_once_with(
            hass_url=self.hass_url, long_lived_token=self.token, logger=logger
        )
        mock_client.startup.assert_called_once()

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_get_websocket_client_reuse_existing(self, mock_client_class):
        """Test reusing existing WebSocket client."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.connected = True
        mock_client.startup = AsyncMock()
        mock_client_class.return_value = mock_client

        # Get the client twice
        client1 = await get_websocket_client(self.hass_url, self.token, logger)
        client2 = await get_websocket_client(self.hass_url, self.token, logger)

        # Verify same client is returned and startup is called only once
        self.assertEqual(client1, client2)
        self.assertEqual(client1, mock_client)
        mock_client_class.assert_called_once()  # Should only be called once
        mock_client.startup.assert_called_once()  # Should only be called once

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_get_websocket_client_reconnect_when_disconnected(
        self, mock_client_class
    ):
        """Test reconnecting when existing client is disconnected."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.startup = AsyncMock()
        mock_client.reconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        # First call - client is connected
        mock_client.connected = True
        client1 = await get_websocket_client(self.hass_url, self.token, logger)

        # Second call - client is now disconnected
        mock_client.connected = False
        client2 = await get_websocket_client(self.hass_url, self.token, logger)

        # Verify same client is returned but reconnect was called
        self.assertEqual(client1, client2)
        mock_client.startup.assert_called_once()
        mock_client.reconnect.assert_called_once()

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_get_websocket_client_startup_timeout(self, mock_client_class):
        """Test handling startup timeout."""
        # Mock the AsyncWebSocketClient with startup timeout
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.startup.side_effect = TimeoutError()
        mock_client_class.return_value = mock_client

        # Should raise ConnectionError due to timeout
        with self.assertRaises(ConnectionError) as cm:
            await get_websocket_client(self.hass_url, self.token, logger)

        self.assertIn("startup timed out", str(cm.exception))

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_get_websocket_client_startup_failure(self, mock_client_class):
        """Test handling startup failure."""
        # Mock the AsyncWebSocketClient with startup failure
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.startup.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        # Should raise the original exception
        with self.assertRaises(Exception) as cm:
            await get_websocket_client(self.hass_url, self.token, logger)

        self.assertIn("Connection failed", str(cm.exception))

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_get_websocket_client_reconnect_timeout(self, mock_client_class):
        """Test handling reconnect timeout."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.startup = AsyncMock()
        mock_client.reconnect.side_effect = TimeoutError()
        mock_client_class.return_value = mock_client

        # First call - successful startup
        mock_client.connected = True
        await get_websocket_client(self.hass_url, self.token, logger)

        # Second call - reconnect timeout
        mock_client.connected = False
        with self.assertRaises(ConnectionError) as cm:
            await get_websocket_client(self.hass_url, self.token, logger)

        self.assertIn("reconnect timed out", str(cm.exception))

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_get_websocket_client_reconnect_failure(self, mock_client_class):
        """Test handling reconnect failure."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.startup = AsyncMock()
        mock_client.reconnect.side_effect = Exception("Reconnect failed")
        mock_client_class.return_value = mock_client

        # First call - successful startup
        mock_client.connected = True
        await get_websocket_client(self.hass_url, self.token, logger)

        # Second call - reconnect failure
        mock_client.connected = False
        with self.assertRaises(Exception) as cm:
            await get_websocket_client(self.hass_url, self.token, logger)

        self.assertIn("Reconnect failed", str(cm.exception))

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_close_global_connection(self, mock_client_class):
        """Test closing the global connection."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.connected = True
        mock_client.startup = AsyncMock()
        mock_client.shutdown = AsyncMock()
        mock_client_class.return_value = mock_client

        # Get a client first
        await get_websocket_client(self.hass_url, self.token, logger)

        # Close the connection
        await close_global_connection()

        # Verify shutdown was called
        mock_client.shutdown.assert_called_once()

        # Verify global client is reset
        import emhass.connection_manager as cm

        self.assertIsNone(cm._global_client)

    async def test_close_global_connection_when_none(self):
        """Test closing global connection when there is no client."""
        # Should not raise exception when no client exists
        await close_global_connection()

        # Verify global client is still None
        import emhass.connection_manager as cm

        self.assertIsNone(cm._global_client)

    def test_is_connected_no_client(self):
        """Test is_connected when no client exists."""
        # Reset global client
        import emhass.connection_manager as cm

        cm._global_client = None

        # Should return False
        self.assertFalse(is_connected())

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_is_connected_with_client(self, mock_client_class):
        """Test is_connected with existing client."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.startup = AsyncMock()
        mock_client_class.return_value = mock_client

        # Test with connected client
        mock_client.connected = True
        await get_websocket_client(self.hass_url, self.token, logger)
        self.assertTrue(is_connected())

        # Test with disconnected client
        mock_client.connected = False
        self.assertFalse(is_connected())

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_concurrent_client_access(self, mock_client_class):
        """Test concurrent access to get_websocket_client."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.connected = True
        mock_client.startup = AsyncMock()
        mock_client_class.return_value = mock_client

        # Create multiple concurrent requests
        async def get_client():
            return await get_websocket_client(self.hass_url, self.token, logger)

        tasks = [get_client() for _ in range(5)]
        clients = await asyncio.gather(*tasks)

        # All should return the same client instance
        for client in clients:
            self.assertEqual(client, mock_client)

        # Client should only be created once
        mock_client_class.assert_called_once()
        mock_client.startup.assert_called_once()

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_get_websocket_client_without_logger(self, mock_client_class):
        """Test getting WebSocket client without logger."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.connected = True
        mock_client.startup = AsyncMock()
        mock_client_class.return_value = mock_client

        # Get the client without logger
        await get_websocket_client(self.hass_url, self.token)

        # Verify client was created with None logger
        mock_client_class.assert_called_once_with(
            hass_url=self.hass_url, long_lived_token=self.token, logger=None
        )

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_client_reset_after_startup_failure(self, mock_client_class):
        """Test that global client is reset after startup failure."""
        # Mock the AsyncWebSocketClient with startup failure
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.startup.side_effect = Exception("Startup failed")
        mock_client_class.return_value = mock_client

        # Should raise exception and reset global client
        with self.assertRaises(Exception) as cm:
            await get_websocket_client(self.hass_url, self.token, logger)

        # Verify global client is reset
        import emhass.connection_manager as cm

        self.assertIsNone(cm._global_client)

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_client_reset_after_reconnect_failure(self, mock_client_class):
        """Test that global client is reset after reconnect failure."""
        # Mock the AsyncWebSocketClient
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.startup = AsyncMock()
        mock_client.reconnect.side_effect = Exception("Reconnect failed")
        mock_client_class.return_value = mock_client

        # First call - successful
        mock_client.connected = True
        await get_websocket_client(self.hass_url, self.token, logger)

        # Second call - reconnect fails
        mock_client.connected = False
        with self.assertRaises(Exception) as cm:
            await get_websocket_client(self.hass_url, self.token, logger)

        # Verify global client is reset
        import emhass.connection_manager as cm

        self.assertIsNone(cm._global_client)

    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_lock_prevents_race_conditions(self, mock_client_class):
        """Test that the async lock prevents race conditions."""
        # Mock the AsyncWebSocketClient with slow startup
        mock_client = AsyncMock(spec=AsyncWebSocketClient)
        mock_client.connected = True
        mock_client_class.return_value = mock_client

        async def slow_startup():
            await asyncio.sleep(0.1)  # Simulate slow startup

        mock_client.startup = slow_startup

        # Start two concurrent requests
        task1 = asyncio.create_task(
            get_websocket_client(self.hass_url, self.token, logger)
        )
        task2 = asyncio.create_task(
            get_websocket_client(self.hass_url, self.token, logger)
        )

        clients = await asyncio.gather(task1, task2)

        # Both should return the same client
        self.assertEqual(clients[0], clients[1])
        # Client should only be created once
        mock_client_class.assert_called_once()


if __name__ == "__main__":
    unittest.main()
