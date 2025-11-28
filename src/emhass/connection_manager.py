import asyncio
import logging

from emhass.websocket_client import AsyncWebSocketClient, ConnectionError

_global_client: AsyncWebSocketClient | None = None
_lock = asyncio.Lock()


async def get_websocket_client(
    hass_url: str, token: str, logger: logging.Logger | None = None
) -> AsyncWebSocketClient:
    """
    Get or create a global WebSocket client connection.

    :param hass_url: The Home Assistant URL
    :type hass_url: str
    :param token: Long-lived token for authentication
    :type token: str
    :param logger: Logger instance for logging
    :type logger: logging.Logger | None
    :return: Connected AsyncWebSocketClient instance
    :rtype: AsyncWebSocketClient
    :raises ConnectionError: If connection cannot be established
    """
    global _global_client

    async with _lock:
        if _global_client is None:
            if logger:
                logger.debug(f"Creating new WebSocket client for {hass_url}")
            _global_client = AsyncWebSocketClient(
                hass_url=hass_url, long_lived_token=token, logger=logger
            )
            try:
                await asyncio.wait_for(_global_client.startup(), timeout=15.0)
                if logger:
                    logger.info("WebSocket client started successfully")
            except TimeoutError:
                if logger:
                    logger.error("WebSocket startup timed out")
                _global_client = None
                raise ConnectionError("WebSocket startup timed out") from None
            except Exception as e:
                if logger:
                    logger.error(f"WebSocket startup failed: {e}")
                _global_client = None
                raise
        elif not _global_client.connected:
            if logger:
                logger.debug("WebSocket client exists but not connected, attempting reconnect")
            try:
                await asyncio.wait_for(_global_client.reconnect(), timeout=15.0)
                if logger:
                    logger.info("WebSocket client reconnected successfully")
            except TimeoutError:
                if logger:
                    logger.error("WebSocket reconnect timed out")
                _global_client = None
                raise ConnectionError("WebSocket reconnect timed out") from None
            except Exception as e:
                if logger:
                    logger.error(f"WebSocket reconnect failed: {e}")
                _global_client = None
                raise
        else:
            if logger:
                logger.debug("Using existing connected WebSocket client")

    return _global_client


async def close_global_connection():
    """
    Close the global WebSocket connection if it exists.
    """
    global _global_client

    async with _lock:
        if _global_client is not None:
            try:
                await _global_client.shutdown()
            except Exception:
                # Ignore errors during shutdown
                pass
            finally:
                _global_client = None


def is_connected() -> bool:
    """
    Check if the global WebSocket client is connected.

    :return: True if connected, False otherwise
    :rtype: bool
    """
    return _global_client is not None and _global_client.connected
