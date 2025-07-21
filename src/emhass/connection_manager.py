import asyncio
import logging

from emhass.websocket_client import AsyncWebSocketClient, ConnectionError

_global_client: AsyncWebSocketClient | None = None
_lock = asyncio.Lock()

async def get_websocket_client(hass_url: str, token: str, logger: logging.Logger | None = None) -> AsyncWebSocketClient:
    global _global_client

    async with _lock:
        if _global_client is None:
            _global_client = AsyncWebSocketClient(hass_url=hass_url, long_lived_token=token, logger=logger)
            try:
                await asyncio.wait_for(_global_client.startup(), timeout=15.0)
            except TimeoutError:
                if logger:
                    logger.error("WebSocket startup timed out")
                _global_client = None
                raise ConnectionError("WebSocket startup timed out")
            except Exception as e:
                if logger:
                    logger.error(f"WebSocket startup failed: {e}")
                _global_client = None
                raise
        elif not _global_client.connected:
            try:
                await asyncio.wait_for(_global_client.reconnect(), timeout=15.0)
            except TimeoutError:
                if logger:
                    logger.error("WebSocket reconnect timed out")
                _global_client = None
                raise ConnectionError("WebSocket reconnect timed out")
            except Exception as e:
                if logger:
                    logger.error(f"WebSocket reconnect failed: {e}")
                _global_client = None
                raise

    return _global_client

async def close_global_connection():
    global _global_client

    if _global_client is not None:
        await _global_client.shutdown()
        _global_client = None

def is_connected() -> bool:
    return _global_client is not None and _global_client.connected
