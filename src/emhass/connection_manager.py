import asyncio
import logging
import os

from emhass.websocket_client import AsyncWebSocketClient, ConnectionError

_global_client: AsyncWebSocketClient | None = None
_lock = asyncio.Lock()

# The forms of the Supervisor proxy URL that may receive the Supervisor token (compared
# without a trailing slash).
SUPERVISOR_URLS = frozenset({"http://supervisor/core/api", "https://supervisor/core/api"})

SUPERVISOR_TOKEN_REFUSED = (
    "The Supervisor token is only sent to the Supervisor, and the Home Assistant URL is "
    "another host, so the token will not be sent. Give a long-lived access token for that "
    "URL in the add-on options (long_lived_token), the secrets file or --key, not in "
    "SUPERVISOR_TOKEN."
)


def is_supervisor_url(url: str | None) -> bool:
    """True when ``url`` is the Supervisor proxy, or unset (which means the Supervisor)."""
    if url is None or not str(url).strip() or str(url).strip() == "empty":
        return True
    return str(url).strip().rstrip("/").lower() in SUPERVISOR_URLS


def canonical_supervisor_url(url: str | None) -> str | None:
    """The Supervisor proxy URL in the exact form the request code builds on.

    A Supervisor form written with other case, spaces or a trailing slash is returned as
    ``http://supervisor/core/api`` or ``https://supervisor/core/api``. Anything else,
    including an unset or ``"empty"`` URL, is returned unchanged.
    """
    if url is None:
        return None
    normalised = str(url).strip().rstrip("/").lower()
    return normalised if normalised in SUPERVISOR_URLS else url


def is_supervisor_token_for_another_host(hass_url: str | None, token: str | None) -> bool:
    """True when ``token`` is the Supervisor token and ``hass_url`` is not the Supervisor proxy.

    The Supervisor token is only for the Supervisor proxy and must never be sent to another host.
    """
    supervisor_token = os.getenv("SUPERVISOR_TOKEN", "").strip()
    if not supervisor_token or str(token or "").strip() != supervisor_token:
        return False
    return not is_supervisor_url(hass_url)


async def _create_and_start_client(
    hass_url: str, token: str, logger: logging.Logger | None
) -> AsyncWebSocketClient:
    """Helper to create and start a new WebSocket client."""
    # Every WebSocket client is created here, so this check covers them all. Reconnects
    # reuse the URL and token checked here, which nothing changes afterwards.
    if is_supervisor_token_for_another_host(hass_url, token):
        if logger:
            logger.error(SUPERVISOR_TOKEN_REFUSED)
        raise ConnectionError(SUPERVISOR_TOKEN_REFUSED)
    if logger:
        logger.debug(f"Creating new WebSocket client for {hass_url}")
    client = AsyncWebSocketClient(hass_url=hass_url, long_lived_token=token, logger=logger)
    try:
        await asyncio.wait_for(client.startup(), timeout=15.0)
        if logger:
            logger.info("WebSocket client started successfully")
        return client
    except TimeoutError:
        if logger:
            logger.error("WebSocket startup timed out")
        raise ConnectionError("WebSocket startup timed out") from None
    except Exception as e:
        if logger:
            logger.error(f"WebSocket startup failed: {e}")
        raise


async def _reconnect_client(client: AsyncWebSocketClient, logger: logging.Logger | None) -> None:
    """Helper to reconnect an existing WebSocket client."""
    if logger:
        logger.debug("WebSocket client exists but not connected, attempting reconnect")
    try:
        await asyncio.wait_for(client.reconnect(), timeout=15.0)
        if logger:
            logger.info("WebSocket client reconnected successfully")
    except TimeoutError:
        if logger:
            logger.error("WebSocket reconnect timed out")
        raise ConnectionError("WebSocket reconnect timed out") from None
    except Exception as e:
        if logger:
            logger.error(f"WebSocket reconnect failed: {e}")
        raise


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
            # Logic extracted to helper
            _global_client = await _create_and_start_client(hass_url, token, logger)
        elif not _global_client.connected:
            # Logic extracted to helper
            try:
                await _reconnect_client(_global_client, logger)
            except Exception:
                # Ensure global is reset if reconnect fails
                _global_client = None
                raise
        elif logger:
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
