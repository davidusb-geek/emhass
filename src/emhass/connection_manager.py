import asyncio
import functools
import logging
import os
import unicodedata
import urllib.parse

from emhass.websocket_client import AsyncWebSocketClient, ConnectionError

try:  # installed with aiohttp, through yarl
    import idna
except ImportError:  # pragma: no cover
    idna = None

_global_client: AsyncWebSocketClient | None = None
_lock = asyncio.Lock()

# The Supervisor proxy URL, the only URL that may receive the Supervisor token (compared
# without a trailing slash). The proxy is HTTP only, so there is no https form.
SUPERVISOR_URLS = frozenset({"http://supervisor/core/api"})

SUPERVISOR_TOKEN_REFUSED = (
    "The Supervisor token is only sent to the Supervisor, and the Home Assistant URL is "
    "another host, so the token will not be sent. Give a long-lived access token for that "
    "URL in the add-on options (long_lived_token), the secrets file or --key, not in "
    "SUPERVISOR_TOKEN."
)

NO_WEBSOCKET_TOKEN = (
    "No Home Assistant token is set for this URL, so the WebSocket connection is not opened."
)


def is_supervisor_url(url: str | None) -> bool:
    """True when ``url`` is the Supervisor proxy, or unset (which means the Supervisor)."""
    if url is None or not str(url).strip() or str(url).strip() == "empty":
        return True
    return str(url).strip().rstrip("/").lower() in SUPERVISOR_URLS


def canonical_supervisor_url(url: str | None) -> str | None:
    """The Supervisor proxy URL in the exact form the request code builds on.

    The Supervisor URL written with other case, spaces or a trailing slash is returned as
    ``http://supervisor/core/api``. Anything else, including an unset or ``"empty"`` URL, is
    returned unchanged.
    """
    if url is None:
        return None
    normalised = str(url).strip().rstrip("/").lower()
    return normalised if normalised in SUPERVISOR_URLS else url


# A Supervisor token at least this long is also found inside a longer value ("Bearer <token>",
# the token with a stray character around it). A shorter one is only compared whole, so a short
# stand-in token cannot match a value by accident.
MIN_CONTAINED_TOKEN = 16


def holds_supervisor_token(token: str | None) -> bool:
    """True when ``token`` is the Supervisor token, or holds it with something around it.

    A copy changed inside (a character inserted, other case, cut short) is not found.
    Writing one takes the token itself, so it gives nothing away.
    """
    supervisor_token = os.getenv("SUPERVISOR_TOKEN", "").strip()
    if not supervisor_token:
        return False
    value = str(token or "")
    if value.strip() == supervisor_token:
        return True
    return len(supervisor_token) >= MIN_CONTAINED_TOKEN and supervisor_token in value


def is_supervisor_request_url(url: str | None) -> bool:
    """True when ``url`` is the Supervisor proxy's base URL or a path under it.

    For a full request URL, where ``is_supervisor_url`` compares base URLs. Unset is not the
    Supervisor here: a request URL is always given in full.
    """
    if url is None:
        return False
    value = str(url).strip().lower()
    return any(value == base or value.startswith(base + "/") for base in SUPERVISOR_URLS)


def is_supervisor_token_for_another_host(hass_url: str | None, token: str | None) -> bool:
    """True when ``token`` holds the Supervisor token and ``hass_url`` is not the Supervisor proxy.

    The Supervisor token is only for the Supervisor proxy and must never be sent to another host.
    """
    return holds_supervisor_token(token) and not is_supervisor_url(hass_url)


# The default hass_url, which is also the example in the docs and in
# secrets_emhass(example).yaml. RFC 2606 reserves .invalid for names that are sure to be
# invalid.
PLACEHOLDER_HASS_URL = "https://homeassistant.invalid:8123/"
# The example the docs and the example secrets file used before, a DuckDNS name.
# Existing installs may still have it set.
OLD_PLACEHOLDER_HASS_URL = "https://myhass.duckdns.org/"
# Both are treated as unset and never get a Home Assistant token. Only these two names
# and names under them match, not every .invalid name.
_PLACEHOLDER_HOSTS = ("homeassistant.invalid", "myhass.duckdns.org")
# The token build_secrets uses when none is set. It is an example, not a secret.
PLACEHOLDER_TOKEN = "thatverylongtokenhere"

PLACEHOLDER_URL_REFUSED = (
    f"hass_url is still the example from the docs ({PLACEHOLDER_HASS_URL} or the older "
    f"{OLD_PLACEHOLDER_HASS_URL}), so nothing is sent there, and no Home Assistant token. "
    "Set hass_url to your own Home Assistant URL (EMHASS_URL, the add-on option hass_url, "
    "the secrets file or --url)."
)

HASS_URL_UNSET_REFUSED = (
    "hass_url is not set, so the Home Assistant token is not sent anywhere. Set hass_url to "
    "your own Home Assistant URL (EMHASS_URL, the add-on option hass_url, the secrets file "
    "or --url)."
)


# Full stops that IDNA (UTS 46) maps to "." in a host name.
_FULL_STOPS = str.maketrans({"\u3002": ".", "\uff0e": ".", "\uff61": "."})


def _idna_2003(name: str) -> bytes:
    """Python's own IDNA codec, which websockets and urllib use for a host name."""
    return name.encode("idna")


# How the clients turn a host name into a DNS name: aiohttp (through yarl) uses the
# idna package's UTS 46 mapping, websockets and urllib use Python's IDNA 2003 codec.
_ENCODERS = [_idna_2003]
if idna is not None:
    _ENCODERS.append(functools.partial(idna.encode, uts46=True))


def _simplify(text: str) -> str:
    """NFKC, the other full stops as ".", and no invisible format characters."""
    text = unicodedata.normalize("NFKC", text).translate(_FULL_STOPS)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _dns_names(host: str) -> set[str]:
    """The DNS names a client may turn ``host`` into, lower case and without a final dot."""
    names = set()
    for form in {host, urllib.parse.unquote(host)}:
        for text in {form, _simplify(form)}:
            names.add(text.casefold())
            for encode in _ENCODERS:
                try:
                    names.add(encode(text).decode("ascii").lower())
                except (UnicodeError, ValueError):
                    pass  # no client can use this name either
    return {name.rstrip(".") for name in names}


def is_placeholder_hass_url(url: str | None) -> bool:
    """True when ``url`` points at one of the example hosts or a name under one.

    The host is taken from the URL as given (so percent-encoded characters in the
    user part cannot move it), then compared in every form the HTTP, WebSocket and
    urllib clients may turn it into: percent-decoded, NFKC, the other full stops,
    without invisible characters, case-folded, and through both IDNA encoders. An
    unparsable URL that names the host counts as a match.
    """
    if url is None:
        return False
    text = str(url).strip()
    if not text or text == "empty":
        return False
    for candidate in (text, _simplify(text)):
        if "://" not in candidate and not candidate.startswith("//"):
            candidate = "//" + candidate
        try:
            host = urllib.parse.urlsplit(candidate).hostname or ""
        except ValueError:
            decoded = _simplify(urllib.parse.unquote(text)).casefold()
            if any(example in decoded for example in _PLACEHOLDER_HOSTS):
                return True
            continue
        if any(
            name == example or name.endswith("." + example)
            for name in _dns_names(host)
            for example in _PLACEHOLDER_HOSTS
        ):
            return True
    return False


def is_refused_placeholder(hass_url: str | None, token: str | None) -> bool:
    """True when ``hass_url`` is an example host and ``token`` is not the example token.

    The example URL with the example token is an install with nothing set, which keeps its
    old behaviour. Any other token there means one was set without a URL (and has been or
    must be dropped), so nothing is sent to the example host.
    """
    return is_placeholder_hass_url(hass_url) and str(token or "").strip() != PLACEHOLDER_TOKEN


def is_real_token(token: str | None) -> bool:
    """True for a token someone set: not empty, not "empty" and not the example token."""
    return str(token or "").strip() not in ("", "empty", PLACEHOLDER_TOKEN)


def is_unset_hass_url(url: str | None) -> bool:
    """True when ``url`` is not set: None, blank or "empty"."""
    return url is None or str(url).strip() in ("", "empty")


def is_token_for_placeholder(hass_url: str | None, token: str | None) -> bool:
    """True when a real token (not empty and not the example one) would go to an example host."""
    return is_real_token(token) and is_placeholder_hass_url(hass_url)


async def _create_and_start_client(
    hass_url: str, token: str, logger: logging.Logger | None
) -> AsyncWebSocketClient:
    """Helper to create and start a new WebSocket client."""
    # Every WebSocket client is created here, so these checks cover them all. Reconnects
    # reuse the URL and token checked here, which nothing changes afterwards, and the
    # client never follows a redirect (websocket_client._ws_connect), so the token is only
    # sent to the host of this URL.
    if is_refused_placeholder(hass_url, token):
        if logger:
            logger.error(PLACEHOLDER_URL_REFUSED)
        raise ConnectionError(PLACEHOLDER_URL_REFUSED)
    if is_supervisor_token_for_another_host(hass_url, token):
        if logger:
            logger.error(SUPERVISOR_TOKEN_REFUSED)
        raise ConnectionError(SUPERVISOR_TOKEN_REFUSED)
    # An empty token can never authenticate, so don't open a connection just to send it.
    if not str(token or "").strip() or str(token).strip() == "empty":
        if logger:
            logger.error(NO_WEBSOCKET_TOKEN)
        raise ConnectionError(NO_WEBSOCKET_TOKEN)
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
