"""Optional API authentication for the EMHASS web server.

Opt-in: with no ``emhass_api_token`` configured, nothing changes. With a token
configured, every request needs one of:

- an ``Authorization: Bearer <token>`` header (rest_command, curl, scripts);
- the web UI session cookie set by ``/login``;
- a request the add-on ingress proxy sent (not decided yet, see
  ``is_trusted_ingress_request``).

``/healthz``, ``/login`` and ``/logout`` stay open. WebSocket handshakes need
the bearer header.

The token is read once, by ``configure_now()`` when the web server module is
imported (in the Gunicorn master, with preload, so every worker shares one
snapshot), from the same places
as the other secrets: the ``EMHASS_API_TOKEN`` environment variable, then
``options.json`` (add-on), then ``secrets_emhass.yaml``, a later source
replacing an earlier one as in ``build_secrets``. A source that leaves it empty
does not replace an earlier one. The request hook looks it up on every request.
If a token source is there but cannot be used (see ``read_api_token``), or the
token is shorter than ``MIN_TOKEN_LENGTH``, every request except ``/healthz``
and ``/logout`` gets a 503 rather than running unauthenticated or with a
guessable token. Only a missing file counts as "not configured".

A request authenticated only by the session cookie must send JSON when it
changes something (any method but GET, HEAD and OPTIONS). A page on another
origin cannot send a JSON content type without a CORS preflight, which EMHASS
does not answer, so this stops cross-origin form posts that would otherwise
carry the cookie. ``SameSite=Strict`` alone does not, because it ignores ports.

The rule stops a page on another port of the same host from *posting* with
the cookie. It does not stop the server on that port from *receiving* it:
browsers send cookies to every port of a host, so whoever runs a service on
another port of the EMHASS host can copy the cookie and replay it. A copied
cookie works until it expires or EMHASS restarts; logging out clears only the
browser's copy.

Everything lives in this module and is attached with ``install(app)``, so the
parts that need a design decision (the UI login and ingress) can be swapped
without touching the routes.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import logging
import os
import pathlib
import secrets
import sys
import time
import urllib.parse

import orjson
import yaml
from quart import Quart, Response, current_app, request, websocket

API_TOKEN_KEY = "emhass_api_token"
API_TOKEN_ENV = "EMHASS_API_TOKEN"

SESSION_COOKIE = "emhass_session"
SESSION_MAX_AGE_S = 30 * 24 * 3600
# Tokens shorter than this are refused (503 on every route but /healthz). The
# bearer check is not rate limited, and a session cookie is an HMAC keyed with
# the token, so a short token could be guessed online or offline from a cookie.
MIN_TOKEN_LENGTH = 16
# Failed attempts are logged as a count, at most once per this many seconds.
FAILURE_LOG_INTERVAL_S = 60.0
# Slows down guessing through the login form, one guess at a time. It does not
# slow guesses sent in parallel; the token length is the real guard.
FAILED_LOGIN_DELAY_S = 1.0
# Largest login form accepted, in bytes, and how many fields it may have. It
# is read before any credential check.
MAX_LOGIN_BODY = 4096
MAX_LOGIN_FIELDS = 16
# Used when the app sets no BODY_TIMEOUT.
LOGIN_BODY_TIMEOUT_S = 60

# Longest expiry field accepted in a session cookie (seconds since 1970).
_MAX_EXPIRY_DIGITS = 12
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# Endpoints that never need credentials.
OPEN_ENDPOINTS = frozenset({"healthz", "api_auth_login", "api_auth_logout"})
# Pages a browser opens directly. Without credentials they redirect to the
# login page instead of answering 401.
PAGE_ENDPOINTS = frozenset({"index", "configuration"})


class ApiTokenError(Exception):
    """A token source exists but does not hold a usable token."""


# loaded: configure() ran. token: the token, or None when none is set.
# failed: configure() could not read a token source, so requests are refused.
_state: dict = {"loaded": False, "token": None, "failed": False}


def _normalise(value) -> str | None:
    """A configured token as a string, or None for "not set".

    Only strings are accepted. YAML reads an unquoted all-digit value as a
    number and can change it on the way (a leading zero makes it octal), so
    a number is refused rather than converted.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiTokenError(f"{API_TOKEN_KEY} must be a string; quote it in YAML")
    return value.strip() or None


def read_api_token(
    secrets_path: pathlib.Path | str | None,
    options_path: pathlib.Path | str | None,
    environ: dict[str, str] | None = None,
) -> str | None:
    """Read the API token from the environment, options.json and the secrets file.

    A later source replaces an earlier one, as in ``build_secrets``. Returns
    None when no source sets it. Raises when a source is there but cannot be
    used, so the caller can fail closed: a path that cannot be read (a
    directory, no permission), a file that does not parse, a non-empty file
    whose top level is not a mapping, or a YAML key with no value (an unquoted
    token that starts with ``#`` reads as a comment). Only a missing file
    means "not configured". Synchronous, so it can run at import.

    The errors are raised ``from None``: a YAML error's text quotes the line
    of the file, which can be the token.
    """
    env = os.environ if environ is None else environ
    token = _normalise(env.get(API_TOKEN_ENV))

    content = _read_source(options_path)
    if content is not None:
        try:
            options = orjson.loads(content) if content.strip() else {}
        except orjson.JSONDecodeError:
            raise ApiTokenError("options file is not valid JSON") from None
        if not isinstance(options, dict):
            raise ApiTokenError("options file is not a JSON object")
        token = _normalise(options.get(API_TOKEN_KEY)) or token

    content = _read_source(secrets_path)
    if content is not None:
        try:
            loaded = yaml.safe_load(content)
        except yaml.YAMLError:
            raise ApiTokenError("secrets file is not valid YAML") from None
        if loaded is None:
            loaded = {}  # empty, or comments only
        if not isinstance(loaded, dict):
            raise ApiTokenError("secrets file is not a YAML mapping")
        if API_TOKEN_KEY in loaded and loaded[API_TOKEN_KEY] is None:
            raise ApiTokenError(f"{API_TOKEN_KEY} has no value; quote the token in YAML")
        token = _normalise(loaded.get(API_TOKEN_KEY)) or token

    return token


def _read_source(path: pathlib.Path | str | None) -> str | None:
    """The text of a token source, or None when there is no such file.

    The file is opened directly rather than checked with ``is_file()``: a
    directory (what Docker creates when a bind-mounted file is missing on the
    host) or a file that cannot be read must fail closed, not count as absent.
    """
    if not path:
        return None
    try:
        return pathlib.Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as e:
        raise ApiTokenError(f"token source cannot be read ({type(e).__name__})") from None


async def load_api_token(
    secrets_path: pathlib.Path | str | None,
    options_path: pathlib.Path | str | None,
    environ: dict[str, str] | None = None,
) -> str | None:
    """``read_api_token`` for async callers."""
    return read_api_token(secrets_path, options_path, environ)


def configure_now(
    secrets_path: pathlib.Path | str | None,
    options_path: pathlib.Path | str | None,
    logger: logging.Logger,
) -> None:
    """Load the token for the request hook. Never logs the token itself.

    Runs before the logging level is set, so only a failure is logged here.
    ``log_status()`` reports the rest once logging is set up.
    """
    try:
        token = read_api_token(secrets_path, options_path)
    except Exception as e:
        _fail(logger, f"Could not read {API_TOKEN_KEY} ({_describe(e)})")
        return
    if token is not None and len(token) < MIN_TOKEN_LENGTH:
        _fail(
            logger,
            f"{API_TOKEN_KEY} is shorter than {MIN_TOKEN_LENGTH} characters; use a long random value",
        )
        return
    _state.update(loaded=True, token=token, failed=False)


def recheck_now(
    secrets_path: pathlib.Path | str | None,
    options_path: pathlib.Path | str | None,
    logger: logging.Logger,
) -> None:
    """Fail closed if a token appeared after the import-time snapshot found none.

    Runs as each worker starts. A token set before the snapshot is kept as it
    is, so changing one still needs a restart. But a server whose snapshot has
    no token must not keep running open once the files hold one, or once they
    cannot be read (a file that was missing at import and written while EMHASS
    started up, for example).
    """
    if _state["failed"] or _state["token"] is not None:
        return
    try:
        token = read_api_token(secrets_path, options_path)
    except Exception as e:
        _fail(logger, f"Could not read {API_TOKEN_KEY} ({_describe(e)})")
        return
    if token is not None:
        _fail(
            logger,
            f"{API_TOKEN_KEY} was set after EMHASS started loading; restart EMHASS to turn "
            "API authentication on",
        )


async def configure(
    secrets_path: pathlib.Path | str | None,
    options_path: pathlib.Path | str | None,
    logger: logging.Logger,
) -> None:
    """``configure_now`` for async callers."""
    configure_now(secrets_path, options_path, logger)


def _describe(error: Exception) -> str:
    """Why a token source was refused, safe to log.

    ApiTokenError messages are fixed text written here. Anything else is
    reported by type only: a YAML error's text quotes the line of the file.
    """
    if isinstance(error, ApiTokenError):
        return str(error)
    return type(error).__name__


def _fail(logger: logging.Logger, reason: str) -> None:
    _state.update(loaded=True, token=None, failed=True)
    logger.error(
        "%s. Every request except /healthz and /logout will get 503 until this is fixed.",
        reason,
    )


def is_loaded() -> bool:
    return _state["loaded"]


def current_token() -> str | None:
    """The token in force for this request.

    Before ``configure()`` has run (initialisation failed early, or a test
    client), only the environment variable can set it.
    """
    if _state["loaded"]:
        return _state["token"]
    try:
        return _normalise(os.environ.get(API_TOKEN_ENV))
    except ApiTokenError:
        return None


def redacted(value):
    """``value`` for a log line, with an ``emhass_api_token`` key masked.

    Only that top-level key; ``TokenLogFilter`` masks the token by value
    wherever else it turns up.
    """
    if isinstance(value, dict) and API_TOKEN_KEY in value:
        return {**value, API_TOKEN_KEY: "***"}
    return value


class TokenLogFilter(logging.Filter):
    """Masks the configured token in a log record's message and traceback.

    The token is never read from a request body, but a client that puts it
    there by mistake (nested, under another key, in a YAML error) must not get
    it written to the console or the action log, whose lines are returned in
    error responses. Attach it with ``mask_token_in()``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        token = current_token()
        if not token:
            return True
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record is the formatter's to report
            return True
        if token in message:
            record.msg = message.replace(token, "***")
            record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text and token in record.exc_text:
            record.exc_text = record.exc_text.replace(token, "***")
        if record.stack_info and token in record.stack_info:
            record.stack_info = record.stack_info.replace(token, "***")
        return True


token_log_filter = TokenLogFilter()


def mask_token_in(logger: logging.Logger) -> None:
    """Attach ``token_log_filter`` to ``logger`` and to each of its handlers.

    On the logger it covers every record logged there, whichever handler
    writes it (Quart's console handler comes from Flask, not quart.logging).
    On the handlers it covers records that reach them from other loggers.
    Safe to call again after adding a handler.
    """
    logger.addFilter(token_log_filter)
    for handler in logger.handlers:
        handler.addFilter(token_log_filter)


def reset() -> None:
    """Forget the loaded token. For tests."""
    _state.update(loaded=False, token=None, failed=False)
    _failures.update(count=0, since=None)


# Failed attempts since the last log line, and when that line was written.
_failures: dict = {"count": 0, "since": None}


def _note_failure() -> None:
    """Count a wrong token or cookie, and log the count at most once a minute.

    Nothing about the attempt itself is logged, only how many there were.
    """
    _failures["count"] += 1
    now = time.monotonic()
    since = _failures["since"]
    if since is not None and now - since < FAILURE_LOG_INTERVAL_S:
        return
    current_app.logger.warning(
        "%d failed API authentication attempt(s) since the last report",
        _failures["count"],
    )
    _failures.update(count=0, since=now)


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    host = host.strip().strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def bind_host(server_ip: str | None) -> str:
    """Best guess at the address the server listens on.

    Under Gunicorn the bind comes from gunicorn.conf.py, which reads the IP
    environment variable. A ``-b`` on the Gunicorn command line is not visible
    here. Run directly, ``main()`` binds to ``server_ip``.
    """
    if "gunicorn" in sys.modules:
        return os.getenv("IP", "0.0.0.0")
    return server_ip or "0.0.0.0"


def log_status(logger: logging.Logger, host: str) -> None:
    """Start-up report: auth on (and a weak token), or a warning when it is off
    and the server is reachable from the network. Never logs the token."""
    if _state["failed"]:
        return  # configure() already logged the error
    if current_token() is not None:
        logger.info("API authentication is enabled")
        return
    if is_loopback(host):
        logger.info("API authentication is off (no %s set; listening on %s)", API_TOKEN_KEY, host)
        return
    logger.warning(
        "No API token is set and the web server listens on %s. Any device that can "
        "reach this port can change the configuration and run actions. Set %s "
        "(secrets file, add-on option or %s environment variable), or listen on "
        "127.0.0.1 only.",
        host,
        API_TOKEN_KEY,
        API_TOKEN_ENV,
    )


# -- credentials --------------------------------------------------------------


def _token_bytes(token: str) -> bytes:
    return token.encode("utf-8")


# Only space and tab around a header value. str.strip() would also remove
# characters such as U+0085 and U+00A0, which here are latin-1 decoded bytes
# of the token itself (the last byte of a UTF-8 "a" with a grave accent is 0xA0).
_HEADER_SPACE = " \t"


def _bearer_matches(token: str, headers=None) -> bool:
    headers = request.headers if headers is None else headers
    header = headers.get("Authorization", "")
    scheme, _, value = header.strip(_HEADER_SPACE).partition(" ")
    value = value.strip(_HEADER_SPACE)
    if scheme.lower() != "bearer" or not value:
        return False
    try:
        # Header values arrive decoded as latin-1. Encoding back gives the bytes
        # on the wire, which for a UTF-8 token are its UTF-8 bytes.
        presented = value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(presented, _token_bytes(token))


# Random key for the session cookie, made when this module is imported: in the
# Gunicorn master with preload_app (set in gunicorn.conf.py), so every worker
# shares it. A restart ends every session, and a captured cookie cannot be
# used to guess the token offline.
_SESSION_KEY = secrets.token_bytes(32)


def _session_mac(token: str, expires: int) -> str:
    # Keyed with the random key and the token, so a cookie also stops working
    # when the token changes.
    key = hmac.new(_SESSION_KEY, _token_bytes(token), hashlib.sha256).digest()
    message = f"emhass-ui-session:{expires}".encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def make_session_value(token: str, now: float | None = None) -> str:
    """Cookie value: expiry time and an HMAC of it.

    Holds no copy of the token and needs no server-side store, so it works
    across Gunicorn workers that share the key. It stops working when EMHASS
    restarts or the token changes. Logging out clears only the browser's copy:
    a copied value stays valid until it expires or EMHASS restarts.
    """
    expires = int(time.time() if now is None else now) + SESSION_MAX_AGE_S
    return f"{expires}.{_session_mac(token, expires)}"


def session_is_valid(token: str, value: str | None, now: float | None = None) -> bool:
    if not value:
        return False
    expires_text, _, mac = value.partition(".")
    if (
        not expires_text.isascii()
        or not expires_text.isdigit()
        or len(expires_text) > _MAX_EXPIRY_DIGITS
        or not mac
    ):
        return False
    expires = int(expires_text)
    if expires < (time.time() if now is None else now):
        return False
    return hmac.compare_digest(
        mac.encode("ascii", "replace"), _session_mac(token, expires).encode()
    )


def is_trusted_ingress_request() -> bool:
    """Seam for the Home Assistant add-on: True for a request that came through ingress.

    Not decided, so nothing is trusted and ingress users log in like anyone else.
    ``request.remote_addr`` and headers such as ``X-Ingress-Path`` or
    ``X-Forwarded-For`` must not be used: a client that reaches the port
    directly can set the headers, and with forwarded-header handling switched
    on, the address follows them. See DESIGN-part2.md.
    """
    return False


def _session_cookie_state(token: str) -> str:
    """ "valid", "invalid" or "absent" for the session cookies in this request.

    Every cookie of that name is checked, not just the first: a page on another
    port of the same host can set one with a longer path, which the browser
    then sends first.
    """
    values = request.cookies.getlist(SESSION_COOKIE)
    if not values:
        return "absent"
    if any(session_is_valid(token, value) for value in values):
        return "valid"
    return "invalid"


def _bearer_presented(headers=None) -> bool:
    headers = request.headers if headers is None else headers
    return headers.get("Authorization", "").strip(_HEADER_SPACE).lower().startswith("bearer")


def _came_over_https() -> bool:
    """True when the browser's side of the connection is HTTPS.

    ``X-Forwarded-Proto`` is honoured even from an untrusted client: all it can
    do is make its own browser keep the cookie off plain HTTP.
    """
    if request.scheme == "https":
        return True
    forwarded = request.headers.get("X-Forwarded-Proto", "")
    return forwarded.split(",")[0].strip(_HEADER_SPACE).lower() == "https"


def _cookie_request_allowed() -> bool:
    """A cookie-authenticated request that changes something must send JSON."""
    return request.method in SAFE_METHODS or request.mimetype == "application/json"


# -- responses ----------------------------------------------------------------


def _relative(target: str) -> str:
    """A Location relative to the current path.

    Behind ingress the browser sees the app under a path prefix that the app
    does not see, so absolute paths would leave the app.
    """
    depth = max(request.path.count("/") - 1, 0)
    return "../" * depth + target


def _redirect(target: str) -> Response:
    response = Response("", status=303)
    response.headers["Location"] = target
    response.headers["Cache-Control"] = "no-store"
    return response


def _unauthorized() -> Response:
    response = Response(
        "EMHASS >> Authentication required. Send the header "
        "'Authorization: Bearer <emhass_api_token>'.\n",
        status=401,
        content_type="text/plain",
    )
    response.headers["WWW-Authenticate"] = 'Bearer realm="EMHASS"'
    response.headers["Cache-Control"] = "no-store"
    return response


def _forbidden_cookie_post() -> Response:
    response = Response(
        "EMHASS >> A request signed in with the web UI cookie must send "
        "'Content-Type: application/json'.\n",
        status=403,
        content_type="text/plain",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def _plain(status: int, text: str) -> Response:
    response = Response(text, status=status, content_type="text/plain")
    response.headers["Cache-Control"] = "no-store"
    return response


def _unavailable() -> Response:
    response = Response(
        "EMHASS >> API authentication is not configured correctly. See the log.\n",
        status=503,
        content_type="text/plain",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


_LOGIN_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>EMHASS login</title>
<style>
body {{ font-family: sans-serif; max-width: 22rem; margin: 4rem auto; padding: 0 1rem; }}
input {{ width: 100%; box-sizing: border-box; padding: .5rem; margin: .5rem 0; }}
.error {{ color: #b00020; }}
</style>
</head>
<body>
<h1>EMHASS</h1>
{message}
<form method="post" action="login">
<label for="token">API token</label>
<input id="token" name="token" type="password" autocomplete="current-password" required autofocus>
<input type="submit" value="Log in">
</form>
</body>
</html>
"""


def _login_page(status: int = 200, failed: bool = False) -> Response:
    message = '<p class="error">That token is not right.</p>' if failed else ""
    response = Response(
        _LOGIN_PAGE.format(message=message), status=status, content_type="text/html"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


# -- hook and routes ----------------------------------------------------------


async def require_api_token():
    """before_request hook. Returns None to let the request through."""
    if request.endpoint in OPEN_ENDPOINTS:
        return None
    if _state["failed"]:
        return _unavailable()
    token = current_token()
    if token is None:
        return None
    if _bearer_matches(token) or is_trusted_ingress_request():
        return None
    cookie = _session_cookie_state(token)
    if cookie == "valid":
        if _cookie_request_allowed():
            return None
        return _forbidden_cookie_post()
    if cookie == "invalid" or _bearer_presented():
        _note_failure()
    if (
        request.method == "GET"
        and request.endpoint in PAGE_ENDPOINTS
        and "text/html" in request.headers.get("Accept", "")
    ):
        return _redirect(_relative("login"))
    return _unauthorized()


async def _read_login_form() -> tuple[dict[str, list[str]], Response | None]:
    """The login form's fields, or a response refusing the body.

    This is the only body read without credentials, so it is read here with a
    small cap rather than through ``request.form``, which would take up to
    MAX_CONTENT_LENGTH (16 MiB) and parse it on the event loop. A body without
    Content-Length (chunked, as a proxy may send it) is capped as it arrives.
    """
    length = request.content_length
    if length is not None and length > MAX_LOGIN_BODY:
        return {}, _plain(413, "EMHASS >> The login form is too large.\n")
    if request.mimetype != "application/x-www-form-urlencoded":
        return {}, _plain(415, "EMHASS >> The login form must be sent as a plain HTML form.\n")
    data = bytearray()
    try:
        async with asyncio.timeout(request.body_timeout or LOGIN_BODY_TIMEOUT_S):
            async for chunk in request.body:
                data.extend(chunk)
                if len(data) > MAX_LOGIN_BODY:
                    return {}, _plain(413, "EMHASS >> The login form is too large.\n")
    except TimeoutError:
        return {}, _plain(408, "EMHASS >> The login form took too long to arrive.\n")
    try:
        fields = urllib.parse.parse_qs(
            data.decode("utf-8", "replace"), max_num_fields=MAX_LOGIN_FIELDS
        )
    except ValueError:
        return {}, _plain(400, "EMHASS >> The login form has too many fields.\n")
    return fields, None


async def login():
    token = current_token()
    if _state["failed"]:
        return _unavailable()
    if token is None:
        return _redirect("./")
    if request.method != "POST":
        return _login_page()
    fields, refused = await _read_login_form()
    if refused is not None:
        return refused
    presented = fields.get("token", [""])[0].strip().encode("utf-8")
    if presented and hmac.compare_digest(presented, _token_bytes(token)):
        response = _redirect("./")
        response.set_cookie(
            SESSION_COOKIE,
            make_session_value(token),
            max_age=SESSION_MAX_AGE_S,
            path="/",
            httponly=True,
            samesite="Strict",
            secure=_came_over_https(),
        )
        return response
    _note_failure()
    await asyncio.sleep(FAILED_LOGIN_DELAY_S)
    return _login_page(status=401, failed=True)


_LOGOUT_PAGE = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>EMHASS log out</title></head>
<body>
<form method="post" action="logout"><input type="submit" value="Log out"></form>
</body>
</html>
"""


async def logout():
    """GET (and HEAD) show a button, only POST logs out, so a link or image on
    another page cannot. A same-site page can still post the form; that only
    clears this browser's cookie. Logging out does not end a copy of the
    cookie held elsewhere: only a restart or a new token does."""
    if request.method != "POST":
        response = Response(_LOGOUT_PAGE, content_type="text/html")
        response.headers["Cache-Control"] = "no-store"
        return response
    response = _redirect("login")
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        samesite="Strict",
        secure=_came_over_https(),
    )
    return response


async def require_api_token_websocket():
    """before_websocket hook: the same rule for a WebSocket handshake.

    EMHASS has no WebSocket route today; this keeps a later one from being
    open by default. Only the bearer header counts here, not the cookie: a
    handshake is not covered by the JSON rule, so a cookie would let a page on
    another port of the same host open one.
    """
    if _state["failed"]:
        return _unavailable()
    token = current_token()
    if token is None:
        return None
    if _bearer_matches(token, websocket.headers) or is_trusted_ingress_request():
        return None
    if _bearer_presented(websocket.headers):
        _note_failure()
    return _unauthorized()


def install(app: Quart) -> None:
    """Attach the hooks and the login routes to ``app``."""
    app.before_request(require_api_token)
    app.before_websocket(require_api_token_websocket)
    app.add_url_rule("/login", "api_auth_login", login, methods=["GET", "POST"])
    app.add_url_rule("/logout", "api_auth_logout", logout, methods=["GET", "POST"])
