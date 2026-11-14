"""Optional API authentication for the EMHASS web server.

Opt-in: with no ``emhass_api_token`` configured, nothing changes. With a token
configured, every request needs one of:

- an ``Authorization: Bearer <token>`` header (rest_command, curl, scripts);
- the web UI session cookie set by ``/login``;
- a request the add-on ingress proxy sent (not decided yet, see
  ``is_trusted_ingress_request``).

``/healthz`` and the login page stay open.

The token is read once, by ``configure_now()`` when the web server module is
imported (in the Gunicorn master, with preload, so every worker shares one
snapshot), from the same places
as the other secrets: the ``EMHASS_API_TOKEN`` environment variable, then
``options.json`` (add-on), then ``secrets_emhass.yaml``, a later source
replacing an earlier one as in ``build_secrets``. A source that leaves it empty
does not replace an earlier one. The request hook looks it up on every request.
If a token source exists but cannot be read, or the token is shorter than
``MIN_TOKEN_LENGTH``, every request except ``/healthz`` and ``/logout`` gets a
503 rather than running unauthenticated or with a guessable token.

A request authenticated only by the session cookie must send JSON when it
changes something (any method but GET, HEAD and OPTIONS). A page on another
origin cannot send a JSON content type without a CORS preflight, which EMHASS
does not answer, so this stops cross-origin form posts that would otherwise
carry the cookie. ``SameSite=Strict`` alone does not, because it ignores ports.

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
import sys
import time

import orjson
import yaml
from quart import Quart, Response, current_app, request

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
# Slows down guessing through the login form.
FAILED_LOGIN_DELAY_S = 1.0

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
    None when no source sets it. Raises when a file that exists cannot be
    parsed, so the caller can fail closed. Synchronous, so it can run at import.
    """
    env = os.environ if environ is None else environ
    token = _normalise(env.get(API_TOKEN_ENV))

    if options_path and pathlib.Path(options_path).is_file():
        content = pathlib.Path(options_path).read_text(encoding="utf-8")
        try:
            options = orjson.loads(content) if content.strip() else {}
        except orjson.JSONDecodeError as e:
            raise ApiTokenError("options file is not valid JSON") from e
        if isinstance(options, dict):
            token = _normalise(options.get(API_TOKEN_KEY)) or token

    if secrets_path and pathlib.Path(secrets_path).is_file():
        content = pathlib.Path(secrets_path).read_text(encoding="utf-8")
        try:
            secrets = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ApiTokenError("secrets file is not valid YAML") from e
        if isinstance(secrets, dict):
            token = _normalise(secrets.get(API_TOKEN_KEY)) or token

    return token


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
        # Only the exception type: a YAML error message quotes the file.
        _fail(logger, f"Could not read {API_TOKEN_KEY} ({type(e).__name__})")
        return
    if token is not None and len(token) < MIN_TOKEN_LENGTH:
        _fail(
            logger,
            f"{API_TOKEN_KEY} is shorter than {MIN_TOKEN_LENGTH} characters; use a long random value",
        )
        return
    _state.update(loaded=True, token=token, failed=False)


async def configure(
    secrets_path: pathlib.Path | str | None,
    options_path: pathlib.Path | str | None,
    logger: logging.Logger,
) -> None:
    """``configure_now`` for async callers."""
    configure_now(secrets_path, options_path, logger)


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
    """``value`` for a log line, with the API token masked if a caller sent it.

    The token is never read from a request body, but a client that puts it
    there by mistake must not get it written to the action log, whose lines
    are returned in error responses.
    """
    if isinstance(value, dict) and API_TOKEN_KEY in value:
        return {**value, API_TOKEN_KEY: "***"}
    return value


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


def _bearer_matches(token: str) -> bool:
    header = request.headers.get("Authorization", "")
    scheme, _, value = header.strip().partition(" ")
    value = value.strip()
    if scheme.lower() != "bearer" or not value:
        return False
    try:
        # Header values arrive decoded as latin-1. Encoding back gives the bytes
        # on the wire, which for a UTF-8 token are its UTF-8 bytes.
        presented = value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(presented, _token_bytes(token))


def _session_mac(token: str, expires: int) -> str:
    message = f"emhass-ui-session:{expires}".encode()
    return hmac.new(_token_bytes(token), message, hashlib.sha256).hexdigest()


def make_session_value(token: str, now: float | None = None) -> str:
    """Cookie value: expiry time and an HMAC of it keyed with the token.

    Holds no copy of the token, needs no server-side store (so it works across
    Gunicorn workers), and stops working when the token changes.
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


def _bearer_presented() -> bool:
    return request.headers.get("Authorization", "").strip().lower().startswith("bearer")


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


async def login():
    token = current_token()
    if _state["failed"]:
        return _unavailable()
    if token is None:
        return _redirect("./")
    if request.method == "GET":
        return _login_page()
    form = await request.form
    presented = form.get("token", "").strip().encode("utf-8")
    if presented and hmac.compare_digest(presented, _token_bytes(token)):
        response = _redirect("./")
        response.set_cookie(
            SESSION_COOKIE,
            make_session_value(token),
            max_age=SESSION_MAX_AGE_S,
            path="/",
            httponly=True,
            samesite="Strict",
            secure=request.scheme == "https",
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
    """GET shows a button, POST logs out, so another page cannot log a user out
    with a plain link or image."""
    if request.method == "GET":
        response = Response(_LOGOUT_PAGE, content_type="text/html")
        response.headers["Cache-Control"] = "no-store"
        return response
    response = _redirect("login")
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


def install(app: Quart) -> None:
    """Attach the hook and the login routes to ``app``."""
    app.before_request(require_api_token)
    app.add_url_rule("/login", "api_auth_login", login, methods=["GET", "POST"])
    app.add_url_rule("/logout", "api_auth_logout", logout, methods=["GET", "POST"])
