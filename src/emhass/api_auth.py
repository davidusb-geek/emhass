"""API access control for the EMHASS web server (GHSA-c39q-j4r5-9xq5).

Closed by default. Every request except ``/healthz``, ``/login`` and ``/logout``
needs one of:

- an ``Authorization: Bearer <token>`` header (rest_command, curl, scripts);
- the web UI session cookie set by ``/login``;
- a request that reached the Home Assistant **ingress port** from the Supervisor
  (see ``is_trusted_ingress_request``). Home Assistant has already signed that
  user in.

WebSocket handshakes need the bearer header (or come from trusted ingress).

The token is read once, by ``configure_now()`` when the web server module is
imported (in the Gunicorn master, with preload, so every worker shares one
snapshot), from the same places as the other secrets: the ``EMHASS_API_TOKEN``
environment variable, then ``options.json`` (add-on), then
``secrets_emhass.yaml``, a later source replacing an earlier one as in
``build_secrets``. A source that leaves it empty does not replace an earlier one.

When none of them sets a token, EMHASS makes one (``secrets.token_urlsafe(32)``)
the first time it starts and keeps it in the data folder in the file
``emhass_api_token`` (mode 0600, never a link). Later starts reuse that file, so
the token survives restarts. Only the path is logged, never the token. In the
add-on, the web UI shows it to a Home Assistant administrator (``/api-token``).
``EMHASS_API_TOKEN_AUTOGENERATE=false`` turns the generation off: with no token
configured, requests then get 503 (trusted ingress is still served, since it needs no token).

``EMHASS_API_AUTH_DISABLED=true`` turns access control off. It is an explicit
operator choice, logged at start-up, and it restores the unauthenticated access
the advisory describes.

If a token source is there but cannot be used (see ``read_api_token``), or the
token is shorter than ``MIN_LENGTH``, every request except ``/healthz`` and
``/logout`` gets a 503 rather than running unauthenticated or with a guessable
token. Only a missing file counts as "not configured". The request hook looks
the token up on every request.

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
import errno
import hashlib
import hmac
import ipaddress
import json
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

# The names of the setting and of the environment variable, not their values.
SETTING_NAME = "emhass_api_token"
ENV_NAME = "EMHASS_API_TOKEN"

SESSION_COOKIE = "emhass_session"
SESSION_MAX_AGE_S = 30 * 24 * 3600
# Tokens shorter than this are refused (503 on every route but /healthz). The
# bearer check is not rate limited, and a session cookie is an HMAC keyed with
# the token, so a short token could be guessed online or offline from a cookie.
MIN_LENGTH = 16
# Explicit operator opt-out (read on every request, so a test or an operator can flip it).
DISABLE_ENV = "EMHASS_API_AUTH_DISABLED"
AUTOGENERATE_ENV = "EMHASS_API_TOKEN_AUTOGENERATE"
TOKEN_FILE_NAME = "emhass_api_token"
# The Home Assistant add-on ingress listener: a port of its own, so ingress is told apart by the
# socket that accepted the connection, which a client cannot choose, and never by a header.
INGRESS_PORT_ENV = "EMHASS_INGRESS_PORT"
INGRESS_PEERS_ENV = "EMHASS_INGRESS_PEERS"
DEFAULT_INGRESS_PORT = 8099
# The Supervisor's address on the hassio network (DOCKER_IPV4_NETWORK_MASK[2]); the developer
# documentation says an add-on must accept ingress connections from this address only.
DEFAULT_INGRESS_PEERS = "172.30.32.2"
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
_state: dict = {"loaded": False, "token": None, "failed": False, "source": None}


# Only ASCII white space around a token. str.strip() would also remove
# characters such as U+0085 and U+00A0, so a token ending in one of them
# could never be matched. Form feed and vertical tab are trimmed too: an
# HTTP header cannot carry them, so a token ending in one could not be sent.
_TOKEN_SPACE = " \t\r\n\f\v"


def _normalise(value) -> str | None:
    """A configured token as a string, or None for "not set".

    Only strings are accepted. YAML reads an unquoted all-digit value as a
    number and can change it on the way (a leading zero makes it octal), so
    a number is refused rather than converted.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiTokenError(f"{SETTING_NAME} must be a string; quote it in YAML")
    try:
        # A lone surrogate (from a YAML escape, or a non-UTF-8 environment
        # variable) cannot be compared or masked, so the token is refused.
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ApiTokenError(f"{SETTING_NAME} is not valid UTF-8 text") from None
    return value.strip(_TOKEN_SPACE) or None


def _top_level_key_count(content: str, name: str) -> int:
    """How many times ``name`` is a key of the top-level YAML mapping in
    ``content``, counting the keys that ``<<`` merge keys bring in."""
    loader = yaml.SafeLoader(content)
    try:
        node = loader.get_single_node()
        if not isinstance(node, yaml.MappingNode):
            return 0
        loader.flatten_mapping(node)
    finally:
        loader.dispose()
    return sum(1 for key, _ in node.value if isinstance(key, yaml.ScalarNode) and key.value == name)


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
    whose top level is not a mapping, the token key twice in the secrets file
    (merge keys included), or a YAML key with no value (an unquoted token that
    starts with ``#`` reads as a comment). Only a missing file means "not
    configured". Synchronous, so it can run at import.

    In options.json a repeated key also keeps the last value; the add-on
    writes that file, so it is not checked.

    The errors are raised ``from None``: a parser's error text can quote the
    line of the file or the value, which can be the token.
    """
    env = os.environ if environ is None else environ
    token = _normalise(env.get(ENV_NAME))

    content = _read_source(options_path)
    if content is not None:
        try:
            options = orjson.loads(content) if content.strip() else {}
        except orjson.JSONDecodeError:
            raise ApiTokenError("options file is not valid JSON") from None
        if not isinstance(options, dict):
            raise ApiTokenError("options file is not a JSON object")
        token = _normalise(options.get(SETTING_NAME)) or token

    content = _read_source(secrets_path)
    if content is not None:
        try:
            loaded = yaml.safe_load(content)
        except yaml.YAMLError:
            raise ApiTokenError("secrets file is not valid YAML") from None
        except Exception:
            # e.g. a !!int tag on a value that is not a number
            raise ApiTokenError("secrets file has a value that cannot be converted") from None
        if loaded is None:
            loaded = {}  # empty, or comments only
        if not isinstance(loaded, dict):
            raise ApiTokenError("secrets file is not a YAML mapping")
        # safe_load keeps only the last of repeated keys, so a second, empty one
        # would turn auth off without a word.
        if _top_level_key_count(content, SETTING_NAME) > 1:
            raise ApiTokenError(f"{SETTING_NAME} is set more than once in the secrets file")
        if SETTING_NAME in loaded and loaded[SETTING_NAME] is None:
            raise ApiTokenError(f"{SETTING_NAME} has no value; quote the token in YAML")
        token = _normalise(loaded.get(SETTING_NAME)) or token

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
        # A link whose target is missing (a secret mount being swapped, for
        # example) is there but cannot be used.
        if os.path.lexists(path):
            raise ApiTokenError("token source is a link to a missing file") from None
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


def auth_disabled() -> bool:
    """True when the operator switched access control off explicitly."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _autogenerate_enabled() -> bool:
    return os.environ.get(AUTOGENERATE_ENV, "").strip().lower() not in {"0", "false", "no", "off"}


def _read_generated(path: pathlib.Path) -> str | None:
    """The token kept in the data folder, or None when there is no such file."""
    if path.is_symlink():
        raise ApiTokenError("the generated token file is a link")
    token = _normalise(_read_source(path))
    if token is not None and len(token) < MIN_LENGTH:
        raise ApiTokenError("the generated token file holds a short token")
    return token


def _read_generated_waiting(path: pathlib.Path) -> str | None:
    """``_read_generated``, but an empty file is given a moment: another worker created it
    and has not written the token yet."""
    for _ in range(40):
        token = _read_generated(path)
        if token is not None or not path.exists():
            return token
        time.sleep(0.05)
    return None


_IS_WINDOWS = os.name == "nt"
# link() errors that mean "this file system has no hard links"
_NO_HARD_LINKS = {
    getattr(errno, name)
    for name in ("EPERM", "ENOTSUP", "EOPNOTSUPP", "ENOSYS", "EINVAL", "EXDEV")
    if hasattr(errno, name)
}
# An empty token file this old was left by a crash between creating and writing it; it holds no secret.
STALE_EMPTY_S = 10.0


def _publish_exclusively(path: pathlib.Path, token: str) -> bool:
    """Make ``path`` hold ``token``, complete or not at all. True when this call made it, False when
    it already existed.

    The token is written and flushed to a private temporary file, which is then hard-linked to
    ``path``: the link fails if ``path`` exists and never exposes a half-written file, even if the
    process dies. A file system without hard links falls back to an exclusive create.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
        except OSError as e:
            if e.errno not in _NO_HARD_LINKS:
                raise  # EIO, ENOSPC and the like are failures, not a reason to use the weaker path
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
        return True
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _remove_stale_temporaries(folder: pathlib.Path) -> None:
    """Best effort: delete temporary token files a killed process left (older than an hour)."""
    try:
        for leftover in folder.glob(f".{TOKEN_FILE_NAME}.*.tmp"):
            if time.time() - leftover.stat().st_mtime > 3600:
                leftover.unlink()
    except OSError:
        pass


LOCK_STALE_S = 30.0


def _acquire_replacement_lock(lock: pathlib.Path) -> bool:
    """Take the lock that serialises replacing a stale token file. A lock left by a killed process
    (older than ``LOCK_STALE_S``) is removed. False when it cannot be had within about two seconds."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(40):
        try:
            os.close(os.open(lock, flags, 0o600))
            return True
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                    lock.unlink()
                    continue
            except OSError:
                pass
            time.sleep(0.05)
        except OSError:
            return False
    return False


def _release_replacement_lock(lock: pathlib.Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


def _discard_stale(path: pathlib.Path) -> None:
    """Delete a token file that holds no token, if it is old and is still the file that was seen.

    Raises ``ApiTokenError`` for a file that holds something else, or one too new to be sure
    another worker is not still writing it.
    """
    try:
        seen = path.stat()
        try:
            blank = not path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raise  # gone already: handled below as "someone else removed it"
        except (OSError, UnicodeDecodeError):
            blank = False
        if not blank:
            raise ApiTokenError("the generated token file holds no usable token; delete it")
        if time.time() - seen.st_mtime < STALE_EMPTY_S:
            raise ApiTokenError(
                "the generated token file is blank (another worker may be writing it)"
            )
        now = path.stat()
        if (now.st_ino, now.st_size) != (seen.st_ino, seen.st_size):
            raise ApiTokenError("the generated token file changed while it was being replaced")
        path.unlink()
    except FileNotFoundError:
        pass  # someone else removed it first
    except OSError:
        raise ApiTokenError("the generated token file is blank and cannot be replaced") from None


def _create_token(path: pathlib.Path, logger: logging.Logger) -> tuple[str, str]:
    """Make a token and publish it at ``path`` (see ``_publish_exclusively``); on losing the race to
    another worker, return that worker's token."""
    token = secrets.token_urlsafe(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _remove_stale_temporaries(path.parent)
        created = _publish_exclusively(path, token)
    except OSError as e:
        raise ApiTokenError(
            f"no token is configured and one cannot be saved in the data folder ({type(e).__name__})"
        ) from None
    if not created:
        again = _read_generated_waiting(path)  # another worker made it first
        if again is None:
            raise ApiTokenError("the generated token file is empty") from None
        return again, "file"
    logger.warning(
        "No %s was configured, so EMHASS generated one and saved it to %s. Send it as "
        "'Authorization: Bearer <token>'. In the Home Assistant add-on, open the web UI and "
        "choose API token.",
        SETTING_NAME,
        path,
    )
    if _IS_WINDOWS:
        logger.warning(
            "On Windows the token file is not restricted to its owner. Set %s instead, or keep "
            "the data folder private.",
            ENV_NAME,
        )
    return token, "new"


def provision_token(data_dir: pathlib.Path | str, logger: logging.Logger) -> tuple[str, str]:
    """Return ``(token, how)`` from the token file in ``data_dir``, making it when it is missing.

    ``how`` is "file" or "new". The file appears complete or not at all and with mode 0600
    (POSIX), so workers starting together agree on one token and a crash cannot leave a
    half-written one. A blank file older than ``STALE_EMPTY_S`` (left by an older version that
    crashed, or by hand) is replaced under a lock, so workers that find it together converge on
    one token. Raises ``ApiTokenError`` when it cannot be read or written, which fails the server
    closed.
    """
    path = pathlib.Path(data_dir) / TOKEN_FILE_NAME
    existing = _read_generated_waiting(path)
    if existing is not None:
        return existing, "file"
    if not path.exists():
        return _create_token(path, logger)
    lock = path.with_name(f".{TOKEN_FILE_NAME}.lock")
    if not _acquire_replacement_lock(lock):
        again = _read_generated_waiting(path)
        if again is not None:
            return again, "file"
        raise ApiTokenError("the generated token file is being replaced by another process")
    try:
        again = _read_generated(path)  # another worker may have finished while we waited
        if again is not None:
            return again, "file"
        _discard_stale(path)
        return _create_token(path, logger)
    finally:
        _release_replacement_lock(lock)


def configure_now(
    secrets_path: pathlib.Path | str | None,
    options_path: pathlib.Path | str | None,
    logger: logging.Logger,
    data_dir: pathlib.Path | str | None = None,
) -> None:
    """Load the token for the request hook. Never logs the token itself.

    Runs before the logging level is set, so only a failure and the one-time
    notice about a generated token are logged here. ``log_status()`` reports the rest once
    logging is set up. With ``data_dir`` and no configured token, one is generated and kept there.
    """
    try:
        token = read_api_token(secrets_path, options_path)
        source = "configured"
        if token is not None and len(token) < MIN_LENGTH:
            _fail(
                logger,
                f"{SETTING_NAME} is shorter than {MIN_LENGTH} characters; use a long random value",
            )
            return
        if (
            token is None
            and data_dir is not None
            and not auth_disabled()
            and _autogenerate_enabled()
        ):
            token, source = provision_token(data_dir, logger)
            source = "generated-new" if source == "new" else "generated"
    except Exception as e:
        _fail(logger, f"Could not read {SETTING_NAME} ({_describe(e)})")
        return
    _state.update(loaded=True, token=token, failed=False, source=source if token else None)


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
    if _state["failed"] or _state["token"] is not None or auth_disabled():
        return
    try:
        token = read_api_token(secrets_path, options_path)
    except Exception as e:
        _fail(logger, f"Could not read {SETTING_NAME} ({_describe(e)})")
        return
    if token is not None:
        _fail(
            logger,
            f"{SETTING_NAME} was set after EMHASS started loading; restart EMHASS to turn "
            "API authentication on",
        )


async def configure(
    secrets_path: pathlib.Path | str | None,
    options_path: pathlib.Path | str | None,
    logger: logging.Logger,
    data_dir: pathlib.Path | str | None = None,
) -> None:
    """``configure_now`` for async callers."""
    configure_now(secrets_path, options_path, logger, data_dir)


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
        "%s. Every request except /healthz and /logout will get 503 until this is fixed "
        "and EMHASS is restarted.",
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
        return _normalise(os.environ.get(ENV_NAME))
    except ApiTokenError:
        return None


def redacted(value):
    """``value`` for a log line, with an ``emhass_api_token`` key masked.

    Only that top-level key; ``TokenLogFilter`` masks the token by value
    wherever else it turns up.
    """
    if isinstance(value, dict) and SETTING_NAME in value:
        return {**value, SETTING_NAME: "***"}
    return value


def _token_forms(token: str) -> list[str]:
    """The token as it can appear in a log line: as is, escaped the way
    ``repr()`` (a dict in a log line) and JSON write it, for a token that holds
    a backslash, a quote, a control or a non-ASCII character, and as it looks
    in a URL (an access log line). Longest first."""
    forms = {
        token,
        repr(token)[1:-1],
        json.dumps(token)[1:-1],
        json.dumps(token, ensure_ascii=False)[1:-1],
        urllib.parse.quote(token, safe=""),
        urllib.parse.quote_plus(token, safe=""),
        urllib.parse.quote(token),
    }
    return sorted(forms, key=len, reverse=True)


def _mask(text: str, forms: list[str]) -> str:
    for form in forms:
        text = text.replace(form, "***")
    return text


class TokenLogFilter(logging.Filter):
    """Masks the configured token in a log record's message and traceback.

    The token is never read from a request body, but a client that puts it
    there by mistake (nested, under another key, in a YAML error) must not get
    it written to the console or the action log, whose lines are returned in
    error responses. ``install_log_masking()`` runs it on every record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        token = current_token()
        if not token:
            return True
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record is the formatter's to report
            return True
        forms = _token_forms(token)
        masked = _mask(message, forms)
        if masked != message:
            record.msg = masked
            record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = _mask(record.exc_text, forms)
        if record.stack_info:
            record.stack_info = _mask(record.stack_info, forms)
        return True


token_log_filter = TokenLogFilter()
_previous_record_factory = None


def _masking_record_factory(*args, **kwargs) -> logging.LogRecord:
    record = _previous_record_factory(*args, **kwargs)
    token_log_filter.filter(record)
    return record


def install_log_masking() -> None:
    """Mask the token in every log record made in this process.

    It runs as each record is created, so it covers every logger (the app's,
    the other ``emhass.*`` loggers, Gunicorn's and Uvicorn's, which print
    request paths) whatever handler writes the record. Records are only
    created for enabled levels, so the cost is one search per logged line.
    Safe to call more than once.
    """
    global _previous_record_factory
    if _previous_record_factory is not None:
        return
    _previous_record_factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(_masking_record_factory)


def reset() -> None:
    """Forget the loaded token. For tests."""
    _state.update(loaded=False, token=None, failed=False, source=None)
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
    """Start-up report of the access control state. Never logs the token."""
    if _state["failed"]:
        return  # configure() already logged the error
    if auth_disabled():
        logger.warning(
            "API access control is OFF (%s is set). Any device that can reach this port can "
            "change the configuration and run actions.",
            DISABLE_ENV,
        )
        return
    if current_token() is None:
        logger.error(
            "API access control is on but no %s is available and none can be generated "
            "(%s=false). Every request except /healthz, /logout and Home Assistant ingress gets 503. Set %s.",
            SETTING_NAME,
            AUTOGENERATE_ENV,
            ENV_NAME,
        )
        return
    how = {
        "configured": "a configured token",
        "generated": "a generated token kept in the data folder",
        "generated-new": "a newly generated token kept in the data folder",
    }.get(_state["source"], "a token")
    port = ingress_port()
    logger.info(
        "API access control is on (%s)%s",
        how,
        f"; Home Assistant ingress is served on port {port}" if port else "",
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


def _api_port() -> int:
    """The API port (``PORT``, as gunicorn.conf.py reads it)."""
    try:
        return int(os.environ.get("PORT", "5000").strip())
    except ValueError:
        return 5000


def ingress_port() -> int | None:
    """The port of the ingress listener, or None when there is none.

    ``EMHASS_INGRESS_PORT``, else 8099 when running as a Home Assistant add-on (the Supervisor sets
    ``SUPERVISOR_TOKEN``). ``gunicorn.conf.py`` makes the same decision when it chooses what to
    listen on; a test keeps the two in step. Never the API port: that would refuse every API client.
    """
    value = os.environ.get(INGRESS_PORT_ENV, "").strip()
    if not value:
        if not os.environ.get("SUPERVISOR_TOKEN"):
            return None
        value = str(DEFAULT_INGRESS_PORT)
    try:
        port = int(value)
    except ValueError:
        return None
    if port == _api_port():
        return None
    return port if 0 < port < 65536 else None


def _ingress_networks() -> list:
    networks = []
    for item in os.environ.get(INGRESS_PEERS_ENV, DEFAULT_INGRESS_PEERS).split(","):
        item = item.strip()
        if item:
            try:
                networks.append(ipaddress.ip_network(item, strict=False))
            except ValueError:
                continue  # an entry that is not an address is ignored, which only narrows trust
    return networks


def _peer_is_supervisor(address: str | None) -> bool:
    try:
        ip = ipaddress.ip_address(address or "")
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return any(ip.version == net.version and ip in net for net in _ingress_networks())


def _scope_pair(key: str, scope=None):
    """(host, port) from the ASGI scope, which the server fills from the socket, or None."""
    value = (request.scope if scope is None else scope).get(key)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[0], value[1]
    return None


def on_ingress_port(scope=None) -> bool:
    """True when this request (or WebSocket handshake) was accepted by the ingress listener."""
    port = ingress_port()
    server = _scope_pair("server", scope)
    return port is not None and server is not None and server[1] == port


def is_trusted_ingress_request(scope=None) -> bool:
    """True for a request Home Assistant's ingress proxy made.

    Two facts must both hold: the connection was accepted by the ingress listener (the port
    only the Supervisor is meant to reach, and one the add-on does not publish), and its peer
    is the Supervisor (``EMHASS_INGRESS_PEERS``, default 172.30.32.2, the address Home
    Assistant's developer documentation says to accept ingress from). Headers such as
    ``X-Ingress-Path``, ``X-Forwarded-For`` and ``X-Remote-User-*`` are never consulted: a
    client that reaches a listener can set them. The server must not rewrite the peer address
    from ``X-Forwarded-For`` (``gunicorn.conf.py`` and ``main()`` switch that off).
    """
    if not on_ingress_port(scope):
        return False
    peer = _scope_pair("client", scope)
    return peer is not None and _peer_is_supervisor(peer[0])


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

    ``X-Forwarded-Proto`` and ``Forwarded: proto=...`` are honoured even from
    an untrusted client: all they can do is make its own browser keep the
    cookie off plain HTTP.
    """
    if request.scheme == "https":
        return True
    forwarded = request.headers.get("X-Forwarded-Proto", "")
    if forwarded.split(",")[0].strip(_HEADER_SPACE).lower() == "https":
        return True
    first = request.headers.get("Forwarded", "").split(",")[0]
    for pair in first.split(";"):
        name, _, value = pair.strip(_HEADER_SPACE).partition("=")
        if name.lower() == "proto" and value.strip('"').lower() == "https":
            return True
    return False


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


def _ingress_refused() -> Response:
    return _plain(
        403,
        "EMHASS >> This port only accepts Home Assistant ingress. Use the API port.\n",
    )


async def require_api_token():
    """before_request hook. Returns None to let the request through."""
    if request.endpoint in OPEN_ENDPOINTS:
        return None
    if _state["failed"]:
        return _unavailable()
    if auth_disabled():
        return None
    if on_ingress_port():
        # The ingress listener serves Home Assistant only. Anything else that reaches it is
        # refused whatever it carries; the Supervisor has already signed its user in.
        if not is_trusted_ingress_request():
            return _ingress_refused()
        if _cookie_request_allowed():
            return None
        return _forbidden_cookie_post()
    token = current_token()
    if token is None:
        return _unavailable()
    if _bearer_matches(token):
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


# The group every Home Assistant administrator is in (homeassistant.auth.const.GROUP_ID_ADMIN)
ADMIN_GROUP = "system-admin"


def ha_administrator_request() -> bool:
    """True when the Supervisor says an administrator made this request.

    The Supervisor adds ``X-Remote-User-Id`` and ``X-Remote-User-Groups`` to what it forwards and
    removes any copy a client sent (``_REQUEST_HEADERS_FILTER`` in its ingress proxy). They can be
    believed here only because ``is_trusted_ingress_request`` already restricts this port to the
    Supervisor. A request with no user, or a user that is not an administrator, is refused, so the
    page stays admin-only even for an add-on that sets ``panel_admin: false``.
    """
    if not request.headers.get("X-Remote-User-Id", "").strip():
        return False
    groups = [g.strip() for g in request.headers.get("X-Remote-User-Groups", "").split(",")]
    return ADMIN_GROUP in groups


def token_page_available() -> bool:
    """True when this request may open /api-token: it came through ingress and the token is
    one EMHASS generated. The web UI shows its link only then."""
    return (
        is_trusted_ingress_request()
        and ha_administrator_request()
        and current_token() is not None
        and _state["source"] in ("generated", "generated-new")
    )


async def api_token_page():
    """Show a generated token to a Home Assistant user who came in through ingress.

    Only for ingress (Home Assistant has signed the user in) and only for a token EMHASS made
    itself: a configured token is already known to whoever configured it. Never cached.
    """
    if not is_trusted_ingress_request():
        return _plain(
            403, "EMHASS >> The API token is shown through Home Assistant ingress only.\n"
        )
    mode = request.headers.get("Sec-Fetch-Mode")
    if (mode is not None and mode.lower() != "navigate") or request.headers.get(
        "Sec-Fetch-Site", ""
    ).lower() == "cross-site":
        # A browser sets these and a page's script cannot change them: script in another add-on's
        # panel (the same Home Assistant origin) must not be able to read the token with fetch().
        return _plain(403, "EMHASS >> Open the API token as a page, not from a script.\n")
    if not ha_administrator_request():
        return _plain(
            403, "EMHASS >> The API token is shown to Home Assistant administrators only.\n"
        )
    if not token_page_available():
        return _plain(404, "EMHASS >> There is no generated API token.\n")
    response = _plain(200, current_token() + "\n")
    response.headers["Referrer-Policy"] = "no-referrer"
    # sandbox without allow-same-origin gives the document an opaque origin: a script on this origin
    # (another add-on's panel) that frames or opens the page cannot read its text
    response.headers["Content-Security-Policy"] = "sandbox"
    return response


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
        return _redirect("./") if auth_disabled() else _unavailable()
    if request.method != "POST":
        return _login_page()
    fields, refused = await _read_login_form()
    if refused is not None:
        return refused
    presented = fields.get("token", [""])[0].strip(_TOKEN_SPACE).encode("utf-8")
    if presented and hmac.compare_digest(presented, _token_bytes(token)):
        response = _redirect("./")
        response.set_cookie(
            SESSION_COOKIE,
            make_session_value(token),
            max_age=SESSION_MAX_AGE_S,
            # No Path: the browser scopes the cookie to the folder /login was served from, so behind
            # an ingress prefix it is not sent to the rest of the origin
            path=None,
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
        path=None,
        httponly=True,
        samesite="Strict",
        secure=_came_over_https(),
    )
    return response


async def require_api_token_websocket():
    """before_websocket hook: the same rule for a WebSocket handshake.

    EMHASS has no WebSocket route today; this keeps a later one from being
    open by default. On the API port only the bearer header counts here, not the cookie: a
    handshake is not covered by the JSON rule, so a cookie would let a page on
    another port of the same host open one.
    """
    if _state["failed"]:
        return _unavailable()
    if auth_disabled():
        return None
    if on_ingress_port(websocket.scope):
        # same rule as an ordinary request: the Supervisor only, whatever else is presented
        return None if is_trusted_ingress_request(websocket.scope) else _ingress_refused()
    token = current_token()
    if token is None:
        return _unavailable()
    if _bearer_matches(token, websocket.headers):
        return None
    if _bearer_presented(websocket.headers):
        _note_failure()
    return _unauthorized()


def install(app: Quart) -> None:
    """Attach the hooks and the login routes to ``app``."""
    app.before_request(require_api_token)
    app.before_websocket(require_api_token_websocket)
    app.add_url_rule("/api-token", "api_auth_token", api_token_page, methods=["GET"])
    app.add_url_rule("/login", "api_auth_login", login, methods=["GET", "POST"])
    app.add_url_rule("/logout", "api_auth_logout", logout, methods=["GET", "POST"])
