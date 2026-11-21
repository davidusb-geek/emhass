"""CWE-117 regression tests for the dry-run endpoint's log lines.

``POST /api/v1/dry-run/<action_name>`` takes the action name from the URL, and
an exception raised while serving it can carry text from the request body. Both
reach the console log, where a carriage return, a newline, an ANSI escape or a
Unicode line separator would forge a log line or drive the terminal.
"""

import asyncio
import json
import logging
import urllib.parse
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from emhass import api_auth, web_server
from emhass.utils import InverterConfigError

FORMAT_CHARS = tuple(map(chr, (0x202E, 0x200B, 0xFEFF)))
HOSTILE = "x\r\nERROR - forged\x1b[31m\x7f\u2028\u2029\x85\t\x00" + "".join(FORMAT_CHARS)
SPECIALS = ("\r", "\n", "\x1b", "\x7f", "\u2028", "\u2029", "\x85", "\t", "\x00", *FORMAT_CHARS)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture(autouse=True)
def _data_path(tmp_path):
    saved = web_server.emhass_conf
    web_server.emhass_conf = {**saved, "data_path": tmp_path}
    yield
    web_server.emhass_conf = saved


@pytest.fixture(autouse=True)
def _app_logger_level():
    """Do not rely on the ambient level: another test module lowers the
    "emhass" logger to CRITICAL at import, which a full run inherits and which
    would drop the ERROR record the dry-run response collects."""
    saved = web_server.app.logger.level
    web_server.app.logger.setLevel(logging.INFO)
    yield
    web_server.app.logger.setLevel(saved)


@pytest.fixture
def logged():
    capture = _Capture()
    saved = web_server.app.logger.level
    web_server.app.logger.addHandler(capture)
    web_server.app.logger.setLevel(logging.DEBUG)
    try:
        yield capture.messages
    finally:
        web_server.app.logger.removeHandler(capture)
        web_server.app.logger.setLevel(saved)


def _post(name: str, body: dict | None = None):
    async def go():
        return await web_server.app.test_client().post(
            "/api/v1/dry-run/" + urllib.parse.quote(name, safe=""), json=body or {}
        )

    return asyncio.run(go())


def _assert_single_safe_line(messages: list[str]) -> None:
    assert messages, "nothing was logged"
    for message in messages:
        for special in SPECIALS:
            assert special not in message, (repr(special), message)
        assert len(message) < 700, len(message)


def test_failure_before_validation_logs_the_url_action_name_safely(logged):
    """app.logger.exception(f"Dry run of {action_name} failed: ...")."""
    with patch.object(web_server, "_open_lock_slot", side_effect=RuntimeError("disk")):
        response = _post(HOSTILE + "A" * 2000)
    assert response.status_code == 500
    _assert_single_safe_line(logged)
    assert any(m.startswith("Dry run of ") and "failed: disk" in m for m in logged)


def test_exception_text_from_the_request_is_logged_safely(logged):
    """The exception message can carry request-body text."""
    with patch.object(web_server, "_dry_run", AsyncMock(side_effect=ValueError(HOSTILE * 80))):
        response = _post("naive-mpc-optim")
    assert response.status_code == 500
    _assert_single_safe_line(logged)


def test_unsupported_action_name_is_logged_safely_and_bounded(logged):
    response = _post(HOSTILE + "B" * 5000)
    assert response.status_code == 400
    _assert_single_safe_line(logged)
    assert any("A dry run is for" in m and "Nothing was run" in m for m in logged)


def test_ordinary_messages_keep_their_wording(logged):
    with patch.object(
        web_server, "_dry_run", AsyncMock(side_effect=RuntimeError("solver crashed: ça va"))
    ):
        _post("naive-mpc-optim")
    assert "Dry run of naive-mpc-optim failed: solver crashed: ça va" in logged
    logged.clear()
    _post("bogus-action")
    assert any(
        m == "A dry run is for perfect-optim, dayahead-optim, naive-mpc-optim, "
        "not 'bogus-action'. Nothing was run"
        for m in logged
    )


def test_response_body_still_reports_the_failure():
    with patch.object(web_server, "_dry_run", AsyncMock(side_effect=RuntimeError("boom"))):
        response = _post("naive-mpc-optim")
        body = asyncio.run(response.get_json())
    assert response.status_code == 500
    assert body["status"] == "not-run"
    assert any("boom" in e for e in body["errors"])


class _Formatted(logging.Handler):
    """Collects each record as a real formatter writes it, traceback included."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


@pytest.fixture
def formatted():
    handler = _Formatted()
    saved = web_server.app.logger.level
    web_server.app.logger.addHandler(handler)
    web_server.app.logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        web_server.app.logger.removeHandler(handler)
        web_server.app.logger.setLevel(saved)


def _failing_dry_run(exc: BaseException):
    with patch.object(web_server, "_dry_run", AsyncMock(side_effect=exc)):
        return _post("naive-mpc-optim")


def test_formatted_traceback_cannot_be_forged_by_exception_text(formatted):
    """logger.exception() appends 'ValueError: <message>' to the record; a
    hostile message must not add physical lines or control characters there."""
    _failing_dry_run(ValueError("benign"))
    benign = formatted[-1]
    formatted.clear()
    _failing_dry_run(ValueError(HOSTILE * 80))
    assert len(formatted) == 1, "one ERROR record per failure"
    hostile = formatted[0]
    assert hostile.splitlines() == hostile.split("\n"), "only \n separates lines"
    for special in SPECIALS[:1] + SPECIALS[2:]:
        assert special not in hostile, repr(special)
    assert len(hostile.splitlines()) == len(benign.splitlines())
    assert not any(line.startswith("ERROR - forged") for line in hostile.splitlines())
    assert len(hostile) < len(benign) + 1500


def test_formatted_traceback_drops_hostile_chained_exception_text(formatted):
    try:
        try:
            raise OSError("INNER-MARKER " + HOSTILE)
        except OSError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as exc:
        chained = exc
    _failing_dry_run(chained)
    text = formatted[0]
    for special in SPECIALS[:1] + SPECIALS[2:]:
        assert special not in text, repr(special)
    assert "RuntimeError: outer" in text
    assert "INNER-MARKER" not in text, "the chained exception must not be printed at all"


def test_ordinary_exception_keeps_type_message_and_frames(formatted):
    _failing_dry_run(KeyError("sensor_x"))
    text = formatted[0]
    assert text.startswith("ERROR - emhass.web_server - Dry run of naive-mpc-optim failed: ")
    assert "Traceback (most recent call last):" in text
    assert "web_server.py" in text and "api_v1_dry_run" in text
    assert text.splitlines()[-1] == "KeyError: 'sensor_x'"
    assert len(formatted) == 1


def test_log_safe_output_is_bounded_even_when_every_character_expands():
    """An escaped character takes at most ten characters (a U+XXXXXXXX escape)."""
    for char, width in (("\x00", 4), (chr(0xE0001), 10)):
        worst = web_server._log_safe(char * 10**6)
        assert len(worst) <= 200 * width + len("...[truncated]")
    assert web_server._log_safe("ok text") == "ok text"


# --------------------------------------------------------------------------
# The API token must be masked before anything is truncated or escaped
# --------------------------------------------------------------------------

TOKEN = "TqK7vXw9RmZ4bNc2HdL8sYf5JgP3uE6a"  # synthetic
TAB_TOKEN = "TabbedQ7Kp3\tSepVq7Kp3ZxW9"  # synthetic, with an embedded tab


@contextmanager
def _token(value: str):
    saved = dict(api_auth._state)
    api_auth._state.update(loaded=True, token=value, failed=False, source="test")
    try:
        yield
    finally:
        api_auth._state.clear()
        api_auth._state.update(saved)


def _assert_no_token_fragment(text: str, token: str, width: int = 6) -> None:
    """Neither the token nor any run of `width` of its characters appears."""
    for part in token.split("\t"):
        for i in range(max(len(part) - width + 1, 0)):
            assert part[i : i + width] not in text, (part[i : i + width], text[:300])


def _everything_logged(logged, formatted, response) -> str:
    body = asyncio.run(response.get_json())
    return "\n".join([*logged, *formatted, *body["errors"]])


@pytest.mark.parametrize("token", [TOKEN, TAB_TOKEN])
@pytest.mark.parametrize("offset", [0, 150, 190, 195, 199, 200, 260])
def test_token_is_masked_before_truncation(logged, formatted, token, offset):
    """A token that straddles the 200-character cut must not leave a prefix."""
    boom = ValueError("y" * offset + token + " tail")
    with _token(token), patch.object(web_server, "_dry_run", AsyncMock(side_effect=boom)):
        response = _post("naive-mpc-optim")
    assert response.status_code == 500
    everything = _everything_logged(logged, formatted, response)
    _assert_no_token_fragment(everything, token)
    if offset <= 197:  # the mask is inside the 200 kept characters
        assert "***" in everything


@pytest.mark.parametrize("token", [TOKEN, TAB_TOKEN])
@pytest.mark.parametrize(
    "render",
    [
        lambda t: t,
        repr,
        json.dumps,
        lambda t: urllib.parse.quote(t, safe=""),
        lambda t: "'" + t + "'",
    ],
    ids=["raw", "repr", "json", "url", "quoted"],
)
def test_token_in_escaped_and_quoted_forms_is_masked(logged, formatted, token, render):
    boom = ValueError("bad value " + render(token))
    with _token(token), patch.object(web_server, "_dry_run", AsyncMock(side_effect=boom)):
        response = _post("naive-mpc-optim")
    _assert_no_token_fragment(_everything_logged(logged, formatted, response), token)


@pytest.mark.parametrize("token", [TOKEN, TAB_TOKEN])
def test_token_in_a_key_error_is_masked(logged, formatted, token):
    """str(KeyError(x)) is repr(x), a third spelling of the token."""
    with (
        _token(token),
        patch.object(web_server, "_dry_run", AsyncMock(side_effect=KeyError(token))),
    ):
        response = _post("naive-mpc-optim")
    _assert_no_token_fragment(_everything_logged(logged, formatted, response), token)


@pytest.mark.parametrize("token", [TOKEN, TAB_TOKEN])
def test_token_in_the_url_action_name_is_masked(logged, formatted, token):
    """On both sinks: the 500 before validation and the 400 for an unknown action."""
    with _token(token):
        unknown = _post(token + "x")
        with patch.object(web_server, "_open_lock_slot", side_effect=RuntimeError("disk")):
            failed = _post(token)
    assert unknown.status_code == 400 and failed.status_code == 500
    errors = asyncio.run(unknown.get_json())["errors"] + asyncio.run(failed.get_json())["errors"]
    _assert_no_token_fragment("\n".join([*logged, *formatted, *errors]), token)


# --------------------------------------------------------------------------
# The response keeps its shape; hostile or oversized text is bounded and escaped
# --------------------------------------------------------------------------


def test_ordinary_error_response_is_unchanged():
    boom = RuntimeError("solver crashed")
    with patch.object(web_server, "_dry_run", AsyncMock(side_effect=boom)):
        response = _post("naive-mpc-optim")
    body = asyncio.run(response.get_json())
    assert response.status_code == 500
    assert body["errors"] == ["Dry run of naive-mpc-optim failed: solver crashed"]
    assert {"dry_run", "action", "status", "plan", "warnings", "errors"} <= set(body)
    assert body["status"] == "not-run" and body["plan"] is None


def test_oversized_error_text_is_bounded_in_the_response_and_stays_intelligible():
    boom = ValueError("first line\n" + "z" * 5000)
    with patch.object(web_server, "_dry_run", AsyncMock(side_effect=boom)):
        response = _post("naive-mpc-optim")
    body = asyncio.run(response.get_json())
    assert response.status_code == 500 and len(body["errors"]) == 1
    message = body["errors"][0]
    assert message.startswith("Dry run of naive-mpc-optim failed: first line")
    assert message.endswith("...[truncated]") and "\n" not in message
    assert len(message) < 200 * 4 + 120


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("naive-mpc-optim", "naive-mpc-optim"),
        ("solver crashed: ça va", "solver crashed: ça va"),
        ("a" + chr(92) + "nb", "a" + chr(92) * 2 + "nb"),  # a literal backslash stays distinct
        ("\r\n", chr(92) + "x0d" + chr(92) + "x0a"),
        ("\x1b[31m", chr(92) + "x1b[31m"),
        ("\x7f", chr(92) + "x7f"),
        ("\x85", chr(92) + "x85"),
    ],
)
def test_log_safe_escapes_backslashes_and_controls_but_keeps_ordinary_text(raw, expected):
    assert web_server._log_safe(raw) == expected


@pytest.mark.parametrize("codepoint", [0x2028, 0x2029, 0x202E, 0x200B, 0xFEFF, 0xA0, 0xE0001])
def test_log_safe_escapes_non_printable_unicode(codepoint):
    kind = "x%02x" if codepoint < 0x100 else "u%04x" if codepoint < 0x10000 else "U%08x"
    expected = "a" + chr(92) + kind % codepoint + "b"
    assert web_server._log_safe("a" + chr(codepoint) + "b") == expected


def test_unsupported_action_message_is_no_less_escaped_than_the_old_repr(logged):
    """The old text was {action_name!r}: bidi overrides and backslashes were escaped."""
    _post("evil" + chr(0x202E) + "name")
    assert not any(chr(0x202E) in m for m in logged)
    logged.clear()
    _post("a" + chr(92) + "nERROR forged")
    assert any("not 'a" + chr(92) * 2 + "nERROR forged'" in m for m in logged)


# --------------------------------------------------------------------------
# Every warning or error a dry run logs is bounded and safe, whoever logs it
# --------------------------------------------------------------------------

LINE_LIMIT = web_server._DRY_RUN_LINE_LIMIT


def _problems(logged, response) -> list[str]:
    errors = asyncio.run(response.get_json())["errors"]
    return [*logged, *errors]


def _assert_bounded_and_safe(messages: list[str]) -> None:
    assert messages, "nothing was logged"
    for message in messages:
        for special in SPECIALS:
            assert special not in message, (repr(special), message[:80])
        assert len(message) <= LINE_LIMIT + len("...[truncated]") + 100, len(message)


@pytest.mark.parametrize("flag", ["dry_run", "entity_save", "weather_forecast_cache"])
def test_a_refused_flag_value_is_bounded_in_the_log_and_the_response(logged, flag):
    """read_flag puts the sent value in its message, up to the 16 MB body limit."""
    response = _post("naive-mpc-optim", {flag: "A" * 5000 + HOSTILE})
    assert response.status_code == 400
    _assert_bounded_and_safe(_problems(logged, response))


def _stub_inputs(build):
    """A dry run whose configuration loads and whose inputs `build` makes."""
    return (
        patch.object(web_server, "_load_params", AsyncMock(return_value=({"x": 1}, "profit"))),
        patch.object(web_server, "set_input_data_dict", build),
    )


def test_a_message_logged_while_the_inputs_are_built_is_bounded_and_safe(logged):
    """treat_runtimeparams logs an inverter fault with logger.error("...: %s", problem)
    on the logger it is given, before any handler of the endpoint runs."""

    async def build(*args, **kwargs):
        args[5].error(
            "number_of_inverters must be a positive integer, got %s", "A" * 5000 + HOSTILE
        )

    load, inputs = _stub_inputs(build)
    with load, inputs:
        response = _post("naive-mpc-optim")
    assert response.status_code == 400
    _assert_bounded_and_safe(_problems(logged, response))


def test_an_inverter_config_error_is_bounded_and_safe_in_the_log_and_the_response(logged):
    """_explain(str(exc)) for an InverterConfigError raised by the run."""
    fault = InverterConfigError("inverter_ac_output_max: bad value " + "A" * 5000 + HOSTILE)
    load, inputs = _stub_inputs(AsyncMock(return_value={"x": 1}))
    with load, inputs, patch.object(web_server, "naive_mpc_optim", AsyncMock(side_effect=fault)):
        response = _post("naive-mpc-optim")
    assert response.status_code == 400
    _assert_bounded_and_safe(_problems(logged, response))


def test_an_ordinary_inverter_message_is_unchanged(logged):
    """No extra escaping: the message already carries repr's own backslashes."""
    ordinary = "inverter_ac_output_max[0]: non-numeric string element " + repr("a" + chr(92) + "nb")
    load, inputs = _stub_inputs(AsyncMock(return_value={"x": 1}))
    with (
        load,
        inputs,
        patch.object(
            web_server, "naive_mpc_optim", AsyncMock(side_effect=InverterConfigError(ordinary))
        ),
    ):
        response = _post("naive-mpc-optim")
    assert response.status_code == 400
    assert ordinary in logged
    assert asyncio.run(response.get_json())["errors"] == [ordinary]


def test_informational_dry_run_lines_are_not_cut(logged):
    """Only warnings and errors are bounded: the INFO line that echoes the request stays whole."""

    async def build(*args, **kwargs):
        args[5].info("x" * 3000)
        args[5].error("stop")

    load, inputs = _stub_inputs(build)
    with load, inputs:
        _post("naive-mpc-optim")
    assert "x" * 3000 in logged


def test_a_live_log_record_is_not_touched():
    """Outside a dry run the filter does nothing, so action_logs.txt lines keep their form."""
    record = logging.LogRecord("emhass", logging.ERROR, __file__, 1, "a %s", ("b" * 5000,), None)
    assert web_server._dry_run_record_safe(record) is True
    assert record.getMessage() == "a " + "b" * 5000


def test_an_exception_logged_while_the_inputs_are_built_is_escaped_in_the_traceback(formatted):
    """logger.exception() prints 'Type: message' and any chained exception itself,
    and logging's stack_info prints the call stack: none may carry request text raw."""

    async def build(*args, **kwargs):
        try:
            try:
                raise OSError("CHAINED-MARKER " + HOSTILE)
            except OSError as inner:
                raise ValueError("INNER " + HOSTILE * 40) from inner
        except ValueError:
            args[5].exception("The inputs failed")
        args[5].warning("with a stack", stack_info=True)

    load, inputs = _stub_inputs(build)
    with load, inputs:
        _post("naive-mpc-optim")
    failed = next(r for r in formatted if "The inputs failed" in r)
    stacked = next(r for r in formatted if "with a stack" in r)
    for text in (failed, stacked):
        for special in SPECIALS[:1] + SPECIALS[2:]:
            assert special not in text, repr(special)
        assert not any(line.startswith("ERROR - forged") for line in text.splitlines())
    assert "Traceback (most recent call last):" in failed and "ValueError: INNER" in failed
    assert "CHAINED-MARKER" not in failed
    assert len(failed) < 3000
    assert "Stack (most recent call last):" in stacked


# --------------------------------------------------------------------------
# With an API token configured, the masking record factory pre-renders exc_text
# --------------------------------------------------------------------------


def _hostile_chain() -> BaseException:
    try:
        try:
            raise OSError("CHAINED-MARKER " + HOSTILE)
        except OSError as inner:
            raise ValueError("OUTER " + TOKEN + HOSTILE + "A" * 5000) from inner
    except ValueError as exc:
        return exc


@pytest.mark.parametrize("route", ["the 500 handler", "a logger.exception call in the run"])
def test_a_configured_token_does_not_bring_back_the_raw_traceback(formatted, route):
    """The record factory caches exc_text from the original exception when a token
    is set; the filter's safe exception must replace that text, not sit beside it."""

    async def build(*args, **kwargs):
        try:
            raise _hostile_chain()
        except ValueError:
            args[5].exception("The inputs failed")

    with _token(TOKEN):
        if route == "the 500 handler":
            _failing_dry_run(_hostile_chain())
        else:
            load, inputs = _stub_inputs(build)
            with load, inputs:
                _post("naive-mpc-optim")
    text = next(r for r in formatted if "Traceback (most recent call last):" in r)
    for special in SPECIALS[:1] + SPECIALS[2:]:
        assert special not in text, repr(special)
    assert not any(line.startswith(("ERROR - forged", "forged")) for line in text.splitlines())
    assert "CHAINED-MARKER" not in text
    assert "ValueError: OUTER" in text
    assert len(text) < 3000
    _assert_no_token_fragment(text, TOKEN)


# --------------------------------------------------------------------------
# A record that cannot be rendered is replaced, never raised and never passed on
# --------------------------------------------------------------------------


class _UnprintableArg:
    def __str__(self) -> str:
        raise RuntimeError("no string")


@pytest.mark.parametrize(
    "emit",
    [
        lambda log: log.warning("bad %d", "x" + HOSTILE),
        lambda log: log.error("%s and %s", HOSTILE),
        lambda log: log.error("%s", _UnprintableArg()),
    ],
    ids=["wrong type", "too few arguments", "argument that cannot be printed"],
)
def test_a_malformed_log_call_neither_raises_into_the_dry_run_nor_leaks(logged, formatted, emit):
    async def build(*args, **kwargs):
        emit(args[5])

    load, inputs = _stub_inputs(build)
    with load, inputs:
        response = _post("naive-mpc-optim")
    assert response.status_code == 400, "a bad log call must not turn the request into a 500"
    errors = asyncio.run(response.get_json())["errors"]
    everything = "\n".join([*logged, *errors, *formatted])
    assert web_server._UNRENDERABLE_RECORD in errors or web_server._UNRENDERABLE_RECORD in logged
    for special in SPECIALS[:1] + SPECIALS[2:]:
        assert special not in everything, repr(special)
    assert "forged" not in everything
