"""CWE-117 regression tests for the dry-run endpoint's log lines.

``POST /api/v1/dry-run/<action_name>`` takes the action name from the URL, and
an exception raised while serving it can carry text from the request body. Both
reach the console log, where a carriage return, a newline, an ANSI escape or a
Unicode line separator would forge a log line or drive the terminal.
"""

import asyncio
import logging
import urllib.parse
from unittest.mock import AsyncMock, patch

import pytest

from emhass import web_server

HOSTILE = "x\r\nERROR - forged\x1b[31m\x7f  \x85\t\x00"
SPECIALS = ("\r", "\n", "\x1b", "\x7f", " ", " ", "\x85", "\t", "\x00")


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


def _post(name: str):
    async def go():
        return await web_server.app.test_client().post(
            "/api/v1/dry-run/" + urllib.parse.quote(name, safe=""), json={}
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
            raise OSError(HOSTILE)
        except OSError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as exc:
        chained = exc
    _failing_dry_run(chained)
    text = formatted[0]
    for special in SPECIALS[:1] + SPECIALS[2:]:
        assert special not in text, repr(special)
    assert "forged" not in text.replace("\u001b", "") or "\nERROR - forged" not in text


def test_ordinary_exception_keeps_type_message_and_frames(formatted):
    _failing_dry_run(KeyError("sensor_x"))
    text = formatted[0]
    assert text.startswith("ERROR - emhass.web_server - Dry run of naive-mpc-optim failed: ")
    assert "Traceback (most recent call last):" in text
    assert "web_server.py" in text and "api_v1_dry_run" in text
    assert text.splitlines()[-1] == "KeyError: 'sensor_x'"
    assert len(formatted) == 1
