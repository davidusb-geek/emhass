"""/set-config writes config.json and params.pkl whole, and one save at a time.

It used to write config.json, then open params.pkl for writing (emptying it) before
building the params that go in it. A failure in that build left a new config.json next
to an empty params.pkl, so every action failed until the next good save, and an action
running during any save could read the empty file. These tests go through the real app
on scratch paths.
"""

import asyncio
import errno
import logging
import os
import pathlib
import pickle
import stat
from unittest.mock import patch

import orjson
import pytest

from emhass import utils

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
logger = logging.getLogger("config_save_test")

SEED = '{"battery_minimum_state_of_charge": 0.3}'
CHANGE = '{"battery_minimum_state_of_charge": 0.25}'


@pytest.fixture
def web(tmp_path, monkeypatch):
    """The real web_server app on scratch paths, seeded with one good save."""
    from emhass import web_server

    conf = {
        "data_path": tmp_path,
        "config_path": tmp_path / "config.json",
        "defaults_path": TEST_ROOT / "src/emhass/data/config_defaults.json",
        "associations_path": TEST_ROOT / "src/emhass/data/associations.csv",
        "legacy_config_path": tmp_path / "legacy_missing.yaml",
        "root_path": TEST_ROOT / "src/emhass/",
    }
    _, secrets = asyncio.run(utils.build_secrets(conf, logger, no_response=True))
    monkeypatch.setattr(web_server, "emhass_conf", conf)
    monkeypatch.setattr(web_server, "params_secrets", secrets)
    client = web_server.app.test_client()

    def post(body: str):
        async def go():
            r = await client.post(
                "/set-config", data=body.encode(), headers={"Content-Type": "application/json"}
            )
            return r.status_code, await r.get_data(as_text=True)

        return asyncio.run(go())

    def files():
        return conf["config_path"].read_bytes(), (tmp_path / "params.pkl").read_bytes()

    assert post(SEED)[0] == 200
    return type(
        "Web",
        (),
        {"conf": conf, "post": staticmethod(post), "files": staticmethod(files), "ws": web_server},
    )


def _failing_on_call(n, real, failure):
    """A build_params stand-in that runs the real one except on call ``n``."""
    calls = {"n": 0}

    async def build(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            if isinstance(failure, Exception):
                raise failure
            return failure
        return await real(*args, **kwargs)

    return build


def test_a_save_writes_both_files_and_they_agree(web):
    assert web.post(CHANGE)[0] == 200
    saved = orjson.loads(web.conf["config_path"].read_bytes())
    _, params = pickle.loads((web.conf["data_path"] / "params.pkl").read_bytes())
    assert saved["battery_minimum_state_of_charge"] == 0.25
    assert params["plant_conf"]["battery_minimum_state_of_charge"] == 0.25


@pytest.mark.parametrize(
    "failure", [RuntimeError("build failed"), False], ids=["raises", "returns-false"]
)
def test_a_failed_params_build_leaves_both_files_as_they_were(web, failure):
    before = web.files()
    # The second build_params call in a save is the one for params.pkl.
    with patch.object(web.ws, "build_params", _failing_on_call(2, web.ws.build_params, failure)):
        try:
            status, _ = web.post(CHANGE)
        except RuntimeError:
            status = 500
    assert status == 500
    assert web.files() == before


def test_params_pkl_keeps_its_old_content_while_a_save_builds_the_new_one(web):
    pkl = web.conf["data_path"] / "params.pkl"
    before = pkl.read_bytes()
    seen = []
    real = web.ws.build_params

    async def build(*args, **kwargs):
        seen.append(pkl.read_bytes())
        return await real(*args, **kwargs)

    with patch.object(web.ws, "build_params", build):
        assert web.post(CHANGE)[0] == 200
    assert seen == [before, before]
    assert pkl.read_bytes() != before


def test_a_config_that_cannot_be_written_leaves_both_files_as_they_were(web):
    before = web.files()
    real = getattr(web.ws, "replace_file", utils.replace_file)
    written = []

    async def replace(path, content):
        written.append(path)
        await real(path, content)

    errors = []

    class Errors(logging.Handler):
        def emit(self, record):
            errors.append(f"{record.getMessage()} {record.exc_info and record.exc_info[1]}")

    # Listen on the app logger itself: other tests turn its level up and its
    # propagation off.
    app_logger, handler = web.ws.app.logger, Errors(logging.ERROR)
    level = app_logger.level
    app_logger.setLevel(logging.ERROR)
    app_logger.addHandler(handler)
    # orjson cannot write an integer outside 64 bits. This used to empty config.json.
    try:
        with patch.object(web.ws, "replace_file", replace, create=True):
            try:
                status, text = web.post('{"battery_nominal_energy_capacity": 18446744073709551616}')
            except TypeError as e:  # the app may re-raise instead of answering 500
                status, text = 500, ""
                errors.append(str(e))
    finally:
        app_logger.removeHandler(handler)
        app_logger.setLevel(level)
    # A 500 from orjson.dumps here; a request-level 400 for the same body is a separate change.
    if status == 500:
        assert any("64-bit" in e for e in errors), errors
    else:
        assert status == 400 and "64 bits" in text
    assert written == []
    assert web.files() == before


def test_a_save_waits_for_another_one_and_gives_409_when_it_does_not_finish(web, monkeypatch):
    monkeypatch.setattr(web.ws, "CONFIG_SAVE_LOCK_WAIT_S", 0.2, raising=False)
    slot = getattr(web.ws, "_config_save_slot", lambda: web.conf["data_path"] / "no-lock.lock")
    before = web.files()
    with open(slot(), "a+") as other_save:
        assert web.ws.try_lock_file(other_save)
        try:
            status, text = web.post(CHANGE)
        finally:
            web.ws.unlock_file(other_save)
    assert status == 409
    assert "Another configuration save is running" in text
    assert web.files() == before
    assert web.post(CHANGE)[0] == 200


def test_a_config_json_mounted_on_its_own_is_written_in_place(web):
    # docker run -v ./config.json:/share/config.json makes config.json a mount point,
    # and renaming over a mount point fails with EBUSY.
    real = getattr(web.ws, "replace_file", utils.replace_file)

    async def replace(path, content):
        if str(path) == str(web.conf["config_path"]):
            raise OSError(errno.EBUSY, "Device or resource busy")
        await real(path, content)

    with patch.object(web.ws, "replace_file", replace, create=True):
        assert web.post(CHANGE)[0] == 200
    saved = orjson.loads(web.conf["config_path"].read_bytes())
    assert saved["battery_minimum_state_of_charge"] == 0.25


def test_any_other_write_failure_is_not_retried_in_place(web):
    before = web.files()
    attempted = []

    async def replace(path, content):
        attempted.append(path)
        raise OSError(errno.ENOSPC, "No space left on device")

    with patch.object(web.ws, "replace_file", replace, create=True):
        try:
            status, _ = web.post(CHANGE)
        except OSError:
            status = 500
    assert status == 500
    assert attempted == [web.conf["config_path"]]  # failed on the first file, not retried
    assert web.files() == before


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to replace an open file")
def test_replace_file_waits_for_a_reader_to_close_the_file_on_windows(tmp_path):
    target = tmp_path / "params.pkl"
    target.write_bytes(b"old")

    async def go():
        with open(target, "rb"):
            task = asyncio.ensure_future(utils.replace_file(target, b"new"))
            await asyncio.sleep(0.2)
            assert not task.done()
        await task

    asyncio.run(go())
    assert target.read_bytes() == b"new"


@pytest.mark.skipif(os.name == "nt", reason="the retry is for Windows only")
def test_replace_file_does_not_retry_a_permission_error_elsewhere(tmp_path):
    calls = []

    def refuse(src, dst):
        calls.append(dst)
        raise PermissionError(errno.EACCES, "Permission denied")

    with patch.object(utils.os, "replace", refuse), pytest.raises(PermissionError):
        asyncio.run(utils.replace_file(tmp_path / "params.pkl", b"new"))
    assert len(calls) == 1


def _can_symlink(tmp_path) -> bool:
    try:
        (tmp_path / "probe-link").symlink_to(tmp_path / "probe-target")
    except OSError:  # Windows without the symlink privilege
        return False
    return True


def test_replace_file_writes_through_a_symlink(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("this account cannot create symlinks")
    target = tmp_path / "real-config.json"
    target.write_text("old")
    link = tmp_path / "config.json"
    link.symlink_to(target)
    asyncio.run(utils.replace_file(link, "new"))
    assert link.is_symlink()
    assert target.read_text() == "new"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_replace_file_keeps_the_permissions_of_the_file_it_replaces(tmp_path):
    target = tmp_path / "config.json"
    target.write_text("old")
    target.chmod(0o640)
    asyncio.run(utils.replace_file(target, "new"))
    assert target.read_text() == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="owner and mode are not copied on Windows")
@pytest.mark.parametrize("call", ["chown", "chmod"])
@pytest.mark.parametrize(
    "error",
    [
        PermissionError(errno.EPERM, "Operation not permitted"),  # not root
        OSError(errno.EINVAL, "Invalid argument"),  # an owner a user namespace cannot map
        OSError(errno.ENOTSUP, "Operation not supported"),  # some FUSE mounts
    ],
    ids=["EPERM", "EINVAL", "ENOTSUP"],
)
def test_replace_file_still_replaces_when_owner_or_mode_cannot_be_copied(tmp_path, call, error):
    target = tmp_path / "config.json"
    target.write_text("old")

    def refuse(*args):
        raise error

    with patch.object(utils.os, call, refuse):
        asyncio.run(utils.replace_file(target, "new"))
    assert target.read_text() == "new"
