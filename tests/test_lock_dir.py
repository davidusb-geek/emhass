"""Where EMHASS keeps its lock and counter files.

On Linux and the BSDs the system temp folder is shared by every account on the
host. A file there with one of EMHASS's fixed names, left by another account
(or by an older install run as another user), must not stop optimizations,
configuration saves or Solcast calls.
"""

import logging
import os
import pathlib
import types
from unittest import mock

import pytest

from emhass import utils, web_server
from emhass.forecast import Forecast

posix_only = pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX accounts only")


@pytest.fixture
def shared_tmp(tmp_path):
    """A stand-in for the shared system temp folder, and a data folder."""
    tmp = tmp_path / "tmp"
    data = tmp_path / "data"
    tmp.mkdir()
    data.mkdir()
    with (
        mock.patch("tempfile.gettempdir", return_value=str(tmp)),
        mock.patch.dict(web_server.emhass_conf, {"data_path": data}),
        mock.patch.object(utils, "_private_temp_dir_warned", False),
    ):
        yield tmp, data


def _foreign_file(path: pathlib.Path) -> None:
    """Something at `path` that this account cannot open for writing. A folder
    stands in for another account's file: open(path, "a+") fails either way,
    and it does so for root too."""
    path.mkdir()


def _slots():
    return {
        "dry run": web_server._dry_run_slot,
        "config save": web_server._config_save_slot,
        "live optimization": web_server._live_optim_slot,
    }


@posix_only
@pytest.mark.parametrize("name", list(_slots()))
def test_a_foreign_file_in_the_shared_temp_folder_does_not_block_the_lock(shared_tmp, name):
    tmp, _ = shared_tmp
    slot = _slots()[name]
    _foreign_file(tmp / slot().name)  # the name it always had, in the shared folder
    try:
        f = open(slot(), "a+")
    except OSError as exc:
        pytest.fail(f"the {name} lock cannot be opened: {exc}")
    with f:
        assert utils.try_lock_file(f)
        utils.unlock_file(f)


@posix_only
def test_a_foreign_file_does_not_block_participant_commits(shared_tmp):
    tmp, _ = shared_tmp
    _foreign_file(tmp / web_server._live_optim_slot().name)
    taken, release = web_server._take_live_optim_slot()
    assert taken == "ok"
    release()


@posix_only
def test_a_foreign_solcast_counter_does_not_refuse_the_call(shared_tmp):
    tmp, data = shared_tmp
    stub = types.SimpleNamespace(
        emhass_conf={"data_path": data}, logger=logging.getLogger("test_lock_dir")
    )
    assert Forecast._solcast_rate_limit_ok(stub)  # creates today's counter
    counters = list(tmp.rglob("emhass_solcast_calls_*.count"))
    assert len(counters) == 1
    (tmp / counters[0].name).unlink(missing_ok=True)
    _foreign_file(tmp / counters[0].name)
    assert Forecast._solcast_rate_limit_ok(stub)


@posix_only
def test_the_files_go_in_a_folder_only_this_account_can_use(shared_tmp):
    tmp, data = shared_tmp
    folder = utils.private_temp_dir(data)
    assert folder == tmp / f"emhass-{os.geteuid()}"
    st = os.lstat(folder)
    assert st.st_uid == os.geteuid()
    assert st.st_mode & 0o777 == 0o700
    for slot in _slots().values():
        assert slot().parent == folder
    assert utils.private_temp_dir(data) == folder  # found again, not re-made


@posix_only
def test_the_folder_is_0700_whatever_the_umask(shared_tmp):
    _, data = shared_tmp
    old = os.umask(0o277)
    try:
        folder = utils.private_temp_dir(data)
    finally:
        os.umask(old)
    assert folder != data
    assert os.lstat(folder).st_mode & 0o777 == 0o700


@posix_only
def test_a_refused_chmod_gives_the_same_answer_on_every_call(shared_tmp):
    """Where chmod is refused (a temp folder on FAT, say), the first call and the
    later ones agree on one folder, so workers lock the same file."""
    _, data = shared_tmp
    old = os.umask(0o277)
    try:
        with mock.patch("os.chmod", side_effect=PermissionError("refused")):
            first = utils.private_temp_dir(data)
            second = utils.private_temp_dir(data)
    finally:
        os.umask(old)
    assert first == second == data


@posix_only
def test_a_refused_chmod_keeps_a_folder_made_0700(shared_tmp):
    """With a usual umask mkdir already gives 0700, so a refused chmod changes
    nothing: every call uses the temp folder."""
    tmp, data = shared_tmp
    old = os.umask(0o022)
    try:
        with mock.patch("os.chmod", side_effect=PermissionError("refused")):
            first = utils.private_temp_dir(data)
            second = utils.private_temp_dir(data)
    finally:
        os.umask(old)
    assert first == second == tmp / f"emhass-{os.geteuid()}"
    assert os.lstat(first).st_mode & 0o777 == 0o700


@posix_only
@pytest.mark.parametrize("mode", [0o500, 0o600, 0o300, 0o000], ids=oct)
def test_an_own_folder_with_owner_bits_off_is_set_to_0700(shared_tmp, mode, caplog):
    """No group or other bits, so no other account can have used it: it is set to
    0700 and used, not given up for the data folder."""
    tmp, data = shared_tmp
    folder = tmp / f"emhass-{os.geteuid()}"
    folder.mkdir(mode=0o700)
    folder.chmod(mode)
    with caplog.at_level(logging.WARNING, logger=utils.__name__):
        assert utils.private_temp_dir(data) == folder
    assert os.lstat(folder).st_mode & 0o777 == 0o700
    assert not caplog.records


@posix_only
def test_a_worker_in_the_mkdir_to_chmod_gap_picks_the_same_folder(shared_tmp):
    """Under a umask that takes owner bits off, mkdir gives 0500 until chmod runs.
    A second worker that looks in that gap must still pick the folder the first
    one is about to use, or the two would lock different files."""
    _, data = shared_tmp
    real_chmod = os.chmod
    seen = []

    def chmod_after_another_call(path, mode, *args, **kwargs):
        if not seen:
            seen.append(None)
            seen.append(utils.private_temp_dir(data))  # the other worker, in the gap
        return real_chmod(path, mode, *args, **kwargs)

    old = os.umask(0o277)
    try:
        with mock.patch("os.chmod", side_effect=chmod_after_another_call):
            first = utils.private_temp_dir(data)
    finally:
        os.umask(old)
    assert seen, "the gap was reached"
    assert first == seen[1] != data
    assert os.lstat(first).st_mode & 0o777 == 0o700


@posix_only
@pytest.mark.parametrize(
    "kind",
    [
        "open to others",
        "group readable",
        "a file",
        "a symlink",
        "another account's",
    ],
)
def test_an_unusable_folder_falls_back_to_the_data_folder(shared_tmp, kind, caplog):
    tmp, data = shared_tmp
    uid = os.geteuid()
    folder = tmp / f"emhass-{uid}"
    if kind == "open to others":
        folder.mkdir(mode=0o700)
        folder.chmod(0o777)
    elif kind == "group readable":
        folder.mkdir(mode=0o700)
        folder.chmod(0o750)
    elif kind == "a file":
        folder.write_text("")
        folder.chmod(0o700)  # only the folder check can reject it
    elif kind == "a symlink":
        target = tmp / "elsewhere"
        target.mkdir(mode=0o700)
        folder.symlink_to(target)
    else:  # made by this account, seen as if this process ran as the next uid
        uid += 1
        folder = tmp / f"emhass-{uid}"
        folder.mkdir(mode=0o700)
    with (
        mock.patch("os.geteuid", return_value=uid),
        caplog.at_level(logging.WARNING, logger=utils.__name__),
    ):
        assert utils.private_temp_dir(data) == data
        assert utils.private_temp_dir(data) == data
        slot = web_server._live_optim_slot()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "warned once, not on every request"
    assert slot.parent == data
    with open(slot, "a+") as f:
        assert utils.try_lock_file(f)
        utils.unlock_file(f)


@posix_only
def test_a_folder_that_cannot_be_made_fails_as_before(shared_tmp, caplog):
    """A temp folder that is missing (or read-only, which root would bypass in a
    test): no fallback, and the lock cannot be opened, so the live optimization
    answers 503 as before."""
    tmp, data = shared_tmp
    missing = tmp / "missing"
    with (
        mock.patch("tempfile.gettempdir", return_value=str(missing)),
        caplog.at_level(logging.WARNING, logger=utils.__name__),
    ):
        folder = utils.private_temp_dir(data)
    assert folder == missing / f"emhass-{os.geteuid()}"
    assert not caplog.records
    with pytest.raises(OSError):
        open(folder / "x.lock", "a+")


@pytest.mark.skipif(hasattr(os, "geteuid"), reason="Windows only")
def test_windows_keeps_its_per_account_temp_folder(shared_tmp):
    tmp, data = shared_tmp
    assert utils.private_temp_dir(data) == tmp
    for slot in _slots().values():
        assert slot().parent == tmp


def test_two_data_folders_get_two_locks(shared_tmp, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    first = web_server._live_optim_slot()
    with mock.patch.dict(web_server.emhass_conf, {"data_path": other}):
        second = web_server._live_optim_slot()
    assert first != second
    assert first.parent == second.parent
