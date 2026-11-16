"""The API token EMHASS generates when none is configured (migration path for existing installs)."""

import os
import pathlib
import stat
import threading
import unittest
from unittest.mock import patch

import yaml

from emhass import api_auth

from .api_auth_support import ADMIN, OUTSIDER, SUPERVISOR, TOKEN, AuthSupportBase

POSIX = os.name == "posix"


class TestProvisioning(AuthSupportBase):
    async def configure(self, **kwargs):
        await api_auth.configure(None, None, self.logger, data_dir=self.data, **kwargs)

    def token_file(self) -> pathlib.Path:
        return self.data / api_auth.TOKEN_FILE_NAME

    async def test_a_token_is_generated_kept_and_enforced(self):
        with self.assertLogs(self.logger, level="WARNING") as logs:
            await self.configure()
        token = api_auth.current_token()
        self.assertGreaterEqual(len(token), 40)
        self.assertEqual(self.token_file().read_text(encoding="utf-8").strip(), token)
        self.assertEqual(api_auth._state["source"], "generated-new")
        # only the path is logged
        text = "\n".join(logs.output)
        self.assertIn(str(self.token_file()), text)
        self.assertNotIn(token, text)
        # and it is enforced
        refused = await self.send("POST", "/action/publish-data", json={})
        self.assertEqual(refused.status_code, 401)
        ok = await self.send(
            "POST", "/action/publish-data", json={}, headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(ok.status_code, 200)

    @unittest.skipUnless(POSIX, "file modes are POSIX")
    async def test_the_file_is_private_to_its_owner(self):
        await self.configure()
        self.assertEqual(stat.S_IMODE(self.token_file().stat().st_mode), 0o600)

    async def test_the_token_survives_a_restart(self):
        await self.configure()
        first = api_auth.current_token()
        api_auth.reset()
        with self.assertNoLogs(self.logger, level="WARNING"):
            await self.configure()
        self.assertEqual(api_auth.current_token(), first)
        self.assertEqual(api_auth._state["source"], "generated")

    async def test_two_workers_starting_together_agree_on_one_token(self):
        results = []

        def start():
            results.append(api_auth.provision_token(self.data, self.logger)[0])

        threads = [threading.Thread(target=start) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(self.token_file().read_text(encoding="utf-8").strip(), results[0])

    async def test_a_file_another_worker_is_still_writing_is_waited_for(self):
        self.token_file().write_text("")  # created, token not written yet
        written = "w" * 43

        def finish():
            import time

            time.sleep(0.2)
            self.token_file().write_text(written + "\n")

        thread = threading.Thread(target=finish)
        thread.start()
        token, how = api_auth.provision_token(self.data, self.logger)
        thread.join()
        self.assertEqual((token, how), (written, "file"))

    async def test_a_configured_token_is_used_and_nothing_is_generated(self):
        for name, setup in {
            "environment": lambda: os.environ.__setitem__(api_auth.ENV_NAME, TOKEN),
            "secrets": lambda: (self.dir / "s.yaml").write_text(
                yaml.safe_dump({api_auth.SETTING_NAME: TOKEN})
            ),
        }.items():
            with self.subTest(name):
                api_auth.reset()
                os.environ.pop(api_auth.ENV_NAME, None)
                setup()
                await api_auth.configure(self.dir / "s.yaml", None, self.logger, data_dir=self.data)
                self.assertEqual(api_auth.current_token(), TOKEN)
                self.assertEqual(api_auth._state["source"], "configured")
                self.assertFalse(self.token_file().exists())

    async def test_switching_generation_off_leaves_the_server_closed(self):
        os.environ[api_auth.AUTOGENERATE_ENV] = "false"
        await self.configure()
        self.assertIsNone(api_auth.current_token())
        self.assertFalse(self.token_file().exists())
        response = await self.send("GET", "/api/v1/last-run")
        self.assertEqual(response.status_code, 503)

    async def test_the_opt_out_generates_nothing(self):
        os.environ[api_auth.DISABLE_ENV] = "true"
        await self.configure()
        self.assertFalse(self.token_file().exists())

    async def test_no_data_folder_given_means_no_generation(self):
        await api_auth.configure(None, None, self.logger)
        self.assertIsNone(api_auth.current_token())

    async def test_bad_files_fail_closed_and_are_never_replaced(self):
        for name, write in {
            "empty": lambda: self.token_file().write_text(""),
            "short": lambda: self.token_file().write_text("tiny-secret\n"),
            "directory": lambda: self.token_file().mkdir(),
        }.items():
            with self.subTest(name):
                api_auth.reset()
                path = self.token_file()
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink(missing_ok=True)
                write()
                with self.assertLogs(self.logger, level="ERROR") as logs:
                    await self.configure()
                self.assertTrue(api_auth._state["failed"])
                self.assertNotIn("tiny-secret", "\n".join(logs.output))
                response = await self.send("GET", "/api/v1/last-run")
                self.assertEqual(response.status_code, 503)

    @unittest.skipUnless(POSIX or os.environ.get("EMHASS_TEST_SYMLINKS"), "needs symlink support")
    async def test_a_link_in_place_of_the_file_is_refused(self):
        outside = self.dir / "outside.txt"
        outside.write_text("x" * 40 + "\n")
        try:
            os.symlink(outside, self.token_file())
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"cannot create symlinks here: {exc}")
        with self.assertLogs(self.logger, level="ERROR"):
            await self.configure()
        self.assertTrue(api_auth._state["failed"])
        self.assertEqual(outside.read_text(), "x" * 40 + "\n")

    @unittest.skipUnless(
        POSIX and hasattr(os, "geteuid") and os.geteuid() != 0, "needs a non-root POSIX user"
    )
    async def test_an_unwritable_data_folder_fails_closed_without_open_access(self):
        self.data.chmod(0o500)
        try:
            with self.assertLogs(self.logger, level="ERROR") as logs:
                await self.configure()
        finally:
            self.data.chmod(0o700)
        self.assertTrue(api_auth._state["failed"])
        self.assertIn("cannot be saved", "\n".join(logs.output))
        self.assertEqual(
            (await self.send("POST", "/action/publish-data", json={})).status_code, 503
        )

    async def test_a_missing_data_folder_is_created(self):
        missing = self.dir / "new" / "data"
        await api_auth.configure(None, None, self.logger, data_dir=missing)
        self.assertTrue((missing / api_auth.TOKEN_FILE_NAME).is_file())

    async def test_the_generated_token_is_masked_in_every_log_record(self):
        await self.configure()
        token = api_auth.current_token()
        import logging

        record = logging.LogRecord("x", logging.INFO, __file__, 1, "body %s", ({"t": token},), None)
        self.assertTrue(api_auth.token_log_filter.filter(record))
        self.assertNotIn(token, record.getMessage())

    async def test_the_generated_token_never_reaches_the_runtime_parameters(self):
        await self.configure()
        token = api_auth.current_token()
        shown = str(api_auth.redacted({"emhass_api_token": token, "other": 1}))
        self.assertNotIn(token, shown)


class TestTokenPage(AuthSupportBase):
    """/api-token: a generated token is shown only to a Home Assistant user who came through ingress."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await api_auth.configure(None, None, self.logger, data_dir=self.data)
        self.token = api_auth.current_token()

    async def test_shown_through_ingress(self):
        response = await self.send("GET", "/api-token", port=8099, peer=SUPERVISOR, headers=ADMIN)
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await response.get_data(as_text=True)).strip(), self.token)
        self.assertIn("no-store", response.headers["Cache-Control"])
        self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(response.mimetype, "text/plain")

    async def test_never_through_the_api_port_even_with_the_token(self):
        for headers in (None, {"Authorization": f"Bearer {self.token}"}):
            response = await self.send("GET", "/api-token", headers=headers, peer=SUPERVISOR)
            self.assertIn(response.status_code, (401, 403))
            self.assertNotIn(self.token, await response.get_data(as_text=True))

    async def test_not_shown_for_a_configured_token(self):
        os.environ[api_auth.ENV_NAME] = TOKEN
        api_auth.reset()
        await api_auth.configure(None, None, self.logger, data_dir=self.data)
        response = await self.send("GET", "/api-token", port=8099, peer=SUPERVISOR, headers=ADMIN)
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(TOKEN, await response.get_data(as_text=True))

    async def test_a_missing_or_non_administrator_identity_is_refused(self):
        cases = {
            "no identity at all": {},
            "user id but no groups": {"X-Remote-User-Id": "u1"},
            "groups but no user id": {"X-Remote-User-Groups": "system-admin"},
            "a normal user": {"X-Remote-User-Id": "u1", "X-Remote-User-Groups": "system-users"},
            "a read-only user": {
                "X-Remote-User-Id": "u1",
                "X-Remote-User-Groups": "system-read-only",
            },
            "lookalike group": {
                "X-Remote-User-Id": "u1",
                "X-Remote-User-Groups": "system-admins,xsystem-admin",
            },
            "blank user id": {"X-Remote-User-Id": " ", "X-Remote-User-Groups": "system-admin"},
        }
        for name, headers in cases.items():
            with self.subTest(name):
                response = await self.send(
                    "GET", "/api-token", port=8099, peer=SUPERVISOR, headers=headers
                )
                self.assertEqual(response.status_code, 403)
                self.assertNotIn(self.token, await response.get_data(as_text=True))

    async def test_forged_administrator_headers_from_a_stranger_do_nothing(self):
        for port in (8099, 5000):
            response = await self.send("GET", "/api-token", port=port, peer=OUTSIDER, headers=ADMIN)
            self.assertIn(response.status_code, (401, 403))
            self.assertNotIn(self.token, await response.get_data(as_text=True))
        # and the Supervisor's address on the API port is still just a client
        response = await self.send("GET", "/api-token", port=5000, peer=SUPERVISOR, headers=ADMIN)
        self.assertIn(response.status_code, (401, 403))

    async def test_not_from_a_stranger_on_the_ingress_port(self):
        response = await self.send("GET", "/api-token", port=8099, peer=OUTSIDER)
        self.assertEqual(response.status_code, 403)
        self.assertNotIn(self.token, await response.get_data(as_text=True))


class TestCrashSafety(AuthSupportBase):
    def token_file(self) -> pathlib.Path:
        return self.data / api_auth.TOKEN_FILE_NAME

    def test_losing_the_race_to_create_the_file_uses_the_winners_token(self):
        winner = "r" * 43
        real_link = os.link

        def link(src, dst, *args, **kwargs):
            pathlib.Path(dst).write_text(winner + "\n")  # another worker published first
            return real_link(src, dst, *args, **kwargs)  # now fails: the file exists

        real_open = os.open

        def guarded_open(path, *args, **kwargs):
            # the exclusive-create fallback must not run when link() reported the file exists
            assert pathlib.Path(path) != self.token_file(), "fell back to creating the final file"
            return real_open(path, *args, **kwargs)

        with (
            patch.object(api_auth.os, "link", side_effect=link),
            patch.object(api_auth.os, "open", side_effect=guarded_open),
        ):
            token, how = api_auth.provision_token(self.data, self.logger)
        self.assertEqual((token, how), (winner, "file"))
        self.assertEqual(self.token_file().read_text().strip(), winner)

    def test_a_failure_while_writing_leaves_no_token_file_and_no_leftovers(self):
        with patch.object(api_auth.os, "fsync", side_effect=OSError("disk full")):
            with self.assertRaises(api_auth.ApiTokenError):
                api_auth.provision_token(self.data, self.logger)
        self.assertEqual(sorted(p.name for p in self.data.iterdir()), [])

    @unittest.skipUnless(POSIX, "file modes are POSIX")
    def test_a_file_system_without_hard_links_still_gets_a_private_complete_file(self):
        with patch.object(
            api_auth.os, "link", side_effect=OSError(__import__("errno").ENOTSUP, "no hard links")
        ):
            token, how = api_auth.provision_token(self.data, self.logger)
        self.assertEqual(how, "new")
        self.assertEqual(self.token_file().read_text().strip(), token)
        self.assertEqual(stat.S_IMODE(self.token_file().stat().st_mode), 0o600)
        self.assertEqual([p.name for p in self.data.iterdir()], [api_auth.TOKEN_FILE_NAME])

    def test_an_old_empty_file_left_by_a_crash_is_replaced(self):
        import time

        self.token_file().write_text("")
        past = time.time() - 3600
        os.utime(self.token_file(), (past, past))
        token, how = api_auth.provision_token(self.data, self.logger)
        self.assertEqual(how, "new")
        self.assertGreaterEqual(len(token), 40)
        self.assertEqual(self.token_file().read_text().strip(), token)

    def test_a_fresh_empty_file_is_not_replaced_under_another_worker(self):
        self.token_file().write_text("")
        with self.assertRaises(api_auth.ApiTokenError):
            api_auth.provision_token(self.data, self.logger)
        self.assertEqual(self.token_file().read_text(), "")

    def test_windows_is_told_the_file_is_not_private(self):
        with (
            patch.object(api_auth, "_IS_WINDOWS", True),
            self.assertLogs(self.logger, level="WARNING") as logs,
        ):
            api_auth.provision_token(self.data, self.logger)
        self.assertTrue(any("not restricted to its owner" in line for line in logs.output))
        with (
            patch.object(api_auth, "_IS_WINDOWS", False),
            self.assertLogs(self.logger, level="WARNING") as logs,
        ):
            self.token_file().unlink()
            api_auth.provision_token(self.data, self.logger)
        self.assertFalse(any("not restricted" in line for line in logs.output))

    def test_an_io_error_from_link_is_a_failure_not_a_reason_to_weaken(self):
        import errno

        with patch.object(api_auth.os, "link", side_effect=OSError(errno.EIO, "I/O error")):
            with self.assertRaises(api_auth.ApiTokenError):
                api_auth.provision_token(self.data, self.logger)
        self.assertEqual(sorted(p.name for p in self.data.iterdir()), [])

    def test_a_file_that_got_its_token_while_waiting_is_used_not_replaced(self):
        self.token_file().write_text("")
        got = "g" * 43
        with (
            patch.object(api_auth, "_read_generated_waiting", return_value=None),
            patch.object(api_auth, "_read_generated", return_value=got),
        ):
            self.assertEqual(api_auth.provision_token(self.data, self.logger), (got, "file"))

    def test_old_leftover_temporaries_are_removed_and_recent_ones_kept(self):
        import time

        old = self.data / f".{api_auth.TOKEN_FILE_NAME}.1.aaaa.tmp"
        recent = self.data / f".{api_auth.TOKEN_FILE_NAME}.2.bbbb.tmp"
        old.write_text("x")
        recent.write_text("y")
        past = time.time() - 7200
        os.utime(old, (past, past))
        api_auth.provision_token(self.data, self.logger)
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())

    def test_a_blank_file_is_treated_like_an_empty_one(self):
        import time

        self.token_file().write_text("  \n\n")
        with self.assertRaises(api_auth.ApiTokenError):
            api_auth.provision_token(
                self.data, self.logger
            )  # fresh: maybe another worker is writing
        past = time.time() - 3600
        os.utime(self.token_file(), (past, past))
        token, how = api_auth.provision_token(self.data, self.logger)
        self.assertEqual(how, "new")
        self.assertEqual(self.token_file().read_text().strip(), token)

    def test_a_file_that_is_not_a_token_is_not_overwritten(self):
        import time

        self.token_file().write_text("garbage\n")
        past = time.time() - 3600
        os.utime(self.token_file(), (past, past))
        with self.assertRaises(api_auth.ApiTokenError):
            api_auth.provision_token(self.data, self.logger)
        self.assertEqual(self.token_file().read_text(), "garbage\n")

    def test_a_file_replaced_between_the_two_looks_is_not_deleted(self):
        import time
        import types

        self.token_file().write_text("")
        past = time.time() - 3600
        os.utime(self.token_file(), (past, past))
        real = pathlib.Path.stat
        calls = []

        def stat(path, *args, **kwargs):
            result = real(path, *args, **kwargs)
            if path == self.token_file():
                calls.append(1)
                if len(calls) == 2:  # the second look finds a different file in its place
                    return types.SimpleNamespace(
                        st_ino=result.st_ino + 1, st_size=result.st_size, st_mtime=result.st_mtime
                    )
            return result

        with patch.object(pathlib.Path, "stat", stat):
            with self.assertRaises(api_auth.ApiTokenError):
                api_auth._discard_stale(self.token_file())
        self.assertTrue(self.token_file().exists())

    def test_non_blank_text_is_never_discarded(self):
        import time

        self.token_file().write_text("something else entirely\n")
        past = time.time() - 3600
        os.utime(self.token_file(), (past, past))
        with self.assertRaises(api_auth.ApiTokenError):
            api_auth._discard_stale(self.token_file())
        self.assertEqual(self.token_file().read_text(), "something else entirely\n")

    def test_a_file_that_vanishes_while_being_looked_at_is_not_an_error(self):
        import time

        self.token_file().write_text("")
        past = time.time() - 3600
        os.utime(self.token_file(), (past, past))
        real = pathlib.Path.read_text

        def vanish(path, *args, **kwargs):
            if path == self.token_file():
                path.unlink()
            return real(path, *args, **kwargs)

        with patch.object(pathlib.Path, "read_text", vanish):
            api_auth._discard_stale(self.token_file())  # does not raise
        self.assertFalse(self.token_file().exists())

    def test_replacement_is_serialised_by_a_lock(self):
        import time

        self.token_file().write_text("")
        past = time.time() - 3600
        os.utime(self.token_file(), (past, past))
        lock = self.data / f".{api_auth.TOKEN_FILE_NAME}.lock"
        lock.write_text("")  # another process is replacing it right now
        with self.assertRaises(api_auth.ApiTokenError):
            api_auth.provision_token(self.data, self.logger)
        self.assertEqual(self.token_file().read_text(), "")  # untouched
        self.assertTrue(lock.exists())  # not ours to remove

    def test_a_lock_left_by_a_killed_process_is_cleared(self):
        import time

        self.token_file().write_text("")
        past = time.time() - 3600
        os.utime(self.token_file(), (past, past))
        lock = self.data / f".{api_auth.TOKEN_FILE_NAME}.lock"
        lock.write_text("")
        os.utime(lock, (past, past))
        token, how = api_auth.provision_token(self.data, self.logger)
        self.assertEqual(how, "new")
        self.assertFalse(lock.exists())  # released afterwards

    def test_two_provisioners_finding_a_stale_file_together_converge(self):
        import time

        self.token_file().write_text("")
        past = time.time() - 3600
        os.utime(self.token_file(), (past, past))
        results = []
        errors = []

        def start():
            try:
                results.append(api_auth.provision_token(self.data, self.logger)[0])
            except api_auth.ApiTokenError as e:  # losing the lock may fail closed: never diverge
                errors.append(str(e))

        threads = [threading.Thread(target=start) for _ in range(6)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertTrue(results)
        self.assertEqual(len(set(results)), 1, "workers ended with different tokens")
        self.assertEqual(self.token_file().read_text().strip(), results[0])


class TestTokenPageIsNotReadableByScript(AuthSupportBase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await api_auth.configure(None, None, self.logger, data_dir=self.data)

    async def get(self, **headers):
        return await self.send(
            "GET", "/api-token", port=8099, peer=SUPERVISOR, headers={**ADMIN, **headers}
        )

    async def test_the_page_cannot_be_read_by_a_framing_script(self):
        response = await self.get()
        self.assertEqual(response.status_code, 200)
        # an opaque origin: the parent page's script cannot read a sandboxed document
        self.assertEqual(response.headers["Content-Security-Policy"], "sandbox")
        self.assertNotIn("allow-same-origin", response.headers["Content-Security-Policy"])

    async def test_a_page_load_is_served_and_a_script_read_is_not(self):
        token = api_auth.current_token()
        for headers, expected in (
            ({}, 200),  # curl, or a browser that sends no fetch metadata
            ({"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "same-origin"}, 200),
            ({"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "none"}, 200),
            (
                {"Sec-Fetch-Mode": "cors", "Sec-Fetch-Site": "same-origin"},
                403,
            ),  # fetch() from another panel
            ({"Sec-Fetch-Mode": "same-origin"}, 403),
            ({"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "cross-site"}, 403),
        ):
            with self.subTest(headers=headers):
                response = await self.get(**headers)
                self.assertEqual(response.status_code, expected)
                if expected == 403:
                    self.assertNotIn(token, await response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()


class TestTokenLink(AuthSupportBase):
    """The web UI shows the link to /api-token only where the page itself is allowed."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await api_auth.configure(None, None, self.logger, data_dir=self.data)

    async def test_link_only_through_ingress_for_a_generated_token(self):
        shown = await self.send("GET", "/", port=8099, peer=SUPERVISOR, headers=ADMIN)
        self.assertIn('href="api-token"', await shown.get_data(as_text=True))
        # no link for a request with no administrator identity
        anonymous = await self.send("GET", "/", port=8099, peer=SUPERVISOR)
        self.assertNotIn('href="api-token"', await anonymous.get_data(as_text=True))
        # never on the API port, even with the token
        token = api_auth.current_token()
        other = await self.send("GET", "/", headers={"Authorization": f"Bearer {token}"})
        self.assertNotIn('href="api-token"', await other.get_data(as_text=True))

    async def test_no_link_for_a_configured_token(self):
        os.environ[api_auth.ENV_NAME] = TOKEN
        api_auth.reset()
        await api_auth.configure(None, None, self.logger, data_dir=self.data)
        page = await self.send("GET", "/", port=8099, peer=SUPERVISOR, headers=ADMIN)
        self.assertNotIn('href="api-token"', await page.get_data(as_text=True))
