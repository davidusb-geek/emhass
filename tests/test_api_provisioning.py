"""The API token EMHASS generates when none is configured (migration path for existing installs)."""

import os
import pathlib
import stat
import threading
import unittest

import yaml

from emhass import api_auth

from .api_auth_support import OUTSIDER, SUPERVISOR, TOKEN, AuthSupportBase

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
        response = await self.send("GET", "/api-token", port=8099, peer=SUPERVISOR)
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
        response = await self.send("GET", "/api-token", port=8099, peer=SUPERVISOR)
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(TOKEN, await response.get_data(as_text=True))

    async def test_not_from_a_stranger_on_the_ingress_port(self):
        response = await self.send("GET", "/api-token", port=8099, peer=OUTSIDER)
        self.assertEqual(response.status_code, 403)
        self.assertNotIn(self.token, await response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()


class TestTokenLink(AuthSupportBase):
    """The web UI shows the link to /api-token only where the page itself is allowed."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await api_auth.configure(None, None, self.logger, data_dir=self.data)

    async def test_link_only_through_ingress_for_a_generated_token(self):
        shown = await self.send("GET", "/", port=8099, peer=SUPERVISOR)
        self.assertIn('href="api-token"', await shown.get_data(as_text=True))
        # never on the API port, even with the token
        token = api_auth.current_token()
        other = await self.send("GET", "/", headers={"Authorization": f"Bearer {token}"})
        self.assertNotIn('href="api-token"', await other.get_data(as_text=True))

    async def test_no_link_for_a_configured_token(self):
        os.environ[api_auth.ENV_NAME] = TOKEN
        api_auth.reset()
        await api_auth.configure(None, None, self.logger, data_dir=self.data)
        page = await self.send("GET", "/", port=8099, peer=SUPERVISOR)
        self.assertNotIn('href="api-token"', await page.get_data(as_text=True))
