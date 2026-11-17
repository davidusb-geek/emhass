"""Helpers shared by the access-control test modules."""

import logging
import os
import pathlib
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from emhass import api_auth, web_server

TOKEN = "correct-horse-battery-staple-0123"
BEARER = {"Authorization": f"Bearer {TOKEN}"}
SUPERVISOR = "172.30.32.2"
# Documentation address ranges, never real hosts
OUTSIDER = "192.0.2.10"
# what the Supervisor adds for a Home Assistant administrator (supervisor/api/ingress.py _init_header)
ADMIN = {"X-Remote-User-Id": "ha-user-1", "X-Remote-User-Groups": "system-admin,system-users"}


class AuthSupportBase(unittest.IsolatedAsyncioTestCase):
    """A data folder, a clean environment and a test client whose connection can be chosen."""

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.data = self.dir / "data"
        self.data.mkdir()
        self.original_conf = web_server.emhass_conf
        web_server.emhass_conf = {"data_path": self.data}
        api_auth.reset()
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for name in (
            api_auth.ENV_NAME,
            api_auth.DISABLE_ENV,
            api_auth.AUTOGENERATE_ENV,
            api_auth.INGRESS_PORT_ENV,
            api_auth.INGRESS_PEERS_ENV,
            "SUPERVISOR_TOKEN",
        ):
            os.environ.pop(name, None)
        self.delay = patch.object(api_auth, "FAILED_LOGIN_DELAY_S", 0)
        self.delay.start()
        self.action = patch("emhass.web_server._action_call_unlocked", new_callable=AsyncMock)
        self.mock_action = self.action.start()
        self.mock_action.return_value = "ran"
        self.client = web_server.app.test_client()
        self.logger = logging.getLogger("test_api_auth_support")

    async def asyncTearDown(self):
        self.action.stop()
        self.delay.stop()
        self.env.stop()
        api_auth.reset()
        web_server.emhass_conf = self.original_conf
        self.tmp.cleanup()

    async def send(self, method, path, *, port=5000, peer=OUTSIDER, headers=None, **kwargs):
        """One request over a connection accepted on ``port`` from ``peer`` (the ASGI scope's
        ``server`` and ``client``, which the real server fills from the socket)."""
        return await self.client.open(
            path,
            method=method,
            headers=headers,
            scope_base={"server": ("0.0.0.0", port), "client": (peer, 40000)},
            **kwargs,
        )

    async def enable(self, token: str = TOKEN):
        os.environ[api_auth.ENV_NAME] = token
        await api_auth.configure(None, None, self.logger)
        self.assertEqual(api_auth.current_token(), token)
