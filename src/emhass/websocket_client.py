import asyncio
import logging
import ssl
import time
import urllib.parse as urlparse
from typing import Any

import orjson
import websockets

logger = logging.getLogger(__name__)

class WebSocketError(Exception): pass
class AuthenticationError(WebSocketError): pass
class ConnectionError(WebSocketError): pass
class RequestError(WebSocketError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

class AsyncWebSocketClient:
    def __init__(
        self,
        hass_url: str,
        long_lived_token: str,
        logger: logging.Logger | None = None,
        ping_interval: int = 30,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 5,
    ):
        self.hass_url = hass_url
        self.token = long_lived_token.strip()
        self.logger = logger or logging.getLogger("AsyncWebSocketClient")
        self.ping_interval = ping_interval
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._ws = None
        self._connected = False
        self._authenticated = False
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._ping_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._reconnects = 0

    @property
    def websocket_url(self) -> str:
        # For supervisor API, use the dedicated websocket endpoint
        if self.hass_url.startswith("http://supervisor/core/api"):
            return "ws://supervisor/core/websocket"
        elif self.hass_url.startswith("https://supervisor/core/api"):
            return "wss://supervisor/core/websocket"
        else:
            # Standard Home Assistant instance
            parsed = urlparse.urlparse(self.hass_url)
            ws_scheme = "wss" if parsed.scheme == "https" else "ws"
            return f"{ws_scheme}://{parsed.netloc}/api/websocket"

    @property
    def connected(self) -> bool:
        return self._connected and self._authenticated

    def _get_ssl_context(self) -> ssl.SSLContext | None:
        if self.websocket_url.startswith("wss://"):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return None

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def startup(self):
        """Connect and authenticate."""
        await self._connect()

    async def shutdown(self):
        """Cleanly close connection."""
        async with self._lock:
            await self._cleanup()

    async def reconnect(self):
        """Force a reconnect."""
        async with self._lock:
            await self._cleanup()
            await asyncio.sleep(1)
            try:
                await asyncio.wait_for(self._connect(), timeout=10.0)
            except TimeoutError as e:
                self.logger.error("Reconnection timed out")
                raise ConnectionError("Reconnection timed out") from e
            except Exception as e:
                self.logger.error(f"Reconnection failed: {e}")
                raise

    async def _connect(self):
        """Internal connect/authenticate and start background tasks."""
        ssl_ctx = self._get_ssl_context()
        self._ws = await websockets.connect(self.websocket_url, ssl=ssl_ctx, ping_interval=None, max_size=None)
        # Authenticate
        msg = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
        data = orjson.loads(msg)
        if data.get("type") != "auth_required":
            raise AuthenticationError("No auth_required")
        auth = orjson.dumps({"type":"auth", "access_token":self.token}).decode()
        await self._ws.send(auth)
        resp = orjson.loads(await asyncio.wait_for(self._ws.recv(), timeout=5.0))
        if resp.get("type") != "auth_ok":
            raise AuthenticationError("Invalid auth")
        self._authenticated = True
        self._connected = True
        self._recv_task = asyncio.create_task(self._receiver())
        self._ping_task = asyncio.create_task(self._ping_loop())
        self._reconnects = 0
        self.logger.info("🔌 Connected and authenticated")

    async def _cleanup(self):
        """Stop tasks and close ws."""
        self._connected = False
        self._authenticated = False
        if self._ping_task:
            self._ping_task.cancel()
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Connection closed"))
        self._pending.clear()

    async def _ping_loop(self):
        """Keep alive ping loop."""
        try:
            while self.connected:
                await asyncio.sleep(self.ping_interval)
                await self._ws.send(orjson.dumps({"id": self._next_id(), "type": "ping"}).decode())
        except Exception:
            pass

    async def _receiver(self):
        """Receive messages and route to pending futures."""
        try:
            while self.connected:
                msg = await self._ws.recv()
                data = orjson.loads(msg)
                # route by id
                mid = data.get("id")
                if mid and mid in self._pending:
                    fut = self._pending.pop(mid)
                    if data.get("type") == "result" and not data.get("success", True):
                        err = data.get("error", {})
                        fut.set_exception(RequestError(err.get("code",""), err.get("message","")))
                    else:
                        fut.set_result(data.get("result", None))
        except Exception as e:
            self.logger.error(f"Receiver error: {e}")
            self._connected = False
            # Cancel pending futures
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("Connection lost"))
            self._pending.clear()

    async def send(self, msg_type: str, **kwargs) -> Any:
        """Send a command and await response."""
        async with self._lock:
            if not self.connected:
                await self.reconnect()
            mid = self._next_id()
            payload = {"id": mid, "type": msg_type, **kwargs}
            fut = asyncio.get_event_loop().create_future()
            self._pending[mid] = fut
            await self._ws.send(orjson.dumps(payload).decode())
        return await fut

    # Convenience API methods:
    async def get_config(self) -> dict[str, Any]:
        return await self.send("get_config")

    async def get_states(self) -> list[dict[str, Any]]:
        return await self.send("get_states")

    async def get_state(self, entity_id: str) -> dict[str, Any] | None:
        states = await self.get_states()
        return next((s for s in states if s["entity_id"] == entity_id), None)

    async def call_service(self, domain: str, service: str, service_data: dict = None, target: dict = None):
        return await self.send("call_service", domain=domain, service=service, service_data=service_data or {}, target=target or {})

    async def get_history(self, **kwargs):
        return await self.send("history/history_during_period", **kwargs)

    async def get_statistics(self, **kwargs):
        return await self.send("recorder/statistics_during_period", **kwargs)

    async def ping_time(self) -> float:
        start = time.perf_counter()
        await self.send("ping")
        return (time.perf_counter() - start) * 1000
