import asyncio
import ssl
import time
import urllib.parse as urlparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import orjson
import websockets
import logging

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
        logger: Optional[logging.Logger] = None,
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
        self._pending: Dict[int, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._ping_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._reconnects = 0

    @property
    def websocket_url(self) -> str:
        parsed = urlparse.urlparse(self.hass_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{ws_scheme}://{parsed.netloc}/api/websocket"

    @property
    def connected(self) -> bool:
        return self._connected and self._authenticated

    def _get_ssl_context(self) -> Optional[ssl.SSLContext]:
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
            except asyncio.TimeoutError:
                self.logger.error("Reconnection timed out")
                raise ConnectionError("Reconnection timed out")
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
    async def get_config(self) -> Dict[str, Any]:
        return await self.send("get_config")

    async def get_states(self) -> List[Dict[str, Any]]:
        return await self.send("get_states")

    async def get_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
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

# """
# Async WebSocket client for Home Assistant based on HomeAssistantAPI patterns.
# This implementation provides persistent connections with automatic ping/pong,
# graceful reconnection, and statistics support.
# """
# import asyncio
# import logging
# import ssl
# import time
# import urllib.parse as urlparse
# from datetime import datetime, timezone
# from typing import Any, Dict, List, Optional

# import orjson
# import websockets
# import websockets.exceptions

# logger = logging.getLogger(__name__)


# class WebSocketError(Exception):
#     """Base exception for WebSocket operations."""
#     pass


# class AuthenticationError(WebSocketError):
#     """Authentication failed."""
#     pass


# class ConnectionError(WebSocketError):
#     """Connection failed."""
#     pass


# class RequestError(WebSocketError):
#     """Request failed."""
#     def __init__(self, code: str, message: str):
#         print("RequestError - websocket.py.py")
#         self.code = code
#         self.message = message
#         super().__init__(f"{code}: {message}")


# class AsyncWebSocketClient:
#     """
#     Manages persistent WebSocket connection to Home Assistant with automatic
#     ping/pong, reconnection, and concurrent request handling.
#     """

#     def __init__(
#         self,
#         hass_url: str,
#         long_lived_token: str,
#         logger: Optional[logging.Logger] = None,
#         ping_interval: int = 30,
#         ping_timeout: int = 20,
#         reconnect_delay: int = 5,
#         max_reconnect_attempts: int = 5,
#         auto_cleanup: bool = True,
#     ):
#         print("AsyncWebSocketManager init - websocket.py.py")
#         self.hass_url = hass_url
#         self.long_lived_token = long_lived_token.strip()
#         self.logger = logger or logging.getLogger(__name__)
#         self.ping_interval = ping_interval
#         self.ping_timeout = ping_timeout
#         self.reconnect_delay = reconnect_delay
#         self.max_reconnect_attempts = max_reconnect_attempts

#         # Connection state
#         self.websocket = None
#         self.is_connected = False
#         self.is_authenticated = False
#         self.ha_version = "unknown"

#         self._client: Optional[AsyncWebSocketClient] = None
#         # self._manager: Optional[AsyncWebSocketManager] = None
#         self._is_temp_instance = auto_cleanup
#         self._startup_attempted = False

#         # Request management
#         self._id_counter = 0
#         self._pending_requests: Dict[int, asyncio.Future] = {}
#         self._connection_lock = asyncio.Lock()

#         # Background tasks
#         self._ping_task: Optional[asyncio.Task] = None
#         self._receiver_task: Optional[asyncio.Task] = None
#         self._reconnect_attempts = 0

#         # Parse WebSocket URL
#         self.websocket_url = self._get_websocket_url()


#     def _get_websocket_url(self) -> str:
#         """Convert HTTP URL to WebSocket URL."""
#         print("_get_websocket_url - websocket.py")
#         parsed = urlparse.urlparse(self.hass_url)
#         if parsed.scheme == "https":
#             ws_scheme = "wss"
#         elif parsed.scheme == "http":
#             ws_scheme = "ws"
#         else:
#             raise ValueError(f"Unsupported scheme: {parsed.scheme}")

#         # Construct WebSocket URL
#         ws_url = f"{ws_scheme}://{parsed.netloc}/api/websocket"
#         return ws_url

#     def _get_ssl_context(self) -> Optional[ssl.SSLContext]:
#         """Get SSL context for secure connections."""
#         print("_get_ssl_context - websocket.py")
#         if self.websocket_url.startswith("wss://"):
#             context = ssl.create_default_context()
#             context.check_hostname = False
#             context.verify_mode = ssl.CERT_NONE
#             return context
#         return None

#     def _get_next_id(self) -> int:
#         """Get next unique message ID."""
#         print("_get_next_id - websocket.py")
#         self._id_counter += 1
#         return self._id_counter

#     async def startup(self) -> None:
#         """Initialize WebSocket connection."""
#         print("AsyncWebSocketManager.startup - websocket.py")
#         print(self._client)
#         # print(self._manager)

#         print(self._startup_attempted)
#         if self._client and self._manager and self._manager.is_connected:
#             return  # Already started successfully

#         if self._startup_attempted and self._manager and not self._manager.is_connected:
#             # Previous startup failed, allow retry
#             self._startup_attempted = False

#         if self._startup_attempted:
#             return  # Startup already in progress or completed

#         self._startup_attempted = True

#         # Validate configuration before attempting connection
#         if not self.hass_url or not self.long_lived_token:
#             raise ConnectionError("Missing required configuration: hass_url or long_lived_token")

#         try:
#             self._manager = AsyncWebSocketManager(
#                 self.hass_url,
#                 self.long_lived_token,
#                 self.logger
#             )
#             await self._manager.connect()
#             self._client = AsyncWebSocketClient(self._manager)
#             print("self._client", self._client)
#         except Exception as e:
#             self._startup_attempted = False
#             raise e

#     async def shutdown(self) -> None:
#         print("AsyncWebSocketManager.shutdown - websocket.py")
#         """Close WebSocket connection."""
#         if self._manager:
#             await self._manager.disconnect()
#             self._manager = None
#             self._client = None
#         self._startup_attempted = False

#     async def is_connection_healthy(self) -> bool:
#         """Check if connection is healthy."""
#         print("AsyncWebSocketManager.is_connection_healthy - websocket.py")
#         if self._manager is None:
#             return False
#         return (
#             self._manager.is_connected and
#             self._manager.is_authenticated
#         )


#     async def connect(self) -> None:
#         """Establish WebSocket connection and authenticate."""
#         print("connect - websocket.py")
#         async with self._connection_lock:
#             if self.is_connected:
#                 return

#             try:
#                 ssl_context = self._get_ssl_context()

#                 self.logger.info(f"Connecting to {self.websocket_url}")
#                 self.websocket = await websockets.connect(
#                     self.websocket_url,
#                     ssl=ssl_context,
#                     ping_interval=None,  # We handle ping ourselves
#                     ping_timeout=None,
#                     close_timeout=15,
#                     max_size=2**24,  # 16MB
#                     max_queue=32
#                 )

#                 # Authenticate first
#                 await asyncio.sleep(1)
#                 await self._authenticate()

#                 # Start message receiver AFTER authentication
#                 self._receiver_task = asyncio.create_task(self._message_receiver())

#                 # Give receiver task time to start
#                 await asyncio.sleep(0.1)

#                 # Start ping task
#                 self._ping_task = asyncio.create_task(self._ping_loop())

#                 self.is_connected = True
#                 self._reconnect_attempts = 0
#                 self.logger.info(f"✅ Connected to Home Assistant {self.ha_version}")

#             except Exception as e:
#                 self.logger.error(f"Failed to connect: {e}")
#                 await self._cleanup()
#                 raise ConnectionError(f"Failed to connect: {e}")

#     async def disconnect(self) -> None:
#         """Disconnect from WebSocket."""
#         print("disconnect - websocket.py")
#         async with self._connection_lock:
#             await self._cleanup()
#             self.logger.info("Disconnected from Home Assistant")

#     async def _cleanup(self) -> None:
#         """Clean up connection and tasks."""
#         print("_cleanup - websocket.py")
#         self.is_connected = False
#         self.is_authenticated = False

#         # Cancel background tasks
#         if self._ping_task and not self._ping_task.done():
#             self._ping_task.cancel()
#             try:
#                 await self._ping_task
#             except asyncio.CancelledError:
#                 pass

#         if self._receiver_task and not self._receiver_task.done():
#             self._receiver_task.cancel()
#             try:
#                 await self._receiver_task
#             except asyncio.CancelledError:
#                 pass

#         # Close WebSocket
#         if self.websocket:
#             try:
#                 if hasattr(self.websocket, 'closed') and not self.websocket.closed:
#                     await self.websocket.close()
#                 elif not hasattr(self.websocket, 'closed'):
#                     await self.websocket.close()
#             except Exception:
#                 pass

#         # Fail pending requests
#         for future in self._pending_requests.values():
#             if not future.done():
#                 future.set_exception(ConnectionError("Connection closed"))
#         self._pending_requests.clear()

#     async def _authenticate(self) -> None:
#         """Authenticate with Home Assistant."""
#         print("_authenticate - websocket.py")
#         try:
#             # Wait for auth_required message
#             print("Waiting for auth_required message")
#             auth_required = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
#             auth_data = orjson.loads(auth_required)
#             print(f"Received auth message: {auth_data}")

#             if auth_data.get("type") != "auth_required":
#                 raise AuthenticationError(f"Expected auth_required, got {auth_data.get('type')}")

#             self.ha_version = auth_data.get("ha_version", "unknown")

#             # Send authentication
#             auth_message = orjson.dumps({
#                 "type": "auth",
#                 "access_token": self.long_lived_token
#             }).decode("utf-8")
#             self.logger.debug("Sending authentication")
#             await self.websocket.send(auth_message)

#             # Wait for auth response
#             auth_response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
#             auth_result = orjson.loads(auth_response)
#             self.logger.debug(f"Received auth response: {auth_result}")

#             if auth_result.get("type") == "auth_ok":
#                 self.is_authenticated = True
#                 self.ha_version = auth_result.get("ha_version", self.ha_version)
#                 self.logger.debug(f"Authentication successful with HA version {self.ha_version}")
#             elif auth_result.get("type") == "auth_invalid":
#                 raise AuthenticationError(auth_result.get("message", "Authentication failed"))
#             else:
#                 raise AuthenticationError(f"Unexpected auth response: {auth_result}")

#         except asyncio.TimeoutError:
#             raise AuthenticationError("Authentication timed out")
#         except Exception as e:
#             self.logger.error(f"Authentication failed: {e}")
#             raise AuthenticationError(f"Authentication failed: {e}")

#     async def _ping_loop(self) -> None:
#         """Background task to send periodic pings."""
#         print("_ping_loop - websocket.py")
#         while self.is_connected:
#             try:
#                 await asyncio.sleep(self.ping_interval)
#                 if self.websocket:
#                     # Check if websocket is still open
#                     try:
#                         if hasattr(self.websocket, 'closed') and self.websocket.closed:
#                             break
#                         ping_id = self._get_next_id()
#                         ping_message = orjson.dumps({"id": ping_id, "type": "ping"}).decode("utf-8")
#                         await self.websocket.send(ping_message)
#                     except Exception as ping_error:
#                         print(f"Ping send failed: {ping_error}")
#                         if self.is_connected:
#                             asyncio.create_task(self._reconnect())
#                         break
#             except Exception as e:
#                 self.logger.warning(f"Ping failed: {e}")
#                 if self.is_connected:
#                     asyncio.create_task(self._reconnect())
#                 break

#     async def _message_receiver(self) -> None:
#         """Background task to receive and route messages."""
#         print("_message_receiver - websocket.py")
#         self.logger.debug("Message receiver task started")
#         while self.is_connected:
#             try:
#                 if not self.websocket:
#                     self.logger.debug("WebSocket is None, stopping receiver")
#                     break

#                 self.logger.debug("Waiting for message...")
#                 message = await self.websocket.recv()
#                 self.logger.debug(f"Received raw message: {message[:100]}...")
#                 data = orjson.loads(message)
#                 self.logger.debug(f"Parsed message: {data.get('type')} with ID {data.get('id')}")
#                 await self._handle_message(data)

#             except websockets.exceptions.ConnectionClosed:
#                 self.logger.warning("WebSocket connection closed")
#                 if self.is_connected:
#                     asyncio.create_task(self._reconnect())
#                 break
#             except (OSError, ConnectionResetError) as e:
#                 self.logger.warning(f"Connection error in message receiver: {e}")
#                 if self.is_connected:
#                     asyncio.create_task(self._reconnect())
#                 break
#             except orjson.JSONDecodeError as e:
#                 self.logger.warning(f"Invalid JSON received: {e}")
#                 continue
#             except Exception as e:
#                 self.logger.error(f"Message receiver error: {e}")
#                 import traceback
#                 self.logger.error(f"Traceback: {traceback.format_exc()}")
#                 if self.is_connected:
#                     asyncio.create_task(self._reconnect())
#                 break
#         self.logger.debug("Message receiver task ended")

#     async def _handle_message(self, data: Dict[str, Any]) -> None:
#         """Route received message to appropriate handler."""
#         print("_handle_message - websocket.py")
#         message_id = data.get("id")
#         message_type = data.get("type")

#         self.logger.debug(f"Received message: {message_type} with ID {message_id}")

#         if message_type == "pong":
#             self.logger.debug(f"Received pong {message_id}")
#             return

#         if message_type == "result":
#             # Check for errors
#             if not data.get("success", True):
#                 error = data.get("error", {})
#                 error_code = error.get("code", "unknown")
#                 error_message = error.get("message", "Unknown error")
#                 exception = RequestError(error_code, error_message)
#                 self.logger.error(f"WebSocket error for message {message_id}: {error_code} - {error_message}")
#             else:
#                 exception = None

#             # Route to pending request
#             if message_id in self._pending_requests:
#                 future = self._pending_requests.pop(message_id)
#                 if not future.done():
#                     if exception:
#                         future.set_exception(exception)
#                     else:
#                         future.set_result(data.get("result"))
#                         self.logger.debug(f"Successfully handled result for message {message_id}")
#             else:
#                 self.logger.warning(f"Received result for unknown message ID: {message_id}")

#         elif message_type == "event":
#             # Handle events (for subscriptions)
#             if message_id in self._pending_requests:
#                 future = self._pending_requests[message_id]
#                 if not future.done():
#                     future.set_result(data.get("event"))
#             else:
#                 self.logger.debug(f"Received event for message ID: {message_id}")
#         else:
#             self.logger.warning(f"Received unknown message type: {message_type}")

#     async def _reconnect(self) -> None:
#         """Attempt to reconnect with exponential backoff."""
#         print("_reconnect - websocket.py")
#         async with self._connection_lock:
#             if self._reconnect_attempts >= self.max_reconnect_attempts:
#                 self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) exceeded")
#                 await self._cleanup()
#                 return

#             self._reconnect_attempts += 1
#             delay = self.reconnect_delay * (2 ** (self._reconnect_attempts - 1))

#             self.logger.warning(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
#             await asyncio.sleep(delay)

#             try:
#                 # Clean up old connection first
#                 if self.websocket:
#                     try:
#                         if hasattr(self.websocket, 'close'):
#                             await self.websocket.close()
#                     except Exception:
#                         pass
#                     self.websocket = None

#                 # Cancel old tasks
#                 if self._ping_task and not self._ping_task.done():
#                     self._ping_task.cancel()
#                 if self._receiver_task and not self._receiver_task.done():
#                     self._receiver_task.cancel()

#                 # Create new connection
#                 ssl_context = self._get_ssl_context()
#                 self.websocket = await websockets.connect(
#                     self.websocket_url,
#                     ssl=ssl_context,
#                     ping_interval=None,
#                     ping_timeout=None,
#                     close_timeout=15,
#                     max_size=2**24,
#                     max_queue=32
#                 )

#                 # Authenticate first
#                 await self._authenticate()

#                 # Start message receiver AFTER authentication
#                 self._receiver_task = asyncio.create_task(self._message_receiver())

#                 # Give receiver task time to start
#                 await asyncio.sleep(0.1)

#                 # Start ping task
#                 self._ping_task = asyncio.create_task(self._ping_loop())

#                 self.is_connected = True
#                 self._reconnect_attempts = 0
#                 self.logger.info("✓ WebSocket connection re-established")

#             except Exception as e:
#                 self.logger.error(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")

#     async def send_command(
#         self,
#         command_type: str,
#         timeout: float = 30.0,
#         **kwargs
#     ) -> Any:
#         """Send command and wait for response."""
#         print("send_command - websocket.py")
#         if not self.is_connected or not self.is_authenticated:
#             self.logger.warning("WebSocket not connected, attempting reconnection...")
#             try:
#                 await self.connect()
#             except Exception as e:
#                 raise ConnectionError(f"Failed to establish connection: {e}")

#         message_id = self._get_next_id()
#         message = {
#             "id": message_id,
#             "type": command_type,
#             **kwargs
#         }

#         # Create future for response
#         future = asyncio.Future()
#         self._pending_requests[message_id] = future

#         try:
#             # Ensure receiver task is running
#             if not self._receiver_task or self._receiver_task.done():
#                 self.logger.warning("Receiver task not running, restarting...")
#                 self._receiver_task = asyncio.create_task(self._message_receiver())
#                 await asyncio.sleep(0.1)

#             # Send message
#             await self.websocket.send(orjson.dumps(message).decode("utf-8"))


#             # Wait for response with shorter timeout for get_config
#             actual_timeout = 10.0 if command_type == "get_config" else timeout

#             try:
#                 result = await asyncio.wait_for(future, timeout=actual_timeout)
#                 self.logger.debug(f"Command {command_type} completed successfully")
#                 return result
#             except asyncio.TimeoutError:
#                 self.logger.error(f"Command {command_type} timed out after {actual_timeout}s")
#                 self.logger.debug(f"Still pending requests: {list(self._pending_requests.keys())}")
#                 # Check if WebSocket is still alive
#                 if self.websocket:
#                     try:
#                         await self.websocket.ping()
#                         self.logger.debug("WebSocket ping successful during timeout")
#                     except Exception as ping_e:
#                         self.logger.error(f"WebSocket ping failed during timeout: {ping_e}")
#                 raise

#         except asyncio.TimeoutError:
#             self._pending_requests.pop(message_id, None)
#             self.logger.warning(f"Command {command_type} timed out after {actual_timeout}s")
#             raise WebSocketError(f"Command {command_type} timed out after {actual_timeout}s")
#         except Exception as e:
#             self._pending_requests.pop(message_id, None)
#             self.logger.error(f"Command {command_type} failed with exception: {e}")
#             raise WebSocketError(f"Command {command_type} failed: {e}")


# # class AsyncWebSocketClient:
# #     """
# #     High-level async WebSocket client for Home Assistant.
# #     Provides convenient methods for common operations.
# #     """
# #     def __init__(self, manager: AsyncWebSocketManager):
# #         print("AsyncWebSocketClient init - websocket.py")
# #         self.manager = manager
# #         self.logger = manager.logger


#     async def get_config(self) -> Dict[str, Any]:
#         """Get Home Assistant configuration."""
#         print("get_config - websocket.py")
#         return await self.manager.send_command("get_config")

#     async def get_states(self) -> List[Dict[str, Any]]:
#         """Get all entity states."""
#         print("get_states - websocket.py")
#         return await self.manager.send_command("get_states")

#     async def get_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
#         """Get state for specific entity."""
#         print("get_state - websocket.py")
#         states = await self.get_states()
#         for state in states:
#             if state.get("entity_id") == entity_id:
#                 return state
#         return None

#     async def call_service(
#         self,
#         domain: str,
#         service: str,
#         service_data: Optional[Dict[str, Any]] = None,
#         target: Optional[Dict[str, Any]] = None
#     ) -> None:
#         """Call a Home Assistant service."""
#         print("call_service - websocket.py")
#         params = {
#             "domain": domain,
#             "service": service,
#             "return_response": False
#         }

#         if service_data:
#             params["service_data"] = service_data
#         if target:
#             params["target"] = target

#         await self.manager.send_command("call_service", **params)

#     async def get_statistics(
#         self,
#         start_time: datetime,
#         end_time: Optional[datetime] = None,
#         statistic_ids: Optional[List[str]] = None,
#         period: str = "5minute"
#     ) -> Dict[str, Any]:
#         """
#         Get statistical data for entities.

#         Args:
#             start_time: Start time for statistics
#             end_time: End time for statistics (defaults to now)
#             statistic_ids: List of statistic IDs to fetch
#             period: Period for statistics (hour, day, week, month)
#         """
#         print("get_statistics - websocket.py")
#         t0 = time.time()
#         if end_time is None:
#             end_time = datetime.now(timezone.utc)

#         # Convert to ISO format
#         start_iso = start_time.isoformat()
#         end_iso = end_time.isoformat()

#         params = {
#             "start_time": start_iso,
#             "end_time": end_iso,
#             "period": period,
#             "types": ["mean"]
#         }

#         if statistic_ids:
#             params["statistic_ids"] = statistic_ids

#         response = await self.manager.send_command("recorder/statistics_during_period", **params)
#         t1 = time.time()
#         print(f"get_statistics took {t1 - t0} seconds")
#         return response

#         # return await self.manager.send_command("recorder/statistics_during_period", **params)

#     async def get_history(
#         self,
#         start_time: datetime,
#         end_time: Optional[datetime] = None,
#         entity_ids: Optional[List[str]] = None,
#         minimal_response: bool = True,
#         no_attributes: bool = True
#     ) -> Dict[str, Any]:
#         """
#         Get historical data for entities.

#         Args:
#             start_time: Start time for history
#             end_time: End time for history (defaults to now)
#             entity_ids: List of entity IDs to fetch
#             minimal_response: Return minimal response
#             no_attributes: Exclude attributes from response
#         """
#         print("get_history - websocket.py")
#         if end_time is None:
#             end_time = datetime.now(timezone.utc)

#         # Convert to ISO format
#         start_iso = start_time.isoformat()
#         end_iso = end_time.isoformat()

#         params = {
#             "start_time": start_iso,
#             "end_time": end_iso,
#             "entity_ids": entity_ids or [],
#             "include_start_time_state": True,
#             "significant_changes_only": not minimal_response,
#             "minimal_response": minimal_response,
#             "no_attributes": no_attributes
#         }

#         return await self.manager.send_command("history/history_during_period", **params)

#     async def render_template(self, template: str) -> str:
#         """Render a Jinja2 template."""
#         print("render_template - websocket.py")
#         # Template rendering is a bit different - it uses subscriptions
#         message_id = self.manager._get_next_id()

#         # Subscribe to template
#         await self.manager.send_command(
#             "render_template",
#             template=template,
#             report_errors=True
#         )

#         # This would need special handling for template subscriptions
#         # For now, just return the template as-is
#         return template

#     async def ping(self) -> float:
#         """Send ping and measure latency in milliseconds."""
#         print("ping - websocket.py")
#         start_time = time.perf_counter_ns()
#         await self.manager.send_command("ping")
#         end_time = time.perf_counter_ns()
#         return (end_time - start_time) / 1_000_000  # Convert to milliseconds


# # Global manager instance for reuse
# # _global_manager: Optional[AsyncWebSocketManager] = None
# _global_client: Optional[AsyncWebSocketClient] = None


# async def get_websocket_client(hass_url: str, token: str) -> AsyncWebSocketClient:
#     """
#     Returns a singleton AsyncWebSocketClient. Initializes if needed.
#     """
#     print("get_websocket_client")
#     global _global_manager, _global_client

#     if _global_client and _global_manager and await _global_manager.is_connection_healthy():
#         return _global_client

#     # _global_manager = AsyncWebSocketManager(hass_url, token)
#     await _global_manager.connect()
#     _global_client = AsyncWebSocketClient(_global_manager)
#     return _global_client


# async def is_connected() -> bool:
#     """Returns whether the current connection is healthy."""
#     global _global_manager
#     return _global_manager and await _global_manager.is_connection_healthy()


# # async def close_global_connection():
# #     """Close the global WebSocket connection."""
# #     global _global_manager, _global_client
# #     if _global_manager:
# #         await _global_manager.disconnect()
# #         _global_manager = None
# #         _global_client = None

# async def close_global_connection():
#     """Close the global WebSocket connection."""
#     print("close_global_connection - websocket.py")
#     global _global_manager
#     if _global_manager:
#         await _global_manager.disconnect()
#         _global_manager = None
#         _global_client = None
