import os

# Bind address: the API port. Every request on it needs credentials (see emhass.api_auth).
bind = [f"{os.getenv('IP', '0.0.0.0')}:{os.getenv('PORT', '5000')}"]

# Home Assistant add-on ingress gets a listener of its own, so ingress is recognised by the socket
# that accepted the connection and never by a header. The add-on does not publish this port and the
# app refuses every peer except the Supervisor on it. Same decision as emhass.api_auth.ingress_port().
_ingress = os.getenv("EMHASS_INGRESS_PORT", "").strip() or (
    "8099" if os.getenv("SUPERVISOR_TOKEN") else ""
)
try:
    _api_port = int(os.getenv("PORT", "5000").strip())
except ValueError:
    _api_port = 5000
# compared as numbers: "05000" is the API port too
if _ingress.isdigit() and 0 < int(_ingress) < 65536 and int(_ingress) != _api_port:
    bind.append(f"0.0.0.0:{int(_ingress)}")

# Worker configuration
workers = int(os.getenv("WEB_CONCURRENCY", 1))
# Note: threads is not used with async workers (uvicorn), only for sync workers

# Worker connections (only for async workers)
worker_connections = int(os.getenv("WORKER_CONNECTIONS", 1000))

# Process management
max_requests = int(os.getenv("MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", 100))
preload_app = True

# Development settings
reload = (os.getenv("WEB_RELOAD", "false").lower()) == "true"

# Timeouts
timeout = int(os.getenv("WEB_TIMEOUT", 4000))
graceful_timeout = 60
keepalive = 2

# Logging
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "warning")

# Security
# Access control trusts the TCP peer address only. uvicorn must not replace it with an
# X-Forwarded-For value, or any process on loopback could choose which peer it appears to be.
forwarded_allow_ips = ""
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
worker_tmp_dir = "/dev/shm"  # Use memory for worker temp files if available


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("EMHASS server is ready with %s workers", workers)


def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info("Worker %s received SIGINT/SIGQUIT", worker.pid)


def on_exit(server):
    """Called just before exiting."""
    server.log.info("EMHASS server is shutting down")
