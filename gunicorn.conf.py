import os

# Bind address
bind = f"{os.getenv('IP', '0.0.0.0')}:{os.getenv('PORT', '5000')}"

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
