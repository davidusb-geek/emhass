#!/bin/bash
set -e

# Set defaults
EMHASS_SERVER_TYPE=${EMHASS_SERVER_TYPE:-async}
WORKER_CLASS=${WORKER_CLASS:-uvicorn.workers.UvicornWorker}
PORT=${PORT:-5000}
IP=${IP:-0.0.0.0}

echo "=================================================="
echo "   Starting EMHASS Energy Management System"
echo "=================================================="
echo "   Configuration:"
echo "   Server Type: $EMHASS_SERVER_TYPE"
echo "   Worker Class: $WORKER_CLASS"
echo "   Bind Address: $IP:$PORT"
echo "   Python Path: $(which python3 2>/dev/null || echo 'N/A')"
echo "   UV Version: $(uv --version 2>/dev/null || echo 'N/A')"
echo "=================================================="

# Validate server type
case "$EMHASS_SERVER_TYPE" in
    sync)
        echo "Using synchronous Flask server with gunicorn"
        echo "Starting gunicorn with WSGI workers..."
        exec uv run --frozen gunicorn emhass.web_server:app -c gunicorn.conf.py
        ;;
    async)
        echo "Using asynchronous Quart server with gunicorn + uvicorn workers"
        echo "Starting gunicorn with $WORKER_CLASS workers..."
        exec uv run --frozen gunicorn emhass.web_server_async:app -c gunicorn.conf.py -k "$WORKER_CLASS"
        ;;
    *)
        echo "  ERROR: Unknown server type '$EMHASS_SERVER_TYPE'"
        echo "  Valid options:"
        echo "   - sync: Flask with gunicorn WSGI workers"
        echo "   - async: Quart with gunicorn + uvicorn ASGI workers (recommended)"
        echo ""
        echo "  Example: docker run -e EMHASS_SERVER_TYPE=async your-image"
        exit 1
        ;;
esac
