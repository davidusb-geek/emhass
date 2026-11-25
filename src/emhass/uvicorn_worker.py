"""The gunicorn worker class the Docker image runs (``-k emhass.uvicorn_worker.UvicornWorker``).

EMHASS serves no WebSocket route, so the worker runs uvicorn with ``ws="none"``: an upgrade request is
answered as plain HTTP and goes through the same auth hook as any other request. With uvicorn 0.54 the
default (``ws="auto"``) picks the websockets sans-io protocol, which leaves an upgrade with more than
128 headers unanswered and, if one is still pending at shutdown, stops the server without running the
Quart ``after_serving`` hooks. ``web_server.main()`` sets the same value for the direct server.
"""

from uvicorn.workers import UvicornWorker as _UvicornWorker

WS_PROTOCOL = "none"


class UvicornWorker(_UvicornWorker):
    CONFIG_KWARGS = {**_UvicornWorker.CONFIG_KWARGS, "ws": WS_PROTOCOL}
