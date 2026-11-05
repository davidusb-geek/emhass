"""Make ``aioresponses`` usable with every ``aiohttp`` the test extra can install.

``aioresponses`` 0.7.9 builds the mocked ``aiohttp.ClientResponse`` by hand, and two of
the things it does stopped working with ``aiohttp`` 3.13/3.14:

* ``ClientResponse.__init__`` gained a required keyword-only ``stream_writer``
  argument, which ``aioresponses`` does not pass (``TypeError`` on every mocked
  request, which the code under test then reports as a request failure).
* Feeding a body larger than the stream's high-water mark (about 128 KiB here) makes
  the ``StreamReader`` call ``pause_reading()`` on its protocol. The protocol
  ``aioresponses`` gives it has no parser, so ``aiohttp`` raises ``AssertionError``.
  A mocked body is already fully in memory, so flow control is meaningless for it.

This module patches the two names ``aioresponses.core`` looks up when it builds a
response. It is test infrastructure only; nothing under ``src/`` imports it. Import
``aioresponses`` from here instead of from the package so the adapter is installed:

    from tests.aioresponses_compat import aioresponses

On an ``aiohttp`` whose ``ClientResponse`` does not take ``stream_writer`` the response
class is left alone, so the same tests run on the older and the current ``aiohttp``.
"""

import inspect
from unittest.mock import Mock

from aiohttp import ClientResponse, StreamReader
from aioresponses import aioresponses, core

__all__ = ["aioresponses", "install", "make_client_response", "stream_reader_factory"]

# Matches the 2**16 limit aioresponses itself uses for the mocked stream.
_STREAM_LIMIT = 2**16


def requires_stream_writer(response_class: type) -> bool:
    """True when ``response_class.__init__`` has a required ``stream_writer`` argument."""
    parameter = inspect.signature(response_class.__init__).parameters.get("stream_writer")
    return parameter is not None and parameter.default is inspect.Parameter.empty


def make_client_response(base: type = ClientResponse) -> type:
    """Return ``base``, or a subclass of it that supplies ``stream_writer`` when required.

    The mocked response was never written to a socket, so a stand-in writer with an
    ``output_size`` of 0 (the only attribute ``aiohttp`` reads from it) is accurate.
    """
    if not requires_stream_writer(base):
        return base

    class CompatClientResponse(base):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("stream_writer", Mock(output_size=0))
            super().__init__(*args, **kwargs)

    CompatClientResponse.__name__ = base.__name__
    CompatClientResponse.__qualname__ = base.__qualname__
    return CompatClientResponse


def stream_reader_factory(loop=None) -> StreamReader:
    """Mocked-body ``StreamReader`` whose protocol ignores pause/resume requests."""
    return StreamReader(Mock(), limit=_STREAM_LIMIT, loop=loop)


def install() -> None:
    """Patch ``aioresponses.core`` for the installed ``aiohttp``; safe to call repeatedly."""
    core.ClientResponse = make_client_response(ClientResponse)
    core.stream_reader_factory = stream_reader_factory


install()
