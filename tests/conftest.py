"""Shared pytest setup.

The web server refuses requests without credentials (``emhass.api_auth``, GHSA-c39q-j4r5-9xq5). Most
older web tests exercise handlers through an unauthenticated test client, so they run with the explicit
operator opt-out. The modules that test access control itself (``test_api_auth*``) get a clean
environment and set up what they need.

The opt-out is set at import time as well, because ``web_server`` reads its token configuration when it
is first imported.
"""

import os

import pytest

# Keep the import-time configuration from generating a token file in a real data folder.
os.environ.setdefault("EMHASS_API_AUTH_DISABLED", "true")

_AUTH_MODULES = ("test_api_auth", "test_api_ingress", "test_api_provisioning")
_AUTH_ENV = (
    "EMHASS_API_AUTH_DISABLED",
    "EMHASS_API_TOKEN",
    "EMHASS_API_TOKEN_AUTOGENERATE",
    "EMHASS_INGRESS_PORT",
    "EMHASS_INGRESS_PEERS",
    "SUPERVISOR_TOKEN",
)


@pytest.fixture(autouse=True)
def _api_auth_environment(request, monkeypatch):
    if request.module.__name__.rsplit(".", 1)[-1] in _AUTH_MODULES:
        for name in _AUTH_ENV:
            monkeypatch.delenv(name, raising=False)
        return
    monkeypatch.setenv("EMHASS_API_AUTH_DISABLED", "true")


@pytest.fixture(autouse=True)
def _own_remote_commits(tmp_path, monkeypatch):
    """A coordinator with remote solvers keeps a record of what it committed
    them to in its data folder; in a test, in a folder of the test's own,
    never in the data folder the tests share."""
    from emhass.coordinator import backend

    monkeypatch.setattr(
        backend,
        "_remote_commits_path",
        lambda opt: tmp_path / "coordinator_remote_commits.json",
        raising=False,
    )


@pytest.fixture(autouse=True)
def _no_plan_stepper_left():
    """A participant's plan stepper runs a thread: none outlives its test (it
    would publish into a later test's data folder)."""
    yield
    import sys

    participant_api = sys.modules.get("emhass.participant_api")
    if participant_api is not None:
        for stepper in list(getattr(participant_api, "_steppers", {}).values()):
            stepper.stop()
        participant_api._steppers.clear()
