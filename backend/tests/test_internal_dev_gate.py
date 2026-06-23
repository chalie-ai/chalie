"""Feature test: the CHALIE_INTERNAL_DEV gate hides every native-mobile surface.

Real production path, zero mocks: the real create_app() Flask app, the real
``@internal_only`` decorator on the real routes, against the real db fixture.
With the flag unset — the released default — the pairing surfaces answer 404 and
/auth/status reports ``internal_dev`` False; with it set they open. This is the
switch that lets the native-mobile feature land on rc without surfacing in prod.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient

pytestmark = pytest.mark.unit


def _client() -> FlaskClient:
    from api import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app.test_client()


def test_disabled_hides_mobile_surfaces(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHALIE_INTERNAL_DEV", raising=False)
    client = _client()

    assert client.get('/auth/status').get_json()["internal_dev"] is False
    # 404 — gated before auth even runs, indistinguishable from a missing route.
    assert client.get('/auth/username').status_code == 404
    assert client.get('/pairing/').status_code == 404


def test_enabled_opens_mobile_surfaces(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHALIE_INTERNAL_DEV", "1")
    client = _client()

    assert client.get('/auth/status').get_json()["internal_dev"] is True
    # Gate open → the route now enforces auth (401 without a cookie session).
    assert client.get('/auth/username').status_code == 401
    # Gate open → the pairing MPA entry is served.
    assert client.get('/pairing/').status_code == 200
