"""Feature tests: who may watch and steer the browser.

Only the signed-in user's own browser session reaches the live view: an
external agent holding a valid bearer token is refused on every browser
endpoint and on the live-page socket, and a page on another site cannot open
the live-page socket even though the user's cookie rides along with it.
Real Flask app, real cookie session and real bearer token, served on a real
socket; no browser is needed or started.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from tests._browser_live import (
    bearer_header,
    close_all,
    closing_code,
    cookie_header,
    launches,
    listed,
    live_server,
    signed_in,
    stream_socket,
    use_own_data_dir,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def no_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    """The browser launches seen so far. Should a refused call start a browser after all,
    its profile is under *tmp_path* and it is closed on the way out."""
    use_own_data_dir(tmp_path, monkeypatch)
    before = launches()
    yield before
    if launches() != before:
        close_all()


@pytest.fixture
def app(db: sqlite3.Connection, no_browser: int) -> Flask:
    from api import create_app  # noqa: PLC0415 — built per test, on the test's database

    built = create_app()
    built.config["TESTING"] = True
    return built


def test_an_external_agent_with_a_valid_token_cannot_reach_the_users_browser(
    app: Flask, no_browser: int
) -> None:
    agent: FlaskClient = app.test_client()
    bearer = bearer_header()
    refused = [
        agent.get("/api/browser/tabs", headers=bearer),
        agent.post(
            "/api/browser/tabs", json={"url": "http://127.0.0.1:9/"}, headers=bearer
        ),
        agent.delete("/api/browser/tabs/any-tab", headers=bearer),
        agent.post(
            "/api/browser/navigate/any-tab",
            json={"url": "http://127.0.0.1:9/"},
            headers=bearer,
        ),
        agent.post(
            "/api/browser/history/any-tab", json={"action": "back"}, headers=bearer
        ),
        agent.post("/api/browser/handoffs/any-handoff", headers=bearer),
    ]
    assert [r.status_code for r in refused] == [403] * 6, [r.data for r in refused]

    client, _token = signed_in(app)
    assert listed(client) == []  # the same list is the user's to read

    with live_server(app) as host, stream_socket(host, "any-tab", bearer) as ws:
        messages, code = closing_code(ws)
    assert (messages, code) == (['{"type": "error", "message": "Unauthorized"}'], 4403)
    assert launches() == no_browser


def test_another_sites_page_cannot_open_the_live_page_with_the_users_cookie(
    app: Flask, no_browser: int
) -> None:
    _client, token = signed_in(app)
    with live_server(app) as host:
        foreign = cookie_header(token) | {"Origin": "https://evil.example"}
        with stream_socket(host, "any-tab", foreign) as ws:
            refused = closing_code(ws)
        # The socket client names the host without its port, so this is the origin
        # matching the Host it sends — what a page served by this server sends.
        own = cookie_header(token) | {"Origin": "http://127.0.0.1"}
        with stream_socket(host, "no-such-tab", own) as ws:
            unknown = closing_code(ws)
    assert refused == (['{"type": "error", "message": "Forbidden origin"}'], 4403)
    assert unknown == (
        ['{"type": "closed"}'],
        4404,
    )  # let through, to a tab that is not open
    assert launches() == no_browser
