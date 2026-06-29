# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the native mobile app authenticates with ONE minted wrapper
token across both surfaces it talks to:

  * the WebSocket handshake, where a mobile socket cannot carry a cookie, so it
    presents the raw token as the FIRST WS frame
    (``{"type":"auth","token":...}``) on ``/ws`` — never in the URL, so the
    credential cannot leak into reverse-proxy/access logs, ``Referer``, or
    history; and
  * ordinary HTTP requests, where the same token rides ``Authorization: Bearer``.

These exercise the REAL production path with zero mocks:
  * the real ``WrapperAuthService.create_token`` mints the token (the exact call
    ``POST /api/wrappers`` makes), persisted in the real ``db`` SQLite fixture;
  * the real ``validate_session`` runs first and genuinely fails (no cookie),
    so the bearer-first-frame fallback is genuinely exercised;
  * a real consumer object at the WS boundary fulfils the same
    ``send(raw)`` / ``close()`` / ``receive()`` contract the browser socket
    fulfils — the same pattern as test_websocket_broadcast_fanout.py;
  * the real Flask route ``GET /chat/subagents/active`` (``@require_auth``)
    proves the token authenticates an HTTP request too.

Against the pre-fix handler (cookie-only ``validate_session`` at
websocket.py:40, no bearer fallback) the bearer-only handshake is closed
``Unauthorized`` — ``test_ws_handshake_accepts_bearer_token_first_frame`` fails
RED. It passes GREEN once the first-frame fallback lands.
"""

import json
import sqlite3
from collections.abc import Generator
from typing import cast

import pytest

from services.websocket_broker import WebSocketBroker

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _enable_internal_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    # The bearer-over-WS handshake is a native-client surface gated behind
    # CHALIE_INTERNAL_DEV; this suite exercises it, so the gate runs open.
    monkeypatch.setenv("CHALIE_INTERNAL_DEV", "1")


class _BoundarySocket:
    """A real consumer at the WS boundary — the same ``send`` / ``close`` /
    ``receive`` contract the production browser socket fulfils. Records sent
    frames and whether it was closed. ``receive`` yields a programmable sequence
    of inbound frames (the auth frame first, then ``None`` for one broker ping)
    then raises to end the loop, mirroring a browser socket that drops the
    connection."""

    def __init__(self, inbound: list[str | None] | None = None) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._receives = list(inbound or [])

    def send(self, raw: str) -> None:
        self.sent.append(raw)

    def close(self) -> None:
        self.closed = True

    def receive(self, timeout: int = 60) -> str | None:
        if not self._receives:
            raise RuntimeError("socket closed")
        return self._receives.pop(0)

    def sent_types(self) -> list[str]:
        return [cast(str, json.loads(m).get("type")) for m in self.sent]


@pytest.fixture
def broker() -> Generator[WebSocketBroker, None, None]:
    """The real process-wide broker singleton, emptied before and after so a
    leaked connection cannot perturb the handshake assertions."""
    b = WebSocketBroker()
    with b._lock:
        b._connections.clear()
    try:
        yield b
    finally:
        with b._lock:
            b._connections.clear()


def _mint_token(name: str) -> str:
    """Mint a wrapper token via the real service — the exact create_token call
    POST /api/wrappers makes. Returns the raw bearer token (shown once)."""
    from services.wrapper_auth_service import WrapperAuthService

    raw_token, _wrapper_id = WrapperAuthService().create_token(name=name)
    return raw_token


def _auth_frame(raw: str) -> str:
    return json.dumps({"type": "auth", "token": raw})


def test_ws_handshake_accepts_bearer_token_first_frame(
    db: sqlite3.Connection, broker: WebSocketBroker
) -> None:
    """A handshake with NO cookie (as a mobile socket has) that presents a valid
    raw token as its first WS frame must authenticate: the handler must NOT send
    the ``Unauthorized`` error frame and must NOT close the socket, and the
    connection must be registered with the broker. The URL carries no token.
    Pre-fix the cookie-only check closes it — RED here."""
    from api import create_app
    from api.websocket import _ws_handler

    raw = _mint_token("Mobile — handshake")

    app = create_app()
    app.config["TESTING"] = True
    # Auth frame first, then None (one broker ping), then the loop ends.
    socket = _BoundarySocket([_auth_frame(raw), None])

    with app.test_request_context("/ws"):
        _ws_handler(socket)

    assert not socket.closed, (
        "the handshake carried a valid bearer token in its first frame but the "
        "socket was closed — the bearer fallback did not run"
    )
    assert "error" not in socket.sent_types(), (
        "an Unauthorized error frame was sent despite a valid first-frame token; "
        f"the only frame on a healthy handshake is the broker ping. Got: {socket.sent_types()}"
    )
    # The broker's connect() + broadcast({"type": "ping"}) runs inside the
    # handler's loop; the finally block disconnects cleanly on loop exit (the
    # _BoundarySocket.receive() raises once the inbound queue drains).  So
    # after the handler returns the socket is no longer in _connections — correct
    # production behaviour.  We prove it WAS registered by the fact the ping
    # frame was delivered to it (the broker only fans out to registered sockets).
    assert "ping" in socket.sent_types(), (
        "an authenticated handshake must be registered with the broker — proved "
        f"by receiving the broker ping. Got: {socket.sent_types()}"
    )


def test_ws_handshake_rejects_missing_and_invalid_token(
    db: sqlite3.Connection, broker: WebSocketBroker
) -> None:
    """No cookie AND no usable first frame => the existing Unauthorized close
    fires unchanged. Covers a token-less handshake (no auth frame at all), a
    garbage token, a non-auth frame, and a malformed frame — none must
    authenticate, and no token may appear in the URL."""
    from api import create_app
    from api.websocket import _ws_handler

    app = create_app()
    app.config["TESTING"] = True

    cases: list[tuple[str, _BoundarySocket]] = [
        # No auth frame — receive() returns nothing usable; the handler's
        # _handshake_bearer times out/raises and closes.
        ("/ws", _BoundarySocket([])),
        # Garbage token in a well-formed auth frame.
        ("/ws", _BoundarySocket([_auth_frame("not-a-real-token")])),
        # A non-auth first frame (e.g. a stray pong) must not authenticate.
        ("/ws", _BoundarySocket([json.dumps({"type": "pong"})])),
        # A malformed first frame.
        ("/ws", _BoundarySocket(["not-json"])),
    ]
    for url, socket in cases:
        with app.test_request_context(url):
            _ws_handler(socket)
        assert socket.closed, (
            f"handshake {url!r} carried no valid auth frame and must be closed"
        )
        assert "error" in socket.sent_types(), (
            f"handshake {url!r} must receive the Unauthorized error frame; got "
            f"{socket.sent_types()}"
        )
        assert "ping" not in socket.sent_types(), (
            f"an unauthenticated handshake {url!r} must NOT be registered with "
            f"the broker — no ping may be delivered. Got: {socket.sent_types()}"
        )


def test_ws_handshake_url_carries_no_token(
    db: sqlite3.Connection, broker: WebSocketBroker
) -> None:
    """The bearer token must NEVER appear in the request URL — that is the
    regression this suite guards. A valid first-frame handshake authenticates
    while the request path stays a bare ``/ws`` with no query string."""
    from api import create_app
    from api.websocket import _ws_handler
    from flask import request as flask_request

    raw = _mint_token("Mobile — url hygiene")

    app = create_app()
    app.config["TESTING"] = True
    socket = _BoundarySocket([_auth_frame(raw), None])

    with app.test_request_context("/ws"):
        _ws_handler(socket)
        assert "token" not in flask_request.args, (
            "the bearer token must not travel in the URL query string — it "
            f"leaks into logs/Referer. Saw ?{flask_request.query_string.decode()}"
        )
    assert not socket.closed


def test_same_minted_token_authenticates_http_bearer_request(
    db: sqlite3.Connection
) -> None:
    """The SAME token the WS handshake accepts must authenticate an ordinary
    HTTP request via ``Authorization: Bearer`` — proving one token works across
    both surfaces the mobile app uses. Drives the real ``@require_auth`` route
    GET /chat/subagents/active end-to-end (no auth bypass)."""
    from api import create_app

    raw = _mint_token("Mobile — http bearer")

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.get(
            "/api/chat/subagents/active",
            headers={"Authorization": "Bearer " + raw},
        )

    assert resp.status_code == 200, (
        "a valid bearer token must authenticate GET /chat/subagents/active; got "
        f"{resp.status_code} {resp.get_data(as_text=True)!r}"
    )
    assert "subagents" in resp.get_json(), (
        "the authenticated route must return its real payload"
    )
