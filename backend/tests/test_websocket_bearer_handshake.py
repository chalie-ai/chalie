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
    passes the raw token as the ``?token=`` query parameter on ``/ws``; and
  * ordinary HTTP requests, where the same token rides ``Authorization: Bearer``.

These exercise the REAL production path with zero mocks:
  * the real ``WrapperAuthService.create_token`` mints the token (the exact call
    ``POST /api/wrappers`` makes), persisted in the real ``db`` SQLite fixture;
  * the real ``validate_session`` runs first and genuinely fails (no cookie),
    so the bearer fallback is genuinely exercised;
  * a real consumer object at the WS boundary fulfils the same
    ``send(raw)`` / ``close()`` / ``receive()`` contract the browser socket
    fulfils — the same pattern as test_websocket_broadcast_fanout.py;
  * the real Flask route ``GET /chat/subagents/active`` (``@require_auth``)
    proves the token authenticates an HTTP request too.

Against the pre-fix handler (cookie-only ``validate_session`` at
websocket.py:40, no ``?token=`` fallback) the bearer-only handshake is closed
``Unauthorized`` — ``test_ws_handshake_accepts_bearer_token_query_param`` fails
RED. It passes GREEN once the fallback lands.
"""

import json
import sqlite3
from collections.abc import Generator
from typing import cast

import pytest

from services.websocket_broker import WebSocketBroker

pytestmark = pytest.mark.unit


class _BoundarySocket:
    """A real consumer at the WS boundary — the same ``send`` / ``close`` /
    ``receive`` contract the production browser socket fulfils. Records sent
    frames and whether it was closed. ``receive`` returns ``None`` once (so the
    handler's loop emits a single ping) then raises to end the loop, mirroring a
    browser socket that drops the connection."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._receives = 0

    def send(self, raw: str) -> None:
        self.sent.append(raw)

    def close(self) -> None:
        self.closed = True

    def receive(self, timeout: int = 60) -> None:
        self._receives += 1
        if self._receives == 1:
            return None  # triggers one broker ping, then we end the loop
        raise RuntimeError("socket closed")

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


def test_ws_handshake_accepts_bearer_token_query_param(
    db: sqlite3.Connection, broker: WebSocketBroker
) -> None:
    """A handshake carrying a valid raw token as ``/ws?token=<raw>`` (no cookie,
    as a mobile socket has) must authenticate: the handler must NOT send the
    ``Unauthorized`` error frame and must NOT close the socket, and the
    connection must be registered with the broker. Pre-fix the cookie-only check
    closes it — RED here."""
    from api import create_app
    from api.websocket import _ws_handler

    raw = _mint_token("Mobile — handshake")

    app = create_app()
    app.config["TESTING"] = True
    socket = _BoundarySocket()

    with app.test_request_context("/ws?token=" + raw):
        _ws_handler(socket)

    assert not socket.closed, (
        "the handshake carried a valid bearer token via ?token= but the socket "
        "was closed — the bearer fallback did not run"
    )
    assert "error" not in socket.sent_types(), (
        "an Unauthorized error frame was sent despite a valid ?token=; the only "
        f"frame on a healthy handshake is the broker ping. Got: {socket.sent_types()}"
    )
    assert socket in broker._connections, (
        "an authenticated handshake must register the socket with the broker"
    )


def test_ws_handshake_rejects_missing_and_invalid_token(
    db: sqlite3.Connection, broker: WebSocketBroker
) -> None:
    """One common path / self-no-op: no cookie AND no usable token => the
    existing Unauthorized close fires unchanged. Covers a token-less handshake
    and a garbage ``?token=`` — neither must authenticate."""
    from api import create_app
    from api.websocket import _ws_handler

    app = create_app()
    app.config["TESTING"] = True

    for url in ("/ws", "/ws?token=not-a-real-token"):
        socket = _BoundarySocket()
        with app.test_request_context(url):
            _ws_handler(socket)
        assert socket.closed, (
            f"handshake {url!r} carried no valid auth and must be closed"
        )
        assert "error" in socket.sent_types(), (
            f"handshake {url!r} must receive the Unauthorized error frame; got "
            f"{socket.sent_types()}"
        )
        assert socket not in broker._connections, (
            f"an unauthenticated handshake {url!r} must NOT be registered"
        )


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
            "/chat/subagents/active",
            headers={"Authorization": "Bearer " + raw},
        )

    assert resp.status_code == 200, (
        "a valid bearer token must authenticate GET /chat/subagents/active; got "
        f"{resp.status_code} {resp.get_data(as_text=True)!r}"
    )
    assert "subagents" in resp.get_json(), (
        "the authenticated route must return its real payload"
    )
