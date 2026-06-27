# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — WebSocket broadcast fans out to every open surface.

A Chalie instance belongs to one user, but that user may have the UI open on
several devices/surfaces at once (phone, laptop, multiple tabs). Every surface
opens its own connection on the single fixed ``/ws`` channel; the server
broadcasts each event on that channel; every connected surface must receive it
so they all stay in sync.

These tests exercise the REAL production path with zero mocks:
  * the real ``WebSocketBroker`` process-wide singleton (the broadcast channel),
  * real consumer objects at the WS boundary (the same ``send(raw)`` contract the
    browser socket fulfils — exactly the pattern the broker calls in production),
  * the real user-message echo tail ``api.chat._broadcast_user_echo``.

Before the multi-surface fix the broker held a single socket reference and a
second connection overwrote the first, so only the most-recently-connected
surface received anything — ``test_disconnect_removes_only_the_closed_surface``
fails RED against that code and passes GREEN with the connection-set broker.
"""

import json
import sqlite3
from collections.abc import Generator
from typing import cast

import pytest

from services.websocket_broker import WebSocketBroker

pytestmark = pytest.mark.unit


# ── helpers ───────────────────────────────────────────────────────────────────


class _Surface:
    """A real consumer at the WS boundary — one open device/tab. The broker
    pushes JSON strings to it exactly as it does to a browser connection."""

    def __init__(self) -> None:
        self.received: list[dict[str, object]] = []

    def send(self, raw: str) -> None:
        self.received.append(json.loads(raw))

    def types(self) -> list[str]:
        return [cast(str, m.get("type")) for m in self.received]


class _DeadSurface:
    """A surface whose socket has died — every ``send`` raises, as a closed or
    half-open browser socket would. Counts calls so we can prove it is pruned."""

    def __init__(self) -> None:
        self.calls = 0

    def send(self, raw: str) -> None:
        self.calls += 1
        raise RuntimeError("socket closed")


@pytest.fixture
def broker() -> Generator[WebSocketBroker, None, None]:
    """The real process-wide broker singleton, emptied before and after the test
    so leaked connections from other tests cannot perturb fan-out assertions."""
    b = WebSocketBroker()
    with b._lock:
        b._connections.clear()
    try:
        yield b
    finally:
        with b._lock:
            b._connections.clear()


# ── B. Targeted disconnect: closing one surface keeps the others live ───────────


def test_disconnect_removes_only_the_closed_surface(db: sqlite3.Connection, broker: WebSocketBroker) -> None:
    """When one surface disconnects, the others must keep receiving. The previous
    single-slot broker cleared the only reference on any disconnect, so the
    surviving tab went dark — this guards against that regression."""
    phone = _Surface()
    laptop = _Surface()
    broker.connect(phone)
    broker.connect(laptop)

    broker.disconnect(phone)  # phone tab closed

    broker.broadcast({"type": "notification", "text": "still here"})

    assert phone.received == [], "a disconnected surface must receive nothing"
    assert laptop.types() == ["notification"], (
        f"the surviving surface must still receive broadcasts; got {laptop.types()}"
    )


# ── C. Dead-socket pruning: a failed send removes that surface, others unaffected ─


def test_broadcast_prunes_dead_surface_and_keeps_delivering(db: sqlite3.Connection, broker: WebSocketBroker) -> None:
    """A surface whose ``send`` raises (closed/half-open socket) is pruned on the
    failing broadcast and is not contacted again, while healthy surfaces continue
    to receive — one stale connection cannot block or accumulate."""
    healthy = _Surface()
    dead = _DeadSurface()
    broker.connect(healthy)
    broker.connect(dead)

    broker.broadcast({"type": "status", "stage": "first"})
    assert dead.calls == 1, "the dead surface should be contacted once before pruning"
    assert healthy.types() == ["status"], "the healthy surface must receive the first push"

    broker.broadcast({"type": "status", "stage": "second"})
    assert dead.calls == 1, (
        "the dead surface must have been pruned after its send failed — it should "
        f"not be contacted on the second broadcast (calls={dead.calls})"
    )
    assert healthy.types() == ["status", "status"], (
        f"the healthy surface must keep receiving after a peer was pruned; got {healthy.types()}"
    )


# ── D. User-message echo: every open surface sees the message that was sent ─────


def test_user_message_echo_reaches_all_surfaces_with_echo_id(broker: WebSocketBroker) -> None:
    """When one surface sends a message, the server echoes it on the shared
    channel via the real ``_broadcast_user_echo``. Every open surface must
    receive a ``user_message`` frame carrying the verbatim text AND the
    ``echo_id`` — the id is what lets the originating surface ignore its own
    echo (it already rendered the bubble) while peers render it.
    """
    phone = _Surface()
    laptop = _Surface()
    broker.connect(phone)
    broker.connect(laptop)

    from api.chat import _broadcast_user_echo

    _broadcast_user_echo("Move dinner to 8pm", "echo-abc-123")

    for surface, name in ((phone, "phone"), (laptop, "laptop")):
        echoes = [m for m in surface.received if m.get("type") == "user_message"]
        assert len(echoes) == 1, (
            f"{name} surface must receive exactly one user_message echo; got "
            f"{surface.types()}"
        )
        assert echoes[0]["content"] == "Move dinner to 8pm", (
            f"{name} surface must receive the verbatim message text (a user bubble "
            "renders escaped plain text, so it is NOT sanitized/altered)"
        )
        assert echoes[0]["echo_id"] == "echo-abc-123", (
            f"{name} surface must receive the echo_id so the sender can de-dup; "
            f"got {echoes[0]}"
        )


def test_post_chat_endpoint_broadcasts_user_echo_to_all_surfaces(
    authed_client: tuple[object, sqlite3.Connection, object], broker: WebSocketBroker
) -> None:
    """End-to-end via the REAL entry point: a ``POST /chat`` carrying ``echo_id``
    must echo the message to every open surface before the turn runs.

    Drives the production Flask route (``post_chat``) against a real DB, with two
    real surfaces connected to the real broker. The echo is broadcast
    synchronously inside ``post_chat`` (before the background turn is spawned),
    so both surfaces hold it by the time the 202 returns — proving the wiring
    from the HTTP form field through to the fan-out the multi-surface sync
    depends on.
    """
    from typing import cast as _cast
    from flask.testing import FlaskClient

    client = _cast(FlaskClient, authed_client[0])

    phone = _Surface()
    laptop = _Surface()
    broker.connect(phone)
    broker.connect(laptop)

    resp = client.post(
        "/chat",
        data={"text": "What's on my calendar today?", "echo_id": "echo-endpoint-1"},
    )
    assert resp.status_code == 202
    assert resp.get_json() == {"status": "accepted"}

    # Cancel the spawned turn so no background processor lingers for later tests.
    client.post("/chat/interrupt")

    for surface, name in ((phone, "phone"), (laptop, "laptop")):
        echoes = [m for m in surface.received if m.get("type") == "user_message"]
        assert len(echoes) == 1, (
            f"{name} surface must receive the user_message echo from the real "
            f"POST /chat; got {surface.types()}"
        )
        assert echoes[0]["content"] == "What's on my calendar today?"
        assert echoes[0]["echo_id"] == "echo-endpoint-1"
