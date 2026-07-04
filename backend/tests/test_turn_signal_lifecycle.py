# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the turn-signal lifecycle on the WebSocket bus.

A user turn narrates itself to every open surface through ONE chokepoint,
``MessageProcessor.broadcast``, gated by a single class flag
``ProcessorConfig.BROADCASTS_STATE`` (only the user channel opts in). The surface
holds no turn data: each signal carries a turn_id (plus, for a live tool call, the
``tool_calls`` row id/name/summary) and the surface pulls every byte back over
REST. These tests drive the REAL production path with zero mocks:

  * ``MessageProcessor.__init__`` also opens a ``turn_executions`` row via
    ``ExecutionTracker``, which broadcasts its own lifecycle frame (a ``state``
    key, never a ``status`` key — see ``execution_tracker._broadcast_lifecycle``).
    That frame is a real, intentional side effect of construction; it is the
    OTHER file's contract (``test_turn_executions.py``), so ``_Surface.types()``
    here filters the wire down to ``status``-bearing frames only, exactly as a
    real surface distinguishes the two frame kinds by key presence.

  * the real process-wide ``WebSocketBroker`` and real consumer surfaces at the
    ``send(raw)`` WS boundary (the exact contract a browser socket fulfils),
  * the real ``ToolDispatcher`` dispatch chokepoint running a real, offline,
    always-available tool (``memory`` recall — local embeddings + SQLite),
  * the real ``MessageProcessor`` persistence steps (``_store_row``/``_end_turn``)
    and the real ``UserConfig`` / ``WebSearchConfig`` against the real test db.

The contract they lock: the live ``tool_called`` id IS the persisted
``tool_calls`` row id IS the id ``tool_done`` resolves — so the surface's live
act-trail pill, its running timer, and the post-step refetched block all key on
one identity, and a non-broadcasting channel emits NOTHING.
"""

import json
import sqlite3
from collections.abc import Generator
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels.user import UserConfig
from configs.channels.web_search import WebSearchConfig
from services.act_trail import ActTrail
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.websocket_broker import WebSocketBroker

pytestmark = pytest.mark.unit

_CHAT = ProcessorConfig.PolicyChannel.CHAT


class _Surface:
    """A real consumer at the WS boundary — one open device/tab. The broker pushes
    JSON strings to it exactly as it does to a browser connection."""

    def __init__(self) -> None:
        self.received: list[dict[str, object]] = []

    def send(self, raw: str) -> None:
        self.received.append(json.loads(raw))

    def types(self) -> list[str]:
        """Only the ``status``-keyed broadcast() signals — a turn_executions
        lifecycle frame carries ``state``, never ``status``, so it is filtered
        out here rather than polluting this file's assertions with a frame
        kind that belongs to ExecutionTracker's own contract."""
        return [cast(str, m["status"]) for m in self.received if "status" in m]


@pytest.fixture
def broker() -> Generator[WebSocketBroker, None, None]:
    """The real process-wide broker singleton, emptied before and after so leaked
    connections from other tests cannot perturb the frame assertions."""
    b = WebSocketBroker()
    with b._lock:
        b._connections.clear()
    try:
        yield b
    finally:
        with b._lock:
            b._connections.clear()


def _user_mp(text: str) -> MessageProcessor:
    """A real user-channel MP placed in its post-input-row production state: the
    real ``MessageProcessor.__init__`` pre-allocates the turn and anchoring input
    row exactly as production does (``make_row_id`` / ``self.ts``), so the
    broadcast chokepoint and dispatch run for real. ``_setup`` itself (thinking
    gate, turn-zero seed — network / unrelated signals) is deliberately not
    driven here."""
    return MessageProcessor(UserConfig(), raw_input=text)


def test_live_tool_call_signals_are_keyed_to_the_persisted_row(
    db: sqlite3.Connection, broker: WebSocketBroker,
) -> None:
    """A tool running mid-ACT emits ``tool_called`` then ``tool_done``, and the id
    they carry IS the ``tool_calls`` row the dispatcher persisted — the single
    identity the surface's live pill + timer and its later refetch all key on.
    This is the real-time info the pre-redesign code dropped."""
    surface = _Surface()
    broker.connect(surface)
    mp = _user_mp("what is the population of Malta")

    ToolDispatcher(mp).dispatch("memory", {"action": "recall", "query": "population of Malta"})

    assert surface.types() == ["tool_called", "tool_done"], (
        f"a live tool call must bracket itself with tool_called→tool_done; got {surface.types()}"
    )
    called, done = (f for f in surface.received if "status" in f)
    assert called["turn_id"] == mp.turn_id
    assert called["name"] == "memory"
    assert called["transcript_row_id"] == mp.anchor
    call_id = called["id"]
    assert isinstance(call_id, int), f"tool_called must name the opened row id; got {call_id!r}"
    assert done["id"] == call_id, "tool_done must resolve the SAME pill tool_called opened"

    # Cross-step contract: the signalled id is exactly the persisted tool_calls row
    # (anchored to mp.anchor) whose result finish() wrote — the FE refetch lands the
    # same call the live pill represented, so nothing double-renders.
    rows = ActTrail().fetch_by_transcript_id(cast(int, mp.anchor))
    assert [r["id"] for r in rows] == [call_id]
    assert rows[0]["tool_name"] == "memory"
    assert rows[0]["result"], "finish() must have written the tool result onto the opened row"


def test_state_signals_fire_from_the_real_persistence_steps(
    db: sqlite3.Connection, broker: WebSocketBroker,
) -> None:
    """Persisting a step row emits ``updated`` (refetch the block) through the
    broadcast() chokepoint — driven through the real ``_store_row``, not the
    broadcast method in isolation. Ending the turn no longer emits a ``status``
    signal of its own: ``_end_turn`` stamps the execution row done via
    ``ExecutionTracker.set_done()``, which broadcasts the turn's lifecycle frame
    (``state=completed``) instead — that frame's own contract is pinned in
    ``test_turn_executions.py``, not here."""
    surface = _Surface()
    broker.connect(surface)
    mp = _user_mp("tell me a fact")

    mp._store_row("Here is your answer.")
    mp._end_turn("Here is your answer.")

    assert surface.types() == ["updated"], (
        f"persistence must emit exactly one updated status signal; got {surface.types()}"
    )
    updated = next(f for f in surface.received if "status" in f)
    assert updated["turn_id"] == mp.turn_id

    # The turn's completion still reaches the wire — as the OTHER frame kind.
    lifecycle_frames = [f for f in surface.received if "state" in f]
    assert [f["state"] for f in lifecycle_frames] == ["working", "completed"], (
        f"construction opens the row (working) and _end_turn's set_done() closes it (completed); "
        f"got {surface.received}"
    )


def test_a_non_broadcasting_channel_emits_nothing(
    db: sqlite3.Connection, broker: WebSocketBroker,
) -> None:
    """``BROADCASTS_STATE`` is the single gate: a delegate channel (the flag off)
    runs the identical setup + dispatch through the identical chokepoint and the
    surface receives NOTHING — no state frames, no tool frames. Proves the lean
    bus stays silent for every channel but the user's."""
    surface = _Surface()
    broker.connect(surface)

    mp = MessageProcessor(WebSearchConfig(_CHAT), raw_input="current population of Malta")
    mp._setup()  # real delegate setup: writes its own row, gates — must stay silent

    ToolDispatcher(mp).dispatch("memory", {"action": "recall", "query": "population of Malta"})

    assert surface.received == [], (
        f"a non-broadcasting channel must emit nothing; got {surface.types()}"
    )
