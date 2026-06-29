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
from services.transcript_service import Transcript
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
        return [cast(str, m.get("type")) for m in self.received]


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
    real ``Transcript.write_input_row`` allocates the turn and anchor exactly as
    ``_setup`` does, so the broadcast chokepoint and dispatch run for real. The
    thinking gate and turn-zero seed (network / unrelated signals) are not the
    subject here and are not driven."""
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, text, {})
    mp.config = UserConfig()
    mp.uid = Transcript.write_input_row("user", "user", text, turn_id=None)
    mp.turn_id = Transcript.turn_id_of_row(mp.uid)
    mp.anchor = mp.uid
    return mp


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
    called, done = surface.received
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
    """Persisting a step row emits ``updated`` (refetch the block); ending the turn
    emits ``done`` (drop the working indicator). Both ride the same chokepoint and
    carry the turn_id the surface coalesces on — driven through the real
    ``_store_row`` / ``_end_turn``, not the broadcast method in isolation."""
    surface = _Surface()
    broker.connect(surface)
    mp = _user_mp("tell me a fact")

    mp._store_row("Here is your answer.")
    mp._end_turn("Here is your answer.")

    assert surface.types() == ["updated", "done"], (
        f"persistence must emit updated then the turn end must emit done; got {surface.types()}"
    )
    updated, done = surface.received
    assert updated["turn_id"] == mp.turn_id
    assert done["turn_id"] == mp.turn_id


def test_a_non_broadcasting_channel_emits_nothing(
    db: sqlite3.Connection, broker: WebSocketBroker,
) -> None:
    """``BROADCASTS_STATE`` is the single gate: a delegate channel (the flag off)
    runs the identical setup + dispatch through the identical chokepoint and the
    surface receives NOTHING — no state frames, no tool frames. Proves the lean
    bus stays silent for every channel but the user's."""
    surface = _Surface()
    broker.connect(surface)

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "current population of Malta", {})
    mp.config = WebSearchConfig(_CHAT)
    mp._setup()  # real delegate setup: writes its own row, gates — must stay silent

    ToolDispatcher(mp).dispatch("memory", {"action": "recall", "query": "population of Malta"})

    assert surface.received == [], (
        f"a non-broadcasting channel must emit nothing; got {surface.types()}"
    )
