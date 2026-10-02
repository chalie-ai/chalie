# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: every completed provider call stores its own assistant row.

A turn is a chain of provider calls. Each completed call is one assistant row:
a call that only dispatched tools stores an EMPTY row before those tools run
(so its tool calls and its thinking trace anchor to it), and only the terminal
call — the one with no tool calls — is the settled row. Readers that show or
replay the conversation must treat a blank step row as plumbing, not as an
empty assistant line, while the live surface still learns about every row.

Same harness as ``test_thinking_trace_spine.py``: the real production entry
point (inert construct, ``begin()``, ``result()``) against the real DB and
services; the only substitution is the LLM network boundary via
``services.provider_service.build_client``. Frames are captured by a real
subscriber on the real socket registry.
"""

import json
import sqlite3
import threading
import time
from typing import cast
from unittest.mock import patch

import pytest

from configs.channels.memory_step import HISTORY_LIMIT
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from models.transcript_thinking import TranscriptThinking
from models.turn_execution import TurnExecution
from services.memory_step_service import memory_step_config
from services.turn_serializer_service import get_service as _serializer
from services.websocket import Websocket

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.build_client"


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()``, in order. Past
    the end of the script it returns a non-empty terminal, so the post-turn
    daemon turns a completed ``user`` turn spawns still finish."""

    _TERMINAL = ProviderResponse(text="done", model="scripted-overflow", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0

    def get_context_limit(self) -> int:
        return 200000

    def send(self, _dto: object) -> ProviderResponse:
        if self.sends >= len(self._responses):
            return self._TERMINAL
        response = self._responses[self.sends]
        self.sends += 1
        return response


class _RecordingClient:
    """A real socket-registry subscriber implementing the registry's
    ``send(str)`` Protocol, capturing every broadcast frame."""

    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def send(self, data: str) -> None:
        self.frames.append(json.loads(data))


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed ``user`` turn
    spawns, so they run to completion inside THIS test's provider+DB patch."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [
            t for t in threading.enumerate()
            if t.name in ("skill-suggest", "thread-gist") or t.name.startswith("turn-")
        ]
        if not pending:
            return
        for t in pending:
            t.join(timeout=deadline - time.monotonic())


def _run(provider: _ScriptedProvider, raw_input: str) -> MessageProcessor:
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return mp


def _assistant_rows(mp: MessageProcessor) -> list[dict[str, object]]:
    return [r for r in Transcript.by_turn(mp.channel, mp.turn_id) if r["role"] == "assistant"]


def _tool_step(text: str, probe: str, thinking: str | None = None) -> ProviderResponse:
    """A provider call that dispatches one (unknown-ability, so inert) tool call."""
    return ProviderResponse(
        text=text, model="scripted", thinking_block=thinking,
        tool_calls=[{"name": "noop_probe", "input": {"q": probe}}],
    )


def _final(text: str, thinking: str | None = None) -> ProviderResponse:
    return ProviderResponse(text=text, model="scripted", thinking_block=thinking, tool_calls=None)


def test_each_provider_call_of_a_turn_stores_its_own_row_with_its_own_work(
    db: sqlite3.Connection,
) -> None:
    """A three-call turn — a tool-only call, a call with prose AND a tool, then
    the final answer — leaves three assistant rows, only the last settled. The
    work of each call (its tool call, its thinking trace) sits on that call's
    own row, and the turn's settle point is the final row: an interim row is
    never the one the turn settles on."""
    assert db is not None
    provider = _ScriptedProvider(
        _tool_step("", "first", thinking="reasoning one"),
        _tool_step("Checking the second thing.", "second", thinking="reasoning two"),
        _final("All done.", thinking="reasoning three"),
    )

    mp = _run(provider, "look into both things")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    rows = _assistant_rows(mp)
    assert [r["content"] for r in rows] == ["", "Checking the second thing.", "All done."]
    assert [r["settled"] for r in rows] == [0, 0, 1]
    row_ids = [cast("int", r["id"]) for r in rows]
    assert Transcript.settle0(mp.channel, mp.turn_id) == row_ids[2]
    # Each call's tool call is anchored to the row of the call that made it.
    probes = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "noop_probe"]
    assert [c.transcript_id for c in probes] == row_ids[:2]
    # Each call's reasoning is anchored to that same call's row.
    traces = TranscriptThinking.by_turn(mp.channel, mp.turn_id)
    assert [(t.transcript_id, t.thinking_trace) for t in traces] == [
        (row_ids[0], "reasoning one"),
        (row_ids[1], "reasoning two"),
        (row_ids[2], "reasoning three"),
    ]


def test_the_live_surface_is_told_about_every_row_a_turn_stores(
    db: sqlite3.Connection,
) -> None:
    """The open conversation view refetches the turn when it is told the turn
    was updated; a row stored without that frame would sit in the database
    unseen until the next unrelated frame. One ``updated`` frame per stored
    assistant row — three for the three-call turn, none lost on the empty
    tool-only step."""
    assert db is not None
    provider = _ScriptedProvider(
        _tool_step("", "first"),
        _tool_step("Checking the second thing.", "second"),
        _final("All done."),
    )
    client = _RecordingClient()
    Websocket._connect(client)
    try:
        mp = _run(provider, "look into both things")
    finally:
        Websocket._disconnect(client)

    rows = _assistant_rows(mp)
    assert len(rows) == 3, "the scripted turn must have stored one row per call"
    updated = [
        f for f in client.frames
        if f.get("status") == "updated" and f.get("turn_id") == mp.turn_id
    ]
    assert len(updated) == len(rows)


def test_replaying_history_skips_the_blank_step_row(db: sqlite3.Connection) -> None:
    """The model reads the previous turn back as conversation. A tool-only step
    stored an empty assistant row; replaying it would show the model a blank
    ``Assistant:`` line it never wrote. The stored turn keeps the blank step
    row, but the next turn's history view holds only the rows that SAID
    something — the question and the answer — one rendered line each."""
    assert db is not None
    first = _run(
        _ScriptedProvider(_tool_step("", "probe"), _final("The sky is blue.")),
        "what colour is the sky",
    )
    assert [r["content"] for r in _assistant_rows(first)] == ["", "The sky is blue."], (
        "precondition: the stored turn holds the blank step row"
    )
    second = MessageProcessor(UserConfig(), raw_input="and the sea?")  # inert (I2)

    history = second.transcript_service.read()

    assert [(r.role, r.content) for r in history] == [
        ("user", "what colour is the sky"),
        ("assistant", "The sky is blue."),
    ]
    rendered = second.prompt_service.previous_messages()
    assert len(rendered.splitlines()) == len(history)
    assert "The sky is blue." in rendered


def test_a_capped_history_is_not_crowded_out_by_blank_step_rows(
    db: sqlite3.Connection,
) -> None:
    """A capped history view keeps the newest N rows. A turn that ran as many
    tool-only steps as the cap stores that many blank rows between the question
    and the answer; counted against the cap they would push the question out,
    leaving a reader that sees an answer with no question. The cap counts only
    rows that said something, so the question survives."""
    assert db is not None
    first = _run(
        _ScriptedProvider(
            *[_tool_step("", f"probe {n}") for n in range(HISTORY_LIMIT)],
            _final("Both are blue."),
        ),
        "what colour are the sky and the sea",
    )
    assert len(_assistant_rows(first)) == HISTORY_LIMIT + 1, (
        "precondition: one blank row per tool-only step, then the answer"
    )
    reader = MessageProcessor(memory_step_config(UserConfig(), []), raw_input="")  # inert (I2)

    history = reader.transcript_service.read()

    assert [(r.role, r.content) for r in history] == [
        ("user", "what colour are the sky and the sea"),
        ("assistant", "Both are blue."),
    ]


def test_the_rendered_thread_shows_the_step_row_with_its_tool_chips(
    db: sqlite3.Connection,
) -> None:
    """The thread view hangs a step's tool pills off the step's own row, so the
    serializer must emit the blank step row as a message of its own — dropping
    it as 'empty' would orphan the pills. The row is a blank, unsettled
    assistant message carrying the call's chip; the user's message does not
    carry that chip; the final answer is the settled message."""
    assert db is not None
    mp = _run(
        _ScriptedProvider(_tool_step("", "probe"), _final("The sky is blue.")),
        "what colour is the sky",
    )

    result = _serializer().serialize(mp.channel, mp.turn_id)

    messages = cast("list[dict[str, object]]", result["messages"])
    rows = _assistant_rows(mp)
    assert [m["id"] for m in messages] == [str(mp.uid), str(rows[0]["id"]), str(rows[1]["id"])]
    user_msg, step_msg, answer_msg = messages
    assert step_msg["role"] == "assistant"
    assert step_msg["content"] == ""
    assert step_msg["settled"] is False
    step_chips = cast("list[dict[str, object]]", step_msg["tool_calls"])
    assert [c["tool_name"] for c in step_chips] == ["noop_probe"]
    assert "noop_probe" not in [
        c["tool_name"] for c in cast("list[dict[str, object]]", user_msg.get("tool_calls", []))
    ]
    assert answer_msg["content"] == "The sky is blue."
    assert answer_msg["settled"] is True
