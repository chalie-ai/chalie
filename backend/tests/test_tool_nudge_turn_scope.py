# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: a tool's follow-up nudge lives for the turn it was made in.

A tool's next-step nudge (its follow-up block) is shown to the model, never
stored:

* it shows on every later step of the turn the call was made in;
* it is gone on the next turn, which rebuilds its context from the transcript;
* the stored ``tool_calls`` row holds the bare result, and so does the tool
  chip the person sees;
* a hand-over written when the turn runs out of room reads the result as the
  model was shown it, nudge included, and no stored row keeps the nudge.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) against the real SQLite DB, with the real ``DispatchService``,
``ToolCallService`` and ``PromptService``. The only substitution is the LLM
network boundary: ``services.provider_service.build_client``. The fake client
replays a scripted sequence of provider responses, one per step, and records
each step prompt. ``nudge_probe`` is a real, DISCOVERABLE ``Ability`` granted
through a CHAT-channel ``allow`` policy row, the production grant path."""

import sqlite3
from typing import cast
from unittest.mock import patch

import pytest

from abilities._ability import FOLLOW_UP_BLOCK, Ability
from abilities._result import ToolResult
from abilities._turn_handover_config import TurnHandoverConfig
from abilities.chat_history_compactor import ChatHistoryCompactionConfig
from configs.channels.user import UserConfig
from configs.enums.ability_category import AbilityCategory
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from exceptions import ContextLimit
from models.policy import Policy
from models.provider_response import ProviderResponse
from models.thread import Thread
from models.turn_execution import TurnExecution
from services.provider_api import ProviderApiRequest
from services.turn_serializer_service import get_service as _serializer
from tests.helpers import NO_PLAN, drain_background_turns, is_planning_request

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.build_client"

_NUDGE = "nudge probe: read the second page before answering"
_BLOCK = FOLLOW_UP_BLOCK.format(text=_NUDGE)


class nudge_probe(Ability):
    """A real, DISCOVERABLE ability whose success carries a follow-up nudge.

    No custom ``__init__``: ``DispatchService._bind()`` builds a fresh instance
    via ``type(template)(mp=...)`` for every dispatch."""

    NAME = "nudge_probe"
    # The registry rejects a discoverable ability with no find_tools heading.
    CATEGORY = AbilityCategory.SYSTEM

    def get_search_tooltip(self) -> str:
        return "follow-up nudge test fixture"

    def get_summary(self) -> str:
        return "Follow-up nudge test fixture — returns a fixed success with a nudge."

    def get_examples(self) -> list[str]:
        return [
            "probe the nudge",
            "test the follow-up nudge",
            "fire the nudge probe",
            "run the nudge probe",
            "hit the nudge probe",
            "trigger the nudge probe",
            "use the nudge probe",
        ]

    def get_parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": [],
        }

    def run(self, params: dict[str, object]) -> ToolResult:
        return ToolResult.ok("nudge probe done")

    def get_follow_up(self, tr: ToolResult) -> str:
        return _NUDGE


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order,
    and records each request's step prompt in ``bodies``.

    Past the end of the script it returns a benign terminal (no-tool) response:
    post-turn daemon turns make provider calls this test does not script for.
    The assertions read the scripted steps' prompts, which always come first."""

    _TERMINAL = ProviderResponse(text="done", model="scripted-terminal", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0
        self.bodies: list[str] = []

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: ProviderApiRequest) -> ProviderResponse:
        if is_planning_request(dto):
            return NO_PLAN
        self.bodies.append(str(dto.messages[0]["content"]))
        if self.sends >= len(self._responses):
            return self._TERMINAL
        response = self._responses[self.sends]
        self.sends += 1
        return response


def _step(name: str, **params: object) -> ProviderResponse:
    """A step that makes one tool call."""
    return ProviderResponse(text="", model="scripted", tool_calls=[{"name": name, "input": params}])


def _answer(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, model="scripted", tool_calls=None)


def _run(provider: _ScriptedProvider, raw_input: str, *, turn_id: int = -1) -> MessageProcessor:
    """Drive a real user turn to termination against *provider*, then quiesce any
    post-turn daemon turns inside the patch so the test stays hermetic."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input, turn_id=turn_id)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        drain_background_turns()
    return mp


def _nudged_turn() -> tuple[MessageProcessor, _ScriptedProvider]:
    """A completed three-step turn: step 1 calls ``nudge_probe``, step 2 calls an
    unregistered tool (no nudge of its own), step 3 answers."""
    Policy.upsert(PolicyChannel.CHAT.value, nudge_probe.NAME, "allow")
    provider = _ScriptedProvider(
        _step(nudge_probe.NAME, q="first"),
        _step("noop_probe", q="second"),
        _answer("All done."),
    )
    mp = _run(provider, "probe the nudge")
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    return mp, provider


def test_nudge_shows_on_every_later_step_of_the_turn(db: sqlite3.Connection) -> None:
    """The nudge is absent before the call and shows, once, on each of the two
    steps after it — including the step after an unrelated call."""
    assert db is not None
    _, provider = _nudged_turn()

    # bodies[k] is step k+1's prompt.
    assert _NUDGE not in provider.bodies[0]
    assert provider.bodies[1].count(_BLOCK) == 1
    assert provider.bodies[2].count(_BLOCK) == 1


def test_nudge_is_gone_on_the_next_turn(db: sqlite3.Connection) -> None:
    """A reply in the same thread — a new turn that shares the thread's
    ``turn_id`` — rebuilds its context from the transcript: its prompt carries
    no trace of the earlier turn's nudge. The skill-suggestion pass that follows
    it reads the thread's stored rows, so it sees the probe's bare result and
    no nudge either."""
    assert db is not None
    mp_1, provider_1 = _nudged_turn()
    assert provider_1.bodies[1].count(_BLOCK) == 1  # the nudge did show in turn 1

    # A thread that already carries its label launches no gist delegate, whose
    # provider call would race the reply's for the scripted answer.
    assert Thread.set_summary(mp_1.channel, mp_1.turn_id, "nudge probe")
    provider_2 = _ScriptedProvider(_answer("Second answer."))
    _run(provider_2, "and now?", turn_id=mp_1.turn_id)

    # bodies[0] is the reply's only step; the thread's stored calls reach the
    # skill-suggestion threshold, so its pass is the next request.
    assert _NUDGE not in provider_2.bodies[0]
    skill_pass = [b for b in provider_2.bodies if "## Completed ACT Loop Trail" in b]
    assert len(skill_pass) == 1
    assert "nudge probe done" in skill_pass[0]
    assert all(_NUDGE not in body for body in provider_2.bodies)


def test_stored_row_and_tool_chip_hold_the_bare_result(db: sqlite3.Connection) -> None:
    """The ``tool_calls`` row keeps the bare result: the success envelope with
    no follow-up block. The tool chip the person sees serves that same text."""
    assert db is not None
    mp, _ = _nudged_turn()

    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == nudge_probe.NAME]
    assert len(rows) == 1
    row = rows[0]
    assert "nudge probe done" in row.result
    assert _NUDGE not in row.result
    assert "[follow_up_instruction]" not in row.result

    chip = _serializer().tool_call(cast(int, row.id))
    assert chip is not None
    assert chip["result"] == row.result


_HANDOVER_MARK = TurnHandoverConfig().system_prompt.splitlines()[0]
_COMPACTION_MARK = ChatHistoryCompactionConfig().system_prompt.splitlines()[0]


class _RunsOutOfRoomProvider:
    """Serves the hand-over and the compactor by their system prompts and
    records each hand-over body. The main turn calls ``nudge_probe``, is then
    refused for length once, and answers on the retry; anything after that
    gets the same answer."""

    def __init__(self) -> None:
        self.main_sends = 0
        self.handover_bodies: list[str] = []

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: ProviderApiRequest) -> ProviderResponse:
        if is_planning_request(dto):
            return NO_PLAN
        system = dto.system
        if _HANDOVER_MARK in system:
            self.handover_bodies.append("\n".join(str(m.get("content") or "") for m in dto.messages))
            return _answer("I was mid-task.")
        if _COMPACTION_MARK in system:
            return _answer("## Now\nthe compacted checkpoint")
        self.main_sends += 1
        if self.main_sends == 1:
            return _step(nudge_probe.NAME, q="first")
        if self.main_sends == 2:
            raise ContextLimit("payload exceeds the model's maximum context length")
        return _answer("All done.")


def _stored_cells_containing(db: sqlite3.Connection, text: str) -> list[str]:
    """Every ``table.column`` of an ordinary table whose stored value contains
    *text*. Full-text index tables are skipped: their shadow tables hold the
    same text and are scanned."""
    tables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL%'"
    )]
    hits: list[str] = []
    for table in tables:
        cursor = db.execute(f'SELECT * FROM "{table}"')
        columns = [c[0] for c in cursor.description]
        for row in cursor:
            for column, value in zip(columns, row):
                if isinstance(value, bytes):
                    value = value.decode("utf-8", "replace")
                if isinstance(value, str) and text in value:
                    hits.append(f"{table}.{column}")
    return hits


def test_handover_reads_the_nudge_and_no_row_stores_it(db: sqlite3.Connection) -> None:
    """When the turn runs out of room after the probe, the hand-over the model
    writes itself is built from the result as the model was shown it — the
    nudge included — and the nudge reaches no stored row."""
    Policy.upsert(PolicyChannel.CHAT.value, nudge_probe.NAME, "allow")
    provider = _RunsOutOfRoomProvider()
    mp = MessageProcessor(UserConfig(), raw_input="probe the nudge")  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        drain_background_turns()

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    assert len(provider.handover_bodies) == 1
    assert _BLOCK in provider.handover_bodies[0]
    assert _stored_cells_containing(db, _NUDGE) == []
