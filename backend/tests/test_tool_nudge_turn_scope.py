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
  chip the person sees.

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
from configs.channels.user import UserConfig
from configs.enums.ability_category import AbilityCategory
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.policy import Policy
from models.provider_response import ProviderResponse
from models.thread_gist import ThreadGist
from models.turn_execution import TurnExecution
from services.provider_api import ProviderApiRequest
from services.turn_serializer_service import get_service as _serializer
from tests.helpers import drain_background_turns

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
    ThreadGist(channel=mp_1.channel, turn_id=mp_1.turn_id, gist="nudge probe").upsert()
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
