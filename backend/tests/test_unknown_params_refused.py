# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: a tool call that names a parameter the tool does not declare is
refused whole, and nothing runs.

Before this, an undeclared key passed through dispatch and every ability
ignored it, so the call ran without it and reported success: a ``schedule``
create sent ``cron="0 4 * * *"`` lost the cron, defaulted every field to ``*``,
and ran every minute. The refusal sits in dispatch, after the key healer, so an
alias the healer rewrites to its declared name still passes.

Every case drives the REAL production entry point end to end, like
``test_edit_file_empty_replace.py``: a real ``MessageProcessor`` turn against
the real SQLite DB (``db`` fixture), the real ``DispatchService``, and the real
abilities. The only substitution is the LLM network boundary
(``services.provider_service.build_client``), scripted to replay one tool call
per step. ``schedule.create``, ``user_device`` and ``home.list_devices`` carry
seeded ``allow`` rows on the ``chat`` channel; ``read`` bypasses the gate; the
calendar case allows ``calendar.create_event`` first, as the person would.
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from abilities.calendar import CalendarAbility
from configs.channels.user import UserConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.policy import Policy
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from tests.helpers import NO_PLAN, drain_background_turns, is_planning_request

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.build_client"


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


def _step(*calls: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(text="", model="scripted", tool_calls=list(calls))


_DONE = ProviderResponse(text="Done.", model="scripted", tool_calls=None)


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order;
    past the end of the script it returns a benign terminal response (a
    completed user turn spawns a post-turn suggestion call it does not script)."""

    _TERMINAL = ProviderResponse(text="", model="scripted-unknown-params", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        if is_planning_request(dto):
            return NO_PLAN
        if self.sends >= len(self._responses):
            return self._TERMINAL
        response = self._responses[self.sends]
        self.sends += 1
        return response


def _run(provider: _ScriptedProvider, raw_input: str) -> MessageProcessor:
    """Drive a real user turn to termination against *provider*, then quiesce any
    post-turn daemon turns inside the patch so the test stays hermetic."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        drain_background_turns()
    return mp


def _calls(mp: MessageProcessor, tool_name: str) -> list[ToolCall]:
    return [c for c in mp.tool_call_service.by_turn() if c.tool_name == tool_name]


def _schedules(db: sqlite3.Connection) -> list[dict[str, object]]:
    rows = db.execute(
        "SELECT message, cron_minute, cron_hour, cron_dom, cron_month, cron_dow"
        " FROM scheduled_items ORDER BY id"
    ).fetchall()
    return [dict(row) for row in rows]


# ── Case 1: an undeclared key refuses the whole call ──────────────────────────


def test_schedule_sent_cron_is_refused_and_creates_nothing(db: sqlite3.Connection) -> None:
    """The exact failure from the field: ``cron`` is not a schedule parameter.
    The call is refused with the approved wording and the declared list, no
    schedule row is written, and the model's corrected call (the cron spelled as
    the declared fields) then creates exactly the schedule that was asked for."""
    provider = _ScriptedProvider(
        _step(_tool("schedule", action="create", message="Water the plants", cron="0 4 * * *")),
        _step(_tool("schedule", action="create", message="Water the plants", minute="0", hour="4")),
        _DONE,
    )

    mp = _run(provider, "remind me to water the plants every day at 4am")

    refused, created = _calls(mp, "schedule")
    assert refused.state == ToolCall.ERROR
    assert refused.result == (
        "[schedule(status=error, code=unknown-params)]\n"
        "Unknown parameter(s): cron. Nothing was done.\n"
        "valid: action | message | start_at | minute | hour | day | month | weekday"
        " | item_id | query | limit\n"
        "[end:schedule]"
    )

    assert created.state == ToolCall.DONE, created.result
    # One row — the corrected call's. The refused call wrote nothing, so there
    # is no every-minute row next to it.
    assert _schedules(db) == [{
        "message": "Water the plants", "cron_minute": "0", "cron_hour": "4",
        "cron_dom": "*", "cron_month": "*", "cron_dow": "*",
    }]


# ── Case 2: a healed alias is not unknown ─────────────────────────────────────


def test_alias_healed_to_a_declared_name_still_runs(db: sqlite3.Connection, tmp_path: Path) -> None:
    """``url`` is not a ``read`` parameter, but the healer rewrites it to
    ``source`` before the refusal looks, so the read runs and returns the file."""
    assert db is not None
    target = tmp_path / "note.txt"
    target.write_text("the alias reached the file\n", encoding="utf-8")

    provider = _ScriptedProvider(_step(_tool("read", url=str(target))), _DONE)

    mp = _run(provider, "read my note")

    (read,) = _calls(mp, "read")
    assert read.state == ToolCall.DONE, read.result
    assert "the alias reached the file" in read.result
    assert "unknown-params" not in read.result


# ── Case 3: a tool that declares no parameters refuses any key ────────────────


def test_tool_with_no_parameters_refuses_any_key(db: sqlite3.Connection) -> None:
    """``user_device`` declares no parameters, so any key it is sent is unknown;
    the refusal carries no ``valid:`` line because there is nothing to list."""
    assert db is not None
    provider = _ScriptedProvider(_step(_tool("user_device", detail="full")), _DONE)

    mp = _run(provider, "what device am I on")

    (call,) = _calls(mp, "user_device")
    assert call.state == ToolCall.ERROR
    assert call.result == (
        "[user_device(status=error, code=unknown-params)]\n"
        "Unknown parameter(s): detail. Nothing was done.\n"
        "[end:user_device]"
    )


# ── Case 4: parameters the tools always read are now declared ─────────────────


def test_newly_declared_calendar_and_home_params_reach_the_tool(db: sqlite3.Connection) -> None:
    """calendar's location/description/calendar_name and home's limit were read
    by the tools but never declared; they are declared now, so a call using them
    passes the refusal and the pre-gate and reaches the tool, which answers that
    the service is not connected (none is, in the test database)."""
    Policy.upsert(PolicyChannel.CHAT.value, f"{CalendarAbility.NAME}.create_event", "allow")
    provider = _ScriptedProvider(
        _step(_tool(
            "calendar", action="create_event", summary="Dentist",
            dtstart="2026-12-01T10:00:00Z", dtend="2026-12-01T11:00:00Z",
            location="Main St", description="Check-up", calendar_name="Personal",
        )),
        _step(_tool("home", action="list_devices", limit=5)),
        _DONE,
    )

    mp = _run(provider, "book the dentist and list my devices")

    (calendar,) = _calls(mp, "calendar")
    (home,) = _calls(mp, "home")
    for call in (calendar, home):
        first_line = call.result.split("\n", 1)[0]
        assert "code=not-connected" in first_line, call.result
