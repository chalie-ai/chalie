# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""``GET /api/threads/all``'s ``working`` flag reads whether the thread has an
execution that has not ended — never whether an assistant row settled.

A turn stopped, crashed or cut off by a restart mid-call leaves an unsettled
assistant row behind; the feed listed such a thread as working forever. Each
test drives a real inert ``MessageProcessor`` through the real execution and
transcript services, ends the turn the way production does, and reads the feed
through the real endpoint. No mocks.
"""

import sqlite3
from collections.abc import Callable

import pytest
from flask.testing import FlaskClient

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

_TURN_ID = 7301
_AuthedClient = tuple[FlaskClient, sqlite3.Connection, object]


def _start(text: str) -> MessageProcessor:
    """A turn opened on the user channel with its input row written, the way
    a send lands one."""
    mp = MessageProcessor(UserConfig(), _TURN_ID, text)  # inert until it runs
    mp.uid = mp.current_transcript_id = mp.transcript_service.append_input(mp.raw_input)
    assert mp.turn_execution_service.open() is not None  # sanity: real open-row write
    return mp


def _feed_working(client: FlaskClient) -> bool:
    resp = client.get("/api/threads/all?type=user")
    assert resp.status_code == 200
    [thread] = [t for t in resp.get_json()["result"] if t["turn_id"] == _TURN_ID]
    return bool(thread["working"])


def _stop(client: FlaskClient, mp: MessageProcessor) -> None:
    resp = client.delete(f"/api/threads/{mp.turn_id}?type=user")
    assert resp.get_json()["result"]["cancelled"] is True  # sanity: a live turn was stopped


def _crash(_client: FlaskClient, mp: MessageProcessor) -> None:
    assert mp.turn_execution_service.finish(TurnExecution.CRASHED, "step raised") is not None


def _restart(_client: FlaskClient, mp: MessageProcessor) -> None:
    assert [r.turn_id for r in mp.turn_execution_service.sweep_orphaned()] == [mp.turn_id]


@pytest.mark.parametrize("end", [_stop, _crash, _restart], ids=["stopped", "crashed", "restart"])
def test_ended_turn_with_unsettled_assistant_row_is_not_working(
    authed_client: _AuthedClient, end: Callable[[FlaskClient, MessageProcessor], None],
) -> None:
    """A provider call wrote its assistant row, unsettled, and the turn then
    ended without settling: the user stopped it, a step raised, or a restart
    swept it. The thread is idle."""
    client, _db, _store = authed_client
    mp = _start("Summarise my inbox.")
    mp.transcript_service.append_assistant("Reading your inbox", settled=False)

    end(client, mp)

    assert _feed_working(client) is False


@pytest.mark.parametrize("replying", [False, True], ids=["first exchange", "reply into a settled thread"])
def test_running_turn_is_working(authed_client: _AuthedClient, replying: bool) -> None:
    """A turn mid-call, its assistant row written and unsettled, is working —
    also when it is a reply into a thread whose first exchange already settled
    and finished."""
    client, _db, _store = authed_client
    if replying:
        opener = _start("What's on my plate today?")
        opener.transcript_service.append_assistant("Nothing scheduled.", settled=True)
        assert opener.turn_execution_service.finish(TurnExecution.COMPLETED) is not None
    mp = _start("And tomorrow?")
    mp.transcript_service.append_assistant("Checking your calendar", settled=False)

    assert _feed_working(client) is True


def test_finished_turn_is_not_working(authed_client: _AuthedClient) -> None:
    """A turn that answered and closed its execution is idle."""
    client, _db, _store = authed_client
    mp = _start("What's the weather?")
    mp.transcript_service.append_assistant("Sunny, 24°C.", settled=True)
    assert mp.turn_execution_service.finish(TurnExecution.COMPLETED) is not None

    assert _feed_working(client) is False
