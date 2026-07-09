# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the cancelled-orphan exclusion added in the undo/cancel
rework: a cancel observed at ``MessageProcessor._step``'s checkpoint discards
the in-flight response before any reply row is stored, leaving a trailing,
reply-less user row behind. Two independent read paths must both hide it:

* ``GET /api/thread/<turn_id>`` (``api.threads._drop_trailing_cancelled_orphan``)
  — the single-turn expand/refetch read.
* ``Transcript.recent_threads`` / ``Transcript.count_turns``
  (``Transcript._thread_query_filter``'s ``_CANCELLED_ORPHAN_HAVING``) — the
  collapsed thread feed.

Both must still show any REAL content a turn already wrote before the cancel
landed — only a turn whose entire content is the cancelled orphan disappears.

Drives the real ``TranscriptService``/``TurnExecutionService`` through real,
inertly-constructed ``MessageProcessor`` instances (mirrors the fork pattern in
``test_threads_serialize_turn.py``), the real Flask app for the REST read/cancel
(``authed_client``), and the real ``Transcript`` model classmethods for the
feed — all against the real, fully-migrated SQLite database. No mocks.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.transcript import Transcript
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

_CHANNEL = "user"


def _open_exchange(turn_id: int, question: str, answer: str) -> tuple[int, int]:
    """A real, settled user+assistant exchange written via the production
    services — the "already-answered" half of a turn a later reply forks
    into (mirrors the opener half of the fork pattern in
    ``test_threads_serialize_turn.py``). Returns (user_row_id, answer_row_id)."""
    mp = MessageProcessor(UserConfig(), turn_id, question)  # inert (I2)
    user_id = mp.transcript_service.append_input(mp.raw_input)
    mp.uid = user_id
    mp.current_transcript_id = user_id
    answer_id = mp.transcript_service.append_assistant(answer)
    return user_id, answer_id


# ── (a) GET /api/thread/<turn_id> ───────────────────────────────────────────


def test_get_thread_drops_trailing_cancelled_orphan_but_keeps_prior_exchange(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A forked reply into an already-answered turn gets cancelled before any
    reply lands (the real ``DELETE`` endpoint, the FE stop button's own call).
    ``GET`` must still return the opener's real user+assistant content, but the
    reply-less trailing user row must be gone — never a dangling, unanswered
    bubble on refetch."""
    client, _db, _store = authed_client
    turn_id = 8301
    opener_user_id, opener_answer_id = _open_exchange(
        turn_id, "What's on my calendar today?", "You have no events today.",
    )

    reply = MessageProcessor(UserConfig(), turn_id, "Actually, add a 3pm meeting.")  # forked reply, inert (I2)
    reply_user_id = reply.transcript_service.append_input(reply.raw_input)
    reply.uid = reply_user_id
    reply.current_transcript_id = reply_user_id
    opened = reply.turn_execution_service.open()
    assert opened is not None  # sanity: the real open-row write succeeded

    cancel_resp = client.delete(f"/api/thread/{turn_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.get_json()["state"] == "cancelled"

    get_resp = client.get(f"/api/thread/{turn_id}")
    assert get_resp.status_code == 200
    ids = [m["id"] for m in get_resp.get_json()["messages"]]

    assert str(opener_user_id) in ids
    assert str(opener_answer_id) in ids
    assert str(reply_user_id) not in ids


def test_get_thread_with_only_a_cancelled_orphan_returns_no_messages(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A single-exchange turn cancelled before any reply: the ENTIRE turn is
    the orphan, so ``GET`` returns it with messages stripped to empty rather
    than a lone, unanswered user bubble."""
    client, _db, _store = authed_client
    turn_id = 8302
    mp = MessageProcessor(UserConfig(), turn_id, "Send an email to Jane about the invoice.")  # inert (I2)
    user_id = mp.transcript_service.append_input(mp.raw_input)
    mp.uid = user_id
    mp.current_transcript_id = user_id
    opened = mp.turn_execution_service.open()
    assert opened is not None  # sanity: the real open-row write succeeded

    cancel_resp = client.delete(f"/api/thread/{turn_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.get_json()["state"] == "cancelled"

    get_resp = client.get(f"/api/thread/{turn_id}")
    assert get_resp.status_code == 200
    assert get_resp.get_json()["messages"] == []


# ── (b) thread feed (recent_threads / count_turns) ──────────────────────────


def test_recent_threads_excludes_turn_whose_only_row_is_a_cancelled_orphan(
    db: sqlite3.Connection,
) -> None:
    """A turn whose only row is a cancelled, reply-less user message must not
    surface in the collapsed feed at all — it never became a real exchange."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 8401
    mp = MessageProcessor(UserConfig(), turn_id, "Cancel me before any reply.")  # inert (I2)
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    mp.current_transcript_id = mp.uid
    assert mp.turn_execution_service.open() is not None  # sanity: real open-row write
    cancelled = mp.turn_execution_service.cancel()
    assert cancelled is not None and cancelled.state == TurnExecution.CANCELLED  # sanity: real cancel

    threads, _has_more, _returned = Transcript.recent_threads(_CHANNEL)

    assert turn_id not in [t.get("turn_id") for t in threads]
    assert Transcript.count_turns(_CHANNEL) == 0


def test_recent_threads_keeps_turn_with_real_exchange_despite_trailing_cancelled_orphan(
    db: sqlite3.Connection,
) -> None:
    """A turn that has at least one real, non-user row (an actual answer) must
    stay in the feed even when its LATEST exchange — a forked reply — was
    cancelled before it got its own answer: the turn is not empty, only its
    trailing reply is."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 8402
    _open_exchange(turn_id, "What's the weather?", "Sunny, 22C.")

    reply = MessageProcessor(UserConfig(), turn_id, "And tomorrow?")  # forked reply, inert (I2)
    reply.uid = reply.transcript_service.append_input(reply.raw_input)
    reply.current_transcript_id = reply.uid
    assert reply.turn_execution_service.open() is not None  # sanity: real open-row write
    cancelled = reply.turn_execution_service.cancel()
    assert cancelled is not None and cancelled.state == TurnExecution.CANCELLED  # sanity: real cancel

    threads, _has_more, _returned = Transcript.recent_threads(_CHANNEL)

    assert turn_id in [t.get("turn_id") for t in threads]
    assert Transcript.count_turns(_CHANNEL) == 1
