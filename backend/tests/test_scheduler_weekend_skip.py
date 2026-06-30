# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test for the scheduler's only remaining skip arm — weekends.

Quiet-hours windows were removed, so the poll no longer skips firing based on
an item's active window. The sole surviving skip is the semantic of the
``weekdays`` recurrence: it must not fire on Sat/Sun, but a recurring item must
NEVER be silently lost — the skipped occurrence is marked ``fired`` and advances
to the next day, walking forward to the next weekday.

Drives the real production entry point ``_poll_and_fire()`` with a genuine due
notification row and a real ``WebSocketBroker`` connection (the delivery
boundary). The ONLY thing controlled is the wall clock — which day it is — via
``local_now``, the environment seam; no business logic is mocked.
"""

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services import scheduler_service
from services.time_utils import parse_utc, utc_now

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def suppress_embed() -> Iterator[None]:
    """The next-occurrence embedding is a fire-and-forget side effect unrelated to
    the skip logic; suppress it so it can't touch the poll's shared connection."""
    with patch("services.scheduler_service.embed_scheduled_item"):
        yield

# Reference Saturday / Monday at noon local — the poll only inspects ``.weekday()``.
_SATURDAY = datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)  # weekday() == 5
_MONDAY = datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)    # weekday() == 0


class _RecordingSocket:
    """A real broker connection that records every payload it is sent."""

    def __init__(self) -> None:
        self.payloads: list[str] = []

    def send(self, payload: str) -> None:
        self.payloads.append(payload)


@pytest.fixture
def listener() -> "object":
    """Register a live socket on the singleton broker; disconnect on teardown."""
    from services.websocket_broker import WebSocketBroker

    sock = _RecordingSocket()
    broker = WebSocketBroker()
    broker.connect(sock)
    try:
        yield sock
    finally:
        broker.disconnect(sock)


def _insert_weekdays_item(db: sqlite3.Connection, item_id: str) -> str:
    """A genuine due, recurring 'weekdays' notification — what the poll claims."""
    due_at = (utc_now() - timedelta(minutes=1)).isoformat()
    db.execute(
        "INSERT INTO scheduled_items (id, item_type, message, due_at, recurrence, status, group_id) "
        "VALUES (?, 'notification', 'Daily standup', ?, 'weekdays', 'pending', ?)",
        (item_id, due_at, item_id),
    )
    db.commit()
    return due_at


def test_weekend_skip_advances_recurring_item_without_firing(
    db: sqlite3.Connection, listener: _RecordingSocket
) -> None:
    """On a weekend the 'weekdays' occurrence is skipped (no broadcast) but the
    series still advances — the original is 'fired' and a fresh pending occurrence
    is created one day later. A recurring item is never silently lost."""
    due_at = _insert_weekdays_item(db, "wd-weekend")

    with patch("services.locale_service.local_now", return_value=_SATURDAY):
        scheduler_service._poll_and_fire()

    # (skip) nothing was delivered — the weekend occurrence did not fire.
    assert listener.payloads == [], (
        f"a 'weekdays' item must not fire on a weekend; broker received {listener.payloads!r}"
    )

    # (claimed) the due row is marked 'fired' and advances — not stuck pending/failed.
    assert db.execute(
        "SELECT status FROM scheduled_items WHERE id='wd-weekend'"
    ).fetchone()[0] == "fired"

    # (not lost) exactly one fresh pending occurrence exists, one day later,
    # in the same series — the recurring item survives the weekend skip.
    nxt = db.execute(
        "SELECT due_at, recurrence FROM scheduled_items "
        "WHERE status='pending' AND group_id='wd-weekend'"
    ).fetchall()
    assert len(nxt) == 1, f"expected exactly one next occurrence, got {nxt!r}"
    assert parse_utc(nxt[0][0]) == parse_utc(due_at) + timedelta(days=1)
    assert nxt[0][1] == "weekdays"


def test_weekday_fires_and_delivers(
    db: sqlite3.Connection, listener: _RecordingSocket
) -> None:
    """The skip is weekend-specific: on a weekday the same 'weekdays' item fires
    and the notification is delivered over the broker."""
    _insert_weekdays_item(db, "wd-weekday")

    with patch("services.locale_service.local_now", return_value=_MONDAY):
        scheduler_service._poll_and_fire()

    assert len(listener.payloads) == 1, (
        f"a 'weekdays' item must fire on a weekday; broker received {listener.payloads!r}"
    )
    assert "Daily standup" in listener.payloads[0]
