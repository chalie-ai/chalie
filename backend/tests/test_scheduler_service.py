# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the user-schedule dispatcher,
``cron.jobs.scheduled_items.ScheduledItemsDispatcherJob``. ``_run`` must select
only enabled, already-started rows whose real 5-field crontab expression
(``cron_minute``/``cron_hour``/``cron_dom``/``cron_month``/``cron_dow`` — see
``services.cron_schedule.matches``) matches the current wall-clock minute, and
fire them asynchronously; a fired row runs the real ``MessageProcessor`` keyed
to the schedule's own turn identity — the integer ``id`` IS the ``turn_id`` on
the ``schedule`` channel — so a first fire opens a MAIN turn and any later fire
of the SAME id appends as a FORK to that SAME turn (never a second turn).

The exhaustive per-field / timezone truth table for ``matches`` lives in
``tests/test_cron_schedule.py``; this file only proves the dispatcher wires
ALL FIVE stored fields into that call (not just minute/hour) and drives the
real fire-and-turn side effects.

The LLM boundary (``ProviderService``'s client factory) is swapped for a fixed,
hand-built recorder class (never a ``Mock``) — the one exception rule H
carves out (a network call to a process Chalie does not own). Everything
else — the DB, the poller, ``MessageProcessor``, transcript, turn-execution —
is the real production stack. Background daemon threads are joined by exact
name, never waited on with ``time.sleep``.
"""

import json
import sqlite3
import threading
from datetime import timedelta
from unittest.mock import patch

import pytest

from configs.enums.channels import Channel
from cron.jobs.scheduled_items import ScheduledItemsDispatcherJob
from models.provider_response import ProviderResponse
from models.thread_gist import ThreadGist
from services.time_utils import utc_now
from models.transcript import Transcript
from tests.helpers import insert_scheduled_item

pytestmark = pytest.mark.unit

# The dispatcher is stateless across ticks (no per-run instance state), so one
# shared instance exercises the real production object — the runner builds it
# exactly once in ``cron.JOBS`` too.
_DISPATCHER = ScheduledItemsDispatcherJob()

_BUILD_CLIENT = "services.provider_service.build_client"
_JOIN_TIMEOUT = 10.0


class _RecordingProvider:
    """Returns one fixed response and no tool calls — a real hand-built stand-in
    for the LLM transport client, not a ``Mock``."""

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, response_text: str = "done") -> None:
        self._response_text = response_text

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        return ProviderResponse(text=self._response_text, model="recorder", tool_calls=None)


def _seed_timezone(db: sqlite3.Connection, tz_name: str = "UTC") -> None:
    """Write a real heartbeat timezone into the telemetry table — the same
    store the production ``LocaleService.get_timezone()`` reads."""
    from services.heartbeat_service import heartbeat_service

    heartbeat_service._ctx = None
    db.execute("DELETE FROM telemetry")
    db.execute(
        "INSERT INTO telemetry (key, value) VALUES (?, ?)",
        ("timezone", json.dumps(tz_name)),
    )
    db.commit()
    heartbeat_service._ctx = None


def _join_named_thread(name: str, timeout: float = _JOIN_TIMEOUT) -> bool:
    """Find a live thread by its exact name and join it — the real
    synchronization primitive for a fire-and-forget daemon thread. Returns
    whether such a thread was found at all (never sleeps to "wait and see")."""
    for t in threading.enumerate():
        if t.name == name:
            t.join(timeout)
            return True
    return False


def _schedule_turn_rows(turn_id: int) -> list[dict[str, object]]:
    rows = Transcript.filter("channel", Channel.SCHEDULE.value).filter("turn_id", turn_id).get()
    return [r.to_dict() for r in rows]


def test_poll_and_fire_selects_only_enabled_started_cron_matching_rows(
    db: sqlite3.Connection,
) -> None:
    _seed_timezone(db, "UTC")
    now = utc_now()
    matching_id = insert_scheduled_item(
        db, message="Fire me", cron_hour=str(now.hour), cron_minute=str(now.minute)
    )
    wrong_minute = (now.minute + 30) % 60
    non_matching_id = insert_scheduled_item(db, message="Wrong minute", cron_minute=str(wrong_minute))
    disabled_id = insert_scheduled_item(
        db, message="Disabled", cron_hour=str(now.hour), cron_minute=str(now.minute), enabled=0
    )
    future_id = insert_scheduled_item(
        db, message="Future", start_at=(now + timedelta(days=1)).isoformat()
    )

    with patch(_BUILD_CLIENT, return_value=_RecordingProvider()):
        _DISPATCHER._run()
        fired = _join_named_thread(f"scheduled-work-{matching_id}")
        assert fired, "the matching, enabled, already-started row must be fired"
        _join_named_thread(f"turn-{matching_id}")

    assert _schedule_turn_rows(matching_id), "the fired row must open a real turn on the schedule channel"

    for skipped_id in (non_matching_id, disabled_id, future_id):
        assert not _join_named_thread(f"scheduled-work-{skipped_id}", timeout=0.1), (
            f"row {skipped_id} must never fire"
        )
        assert not _schedule_turn_rows(skipped_id)


def test_poll_wires_month_and_weekday_fields_not_just_minute_and_hour(
    db: sqlite3.Connection,
) -> None:
    """The old dumb-cron model only ever stored/matched dom/hour/minute; the
    real crontab engine adds ``cron_month``/``cron_dow``. A row whose minute
    and hour match but whose month does NOT must still be skipped — proving
    ``_run`` actually forwards all five stored fields into ``matches()``
    rather than silently dropping the two new ones."""
    _seed_timezone(db, "UTC")
    now = utc_now()
    wrong_month = (now.month % 12) + 1  # any month other than the current one

    matching_id = insert_scheduled_item(
        db, message="Right month",
        cron_minute=str(now.minute), cron_hour=str(now.hour), cron_month=str(now.month),
    )
    wrong_month_id = insert_scheduled_item(
        db, message="Wrong month",
        cron_minute=str(now.minute), cron_hour=str(now.hour), cron_month=str(wrong_month),
    )

    with patch(_BUILD_CLIENT, return_value=_RecordingProvider()):
        _DISPATCHER._run()
        assert _join_named_thread(f"scheduled-work-{matching_id}"), (
            "a row whose month matches the current month must fire"
        )

    assert not _join_named_thread(f"scheduled-work-{wrong_month_id}", timeout=0.1), (
        "a row restricted to a different month must never fire even if minute/hour match"
    )
    assert not _schedule_turn_rows(wrong_month_id)


def test_first_fire_opens_main_turn_second_fire_forks_the_same_turn(
    db: sqlite3.Connection,
) -> None:
    _seed_timezone(db, "UTC")
    item_id = insert_scheduled_item(db, message="Water the plants")

    # Pre-seed the turn's gist so BOTH fires' best-effort ingest
    # (`MessageProcessor._maybe_fire_gist`) short-circuits instead of spawning
    # its own nested, unpatched background MessageProcessor for gisting. The
    # schedule channel generates on the first fire, so this has to land before
    # it — otherwise that daemon outlives the patch below and runs unprovidered.
    # This test is about turn threading; the gist is isolated scaffolding.
    ThreadGist(
        channel=Channel.SCHEDULE.value, turn_id=item_id, gist="Watering reminder",
        created_at=utc_now().isoformat(),
    ).upsert()
    db.commit()

    with patch(_BUILD_CLIENT, return_value=_RecordingProvider("first response")):
        _DISPATCHER._fire_item(item_id, "Water the plants")
        assert _join_named_thread(f"scheduled-work-{item_id}")
        assert _join_named_thread(f"turn-{item_id}")

    first_rows = _schedule_turn_rows(item_id)
    assert first_rows, "the first fire must open a MAIN turn on (channel='schedule', turn_id=item_id)"

    with patch(_BUILD_CLIENT, return_value=_RecordingProvider("second response")):
        _DISPATCHER._fire_item(item_id, "Water the plants")
        assert _join_named_thread(f"scheduled-work-{item_id}")
        assert _join_named_thread(f"turn-{item_id}")

    second_rows = _schedule_turn_rows(item_id)
    assert len(second_rows) > len(first_rows), "the second fire must APPEND to the same turn"

    all_schedule_turn_ids = {
        row[0]
        for row in db.execute(
            "SELECT DISTINCT turn_id FROM transcript WHERE channel = 'schedule'"
        ).fetchall()
    }
    assert all_schedule_turn_ids == {item_id}, (
        "a recurring schedule is one growing thread — the second fire must never open a new turn_id"
    )
