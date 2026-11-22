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

import logging
import sqlite3
import threading
from datetime import timedelta
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from configs.channels import config_for
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from controllers.message_processor import MessageProcessor
from cron.jobs.scheduled_items import ScheduledItemsDispatcherJob
from models.provider_response import ProviderResponse
from models.thread_gist import ThreadGist
from models.turn_execution import TurnExecution
from services.time_utils import utc_now
from models.transcript import Transcript
from tests.helpers import insert_scheduled_item
from tests._delegates import ProcessDeath, process_deaths_expected

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
    """Persist a heartbeat timezone into the telemetry JSON snapshot — the same
    store the production ``locale_service.get_timezone()`` reads. The ``db``
    fixture stays a parameter: it is what redirects the snapshot path into this
    test's tmp dir."""
    from services.telemetry_service import TelemetryService

    TelemetryService.write({"timezone": tz_name})


def _join_named_thread(name: str, timeout: float = _JOIN_TIMEOUT) -> bool:
    """Find a live thread by its exact name and join it — the real
    synchronization primitive for a fire-and-forget daemon thread. Returns
    whether such a thread was found at all (never sleeps to "wait and see")."""
    for t in threading.enumerate():
        if t.name == name:
            t.join(timeout)
            return True
    return False


def _label_the_first_fire(db: sqlite3.Connection, gist: str) -> None:
    """Store the label of the thread a schedule's first fire opens, before it
    fires, so no fire launches a label-writing turn of its own
    (``MessageProcessor._maybe_fire_gist``) that would outlive the provider
    patch. A label is found by its thread's opening row id, and the first
    fire's input row is the next transcript row written."""
    seq = db.execute("SELECT seq FROM sqlite_sequence WHERE name = 'transcript'").fetchone()
    ThreadGist(opening_id=(int(seq[0]) if seq else 0) + 1, gist=gist).upsert()
    db.commit()


def _assert_labelled(item_id: int, gist: str) -> None:
    """The label stored before the first fire sits on the thread it opened."""
    assert ThreadGist.bulk_get(Channel.SCHEDULE.value, [item_id]) == {item_id: gist}


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
        assert _join_named_thread(f"turn-{matching_id}")

    assert not _join_named_thread(f"scheduled-work-{wrong_month_id}", timeout=0.1), (
        "a row restricted to a different month must never fire even if minute/hour match"
    )
    assert not _schedule_turn_rows(wrong_month_id)


def test_first_fire_opens_main_turn_second_fire_forks_the_same_turn(
    db: sqlite3.Connection,
) -> None:
    _seed_timezone(db, "UTC")
    item_id = insert_scheduled_item(db, message="Water the plants")

    # Label the thread before its first fire so neither fire's best-effort
    # label ingest spawns its own background turn. The schedule channel labels
    # on the first fire, so this has to land before it. This test is about turn
    # threading; the label is isolated scaffolding.
    _label_the_first_fire(db, "Watering reminder")

    with patch(_BUILD_CLIENT, return_value=_RecordingProvider("first response")):
        _DISPATCHER._fire_item(item_id, "Water the plants")
        assert _join_named_thread(f"scheduled-work-{item_id}")
        assert _join_named_thread(f"turn-{item_id}")
    _assert_labelled(item_id, "Watering reminder")

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


class _GatedProvider:
    """Holds every call until the test opens the gate — a hand-built stand-in
    for the LLM transport client (never a ``Mock``) that keeps a fired run
    working for as long as the test needs, the way a slow model call does."""

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.gate = threading.Event()

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        self.entered.set()
        assert self.gate.wait(_JOIN_TIMEOUT), "the test never opened the provider gate"
        return ProviderResponse(text="checked", model="gated", tool_calls=None)

    def abort(self) -> None:
        self.gate.set()


def _schedule_executions(turn_id: int) -> list[TurnExecution]:
    return list(
        TurnExecution.filter("channel", Channel.SCHEDULE.value)
        .filter("turn_id", turn_id)
        .order_by("id ASC")
        .get()
    )


def test_every_minute_schedule_skips_fires_while_its_run_is_still_going(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    """A schedule with every crontab field ``*`` matches every poll. While its
    first run is still waiting on the model, three further polls (three later
    minutes) must each skip it with one log line and open no second run on its
    thread; once that run finishes, the next poll fires again onto the SAME
    thread."""
    _seed_timezone(db, "UTC")
    item_id = insert_scheduled_item(db, message="Daily bloat check")
    # Label the thread up front so no fire spawns its own background labelling
    # turn (see the fork test above); this test is about overlap, not labels.
    _label_the_first_fire(db, "Bloat check")
    caplog.set_level(logging.INFO, logger="cron.jobs.scheduled_items")
    provider = _GatedProvider()

    with patch(_BUILD_CLIENT, return_value=provider):
        try:
            assert _DISPATCHER._run() == "fired 1 of 1 due row(s)"
            assert _join_named_thread(f"scheduled-work-{item_id}")
            assert provider.entered.wait(_JOIN_TIMEOUT), "the first run never reached the model"

            [first] = _schedule_executions(item_id)
            assert first.state == TurnExecution.WORKING and first.ended_at is None
            _assert_labelled(item_id, "Bloat check")

            for _ in range(3):
                assert _DISPATCHER._run() == "fired 0 of 1 due row(s)"
                assert not _join_named_thread(f"scheduled-work-{item_id}", timeout=0.1), (
                    "a fire must never start while the schedule's last run is still going"
                )
            assert [e.id for e in _schedule_executions(item_id)] == [first.id], (
                "the skipped fires must open no second run on the schedule's thread"
            )
            skips = [
                r.message for r in caplog.records
                if r.message.startswith(f"[SCHEDULER] Skipped schedule {item_id}:")
            ]
            assert skips == [
                f"[SCHEDULER] Skipped schedule {item_id}: its run started at "
                f"{first.started_at} is still going"
            ] * 3, "each skipped fire logs exactly one line"
        finally:
            provider.gate.set()
        assert _join_named_thread(f"turn-{item_id}")

        [finished] = _schedule_executions(item_id)
        assert finished.state == TurnExecution.COMPLETED and finished.ended_at is not None

        assert _DISPATCHER._run() == "fired 1 of 1 due row(s)", (
            "once the last run has finished, the next match fires again"
        )
        assert _join_named_thread(f"scheduled-work-{item_id}")
        assert _join_named_thread(f"turn-{item_id}")

    executions = _schedule_executions(item_id)
    assert [e.id for e in executions][0] == first.id and len(executions) == 2
    assert all(e.state == TurnExecution.COMPLETED for e in executions)
    turn_ids = {
        row[0] for row in db.execute(
            "SELECT DISTINCT turn_id FROM transcript WHERE channel = 'schedule'"
        ).fetchall()
    }
    assert turn_ids == {item_id}, "the second run appends to the schedule's own thread"


class _DyingProvider:
    """A model call that is the last thing its process does — a hand-built
    stand-in for the LLM transport client, never a ``Mock``."""

    CONTENT_FIELD_LABEL = "message.content"

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        raise ProcessDeath


def test_a_run_a_restart_cut_off_blocks_fires_until_the_boot_sweep_ends_it(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    """A run whose process died leaves its execution open. Until the next boot
    sweeps it, that run is never assumed finished, so the schedule still skips;
    once the boot sweep stamps it crashed, the next match fires again onto the
    same thread."""
    _seed_timezone(db, "UTC")
    item_id = insert_scheduled_item(db, message="Daily bloat check")
    _label_the_first_fire(db, "Bloat check")
    caplog.set_level(logging.INFO, logger="cron.jobs.scheduled_items")

    with patch(_BUILD_CLIENT, return_value=_DyingProvider()), process_deaths_expected():
        assert _DISPATCHER._run() == "fired 1 of 1 due row(s)"
        assert _join_named_thread(f"scheduled-work-{item_id}")
        assert _join_named_thread(f"turn-{item_id}")
    _assert_labelled(item_id, "Bloat check")

    [cut] = _schedule_executions(item_id)
    assert cut.state == TurnExecution.WORKING and cut.ended_at is None, (
        "a run whose process died is left open for the boot sweep"
    )
    assert _DISPATCHER._run() == "fired 0 of 1 due row(s)", (
        "a run left open is never assumed finished before the boot sweep"
    )
    assert not _join_named_thread(f"scheduled-work-{item_id}", timeout=0.1)
    assert [r.message for r in caplog.records if r.message.startswith("[SCHEDULER] Skipped")] == [
        f"[SCHEDULER] Skipped schedule {item_id}: its run started at {cut.started_at} is still going"
    ]

    swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()
    assert [(r.id, r.channel, r.turn_id, r.state, r.stop_reason) for r in swept] == [
        (cut.id, Channel.SCHEDULE.value, item_id, TurnExecution.CRASHED, "process death")
    ]

    with patch(_BUILD_CLIENT, return_value=_RecordingProvider("checked")):
        assert _DISPATCHER._run() == "fired 1 of 1 due row(s)", (
            "once the boot sweep has ended the cut-off run, the next match fires again"
        )
        assert _join_named_thread(f"scheduled-work-{item_id}")
        assert _join_named_thread(f"turn-{item_id}")

    executions = _schedule_executions(item_id)
    assert executions[0].id == cut.id
    assert [e.state for e in executions] == [TurnExecution.CRASHED, TurnExecution.COMPLETED]
    assert list(TurnExecution.open_rows()) == []


def test_a_reply_running_in_the_schedules_thread_blocks_its_fires(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    """A person's reply into the schedule's thread runs on the schedule's own
    turn. While that reply's run is still going, a match is skipped exactly as
    if the schedule had started the run itself."""
    client, _db, _store = authed_client
    _seed_timezone(db, "UTC")
    item_id = insert_scheduled_item(db, message="Daily bloat check")
    _label_the_first_fire(db, "Bloat check")
    caplog.set_level(logging.INFO, logger="cron.jobs.scheduled_items")

    with patch(_BUILD_CLIENT, return_value=_RecordingProvider("checked")):
        assert _DISPATCHER._run() == "fired 1 of 1 due row(s)"
        assert _join_named_thread(f"scheduled-work-{item_id}")
        assert _join_named_thread(f"turn-{item_id}")
    _assert_labelled(item_id, "Bloat check")

    provider = _GatedProvider()
    with patch(_BUILD_CLIENT, return_value=provider):
        try:
            resp = client.post(
                f"/api/threads/{item_id}", data={"text": "anything new?", "type": "scheduled"},
            )
            assert resp.status_code == 200, resp.get_json()
            assert provider.entered.wait(_JOIN_TIMEOUT), "the reply's run never reached the model"
            fired_run, reply_run = _schedule_executions(item_id)
            assert fired_run.state == TurnExecution.COMPLETED
            assert reply_run.state == TurnExecution.WORKING and reply_run.ended_at is None

            assert _DISPATCHER._run() == "fired 0 of 1 due row(s)"
            assert not _join_named_thread(f"scheduled-work-{item_id}", timeout=0.1), (
                "a fire must never start while a reply's run on the schedule's thread is going"
            )
            assert [
                r.message for r in caplog.records if r.message.startswith("[SCHEDULER] Skipped")
            ] == [
                f"[SCHEDULER] Skipped schedule {item_id}: its run started at "
                f"{reply_run.started_at} is still going"
            ]
        finally:
            provider.gate.set()
        assert _join_named_thread(f"turn-{item_id}")

    assert [e.state for e in _schedule_executions(item_id)] == [TurnExecution.COMPLETED] * 2
