"""Feature test for ``DmnJob``'s spine-advanced gate.

DMN is the one cognition job that manufactures its own inputs — every
reflection becomes a memory the next reflection reads. Unlike the
watermark-idempotent jobs it must not re-fire through a single idle stretch, or
it reflects on its own prior reflections (a self-feeding memory loop). The
gate: reflect only when a user message has landed since DMN last reflected.

This exercises that gate against the REAL ``transcript`` table and the REAL
durable last-fired clock (``cron:dmn:last_fired``) — never a mock. ``_run``
itself (the DMNMessageProcessor call) is not driven here; the whole point of
the fix is that the decision to fire lives in ``should_run`` so a skipped tick
never advances the clock.
"""

import sqlite3
from datetime import datetime, timedelta

import pytest

from cron.jobs.dmn import DmnJob
from services.memory_store import MemoryStore
from services.time_utils import utc_now
from services.user_synthesis import UserSynthesis

pytestmark = pytest.mark.integration


def _seed_user_message(db: sqlite3.Connection, when: datetime) -> None:
    """Insert one user-role transcript row — the spine ``last_user_message_at`` reads."""
    db.execute(
        "INSERT INTO transcript (role, content, channel, created_at) "
        "VALUES ('user', 'hi', 'test', ?)",
        (when.isoformat(),),
    )
    db.commit()


# ── The spine gate in isolation ───────────────────────────────────────────────


def test_never_reflected_allows_first_reflection(db: sqlite3.Connection, store: MemoryStore) -> None:
    """No durable last-fired stamp → the first idle reflection is always allowed."""
    _seed_user_message(db, utc_now() - timedelta(minutes=40))
    job = DmnJob()
    assert job._get_last_fired().load() is None
    assert job._spine_advanced() is True


def test_user_spoke_since_last_reflection_allows_reflection(db: sqlite3.Connection, store: MemoryStore) -> None:
    """A user message newer than the last reflection is new spine → fire."""
    job = DmnJob()
    job._persist_fired(utc_now() - timedelta(minutes=30))
    _seed_user_message(db, utc_now() - timedelta(minutes=10))
    assert job._spine_advanced() is True


def test_no_new_message_since_last_reflection_skips(db: sqlite3.Connection, store: MemoryStore) -> None:
    """Reflected AFTER the newest user message → nothing new; skip (no self-feed)."""
    _seed_user_message(db, utc_now() - timedelta(minutes=40))
    job = DmnJob()
    job._persist_fired(utc_now() - timedelta(minutes=5))
    assert job._spine_advanced() is False


def test_already_fired_with_no_user_message_ever_skips(db: sqlite3.Connection, store: MemoryStore) -> None:
    """Fired once, no user rows at all → no spine to advance; skip."""
    job = DmnJob()
    job._persist_fired(utc_now() - timedelta(minutes=5))
    assert job._spine_advanced() is False


# ── The gate composed into should_run ─────────────────────────────────────────


def test_should_run_true_when_idle_synthesis_and_spine_all_pass(db: sqlite3.Connection, store: MemoryStore) -> None:
    """Idle 45 min, a synthesis exists, never reflected → the job may fire."""
    _seed_user_message(db, utc_now() - timedelta(minutes=45))
    UserSynthesis.upsert("a portrait of the user", shorthand=True)
    job = DmnJob()  # never fired → spine + interval both clear
    assert job.should_run() is True


def test_should_run_false_when_spine_stale(db: sqlite3.Connection, store: MemoryStore) -> None:
    """Idle + interval + synthesis all pass, but DMN already reflected since the
    user last spoke → the spine gate alone must block the re-fire."""
    _seed_user_message(db, utc_now() - timedelta(minutes=45))
    UserSynthesis.upsert("a portrait of the user", shorthand=True)
    job = DmnJob()
    job._persist_fired(utc_now() - timedelta(minutes=10))  # interval elapsed, spine stale
    assert job.should_run() is False
