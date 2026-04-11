# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for DMNService._active_topic() — Option A whitelist (Commit 9 lockstep fix).

After the Commit 9 channel collapse, 'dmn', 'goal_pursuit', and 'scheduled' are
flat channel strings in the transcript table. The old LIKE-exclusion approach no
longer works. _active_topic() now uses a strict whitelist:
    WHERE channel = 'user'

Tests use a real in-memory SQLite DB built from the production schema via the
`db` fixture (no hard-coded DDL — per feedback_test_ddl_drift.md).

North star: /Volumes/llm/chalie-plans/message-processing.md
Plan: /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md § Commit 9
Caller patches: /Volumes/llm/chalie-plans/scratch/commit_9_caller_patches.md § 1
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_dmn_service(store):
    """Instantiate DMNService with the test db + store, bypassing process singletons.

    Infrastructure injection only — the returned service uses real DB + store
    for all method calls. Pattern mirrors test_dmn_service.py._make_service().
    """
    from services.dmn_service import DMNService
    from services.database_service import get_shared_db_service

    with patch('services.database_service.get_shared_db_service',
               return_value=get_shared_db_service()), \
         patch('services.memory_client.MemoryClientService.create_connection',
               return_value=store):
        svc = DMNService()
    return svc


def _insert_transcript(db, channel, role='user', content='hello', ts='2026-04-10T10:00:00'):
    db.execute(
        "INSERT INTO transcript (channel, role, content, created_at) "
        "VALUES (?, ?, ?, ?)",
        (channel, role, content, ts),
    )
    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Whitelist contract tests
# ─────────────────────────────────────────────────────────────────────────────

class TestActiveTopicWhitelist:
    """_active_topic() uses WHERE channel = 'user' — strict whitelist, Option A."""

    def test_returns_user_when_only_user_rows_present(self, db, store):
        """Happy path: single user row → returns 'user'."""
        _insert_transcript(db, channel='user', ts='2026-04-10T10:00:00')

        svc = _make_dmn_service(store)
        assert svc._active_topic() == 'user'

    def test_returns_general_fallback_when_no_rows(self, db, store):
        """Empty transcript → returns 'general' (the hardcoded fallback)."""
        svc = _make_dmn_service(store)
        assert svc._active_topic() == 'general'

    def test_returns_general_when_only_dmn_rows(self, db, store):
        """Flat 'dmn' channel is excluded by whitelist → returns 'general'."""
        _insert_transcript(db, channel='dmn', role='proactive_thought',
                           content='background thought', ts='2026-04-10T12:00:00')

        svc = _make_dmn_service(store)
        assert svc._active_topic() == 'general'

    def test_returns_general_when_only_goal_pursuit_rows(self, db, store):
        """Flat 'goal_pursuit' channel is excluded by whitelist → returns 'general'."""
        _insert_transcript(db, channel='goal_pursuit', role='goal_pursuit',
                           content='pursuing goal', ts='2026-04-10T12:00:00')

        svc = _make_dmn_service(store)
        assert svc._active_topic() == 'general'

    def test_returns_general_when_only_scheduled_rows(self, db, store):
        """Flat 'scheduled' channel is excluded by whitelist → returns 'general'."""
        _insert_transcript(db, channel='scheduled', role='scheduled',
                           content='running task', ts='2026-04-10T12:00:00')

        svc = _make_dmn_service(store)
        assert svc._active_topic() == 'general'

    def test_user_channel_returned_even_when_dmn_is_more_recent(self, db, store):
        """Whitelist: 'user' row at 10:00 wins over 'dmn' row at 12:00."""
        _insert_transcript(db, channel='user', ts='2026-04-10T10:00:00')
        _insert_transcript(db, channel='dmn', role='proactive_thought',
                           content='later thought', ts='2026-04-10T12:00:00')

        svc = _make_dmn_service(store)
        assert svc._active_topic() == 'user'

    def test_user_channel_returned_even_when_goal_pursuit_is_more_recent(self, db, store):
        """Whitelist: 'user' row at 10:00 wins over 'goal_pursuit' row at 12:00."""
        _insert_transcript(db, channel='user', ts='2026-04-10T10:00:00')
        _insert_transcript(db, channel='goal_pursuit', role='goal_pursuit',
                           content='pursuing', ts='2026-04-10T12:00:00')

        svc = _make_dmn_service(store)
        assert svc._active_topic() == 'user'

    def test_user_channel_returned_even_when_scheduled_is_more_recent(self, db, store):
        """Whitelist: 'user' row at 10:00 wins over 'scheduled' row at 12:00."""
        _insert_transcript(db, channel='user', ts='2026-04-10T10:00:00')
        _insert_transcript(db, channel='scheduled', role='scheduled',
                           content='task ran', ts='2026-04-10T12:00:00')

        svc = _make_dmn_service(store)
        assert svc._active_topic() == 'user'

    def test_most_recent_user_row_returned_when_multiple_user_rows(self, db, store):
        """Most recent 'user' row wins when there are multiple user turns."""
        _insert_transcript(db, channel='user', content='first turn',
                           ts='2026-04-10T09:00:00')
        _insert_transcript(db, channel='user', content='second turn',
                           ts='2026-04-10T11:00:00')

        svc = _make_dmn_service(store)
        # The row returned is always 'user'; we verify the channel value (not content)
        assert svc._active_topic() == 'user'

    def test_all_three_background_channels_excluded_simultaneously(self, db, store):
        """dmn + goal_pursuit + scheduled all inserted; no user row → returns 'general'."""
        _insert_transcript(db, channel='dmn', role='proactive_thought',
                           ts='2026-04-10T11:00:00')
        _insert_transcript(db, channel='goal_pursuit', role='goal_pursuit',
                           ts='2026-04-10T11:30:00')
        _insert_transcript(db, channel='scheduled', role='scheduled',
                           ts='2026-04-10T12:00:00')

        svc = _make_dmn_service(store)
        assert svc._active_topic() == 'general'

    def test_db_exception_returns_general_fallback(self, db, store):
        """If the DB raises, _active_topic() swallows and returns 'general'."""
        svc = _make_dmn_service(store)

        # Patch connection() to raise
        original_connection = svc._db.connection

        class _RaisingCtx:
            def __enter__(self): raise RuntimeError('simulated DB failure')
            def __exit__(self, *a): pass

        svc._db.connection = lambda: _RaisingCtx()
        try:
            result = svc._active_topic()
        finally:
            svc._db.connection = original_connection

        assert result == 'general'
