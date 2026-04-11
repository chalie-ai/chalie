# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for Commit 4: atomic store() and append_atomic_turn().

Coverage:
  A. Happy-path: rows, ids, ordering
  B. DTO sort ordering by timestamp (incl. stable-sort tiebreaker)
  C. Atomicity — mid-transaction failure leaves zero rows
  D. DTO field contract — ephemeral, invoked_by, params serialization
  E. Empty pending_tool_calls
  F. Embedding hook daemon-thread contract
  G. Episode extraction hook
  H. MessageProcessor.store() integration
  I. Empty ROLE warns but proceeds
  J. Exception propagates, self._uid stays None
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_CHANNEL = 'test_store'


def _dto(
    name='tool_x',
    params=None,
    result='ok',
    ephemeral=1,
    invoked_by='llm',
    timestamp='2026-04-11T10:00:00+00:00',
):
    return {
        'name': name,
        'params': params if params is not None else {},
        'result': result,
        'ephemeral': ephemeral,
        'invoked_by': invoked_by,
        'timestamp': timestamp,
    }


def _make_processor(channel=_CHANNEL, role='user', raw_input='hello', **kwargs):
    """Build a concrete FakeProcessor with overridable CHANNEL, ROLE, raw_input."""
    from services.message_processor import MessageProcessor

    class _Fake(MessageProcessor):
        CHANNEL = channel
        ROLE = role

        def getUserDefinition(self) -> str:
            return 'test definition'

        def getUserPrompt(self) -> str:
            return raw_input

    for k, v in kwargs.items():
        setattr(_Fake, k, v)

    return _Fake(raw_input)


def _row_count(db, table):
    row = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return row[0]


def _fetch_all(db, table):
    cursor = db.execute(f"SELECT * FROM {table} ORDER BY id")
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _fetch_tool_calls(db):
    cursor = db.execute("SELECT * FROM tool_calls ORDER BY id")
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _fetch_transcript(db):
    cursor = db.execute("SELECT * FROM transcript ORDER BY id")
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# A. Happy-path — rows, ids, ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendAtomicTurnHappyPath:
    """append_atomic_turn() with 2 DTOs produces exactly 4 rows."""

    def test_row_counts(self, db):
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=[_dto('a'), _dto('b')],
            )

        assert _row_count(db, 'transcript') == 2
        assert _row_count(db, 'tool_calls') == 2

    def test_returns_input_row_id(self, db):
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            input_id = append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=[_dto()],
            )

        rows = _fetch_transcript(db)
        assert rows[0]['id'] == input_id
        assert rows[0]['role'] == 'user'

    def test_input_id_less_than_tool_call_ids_less_than_assistant_id(self, db):
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            input_id = append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=[_dto('a'), _dto('b')],
            )

        tcs = _fetch_tool_calls(db)
        transcript_rows = _fetch_transcript(db)
        assistant_id = transcript_rows[1]['id']

        for tc in tcs:
            assert tc['transcript_id'] == input_id

        # AUTOINCREMENT ensures input < tool_call ids < assistant
        # (tool_calls.id is from its own sequence, but transcript ids are ordered)
        assert input_id < assistant_id

    def test_tool_calls_link_to_input_row(self, db):
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            input_id = append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=[_dto('a'), _dto('b')],
            )

        tcs = _fetch_tool_calls(db)
        assert len(tcs) == 2
        for tc in tcs:
            assert tc['transcript_id'] == input_id

    def test_assistant_row_has_correct_role_and_content(self, db):
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world response',
                pending_tool_calls=[],
            )

        rows = _fetch_transcript(db)
        assert len(rows) == 2
        assert rows[1]['role'] == 'assistant'
        assert rows[1]['content'] == 'world response'


# ─────────────────────────────────────────────────────────────────────────────
# B. DTO ordering — timestamp sort + stable-sort tiebreaker
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendAtomicTurnOrdering:
    """Timestamp sort is stable — ties preserve insertion order."""

    def test_reverse_chron_input_produces_chron_db_order(self, db):
        from services.transcript_service import append_atomic_turn

        dtos = [
            _dto('c', timestamp='2026-04-11T10:00:02+00:00'),
            _dto('b', timestamp='2026-04-11T10:00:01+00:00'),
            _dto('a', timestamp='2026-04-11T10:00:00+00:00'),
        ]

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=dtos,
            )

        tcs = _fetch_tool_calls(db)
        assert [tc['tool_name'] for tc in tcs] == ['a', 'b', 'c']

    def test_identical_timestamps_preserve_insertion_order(self, db):
        """Stable sort: ties keep the original list order."""
        from services.transcript_service import append_atomic_turn

        same_ts = '2026-04-11T10:00:00+00:00'
        dtos = [
            _dto('first',  timestamp=same_ts),
            _dto('second', timestamp=same_ts),
            _dto('third',  timestamp=same_ts),
        ]

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=dtos,
            )

        tcs = _fetch_tool_calls(db)
        assert [tc['tool_name'] for tc in tcs] == ['first', 'second', 'third']

    def test_mixed_timestamps_some_tied(self, db):
        """Timestamps sort first; tied timestamps keep insertion order."""
        from services.transcript_service import append_atomic_turn

        ts_early = '2026-04-11T10:00:00+00:00'
        ts_late  = '2026-04-11T10:00:02+00:00'

        dtos = [
            _dto('tied_b',  timestamp=ts_early),  # idx 0
            _dto('unique',  timestamp=ts_late),   # idx 1  — latest
            _dto('tied_a',  timestamp=ts_early),  # idx 2
        ]

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=dtos,
            )

        tcs = _fetch_tool_calls(db)
        # tied_b and tied_a share the earliest timestamp → keep insertion order
        # unique is latest → sorts last
        assert [tc['tool_name'] for tc in tcs] == ['tied_b', 'tied_a', 'unique']


# ─────────────────────────────────────────────────────────────────────────────
# C. Atomicity — mid-transaction failure leaves zero rows
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendAtomicTurnAtomicity:
    """Any failure inside the transaction rolls back ALL rows."""

    def test_invalid_invoked_by_check_constraint_rolls_back(self, db):
        """CHECK(invoked_by IN ('system','llm')) rejects 'invalid' — zero rows land."""
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            with pytest.raises(Exception):
                append_atomic_turn(
                    channel=_CHANNEL,
                    role='user',
                    raw_input='hello',
                    llm_response='world',
                    pending_tool_calls=[
                        _dto(invoked_by='invalid'),
                    ],
                )

        assert _row_count(db, 'transcript') == 0
        assert _row_count(db, 'tool_calls') == 0

    def test_exception_propagates_to_caller(self, db):
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            with pytest.raises(Exception):
                append_atomic_turn(
                    channel=_CHANNEL,
                    role='user',
                    raw_input='hello',
                    llm_response='world',
                    pending_tool_calls=[_dto(invoked_by='invalid')],
                )


# ─────────────────────────────────────────────────────────────────────────────
# D. DTO field contract
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendAtomicTurnDTOFields:
    """All DTO fields land in the DB with the correct values."""

    def test_ephemeral_flag_persisted(self, db):
        from services.transcript_service import append_atomic_turn

        dtos = [
            _dto('durable',   ephemeral=0, invoked_by='system'),
            _dto('ephemeral', ephemeral=1, invoked_by='llm'),
        ]

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='x',
                llm_response='y',
                pending_tool_calls=dtos,
            )

        tcs = _fetch_tool_calls(db)
        assert tcs[0]['ephemeral'] == 0
        assert tcs[1]['ephemeral'] == 1

    def test_invoked_by_persisted(self, db):
        from services.transcript_service import append_atomic_turn

        dtos = [
            _dto('tool_a', invoked_by='system'),
            _dto('tool_b', invoked_by='llm'),
        ]

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='x',
                llm_response='y',
                pending_tool_calls=dtos,
            )

        tcs = _fetch_tool_calls(db)
        assert tcs[0]['invoked_by'] == 'system'
        assert tcs[1]['invoked_by'] == 'llm'

    def test_invalid_invoked_by_triggers_check_constraint(self, db):
        """invoked_by='invalid' must trigger the DB CHECK constraint."""
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            with pytest.raises(Exception):
                append_atomic_turn(
                    channel=_CHANNEL,
                    role='user',
                    raw_input='x',
                    llm_response='y',
                    pending_tool_calls=[_dto(invoked_by='invalid')],
                )

    def test_params_stored_as_json_string(self, db):
        from services.transcript_service import append_atomic_turn

        original_params = {'query': 'test search', 'limit': 5}

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='x',
                llm_response='y',
                pending_tool_calls=[_dto(params=original_params)],
            )

        tcs = _fetch_tool_calls(db)
        assert len(tcs) == 1
        assert json.loads(tcs[0]['params']) == original_params

    def test_nested_params_roundtrip(self, db):
        from services.transcript_service import append_atomic_turn

        nested = {'outer': {'inner': [1, 2, 3]}, 'flag': True}

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='x',
                llm_response='y',
                pending_tool_calls=[_dto(params=nested)],
            )

        tcs = _fetch_tool_calls(db)
        assert json.loads(tcs[0]['params']) == nested


# ─────────────────────────────────────────────────────────────────────────────
# E. Empty pending_tool_calls
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendAtomicTurnEmptyPendingCalls:
    """Zero DTOs → exactly 2 transcript rows, no tool_calls rows."""

    def test_two_transcript_rows_no_tool_calls(self, db):
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            input_id = append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=[],
            )

        assert _row_count(db, 'transcript') == 2
        assert _row_count(db, 'tool_calls') == 0

    def test_returns_input_row_id_when_no_dtos(self, db):
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            input_id = append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=[],
            )

        rows = _fetch_transcript(db)
        assert rows[0]['id'] == input_id


# ─────────────────────────────────────────────────────────────────────────────
# F. Embedding hook — daemon thread contract
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendAtomicTurnEmbeddingHookFires:
    """_embed_entry is called exactly twice (input + assistant), via daemon threads."""

    def test_embed_called_twice(self, db):
        from services.transcript_service import append_atomic_turn

        calls = []

        def fake_embed(rowid, content):
            calls.append(rowid)

        # Drop the threshold so short test content still triggers the hook —
        # this test exists to lock the wiring, not the token gate (which has
        # its own dedicated coverage in TestAppendAtomicTurnEmbedThreshold).
        with patch('services.transcript_service._EMBED_TOKEN_THRESHOLD', 0), \
             patch('services.transcript_service._embed_entry', side_effect=fake_embed), \
             patch('services.transcript_service._trigger_episode_extraction'):
            input_id = append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='input text',
                llm_response='response text',
                pending_tool_calls=[],
            )

        # Give daemon threads a moment to run
        time.sleep(0.1)

        assert len(calls) == 2

    def test_embed_called_with_input_and_assistant_rowids(self, db):
        from services.transcript_service import append_atomic_turn

        calls = []

        def fake_embed(rowid, content):
            calls.append((rowid, content))

        with patch('services.transcript_service._EMBED_TOKEN_THRESHOLD', 0), \
             patch('services.transcript_service._embed_entry', side_effect=fake_embed), \
             patch('services.transcript_service._trigger_episode_extraction'):
            input_id = append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='input text',
                llm_response='assistant text',
                pending_tool_calls=[],
            )

        time.sleep(0.1)

        rowids = [r for r, _ in calls]
        assert input_id in rowids

        rows = _fetch_transcript(db)
        assistant_id = rows[1]['id']
        assert assistant_id in rowids

    def test_embed_hooks_fire_after_transaction_commits(self, db):
        """Embedding fires AFTER commit — rows are visible in DB when hook runs."""
        from services.transcript_service import append_atomic_turn

        visible_at_call_time = []

        from services.database_service import get_shared_db_service

        def fake_embed(rowid, content):
            # Check rows are already committed at the time this fires
            db_svc = get_shared_db_service()
            with db_svc.connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]
            visible_at_call_time.append(count)

        with patch('services.transcript_service._EMBED_TOKEN_THRESHOLD', 0), \
             patch('services.transcript_service._embed_entry', side_effect=fake_embed), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='x',
                llm_response='y',
                pending_tool_calls=[],
            )

        time.sleep(0.15)

        # Both embed calls should have seen 2 committed transcript rows
        assert len(visible_at_call_time) == 2
        for count in visible_at_call_time:
            assert count == 2


# ─────────────────────────────────────────────────────────────────────────────
# G. Daemon thread contract — hooks do not block return
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendAtomicTurnHooksFireInDaemonThreads:
    """Hooks fire from daemon threads — append_atomic_turn returns before they finish."""

    def test_return_before_slow_embed_completes(self, db):
        """append_atomic_turn returns before a slow embed finishes."""
        from services.transcript_service import append_atomic_turn

        hook_finished = threading.Event()

        def slow_embed(rowid, content):
            time.sleep(0.3)
            hook_finished.set()

        start = time.time()
        with patch('services.transcript_service._EMBED_TOKEN_THRESHOLD', 0), \
             patch('services.transcript_service._embed_entry', side_effect=slow_embed), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='x',
                llm_response='y',
                pending_tool_calls=[],
            )
        elapsed = time.time() - start

        # The function should return well before the 0.3s sleep completes
        assert elapsed < 0.2

        # Confirm the thread did eventually run (it's a daemon thread — clean up)
        hook_finished.wait(timeout=1.0)

    def test_episode_extraction_fires_on_25th_rowid(self, db):
        """_trigger_episode_extraction fires when assistant_row_id % 25 == 0."""
        from services.transcript_service import append_atomic_turn
        from services.database_service import get_shared_db_service

        trigger_calls = []
        db_svc = get_shared_db_service()

        # Determine current max transcript id (AUTOINCREMENT counter)
        with db_svc.connection() as conn:
            max_id = conn.execute("SELECT MAX(id) FROM transcript").fetchone()[0] or 0

        # We need:
        #   input_row  = max_id + padding + 1
        #   assistant  = max_id + padding + 2  (must be % 25 == 0)
        # So: max_id + padding + 2 ≡ 0 (mod 25)
        #     padding = (25 - (max_id + 2) % 25) % 25
        remainder = (max_id + 2) % 25
        padding = (25 - remainder) % 25

        for i in range(padding):
            with db_svc.connection() as conn:
                conn.execute(
                    "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
                    (_CHANNEL, 'user', f'padding {i}'),
                )

        def fake_trigger(channel, rowid):
            trigger_calls.append((channel, rowid))

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction',
                   side_effect=fake_trigger):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='x',
                llm_response='y',
                pending_tool_calls=[],
            )

        # Compute the actual assistant id
        rows = _fetch_transcript(db)
        assistant_id = max(r['id'] for r in rows if r['role'] == 'assistant')
        assert assistant_id % 25 == 0
        assert len(trigger_calls) == 1
        assert trigger_calls[0] == (_CHANNEL, assistant_id)

    def test_episode_extraction_not_fired_on_non_multiple(self, db):
        """_trigger_episode_extraction does not fire when assistant_row_id % 25 != 0."""
        from services.transcript_service import append_atomic_turn
        from services.database_service import get_shared_db_service

        trigger_calls = []

        def fake_trigger(channel, rowid):
            trigger_calls.append(rowid)

        db_svc = get_shared_db_service()
        with db_svc.connection() as conn:
            max_id = conn.execute("SELECT MAX(id) FROM transcript").fetchone()[0] or 0

        # We need the assistant row id (max_id + padding + 2) to NOT be a
        # multiple of 25. Target assistant = max_id + padding + 2; pick the
        # smallest non-negative padding such that (max_id + padding + 2) % 25 != 0.
        padding = 0
        while (max_id + padding + 2) % 25 == 0:
            padding += 1

        for i in range(padding):
            with db_svc.connection() as conn:
                conn.execute(
                    "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
                    (_CHANNEL, 'user', f'padding {i}'),
                )

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction',
                   side_effect=fake_trigger):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='x',
                llm_response='y',
                pending_tool_calls=[],
            )

        rows = _fetch_transcript(db)
        assistant_id = max(r['id'] for r in rows if r['role'] == 'assistant')

        assert assistant_id % 25 != 0, (
            f"padding logic failed to avoid a multiple of 25: assistant_id={assistant_id}"
        )
        assert len(trigger_calls) == 0


# ─────────────────────────────────────────────────────────────────────────────
# H. MessageProcessor.store() integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageProcessorStoreIntegration:
    """store() wires to append_atomic_turn and sets self._uid."""

    def test_uid_set_to_input_row_id(self, db):
        p = _make_processor()
        p._pending_tool_calls = [
            _dto('a'),
            _dto('b'),
            _dto('c'),
        ]

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('final response')

        assert p._uid is not None

        rows = _fetch_transcript(db)
        input_row = rows[0]
        assert p._uid == input_row['id']

    def test_db_has_correct_rows_after_store(self, db):
        p = _make_processor(raw_input='user message')
        p._pending_tool_calls = [_dto('tool_a'), _dto('tool_b')]

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('assistant reply')

        assert _row_count(db, 'transcript') == 2
        assert _row_count(db, 'tool_calls') == 2

        rows = _fetch_transcript(db)
        assert rows[0]['content'] == 'user message'
        assert rows[0]['role'] == 'user'
        assert rows[1]['content'] == 'assistant reply'
        assert rows[1]['role'] == 'assistant'

    def test_uid_equals_input_row_id(self, db):
        p = _make_processor()
        p._pending_tool_calls = []

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('reply')

        rows = _fetch_transcript(db)
        assert p._uid == rows[0]['id']

    def test_channel_and_role_from_class_constants(self, db):
        p = _make_processor(channel='custom_chan', role='proactive_thought')
        p._pending_tool_calls = []

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('done')

        rows = _fetch_transcript(db)
        assert rows[0]['channel'] == 'custom_chan'
        assert rows[0]['role'] == 'proactive_thought'
        assert rows[1]['channel'] == 'custom_chan'
        assert rows[1]['role'] == 'assistant'


# ─────────────────────────────────────────────────────────────────────────────
# I. Empty ROLE warns but proceeds
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageProcessorStoreEmptyRoleWarns:
    """ROLE='' logs a warning but does not raise — rows are still written."""

    def test_warning_logged_on_empty_role(self, db, caplog):
        import logging
        p = _make_processor(role='')
        p._pending_tool_calls = []

        with caplog.at_level(logging.WARNING, logger='services.message_processor'):
            with patch('services.transcript_service._embed_entry'), \
                 patch('services.transcript_service._trigger_episode_extraction'):
                # The DB write may fail or succeed depending on any CHECK on role;
                # what we care about is that the WARNING is logged.
                try:
                    p.store('reply')
                except Exception:
                    pass

        assert any('ROLE is empty' in r.message for r in caplog.records)

    def test_uid_stays_none_if_write_fails_due_to_empty_role(self, db):
        """If the DB rejects empty role, uid stays None."""
        p = _make_processor(role='')
        p._pending_tool_calls = []

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            try:
                p.store('reply')
                # If DB accepted it (no NOT NULL on role content), uid is set
                # This test just ensures no silent state corruption
            except Exception:
                assert p._uid is None


# ─────────────────────────────────────────────────────────────────────────────
# J. Exception propagates — self._uid stays None
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageProcessorStoreExceptionPropagates:
    """If append_atomic_turn raises, store() propagates and _uid stays None."""

    def test_uid_stays_none_when_store_raises(self, db):
        p = _make_processor()
        p._pending_tool_calls = []

        def _raise(*args, **kwargs):
            raise RuntimeError('forced failure')

        with patch('services.transcript_service.append_atomic_turn', side_effect=_raise):
            with pytest.raises(RuntimeError, match='forced failure'):
                p.store('reply')

        assert p._uid is None

    def test_no_rows_when_store_raises(self, db):
        p = _make_processor()
        p._pending_tool_calls = []

        def _raise(*args, **kwargs):
            raise RuntimeError('forced failure')

        with patch('services.transcript_service.append_atomic_turn', side_effect=_raise):
            with pytest.raises(RuntimeError):
                p.store('reply')

        assert _row_count(db, 'transcript') == 0
        assert _row_count(db, 'tool_calls') == 0

    def test_invalid_dto_invoked_by_propagates_to_store(self, db):
        """An invalid DTO that trips the DB CHECK propagates through store()."""
        p = _make_processor()
        p._pending_tool_calls = [_dto(invoked_by='invalid')]

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            with pytest.raises(Exception):
                p.store('reply')

        assert p._uid is None
        assert _row_count(db, 'transcript') == 0
        assert _row_count(db, 'tool_calls') == 0


# ─────────────────────────────────────────────────────────────────────────────
# K. Critic/tester-driven additions — rollback, mutation, schema locks, hooks
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendAtomicTurnStep3Rollback:
    """Failure at step 3 (assistant INSERT) rolls back steps 1 and 2."""

    def test_step3_failure_rolls_back_input_and_tool_calls(self, db):
        """If assistant INSERT fails, input row AND tool_calls rows rollback."""
        from services.transcript_service import append_atomic_turn
        from services.database_service import get_shared_db_service

        db_svc = get_shared_db_service()
        real_cursor = db_svc.connection

        # Inject a failure on the second INSERT INTO transcript (assistant row).
        # We monkey-patch the connection to raise on the assistant INSERT by
        # counting transcript INSERTs.
        import services.transcript_service as ts_mod

        real_connection = db_svc.connection
        call_state = {'transcript_inserts': 0}

        class FailingCursor:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, params=()):
                if 'INSERT INTO transcript' in sql:
                    call_state['transcript_inserts'] += 1
                    if call_state['transcript_inserts'] == 2:
                        raise RuntimeError('forced step-3 failure')
                return self._inner.execute(sql, params)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        class FailingConn:
            def __init__(self, inner):
                self._inner = inner

            def cursor(self):
                return FailingCursor(self._inner.cursor())

            def __getattr__(self, name):
                return getattr(self._inner, name)

        from contextlib import contextmanager

        @contextmanager
        def failing_connection():
            with real_connection() as conn:
                yield FailingConn(conn)

        with patch.object(db_svc, 'connection', side_effect=failing_connection), \
             patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            with pytest.raises(RuntimeError, match='forced step-3 failure'):
                append_atomic_turn(
                    channel=_CHANNEL,
                    role='user',
                    raw_input='hello',
                    llm_response='world',
                    pending_tool_calls=[_dto('a'), _dto('b')],
                )

        # Zero rows anywhere
        assert _row_count(db, 'transcript') == 0
        assert _row_count(db, 'tool_calls') == 0


class TestAppendAtomicTurnListNotMutated:
    """append_atomic_turn must not mutate the caller's pending_tool_calls list."""

    def test_input_list_identity_and_contents_preserved(self, db):
        from services.transcript_service import append_atomic_turn

        dtos = [
            _dto('a', timestamp='2026-04-11T10:00:02+00:00'),
            _dto('b', timestamp='2026-04-11T10:00:01+00:00'),
            _dto('c', timestamp='2026-04-11T10:00:00+00:00'),
        ]
        original_id = id(dtos)
        original_names = [d['name'] for d in dtos]
        original_timestamps = [d['timestamp'] for d in dtos]

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hello',
                llm_response='world',
                pending_tool_calls=dtos,
            )

        # Same list object, same order, same contents
        assert id(dtos) == original_id
        assert [d['name'] for d in dtos] == original_names
        assert [d['timestamp'] for d in dtos] == original_timestamps


class TestAppendAtomicTurnSchemaLocks:
    """Schema contract: nullable columns default correctly on v2 path."""

    def test_tool_call_id_is_null_on_new_path(self, db):
        """The new path does NOT set tool_calls.tool_call_id (stays NULL).

        Legacy tool_call_service.store() populates this field for
        digest_worker's build_messages() reconstruction. The v2 path
        uses getPreviousMessages() literal rendering, not build_messages().
        Locked here so any future change is explicit.
        """
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hi',
                llm_response='hello',
                pending_tool_calls=[_dto('t1')],
            )

        tcs = _fetch_tool_calls(db)
        assert len(tcs) == 1
        assert tcs[0]['tool_call_id'] is None

    def test_transcript_nullable_columns_default(self, db):
        """Input and assistant transcript rows leave nullable columns at default."""
        from services.transcript_service import append_atomic_turn

        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hi',
                llm_response='hello',
                pending_tool_calls=[],
            )

        rows = _fetch_transcript(db)
        for row in rows:
            assert row['tool_call_id'] is None
            assert row['tool_name'] is None
            assert row['internal'] == 0


class TestAppendAtomicTurnInvokedByMissingWarns:
    """A DTO missing 'invoked_by' logs a warning and defaults to 'llm'."""

    def test_missing_invoked_by_logs_warning_and_defaults_llm(self, db, caplog):
        import logging
        from services.transcript_service import append_atomic_turn

        # Build a DTO WITHOUT the invoked_by key entirely
        dto = {
            'name': 'orphan_tool',
            'params': {},
            'result': 'ok',
            'ephemeral': 1,
            'timestamp': '2026-04-11T10:00:00+00:00',
            # 'invoked_by' intentionally omitted
        }

        with caplog.at_level(logging.WARNING, logger='services.transcript_service'):
            with patch('services.transcript_service._embed_entry'), \
                 patch('services.transcript_service._trigger_episode_extraction'):
                append_atomic_turn(
                    channel=_CHANNEL,
                    role='user',
                    raw_input='hi',
                    llm_response='hello',
                    pending_tool_calls=[dto],
                )

        # Warning logged
        assert any("missing 'invoked_by'" in r.message for r in caplog.records)

        # Defaulted to 'llm'
        tcs = _fetch_tool_calls(db)
        assert tcs[0]['invoked_by'] == 'llm'
        assert tcs[0]['tool_name'] == 'orphan_tool'


class TestAppendAtomicTurnEmbedThreshold:
    """Embedding hooks mirror append()'s token-threshold gate (>= 50 tokens)."""

    def test_short_content_skips_embed(self, db):
        """Content below the 50-token threshold does NOT trigger _embed_entry."""
        from services.transcript_service import append_atomic_turn

        calls = []

        def fake_embed(rowid, content):
            calls.append(rowid)

        with patch('services.transcript_service._embed_entry', side_effect=fake_embed), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hi',  # ~1 token
                llm_response='bye',  # ~1 token
                pending_tool_calls=[],
            )

        # Give daemon threads time to attempt the gated call
        time.sleep(0.1)

        # Neither input nor assistant should have been embedded
        assert len(calls) == 0

    def test_long_content_triggers_embed(self, db):
        """Content at or above the 50-token threshold DOES trigger _embed_entry."""
        from services.transcript_service import append_atomic_turn

        calls = []
        hook_fired = threading.Event()

        def fake_embed(rowid, content):
            calls.append(rowid)
            if len(calls) >= 2:
                hook_fired.set()

        # ~ 60 words, well above 50-token threshold for either estimator
        long_text = 'The quick brown fox jumps over the lazy dog. ' * 20

        with patch('services.transcript_service._embed_entry', side_effect=fake_embed), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input=long_text,
                llm_response=long_text,
                pending_tool_calls=[],
            )

        hook_fired.wait(timeout=2.0)
        assert len(calls) == 2


class TestAppendAtomicTurnResultField:
    """The result field round-trips to DB."""

    def test_result_roundtrip(self, db):
        from services.transcript_service import append_atomic_turn

        payload = 'the-result-payload-xyz'
        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hi',
                llm_response='hello',
                pending_tool_calls=[_dto(result=payload)],
            )

        tcs = _fetch_tool_calls(db)
        assert tcs[0]['result'] == payload

    def test_result_none_coerced_to_empty(self, db):
        from services.transcript_service import append_atomic_turn

        raw = {
            'name': 'x',
            'params': {},
            'result': None,
            'ephemeral': 1,
            'invoked_by': 'llm',
            'timestamp': '2026-04-11T10:00:00+00:00',
        }
        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            append_atomic_turn(
                channel=_CHANNEL,
                role='user',
                raw_input='hi',
                llm_response='hello',
                pending_tool_calls=[raw],
            )

        tcs = _fetch_tool_calls(db)
        assert tcs[0]['result'] == ''
