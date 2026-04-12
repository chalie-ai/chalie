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
from unittest.mock import patch

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
    from services.system_message_prompt import SystemMessagePrompt

    class _StubPrompt(SystemMessagePrompt):
        _SYSTEM_PROMPT = ''

    class _Fake(MessageProcessor):
        CHANNEL = channel
        ROLE = role
        SYSTEM_PROMPT_CLASS = _StubPrompt

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
            append_atomic_turn(
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
            append_atomic_turn(
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

        def _raise(*_args, **_kwargs):
            raise RuntimeError('forced failure')

        with patch('services.transcript_service.append_atomic_turn', side_effect=_raise):
            with pytest.raises(RuntimeError, match='forced failure'):
                p.store('reply')

        assert p._uid is None

    def test_no_rows_when_store_raises(self, db):
        p = _make_processor()
        p._pending_tool_calls = []

        def _raise(*_args, **_kwargs):
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

        # Inject a failure on the second INSERT INTO transcript (assistant row).
        # We monkey-patch the connection to raise on the assistant INSERT by
        # counting transcript INSERTs.
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


# =============================================================================
# RESCUED: getPreviousMessages() tests (from test_message_processor_v2_base.py)
# Only tests that use the real `db` fixture are included.
# =============================================================================

import logging as _logging  # noqa: E402  (appended block — already imported via std path)

_GPM_USER_DEF = "The user is a human named Alice interacting via the chat interface."
_GPM_USER_PROMPT = "What time is it?"
_GPM_CHANNEL = 'test_channel'


def _stub_gpm_prompt_cls():
    from services.system_message_prompt import SystemMessagePrompt

    class _StubPrompt(SystemMessagePrompt):
        _SYSTEM_PROMPT = ''

    return _StubPrompt


class _GPMFakeProcessor:
    """Concrete MessageProcessor subclass for getPreviousMessages() tests."""

    _CHANNEL = _GPM_CHANNEL
    _ROLE = 'test_role'

    @staticmethod
    def make(**kwargs):
        from services.message_processor import MessageProcessor

        class _Fake(MessageProcessor):
            CHANNEL = _GPMFakeProcessor._CHANNEL
            ROLE = _GPMFakeProcessor._ROLE
            SYSTEM_PROMPT_CLASS = _stub_gpm_prompt_cls()

            def getUserDefinition(self) -> str:
                return _GPM_USER_DEF

            def getUserPrompt(self) -> str:
                return _GPM_USER_PROMPT

        for k, v in kwargs.items():
            setattr(_Fake, k, v)

        return _Fake('test raw input', {'key': 'value'})

    @staticmethod
    def cls():
        from services.message_processor import MessageProcessor

        class _Fake(MessageProcessor):
            CHANNEL = _GPMFakeProcessor._CHANNEL
            ROLE = _GPMFakeProcessor._ROLE
            SYSTEM_PROMPT_CLASS = _stub_gpm_prompt_cls()

            def getUserDefinition(self) -> str:
                return _GPM_USER_DEF

            def getUserPrompt(self) -> str:
                return _GPM_USER_PROMPT

        return _Fake


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — empty channel → ''
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesEmpty:
    def test_empty_channel_returns_empty_string(self, db):
        """No transcript rows and no compaction → empty string."""
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert result == ''

    def test_returns_str_type(self, db):
        p = _GPMFakeProcessor.make()
        assert isinstance(p.getPreviousMessages(), str)


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — transcript rows + ephemeral filtering
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesTranscript:
    def _seed_transcript(self, db, channel=_GPM_CHANNEL):
        """Insert two transcript rows and one tool_calls row each."""
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Hello world', '2026-04-10 10:00:00')",
            (channel,)
        )
        uid1 = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'assistant', 'Hi there', '2026-04-10 10:01:00')",
            (channel,)
        )
        uid2 = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Durable (ephemeral=0) tool_call linked to uid1 — MUST appear
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, "
            "invoked_by, ephemeral, created_at) "
            "VALUES (?, 'memory', '{}', 'User likes dark mode', 'system', 0, '2026-04-10 10:00:30')",
            (uid1,)
        )

        # Ephemeral (ephemeral=1) tool_call linked to uid2 — must NOT appear
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, "
            "invoked_by, ephemeral, created_at) "
            "VALUES (?, 'read', '{}', 'Web page content', 'llm', 1, '2026-04-10 10:01:30')",
            (uid2,)
        )

        db.commit()
        return uid1, uid2

    def test_user_row_appears(self, db):
        self._seed_transcript(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'user: Hello world' in result
        assert 'USER: Hello world' not in result

    def test_assistant_row_appears(self, db):
        self._seed_transcript(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Assistant: Hi there' in result
        assert 'ASSISTANT: Hi there' not in result

    def test_durable_tool_call_appears(self, db):
        self._seed_transcript(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert '[memory()] User likes dark mode' in result
        assert 'TOOL(memory)' not in result

    def test_ephemeral_tool_call_does_not_appear(self, db):
        self._seed_transcript(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Web page content' not in result

    def test_timestamp_format(self, db):
        self._seed_transcript(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert '[2026-04-10 10:00]' in result

    def test_rows_ordered_oldest_first(self, db):
        self._seed_transcript(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        user_pos = result.index('user: Hello world')
        asst_pos = result.index('Assistant: Hi there')
        assert user_pos < asst_pos

    def test_other_channel_rows_excluded(self, db):
        """Rows from a different channel must not appear."""
        self._seed_transcript(db, channel=_GPM_CHANNEL)
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('other_channel', 'user', 'Other channel content', '2026-04-10 10:00:00')"
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Other channel content' not in result


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — compaction row causes prepend + watermark
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesCompaction:
    def _seed_with_compaction(self, db, channel=_GPM_CHANNEL):
        """Insert three transcript rows; set compaction watermark at first."""
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Old message before compaction', '2026-04-09 09:00:00')",
            (channel,)
        )
        wm_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'New message after compaction', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'assistant', 'New reply after compaction', '2026-04-10 10:01:00')",
            (channel,)
        )

        db.execute(
            "INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, "
            "token_count, updated_at) "
            "VALUES (?, 'COMPACTED: previous context here', ?, 50, '2026-04-10 09:30:00')",
            (channel, wm_id)
        )
        db.commit()
        return wm_id

    def test_compacted_text_prepended(self, db):
        self._seed_with_compaction(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert result.startswith('COMPACTED: previous context here')

    def test_pre_watermark_rows_not_re_rendered(self, db):
        self._seed_with_compaction(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Old message before compaction' not in result

    def test_post_watermark_rows_appear(self, db):
        self._seed_with_compaction(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'New message after compaction' in result
        assert 'New reply after compaction' in result

    def test_compacted_text_before_new_rows(self, db):
        self._seed_with_compaction(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        compacted_pos = result.index('COMPACTED:')
        new_msg_pos = result.index('New message after compaction')
        assert compacted_pos < new_msg_pos

    def test_token_budget_ignored_in_commit2(self, db):
        """token_budget is accepted but silently ignored."""
        self._seed_with_compaction(db)
        p = _GPMFakeProcessor.make()
        result_no_budget = p.getPreviousMessages()
        result_with_budget = p.getPreviousMessages(token_budget=100)
        assert result_no_budget == result_with_budget


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — send() + store() wiring (db-only stubs)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesSendStoreWiring:
    """send() and store() are wired — confirm they operate against the real DB."""

    def test_send_is_callable(self, db):
        """send() is wired — no longer raises NotImplementedError."""
        from services.llm_service import LLMResponse
        p = _GPMFakeProcessor.make()
        fake_response = LLMResponse(text='ok', model='m', provider='p', tool_calls=None)
        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.return_value = fake_response
            result = p.send()
        assert isinstance(result, str)

    def test_store_is_callable(self, db):
        """store() is wired — calls DB, no longer raises NotImplementedError."""
        p = _GPMFakeProcessor.make()
        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('final response')
        assert p._uid is not None

    def test_send_is_wired_in_commit_6(self, db):
        """send() calls provider, does not raise NotImplementedError."""
        from services.llm_service import LLMResponse
        p = _GPMFakeProcessor.make()
        fake_response = LLMResponse(text='done', model='m', provider='p', tool_calls=None)
        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.return_value = fake_response
            result = p.send()
        assert result == 'done'

    def test_store_is_wired_in_commit_4(self, db):
        """store() calls DB, does not raise NotImplementedError."""
        p = _GPMFakeProcessor.make()
        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('final response')
        assert p._uid is not None

    def test_send_accepts_request_id(self, db):
        """send() accepts an optional request_id parameter."""
        from services.llm_service import LLMResponse
        p = _GPMFakeProcessor.make()
        fake_response = LLMResponse(text='ok', model='m', provider='p', tool_calls=None)
        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.return_value = fake_response
            result = p.send('req-id')
        assert isinstance(result, str)

    def test_store_not_caught_as_plain_exception(self, db):
        """store() calls DB, no longer raises NotImplementedError."""
        p = _GPMFakeProcessor.make()
        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('response text')
        assert p._uid is not None


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — compaction only, no new rows above watermark
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesCompactionOnlyNoNewRows:
    """Watermark is at or above all transcript rows → output is only compacted_text."""

    def test_returns_compacted_text_alone(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Old message', '2026-04-09 09:00:00')",
            (channel,)
        )
        old_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        db.execute(
            "INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, 'COMPACTED: all prior context', ?, 10, '2026-04-09 10:00:00')",
            (channel, old_id)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()

        assert result == 'COMPACTED: all prior context'
        assert 'Old message' not in result

    def test_no_trailing_newline_when_only_compaction(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Seed', '2026-04-09 09:00:00')",
            (channel,)
        )
        seed_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        db.execute(
            "INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, 'COMPACT BLOCK', ?, 5, '2026-04-09 10:00:00')",
            (channel, seed_id)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert result == 'COMPACT BLOCK'


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — compaction + one new transcript row above watermark
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesCompactionPlusOneRow:
    def test_compaction_followed_by_new_row(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Pre-compaction message', '2026-04-09 08:00:00')",
            (channel,)
        )
        watermark_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Post-compaction message', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        db.execute(
            "INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, 'COMPACTION SUMMARY', ?, 20, '2026-04-09 09:00:00')",
            (channel, watermark_id)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()

        assert result.startswith('COMPACTION SUMMARY')
        assert 'Post-compaction message' in result
        assert 'Pre-compaction message' not in result

    def test_compaction_before_new_row_in_output(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Seed row', '2026-04-09 08:00:00')",
            (channel,)
        )
        wm = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'assistant', 'Fresh reply', '2026-04-10 09:00:00')",
            (channel,)
        )
        db.commit()

        db.execute(
            "INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, 'COMPACT_START', ?, 5, '2026-04-09 09:00:00')",
            (channel, wm)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        compact_pos = result.index('COMPACT_START')
        new_row_pos = result.index('Fresh reply')
        assert compact_pos < new_row_pos


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — multiple channels, no cross-channel leakage
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesChannelIsolation:
    """Each processor only sees its own CHANNEL's rows."""

    def _seed_all_channels(self, db):
        for channel in ('user', 'dmn', 'goal_pursuit', 'scheduled', _GPM_CHANNEL):
            db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES (?, 'user', ?, '2026-04-10 10:00:00')",
                (channel, f"Content from {channel}")
            )
        db.commit()

    def test_test_channel_does_not_see_user_channel(self, db):
        self._seed_all_channels(db)
        p = _GPMFakeProcessor.make()  # CHANNEL='test_channel'
        result = p.getPreviousMessages()
        assert 'Content from user' not in result
        assert f'Content from {_GPM_CHANNEL}' in result

    def test_test_channel_does_not_see_dmn_channel(self, db):
        self._seed_all_channels(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Content from dmn' not in result

    def test_test_channel_does_not_see_goal_pursuit_channel(self, db):
        self._seed_all_channels(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Content from goal_pursuit' not in result

    def test_each_channel_sees_only_its_own_rows(self, db):
        """Build four processors with different CHANNELs and assert no bleed."""
        self._seed_all_channels(db)
        from services.message_processor import MessageProcessor

        for channel in ('user', 'dmn', 'goal_pursuit', 'scheduled'):
            class _Chan(MessageProcessor):
                CHANNEL = channel
                ROLE = 'user'
                SYSTEM_PROMPT_CLASS = _stub_gpm_prompt_cls()

                def getUserDefinition(self):
                    return _GPM_USER_DEF

                def getUserPrompt(self):
                    return _GPM_USER_PROMPT

            p = _Chan('input')
            result = p.getPreviousMessages()
            assert f'Content from {channel}' in result, (
                f"Channel {channel!r} should see its own content"
            )
            for other in ('user', 'dmn', 'goal_pursuit', 'scheduled', _GPM_CHANNEL):
                if other != channel:
                    assert f'Content from {other}' not in result, (
                        f"Channel {channel!r} leaked content from {other!r}"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — durable tool_call interleaving order
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesDurableToolInterleaving:
    """Two durable tool_calls linked to the same transcript row must appear
    immediately after that row, ordered by created_at."""

    def test_two_durable_tool_calls_appear_under_their_transcript_row(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'The input row', '2026-04-10 10:00:00')",
            (channel,)
        )
        tid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, invoked_by, ephemeral, created_at) "
            "VALUES (?, 'memory', '{}', 'Result from memory', 'system', 0, '2026-04-10 10:00:10')",
            (tid,)
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, invoked_by, ephemeral, created_at) "
            "VALUES (?, 'find_tools', '{}', 'Found weather tool', 'llm', 0, '2026-04-10 10:00:20')",
            (tid,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()

        assert '[memory()] Result from memory' in result
        assert '[find_tools()] Found weather tool' in result
        assert 'TOOL(memory)' not in result
        assert 'TOOL(find_tools)' not in result

        input_pos = result.index('The input row')
        memory_pos = result.index('[memory()] Result from memory')
        find_pos = result.index('[find_tools()] Found weather tool')
        assert input_pos < memory_pos < find_pos

    def test_tool_calls_ordered_by_timestamp(self, db):
        """Earlier created_at → appears first in output."""
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Input', '2026-04-10 10:00:00')",
            (channel,)
        )
        tid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, invoked_by, ephemeral, created_at) "
            "VALUES (?, 'second_tool', '{}', 'Second result', 'llm', 0, '2026-04-10 10:00:30')",
            (tid,)
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, invoked_by, ephemeral, created_at) "
            "VALUES (?, 'first_tool', '{}', 'First result', 'llm', 0, '2026-04-10 10:00:05')",
            (tid,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()

        first_pos = result.index('[first_tool()] First result')
        second_pos = result.index('[second_tool()] Second result')
        assert first_pos < second_pos


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — ephemeral=1 for same transcript row is dropped
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesEphemeralSiblingDropped:
    def test_ephemeral_sibling_dropped_when_durable_present(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Row with mixed tool_calls', '2026-04-10 10:00:00')",
            (channel,)
        )
        tid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, invoked_by, ephemeral, created_at) "
            "VALUES (?, 'memory', '{}', 'DURABLE_RESULT', 'system', 0, '2026-04-10 10:00:05')",
            (tid,)
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, invoked_by, ephemeral, created_at) "
            "VALUES (?, 'read', '{}', 'EPHEMERAL_RESULT', 'llm', 1, '2026-04-10 10:00:10')",
            (tid,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()

        assert 'DURABLE_RESULT' in result
        assert 'EPHEMERAL_RESULT' not in result


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — exact timestamp format regression
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesTimestampFormat:
    def test_iso_timestamp_with_offset_formatted_correctly(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Timestamp test', '2026-04-10T14:03:27+00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert '[2026-04-10 14:03]' in result

    def test_no_seconds_in_timestamp(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Seconds check', '2026-04-10T14:03:27+00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert '14:03:27' not in result

    def test_naive_sqlite_timestamp_handled(self, db):
        """Legacy rows use SQLite's naive format; must not crash."""
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Legacy naive timestamp', '2026-04-10 14:03:27')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Legacy naive timestamp' in result
        assert '[2026-04-10 14:03]' in result


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — role rendering per north star
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesRoleCase:
    """Lock the role rendering convention: lowercase for inputs, title-case for
    assistant only."""

    def test_user_role_rendered_lowercase(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Hello', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'user: Hello' in result
        assert 'USER: Hello' not in result

    def test_assistant_role_rendered_titlecase(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'assistant', 'Hi there', '2026-04-10 10:01:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Assistant: Hi there' in result
        assert 'ASSISTANT: Hi there' not in result
        assert 'assistant: Hi there' not in result

    def test_arbitrary_role_rendered_lowercase(self, db):
        """Any non-assistant role string stays lowercase."""
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'proactive_thought', 'Background thought', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'proactive_thought: Background thought' in result
        assert 'PROACTIVE_THOUGHT: Background thought' not in result

    def test_goal_pursuit_role_rendered_lowercase(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'goal_pursuit', 'Pursuit tick', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'goal_pursuit: Pursuit tick' in result

    def test_scheduled_role_rendered_lowercase(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'scheduled', 'Timer fired', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'scheduled: Timer fired' in result


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — token_budget parameter accepted and ignored
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesTokenBudget:
    """token_budget is accepted without error; output is identical regardless."""

    def test_none_and_integer_budget_produce_identical_output(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Budget test', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        without_budget = p.getPreviousMessages()
        with_budget = p.getPreviousMessages(token_budget=100)
        assert without_budget == with_budget

    def test_large_budget_does_not_change_output(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Large budget test', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result_no_budget = p.getPreviousMessages()
        result_large_budget = p.getPreviousMessages(token_budget=999999)
        assert result_no_budget == result_large_budget


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — empty channel returns '' even when other channels
# have data
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesEmptyChannelWithOtherData:
    def test_returns_empty_when_no_rows_for_this_channel(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('other_channel', 'user', 'Not mine', '2026-04-10 10:00:00')"
        )
        db.commit()

        p = _GPMFakeProcessor.make()  # CHANNEL='test_channel'
        result = p.getPreviousMessages()
        assert result == ''


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — empty content row does not raise
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesEmptyContent:
    def test_empty_content_row_rendered_without_crash(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', '', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'user: ' in result
        assert 'USER: ' not in result


# ─────────────────────────────────────────────────────────────────────────────
# getPreviousMessages() — explicit since_id=0 path
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesSinceIdZero:
    def test_no_compaction_returns_all_rows_including_first(self, db):
        """With no compaction, even the very first row must surface."""
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'First ever message', '2026-04-08 09:00:00')",
            (channel,)
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'assistant', 'First reply', '2026-04-08 09:01:00')",
            (channel,)
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Second message', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()

        assert 'First ever message' in result
        assert 'First reply' in result
        assert 'Second message' in result

    def test_since_id_zero_passed_when_no_compaction(self, db):
        """Watermark is exactly 0 (not None) when no compaction exists."""
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Lock the watermark', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        captured_kwargs = {}

        from services import transcript_service as _tsvc
        real_get_recent = _tsvc.get_recent

        def spy(channel_arg, **kwargs):
            captured_kwargs.update(kwargs)
            return real_get_recent(channel_arg, **kwargs)

        with patch('services.transcript_service.get_recent', side_effect=spy):
            p = _GPMFakeProcessor.make()
            p.getPreviousMessages()

        assert captured_kwargs.get('since_id') == 0, (
            f"Expected since_id=0 when no compaction, got {captured_kwargs.get('since_id')!r}"
        )
        from services.message_processor import MessageProcessor
        assert captured_kwargs.get('limit') == MessageProcessor._TRANSCRIPT_FETCH_LIMIT


# =============================================================================
# RESCUED: compaction-specific db tests
# (from test_message_processor_v2_compaction.py — B, D8-D10, E groups)
# =============================================================================

_COMPACT_CHANNEL = 'test_compact_channel'
_COMPACT_USER_DEF = "The user is 'test' — compaction unit tests."


def _make_compact_llm_response(text='summary text', tool_calls=None):
    from services.llm_service import LLMResponse
    return LLMResponse(
        text=text,
        model='test-model',
        provider='mock',
        tool_calls=tool_calls,
    )


def _make_compact_processor(channel=_COMPACT_CHANNEL, role='user', raw_input='hello', **kwargs):
    from services.message_processor import MessageProcessor
    from services.system_message_prompt import SystemMessagePrompt

    class _StubPrompt(SystemMessagePrompt):
        _SYSTEM_PROMPT = ''

    _prompt = raw_input

    class FakeCompactingProcessor(MessageProcessor):
        CHANNEL = channel
        ROLE = role
        SYSTEM_PROMPT_CLASS = _StubPrompt

        def getUserDefinition(self) -> str:
            return _COMPACT_USER_DEF

        def getUserPrompt(self) -> str:
            return _prompt

    for k, v in kwargs.items():
        setattr(FakeCompactingProcessor, k, v)

    return FakeCompactingProcessor(raw_input)


def _compact_seed_transcript_row(db, channel, role='user', content='test content'):
    """Insert a transcript row and return its id (needed for FK in compactions)."""
    cursor = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, role, content),
    )
    db.commit()
    return cursor.lastrowid


def _compact_seed_compaction(db, channel, compacted_text, compacted_up_to_id=None,
                             token_count=100, overflow_content=None):
    """Insert a compactions row for test setup."""
    if compacted_up_to_id is None:
        compacted_up_to_id = _compact_seed_transcript_row(db, channel)

    db.execute(
        """
        INSERT INTO compactions
            (channel, compacted_text, compacted_up_to_id, token_count, updated_at, overflow_content)
        VALUES (?, ?, ?, ?, '2026-04-11T10:00:00+00:00', ?)
        """,
        (channel, compacted_text, compacted_up_to_id, token_count, overflow_content),
    )
    db.commit()


def _compact_get_compaction_row(db, channel):
    cursor = db.execute(
        "SELECT compacted_text, compacted_up_to_id, token_count, updated_at, overflow_content "
        "FROM compactions WHERE channel = ?",
        (channel,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        'compacted_text': row[0],
        'compacted_up_to_id': row[1],
        'token_count': row[2],
        'updated_at': row[3],
        'overflow_content': row[4],
    }


def _compact_dto(
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


# ─────────────────────────────────────────────────────────────────────────────
# _wrap_with_checkpoint — real DB tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWrapWithCheckpoint:
    """Module-private _wrap_with_checkpoint function behavior against real DB."""

    def test_no_compaction_row_returns_bare_body(self, db):
        from services.message_processor import _wrap_with_checkpoint
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'hello user')
        assert result == 'hello user'

    def test_row_with_content_wraps_with_checkpoint_header(self, db):
        from services.message_processor import _wrap_with_checkpoint
        _compact_seed_compaction(db, _COMPACT_CHANNEL, 'Previously: we talked about movies.')
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'user: What is next?')
        assert result.startswith(
            '### Checkpoint - What you were previously discussing / doing\n'
        )
        assert 'Previously: we talked about movies.' in result
        assert "### Current State - What's happening in the current turn\n" in result
        assert result.endswith('user: What is next?')

    def test_row_with_content_exact_envelope_format(self, db):
        from services.message_processor import _wrap_with_checkpoint
        _compact_seed_compaction(db, _COMPACT_CHANNEL, 'checkpoint content')
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'current body')
        expected = (
            "### Checkpoint - What you were previously discussing / doing\n"
            "checkpoint content\n"
            "\n"
            "---\n"
            "### Current State - What's happening in the current turn\n"
            "current body"
        )
        assert result == expected

    def test_row_with_empty_compacted_text_returns_bare_body(self, db):
        from services.message_processor import _wrap_with_checkpoint
        _compact_seed_compaction(db, _COMPACT_CHANNEL, '')
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'body text')
        assert result == 'body text'

    def test_row_with_whitespace_only_compacted_text_returns_bare_body(self, db):
        from services.message_processor import _wrap_with_checkpoint
        _compact_seed_compaction(db, _COMPACT_CHANNEL, '   \n\t  ')
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'body text')
        assert result == 'body text'


# ─────────────────────────────────────────────────────────────────────────────
# _run_stage2_act_restart — DB-writing tests (d8, d9, d10)
# ─────────────────────────────────────────────────────────────────────────────


class TestStage2ActRestartDbWrites:
    """Stage 2 compaction writes and updates real compactions table rows."""

    def test_d8_stage2_writes_fresh_compaction_row_when_none_existed(self, db):
        """When no prior compactions row exists, Stage 2 writes a fresh one."""
        t1_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hello')
        t2_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'assistant', 'hi')

        p = _make_compact_processor()
        p._act_trail = []
        p._pending_tool_calls = []

        llm_resp = _make_compact_llm_response(text='fresh compaction summary')

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_service.get_entries_since', return_value=[
                 {'id': t1_id, 'role': 'user', 'content': 'hello', 'tool_name': None},
                 {'id': t2_id, 'role': 'assistant', 'content': 'hi', 'tool_name': None},
             ]):
            mock_inst.return_value.send_messages.return_value = llm_resp
            result = p._run_stage2_act_restart()

        assert result is True
        row = _compact_get_compaction_row(db, _COMPACT_CHANNEL)
        assert row is not None
        assert 'fresh compaction summary' in row['compacted_text']

    def test_d9_stage2_updates_existing_compaction_row(self, db):
        """When a compactions row exists, Stage 2 updates it in-place via UPSERT."""
        old_t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'old turn')
        _compact_seed_compaction(db, _COMPACT_CHANNEL, 'old summary',
                                 compacted_up_to_id=old_t_id)

        new_t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'new turn')

        p = _make_compact_processor()
        p._act_trail = []
        p._pending_tool_calls = []

        llm_resp = _make_compact_llm_response(text='updated compaction summary')

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_service.get_entries_since', return_value=[
                 {'id': new_t_id, 'role': 'user', 'content': 'new turn', 'tool_name': None},
             ]):
            mock_inst.return_value.send_messages.return_value = llm_resp
            p._run_stage2_act_restart()

        new_row = _compact_get_compaction_row(db, _COMPACT_CHANNEL)
        assert new_row is not None
        assert new_row['compacted_text'] == 'updated compaction summary'

    def test_d10_upsert_does_not_touch_overflow_content(self, db):
        """Stage 2's UPSERT does NOT overwrite overflow_content (legacy field)."""
        legacy_overflow = 'legacy overflow data'
        old_t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'old turn')
        _compact_seed_compaction(db, _COMPACT_CHANNEL, 'old summary',
                                 compacted_up_to_id=old_t_id,
                                 overflow_content=legacy_overflow)

        new_t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'new turn')

        p = _make_compact_processor()
        p._act_trail = []
        p._pending_tool_calls = []

        llm_resp = _make_compact_llm_response(text='new compaction text')

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_service.get_entries_since', return_value=[
                 {'id': new_t_id, 'role': 'user', 'content': 'turn', 'tool_name': None},
             ]):
            mock_inst.return_value.send_messages.return_value = llm_resp
            p._run_stage2_act_restart()

        row = _compact_get_compaction_row(db, _COMPACT_CHANNEL)
        assert row['overflow_content'] == legacy_overflow


# ─────────────────────────────────────────────────────────────────────────────
# _run_full_compaction — direct DB tests (e1–e6)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunFullCompaction:
    """Direct tests for _run_full_compaction against real DB."""

    def test_e1_no_entries_and_no_prior_checkpoint_skips_llm(self, db, caplog):
        """With no entries AND no prior checkpoint, _run_full_compaction returns
        None WITHOUT calling the LLM and without writing to the compactions table."""
        p = _make_compact_processor()

        with caplog.at_level(_logging.WARNING):
            with patch('services.providers.Providers.instance') as mock_inst, \
                 patch('services.compaction_service.get_entries_since', return_value=[]), \
                 patch('services.compaction_service.get_compaction', return_value=None):
                result = p._run_full_compaction()
                mock_inst.return_value.send_messages.assert_not_called()

        assert result is None
        row = _compact_get_compaction_row(db, _COMPACT_CHANNEL)
        assert row is None
        assert any(
            'no entries' in rec.message and 'skipping LLM call' in rec.message
            for rec in caplog.records
        )

    def test_e2_llm_returns_empty_text_returns_none(self, db):
        """When LLM returns empty text, _run_full_compaction returns None without DB write."""
        t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hi')
        p = _make_compact_processor()
        llm_resp = _make_compact_llm_response(text='')

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_service.get_entries_since', return_value=[
                 {'id': t_id, 'role': 'user', 'content': 'hi', 'tool_name': None},
             ]):
            mock_inst.return_value.send_messages.return_value = llm_resp
            result = p._run_full_compaction()

        assert result is None
        row = _compact_get_compaction_row(db, _COMPACT_CHANNEL)
        assert row is None

    def test_e3_llm_raises_returns_none(self, db, caplog):
        """When LLM call raises, _run_full_compaction returns None without DB write."""
        t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hi')
        p = _make_compact_processor()

        with caplog.at_level(_logging.ERROR):
            with patch('services.providers.Providers.instance') as mock_inst, \
                 patch('services.compaction_service.get_entries_since', return_value=[
                     {'id': t_id, 'role': 'user', 'content': 'hi', 'tool_name': None},
                 ]):
                mock_inst.return_value.send_messages.side_effect = RuntimeError('LLM down')
                result = p._run_full_compaction()

        assert result is None
        row = _compact_get_compaction_row(db, _COMPACT_CHANNEL)
        assert row is None

    def test_e4_happy_path_appends_compaction_dto_writes_row_returns_text(self, db):
        """Happy path: appends compaction DTO, writes DB row, returns summary text."""
        t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hello')
        p = _make_compact_processor()
        llm_resp = _make_compact_llm_response(text='happy path summary')

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_service.get_entries_since', return_value=[
                 {'id': t_id, 'role': 'user', 'content': 'hello', 'tool_name': None},
             ]):
            mock_inst.return_value.send_messages.return_value = llm_resp
            result = p._run_full_compaction()

        assert result == 'happy path summary'

        compaction_dtos = [d for d in p._pending_tool_calls if d['name'] == 'compaction']
        assert len(compaction_dtos) == 1
        c = compaction_dtos[0]
        assert c['ephemeral'] == 0
        assert c['invoked_by'] == 'system'
        assert c['result'] == 'happy path summary'
        assert c['params'] == {}

        row = _compact_get_compaction_row(db, _COMPACT_CHANNEL)
        assert row is not None
        assert row['compacted_text'] == 'happy path summary'

    def test_e5_watermark_set_to_max_entry_id(self, db):
        """Watermark is set to max(entry['id']) from the entries list."""
        id3 = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'a')
        id5 = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'c')
        id7 = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'assistant', 'b')

        p = _make_compact_processor()
        entries = [
            {'id': id3, 'role': 'user', 'content': 'a', 'tool_name': None},
            {'id': id7, 'role': 'assistant', 'content': 'b', 'tool_name': None},
            {'id': id5, 'role': 'user', 'content': 'c', 'tool_name': None},
        ]
        llm_resp = _make_compact_llm_response(text='compacted')

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_service.get_entries_since', return_value=entries):
            mock_inst.return_value.send_messages.return_value = llm_resp
            p._run_full_compaction()

        row = _compact_get_compaction_row(db, _COMPACT_CHANNEL)
        assert row['compacted_up_to_id'] == id7

    def test_e6_upsert_uses_self_job_for_llm_call(self, db):
        """LLM call uses job=self.JOB (not a hardcoded legacy job string)."""
        t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hi')

        class CustomJobProcessor(_make_compact_processor().__class__):
            JOB = 'custom-job-name'

        p = CustomJobProcessor('hi')

        captured_jobs = []
        llm_resp = _make_compact_llm_response(text='compacted')

        def fake_send(_system_prompt, _messages, job=None, **_kw):  # noqa: ARG001
            captured_jobs.append(job)
            return llm_resp

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_service.get_entries_since', return_value=[
                 {'id': t_id, 'role': 'user', 'content': 'hi', 'tool_name': None},
             ]):
            mock_inst.return_value.send_messages.side_effect = fake_send
            p._run_full_compaction()

        assert 'custom-job-name' in captured_jobs
