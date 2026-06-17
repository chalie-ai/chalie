# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for get_previous_messages() and the _wrap_with_checkpoint envelope.

Uses real in-memory SQLite DB — no mocks for the data path. The watermark /
overflow / compaction-write behaviour is covered by the feature tests in
test_compaction_watermark.py and the end-to-end scenario suite; the old singleton-mocked
_run_full_compaction / _handle_overflow tests were removed with the redesign.
"""

import pytest

pytestmark = pytest.mark.unit

# =============================================================================
# RESCUED: get_previous_messages() tests (from test_message_processor_v2_base.py)
# Only tests that use the real `db` fixture are included.
# =============================================================================


_GPM_USER_DEF = "The user is a human named Alice interacting via the chat interface."
_GPM_USER_PROMPT = "What time is it?"
_GPM_CHANNEL = 'test_channel'


def _gpm_config(channel=_GPM_CHANNEL, role='test_role', suppress_history=False):
    from services.processor_config import ProcessorConfig
    from tests.helpers import StubProcessorConfig

    return StubProcessorConfig(
        channel=channel,
        role=role,
        policy_channel=ProcessorConfig.PolicyChannel.CHAT,
        build_user_prompt=lambda _mp: _GPM_USER_PROMPT,
        build_user_definition=lambda _mp: _GPM_USER_DEF,
        build_system_prompt=lambda _mp: '',
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=5,
        skip_transcript=False,
        skip_input_row=False,
        suppress_history=suppress_history,
        broadcast_to=None,
        memory_seed=False,
    )


class _GPMFakeProcessor:
    _CHANNEL = _GPM_CHANNEL
    _ROLE = 'test_role'

    @staticmethod
    def make(channel=_GPM_CHANNEL, suppress_history=False, **kwargs):
        from services.message_processor import MessageProcessor

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, 'test raw input', {'key': 'value'})
        mp.config = _gpm_config(channel=channel, suppress_history=suppress_history)
        mp.uid = None
        for k, v in kwargs.items():
            setattr(mp, k, v)
        return mp


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — transcript rows (tool_calls excluded from history)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesTranscript:
    def _seed_transcript(self, db, channel=_GPM_CHANNEL):
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

        # tool_calls rows are now all durable (no ephemeral column)
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, created_at) "
            "VALUES (?, 'memory', '{}', 'User likes dark mode', '2026-04-10 10:00:30')",
            (uid1,)
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, created_at) "
            "VALUES (?, 'read', '{}', 'Web page content', '2026-04-10 10:01:30')",
            (uid2,)
        )

        db.commit()
        return uid1, uid2

    def test_tool_calls_do_not_appear_in_previous_messages(self, db):
        """History replay no longer renders tool_calls rows (decision 2).
        Tool calls are retained durably but excluded from get_previous_messages()
        to keep the history prompt clean and unambiguous."""
        self._seed_transcript(db)
        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()
        # transcript rows render (user/assistant content)
        assert 'Hello world' in result
        assert 'Hi there' in result
        # tool_calls are NOT injected into the history prompt
        assert 'User likes dark mode' not in result
        assert 'Web page content' not in result


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — multiple channels, no cross-channel leakage
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesChannelIsolation:
    def _seed_all_channels(self, db):
        for channel in ('user', 'dmn', 'subagent', 'scheduled', _GPM_CHANNEL):
            db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES (?, 'user', ?, '2026-04-10 10:00:00')",
                (channel, f"Content from {channel}")
            )
        db.commit()

    def test_test_channel_isolation(self, db):
        self._seed_all_channels(db)
        p = _GPMFakeProcessor.make()  # uses test_channel as its CHANNEL
        result = p.get_previous_messages()
        assert f'Content from {_GPM_CHANNEL}' in result
        for other in ('user', 'dmn', 'subagent', 'scheduled'):
            assert f'Content from {other}' not in result


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — exact timestamp format regression
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
        result = p.get_previous_messages()
        assert '[2026-04-10 14:03]' in result


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — window-only, NO fixed row cap (compact-first)
#
# CANONICAL DESIGN (supersedes the trim-first build): there is
# NO provider-layer trim. get_previous_messages() renders EVERY watermark-bounded
# row. When the FULL request reaches the cap the ACT loop fires compaction BEFORE
# sending (compact-first) — it never sends a trimmed/partial view. The only
# drop-oldest is the ``drop_oldest`` PARAM, used solely by ChatHistoryCompactor's
# rare bare-request fallback (step 4.2) when even the tool-free compaction request
# overflows. _previous_rows() is id-ASC, so drop_oldest skips the OLDEST rows.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesWindowFit:
    def _seed_rows(self, db, n, channel=_GPM_CHANNEL):
        for i in range(n):
            db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES (?, 'user', ?, '2026-04-10 10:00:00')",
                (channel, f"line-{i:04d}-end"),
            )
        db.commit()

    def test_renders_all_rows_uncapped(self, db):
        # Well past the retired 50-row cap — every row must still render.
        n = 60
        self._seed_rows(db, n)
        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()
        # One line per row (tool_calls not injected into history), no fixed cap.
        assert len(result.splitlines()) == n
        assert "line-0000-end" in result
        assert f"line-{n - 1:04d}-end" in result

    def test_drop_oldest_param_skips_oldest_rows(self, db):
        # The 4.2 fallback drops the three oldest rows from the compaction input.
        n = 10
        self._seed_rows(db, n)
        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages(drop_oldest=3)
        assert len(result.splitlines()) == n - 3
        for i in range(3):
            assert f"line-{i:04d}-end" not in result   # oldest skipped
        assert "line-0003-end" in result               # first surviving row
        assert f"line-{n - 1:04d}-end" in result        # newest always kept

    def test_drop_oldest_param_at_or_beyond_count_returns_empty(self, db):
        # Skipping every row leaves nothing to render — the fallback floor relies
        # on this to know when there is nothing left to compact.
        n = 2
        self._seed_rows(db, n)
        p = _GPMFakeProcessor.make()
        assert p.get_previous_messages(drop_oldest=2) == ""
        assert p.get_previous_messages(drop_oldest=5) == ""


# =============================================================================
# Trail compaction excludes the automatic memory-recall seed
#
# Canonical design §2-3: ToolChainCompactor no-ops when there are no tool calls
# "excluding internal ones (compaction + automatic memory recall)", and when it
# DOES compact it excludes the automatic memory recall from the act-trail. The
# seed is the turn-0 memory(action=recall, _auto=True) dispatch; a model-issued
# memory call has no _auto flag and stays real trail. Real DB rows, real
# _has_trail / _render_act_trail off a uid-bound processor — zero mocks.
# =============================================================================
class TestTrailExcludesAutoMemoryRecall:
    def _seed_turn(self, db, channel=_GPM_CHANNEL):
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'turn input', '2026-04-10 10:00:00')",
            (channel,),
        )
        tid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()
        return tid

    def _add_tool_call(self, db, tid, tool_name, params_json, result, when):
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (tid, tool_name, params_json, result, when),
        )
        db.commit()

    def test_auto_memory_only_trail_is_no_op(self, db):
        # Only the turn-0 auto memory seed fired — nothing real to compact.
        tid = self._seed_turn(db)
        self._add_tool_call(
            db, tid, "memory",
            '{"action": "recall", "query": "who is boss", "_auto": true}',
            "recalled: boss is Alex", "2026-04-10 10:00:05",
        )
        p = _GPMFakeProcessor.make(uid=tid)
        assert p._has_trail() is False
        # Default render (per-turn display) still shows the auto-recall...
        assert "recalled: boss is Alex" in p._render_act_trail()
        # ...but the compaction input drops it, leaving nothing to compact.
        assert p._render_act_trail(for_compaction=True) == ""

    def test_auto_memory_plus_real_call_compacts_only_real(self, db):
        tid = self._seed_turn(db)
        self._add_tool_call(
            db, tid, "memory",
            '{"action": "recall", "query": "q", "_auto": true}',
            "auto recalled facts", "2026-04-10 10:00:05",
        )
        self._add_tool_call(
            db, tid, "search", '{"query": "weather"}',
            "sunny 24C", "2026-04-10 10:00:10",
        )
        p = _GPMFakeProcessor.make(uid=tid)
        assert p._has_trail() is True
        compaction_input = p._render_act_trail(for_compaction=True)
        assert "sunny 24C" in compaction_input            # real call compacted
        assert "auto recalled facts" not in compaction_input  # auto seed excluded
        # Default per-turn render keeps both.
        assert "auto recalled facts" in p._render_act_trail()

    def test_model_initiated_memory_counts_as_real_trail(self, db):
        # A model-issued memory call has NO _auto flag — it is genuine activity.
        tid = self._seed_turn(db)
        self._add_tool_call(
            db, tid, "memory",
            '{"action": "recall", "query": "remind me"}',
            "model recalled X", "2026-04-10 10:00:05",
        )
        p = _GPMFakeProcessor.make(uid=tid)
        assert p._has_trail() is True
        assert "model recalled X" in p._render_act_trail(for_compaction=True)


# =============================================================================
# _wrap_with_checkpoint — real DB tests
#
# The checkpoint summary is read from the canonical watermark home: a transcript
# row with role='compaction' (design §3.6, get_compaction). Seed it via the
# production factory transcript_service.write_input_row — the exact call _compact()
# makes — never a hand-rolled INSERT or the retired tool_calls audit-row model.
# =============================================================================


_COMPACT_CHANNEL = 'test_compact_channel'


class TestWrapWithCheckpoint:
    def test_row_with_content_exact_envelope_format(self, db):
        from services.message_processor import _wrap_with_checkpoint
        from services import transcript_service

        # Production writes the compaction summary as a transcript row with
        # role='compaction'; its own id is the watermark.
        transcript_service.write_input_row(_COMPACT_CHANNEL, 'compaction', 'checkpoint content')

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
