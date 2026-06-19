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
        # created_at defaults to now() — recent rows that survive the 6h age cut-off.
        db.execute(
            "INSERT INTO transcript (channel, role, content) "
            "VALUES (?, 'user', 'Hello world')",
            (channel,)
        )
        uid1 = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        db.execute(
            "INSERT INTO transcript (channel, role, content) "
            "VALUES (?, 'assistant', 'Hi there')",
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
                "INSERT INTO transcript (channel, role, content) "
                "VALUES (?, 'user', ?)",
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
        from datetime import timedelta

        from services.time_utils import utc_now

        channel = _GPM_CHANNEL
        # A recent ISO-8601 created_at with a UTC offset (within the 6h window) must
        # render minute-precision as [YYYY-MM-DD HH:MM].
        recent = utc_now() - timedelta(minutes=5)
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Timestamp test', ?)",
            (channel, recent.isoformat())
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()
        assert f"[{recent.strftime('%Y-%m-%d %H:%M')}]" in result


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
                "INSERT INTO transcript (channel, role, content) "
                "VALUES (?, 'user', ?)",
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
