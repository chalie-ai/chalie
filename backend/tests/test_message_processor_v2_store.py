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
test_compaction_watermark.py and nightly scenario 120; the old singleton-mocked
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
    """A flat ProcessorConfig for get_previous_messages() tests.

    Post-§9b there are no MessageProcessor subclasses (P1) — every turn is a
    single flat MessageProcessor driven by a per-turn ProcessorConfig.
    """
    from services.processor_config import ProcessorConfig
    from tests.helpers import StubProcessorConfig

    return StubProcessorConfig(
        channel=channel,
        role=role,
        policy_channel=ProcessorConfig.POLICY_CHANNEL.CHAT,
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
    """Flat config-carrying MessageProcessor builder for get_previous_messages()
    tests (no subclass — P1 / §7a)."""

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
# get_previous_messages() — transcript rows + ephemeral filtering
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
            "ephemeral, created_at) "
            "VALUES (?, 'memory', '{}', 'User likes dark mode', 0, '2026-04-10 10:00:30')",
            (uid1,)
        )

        # Ephemeral (ephemeral=1) tool_call linked to uid2 — must NOT appear
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, "
            "ephemeral, created_at) "
            "VALUES (?, 'read', '{}', 'Web page content', 1, '2026-04-10 10:01:30')",
            (uid2,)
        )

        db.commit()
        return uid1, uid2

    def test_durable_tool_call_appears(self, db):
        self._seed_transcript(db)
        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()
        assert '[memory()] User likes dark mode' in result
        assert 'TOOL(memory)' not in result


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — multiple channels, no cross-channel leakage
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesChannelIsolation:
    """Each processor only sees its own CHANNEL's rows."""

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
# get_previous_messages() — durable tool_call interleaving order
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
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'memory', '{}', 'Result from memory', 0, '2026-04-10 10:00:10')",
            (tid,)
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'find_tools', '{}', 'Found weather tool', 0, '2026-04-10 10:00:20')",
            (tid,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()

        assert '[memory()] Result from memory' in result
        assert '[find_tools()] Found weather tool' in result
        assert 'TOOL(memory)' not in result
        assert 'TOOL(find_tools)' not in result

        input_pos = result.index('The input row')
        memory_pos = result.index('[memory()] Result from memory')
        find_pos = result.index('[find_tools()] Found weather tool')
        assert input_pos < memory_pos < find_pos


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — ephemeral=1 for same transcript row is dropped
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
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'memory', '{}', 'DURABLE_RESULT', 0, '2026-04-10 10:00:05')",
            (tid,)
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'read', '{}', 'EPHEMERAL_RESULT', 1, '2026-04-10 10:00:10')",
            (tid,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()

        assert 'DURABLE_RESULT' in result
        assert 'EPHEMERAL_RESULT' not in result


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
# get_previous_messages() — role rendering per north star
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesRoleCase:
    """Lock the role rendering convention: lowercase for inputs, title-case for assistant only."""

    def test_role_rendering_conventions(self, db):
        channel = _GPM_CHANNEL
        for role, content in [
            ('user', 'Hello'),
            ('assistant', 'Hi there'),
            ('subagent', 'Pursuit tick'),
        ]:
            db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES (?, ?, ?, '2026-04-10 10:00:00')",
                (channel, role, content)
            )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()
        assert 'user: Hello' in result
        assert 'Assistant: Hi there' in result
        assert 'assistant: Hi there' not in result
        assert 'subagent: Pursuit tick' in result


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — empty channel returns '' even when other channels
# have data
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesEmptyChannelWithOtherData:
    def test_returns_empty_when_no_rows_for_this_channel(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('other_channel', 'user', 'Not mine', '2026-04-10 10:00:00')"
        )
        db.commit()

        p = _GPMFakeProcessor.make()  # uses test_channel as its CHANNEL
        result = p.get_previous_messages()
        assert result == ''


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — window-only fit (NO fixed row cap)
#
# SPEC CHANGE (2026-06-05, Dylan — supersedes the 50-row render cap): compaction
# no longer caps the rendered history at a fixed row count, and the proactive
# turn-count trigger and COMPACTION_ROW_WINDOW were removed. get_previous_messages()
# renders EVERY watermark-bounded row; the live request is bounded by context-
# window size in Providers._fit_request, which drops the oldest rows (via
# drop_oldest_previous_message) until the request reserves max(10% window, 8k)
# response headroom. The dropped rows are summarised into the next turn's
# checkpoint (_dispatch_compaction resets _history_drop first), so nothing is
# lost. Dylan: "Drop the 50 turn cap. Let's go back to solely relying on context
# window size … loop and drop the oldest message until it fits."
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesWindowFit:
    """get_previous_messages() renders every row; drop_oldest trims the oldest."""

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
        # One line per row (no durable tool_calls), no fixed cap.
        assert len(result.splitlines()) == n
        assert "line-0000-end" in result
        assert f"line-{n - 1:04d}-end" in result

    def test_drop_oldest_trims_from_the_front(self, db):
        n = 10
        self._seed_rows(db, n)
        p = _GPMFakeProcessor.make()
        # The fit loop drops the three oldest rows.
        assert p.drop_oldest_previous_message() is True
        assert p.drop_oldest_previous_message() is True
        assert p.drop_oldest_previous_message() is True
        result = p.get_previous_messages()
        assert len(result.splitlines()) == n - 3
        for i in range(3):
            assert f"line-{i:04d}-end" not in result   # oldest dropped
        assert "line-0003-end" in result               # first surviving row
        assert f"line-{n - 1:04d}-end" in result        # newest always kept

    def test_drop_oldest_terminates_when_history_exhausted(self, db):
        # The fit loop's termination guarantee: drop_oldest returns False once
        # every row is gone, so _fit_request can never loop forever.
        n = 2
        self._seed_rows(db, n)
        p = _GPMFakeProcessor.make()
        assert p.drop_oldest_previous_message() is True
        assert p.drop_oldest_previous_message() is True
        assert p.drop_oldest_previous_message() is False
        assert p.get_previous_messages() == ""


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
    """Module-private _wrap_with_checkpoint function behavior against real DB."""

    def test_no_compaction_row_returns_bare_body(self, db):
        from services.message_processor import _wrap_with_checkpoint
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'hello user')
        assert result == 'hello user'

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
