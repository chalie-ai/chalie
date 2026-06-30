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

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor
    from tests.helpers import StubProcessorConfig

pytestmark = pytest.mark.unit

# =============================================================================
# RESCUED: get_previous_messages() tests (from test_message_processor_v2_base.py)
# Only tests that use the real `db` fixture are included.
# =============================================================================


_GPM_USER_DEF = "The user is a human named Alice interacting via the chat interface."
_GPM_USER_PROMPT = "What time is it?"
_GPM_CHANNEL = 'test_channel'


def _settled_turn(channel: str, contents: list[str]) -> list[int]:
    """Seed one settled turn through the production writers; return its row ids
    in id order. ``contents[0]`` is the user opener; the middle items are
    tool-bearing assistant steps (a real tool keeps each below settle0); the
    last is the no-tool answer (settle0). Every row lands in the MAIN spine
    (``id <= settle0``), so the whole list renders in get_previous_messages."""
    from services.transcript_service import Transcript
    from services.act_trail import ActTrail

    in_id = Transcript.write_input_row(channel, "user", contents[0])
    tid = Transcript.turn_id_of_row(in_id)
    ids = [in_id]
    for text in contents[1:-1]:
        step = Transcript.write_assistant_row(channel, text, turn_id=tid)
        ActTrail().record(tool_name="search_files", params={}, result="", transcript_id=step)
        ids.append(step)
    ids.append(Transcript.write_assistant_row(channel, contents[-1], turn_id=tid))
    return ids


def _gpm_config(channel: str = _GPM_CHANNEL, role: str = 'test_role', suppress_history: bool = False) -> StubProcessorConfig:
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
    def make(channel: str = _GPM_CHANNEL, suppress_history: bool = False, **kwargs: object) -> MessageProcessor:
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
    def _seed_transcript(self, db: sqlite3.Connection, channel: str = _GPM_CHANNEL) -> None:
        """A settled turn whose pre-settle assistant step carries two tool calls.
        The transcript rows are spine rows (``id <= settle0``); the tool_calls
        results live in their own table and must never reach the history prompt."""
        from services.transcript_service import Transcript
        from services.act_trail import ActTrail

        in_id = Transcript.write_input_row(channel, "user", "Hello world")
        tid = Transcript.turn_id_of_row(in_id)
        step = Transcript.write_assistant_row(channel, "Hi there", turn_id=tid)
        ActTrail().record(tool_name="memory", params={}, result="User likes dark mode", transcript_id=step)
        ActTrail().record(tool_name="read", params={}, result="Web page content", transcript_id=step)
        Transcript.write_assistant_row(channel, "settled answer", turn_id=tid)
        db.commit()

    def test_tool_calls_do_not_appear_in_previous_messages(self, db: sqlite3.Connection) -> None:
        """History replay no longer renders tool_calls rows (decision 2).
        Tool calls are retained durably but excluded from get_previous_messages()
        to keep the history prompt clean and unambiguous."""
        self._seed_transcript(db)
        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()
        # transcript rows render (user opener + the assistant step content)
        assert 'Hello world' in result
        assert 'Hi there' in result
        # tool_calls are NOT injected into the history prompt
        assert 'User likes dark mode' not in result
        assert 'Web page content' not in result


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — multiple channels, no cross-channel leakage
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesChannelIsolation:
    def _seed_all_channels(self, db: sqlite3.Connection) -> None:
        for channel in ('user', 'dmn', 'subagent', 'schedule', _GPM_CHANNEL):
            _settled_turn(channel, [f"Content from {channel}", f"Answer for {channel}"])
        db.commit()

    def test_test_channel_isolation(self, db: sqlite3.Connection) -> None:
        self._seed_all_channels(db)
        p = _GPMFakeProcessor.make()  # uses test_channel as its CHANNEL
        result = p.get_previous_messages()
        assert f'Content from {_GPM_CHANNEL}' in result
        for other in ('user', 'dmn', 'subagent', 'schedule'):
            assert f'Content from {other}' not in result


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — exact timestamp format regression
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesTimestampFormat:
    def test_iso_timestamp_with_offset_formatted_correctly(self, db: sqlite3.Connection) -> None:
        from datetime import timedelta

        from services.time_utils import utc_now

        channel = _GPM_CHANNEL
        # A settled turn whose opener carries an ISO-8601 created_at with a UTC
        # offset must render minute-precision as [YYYY-MM-DD HH:MM]. The opener is
        # a spine row, so its timestamp surfaces in the history block.
        recent = utc_now() - timedelta(minutes=5)
        in_id = _settled_turn(channel, ['Timestamp test', 'answer'])[0]
        db.execute(
            "UPDATE transcript SET created_at = ? WHERE id = ?",
            (recent.isoformat(), in_id),
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()
        assert f"[{recent.strftime('%Y-%m-%d %H:%M')}]" in result


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — every row of the settled spine, NO per-row cap
#
# CANONICAL DESIGN: get_previous_messages() renders EVERY row of
# the spine the watermark exposes — there is no provider-layer trim and no fixed
# per-row cap. When the FULL request reaches the context cap the ACT loop fires
# compaction BEFORE sending (compact-first); it never sends a trimmed view. The
# only drop-oldest is the ``drop_oldest`` PARAM, used solely by
# ChatHistoryCompactor's bare-request fallback (step 4.2) when even the tool-free
# compaction request overflows. _previous_rows() is id-ASC, so drop_oldest skips
# the OLDEST rows. Seeding one settled turn pins the exact row count without the
# spine's recent-turn bound interfering.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesWindowFit:
    def _seed_rows(self, db: sqlite3.Connection, n: int, channel: str = _GPM_CHANNEL) -> None:
        # One settled turn of exactly n spine rows (opener + n-2 steps + settle),
        # content ``line-NNNN-end`` in id order.
        _settled_turn(channel, [f"line-{i:04d}-end" for i in range(n)])
        db.commit()

    def test_renders_every_row_of_the_settled_turn(self, db: sqlite3.Connection) -> None:
        # Well past the retired 50-row cap — every row of the turn still renders.
        n = 60
        self._seed_rows(db, n)
        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()
        # One line per row (tool_calls not injected into history), no per-row cap.
        assert len(result.splitlines()) == n
        assert "line-0000-end" in result
        assert f"line-{n - 1:04d}-end" in result

    def test_drop_oldest_param_skips_oldest_rows(self, db: sqlite3.Connection) -> None:
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

    def test_drop_oldest_param_at_or_beyond_count_returns_empty(self, db: sqlite3.Connection) -> None:
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
# The checkpoint summary is read from the dedicated compactions table via
# get_compaction(channel, for_turn_id) — the MAIN spine reads the for_turn_id IS
# NULL axis. Seed it through the production writer compaction_persistence
# .write_compaction — the exact call the compactor's run() makes — never a
# hand-rolled INSERT or the retired role='compaction' transcript row.
# =============================================================================


_COMPACT_CHANNEL = 'test_compact_channel'


class TestWrapWithCheckpoint:
    def test_row_with_content_exact_envelope_format(self, db: sqlite3.Connection) -> None:
        from services.message_processor import _wrap_with_checkpoint
        from services import compaction_persistence

        # The compactor writes the MAIN checkpoint on the for_turn_id IS NULL axis
        # (compacted_up_to is a turn_id); the envelope reads that same axis.
        compaction_persistence.write_compaction(_COMPACT_CHANNEL, None, 1, 'checkpoint content')

        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'current body', None)
        expected = (
            "### Checkpoint - What you were previously discussing / doing\n"
            "checkpoint content\n"
            "\n"
            "---\n"
            "### Current State - What's happening in the current turn\n"
            "current body"
        )
        assert result == expected
