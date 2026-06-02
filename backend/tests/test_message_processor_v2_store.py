# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for get_previous_messages(), compaction, and store() hot paths.

Uses real in-memory SQLite DB — no mocks for data path.
"""

import logging as _logging
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

# =============================================================================
# RESCUED: get_previous_messages() tests (from test_message_processor_v2_base.py)
# Only tests that use the real `db` fixture are included.
# =============================================================================


_GPM_USER_DEF = "The user is a human named Alice interacting via the chat interface."
_GPM_USER_PROMPT = "What time is it?"
_GPM_CHANNEL = 'test_channel'


def _stub_gpm_prompt_cls():
    from services.system_message_prompt import SystemMessagePrompt

    class _StubPrompt(SystemMessagePrompt):
        _SYSTEM_PROMPT = ''

    return _StubPrompt


class _GPMFakeProcessor:
    """Concrete MessageProcessor subclass for get_previous_messages() tests."""

    _CHANNEL = _GPM_CHANNEL
    _ROLE = 'test_role'

    @staticmethod
    def make(**kwargs):
        from services.message_processor import MessageProcessor

        class _Fake(MessageProcessor):
            CHANNEL = _GPMFakeProcessor._CHANNEL
            ROLE = _GPMFakeProcessor._ROLE
            SYSTEM_PROMPT_CLASS = _stub_gpm_prompt_cls()

            def get_user_definition(self) -> str:
                return _GPM_USER_DEF

            def get_user_prompt(self) -> str:
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

            def get_user_definition(self) -> str:
                return _GPM_USER_DEF

            def get_user_prompt(self) -> str:
                return _GPM_USER_PROMPT

        return _Fake


# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — empty channel → ''
# ─────────────────────────────────────────────────────────────────────────────





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
# get_previous_messages() — compaction row causes prepend + watermark
# ─────────────────────────────────────────────────────────────────────────────


def _seed_compaction_via_tool_calls(db, channel, compacted_text, compacted_up_to_id):
    """Seed a success compaction row via the new tool_calls + transcript join.

    Attaches the tool_calls row to the compacted_up_to_id transcript row as the
    FK parent. This avoids inserting a phantom transcript row above the watermark
    that would otherwise appear in get_previous_messages() output.
    Returns the tool_call id.
    """
    import json
    cursor = db.execute(
        "INSERT INTO tool_calls "
        "(transcript_id, tool_name, params, result, ephemeral, created_at) "
        "VALUES (?, 'compaction', ?, ?, 0, '2026-01-01 00:00:01')",
        (
            compacted_up_to_id,
            json.dumps({"compacted_up_to_id": compacted_up_to_id, "status": "success"}),
            compacted_text,
        )
    )
    db.commit()
    return cursor.lastrowid


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

        _seed_compaction_via_tool_calls(db, channel, 'COMPACTED: previous context here', wm_id)
        return wm_id

    def test_compacted_text_prepended(self, db):
        self._seed_with_compaction(db)
        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()
        assert result.startswith('COMPACTED: previous context here')




# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — send() + store() wiring (db-only stubs)
# ─────────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — compaction only, no new rows above watermark
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — compaction + one new transcript row above watermark
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

        _seed_compaction_via_tool_calls(db, channel, 'COMPACTION SUMMARY', watermark_id)

        p = _GPMFakeProcessor.make()
        result = p.get_previous_messages()

        assert result.startswith('COMPACTION SUMMARY')
        assert 'Post-compaction message' in result
        assert 'Pre-compaction message' not in result



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
# get_previous_messages() — token_budget parameter accepted and ignored
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesTokenBudget:
    """token_budget is accepted without error; output is identical regardless."""




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
# get_previous_messages() — empty content row does not raise
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# get_previous_messages() — explicit since_id=0 path
# ─────────────────────────────────────────────────────────────────────────────




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

        def get_user_definition(self) -> str:
            return _COMPACT_USER_DEF

        def get_user_prompt(self) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# _wrap_with_checkpoint — real DB tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWrapWithCheckpoint:
    """Module-private _wrap_with_checkpoint function behavior against real DB."""

    def test_no_compaction_row_returns_bare_body(self, db):
        from services.message_processor import _wrap_with_checkpoint
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'hello user')
        assert result == 'hello user'

    def test_row_with_content_exact_envelope_format(self, db):
        from services.message_processor import _wrap_with_checkpoint
        anchor = _compact_seed_transcript_row(db, _COMPACT_CHANNEL)
        _seed_compaction_via_tool_calls(db, _COMPACT_CHANNEL, 'checkpoint content', anchor)
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



# ─────────────────────────────────────────────────────────────────────────────
# _handle_overflow — DB-writing tests (d8, d9, d10)
# ─────────────────────────────────────────────────────────────────────────────


def _compact_get_audit_row(db, channel):
    """Return the most recent success compaction audit row from tool_calls for channel."""
    row = db.execute(
        """
        SELECT tc.result, json_extract(tc.params, '$.compacted_up_to_id') AS compacted_up_to_id
        FROM tool_calls tc
        JOIN transcript t ON t.id = tc.transcript_id
        WHERE tc.tool_name = 'compaction'
          AND t.channel = ?
          AND json_extract(tc.params, '$.status') = 'success'
        ORDER BY tc.id DESC
        LIMIT 1
        """,
        (channel,),
    ).fetchone()
    if not row:
        return None
    return {'compacted_text': row[0], 'compacted_up_to_id': row[1]}


class TestHandleOverflowDbWrites:
    """_handle_overflow writes append-only tool_calls audit rows (not compactions table)."""

    def test_d8_overflow_writes_fresh_audit_row_when_none_existed(self, db):
        """When no prior compaction exists, _handle_overflow writes a fresh tool_calls row."""
        t1_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hello')
        t2_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'assistant', 'hi')

        p = _make_compact_processor()
        p._act_trail = []
        p._uid = t1_id

        llm_resp = _make_compact_llm_response(
            text='<analysis>notes</analysis><summary>fresh compaction summary</summary>'
        )

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_persistence.get_entries_since', return_value=[
                 {'id': t1_id, 'role': 'user', 'content': 'hello', 'tool_name': None},
                 {'id': t2_id, 'role': 'assistant', 'content': 'hi', 'tool_name': None},
             ]), \
             patch('services.compaction_persistence.get_compaction', return_value=None):
            mock_inst.return_value.send_messages.return_value = llm_resp
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0
            result = p._run_full_compaction()

        assert result == 'fresh compaction summary'
        row = _compact_get_audit_row(db, _COMPACT_CHANNEL)
        assert row is not None
        assert row['compacted_text'] == 'fresh compaction summary'

    def test_d9_overflow_appends_new_row_on_top_of_existing(self, db):
        """A second compaction appends a new row — both coexist, latest wins."""
        old_t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'old turn')
        _seed_compaction_via_tool_calls(db, _COMPACT_CHANNEL, 'old summary', old_t_id)

        new_t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'new turn')

        p = _make_compact_processor()
        p._act_trail = []
        p._uid = new_t_id

        llm_resp = _make_compact_llm_response(
            text='<analysis>notes</analysis><summary>updated compaction summary</summary>'
        )

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_persistence.get_entries_since', return_value=[
                 {'id': new_t_id, 'role': 'user', 'content': 'new turn', 'tool_name': None},
             ]), \
             patch('services.compaction_persistence.get_compaction', return_value={
                 'compacted_text': 'old summary',
                 'compacted_up_to_id': old_t_id,
             }):
            mock_inst.return_value.send_messages.return_value = llm_resp
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0
            p._run_full_compaction()

        new_row = _compact_get_audit_row(db, _COMPACT_CHANNEL)
        assert new_row is not None
        assert new_row['compacted_text'] == 'updated compaction summary'

    def test_d10_failure_audit_row_invisible_to_canonical_lookup(self, db):
        """Failure audit rows are stored (status=failure) but invisible to get_compaction()."""
        t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'a turn')

        p = _make_compact_processor()
        p._act_trail = []
        p._uid = t_id

        # LLM returns text without <summary> tags — causes a failure audit row
        llm_resp = _make_compact_llm_response(text='this has no summary tags at all')

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_persistence.get_entries_since', return_value=[
                 {'id': t_id, 'role': 'user', 'content': 'a turn', 'tool_name': None},
             ]), \
             patch('services.compaction_persistence.get_compaction', return_value=None):
            mock_inst.return_value.send_messages.return_value = llm_resp
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0
            result = p._run_full_compaction()

        assert result is None
        # Failure row exists in DB but is invisible to canonical lookup
        assert _compact_get_audit_row(db, _COMPACT_CHANNEL) is None
        failure_row = db.execute(
            "SELECT json_extract(params, '$.status') FROM tool_calls "
            "WHERE tool_name='compaction' AND transcript_id=?",
            (t_id,),
        ).fetchone()
        assert failure_row is not None
        assert failure_row[0] == 'failure'


# ─────────────────────────────────────────────────────────────────────────────
# _run_full_compaction — direct DB tests (e1–e6)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunFullCompaction:
    """Direct tests for _run_full_compaction against real DB."""

    def test_e1_no_entries_and_no_prior_checkpoint_skips_llm(self, db, caplog):
        """With no entries AND no prior checkpoint, _run_full_compaction returns
        None WITHOUT calling the LLM and without writing any audit row."""
        p = _make_compact_processor()

        with caplog.at_level(_logging.WARNING):
            with patch('services.providers.Providers.instance') as mock_inst, \
                 patch('services.compaction_persistence.get_entries_since', return_value=[]), \
                 patch('services.compaction_persistence.get_compaction', return_value=None):
                result = p._run_full_compaction()
                mock_inst.return_value.send_messages.assert_not_called()

        assert result is None
        assert _compact_get_audit_row(db, _COMPACT_CHANNEL) is None
        assert any(
            'no entries' in rec.message and 'skipping LLM call' in rec.message
            for rec in caplog.records
        )


    def test_e4_happy_path_writes_audit_row_and_returns_summary(self, db):
        """Happy path: LLM returns <summary> tags, audit row written, summary text returned."""
        t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hello')
        p = _make_compact_processor()
        p._uid = t_id
        llm_resp = _make_compact_llm_response(
            text='<analysis>context notes</analysis><summary>happy path summary</summary>'
        )

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_persistence.get_entries_since', return_value=[
                 {'id': t_id, 'role': 'user', 'content': 'hello', 'tool_name': None},
             ]), \
             patch('services.compaction_persistence.get_compaction', return_value=None):
            mock_inst.return_value.send_messages.return_value = llm_resp
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0
            result = p._run_full_compaction()

        assert result == 'happy path summary'

        row = _compact_get_audit_row(db, _COMPACT_CHANNEL)
        assert row is not None
        assert row['compacted_text'] == 'happy path summary'

        # ephemeral=0 so it is persistent (not ephemeral)
        tc_row = db.execute(
            "SELECT ephemeral FROM tool_calls WHERE transcript_id=? AND tool_name='compaction'",
            (t_id,),
        ).fetchone()
        assert tc_row is not None
        assert tc_row[0] == 0

    def test_e5_watermark_set_to_max_entry_id(self, db):
        """Watermark is set to max(entry['id']) from the entries fed to the LLM."""
        id3 = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'a')
        id5 = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'c')
        id7 = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'assistant', 'b')

        p = _make_compact_processor()
        p._uid = id3
        entries = [
            {'id': id3, 'role': 'user', 'content': 'a', 'tool_name': None},
            {'id': id7, 'role': 'assistant', 'content': 'b', 'tool_name': None},
            {'id': id5, 'role': 'user', 'content': 'c', 'tool_name': None},
        ]
        llm_resp = _make_compact_llm_response(
            text='<summary>compacted</summary>'
        )

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_persistence.get_entries_since', return_value=entries), \
             patch('services.compaction_persistence.get_compaction', return_value=None):
            mock_inst.return_value.send_messages.return_value = llm_resp
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0
            p._run_full_compaction()

        row = _compact_get_audit_row(db, _COMPACT_CHANNEL)
        assert row is not None
        assert row['compacted_up_to_id'] == id7


# =============================================================================
# End-to-end ACT loop glue: scenario 120/121 reproduce the same path —
# compact_at is set tiny, the assembled payload must blow past it on the very
# first iteration, _handle_overflow must run, an append-only success audit row
# must be written, and the loop must restart and return a final answer.
# =============================================================================


class TestActLoopOverflowEndToEnd:
    """Old-path overflow end-to-end tests.

    The test that exercised ContinuityCompactionProcessor.send() via the old
    _handle_overflow path has been removed (§9b T6).  The class is retained as
    a tombstone to avoid confusing git blame — remaining coverage lives in the
    new D-series tests for the flat _compact_trail / _compact_history paths.
    """

    _CHANNEL = 'test_act_overflow_channel'


# ─────────────────────────────────────────────────────────────────────────────
# exclude_id filters the current turn's transcript row from compaction input
# Regression: without exclude_id the current unanswered user message enters
# the compaction LLM prompt, potentially confusing the continuity summary.
# ─────────────────────────────────────────────────────────────────────────────


class TestRunFullCompactionExcludeId:
    """_run_full_compaction(exclude_id=X) must not include transcript row X
    in the entries fed to the LLM.  This is the mechanism that prevents the
    current (unanswered) user turn from polluting the compaction context.
    """

    _CHANNEL = 'test_exclude_id_channel'

    def test_current_turn_row_absent_from_compaction_entries(self, db):
        """The transcript row for _uid is not included in the LLM input."""
        # Prior conversation that should be included
        prior_id = _compact_seed_transcript_row(
            db, self._CHANNEL, 'user', 'prior message',
        )
        prior_reply = _compact_seed_transcript_row(
            db, self._CHANNEL, 'assistant', 'prior reply',
        )
        # Current turn row — must be excluded
        current_id = _compact_seed_transcript_row(
            db, self._CHANNEL, 'user', 'SHOULD NOT APPEAR IN COMPACTION',
        )

        captured_inputs: list[list] = []

        def fake_send(system_prompt, messages, job=None, **_kw):
            # Capture the full message list passed to the compaction LLM
            captured_inputs.append(messages)
            from services.llm_service import LLMResponse
            return LLMResponse(
                text='<summary>compact result</summary>',
                model='test', provider='mock', tool_calls=None,
            )

        p = _make_compact_processor(channel=self._CHANNEL)
        p._uid = current_id

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_persistence.get_compaction', return_value=None), \
             patch('services.compaction_persistence.get_entries_since',
                   return_value=[
                       {'id': prior_id, 'role': 'user', 'content': 'prior message',
                        'tool_name': None},
                       {'id': prior_reply, 'role': 'assistant', 'content': 'prior reply',
                        'tool_name': None},
                       {'id': current_id, 'role': 'user',
                        'content': 'SHOULD NOT APPEAR IN COMPACTION',
                        'tool_name': None},
                   ]):
            mock_inst.return_value.send_messages.side_effect = fake_send
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0
            result = p._run_full_compaction(exclude_id=current_id)

        assert result == 'compact result'
        # The LLM must have been called exactly once
        assert len(captured_inputs) == 1, "LLM should be called exactly once"
        # Flatten all message content from the captured call
        all_content = ' '.join(
            str(msg.get('content', ''))
            for msg in captured_inputs[0]
        )
        assert 'SHOULD NOT APPEAR IN COMPACTION' not in all_content, (
            "The current turn's transcript row leaked into the compaction LLM input. "
            "exclude_id filtering is broken."
        )
        # Prior rows must still be present
        assert 'prior message' in all_content or 'prior reply' in all_content, (
            "Prior transcript rows must be included in the compaction input"
        )
