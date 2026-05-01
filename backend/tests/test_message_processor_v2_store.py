# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for getPreviousMessages(), compaction, and store() hot paths.

Uses real in-memory SQLite DB — no mocks for data path.
"""

import logging as _logging
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

# =============================================================================
# RESCUED: getPreviousMessages() tests (from test_message_processor_v2_base.py)
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


def _seed_compaction_via_tool_calls(db, channel, compacted_text, compacted_up_to_id):
    """Seed a success compaction row via the new tool_calls + transcript join.

    Attaches the tool_calls row to the compacted_up_to_id transcript row as the
    FK parent. This avoids inserting a phantom transcript row above the watermark
    that would otherwise appear in getPreviousMessages() output.
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
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.return_value = fake_response
            mock_inst.return_value.get_compact_at.return_value = 32_000
            result = p.send()
        assert isinstance(result, str)

    def test_store_is_callable(self, db):
        """store() is wired — calls DB, no longer raises NotImplementedError."""
        p = _GPMFakeProcessor.make()
        with patch('services.transcript_service._trigger_episode_extraction'):
            p.store('final response')
        # store() writes assistant row; no exception = pass

    def test_send_is_wired_in_commit_6(self, db):
        """send() calls provider, does not raise NotImplementedError."""
        from services.llm_service import LLMResponse
        p = _GPMFakeProcessor.make()
        fake_response = LLMResponse(text='done', model='m', provider='p', tool_calls=None)
        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.return_value = fake_response
            mock_inst.return_value.get_compact_at.return_value = 32_000
            result = p.send()
        assert result == 'done'

    def test_store_is_wired_in_commit_4(self, db):
        """store() calls DB, does not raise NotImplementedError."""
        p = _GPMFakeProcessor.make()
        with patch('services.transcript_service._trigger_episode_extraction'):
            p.store('final response')
        # store() writes assistant row; no exception = pass

    def test_send_accepts_request_id(self, db):
        """send() accepts an optional request_id parameter."""
        from services.llm_service import LLMResponse
        p = _GPMFakeProcessor.make()
        fake_response = LLMResponse(text='ok', model='m', provider='p', tool_calls=None)
        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.return_value = fake_response
            mock_inst.return_value.get_compact_at.return_value = 32_000
            result = p.send('req-id')
        assert isinstance(result, str)

    def test_store_not_caught_as_plain_exception(self, db):
        """store() calls DB, no longer raises NotImplementedError."""
        p = _GPMFakeProcessor.make()
        with patch('services.transcript_service._trigger_episode_extraction'):
            p.store('response text')
        # store() writes assistant row; no exception = pass


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

        _seed_compaction_via_tool_calls(db, channel, 'COMPACTED: all prior context', old_id)

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

        _seed_compaction_via_tool_calls(db, channel, 'COMPACT BLOCK', seed_id)

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

        _seed_compaction_via_tool_calls(db, channel, 'COMPACTION SUMMARY', watermark_id)

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

        _seed_compaction_via_tool_calls(db, channel, 'COMPACT_START', wm)

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
        for channel in ('user', 'dmn', 'subagent', 'scheduled', _GPM_CHANNEL):
            db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES (?, 'user', ?, '2026-04-10 10:00:00')",
                (channel, f"Content from {channel}")
            )
        db.commit()

    def test_test_channel_does_not_see_user_channel(self, db):
        self._seed_all_channels(db)
        p = _GPMFakeProcessor.make()  # uses test_channel as its CHANNEL
        result = p.getPreviousMessages()
        assert 'Content from user' not in result
        assert f'Content from {_GPM_CHANNEL}' in result

    def test_test_channel_does_not_see_dmn_channel(self, db):
        self._seed_all_channels(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Content from dmn' not in result

    def test_test_channel_does_not_see_subagent_channel(self, db):
        self._seed_all_channels(db)
        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Content from subagent' not in result

    def test_each_channel_sees_only_its_own_rows(self, db):
        """Build four processors with different CHANNELs and assert no bleed."""
        self._seed_all_channels(db)
        from services.message_processor import MessageProcessor

        for channel in ('user', 'dmn', 'subagent', 'scheduled'):
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
            for other in ('user', 'dmn', 'subagent', 'scheduled', _GPM_CHANNEL):
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
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'second_tool', '{}', 'Second result', 0, '2026-04-10 10:00:30')",
            (tid,)
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'first_tool', '{}', 'First result', 0, '2026-04-10 10:00:05')",
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

    def test_subagent_role_rendered_lowercase(self, db):
        channel = _GPM_CHANNEL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'subagent', 'Pursuit tick', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = _GPMFakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'subagent: Pursuit tick' in result

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

        p = _GPMFakeProcessor.make()  # uses test_channel as its CHANNEL
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
        anchor = _compact_seed_transcript_row(db, _COMPACT_CHANNEL)
        _seed_compaction_via_tool_calls(db, _COMPACT_CHANNEL,
                                        'Previously: we talked about movies.', anchor)
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'user: What is next?')
        assert result.startswith(
            '### Checkpoint - What you were previously discussing / doing\n'
        )
        assert 'Previously: we talked about movies.' in result
        assert "### Current State - What's happening in the current turn\n" in result
        assert result.endswith('user: What is next?')

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

    def test_row_with_empty_compacted_text_returns_bare_body(self, db):
        from services.message_processor import _wrap_with_checkpoint
        anchor = _compact_seed_transcript_row(db, _COMPACT_CHANNEL)
        _seed_compaction_via_tool_calls(db, _COMPACT_CHANNEL, '', anchor)
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'body text')
        assert result == 'body text'

    def test_row_with_whitespace_only_compacted_text_returns_bare_body(self, db):
        from services.message_processor import _wrap_with_checkpoint
        anchor = _compact_seed_transcript_row(db, _COMPACT_CHANNEL)
        _seed_compaction_via_tool_calls(db, _COMPACT_CHANNEL, '   \n\t  ', anchor)
        result = _wrap_with_checkpoint(_COMPACT_CHANNEL, 'body text')
        assert result == 'body text'


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
            mock_inst.return_value.get_compact_at.return_value = 32_000
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
            mock_inst.return_value.get_compact_at.return_value = 32_000
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
            mock_inst.return_value.get_compact_at.return_value = 32_000
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

    def test_e2_llm_returns_no_summary_tags_returns_none(self, db):
        """When LLM returns text without <summary> tags, returns None and writes failure row."""
        t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hi')
        p = _make_compact_processor()
        p._uid = t_id
        llm_resp = _make_compact_llm_response(text='plain text without tags')

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_persistence.get_entries_since', return_value=[
                 {'id': t_id, 'role': 'user', 'content': 'hi', 'tool_name': None},
             ]), \
             patch('services.compaction_persistence.get_compaction', return_value=None):
            mock_inst.return_value.send_messages.return_value = llm_resp
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.get_compact_at.return_value = 32_000
            result = p._run_full_compaction()

        assert result is None
        assert _compact_get_audit_row(db, _COMPACT_CHANNEL) is None  # failure invisible

    def test_e3_llm_raises_returns_none(self, db, caplog):
        """When LLM call raises, _run_full_compaction returns None without a success row."""
        t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hi')
        p = _make_compact_processor()
        p._uid = t_id

        with caplog.at_level(_logging.ERROR):
            with patch('services.providers.Providers.instance') as mock_inst, \
                 patch('services.compaction_persistence.get_entries_since', return_value=[
                     {'id': t_id, 'role': 'user', 'content': 'hi', 'tool_name': None},
                 ]), \
                 patch('services.compaction_persistence.get_compaction', return_value=None):
                mock_inst.return_value.send_messages.side_effect = RuntimeError('LLM down')
                mock_inst.return_value.get_context_limit.return_value = 32_000
                mock_inst.return_value.get_compact_at.return_value = 32_000
                result = p._run_full_compaction()

        assert result is None
        assert _compact_get_audit_row(db, _COMPACT_CHANNEL) is None

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
            mock_inst.return_value.get_compact_at.return_value = 32_000
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
            mock_inst.return_value.get_compact_at.return_value = 32_000
            p._run_full_compaction()

        row = _compact_get_audit_row(db, _COMPACT_CHANNEL)
        assert row is not None
        assert row['compacted_up_to_id'] == id7

    def test_e6_compaction_always_uses_frontal_cortex_unified(self, db):
        """Compaction LLM call always uses 'frontal-cortex-unified' regardless of
        the host processor's JOB. ContinuityCompactionProcessor hardcodes JOB so
        all compaction work is logged and provider-routed under one cognitive job."""
        t_id = _compact_seed_transcript_row(db, _COMPACT_CHANNEL, 'user', 'hi')

        class CustomJobProcessor(_make_compact_processor().__class__):
            JOB = 'custom-job-name'  # host JOB must NOT leak into compaction

        p = CustomJobProcessor('hi')
        p._uid = t_id

        captured_jobs = []
        llm_resp = _make_compact_llm_response(
            text='<summary>compacted</summary>'
        )

        def fake_send(_system_prompt, _messages, job=None, **_kw):  # noqa: ARG001
            captured_jobs.append(job)
            return llm_resp

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.compaction_persistence.get_entries_since', return_value=[
                 {'id': t_id, 'role': 'user', 'content': 'hi', 'tool_name': None},
             ]), \
             patch('services.compaction_persistence.get_compaction', return_value=None):
            mock_inst.return_value.send_messages.side_effect = fake_send
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.get_compact_at.return_value = 32_000
            p._run_full_compaction()

        assert captured_jobs == ['frontal-cortex-unified']
        assert 'custom-job-name' not in captured_jobs
