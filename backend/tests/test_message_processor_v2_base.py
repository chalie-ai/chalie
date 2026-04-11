# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for services.message_processor — Commit 2 base class.

All tests are purely additive — nothing in production code is changed.
The module is imported directly; no callers are wired up yet.

Test coverage:
  1.  Base constructor fields
  2.  Default class constants
  3.  getSystemPrompt() raises TypeError when SYSTEM_PROMPT_CLASS is the abstract base
  4.  getSystemPrompt() with a subclass that sets SYSTEM_PROMPT_CLASS
  5.  getTools() — empty NATIVE_TOOLS + empty _discovered_tools → []
  6.  getTools() — native + dynamic → both returned
  7.  getTools() — same tool in native and dynamic → dedupe (first-seen wins)
  8.  getActLoopTrail() — empty → ''
  9.  getActLoopTrail() — multi-entry → newline-joined string
  10. getDynamicTools() — identity (no copy)
  11. getPreviousMessages() — empty channel → ''
  12. getPreviousMessages() — transcript rows + ephemeral filtering
  13. getPreviousMessages() — compaction row causes prepend + watermark
  14. Abstract methods raise NotImplementedError; postTurn is a no-op
  15. getUserPrompt() / getUserDefinition() raise on bare base
  16. Per-turn instance isolation (mutable defaults are per-instance)
  17. Import sanity: no DB, no LLM, no thread spawn on import
"""

import threading
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Shared fake subclass fixture
# ─────────────────────────────────────────────────────────────────────────────

_USER_DEF = "The user is a human named Alice interacting via the chat interface."
_USER_PROMPT = "What time is it?"


def _stub_prompt_cls():
    """Minimal concrete ``SystemMessagePrompt`` used as the default for
    ``FakeProcessor``. Any test that wants to exercise the abstract-default
    contract overrides ``SYSTEM_PROMPT_CLASS`` back to the bare base."""
    from services.system_message_prompt import SystemMessagePrompt

    class _StubPrompt(SystemMessagePrompt):
        _SYSTEM_PROMPT = ''

    return _StubPrompt


class FakeProcessor:
    """Concrete MessageProcessor subclass used in all tests below.

    Imported lazily inside each test class so that the import sanity test
    (TestImport) can verify a clean cold import.
    """

    _CHANNEL = 'test_channel'
    _ROLE = 'test_role'

    @staticmethod
    def make(**kwargs):
        """Build a FakeProcessor instance with optional overrides."""
        from services.message_processor import MessageProcessor

        class _Fake(MessageProcessor):
            CHANNEL = FakeProcessor._CHANNEL
            ROLE = FakeProcessor._ROLE
            SYSTEM_PROMPT_CLASS = _stub_prompt_cls()

            def getUserDefinition(self) -> str:
                return _USER_DEF

            def getUserPrompt(self) -> str:
                return _USER_PROMPT

        for k, v in kwargs.items():
            setattr(_Fake, k, v)

        return _Fake('test raw input', {'key': 'value'})

    @staticmethod
    def cls():
        from services.message_processor import MessageProcessor

        class _Fake(MessageProcessor):
            CHANNEL = FakeProcessor._CHANNEL
            ROLE = FakeProcessor._ROLE
            SYSTEM_PROMPT_CLASS = _stub_prompt_cls()

            def getUserDefinition(self) -> str:
                return _USER_DEF

            def getUserPrompt(self) -> str:
                return _USER_PROMPT

        return _Fake


# ─────────────────────────────────────────────────────────────────────────────
# 17. Import sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestImport:
    """Import must be side-effect free."""

    def test_import_does_not_hit_database(self):
        with patch('services.database_service.get_shared_db_service') as mock_db:
            import importlib
            import services.message_processor as _mod
            importlib.reload(_mod)
            mock_db.assert_not_called()

    def test_import_does_not_spawn_threads(self):
        before = threading.active_count()
        import importlib
        import services.message_processor as _mod
        importlib.reload(_mod)
        after = threading.active_count()
        assert after == before, (
            f"Import spawned {after - before} thread(s) — must be side-effect free"
        )

    def test_import_does_not_call_llm(self):
        with patch('services.llm_service.create_llm_service') as mock_llm:
            import importlib
            import services.message_processor as _mod
            importlib.reload(_mod)
            mock_llm.assert_not_called()

    def test_import_exports_message_processor(self):
        from services.message_processor import MessageProcessor
        assert MessageProcessor is not None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constructor
# ─────────────────────────────────────────────────────────────────────────────


class TestConstructor:
    def test_raw_input_stored(self):
        p = FakeProcessor.make()
        assert p._raw_input == 'test raw input'

    def test_metadata_stored(self):
        p = FakeProcessor.make()
        assert p._metadata == {'key': 'value'}

    def test_metadata_defaults_to_empty_dict_when_none(self):
        Cls = FakeProcessor.cls()
        p = Cls('input', None)
        assert p._metadata == {}

    def test_memory_seed_is_none(self):
        p = FakeProcessor.make()
        assert p._memory_seed is None

    def test_act_trail_is_empty_list(self):
        p = FakeProcessor.make()
        assert p._act_trail == []
        assert isinstance(p._act_trail, list)

    def test_pending_tool_calls_is_empty_list(self):
        p = FakeProcessor.make()
        assert p._pending_tool_calls == []
        assert isinstance(p._pending_tool_calls, list)

    def test_discovered_tools_is_empty_list(self):
        p = FakeProcessor.make()
        assert p._discovered_tools == []
        assert isinstance(p._discovered_tools, list)

    def test_uid_is_none(self):
        p = FakeProcessor.make()
        assert p._uid is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Default class constants
# ─────────────────────────────────────────────────────────────────────────────


class TestClassConstants:
    def test_job(self):
        from services.message_processor import MessageProcessor
        assert MessageProcessor.JOB == 'frontal-cortex-unified'

    def test_max_iterations(self):
        from services.message_processor import MessageProcessor
        assert MessageProcessor.MAX_ITERATIONS == 30

    def test_max_timeout(self):
        from services.message_processor import MessageProcessor
        assert MessageProcessor.MAX_TIMEOUT == 900

    def test_native_tools_is_empty_list(self):
        from services.message_processor import MessageProcessor
        assert MessageProcessor.NATIVE_TOOLS == []

    def test_compaction_prompt_is_non_empty_string(self):
        # Commit 7: prompt is now populated (was placeholder '' in Commit 2).
        from services.message_processor import MessageProcessor
        assert isinstance(MessageProcessor.COMPACTION_PROMPT, str)
        assert len(MessageProcessor.COMPACTION_PROMPT) > 0

    def test_tool_compaction_prompt_is_non_empty_string(self):
        # Commit 7: prompt is now populated (was placeholder '' in Commit 2).
        from services.message_processor import MessageProcessor
        assert isinstance(MessageProcessor.TOOL_COMPACTION_PROMPT, str)
        assert len(MessageProcessor.TOOL_COMPACTION_PROMPT) > 0

    def test_system_prompt_class_is_base(self):
        from services.message_processor import MessageProcessor
        from services.system_message_prompt import SystemMessagePrompt
        assert MessageProcessor.SYSTEM_PROMPT_CLASS is SystemMessagePrompt


# ─────────────────────────────────────────────────────────────────────────────
# 3. getSystemPrompt() raises TypeError when SYSTEM_PROMPT_CLASS is the abstract base
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSystemPromptDefault:
    """The module-level default ``SYSTEM_PROMPT_CLASS`` on ``MessageProcessor``
    is the abstract ``SystemMessagePrompt`` base — any subclass that forgets
    to override it must fail loudly at getSystemPrompt() call time, not
    silently produce a blank body."""

    def test_module_default_is_abstract_base(self):
        from services.message_processor import MessageProcessor
        from services.system_message_prompt import SystemMessagePrompt
        assert MessageProcessor.SYSTEM_PROMPT_CLASS is SystemMessagePrompt

    def test_abstract_default_raises_type_error(self):
        """If SYSTEM_PROMPT_CLASS is left as the abstract base, calling
        getSystemPrompt() raises TypeError — loud-failure contract."""
        from services.system_message_prompt import SystemMessagePrompt
        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=SystemMessagePrompt)
        with pytest.raises(TypeError):
            p.getSystemPrompt()

    def test_concrete_override_produces_expected_format(self):
        from services.system_message_prompt import SystemMessagePrompt

        class _Concrete(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'BODY'

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=_Concrete)
        result = p.getSystemPrompt()
        assert result == f"{_USER_DEF}\n\nBODY"

    def test_user_definition_is_first_line(self):
        from services.system_message_prompt import SystemMessagePrompt

        class _Concrete(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'BODY'

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=_Concrete)
        result = p.getSystemPrompt()
        assert result.startswith(_USER_DEF)

    def test_double_newline_separator(self):
        from services.system_message_prompt import SystemMessagePrompt

        class _Concrete(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'BODY'

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=_Concrete)
        result = p.getSystemPrompt()
        assert '\n\n' in result


# ─────────────────────────────────────────────────────────────────────────────
# 4. getSystemPrompt() with a custom SYSTEM_PROMPT_CLASS
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSystemPromptCustomClass:
    def test_custom_class_body_appended(self):
        from services.system_message_prompt import SystemMessagePrompt

        class _SentinelPrompt(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'SENTINEL_BODY'

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=_SentinelPrompt)
        result = p.getSystemPrompt()
        assert result == f"{_USER_DEF}\n\nSENTINEL_BODY"

    def test_unified_system_message_prompt_wired(self):
        """SYSTEM_PROMPT_CLASS can be set to any concrete SystemMessagePrompt subclass."""
        from services.system_message_prompt import SystemMessagePrompt

        class _MockUnified(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'UNIFIED_MOCK'

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=_MockUnified)
        result = p.getSystemPrompt()
        assert 'UNIFIED_MOCK' in result
        assert result.startswith(_USER_DEF)

    def test_fresh_instance_created_each_call(self):
        """getSystemPrompt() creates a new SYSTEM_PROMPT_CLASS instance each call."""
        call_count = {'n': 0}

        from services.system_message_prompt import SystemMessagePrompt

        class _Counter(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'body'

            def __init__(self):
                call_count['n'] += 1

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=_Counter)
        p.getSystemPrompt()
        p.getSystemPrompt()
        assert call_count['n'] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 5. getTools() — empty NATIVE_TOOLS + empty _discovered_tools → []
# ─────────────────────────────────────────────────────────────────────────────


class TestGetToolsEmpty:
    def test_returns_empty_list_when_no_tools(self):
        p = FakeProcessor.make()
        # Both NATIVE_TOOLS=[] and _discovered_tools=[] → []
        result = p.getTools()
        assert result == []

    def test_return_type_is_list(self):
        p = FakeProcessor.make()
        assert isinstance(p.getTools(), list)


# ─────────────────────────────────────────────────────────────────────────────
# 6. getTools() — native + dynamic → both returned
# ─────────────────────────────────────────────────────────────────────────────


class TestGetToolsBothSources:
    def test_native_and_dynamic_combined(self):
        _native_schema = {
            'name': 'memory',
            'description': 'Memory skill',
            'input_schema': {'type': 'object', 'properties': {}, 'required': []},
        }
        _dynamic_schema = {
            'name': 'weather_tool',
            'description': 'Weather external tool',
            'input_schema': {'type': 'object', 'properties': {}, 'required': []},
        }

        with patch(
            'services.tool_schema_service.get_skill_schemas',
            return_value=[_native_schema]
        ):
            p = FakeProcessor.make(NATIVE_TOOLS=['memory'])
            p._discovered_tools.append(_dynamic_schema)
            result = p.getTools()

        assert len(result) == 2
        names = {t['name'] for t in result}
        assert names == {'memory', 'weather_tool'}

    def test_only_dynamic_tools_when_native_empty(self):
        _dynamic = {
            'name': 'search_tool',
            'description': 'Search',
            'input_schema': {'type': 'object', 'properties': {}, 'required': []},
        }
        p = FakeProcessor.make()
        p._discovered_tools.append(_dynamic)
        result = p.getTools()
        assert len(result) == 1
        assert result[0]['name'] == 'search_tool'


# ─────────────────────────────────────────────────────────────────────────────
# 7. getTools() — same tool in native and dynamic → dedupe (first-seen wins)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetToolsDedupe:
    def test_native_wins_over_dynamic_duplicate(self):
        _native_schema = {
            'name': 'memory',
            'description': 'Native memory',
            'input_schema': {'type': 'object', 'properties': {}, 'required': []},
        }
        _dynamic_duplicate = {
            'name': 'memory',
            'description': 'Dynamic memory — should be dropped',
            'input_schema': {'type': 'object', 'properties': {}, 'required': []},
        }

        with patch(
            'services.tool_schema_service.get_skill_schemas',
            return_value=[_native_schema]
        ):
            p = FakeProcessor.make(NATIVE_TOOLS=['memory'])
            p._discovered_tools.append(_dynamic_duplicate)
            result = p.getTools()

        assert len(result) == 1
        assert result[0]['description'] == 'Native memory'

    def test_no_duplicates_within_native(self):
        _schema = {
            'name': 'memory',
            'description': 'Memory',
            'input_schema': {'type': 'object', 'properties': {}, 'required': []},
        }
        # get_skill_schemas returns the same schema twice (shouldn't happen in
        # production, but the deduplication must handle it)
        with patch(
            'services.tool_schema_service.get_skill_schemas',
            return_value=[_schema, _schema]
        ):
            p = FakeProcessor.make(NATIVE_TOOLS=['memory', 'memory'])
            result = p.getTools()

        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. getActLoopTrail() — empty → ''
# ─────────────────────────────────────────────────────────────────────────────


class TestGetActLoopTrailEmpty:
    def test_empty_returns_empty_string(self):
        p = FakeProcessor.make()
        assert p.getActLoopTrail() == ''

    def test_type_is_str(self):
        p = FakeProcessor.make()
        assert isinstance(p.getActLoopTrail(), str)


# ─────────────────────────────────────────────────────────────────────────────
# 9. getActLoopTrail() — multi-entry → newline-joined string
# ─────────────────────────────────────────────────────────────────────────────


class TestGetActLoopTrailMulti:
    def test_single_entry(self):
        p = FakeProcessor.make()
        p._act_trail.append('[memory(radius=0.5)] User prefers dark mode')
        assert p.getActLoopTrail() == '[memory(radius=0.5)] User prefers dark mode'

    def test_multiple_entries_joined_by_newline(self):
        p = FakeProcessor.make()
        p._act_trail.append('[memory(radius=0.5)] A')
        p._act_trail.append('[read(url=example.com)] B')
        p._act_trail.append('[tool_synthesis] C')
        result = p.getActLoopTrail()
        assert result == '[memory(radius=0.5)] A\n[read(url=example.com)] B\n[tool_synthesis] C'

    def test_newline_not_appended_at_end(self):
        p = FakeProcessor.make()
        p._act_trail.append('A')
        p._act_trail.append('B')
        assert not p.getActLoopTrail().endswith('\n')


# ─────────────────────────────────────────────────────────────────────────────
# 10. getDynamicTools() returns self._discovered_tools (identity check)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetDynamicTools:
    def test_returns_same_list_object(self):
        p = FakeProcessor.make()
        assert p.getDynamicTools() is p._discovered_tools

    def test_mutations_are_reflected(self):
        p = FakeProcessor.make()
        schema = {'name': 'foo', 'description': 'bar', 'input_schema': {}}
        p._discovered_tools.append(schema)
        assert p.getDynamicTools() == [schema]

    def test_empty_by_default(self):
        p = FakeProcessor.make()
        assert p.getDynamicTools() == []


# ─────────────────────────────────────────────────────────────────────────────
# 11. getPreviousMessages() — empty channel → ''
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesEmpty:
    def test_empty_channel_returns_empty_string(self, db):
        """No transcript rows and no compaction → empty string."""
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert result == ''

    def test_returns_str_type(self, db):
        p = FakeProcessor.make()
        assert isinstance(p.getPreviousMessages(), str)


# ─────────────────────────────────────────────────────────────────────────────
# 12. getPreviousMessages() — transcript rows + ephemeral filtering
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesTranscript:
    def _seed_transcript(self, db, channel='test_channel'):
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
        # North star format: lowercase 'user:' for transcript input rows.
        self._seed_transcript(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'user: Hello world' in result
        assert 'USER: Hello world' not in result

    def test_assistant_row_appears(self, db):
        # North star format: title-case 'Assistant:' (capital A) — the sole
        # exception to the lowercase input-row rule.
        self._seed_transcript(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Assistant: Hi there' in result
        assert 'ASSISTANT: Hi there' not in result

    def test_durable_tool_call_appears(self, db):
        # North star format: bare `[tool_name(k=v;k=v)] result`, no timestamp
        # prefix, no TOOL() wrapper. Empty params dict → `[memory()] …`.
        self._seed_transcript(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert '[memory()] User likes dark mode' in result
        assert 'TOOL(memory)' not in result

    def test_ephemeral_tool_call_does_not_appear(self, db):
        self._seed_transcript(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Web page content' not in result

    def test_timestamp_format(self, db):
        self._seed_transcript(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        # Expect [YYYY-MM-DD HH:MM] prefix
        assert '[2026-04-10 10:00]' in result

    def test_rows_ordered_oldest_first(self, db):
        self._seed_transcript(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        user_pos = result.index('user: Hello world')
        asst_pos = result.index('Assistant: Hi there')
        assert user_pos < asst_pos

    def test_other_channel_rows_excluded(self, db):
        """Rows from a different channel must not appear."""
        self._seed_transcript(db, channel='test_channel')
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('other_channel', 'user', 'Other channel content', '2026-04-10 10:00:00')"
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Other channel content' not in result


# ─────────────────────────────────────────────────────────────────────────────
# 13. getPreviousMessages() — compaction row causes prepend + watermark
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesCompaction:
    def _seed_with_compaction(self, db, channel='test_channel'):
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
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert result.startswith('COMPACTED: previous context here')

    def test_pre_watermark_rows_not_re_rendered(self, db):
        self._seed_with_compaction(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Old message before compaction' not in result

    def test_post_watermark_rows_appear(self, db):
        self._seed_with_compaction(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'New message after compaction' in result
        assert 'New reply after compaction' in result

    def test_compacted_text_before_new_rows(self, db):
        self._seed_with_compaction(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        compacted_pos = result.index('COMPACTED:')
        new_msg_pos = result.index('New message after compaction')
        assert compacted_pos < new_msg_pos

    def test_token_budget_ignored_in_commit2(self, db):
        """token_budget is accepted but silently ignored in Commit 2."""
        self._seed_with_compaction(db)
        p = FakeProcessor.make()
        # Should not raise, budget is ignored
        result_no_budget = p.getPreviousMessages()
        result_with_budget = p.getPreviousMessages(token_budget=100)
        assert result_no_budget == result_with_budget


# ─────────────────────────────────────────────────────────────────────────────
# 14. Abstract methods raise; postTurn is a no-op
# ─────────────────────────────────────────────────────────────────────────────


class TestAbstractStubs:
    def test_send_is_callable(self, db):
        """send() is wired in Commit 6 — no longer raises NotImplementedError."""
        from unittest.mock import patch
        from services.llm_service import LLMResponse
        p = FakeProcessor.make()
        fake_response = LLMResponse(text='ok', model='m', provider='p', tool_calls=None)
        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.return_value = fake_response
            result = p.send()
        assert isinstance(result, str)

    def test_store_is_callable(self, db):
        """store() is wired in Commit 4 — it calls DB, no longer raises NotImplementedError."""
        p = FakeProcessor.make()
        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('final response')
        assert p._uid is not None

    def test_post_turn_is_noop(self):
        p = FakeProcessor.make()
        # Must not raise and must return None
        result = p.postTurn()
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 14a. _run_memory_seed() — Commit 5 base no-op slot
# ─────────────────────────────────────────────────────────────────────────────


class TestRunMemorySeedBaseNoop:
    """Commit 5 — the base ``_run_memory_seed()`` hook is a pure no-op.

    Exists so ``send()`` (Commit 6) can call it unconditionally. User-channel
    subclasses override in Commit 8 to run the memory auto-seed. Background
    channels (DMN, goal-pursuit, scheduled) inherit the no-op.
    """

    def test_returns_none(self):
        p = FakeProcessor.make()
        result = p._run_memory_seed()
        assert result is None

    def test_does_not_raise(self):
        """Must be a no-op — no exceptions, no side effects."""
        p = FakeProcessor.make()
        # Should not raise for any call shape
        p._run_memory_seed()
        p._run_memory_seed()  # idempotent
        p._run_memory_seed()  # still fine

    def test_does_not_mutate_memory_seed(self):
        """Base no-op must leave ``_memory_seed`` as constructor default (None)."""
        p = FakeProcessor.make()
        assert p._memory_seed is None  # constructor default
        p._run_memory_seed()
        assert p._memory_seed is None  # still None after no-op

    def test_does_not_touch_pending_tool_calls(self):
        """Base no-op must not append any DTO to ``_pending_tool_calls``.

        Subclass overrides in Commit 8 will append a durable ``memory`` DTO;
        the base must not, because background channels (DMN/goal/scheduled)
        inherit this and should never produce a spurious memory row.
        """
        p = FakeProcessor.make()
        assert p._pending_tool_calls == []
        p._run_memory_seed()
        assert p._pending_tool_calls == []

    def test_does_not_touch_act_trail(self):
        """Base no-op must not append anything to ``_act_trail`` either."""
        p = FakeProcessor.make()
        assert p._act_trail == []
        p._run_memory_seed()
        assert p._act_trail == []

    def test_is_overridable(self):
        """Subclasses can override with custom behaviour without special
        machinery — confirm the hook is a plain instance method."""
        from services.message_processor import MessageProcessor

        class SeedingProcessor(MessageProcessor):
            CHANNEL = 'seeding'
            ROLE = 'user'
            SYSTEM_PROMPT_CLASS = _stub_prompt_cls()

            def getUserDefinition(self) -> str:
                return 'def'

            def getUserPrompt(self) -> str:
                return 'prompt'

            def _run_memory_seed(self) -> None:
                self._memory_seed = 'seeded'

        p = SeedingProcessor('raw', {})
        assert p._memory_seed is None
        p._run_memory_seed()
        assert p._memory_seed == 'seeded'

    def test_base_preserves_noop_when_subclass_does_not_override(self):
        """Subclass that does NOT override inherits the base no-op unchanged."""
        p = FakeProcessor.make()
        p._memory_seed = 'pre-existing'  # something else set it
        p._run_memory_seed()
        # Base no-op must not clobber pre-existing state
        assert p._memory_seed == 'pre-existing'


# ─────────────────────────────────────────────────────────────────────────────
# 14b. handleTool() — Commit 3 basic dispatch tests
#      (These complement the exhaustive suite in
#       test_message_processor_v2_handle_tool.py)
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolBasic:
    """Basic handleTool() contract wired in Commit 3."""

    def test_handle_tool_dispatches(self):
        """Happy path: calls dispatch_action once with correct args, appends
        DTO + trail line, and returns the result string."""
        from unittest.mock import patch

        fake_dispatch = {'result': 'Memory result text', 'status': 'success'}
        p = FakeProcessor.make()

        with patch(
            'services.act_dispatcher_service.ActDispatcherService'
        ) as MockDispatcher:
            MockDispatcher.return_value.dispatch_action.return_value = fake_dispatch
            result = p.handleTool({'name': 'memory', 'input': {'query': 'hello'}})

        MockDispatcher.return_value.dispatch_action.assert_called_once_with(
            FakeProcessor._CHANNEL, {'type': 'memory', 'query': 'hello'}
        )
        assert result == 'Memory result text'
        assert len(p._pending_tool_calls) == 1
        assert len(p._act_trail) == 1

    def test_handle_tool_returns_string(self):
        """Return type is always str even when dispatch result is non-string."""
        from unittest.mock import patch

        for raw_result in [None, 42, {'nested': 'dict'}, 3.14]:
            p = FakeProcessor.make()
            fake_dispatch = {'result': raw_result, 'status': 'success'}
            with patch('services.act_dispatcher_service.ActDispatcherService') as MockDispatcher:
                MockDispatcher.return_value.dispatch_action.return_value = fake_dispatch
                result = p.handleTool({'name': 'memory', 'input': {}})
            assert isinstance(result, str), (
                f"handleTool() must return str, got {type(result).__name__} for raw={raw_result!r}"
            )

    def test_handle_tool_traps_dispatch_errors(self):
        """When dispatch raises, result is ERROR string, DTO + trail still appended,
        no exception propagates."""
        from unittest.mock import patch

        p = FakeProcessor.make()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockDispatcher:
            MockDispatcher.return_value.dispatch_action.side_effect = RuntimeError('boom')
            result = p.handleTool({'name': 'memory', 'input': {}})

        assert result.startswith('ERROR: memory failed:')
        assert 'boom' in result
        assert len(p._pending_tool_calls) == 1
        assert len(p._act_trail) == 1
        # Result captured in DTO matches what was returned
        assert p._pending_tool_calls[0]['result'] == result


# ─────────────────────────────────────────────────────────────────────────────
# 15. getUserPrompt() / getUserDefinition() raise on bare base
# ─────────────────────────────────────────────────────────────────────────────


class TestAbstractMethodsOnBase:
    def test_get_user_prompt_raises_on_base(self):
        from services.message_processor import MessageProcessor
        p = MessageProcessor.__new__(MessageProcessor)
        with pytest.raises(NotImplementedError):
            p.getUserPrompt()

    def test_get_user_definition_raises_on_base(self):
        from services.message_processor import MessageProcessor
        p = MessageProcessor.__new__(MessageProcessor)
        with pytest.raises(NotImplementedError):
            p.getUserDefinition()

    def test_fake_subclass_does_not_raise(self):
        p = FakeProcessor.make()
        # Should not raise
        assert p.getUserDefinition() == _USER_DEF
        assert p.getUserPrompt() == _USER_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# 16. Per-turn instance isolation
# ─────────────────────────────────────────────────────────────────────────────


class TestInstanceIsolation:
    def test_act_trail_is_independent(self):
        p1 = FakeProcessor.make()
        p2 = FakeProcessor.make()
        p1._act_trail.append('item from p1')
        assert p2._act_trail == []

    def test_pending_tool_calls_is_independent(self):
        p1 = FakeProcessor.make()
        p2 = FakeProcessor.make()
        p1._pending_tool_calls.append({'name': 'memory'})
        assert p2._pending_tool_calls == []

    def test_discovered_tools_is_independent(self):
        p1 = FakeProcessor.make()
        p2 = FakeProcessor.make()
        p1._discovered_tools.append({'name': 'weather_tool'})
        assert p2._discovered_tools == []

    def test_memory_seed_is_independent(self):
        p1 = FakeProcessor.make()
        p2 = FakeProcessor.make()
        p1._memory_seed = 'User likes pizza'
        assert p2._memory_seed is None

    def test_uid_is_independent(self):
        p1 = FakeProcessor.make()
        p2 = FakeProcessor.make()
        p1._uid = 42
        assert p2._uid is None

    def test_class_level_mutation_does_not_leak(self):
        """The mutable defaults (NATIVE_TOOLS=[]) must not be shared instances."""
        Cls = FakeProcessor.cls()
        p1 = Cls('input1')
        p2 = Cls('input2')

        # Instance-level lists must be independent of class-level defaults
        # (_act_trail etc are created fresh in __init__)
        p1._act_trail.append('x')
        assert p2._act_trail == []


# ─────────────────────────────────────────────────────────────────────────────
# 18–44  Gap tests added by Tester (Commit 2 supplement)
#
# Categories:
#   18–21  getSystemPrompt() deep coverage
#   22–27  getTools() deep coverage
#   28–38  getPreviousMessages() — hardest method
#   39–42  Stubs (specific exception classes)
#   43–44  Instance isolation — cross-instance list safety
#   45     Import safety — no heavy top-level dependencies
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 18. getSystemPrompt() — real DMNSystemMessagePrompt wired via SYSTEM_PROMPT_CLASS
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSystemPromptFileBackedDMN:
    """Wire a real concrete subclass (DMNSystemMessagePrompt) and verify the
    output contains both getUserDefinition() and the inlined prompt body.

    DMNSystemMessagePrompt returns an inlined Python constant — no file I/O,
    no fallbacks. The body is guaranteed non-empty by its class contract.
    """

    def test_output_contains_user_definition(self):
        from services.system_message_prompt import DMNSystemMessagePrompt

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=DMNSystemMessagePrompt)
        result = p.getSystemPrompt()
        assert result.startswith(_USER_DEF), (
            "getUserDefinition() must be the first line of getSystemPrompt()"
        )

    def test_output_contains_dmn_prompt_body(self):
        from services.system_message_prompt import DMNSystemMessagePrompt

        # The DMN prompt body is either the on-disk file or the legacy fallback.
        # Both contain the word "Chalie" — safe sentinel that does not drift.
        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=DMNSystemMessagePrompt)
        result = p.getSystemPrompt()
        assert 'Chalie' in result

    def test_double_newline_separator_present(self):
        from services.system_message_prompt import DMNSystemMessagePrompt

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=DMNSystemMessagePrompt)
        result = p.getSystemPrompt()
        assert '\n\n' in result

    def test_user_def_followed_by_double_newline(self):
        from services.system_message_prompt import DMNSystemMessagePrompt

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=DMNSystemMessagePrompt)
        result = p.getSystemPrompt()
        # The exact separator is '{user_def}\n\n{body}'
        assert result[len(_USER_DEF): len(_USER_DEF) + 2] == '\n\n'


# ─────────────────────────────────────────────────────────────────────────────
# 19. getSystemPrompt() — UnifiedSystemMessagePrompt zero-arg construction
#     Verifies the base passes no positional args to SYSTEM_PROMPT_CLASS()
#     so UnifiedSystemMessagePrompt's defaults (original_prompt='', thread_id=None)
#     are used.  This is the Commit 2 contract; Commit 8 will pass explicit args.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSystemPromptUnifiedZeroArgContract:
    """Lock the Commit 2 contract: SYSTEM_PROMPT_CLASS() is called with zero args."""

    def test_unified_constructed_with_zero_args(self):
        """If UnifiedSystemMessagePrompt required positional args, this would raise TypeError."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=UnifiedSystemMessagePrompt)
        # Must not raise — UnifiedSystemMessagePrompt has default params only
        result = p.getSystemPrompt()
        assert result.startswith(_USER_DEF)

    def test_unified_returns_non_empty_body_with_real_db(self, db):
        """With a real DB (migrations applied), UnifiedSystemMessagePrompt.getPrompt()
        returns the UNIFIED template, not an empty string.

        This test wires Commit 1 into Commit 2's socket and catches any
        construction-args mismatch between the two classes.
        """
        from services.system_message_prompt import UnifiedSystemMessagePrompt

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=UnifiedSystemMessagePrompt)
        result = p.getSystemPrompt()
        # The UNIFIED template is non-trivial (>200 chars), so body is not ''
        body = result[len(_USER_DEF) + 2:]  # strip user_def + '\n\n'
        assert len(body) > 100, f"Expected non-empty UNIFIED body, got: {body!r}"

    def test_format_is_user_def_newline_newline_body(self, db):
        """Exact format regression: '{user_def}\\n\\n{body}'."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=UnifiedSystemMessagePrompt)
        result = p.getSystemPrompt()
        prefix = f"{_USER_DEF}\n\n"
        assert result.startswith(prefix)


# ─────────────────────────────────────────────────────────────────────────────
# 20. getSystemPrompt() — SYSTEM_PROMPT_CLASS that raises on instantiation
#     The exception must propagate; there must be no silent fallback.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSystemPromptExceptionPropagation:
    def test_raises_when_system_prompt_class_init_raises(self):
        from services.system_message_prompt import SystemMessagePrompt

        class _Exploding(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'unreachable'

            def __init__(self):
                raise RuntimeError("deliberate explosion in SYSTEM_PROMPT_CLASS")

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=_Exploding)
        with pytest.raises(RuntimeError, match="deliberate explosion"):
            p.getSystemPrompt()

    def test_raises_when_get_prompt_raises(self):
        from services.system_message_prompt import SystemMessagePrompt

        class _ExplodingPrompt(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'unreachable'

            def getPrompt(self) -> str:
                raise ValueError("getPrompt exploded")

        p = FakeProcessor.make(SYSTEM_PROMPT_CLASS=_ExplodingPrompt)
        with pytest.raises(ValueError, match="getPrompt exploded"):
            p.getSystemPrompt()


# ─────────────────────────────────────────────────────────────────────────────
# 21. getSystemPrompt() — fresh instance per call (regression lock)
#     Already tested (test class 4), but included here for completeness
#     alongside the other gap categories.
# (Skipped as duplicate — test_fresh_instance_created_each_call covers it)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 22. getTools() — non-existent skill name in NATIVE_TOOLS is silently dropped
# ─────────────────────────────────────────────────────────────────────────────


class TestGetToolsNonExistentSkill:
    """get_skill_schemas() silently drops unknown names (logs a warning).
    getTools() must propagate that behaviour — no exception, empty result.
    """

    def test_unknown_skill_silently_dropped(self, caplog):
        import logging

        p = FakeProcessor.make(NATIVE_TOOLS=['__does_not_exist__'])
        with caplog.at_level(logging.WARNING, logger='services.tool_schema_service'):
            result = p.getTools()
        assert result == []

    def test_unknown_skill_emits_warning(self, caplog):
        import logging

        p = FakeProcessor.make(NATIVE_TOOLS=['__ghost_skill__'])
        with caplog.at_level(logging.WARNING, logger='services.tool_schema_service'):
            p.getTools()
        assert any('__ghost_skill__' in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# 23. getTools() — real skill name returns expected top-level keys
# ─────────────────────────────────────────────────────────────────────────────


class TestGetToolsRealSkill:
    """find_tools is a cognitive primitive available in all environments
    (its handler module requires no DB).  Use it as the cheapest real skill.
    """

    def test_find_tools_returns_name_key(self):
        from services.tool_schema_service import clear_cache
        clear_cache()

        p = FakeProcessor.make(NATIVE_TOOLS=['find_tools'])
        result = p.getTools()
        assert len(result) == 1
        assert result[0]['name'] == 'find_tools'

    def test_find_tools_returns_description_key(self):
        from services.tool_schema_service import clear_cache
        clear_cache()

        p = FakeProcessor.make(NATIVE_TOOLS=['find_tools'])
        result = p.getTools()
        assert 'description' in result[0]
        assert isinstance(result[0]['description'], str)

    def test_find_tools_returns_input_schema_key(self):
        from services.tool_schema_service import clear_cache
        clear_cache()

        p = FakeProcessor.make(NATIVE_TOOLS=['find_tools'])
        result = p.getTools()
        assert 'input_schema' in result[0]
        assert isinstance(result[0]['input_schema'], dict)


# ─────────────────────────────────────────────────────────────────────────────
# 24. getTools() — native tool wins when dynamic has same name
#     (already tested in TestGetToolsDedupe but with mocked get_skill_schemas;
#      this variant uses a real skill to lock the behavior end-to-end)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetToolsNativeWinsOverDynamic:
    """When the same tool name appears in NATIVE_TOOLS and _discovered_tools,
    the native (first-seen) schema wins and the dynamic duplicate is dropped.
    """

    def test_native_description_is_preserved_not_dynamic(self):
        from services.tool_schema_service import clear_cache
        clear_cache()

        # Inject a dynamic tool with the same name but a different description
        dynamic_duplicate = {
            'name': 'find_tools',
            'description': 'DYNAMIC OVERRIDE — must be dropped',
            'input_schema': {'type': 'object', 'properties': {}, 'required': []},
        }
        p = FakeProcessor.make(NATIVE_TOOLS=['find_tools'])
        p._discovered_tools.append(dynamic_duplicate)
        result = p.getTools()

        assert len(result) == 1
        assert result[0]['description'] != 'DYNAMIC OVERRIDE — must be dropped'


# ─────────────────────────────────────────────────────────────────────────────
# 25. getTools() — getDynamicTools() is called exactly once per getTools() call
# ─────────────────────────────────────────────────────────────────────────────


class TestGetToolsDynamicToolsCalledOnce:
    """getDynamicTools() must be called exactly once per getTools() invocation.

    Commit 6's send() loop calls getTools() once per iteration; if getDynamicTools()
    were called multiple times per iteration that would compound needlessly.
    """

    def test_get_dynamic_tools_called_once(self):
        from services.message_processor import MessageProcessor

        call_count = {'n': 0}

        class _Counting(MessageProcessor):
            CHANNEL = 'test'
            ROLE = 'user'

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return _USER_PROMPT

            def getDynamicTools(self):
                call_count['n'] += 1
                return []

        p = _Counting('input')
        p.getTools()
        assert call_count['n'] == 1

    def test_get_dynamic_tools_called_once_per_invocation(self):
        """Two separate getTools() calls → two getDynamicTools() calls (not four)."""
        from services.message_processor import MessageProcessor

        call_count = {'n': 0}

        class _Counting(MessageProcessor):
            CHANNEL = 'test'
            ROLE = 'user'

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return _USER_PROMPT

            def getDynamicTools(self):
                call_count['n'] += 1
                return []

        p = _Counting('input')
        p.getTools()
        p.getTools()
        assert call_count['n'] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 26. getTools() — empty NATIVE_TOOLS + empty _discovered_tools → [] not None
# ─────────────────────────────────────────────────────────────────────────────


class TestGetToolsEmptyIsNotNone:
    def test_returns_list_not_none(self):
        p = FakeProcessor.make()
        result = p.getTools()
        assert result is not None
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# 27. getTools() — mutating returned list does not affect _discovered_tools
#     (contract: identity — no defensive copy, but mutation of the RETURN VALUE
#      must not silently corrupt the internal state through a hidden alias)
#
# The current implementation builds a fresh `result` list in getTools(), so
# mutating the returned list cannot corrupt _discovered_tools.  This test
# locks that invariant so a future refactor that returns a reference would be
# caught immediately.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetToolsReturnValueIsolation:
    def test_mutating_returned_list_does_not_affect_discovered_tools(self):
        schema = {'name': 'my_tool', 'description': 'test', 'input_schema': {}}
        p = FakeProcessor.make()
        p._discovered_tools.append(schema)

        returned = p.getTools()
        returned.clear()  # mutate the returned list

        # _discovered_tools must be untouched
        assert len(p._discovered_tools) == 1
        assert p._discovered_tools[0]['name'] == 'my_tool'


# ─────────────────────────────────────────────────────────────────────────────
# 28. getPreviousMessages() — compaction exists, zero transcript rows above watermark
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesCompactionOnlyNoNewRows:
    """Watermark is at or above all transcript rows → output is only compacted_text."""

    def test_returns_compacted_text_alone(self, db):
        channel = 'test_channel'
        # Insert a transcript row
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Old message', '2026-04-09 09:00:00')",
            (channel,)
        )
        old_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        # Set compaction watermark at the row we just inserted
        db.execute(
            "INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, 'COMPACTED: all prior context', ?, 10, '2026-04-09 10:00:00')",
            (channel, old_id)
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()

        # Output is the compacted text only — not the old transcript row
        assert result == 'COMPACTED: all prior context'
        assert 'Old message' not in result

    def test_no_trailing_newline_when_only_compaction(self, db):
        channel = 'test_channel'
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

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        # '\n'.join(['COMPACT BLOCK']) has no leading/trailing newline
        assert result == 'COMPACT BLOCK'


# ─────────────────────────────────────────────────────────────────────────────
# 29. getPreviousMessages() — compaction + one new transcript row above watermark
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesCompactionPlusOneRow:
    def test_compaction_followed_by_new_row(self, db):
        channel = 'test_channel'
        # Old row that will become the watermark
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Pre-compaction message', '2026-04-09 08:00:00')",
            (channel,)
        )
        watermark_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        # New row after watermark
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

        p = FakeProcessor.make()
        result = p.getPreviousMessages()

        assert result.startswith('COMPACTION SUMMARY')
        assert 'Post-compaction message' in result
        assert 'Pre-compaction message' not in result

    def test_compaction_before_new_row_in_output(self, db):
        channel = 'test_channel'
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

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        compact_pos = result.index('COMPACT_START')
        new_row_pos = result.index('Fresh reply')
        assert compact_pos < new_row_pos


# ─────────────────────────────────────────────────────────────────────────────
# 30. getPreviousMessages() — multiple channels, no cross-channel leakage
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesChannelIsolation:
    """Each processor only sees its own CHANNEL's rows."""

    def _seed_all_channels(self, db):
        for channel in ('user', 'dmn', 'goal_pursuit', 'scheduled', 'test_channel'):
            db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES (?, 'user', ?, '2026-04-10 10:00:00')",
                (channel, f"Content from {channel}")
            )
        db.commit()

    def test_test_channel_does_not_see_user_channel(self, db):
        self._seed_all_channels(db)
        p = FakeProcessor.make()  # CHANNEL='test_channel'
        result = p.getPreviousMessages()
        assert 'Content from user' not in result
        assert 'Content from test_channel' in result

    def test_test_channel_does_not_see_dmn_channel(self, db):
        self._seed_all_channels(db)
        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Content from dmn' not in result

    def test_test_channel_does_not_see_goal_pursuit_channel(self, db):
        self._seed_all_channels(db)
        p = FakeProcessor.make()
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

                def getUserDefinition(self):
                    return _USER_DEF

                def getUserPrompt(self):
                    return _USER_PROMPT

            # Capture channel in closure via default argument
            p = _Chan('input')
            result = p.getPreviousMessages()
            assert f'Content from {channel}' in result, (
                f"Channel {channel!r} should see its own content"
            )
            # Must NOT see any other channel's content
            for other in ('user', 'dmn', 'goal_pursuit', 'scheduled', 'test_channel'):
                if other != channel:
                    assert f'Content from {other}' not in result, (
                        f"Channel {channel!r} leaked content from {other!r}"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# 31. getPreviousMessages() — durable tool_call interleaving order
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesDurableToolInterleaving:
    """Two durable tool_calls linked to the same transcript row must appear
    immediately after that row, ordered by created_at (the SQL ORDER BY in
    get_by_transcript_ids).
    """

    def test_two_durable_tool_calls_appear_under_their_transcript_row(self, db):
        channel = 'test_channel'
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

        p = FakeProcessor.make()
        result = p.getPreviousMessages()

        assert '[memory()] Result from memory' in result
        assert '[find_tools()] Found weather tool' in result
        assert 'TOOL(memory)' not in result
        assert 'TOOL(find_tools)' not in result

        # Both appear after the transcript row
        input_pos = result.index('The input row')
        memory_pos = result.index('[memory()] Result from memory')
        find_pos = result.index('[find_tools()] Found weather tool')
        assert input_pos < memory_pos < find_pos

    def test_tool_calls_ordered_by_timestamp(self, db):
        """Earlier created_at → appears first in output."""
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Input', '2026-04-10 10:00:00')",
            (channel,)
        )
        tid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        # Insert in reverse timestamp order to verify ORDER BY created_at
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

        p = FakeProcessor.make()
        result = p.getPreviousMessages()

        first_pos = result.index('[first_tool()] First result')
        second_pos = result.index('[second_tool()] Second result')
        assert first_pos < second_pos


# ─────────────────────────────────────────────────────────────────────────────
# 32. getPreviousMessages() — ephemeral=1 for same transcript row is dropped
#     even when a durable sibling is present for the same row
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesEphemeralSiblingDropped:
    def test_ephemeral_sibling_dropped_when_durable_present(self, db):
        channel = 'test_channel'
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

        p = FakeProcessor.make()
        result = p.getPreviousMessages()

        assert 'DURABLE_RESULT' in result
        assert 'EPHEMERAL_RESULT' not in result


# ─────────────────────────────────────────────────────────────────────────────
# 33. getPreviousMessages() — exact timestamp format regression
#     Input: '2026-04-10T14:03:27+00:00'  Output prefix: '[2026-04-10 14:03]'
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesTimestampFormat:
    def test_iso_timestamp_with_offset_formatted_correctly(self, db):
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Timestamp test', '2026-04-10T14:03:27+00:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert '[2026-04-10 14:03]' in result

    def test_no_seconds_in_timestamp(self, db):
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Seconds check', '2026-04-10T14:03:27+00:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        # Seconds (:27) must not appear in the timestamp block
        assert '14:03:27' not in result

    def test_naive_sqlite_timestamp_handled(self, db):
        """Legacy rows use SQLite's naive format '2026-04-10 14:03:27'.
        parse_utc handles this gracefully — getPreviousMessages() must not crash.
        """
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Legacy naive timestamp', '2026-04-10 14:03:27')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        # Must not raise
        result = p.getPreviousMessages()
        assert 'Legacy naive timestamp' in result
        assert '[2026-04-10 14:03]' in result


# ─────────────────────────────────────────────────────────────────────────────
# 34. getPreviousMessages() — role rendering per north star
#
# North star § Literal format:
#   - Transcript input rows (user, proactive_thought, goal_pursuit, scheduled)
#     render with lowercase role labels.
#   - The assistant row is the sole exception — rendered title case 'Assistant:'.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesRoleCase:
    """Lock the role rendering convention: lowercase for inputs, title-case
    for assistant only."""

    def test_user_role_rendered_lowercase(self, db):
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Hello', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'user: Hello' in result
        assert 'USER: Hello' not in result

    def test_assistant_role_rendered_titlecase(self, db):
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'assistant', 'Hi there', '2026-04-10 10:01:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'Assistant: Hi there' in result
        assert 'ASSISTANT: Hi there' not in result
        assert 'assistant: Hi there' not in result

    def test_arbitrary_role_rendered_lowercase(self, db):
        """Any non-assistant role string (e.g. proactive_thought) stays
        lowercase — only 'assistant' is title-cased."""
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'proactive_thought', 'Background thought', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'proactive_thought: Background thought' in result
        assert 'PROACTIVE_THOUGHT: Background thought' not in result

    def test_goal_pursuit_role_rendered_lowercase(self, db):
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'goal_pursuit', 'Pursuit tick', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'goal_pursuit: Pursuit tick' in result

    def test_scheduled_role_rendered_lowercase(self, db):
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'scheduled', 'Timer fired', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        assert 'scheduled: Timer fired' in result


# ─────────────────────────────────────────────────────────────────────────────
# 35. getPreviousMessages() — token_budget parameter accepted and ignored
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesTokenBudget:
    """token_budget is a Commit 7 stub — accepted without error, output identical."""

    def test_none_and_integer_budget_produce_identical_output(self, db):
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Budget test', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        without_budget = p.getPreviousMessages()
        with_budget = p.getPreviousMessages(token_budget=100)
        assert without_budget == with_budget

    def test_large_budget_does_not_change_output(self, db):
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Large budget test', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        result_no_budget = p.getPreviousMessages()
        result_large_budget = p.getPreviousMessages(token_budget=999999)
        assert result_no_budget == result_large_budget


# ─────────────────────────────────────────────────────────────────────────────
# 36. getPreviousMessages() — empty channel returns '' (explicit assertion
#     that '' is returned even when there are rows for OTHER channels)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesEmptyChannelWithOtherData:
    def test_returns_empty_when_no_rows_for_this_channel(self, db):
        # Seed a different channel — test_channel processor must still return ''
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('other_channel', 'user', 'Not mine', '2026-04-10 10:00:00')"
        )
        db.commit()

        p = FakeProcessor.make()  # CHANNEL='test_channel'
        result = p.getPreviousMessages()
        assert result == ''


# ─────────────────────────────────────────────────────────────────────────────
# 37. getPreviousMessages() — null/empty role handled gracefully
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesNullRole:
    """Verify the `or 'unknown'` guard in getPreviousMessages() for the role field.

    The transcript.role column has a NOT NULL constraint, so a database-level NULL
    cannot be inserted.  Instead, test that when the Python dict returned by
    get_recent() has role=None (which could happen if a future schema change or
    custom query omits the column), the fallback renders cleanly as lowercase
    'unknown' — matching the north star input-row case convention (lowercase
    for everything except 'assistant').

    We verify this by patching transcript_service.get_recent to return a row
    with role=None, which exercises the exact code path:
        raw_role = (entry.get('role') or 'unknown')
        role_label = 'Assistant' if raw_role == 'assistant' else raw_role
    """

    def test_none_role_falls_back_to_unknown_lowercase(self):
        """If get_recent() ever returns a row dict with role=None,
        getPreviousMessages() must render it as 'unknown:' (lowercase)
        without crashing.
        """
        from unittest.mock import patch

        fake_row = {
            'id': 1,
            'role': None,          # defensive path: role is missing
            'content': 'Null role content',
            'created_at': '2026-04-10 10:00:00',
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None):
            p = FakeProcessor.make()
            result = p.getPreviousMessages()

        assert 'Null role content' in result
        assert 'unknown:' in result
        assert 'UNKNOWN:' not in result


# ─────────────────────────────────────────────────────────────────────────────
# 38. getPreviousMessages() — empty content row does not raise
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesEmptyContent:
    def test_empty_content_row_rendered_without_crash(self, db):
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', '', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        p = FakeProcessor.make()
        result = p.getPreviousMessages()
        # Row is rendered — empty content means 'user: ' (lowercase, trailing space)
        assert 'user: ' in result
        assert 'USER: ' not in result


# ─────────────────────────────────────────────────────────────────────────────
# 39–42. Stubs — specific exception classes
# ─────────────────────────────────────────────────────────────────────────────


class TestStubExceptionTypes:
    """Verify stubs raise the exact exception class (NotImplementedError),
    not a broader Exception or a subclass.

    Note: handleTool() is no longer a stub (wired in Commit 3).
          store() is no longer a stub (wired in Commit 4).
          send() is no longer a stub (wired in Commit 6).
    """

    def test_send_is_wired_in_commit_6(self, db):
        """send() is wired in Commit 6 — calls provider, does not raise NotImplementedError."""
        from unittest.mock import patch
        from services.llm_service import LLMResponse
        p = FakeProcessor.make()
        fake_response = LLMResponse(text='done', model='m', provider='p', tool_calls=None)
        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.return_value = fake_response
            result = p.send()
        assert result == 'done'

    def test_store_is_wired_in_commit_4(self, db):
        """store() is wired in Commit 4 — calls DB, does not raise NotImplementedError."""
        p = FakeProcessor.make()
        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('final response')
        assert p._uid is not None

    def test_post_turn_returns_none_not_not_implemented(self):
        """postTurn() is overridable, NOT a stub — must return None, not raise."""
        p = FakeProcessor.make()
        result = p.postTurn()
        assert result is None

    def test_send_accepts_request_id(self, db):
        """send() accepts an optional request_id parameter (Commit 6 contract)."""
        from unittest.mock import patch
        from services.llm_service import LLMResponse
        p = FakeProcessor.make()
        fake_response = LLMResponse(text='ok', model='m', provider='p', tool_calls=None)
        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.return_value = fake_response
            result = p.send('req-id')
        assert isinstance(result, str)

    def test_store_not_caught_as_plain_exception(self, db):
        """store() is wired in Commit 4 — calls DB, no longer raises NotImplementedError."""
        p = FakeProcessor.make()
        with patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            p.store('response text')
        assert p._uid is not None


# ─────────────────────────────────────────────────────────────────────────────
# 43. Instance isolation — two instances of SAME subclass, _pending_tool_calls
# ─────────────────────────────────────────────────────────────────────────────


class TestInstanceIsolationPendingToolCalls:
    """Two instances of the same concrete subclass must not share
    _pending_tool_calls even if one was constructed before the other.
    """

    def test_pending_tool_calls_no_cross_talk(self):
        Cls = FakeProcessor.cls()
        p1 = Cls('input A')
        p2 = Cls('input B')

        p1._pending_tool_calls.append({'name': 'memory', 'result': 'A'})
        assert p2._pending_tool_calls == [], (
            "_pending_tool_calls must be instance-scoped; p2 must be empty"
        )

    def test_both_instances_can_accumulate_independently(self):
        Cls = FakeProcessor.cls()
        p1 = Cls('input A')
        p2 = Cls('input B')

        p1._pending_tool_calls.append({'name': 'tool_a', 'result': 'r1'})
        p2._pending_tool_calls.append({'name': 'tool_b', 'result': 'r2'})
        p2._pending_tool_calls.append({'name': 'tool_c', 'result': 'r3'})

        assert len(p1._pending_tool_calls) == 1
        assert len(p2._pending_tool_calls) == 2
        assert p1._pending_tool_calls[0]['name'] == 'tool_a'


# ─────────────────────────────────────────────────────────────────────────────
# 44. Instance isolation — _discovered_tools list is instance-scoped
# ─────────────────────────────────────────────────────────────────────────────


class TestInstanceIsolationDiscoveredTools:
    def test_discovered_tools_no_cross_talk(self):
        Cls = FakeProcessor.cls()
        p1 = Cls('input 1')
        p2 = Cls('input 2')

        p1._discovered_tools.append({'name': 'weather_tool', 'description': 'Weather'})
        assert p2._discovered_tools == [], (
            "_discovered_tools must be instance-scoped"
        )

    def test_clearing_one_does_not_affect_other(self):
        Cls = FakeProcessor.cls()
        p1 = Cls('input 1')
        p2 = Cls('input 2')

        p1._discovered_tools.append({'name': 'tool_x'})
        p2._discovered_tools.append({'name': 'tool_y'})

        p1._discovered_tools.clear()
        assert p2._discovered_tools == [{'name': 'tool_y'}]


# ─────────────────────────────────────────────────────────────────────────────
# 45. Import safety — no top-level imports of heavy/coupled modules
#     The module must load without pulling in LLM clients, embeddings, torch,
#     or ActDispatcherService.  This ensures the module stays lightweight and
#     does not create circular imports or slow cold starts.
# ─────────────────────────────────────────────────────────────────────────────


class TestImportSafety:
    """Inspect the module source directly to assert no blocked top-level imports.

    The blocklist represents modules that would couple the abstract base to
    concrete infrastructure, creating import-time side effects or slow starts.
    """

    _BLOCKLIST = [
        'torch',
        'ollama',
        'openai',
        'anthropic',
        'ActDispatcherService',
        'create_llm_service',
        'LLMService',
        'ProviderService',
        'EmbeddingService',
        'embedding_service',
    ]

    def test_no_blocked_top_level_imports(self):
        """Read the module source and verify none of the blocklisted identifiers
        appear in top-level import statements.
        """
        import ast
        import pathlib

        source_path = pathlib.Path(__file__).parent.parent / 'services' / 'message_processor.py'
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source)

        top_level_imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    top_level_imports.append(f"{module}.{alias.name}")
                    top_level_imports.append(alias.name)
                    top_level_imports.append(module)

        for blocked in self._BLOCKLIST:
            for imp in top_level_imports:
                assert blocked not in imp, (
                    f"Blocked import {blocked!r} found in top-level imports of "
                    f"message_processor.py: {imp!r}"
                )

    def test_lazy_imports_inside_methods_are_acceptable(self):
        """Services imported lazily inside methods (compaction_service,
        transcript_service, tool_schema_service) do not appear at the top level.
        This test confirms the design: only logging, system_message_prompt,
        and time_utils are top-level imports.
        """
        import ast
        import pathlib

        source_path = pathlib.Path(__file__).parent.parent / 'services' / 'message_processor.py'
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source)

        actual_top_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    actual_top_level.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                actual_top_level.add(module)
                for alias in node.names:
                    actual_top_level.add(alias.name)

        # Lazy services must NOT be in top-level
        lazy_services = [
            'compaction_service', 'transcript_service',
            'tool_schema_service', 'get_skill_schemas',
            'ToolCallService', 'tool_call_service',
        ]
        for svc in lazy_services:
            assert svc not in actual_top_level, (
                f"{svc!r} should be a lazy import inside a method, "
                f"not a top-level import"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 46. _format_ts() helper — direct unit tests (arbiter additions)
#
# These cover the silent-data-corruption edge case the critic flagged:
# parse_utc('') and parse_utc(None) return datetime.min (0001-01-01), which
# would otherwise render as a bogus "year 1" timestamp in Previous Messages.
# _format_ts() now returns a placeholder and logs a warning instead.
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatTsHelper:
    """Lock the _format_ts() guard against missing / unparseable timestamps."""

    def test_valid_sqlite_naive_timestamp(self):
        from services.message_processor import _format_ts
        # SQLite's naive format — parse_utc normalises the space → 'T'
        assert _format_ts('2026-04-10 14:03:27') == '2026-04-10 14:03'

    def test_valid_iso_timestamp_with_offset(self):
        from services.message_processor import _format_ts
        assert _format_ts('2026-04-10T14:03:27+00:00') == '2026-04-10 14:03'

    def test_none_returns_placeholder(self, caplog):
        import logging
        from services.message_processor import _format_ts, _MISSING_TS_PLACEHOLDER
        with caplog.at_level(logging.WARNING, logger='services.message_processor'):
            result = _format_ts(None, row_kind='transcript', row_id=42)
        assert result == _MISSING_TS_PLACEHOLDER
        # Warning was emitted referencing the row id
        assert any('42' in r.message for r in caplog.records)

    def test_empty_string_returns_placeholder(self, caplog):
        import logging
        from services.message_processor import _format_ts, _MISSING_TS_PLACEHOLDER
        with caplog.at_level(logging.WARNING, logger='services.message_processor'):
            result = _format_ts('', row_kind='transcript', row_id=7)
        assert result == _MISSING_TS_PLACEHOLDER
        assert any('7' in r.message for r in caplog.records)

    def test_whitespace_only_returns_placeholder(self, caplog):
        import logging
        from services.message_processor import _format_ts, _MISSING_TS_PLACEHOLDER
        with caplog.at_level(logging.WARNING, logger='services.message_processor'):
            result = _format_ts('   ', row_kind='tool_call', row_id=9)
        assert result == _MISSING_TS_PLACEHOLDER

    def test_unparseable_string_returns_placeholder(self, caplog):
        import logging
        from services.message_processor import _format_ts, _MISSING_TS_PLACEHOLDER
        with caplog.at_level(logging.WARNING, logger='services.message_processor'):
            result = _format_ts('not-a-timestamp', row_kind='transcript', row_id=1)
        # parse_utc returns datetime.min for unparseable → year <= 1 → placeholder
        assert result == _MISSING_TS_PLACEHOLDER

    def test_placeholder_has_correct_width(self):
        from services.message_processor import _MISSING_TS_PLACEHOLDER
        # Must be the same 16-char width as a valid 'YYYY-MM-DD HH:MM' slot
        assert len(_MISSING_TS_PLACEHOLDER) == len('2026-04-10 14:03')

    def test_placeholder_does_not_contain_year_1(self):
        from services.message_processor import _MISSING_TS_PLACEHOLDER
        # Must never accidentally render a real-looking date
        assert '0001' not in _MISSING_TS_PLACEHOLDER
        assert '1970' not in _MISSING_TS_PLACEHOLDER


# ─────────────────────────────────────────────────────────────────────────────
# 47. getPreviousMessages() — null created_at on a row renders placeholder,
#     never a bogus 0001-01-01 timestamp
#
# The transcript.created_at column has a default of CURRENT_TIMESTAMP so a
# real NULL is unlikely to hit the DB, but the critic flagged this as a
# silent data-corruption risk. This test patches transcript_service.get_recent
# to hand back a row with missing created_at and asserts the prompt stays
# LLM-safe.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesMissingCreatedAt:
    def test_missing_created_at_renders_placeholder(self):
        from unittest.mock import patch
        from services.message_processor import _MISSING_TS_PLACEHOLDER

        fake_row = {
            'id': 1,
            'role': 'user',
            'content': 'Row with no timestamp',
            'created_at': None,
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.tool_call_service.ToolCallService') as mock_tcs_cls:
            mock_tcs_cls.return_value.get_by_transcript_ids.return_value = {}
            p = FakeProcessor.make()
            result = p.getPreviousMessages()

        # Placeholder rendered instead of the parse_utc sentinel
        assert _MISSING_TS_PLACEHOLDER in result
        assert 'Row with no timestamp' in result
        # Must NOT leak the datetime.min sentinel anywhere
        assert '0001-01-01' not in result

    def test_empty_string_created_at_renders_placeholder(self):
        from unittest.mock import patch
        from services.message_processor import _MISSING_TS_PLACEHOLDER

        fake_row = {
            'id': 2,
            'role': 'user',
            'content': 'Row with empty-string timestamp',
            'created_at': '',
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.tool_call_service.ToolCallService') as mock_tcs_cls:
            mock_tcs_cls.return_value.get_by_transcript_ids.return_value = {}
            p = FakeProcessor.make()
            result = p.getPreviousMessages()

        assert _MISSING_TS_PLACEHOLDER in result
        assert '0001-01-01' not in result

    def test_missing_created_at_on_durable_tool_call_still_renders_bare(self):
        """Durable tool_calls render BARE under the north star — no timestamp
        prefix, so a null created_at on a tool_call simply drops out of the
        output. The row still renders as `[tool_name(params)] result` and
        inherits its owning transcript row's timestamp implicitly by position.
        """
        from unittest.mock import patch

        fake_row = {
            'id': 10,
            'role': 'user',
            'content': 'Good row',
            'created_at': '2026-04-10 10:00:00',
        }
        fake_tc = {
            'id': 99,
            'tool_name': 'memory',
            'params': '{}',
            'result': 'Durable result with null timestamp',
            'created_at': None,
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.tool_call_service.ToolCallService') as mock_tcs_cls:
            mock_tcs_cls.return_value.get_by_transcript_ids.return_value = {10: [fake_tc]}
            p = FakeProcessor.make()
            result = p.getPreviousMessages()

        # Transcript row renders real timestamp in north star format
        assert '[2026-04-10 10:00] user: Good row' in result
        # Tool call renders bare — no timestamp, no TOOL() wrapper
        assert '[memory()] Durable result with null timestamp' in result
        # Year-1 placeholder must never leak through
        assert '0001-01-01' not in result


# ─────────────────────────────────────────────────────────────────────────────
# 48. getPreviousMessages() — explicit since_id=0 path
#
# When no compaction exists, the watermark is 0 and get_recent() is called
# with since_id=0. transcript ids start at 1 so `id > 0` matches every row.
# Lock that we see the very first row (id=1) in the output.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesSinceIdZero:
    def test_no_compaction_returns_all_rows_including_first(self, db):
        """With no compaction, even row id=1 must surface in Previous Messages."""
        channel = 'test_channel'
        # Insert multiple rows; first one will have the lowest id in this channel
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

        p = FakeProcessor.make()
        result = p.getPreviousMessages()

        # All three rows must appear — nothing is cut off by the since_id=0 path
        assert 'First ever message' in result
        assert 'First reply' in result
        assert 'Second message' in result

    def test_since_id_zero_passed_when_no_compaction(self, db):
        """Assert the watermark is exactly 0 (not None) when no compaction exists.

        This locks the contract of the `watermark = compaction[...] if compaction else 0`
        expression. `since_id=0` + `id > 0` returns every row; `since_id=None`
        would take a different branch in transcript_service.get_recent().
        """
        from unittest.mock import patch
        channel = 'test_channel'
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES (?, 'user', 'Lock the watermark', '2026-04-10 10:00:00')",
            (channel,)
        )
        db.commit()

        captured_kwargs = {}

        real_get_recent = None
        from services import transcript_service as _tsvc
        real_get_recent = _tsvc.get_recent

        def spy(channel_arg, **kwargs):
            captured_kwargs.update(kwargs)
            return real_get_recent(channel_arg, **kwargs)

        with patch('services.transcript_service.get_recent', side_effect=spy):
            p = FakeProcessor.make()
            p.getPreviousMessages()

        assert captured_kwargs.get('since_id') == 0, (
            f"Expected since_id=0 when no compaction, got {captured_kwargs.get('since_id')!r}"
        )
        # And the limit was the class constant, not a magic number
        from services.message_processor import MessageProcessor
        assert captured_kwargs.get('limit') == MessageProcessor._TRANSCRIPT_FETCH_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# 49. _TRANSCRIPT_FETCH_LIMIT constant — sanity + overrideability
# ─────────────────────────────────────────────────────────────────────────────


class TestTranscriptFetchLimit:
    def test_default_value(self):
        from services.message_processor import MessageProcessor
        assert MessageProcessor._TRANSCRIPT_FETCH_LIMIT == 2000

    def test_is_class_level_not_instance_level(self):
        from services.message_processor import MessageProcessor
        assert '_TRANSCRIPT_FETCH_LIMIT' in vars(MessageProcessor)

    def test_subclass_can_override(self):
        """A subclass that needs a tighter fetch ceiling can narrow it."""
        from services.message_processor import MessageProcessor

        class _Narrow(MessageProcessor):
            CHANNEL = 'narrow_channel'
            ROLE = 'user'
            _TRANSCRIPT_FETCH_LIMIT = 500

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return _USER_PROMPT

        assert _Narrow._TRANSCRIPT_FETCH_LIMIT == 500
        # Base is untouched
        assert MessageProcessor._TRANSCRIPT_FETCH_LIMIT == 2000


# ─────────────────────────────────────────────────────────────────────────────
# 50. getPreviousMessages() — compaction pseudo-tool filter (Decision 4B)
#
# The `compaction` durable tool_call is an audit-only DTO. Its content is
# already replayed at the top of Previous Messages via the `compactions`
# table prepend (step 2 of the algorithm). Rendering it a second time as a
# durable tool_call under the owning input row would double-show the summary.
#
# Locked: the compaction pseudo-tool MUST be filtered at the call site,
# never at the ToolCallService layer.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesCompactionFilter:
    def test_compaction_durable_tool_call_filtered(self):
        """A durable `compaction` tool_call linked to a transcript row must
        NOT surface in Previous Messages — its content is already replayed
        via the `compactions` table prepend."""
        from unittest.mock import patch

        fake_row = {
            'id': 42,
            'role': 'user',
            'content': 'The input row',
            'created_at': '2026-04-10 10:00:00',
        }
        fake_compaction_tc = {
            'id': 99,
            'tool_name': 'compaction',
            'params': '{"strategy": "bulk"}',
            'result': 'THIS_SHOULD_NOT_APPEAR',
            'invoked_by': 'system',
            'ephemeral': 0,
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.tool_call_service.ToolCallService') as mock_tcs_cls:
            mock_tcs_cls.return_value.get_by_transcript_ids.return_value = {
                42: [fake_compaction_tc]
            }
            p = FakeProcessor.make()
            result = p.getPreviousMessages()

        # The transcript row itself still renders
        assert 'The input row' in result
        # But the compaction pseudo-tool payload MUST NOT appear
        assert 'THIS_SHOULD_NOT_APPEAR' not in result
        assert '[compaction(' not in result
        assert '[compaction()' not in result

    def test_memory_durable_tool_call_still_renders(self):
        """Negative control: a `memory` durable tool_call under the same row
        DOES render. Proves the filter is scoped to `compaction` only."""
        from unittest.mock import patch

        fake_row = {
            'id': 42,
            'role': 'user',
            'content': 'Input row',
            'created_at': '2026-04-10 10:00:00',
        }
        fake_memory_tc = {
            'id': 88,
            'tool_name': 'memory',
            'params': '{"radius": 0.15}',
            'result': 'User visited Mykonos in 2024',
            'invoked_by': 'system',
            'ephemeral': 0,
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.tool_call_service.ToolCallService') as mock_tcs_cls:
            mock_tcs_cls.return_value.get_by_transcript_ids.return_value = {
                42: [fake_memory_tc]
            }
            p = FakeProcessor.make()
            result = p.getPreviousMessages()

        assert '[memory(radius=0.15)] User visited Mykonos in 2024' in result

    def test_mixed_compaction_and_memory_only_memory_surfaces(self):
        """Two durable tool_calls on the same row: one `compaction`, one
        `memory`. The `memory` row surfaces; the `compaction` row does not."""
        from unittest.mock import patch

        fake_row = {
            'id': 42,
            'role': 'user',
            'content': 'The input row',
            'created_at': '2026-04-10 10:00:00',
        }
        fake_memory_tc = {
            'id': 100,
            'tool_name': 'memory',
            'params': '{}',
            'result': 'SEED_SURVIVES',
            'invoked_by': 'system',
            'ephemeral': 0,
        }
        fake_compaction_tc = {
            'id': 101,
            'tool_name': 'compaction',
            'params': '{}',
            'result': 'COMPACTION_DROPPED',
            'invoked_by': 'system',
            'ephemeral': 0,
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.tool_call_service.ToolCallService') as mock_tcs_cls:
            mock_tcs_cls.return_value.get_by_transcript_ids.return_value = {
                42: [fake_memory_tc, fake_compaction_tc]
            }
            p = FakeProcessor.make()
            result = p.getPreviousMessages()

        assert 'SEED_SURVIVES' in result
        assert 'COMPACTION_DROPPED' not in result

    def test_never_render_frozenset_exported(self):
        """The filter set is a module-level frozenset so any future caller
        (tests, tooling) can consult it without copying the list."""
        from services.message_processor import _NEVER_RENDER_IN_PREVIOUS
        assert isinstance(_NEVER_RENDER_IN_PREVIOUS, frozenset)
        assert 'compaction' in _NEVER_RENDER_IN_PREVIOUS
        # Sanity: the set is frozen — not a vanilla mutable set.
        assert not hasattr(_NEVER_RENDER_IN_PREVIOUS, 'add')


# ─────────────────────────────────────────────────────────────────────────────
# 51. _render_params() — canonical source shared by live + historical rendering
#
# The canonical rendering helper used by both handleTool() (Commit 3 — live
# ACT trail append) and getPreviousMessages() (Commit 2a — historical durable
# tool_call replay from DB). Locked as a single source of truth so sibling
# processors never diverge.
# ─────────────────────────────────────────────────────────────────────────────


class TestRenderParamsCanonicalSource:
    def test_empty_dict_renders_empty_string(self):
        p = FakeProcessor.make()
        assert p._render_params({}) == ''

    def test_none_renders_empty_string(self):
        """Defensive path: falsy input → empty string, never crashes."""
        p = FakeProcessor.make()
        assert p._render_params(None) == ''

    def test_single_key_renders_key_equals_value(self):
        p = FakeProcessor.make()
        assert p._render_params({'query': 'hello'}) == 'query=hello'

    def test_multi_key_semicolon_joined(self):
        """Separator is `;`, matching the north star ACT loop trail format."""
        p = FakeProcessor.make()
        rendered = p._render_params({'query': 'hello', 'radius': 0.15})
        assert rendered == 'query=hello;radius=0.15'

    def test_insertion_order_preserved(self):
        """Py 3.7+ dicts preserve insertion order; _render_params must not sort."""
        p = FakeProcessor.make()
        rendered = p._render_params({'z_last': 1, 'a_first': 2})
        assert rendered == 'z_last=1;a_first=2'

    def test_numeric_values_stringified(self):
        p = FakeProcessor.make()
        assert p._render_params({'n': 42}) == 'n=42'
        assert p._render_params({'f': 3.14}) == 'f=3.14'

    def test_boolean_values_stringified(self):
        p = FakeProcessor.make()
        assert p._render_params({'flag': True}) == 'flag=True'


# ─────────────────────────────────────────────────────────────────────────────
# 52. _parse_tc_params() — DB string → dict normalisation
#
# tool_calls.params is stored as a JSON string. The renderer accepts both
# already-parsed dicts (tests) and raw DB strings (prod) and never crashes
# on malformed data.
# ─────────────────────────────────────────────────────────────────────────────


class TestParseTcParams:
    def test_none_returns_empty_dict(self):
        from services.message_processor import _parse_tc_params
        assert _parse_tc_params(None) == {}

    def test_empty_string_returns_empty_dict(self):
        from services.message_processor import _parse_tc_params
        assert _parse_tc_params('') == {}

    def test_pre_parsed_dict_passthrough(self):
        from services.message_processor import _parse_tc_params
        assert _parse_tc_params({'a': 1}) == {'a': 1}

    def test_valid_json_string_parsed(self):
        from services.message_processor import _parse_tc_params
        assert _parse_tc_params('{"query": "hello", "radius": 0.15}') == {
            'query': 'hello', 'radius': 0.15
        }

    def test_malformed_json_returns_empty_dict(self):
        """Bad JSON from the DB must never crash rendering — return empty
        and let the line render as `[tool_name()] result`."""
        from services.message_processor import _parse_tc_params
        assert _parse_tc_params('not-json') == {}
        assert _parse_tc_params('{broken') == {}

    def test_non_dict_json_returns_empty_dict(self):
        """JSON list/scalar must not become the params dict."""
        from services.message_processor import _parse_tc_params
        assert _parse_tc_params('[1, 2, 3]') == {}
        assert _parse_tc_params('"just a string"') == {}
        assert _parse_tc_params('42') == {}


# ─────────────────────────────────────────────────────────────────────────────
# 53. getPreviousMessages() — params rendering parity with live ACT trail
#
# The historical renderer must produce the same `[name(k=v;k=v)] result`
# format as the live ACT trail append in handleTool() (Commit 3). This is
# tested end-to-end by feeding a mocked durable tool_call through the
# rendering pipeline and asserting the final line matches _render_params'
# direct output.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPreviousMessagesParamsRenderingParity:
    def test_params_from_db_string_rendered_via_canonical_source(self):
        """Tool_call params loaded as a JSON string from the DB must render
        identically to the live ACT trail format."""
        from unittest.mock import patch

        fake_row = {
            'id': 1,
            'role': 'user',
            'content': 'Ask for memory',
            'created_at': '2026-04-10 10:00:00',
        }
        fake_tc = {
            'id': 2,
            'tool_name': 'memory',
            'params': '{"radius": 0.15}',
            'result': 'User visited Mykonos in 2024',
            'ephemeral': 0,
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.tool_call_service.ToolCallService') as mock_tcs_cls:
            mock_tcs_cls.return_value.get_by_transcript_ids.return_value = {1: [fake_tc]}
            p = FakeProcessor.make()
            result = p.getPreviousMessages()

        # Canonical source matches directly
        expected_params_str = p._render_params({'radius': 0.15})
        assert expected_params_str == 'radius=0.15'
        expected_line = f'[memory({expected_params_str})] User visited Mykonos in 2024'
        assert expected_line in result

    def test_empty_params_renders_empty_parens(self):
        """Empty dict params → `[tool_name()] result` (empty parens)."""
        from unittest.mock import patch

        fake_row = {
            'id': 1,
            'role': 'user',
            'content': 'row',
            'created_at': '2026-04-10 10:00:00',
        }
        fake_tc = {
            'id': 2,
            'tool_name': 'memory',
            'params': '{}',
            'result': 'empty params result',
            'ephemeral': 0,
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.tool_call_service.ToolCallService') as mock_tcs_cls:
            mock_tcs_cls.return_value.get_by_transcript_ids.return_value = {1: [fake_tc]}
            p = FakeProcessor.make()
            result = p.getPreviousMessages()

        assert '[memory()] empty params result' in result

    def test_malformed_params_string_falls_back_gracefully(self):
        """Malformed params JSON → `[tool_name()] result` — never crash."""
        from unittest.mock import patch

        fake_row = {
            'id': 1,
            'role': 'user',
            'content': 'row',
            'created_at': '2026-04-10 10:00:00',
        }
        fake_tc = {
            'id': 2,
            'tool_name': 'memory',
            'params': 'not-json-at-all',
            'result': 'survived bad params',
            'ephemeral': 0,
        }

        with patch('services.transcript_service.get_recent', return_value=[fake_row]), \
             patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.tool_call_service.ToolCallService') as mock_tcs_cls:
            mock_tcs_cls.return_value.get_by_transcript_ids.return_value = {1: [fake_tc]}
            p = FakeProcessor.make()
            # Must not raise
            result = p.getPreviousMessages()

        assert '[memory()] survived bad params' in result


# ─────────────────────────────────────────────────────────────────────────────
# 50. NATIVE_TOOLS type annotation — list[str] not bare list
# ─────────────────────────────────────────────────────────────────────────────


class TestNativeToolsTypeAnnotation:
    def test_native_tools_annotation_is_list_str(self):
        """Regression lock: NATIVE_TOOLS is annotated as list[str], not bare list."""
        from services.message_processor import MessageProcessor
        annotations = MessageProcessor.__annotations__
        assert 'NATIVE_TOOLS' in annotations
        annotation = annotations['NATIVE_TOOLS']
        # Python 3.9+ evaluates `list[str]` to a generic alias; its __args__ is (str,)
        args = getattr(annotation, '__args__', ())
        assert args == (str,), (
            f"Expected NATIVE_TOOLS: list[str], got {annotation!r}"
        )
