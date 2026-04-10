# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for MessageProcessor.handleTool() — Commit 3.

Covers the full handleTool() specification from the north star:
  /Volumes/llm/chalie-plans/message-processing.md § "handleTool() semantics"

Test groups:
  A. Dispatch + DTO + trail lockstep (success path)
  B. Exception trapping — ERROR string, no re-raise, DTO + trail still appended
  C. find_tools side effect — schema discovery + deduplication
  D. Non-find_tools tool — no side effect on _discovered_tools
  E. DTO field contract — name, params, result, ephemeral, invoked_by, timestamp
  F. ACT trail format — canonical `;` separator via _render_params
  G. Multiple sequential calls — accumulation + timestamp monotonicity
  H. Exception path does NOT trigger find_tools side effect
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────

_USER_DEF = "The user is a human named Alice interacting via the chat interface."
_USER_PROMPT = "What time is it?"
_CHANNEL = 'test_channel'


def _make_processor(**kwargs):
    """Build a concrete FakeProcessor instance for handleTool tests."""
    from services.message_processor_v2 import MessageProcessor

    class _Fake(MessageProcessor):
        CHANNEL = _CHANNEL
        ROLE = 'test_role'

        def getUserDefinition(self) -> str:
            return _USER_DEF

        def getUserPrompt(self) -> str:
            return _USER_PROMPT

    for k, v in kwargs.items():
        setattr(_Fake, k, v)

    return _Fake('test raw input')


def _dispatch_ok(result_text: str, extra: dict | None = None) -> dict:
    """Build a fake successful dispatch_action return dict."""
    d = {'result': result_text, 'status': 'success'}
    if extra:
        d.update(extra)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# A. Dispatch + DTO + trail lockstep (success path)
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolSuccessPath:
    """Happy-path: dispatch succeeds, DTO + trail appended in lockstep."""

    def test_calls_dispatch_action_once(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('ok')
            p.handleTool({'name': 'memory', 'input': {'query': 'test'}})

        MockCls.return_value.dispatch_action.assert_called_once_with(
            _CHANNEL, {'type': 'memory', 'query': 'test'}
        )

    def test_action_dict_construction_merges_input(self):
        """action = {'type': tool_name, **tc.get('input', {})}"""
        p = _make_processor()
        captured = {}
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            def spy(channel, action):
                captured.update(action)
                return _dispatch_ok('result')
            MockCls.return_value.dispatch_action.side_effect = spy
            p.handleTool({'name': 'schedule', 'input': {'date': '2026-05-01', 'note': 'dentist'}})

        assert captured == {'type': 'schedule', 'date': '2026-05-01', 'note': 'dentist'}

    def test_dto_appended_to_pending_tool_calls(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('Memory result')
            p.handleTool({'name': 'memory', 'input': {'query': 'hello'}})

        assert len(p._pending_tool_calls) == 1

    def test_trail_appended_to_act_trail(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('Memory result')
            p.handleTool({'name': 'memory', 'input': {'query': 'hello'}})

        assert len(p._act_trail) == 1

    def test_dto_and_trail_lengths_match_after_call(self):
        """Lockstep: both lists always have equal length."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('r')
            p.handleTool({'name': 'memory', 'input': {}})

        assert len(p._pending_tool_calls) == len(p._act_trail)

    def test_returns_result_string(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('Expected result')
            result = p.handleTool({'name': 'memory', 'input': {}})

        assert result == 'Expected result'

    def test_return_type_is_always_str(self):
        """dispatch.get('result', '') is passed through str()."""
        for raw in [None, 42, 3.14, {'nested': 'dict'}, ['list'], True]:
            p = _make_processor()
            with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
                MockCls.return_value.dispatch_action.return_value = {'result': raw, 'status': 'success'}
                result = p.handleTool({'name': 'memory', 'input': {}})
            assert isinstance(result, str), (
                f"Expected str, got {type(result).__name__} for raw={raw!r}"
            )

    def test_execution_gate_false(self):
        """ActDispatcherService must be constructed with execution_gate=False."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('r')
            p.handleTool({'name': 'memory', 'input': {}})

        MockCls.assert_called_once_with(execution_gate=False)

    def test_no_input_key_falls_back_to_empty_dict(self):
        """tc without 'input' key must not raise — falls back to {}."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('r')
            result = p.handleTool({'name': 'memory'})  # no 'input' key

        assert isinstance(result, str)
        assert len(p._pending_tool_calls) == 1

    def test_empty_input_dict_produces_empty_action_params(self):
        """With input={}, action is just {'type': tool_name}."""
        p = _make_processor()
        captured = {}
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            def spy(channel, action):
                captured.update(action)
                return _dispatch_ok('r')
            MockCls.return_value.dispatch_action.side_effect = spy
            p.handleTool({'name': 'memory', 'input': {}})

        assert captured == {'type': 'memory'}


# ─────────────────────────────────────────────────────────────────────────────
# B. Exception trapping
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolExceptionTrapping:
    """dispatch_action raises → ERROR string, no re-raise, DTO + trail appended."""

    def test_does_not_reraise_on_exception(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = RuntimeError('network down')
            # Must not raise — exception is swallowed
            result = p.handleTool({'name': 'memory', 'input': {}})

        assert isinstance(result, str)

    def test_error_string_format(self):
        """ERROR: {tool_name} failed: {exc}"""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = ValueError('bad param')
            result = p.handleTool({'name': 'schedule', 'input': {}})

        assert result == 'ERROR: schedule failed: bad param'

    def test_error_string_uses_tool_name_from_tc(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = Exception('boom')
            result = p.handleTool({'name': 'document', 'input': {}})

        assert result.startswith('ERROR: document failed:')

    def test_dto_still_appended_on_exception(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = RuntimeError('crash')
            p.handleTool({'name': 'memory', 'input': {}})

        assert len(p._pending_tool_calls) == 1

    def test_trail_still_appended_on_exception(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = RuntimeError('crash')
            p.handleTool({'name': 'memory', 'input': {}})

        assert len(p._act_trail) == 1

    def test_dto_and_trail_lengths_equal_on_exception(self):
        """Lockstep holds even on failure."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = RuntimeError('crash')
            p.handleTool({'name': 'memory', 'input': {}})

        assert len(p._pending_tool_calls) == len(p._act_trail)

    def test_dto_result_matches_returned_error_string(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = KeyError('missing_key')
            result = p.handleTool({'name': 'memory', 'input': {}})

        assert p._pending_tool_calls[0]['result'] == result

    def test_trail_contains_error_string(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = RuntimeError('timeout')
            result = p.handleTool({'name': 'memory', 'input': {}})

        assert result in p._act_trail[0]

    def test_various_exception_types_all_trapped(self):
        for exc_cls, msg in [
            (RuntimeError, 'runtime'),
            (ValueError, 'value'),
            (TypeError, 'type error'),
            (ConnectionError, 'conn'),
            (Exception, 'generic'),
        ]:
            p = _make_processor()
            with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
                MockCls.return_value.dispatch_action.side_effect = exc_cls(msg)
                result = p.handleTool({'name': 'memory', 'input': {}})
            assert 'ERROR' in result
            assert len(p._pending_tool_calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# C. find_tools side effect
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolFindToolsSideEffect:
    """find_tools dispatch → _discovered_tools extended, deduped by name."""

    def test_find_tools_extends_discovered_tools(self):
        p = _make_processor()
        fake_schemas = [
            {'name': 'weather_tool', 'description': 'Get weather', 'input_schema': {}},
        ]
        fake_dispatch = _dispatch_ok(
            'Found 1 tool: weather_tool',
            extra={'_discovered_tools': ['weather_tool']},
        )
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas', return_value=fake_schemas):
            MockCls.return_value.dispatch_action.return_value = fake_dispatch
            p.handleTool({'name': 'find_tools', 'input': {'query': 'weather'}})

        assert len(p._discovered_tools) == 1
        assert p._discovered_tools[0]['name'] == 'weather_tool'

    def test_find_tools_dedupes_by_name(self):
        """A second find_tools call returning the same tool name is a no-op."""
        p = _make_processor()
        fake_schemas = [
            {'name': 'weather_tool', 'description': 'Get weather', 'input_schema': {}},
        ]
        fake_dispatch = _dispatch_ok(
            'Found 1 tool: weather_tool',
            extra={'_discovered_tools': ['weather_tool']},
        )
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas', return_value=fake_schemas):
            MockCls.return_value.dispatch_action.return_value = fake_dispatch
            p.handleTool({'name': 'find_tools', 'input': {'query': 'weather'}})
            # Second call with same tool name — must not duplicate
            p.handleTool({'name': 'find_tools', 'input': {'query': 'weather forecast'}})

        assert len(p._discovered_tools) == 1

    def test_find_tools_adds_new_tools_but_not_duplicates(self):
        """Second call brings a NEW tool — only the new one is added."""
        p = _make_processor()
        weather_schema = {'name': 'weather_tool', 'description': 'Weather', 'input_schema': {}}
        news_schema = {'name': 'news_tool', 'description': 'News', 'input_schema': {}}

        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas') as mock_schemas:
            # First call
            mock_schemas.return_value = [weather_schema]
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok(
                'Found 1 tool: weather_tool',
                extra={'_discovered_tools': ['weather_tool']},
            )
            p.handleTool({'name': 'find_tools', 'input': {'query': 'weather'}})

            # Second call: same weather + new news
            mock_schemas.return_value = [weather_schema, news_schema]
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok(
                'Found 2 tools: weather_tool, news_tool',
                extra={'_discovered_tools': ['weather_tool', 'news_tool']},
            )
            p.handleTool({'name': 'find_tools', 'input': {'query': 'news'}})

        names = [t['name'] for t in p._discovered_tools]
        assert sorted(names) == ['news_tool', 'weather_tool']

    def test_find_tools_empty_discovered_list_is_noop(self):
        """dispatch returns _discovered_tools=[] → get_external_tool_schemas not called."""
        p = _make_processor()
        fake_dispatch = _dispatch_ok(
            'No tools found',
            extra={'_discovered_tools': []},
        )
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas') as mock_schemas:
            MockCls.return_value.dispatch_action.return_value = fake_dispatch
            p.handleTool({'name': 'find_tools', 'input': {'query': 'xyz'}})

        mock_schemas.assert_not_called()
        assert p._discovered_tools == []

    def test_find_tools_missing_key_is_noop(self):
        """dispatch has no '_discovered_tools' key → no side effect."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas') as mock_schemas:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('Found nothing')
            p.handleTool({'name': 'find_tools', 'input': {'query': 'xyz'}})

        mock_schemas.assert_not_called()
        assert p._discovered_tools == []

    def test_find_tools_result_is_confirmation_string_not_schemas(self):
        """LLM sees the confirmation string, not the raw schema list."""
        p = _make_processor()
        confirmation = 'Found 1 tool: weather_tool'
        fake_schemas = [{'name': 'weather_tool', 'description': 'Weather', 'input_schema': {}}]
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas', return_value=fake_schemas):
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok(
                confirmation,
                extra={'_discovered_tools': ['weather_tool']},
            )
            result = p.handleTool({'name': 'find_tools', 'input': {'query': 'weather'}})

        assert result == confirmation
        # DTO result is also the confirmation string
        assert p._pending_tool_calls[0]['result'] == confirmation


# ─────────────────────────────────────────────────────────────────────────────
# D. Non-find_tools tool — no side effect on _discovered_tools
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolNonFindToolsNoSideEffect:
    """Any tool other than find_tools must NOT touch _discovered_tools."""

    def test_memory_tool_does_not_call_get_external_tool_schemas(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas') as mock_schemas:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('Memory result')
            p.handleTool({'name': 'memory', 'input': {'query': 'something'}})

        mock_schemas.assert_not_called()

    def test_schedule_tool_does_not_mutate_discovered_tools(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('Scheduled!')
            p.handleTool({'name': 'schedule', 'input': {'date': '2026-05-01'}})

        assert p._discovered_tools == []

    def test_arbitrary_tool_leaves_discovered_tools_empty(self):
        for tool_name in ['document', 'introspect', 'associate', 'notes', 'goals']:
            p = _make_processor()
            with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
                 patch('services.tool_schema_service.get_external_tool_schemas') as mock_schemas:
                MockCls.return_value.dispatch_action.return_value = _dispatch_ok('result')
                p.handleTool({'name': tool_name, 'input': {}})

            assert p._discovered_tools == [], f"{tool_name} must not affect _discovered_tools"
            mock_schemas.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# E. DTO field contract
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolDTOFields:
    """Every DTO in _pending_tool_calls must have exact field shape."""

    def _dto(self, tc: dict, result_text: str = 'result') -> dict:
        """Run handleTool and return the appended DTO."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok(result_text)
            p.handleTool(tc)
        return p._pending_tool_calls[0]

    def test_dto_name_field(self):
        dto = self._dto({'name': 'memory', 'input': {'query': 'q'}})
        assert dto['name'] == 'memory'

    def test_dto_params_field_is_input_dict(self):
        dto = self._dto({'name': 'memory', 'input': {'query': 'hello', 'radius': 0.5}})
        assert dto['params'] == {'query': 'hello', 'radius': 0.5}

    def test_dto_params_empty_when_no_input(self):
        dto = self._dto({'name': 'memory'})
        assert dto['params'] == {}

    def test_dto_result_field(self):
        dto = self._dto({'name': 'memory', 'input': {}}, result_text='Memory content here')
        assert dto['result'] == 'Memory content here'

    def test_dto_ephemeral_is_1(self):
        dto = self._dto({'name': 'memory', 'input': {}})
        assert dto['ephemeral'] == 1

    def test_dto_invoked_by_is_llm(self):
        dto = self._dto({'name': 'memory', 'input': {}})
        assert dto['invoked_by'] == 'llm'

    def test_dto_timestamp_is_utc_iso_string(self):
        """Timestamp must be a non-empty ISO-8601 string from utc_now()."""
        dto = self._dto({'name': 'memory', 'input': {}})
        ts = dto['timestamp']
        assert isinstance(ts, str)
        assert len(ts) > 0
        # Must parse as ISO 8601 — datetime.fromisoformat() accepts the +00:00 format
        from datetime import datetime
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None, "Timestamp must be timezone-aware (UTC)"

    def test_dto_timestamp_uses_utc_now_not_datetime_now(self):
        """Verify the timestamp comes from utc_now(), not datetime.now()."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('r')
            p.handleTool({'name': 'memory', 'input': {}})

        ts = p._pending_tool_calls[0]['timestamp']
        from datetime import datetime
        parsed = datetime.fromisoformat(ts)
        # UTC offset must be zero
        assert parsed.utcoffset().total_seconds() == 0

    def test_dto_all_required_keys_present(self):
        dto = self._dto({'name': 'memory', 'input': {}})
        for key in ('name', 'params', 'result', 'ephemeral', 'invoked_by', 'timestamp'):
            assert key in dto, f"DTO missing required key: {key!r}"

    def test_dto_on_exception_path_has_all_fields(self):
        """Exception path also produces a fully-populated DTO."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = RuntimeError('crash')
            p.handleTool({'name': 'memory', 'input': {'q': 'x'}})

        dto = p._pending_tool_calls[0]
        for key in ('name', 'params', 'result', 'ephemeral', 'invoked_by', 'timestamp'):
            assert key in dto, f"Exception-path DTO missing key: {key!r}"
        assert dto['ephemeral'] == 1
        assert dto['invoked_by'] == 'llm'


# ─────────────────────────────────────────────────────────────────────────────
# F. ACT trail format — canonical `;` separator
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolActTrailFormat:
    """Trail line must be `[tool_name(k1=v1;k2=v2)] result` exactly."""

    def _trail(self, tc: dict, result_text: str = 'trail result') -> str:
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok(result_text)
            p.handleTool(tc)
        return p._act_trail[0]

    def test_no_params_format(self):
        line = self._trail({'name': 'memory', 'input': {}})
        assert line == '[memory()] trail result'

    def test_single_param_format(self):
        line = self._trail({'name': 'memory', 'input': {'query': 'hello'}})
        assert line == '[memory(query=hello)] trail result'

    def test_multi_param_semicolon_separator(self):
        """Multiple params joined by `;` not `,` or ` `."""
        line = self._trail({'name': 'schedule', 'input': {'date': '2026-05-01', 'note': 'dentist'}})
        assert line == '[schedule(date=2026-05-01;note=dentist)] trail result'

    def test_no_input_key_same_as_empty_input(self):
        """tc without 'input' key → params rendered as empty → `[tool_name()]`."""
        line = self._trail({'name': 'memory'})
        assert line.startswith('[memory()] ')

    def test_trail_result_matches_return_value(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('Exact result text')
            result = p.handleTool({'name': 'memory', 'input': {}})

        assert result in p._act_trail[0]
        assert result == 'Exact result text'

    def test_trail_on_exception_uses_error_string(self):
        """Trail line on exception includes the ERROR: … string."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = RuntimeError('fail')
            result = p.handleTool({'name': 'memory', 'input': {}})

        assert result in p._act_trail[0]
        assert 'ERROR' in p._act_trail[0]

    def test_uses_canonical_render_params_not_reimplementation(self):
        """Trail rendering uses self._render_params() — verify via multi-param output."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('r')
            p.handleTool({'name': 'tool', 'input': {'a': '1', 'b': '2', 'c': '3'}})

        line = p._act_trail[0]
        # Canonical format: a=1;b=2;c=3 (semicolons, no spaces, insertion order)
        assert 'a=1;b=2;c=3' in line


# ─────────────────────────────────────────────────────────────────────────────
# G. Multiple sequential calls — accumulation + timestamp monotonicity
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolMultipleCalls:
    """Sequential handleTool() calls accumulate in order."""

    def test_dtos_accumulate_in_call_order(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = [
                _dispatch_ok('result_a'),
                _dispatch_ok('result_b'),
                _dispatch_ok('result_c'),
            ]
            p.handleTool({'name': 'memory', 'input': {'q': 'a'}})
            p.handleTool({'name': 'schedule', 'input': {'date': 'b'}})
            p.handleTool({'name': 'document', 'input': {'action': 'c'}})

        assert len(p._pending_tool_calls) == 3
        assert p._pending_tool_calls[0]['name'] == 'memory'
        assert p._pending_tool_calls[1]['name'] == 'schedule'
        assert p._pending_tool_calls[2]['name'] == 'document'

    def test_trail_accumulates_in_call_order(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = [
                _dispatch_ok('r1'),
                _dispatch_ok('r2'),
            ]
            p.handleTool({'name': 'memory', 'input': {}})
            p.handleTool({'name': 'schedule', 'input': {}})

        assert len(p._act_trail) == 2
        assert 'memory' in p._act_trail[0]
        assert 'schedule' in p._act_trail[1]

    def test_dto_and_trail_lengths_always_equal(self):
        """Lockstep holds across N calls including mixed success+exception."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = [
                _dispatch_ok('ok1'),
                RuntimeError('boom'),
                _dispatch_ok('ok3'),
            ]
            p.handleTool({'name': 'memory', 'input': {}})
            p.handleTool({'name': 'schedule', 'input': {}})
            p.handleTool({'name': 'document', 'input': {}})

        assert len(p._pending_tool_calls) == len(p._act_trail) == 3

    def test_timestamps_are_monotonically_non_decreasing(self):
        """Every DTO timestamp must be >= the previous one."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('r')
            for i in range(5):
                p.handleTool({'name': f'tool_{i}', 'input': {}})

        from datetime import datetime
        timestamps = [
            datetime.fromisoformat(dto['timestamp'])
            for dto in p._pending_tool_calls
        ]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamp at index {i} is before index {i-1}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# H. Exception path does NOT trigger find_tools side effect
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolExceptionNoFindToolsSideEffect:
    """When dispatch raises for a find_tools call, dispatch={} so no side effect fires."""

    def test_exception_on_find_tools_does_not_call_get_external_tool_schemas(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas') as mock_schemas:
            MockCls.return_value.dispatch_action.side_effect = RuntimeError('find_tools crashed')
            p.handleTool({'name': 'find_tools', 'input': {'query': 'weather'}})

        mock_schemas.assert_not_called()

    def test_exception_on_find_tools_leaves_discovered_tools_empty(self):
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.side_effect = RuntimeError('crash')
            p.handleTool({'name': 'find_tools', 'input': {'query': 'weather'}})

        assert p._discovered_tools == []

    def test_exception_path_dispatch_is_empty_dict(self):
        """On exception, dispatch={} — no '_discovered_tools' key → guard activates."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas') as mock_schemas:
            MockCls.return_value.dispatch_action.side_effect = Exception('crash')
            result = p.handleTool({'name': 'find_tools', 'input': {}})

        # ERROR string returned, no schema calls
        assert 'ERROR' in result
        mock_schemas.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# I. Malformed tc — DTO + trail still land
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolMalformedTc:
    """Malformed tc dicts must never raise; DTO + trail land in lockstep with name='unknown'."""

    def test_malformed_tc_missing_name_still_creates_dto_and_trail(self):
        """tc without a 'name' key must not raise."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('r')
            # Must not raise
            result = p.handleTool({'input': {'q': 'x'}})

        assert isinstance(result, str)
        assert len(p._pending_tool_calls) == 1
        assert len(p._act_trail) == 1
        assert p._pending_tool_calls[0]['name'] == 'unknown'
        assert p._act_trail[0].startswith('[unknown(')

    def test_malformed_tc_name_is_none_still_creates_dto_and_trail(self):
        """tc['name'] = None → fallback to 'unknown', no crash."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('r')
            result = p.handleTool({'name': None, 'input': {}})

        assert isinstance(result, str)
        assert len(p._pending_tool_calls) == 1
        assert len(p._act_trail) == 1
        assert p._pending_tool_calls[0]['name'] == 'unknown'
        assert p._act_trail[0].startswith('[unknown(')

    def test_empty_tc_dict_still_creates_dto_and_trail(self):
        """tc = {} → name='unknown', params={}, no crash."""
        p = _make_processor()
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('r')
            result = p.handleTool({})

        assert isinstance(result, str)
        assert len(p._pending_tool_calls) == 1
        assert len(p._act_trail) == 1
        dto = p._pending_tool_calls[0]
        assert dto['name'] == 'unknown'
        assert dto['params'] == {}
        assert p._act_trail[0] == '[unknown()] r'

    def test_render_params_failure_still_appends_dto_and_trail_in_lockstep(self):
        """If _render_params blows up, DTO + trail still land with lengths equal."""
        p = _make_processor()

        def bomb(_):
            raise RuntimeError('render_params exploded')

        # Monkeypatch the bound method on the instance
        p._render_params = bomb  # type: ignore[method-assign]

        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls:
            MockCls.return_value.dispatch_action.return_value = _dispatch_ok('ok')
            result = p.handleTool({'name': 'memory', 'input': {'q': 'x'}})

        assert isinstance(result, str)
        assert len(p._pending_tool_calls) == 1
        assert len(p._act_trail) == 1
        # Trail still lands with an empty params rendering, not a crash
        assert p._act_trail[0].startswith('[memory(')


# ─────────────────────────────────────────────────────────────────────────────
# J. find_tools status check + defensive dedupe
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleToolFindToolsDefensive:
    """Side effect only fires on status=='success'; dedupe survives schemas lacking a name key."""

    def test_find_tools_error_status_skips_side_effect(self):
        """Dispatch returns status='error' with a _discovered_tools key — side effect must NOT fire."""
        p = _make_processor()
        fake_dispatch = {
            'result': 'Some partial response',
            'status': 'error',
            '_discovered_tools': ['weather_tool'],
        }
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas') as mock_schemas:
            MockCls.return_value.dispatch_action.return_value = fake_dispatch
            p.handleTool({'name': 'find_tools', 'input': {'query': 'weather'}})

        mock_schemas.assert_not_called()
        assert p._discovered_tools == []

    def test_find_tools_missing_status_skips_side_effect(self):
        """Dispatch dict missing 'status' key — defensive check rejects it."""
        p = _make_processor()
        fake_dispatch = {
            'result': 'Some response',
            '_discovered_tools': ['weather_tool'],
            # no 'status' key
        }
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas') as mock_schemas:
            MockCls.return_value.dispatch_action.return_value = fake_dispatch
            p.handleTool({'name': 'find_tools', 'input': {'query': 'weather'}})

        mock_schemas.assert_not_called()
        assert p._discovered_tools == []

    def test_discovered_tools_dedupe_survives_schema_without_name_key(self):
        """Pre-populate _discovered_tools with a dict lacking 'name' — must not KeyError."""
        p = _make_processor()
        # Pre-populate with a degenerate schema that lacks 'name'
        p._discovered_tools.append({'description': 'no name here', 'input_schema': {}})

        new_schemas = [
            {'name': 'weather_tool', 'description': 'Weather', 'input_schema': {}},
        ]
        fake_dispatch = _dispatch_ok(
            'Found 1 tool: weather_tool',
            extra={'_discovered_tools': ['weather_tool']},
        )
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas', return_value=new_schemas):
            MockCls.return_value.dispatch_action.return_value = fake_dispatch
            # Must not raise
            p.handleTool({'name': 'find_tools', 'input': {'query': 'weather'}})

        # Degenerate entry preserved, new weather_tool added (dedupe didn't reject it)
        assert len(p._discovered_tools) == 2
        names = [t.get('name') for t in p._discovered_tools]
        assert 'weather_tool' in names

    def test_discovered_tools_dedupe_survives_new_schema_without_name_key(self):
        """get_external_tool_schemas returns an entry without a 'name' key — must not crash or add it."""
        p = _make_processor()
        bad_schemas = [
            {'description': 'nameless schema', 'input_schema': {}},  # no 'name'
            {'name': 'weather_tool', 'description': 'Weather', 'input_schema': {}},
        ]
        fake_dispatch = _dispatch_ok(
            'Found 2 tools',
            extra={'_discovered_tools': ['weather_tool', 'nameless']},
        )
        with patch('services.act_dispatcher_service.ActDispatcherService') as MockCls, \
             patch('services.tool_schema_service.get_external_tool_schemas', return_value=bad_schemas):
            MockCls.return_value.dispatch_action.return_value = fake_dispatch
            p.handleTool({'name': 'find_tools', 'input': {'query': 'xyz'}})

        # Only the valid (named) schema lands
        assert len(p._discovered_tools) == 1
        assert p._discovered_tools[0]['name'] == 'weather_tool'
