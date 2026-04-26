"""Contract tests for tool-event emission in MessageProcessor.handleTool().

These tests verify the event shape emitted via _emit_tool_event — they do
NOT test dispatch logic, LLM behaviour, or any other service. The fake
dispatcher is the minimum substitution needed because handleTool() calls
self._dispatcher.dispatch_action(); we cannot wire real innate skills in
a unit test scope.

Per feedback_test_philosophy.md: hot-path scope discipline, not mock theater.
"""

import pytest
from services.user_message_processor import UserMessageProcessor

pytestmark = pytest.mark.unit


class _FakeDispatcher:
    """Minimal dispatcher whose dispatch_action always succeeds."""

    def __init__(self, raise_exc=None):
        self._raise_exc = raise_exc
        self.handlers = {}

    def dispatch_action(self, channel, action):
        if self._raise_exc is not None:
            raise self._raise_exc
        return {'status': 'success', 'result': 'ok'}


class TestToolEventEmission:

    def _make_proc(self, events, raise_exc=None):
        """Construct a UMP with the event capture callback and a fake dispatcher."""
        proc = UserMessageProcessor(raw_input='test', on_tool_event=events.append)
        proc._dispatcher = _FakeDispatcher(raise_exc=raise_exc)
        proc._uid = 1
        proc._current_iteration = 2
        return proc

    def test_handle_tool_emits_start_then_end_with_stable_call_id_and_ok_true(self):
        events = []
        proc = self._make_proc(events)
        proc.handleTool({'id': 'toolu_abc', 'name': 'memory', 'input': {'q': 'x'}})

        assert len(events) == 2, f"Expected 2 events, got {len(events)}"

        start = events[0]
        assert start['type'] == 'act_tool_start'
        assert start['call_id'] == 'toolu_abc'
        assert start['name'] == 'memory'
        assert start['iter'] == 2

        end = events[1]
        assert end['type'] == 'act_tool_end'
        assert end['call_id'] == 'toolu_abc'
        assert isinstance(end['ms'], int) and end['ms'] >= 0
        assert end['ok'] is True

    def test_handle_tool_emits_ok_false_on_dispatch_exception_and_mints_call_id_when_missing(self):
        events = []
        proc = self._make_proc(events, raise_exc=RuntimeError('boom'))
        proc.handleTool({'name': 'broken_tool', 'input': {}})

        assert len(events) == 2, f"Expected 2 events, got {len(events)}"

        start = events[0]
        assert start['type'] == 'act_tool_start'
        assert start['name'] == 'broken_tool'
        minted_id = start['call_id']
        assert minted_id and len(minted_id) > 0

        end = events[1]
        assert end['call_id'] == minted_id, "call_id must be stable across start/end pair"
        assert end['ok'] is False
