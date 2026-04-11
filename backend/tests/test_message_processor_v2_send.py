# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for MessageProcessor.send() — Commit 6.

Covers the full send() specification from the north star:
  /Volumes/llm/chalie-plans/message-processing.md § "Flow"
  /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md § Commit 6

Test groups:
  A. One-shot — no tool calls → immediate break, single store(), single postTurn()
  B. Two-iteration — tool call + synthesis → DTO accumulation, store() at end
  C. MAX_ITERATIONS cap — safety cap respected
  D. MAX_TIMEOUT cap — wall-clock safety cap respected
  E. Provider exception semantics — store() and postTurn() NOT called; propagates
  F. postTurn() exception after store() — persisted rows survive, exception swallowed
  G. tool_synthesis DTO shape — ephemeral=1, invoked_by='system', params={}, ISO ts
  H. Empty narration path — llm_response.text is '' → no tool_synthesis DTO
  I. _emit_narration called on base (no-op) and on override subclass
  J. _drain_steering returns [] on base → no user_steer DTOs
  K. Steering drained mid-loop → user_steer DTO appears with correct shape
  L. find_tools discovery propagates to getTools() on next iteration
  M. Narration subclass override fires on_narration callback per iteration
  N. bind_current_processor context set during send(), cleared on exit
"""

import time
from unittest.mock import MagicMock, call, patch

import pytest

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

_USER_DEF = "The user is 'test' — a fake processor used in unit tests."
_CHANNEL = 'test_send_channel'
_ROLE = 'test_role'


def _make_llm_response(text='', tool_calls=None):
    """Build a minimal LLMResponse-like object with .text and .tool_calls."""
    from services.llm_service import LLMResponse
    return LLMResponse(
        text=text,
        model='test-model',
        provider='mock',
        tool_calls=tool_calls,
    )


def _make_processor(
    user_prompt='hello',
    raw_input='hello',
    max_iterations=30,
    max_timeout=900,
    **kwargs,
):
    """Build a concrete FakeProcessor instance for send() tests."""
    from services.message_processor import MessageProcessor

    _prompt = user_prompt

    class _Fake(MessageProcessor):
        CHANNEL = _CHANNEL
        ROLE = _ROLE
        MAX_ITERATIONS = max_iterations
        MAX_TIMEOUT = max_timeout

        def getUserDefinition(self) -> str:
            return _USER_DEF

        def getUserPrompt(self) -> str:
            return _prompt

    for k, v in kwargs.items():
        setattr(_Fake, k, v)

    return _Fake(raw_input)


def _mock_store(processor):
    """Patch store() on a processor instance to capture calls, set _uid."""
    original_store = processor.store

    calls_made = []

    def _fake_store(llm_response):
        calls_made.append(llm_response)
        processor._uid = 99  # simulate successful write

    processor.store = _fake_store
    return calls_made


# ─────────────────────────────────────────────────────────────────────────────
# A. One-shot — no tool calls
# ─────────────────────────────────────────────────────────────────────────────


class TestOneShotNoToolCalls:
    """Provider returns text with no tool_calls → break immediately, store once."""

    def _run(self, response_text='Hello from LLM'):
        p = _make_processor()
        store_calls = _mock_store(p)
        post_calls = []
        p.postTurn = lambda: post_calls.append(True)

        llm_resp = _make_llm_response(text=response_text, tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.return_value = llm_resp
            result = p.send()

        return result, store_calls, post_calls, p

    def test_returns_final_text(self):
        result, _, _, _ = self._run('Final answer.')
        assert result == 'Final answer.'

    def test_store_called_exactly_once(self):
        _, store_calls, _, _ = self._run()
        assert len(store_calls) == 1

    def test_store_receives_final_text(self):
        result, store_calls, _, _ = self._run('my response')
        assert store_calls[0] == 'my response'

    def test_post_turn_called_once(self):
        _, _, post_calls, _ = self._run()
        assert len(post_calls) == 1

    def test_no_tool_synthesis_dto(self):
        """No tool calls means no tool_synthesis DTO appended."""
        _, _, _, p = self._run()
        names = [d['name'] for d in p._pending_tool_calls]
        assert 'tool_synthesis' not in names

    def test_provider_called_exactly_once(self):
        p = _make_processor()
        _mock_store(p)
        llm_resp = _make_llm_response(text='x', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.return_value = llm_resp
            p.send()

        assert mock_inst.return_value.send_messages.call_count == 1

    def test_empty_text_returns_empty_string(self):
        """Provider returns empty text → send() returns ''."""
        result, _, _, _ = self._run('')
        assert result == ''

    def test_messages_is_single_element_list(self):
        """The user body is wrapped as a single-element messages[] for transport."""
        p = _make_processor(user_prompt='user body text')
        _mock_store(p)
        llm_resp = _make_llm_response(text='x', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.return_value = llm_resp
            p.send()

        _, messages_arg, *_ = mock_inst.return_value.send_messages.call_args[0]
        assert messages_arg == [{'role': 'user', 'content': 'user body text'}]


# ─────────────────────────────────────────────────────────────────────────────
# B. Two-iteration — tool call then clean response
# ─────────────────────────────────────────────────────────────────────────────


class TestTwoIteration:
    """First provider call returns one tool call; second returns text only."""

    def _build_tc(self, name='search', input_=None):
        return {'name': name, 'input': input_ or {'q': 'test'}}

    def test_handle_tool_called_once(self):
        p = _make_processor()
        _mock_store(p)

        tc = self._build_tc()
        iter1 = _make_llm_response(text='Searching…', tool_calls=[tc])
        iter2 = _make_llm_response(text='Done.', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'result text'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        assert mock_ht.call_count == 1

    def test_pending_tool_calls_has_tool_synthesis_dto(self):
        """After iteration 1 with narration text, a tool_synthesis DTO is appended."""
        p = _make_processor()
        _mock_store(p)

        tc = self._build_tc()
        iter1 = _make_llm_response(text='Looking into this…', tool_calls=[tc])
        iter2 = _make_llm_response(text='Done.', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'result'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        names = [d['name'] for d in p._pending_tool_calls]
        assert 'tool_synthesis' in names

    def test_provider_called_twice(self):
        p = _make_processor()
        _mock_store(p)

        tc = self._build_tc()
        iter1 = _make_llm_response(text='', tool_calls=[tc])
        iter2 = _make_llm_response(text='result', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        assert mock_inst.return_value.send_messages.call_count == 2

    def test_store_called_once_at_end(self):
        """store() must be called exactly once, after the ACT loop terminates."""
        p = _make_processor()
        store_calls = _mock_store(p)

        tc = self._build_tc()
        iter1 = _make_llm_response(text='step', tool_calls=[tc])
        iter2 = _make_llm_response(text='final', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        assert len(store_calls) == 1
        assert store_calls[0] == 'final'

    def test_act_trail_has_tool_synthesis_entry(self):
        """ACT trail must include a [tool_synthesis] line for the narration."""
        p = _make_processor()
        _mock_store(p)

        tc = self._build_tc()
        iter1 = _make_llm_response(text='narration here', tool_calls=[tc])
        iter2 = _make_llm_response(text='final', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        assert any('[tool_synthesis]' in line for line in p._act_trail)

    def test_final_return_is_last_llm_text(self):
        p = _make_processor()
        _mock_store(p)

        tc = self._build_tc()
        iter1 = _make_llm_response(text='thinking', tool_calls=[tc])
        iter2 = _make_llm_response(text='the answer', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            result = p.send()

        assert result == 'the answer'

    def test_tool_synthesis_recorded_before_handle_tool(self):
        """tool_synthesis DTO must be appended BEFORE handleTool() dispatches
        the tool calls for the same iteration.

        Locks north star § Storage Model transcript-timeline example:
        tool_synthesis precedes the tool_call DTOs it introduces. The
        narration was emitted by the LLM ahead of the tool_use block, so
        the stored timeline must reflect that semantic order.
        """
        p = _make_processor()
        _mock_store(p)

        tc = self._build_tc()
        iter1 = _make_llm_response(
            text='Let me check movies first', tool_calls=[tc]
        )
        iter2 = _make_llm_response(text='done', tool_calls=None)

        # Record the order of append operations: 'synth' vs 'ht'.
        events: list[str] = []

        class _TrackingList(list):
            def append(self, dto):
                if isinstance(dto, dict) and dto.get('name') == 'tool_synthesis':
                    events.append('synth')
                super().append(dto)

        # Swap in a tracking list for _pending_tool_calls.
        p._pending_tool_calls = _TrackingList(p._pending_tool_calls)

        def _fake_handle_tool(tc_arg):
            events.append('ht')
            return 'r'

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool', side_effect=_fake_handle_tool):
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        # The synthesis append for iteration 1 MUST come before the handleTool
        # dispatch for iteration 1's tool call.
        assert 'synth' in events and 'ht' in events
        assert events.index('synth') < events.index('ht')


# ─────────────────────────────────────────────────────────────────────────────
# C. MAX_ITERATIONS cap
# ─────────────────────────────────────────────────────────────────────────────


class TestMaxIterationsCap:
    """Loop exits after MAX_ITERATIONS even if LLM keeps returning tool calls."""

    def test_loop_exits_at_cap(self):
        cap = 5
        p = _make_processor(max_iterations=cap)
        _mock_store(p)

        # Every provider call returns another tool call — loop should exit at cap
        tc = {'name': 'loop_tool', 'input': {}}
        response_with_tools = _make_llm_response(text='', tool_calls=[tc])

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.return_value = response_with_tools
            p.send()

        # Provider is called at most cap times (while loop checks BEFORE calling)
        assert mock_inst.return_value.send_messages.call_count <= cap

    def test_store_still_called_after_cap(self):
        """Even when capped, store() is called once."""
        cap = 3
        p = _make_processor(max_iterations=cap)
        store_calls = _mock_store(p)

        tc = {'name': 'loop_tool', 'input': {}}
        response_with_tools = _make_llm_response(text='step', tool_calls=[tc])

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.return_value = response_with_tools
            p.send()

        assert len(store_calls) == 1

    def test_final_text_empty_on_iteration_cap(self):
        """When the loop exits via MAX_ITERATIONS, final_text is '' — never
        the last iteration's mid-loop narration.

        Locks north star § Storage Model: "Write final assistant row …
        content = llm_response" where llm_response is the *terminating*
        text. Cap-exit has no terminating text, so the assistant row is
        empty rather than a misleading snippet of partial thinking.
        """
        cap = 3
        p = _make_processor(max_iterations=cap)
        store_calls = _mock_store(p)

        tc = {'name': 'loop_tool', 'input': {}}
        response_with_tools = _make_llm_response(
            text='partial thought — do NOT persist as assistant row',
            tool_calls=[tc],
        )

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.return_value = response_with_tools
            result = p.send()

        assert result == ''
        assert len(store_calls) == 1
        assert store_calls[0] == ''


# ─────────────────────────────────────────────────────────────────────────────
# D. MAX_TIMEOUT cap
# ─────────────────────────────────────────────────────────────────────────────


class TestMaxTimeoutCap:
    """Loop exits when wall-clock time exceeds MAX_TIMEOUT."""

    def _counter_time(self, trip_after: int, elapsed: float):
        """Counter-based time.time mock.

        First N calls return 0.0 (loop_start + checks up to `trip_after`),
        after that every call returns `elapsed` (blows past MAX_TIMEOUT).

        Safer than `iter([...])` because the test doesn't need to know
        exactly how many time.time() calls the loop body makes — the
        implementation can add additional calls (e.g. for compaction
        measurement in Commit 7) without the mock falling off the end.
        """
        counter = {'n': 0}

        def _now():
            counter['n'] += 1
            return 0.0 if counter['n'] <= trip_after else elapsed

        return _now

    def test_timeout_exits_loop(self):
        """Mock time.time to simulate elapsed > MAX_TIMEOUT after the first call."""
        p = _make_processor(max_timeout=10)
        _mock_store(p)

        tc = {'name': 'slow_tool', 'input': {}}
        response_with_tools = _make_llm_response(text='still working', tool_calls=[tc])
        response_clean = _make_llm_response(text='done', tool_calls=None)

        # loop_start + first while-check = 2 time.time() calls that see 0.0.
        # Third call onwards returns 11.0 which trips the MAX_TIMEOUT=10 gate.
        now = self._counter_time(trip_after=2, elapsed=11.0)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht, \
             patch('services.message_processor.time') as mock_time:
            mock_time.time.side_effect = now
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [
                response_with_tools,
                response_clean,
            ]
            p.send()

        # Only one provider call should happen (timeout prevents second iteration)
        assert mock_inst.return_value.send_messages.call_count == 1

    def test_store_called_after_timeout(self):
        """store() is still called when the loop exits via timeout."""
        p = _make_processor(max_timeout=5)
        store_calls = _mock_store(p)

        tc = {'name': 'slow_tool', 'input': {}}
        response = _make_llm_response(text='partial', tool_calls=[tc])

        # Same trip_after=2 pattern as above.
        now = self._counter_time(trip_after=2, elapsed=6.0)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht, \
             patch('services.message_processor.time') as mock_time:
            mock_time.time.side_effect = now
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.return_value = response
            p.send()

        assert len(store_calls) == 1

    def test_final_text_empty_on_timeout_cap(self):
        """When the loop exits via timeout, the stored final_text is '' — the
        last iteration's narration is NEVER promoted to the assistant row."""
        p = _make_processor(max_timeout=10)
        store_calls = _mock_store(p)

        tc = {'name': 'slow_tool', 'input': {}}
        # Mid-loop narration that MUST NOT become the final_text.
        response_with_tools = _make_llm_response(
            text='mid-loop narration that must not land as final', tool_calls=[tc]
        )

        now = self._counter_time(trip_after=2, elapsed=11.0)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht, \
             patch('services.message_processor.time') as mock_time:
            mock_time.time.side_effect = now
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.return_value = response_with_tools
            result = p.send()

        assert result == ''
        assert len(store_calls) == 1
        assert store_calls[0] == ''


# ─────────────────────────────────────────────────────────────────────────────
# E. Provider exception semantics
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderException:
    """send_messages raising must propagate; store() and postTurn() must not fire."""

    def test_exception_propagates(self):
        p = _make_processor()
        _mock_store(p)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.side_effect = RuntimeError('LLM down')
            with pytest.raises(RuntimeError, match='LLM down'):
                p.send()

    def test_store_not_called_on_provider_error(self):
        p = _make_processor()
        store_calls = _mock_store(p)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.side_effect = RuntimeError('LLM down')
            with pytest.raises(RuntimeError):
                p.send()

        assert store_calls == []

    def test_post_turn_not_called_on_provider_error(self):
        p = _make_processor()
        _mock_store(p)
        post_calls = []
        p.postTurn = lambda: post_calls.append(True)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.side_effect = RuntimeError('LLM down')
            with pytest.raises(RuntimeError):
                p.send()

        assert post_calls == []

    def test_uid_stays_none_on_provider_error(self):
        """_uid must remain None — no rows were written."""
        p = _make_processor()

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.side_effect = RuntimeError('error')
            with pytest.raises(RuntimeError):
                p.send()

        assert p._uid is None

    def test_db_has_zero_rows_on_provider_error(self, db):
        """End-to-end: real DB gets zero rows when provider errors."""
        from services.message_processor import MessageProcessor

        class _Fake(MessageProcessor):
            CHANNEL = 'test_send_err'
            ROLE = 'user'

            def getUserDefinition(self):
                return 'fake user'

            def getUserPrompt(self):
                return 'hello'

        p = _Fake('hello')

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.transcript_service._embed_entry'), \
             patch('services.transcript_service._trigger_episode_extraction'):
            mock_inst.return_value.send_messages.side_effect = RuntimeError('LLM down')
            with pytest.raises(RuntimeError):
                p.send()

        row = db.execute("SELECT COUNT(*) FROM transcript WHERE channel='test_send_err'").fetchone()
        assert row[0] == 0


# ─────────────────────────────────────────────────────────────────────────────
# F. postTurn() exception after store()
# ─────────────────────────────────────────────────────────────────────────────


class TestPostTurnException:
    """postTurn() error is caught + logged; stored rows survive; send() returns normally."""

    def test_send_returns_normally_when_post_turn_raises(self):
        p = _make_processor()
        _mock_store(p)
        p.postTurn = lambda: (_ for _ in ()).throw(ValueError('post-turn boom'))

        llm_resp = _make_llm_response(text='response text', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.return_value = llm_resp
            result = p.send()

        assert result == 'response text'

    def test_store_was_called_before_post_turn_error(self):
        p = _make_processor()
        store_calls = _mock_store(p)
        p.postTurn = lambda: (_ for _ in ()).throw(ValueError('boom'))

        llm_resp = _make_llm_response(text='x', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.return_value = llm_resp
            p.send()

        assert len(store_calls) == 1

    def test_post_turn_error_is_logged(self, caplog):
        import logging
        p = _make_processor()
        _mock_store(p)
        p.postTurn = lambda: (_ for _ in ()).throw(ValueError('logged boom'))

        llm_resp = _make_llm_response(text='x', tool_calls=None)

        with caplog.at_level(logging.ERROR, logger='services.message_processor'):
            with patch('services.providers.Providers.instance') as mock_inst:
                mock_inst.return_value.send_messages.return_value = llm_resp
                p.send()

        assert any('POSTTURN' in r.message for r in caplog.records)

    def test_store_exception_propagates_and_post_turn_not_called(self):
        """If store() raises, the exception propagates AND postTurn() is
        never called. The atomic-store contract requires that post-turn
        fan-out only runs when the turn is actually persisted.
        """
        p = _make_processor()

        def _boom(_response):
            raise RuntimeError('store boom')

        p.store = _boom
        post_calls = []
        p.postTurn = lambda: post_calls.append(1)

        llm_resp = _make_llm_response(text='final', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.return_value = llm_resp
            with pytest.raises(RuntimeError, match='store boom'):
                p.send()

        # postTurn() must NOT have been called
        assert post_calls == []


# ─────────────────────────────────────────────────────────────────────────────
# G. tool_synthesis DTO shape
# ─────────────────────────────────────────────────────────────────────────────


class TestToolSynthesisDtoShape:
    """tool_synthesis DTO must conform to the north star DTO contract."""

    def _get_synthesis_dto(self, narration_text='mid-loop narration'):
        p = _make_processor()
        _mock_store(p)

        tc = {'name': 'some_tool', 'input': {}}
        iter1 = _make_llm_response(text=narration_text, tool_calls=[tc])
        iter2 = _make_llm_response(text='final', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        return next(d for d in p._pending_tool_calls if d['name'] == 'tool_synthesis')

    def test_name_is_tool_synthesis(self):
        dto = self._get_synthesis_dto()
        assert dto['name'] == 'tool_synthesis'

    def test_ephemeral_is_1(self):
        dto = self._get_synthesis_dto()
        assert dto['ephemeral'] == 1

    def test_invoked_by_is_system(self):
        dto = self._get_synthesis_dto()
        assert dto['invoked_by'] == 'system'

    def test_params_is_empty_dict(self):
        dto = self._get_synthesis_dto()
        assert dto['params'] == {}

    def test_result_is_narration_text(self):
        dto = self._get_synthesis_dto('narration content here')
        assert dto['result'] == 'narration content here'

    def test_timestamp_is_iso_string(self):
        dto = self._get_synthesis_dto()
        ts = dto['timestamp']
        assert isinstance(ts, str)
        # ISO-8601 UTC strings end with +00:00
        assert '+00:00' in ts or ts.endswith('Z')

    def test_timestamp_key_present(self):
        dto = self._get_synthesis_dto()
        assert 'timestamp' in dto


# ─────────────────────────────────────────────────────────────────────────────
# H. Empty narration path
# ─────────────────────────────────────────────────────────────────────────────


class TestEmptyNarrationPath:
    """When llm_response.text is empty, no tool_synthesis DTO is appended."""

    def test_no_synthesis_dto_when_text_empty(self):
        p = _make_processor()
        _mock_store(p)

        tc = {'name': 'some_tool', 'input': {}}
        iter1 = _make_llm_response(text='', tool_calls=[tc])
        iter2 = _make_llm_response(text='final', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        names = [d['name'] for d in p._pending_tool_calls]
        assert 'tool_synthesis' not in names

    def test_no_synthesis_dto_when_text_none(self):
        """LLMResponse.text=None (falsy) also skips tool_synthesis DTO."""
        p = _make_processor()
        _mock_store(p)

        tc = {'name': 'some_tool', 'input': {}}

        from services.llm_service import LLMResponse
        iter1 = LLMResponse(text=None, model='m', provider='p', tool_calls=[tc])
        iter2 = _make_llm_response(text='final', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        names = [d['name'] for d in p._pending_tool_calls]
        assert 'tool_synthesis' not in names

    def test_loop_continues_on_empty_narration(self):
        """Empty narration doesn't break the loop — second call still happens."""
        p = _make_processor()
        _mock_store(p)

        tc = {'name': 'some_tool', 'input': {}}
        iter1 = _make_llm_response(text='', tool_calls=[tc])
        iter2 = _make_llm_response(text='result', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            result = p.send()

        assert result == 'result'
        assert mock_inst.return_value.send_messages.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# I. _emit_narration — base no-op + subclass override
# ─────────────────────────────────────────────────────────────────────────────


class TestEmitNarration:
    """_emit_narration base is a no-op; subclass overrides fire per iteration."""

    def test_base_emit_narration_is_noop(self):
        """Base _emit_narration() must not raise and must return None."""
        from services.message_processor import MessageProcessor

        class _Fake(MessageProcessor):
            CHANNEL = _CHANNEL
            ROLE = _ROLE

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return 'prompt'

        p = _Fake('input')
        result = p._emit_narration('some text', 0)
        assert result is None

    def test_subclass_override_fires_on_narration(self):
        """A subclass that overrides _emit_narration gets called with text + iteration."""
        from services.message_processor import MessageProcessor

        narrations = []

        class _FakeWithNarration(MessageProcessor):
            CHANNEL = _CHANNEL
            ROLE = _ROLE

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return 'prompt'

            def _emit_narration(self, text, iteration):
                narrations.append((text, iteration))

        p = _FakeWithNarration('input')
        # Patch store to avoid DB
        p.store = lambda r: setattr(p, '_uid', 99)

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='narration A', tool_calls=[tc])
        iter2 = _make_llm_response(text='final', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        assert len(narrations) == 1
        assert narrations[0] == ('narration A', 0)

    def test_narration_fires_once_per_iteration_with_text(self):
        """Two iterations with narration text → two _emit_narration calls."""
        from services.message_processor import MessageProcessor

        narrations = []

        class _FakeWithNarration(MessageProcessor):
            CHANNEL = _CHANNEL
            ROLE = _ROLE

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return 'prompt'

            def _emit_narration(self, text, iteration):
                narrations.append((text, iteration))

        p = _FakeWithNarration('input')
        p.store = lambda r: setattr(p, '_uid', 99)

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='step 1', tool_calls=[tc])
        iter2 = _make_llm_response(text='step 2', tool_calls=[tc])
        iter3 = _make_llm_response(text='final', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2, iter3]
            p.send()

        assert len(narrations) == 2
        assert narrations[0] == ('step 1', 0)
        assert narrations[1] == ('step 2', 1)


# ─────────────────────────────────────────────────────────────────────────────
# J. _drain_steering returns [] on base
# ─────────────────────────────────────────────────────────────────────────────


class TestDrainSteeringBase:
    """Base _drain_steering always returns [] → no user_steer DTOs."""

    def test_base_returns_empty_list(self):
        p = _make_processor()
        result = p._drain_steering('some-request-id')
        assert result == []

    def test_base_returns_empty_list_when_no_request_id(self):
        p = _make_processor()
        result = p._drain_steering(None)
        assert result == []

    def test_no_user_steer_dtos_on_base(self):
        p = _make_processor()
        _mock_store(p)

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='step', tool_calls=[tc])
        iter2 = _make_llm_response(text='done', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send(request_id='req-123')

        names = [d['name'] for d in p._pending_tool_calls]
        assert 'user_steer' not in names


# ─────────────────────────────────────────────────────────────────────────────
# K. Steering drained mid-loop
# ─────────────────────────────────────────────────────────────────────────────


class TestSteeringMidLoop:
    """When _drain_steering returns messages, user_steer DTOs are appended."""

    def _make_processor_with_steering(self, steer_messages):
        """Build a processor whose _drain_steering yields steer_messages once."""
        from services.message_processor import MessageProcessor

        _steers = list(steer_messages)
        _yielded = [False]

        class _FakeWithSteering(MessageProcessor):
            CHANNEL = _CHANNEL
            ROLE = _ROLE

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return 'prompt'

            def _drain_steering(self, request_id):
                if not _yielded[0]:
                    _yielded[0] = True
                    return _steers
                return []

        p = _FakeWithSteering('input')
        p.store = lambda r: setattr(p, '_uid', 99)
        return p

    def test_user_steer_dto_appended(self):
        p = self._make_processor_with_steering(['please be brief'])

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='', tool_calls=[tc])
        iter2 = _make_llm_response(text='done', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        names = [d['name'] for d in p._pending_tool_calls]
        assert 'user_steer' in names

    def test_user_steer_dto_shape(self):
        steer_text = 'be more concise'
        p = self._make_processor_with_steering([steer_text])

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='', tool_calls=[tc])
        iter2 = _make_llm_response(text='done', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        dto = next(d for d in p._pending_tool_calls if d['name'] == 'user_steer')
        assert dto['ephemeral'] == 1
        assert dto['invoked_by'] == 'system'
        assert dto['params'] == {}
        assert dto['result'] == steer_text
        assert isinstance(dto['timestamp'], str)
        assert '+00:00' in dto['timestamp']

    def test_steer_trail_entry_has_feedback_marker(self):
        p = self._make_processor_with_steering(['use bullet points'])

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='', tool_calls=[tc])
        iter2 = _make_llm_response(text='done', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        steer_lines = [l for l in p._act_trail if '[steer(' in l]
        assert len(steer_lines) == 1
        assert 'use bullet points' in steer_lines[0]
        assert 'observe in detail' in steer_lines[0]

    def test_multiple_steers_produce_multiple_dtos(self):
        p = self._make_processor_with_steering(['steer A', 'steer B'])

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='', tool_calls=[tc])
        iter2 = _make_llm_response(text='done', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        steer_dtos = [d for d in p._pending_tool_calls if d['name'] == 'user_steer']
        assert len(steer_dtos) == 2
        results = {d['result'] for d in steer_dtos}
        assert results == {'steer A', 'steer B'}


# ─────────────────────────────────────────────────────────────────────────────
# L. find_tools discovery propagates to getTools()
# ─────────────────────────────────────────────────────────────────────────────


class TestFindToolsDiscovery:
    """handleTool() updating _discovered_tools is visible in getTools() next iteration."""

    def test_discovered_tools_appear_in_get_tools(self):
        """After handleTool extends _discovered_tools, getTools() includes them."""
        p = _make_processor()
        _mock_store(p)

        # Simulate handleTool side effect: inject a fake external schema
        fake_schema = {'name': 'weather_tool', 'description': 'Get weather', 'parameters': {}}

        def _fake_handle_tool(tc):
            p._discovered_tools.append(fake_schema)
            return 'discovered weather_tool'

        tc = {'name': 'find_tools', 'input': {'query': 'weather'}}
        iter1 = _make_llm_response(text='', tool_calls=[tc])
        iter2 = _make_llm_response(text='done', tool_calls=None)

        tools_on_iter2 = []

        original_get_tools = p.getTools

        call_count = [0]

        def _patched_get_tools():
            call_count[0] += 1
            result = original_get_tools()
            if call_count[0] >= 2:
                tools_on_iter2.extend(result)
            return result

        p.getTools = _patched_get_tools
        p.handleTool = _fake_handle_tool

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.tool_schema_service.get_skill_schemas', return_value=[]):
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        tool_names = [t['name'] for t in tools_on_iter2]
        assert 'weather_tool' in tool_names

    def test_discovered_tools_survive_across_iterations(self):
        """Tools discovered in iteration N are still in _discovered_tools in N+1."""
        p = _make_processor()
        _mock_store(p)

        fake_schema = {'name': 'movie_tool', 'description': 'movies', 'parameters': {}}

        def _fake_handle_tool(tc):
            if tc['name'] == 'find_tools':
                p._discovered_tools.append(fake_schema)
            return 'ok'

        tc1 = {'name': 'find_tools', 'input': {}}
        tc2 = {'name': 'movie_tool', 'input': {}}
        iter1 = _make_llm_response(text='', tool_calls=[tc1])
        iter2 = _make_llm_response(text='', tool_calls=[tc2])
        iter3 = _make_llm_response(text='final', tool_calls=None)

        p.handleTool = _fake_handle_tool

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch('services.tool_schema_service.get_skill_schemas', return_value=[]):
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2, iter3]
            p.send()

        assert any(t['name'] == 'movie_tool' for t in p._discovered_tools)


# ─────────────────────────────────────────────────────────────────────────────
# M. Narration subclass override — on_narration callback
# ─────────────────────────────────────────────────────────────────────────────


class TestNarrationSubclassOverride:
    """Subclass with on_narration callback fires it once per iteration with text."""

    def _make_narration_processor(self):
        from services.message_processor import MessageProcessor

        narration_log = []

        class _FakeWithCallback(MessageProcessor):
            CHANNEL = _CHANNEL
            ROLE = _ROLE

            def __init__(self, raw_input, on_narration=None):
                super().__init__(raw_input)
                self._on_narration = on_narration

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return 'prompt'

            def _emit_narration(self, text, iteration):
                if self._on_narration:
                    try:
                        self._on_narration(text, iteration)
                    except Exception:
                        pass

        callback = lambda text, it: narration_log.append((text, it))
        p = _FakeWithCallback('input', on_narration=callback)
        p.store = lambda r: setattr(p, '_uid', 99)
        return p, narration_log

    def test_callback_fires_with_text_and_iteration(self):
        p, log = self._make_narration_processor()

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='Processing…', tool_calls=[tc])
        iter2 = _make_llm_response(text='final answer', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            p.send()

        assert len(log) == 1
        assert log[0] == ('Processing…', 0)

    def test_callback_not_fired_when_none(self):
        from services.message_processor import MessageProcessor

        class _FakeWithCallback(MessageProcessor):
            CHANNEL = _CHANNEL
            ROLE = _ROLE

            def __init__(self, raw_input):
                super().__init__(raw_input)
                self._on_narration = None

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return 'prompt'

            def _emit_narration(self, text, iteration):
                if self._on_narration:
                    self._on_narration(text, iteration)

        p = _FakeWithCallback('input')
        p.store = lambda r: setattr(p, '_uid', 99)

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='narration', tool_calls=[tc])
        iter2 = _make_llm_response(text='final', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst, \
             patch.object(p, 'handleTool') as mock_ht:
            mock_ht.return_value = 'r'
            mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
            # Must not raise
            p.send()

    def test_callback_exception_does_not_kill_loop(self, caplog):
        """If _emit_narration raises, the ACT loop must continue.

        North star § ACT Loop step 4: "Exceptions from the callback are
        logged and swallowed — they must never kill the ACT loop." This
        catch lives in the BASE class's send() so every subclass gets the
        guarantee (not just UserMessageProcessor).
        """
        import logging
        from services.message_processor import MessageProcessor

        class _FakeBoomNarration(MessageProcessor):
            CHANNEL = _CHANNEL
            ROLE = _ROLE

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                return 'prompt'

            def _emit_narration(self, text, iteration):
                raise RuntimeError('callback boom')

        p = _FakeBoomNarration('input')
        store_calls = _mock_store(p)

        tc = {'name': 'tool', 'input': {}}
        iter1 = _make_llm_response(text='narration', tool_calls=[tc])
        iter2 = _make_llm_response(text='final', tool_calls=None)

        with caplog.at_level(logging.ERROR, logger='services.message_processor'):
            with patch('services.providers.Providers.instance') as mock_inst, \
                 patch.object(p, 'handleTool') as mock_ht:
                mock_ht.return_value = 'r'
                mock_inst.return_value.send_messages.side_effect = [iter1, iter2]
                # No exception — the base swallows it.
                result = p.send()

        # Loop continued past the raising callback and completed cleanly.
        assert result == 'final'
        assert len(store_calls) == 1
        # The swallowed exception is logged at ERROR level.
        assert any('_emit_narration' in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# N. bind_current_processor context
# ─────────────────────────────────────────────────────────────────────────────


class TestBindCurrentProcessor:
    """current_processor() is bound during send() and cleared on exit."""

    def test_current_processor_set_during_send(self):
        from services.message_processor import current_processor, MessageProcessor

        captured = []

        class _FakeCapture(MessageProcessor):
            CHANNEL = _CHANNEL
            ROLE = _ROLE

            def getUserDefinition(self):
                return _USER_DEF

            def getUserPrompt(self):
                captured.append(current_processor())
                return 'prompt'

        p = _FakeCapture('input')
        p.store = lambda r: setattr(p, '_uid', 99)

        llm_resp = _make_llm_response(text='answer', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.return_value = llm_resp
            p.send()

        assert len(captured) == 1
        assert captured[0] is p

    def test_current_processor_cleared_after_send(self):
        from services.message_processor import current_processor

        p = _make_processor()
        _mock_store(p)

        llm_resp = _make_llm_response(text='answer', tool_calls=None)

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.return_value = llm_resp
            p.send()

        assert current_processor() is None

    def test_current_processor_cleared_even_on_exception(self):
        """bind_current_processor must clean up even when provider raises."""
        from services.message_processor import current_processor

        p = _make_processor()

        with patch('services.providers.Providers.instance') as mock_inst:
            mock_inst.return_value.send_messages.side_effect = RuntimeError('boom')
            with pytest.raises(RuntimeError):
                p.send()

        assert current_processor() is None
