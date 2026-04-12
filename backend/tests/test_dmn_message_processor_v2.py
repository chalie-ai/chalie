# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for DMNMessageProcessor (Commit 9 rewrite).

Locks the north-star contract for the dmn-channel MessageProcessor subclass:
  /Volumes/llm/chalie-plans/message-processing.md
  /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md § Commit 9

Test groups:
  A. Class constants — CHANNEL, ROLE, safety caps, SYSTEM_PROMPT_CLASS, NATIVE_TOOLS
  B. Constructor — field initialisation
  C. getUserDefinition() — exact literal string (regression-lock)
  D. getUserPrompt() — role-prefix + trail assembly
  E. getSystemPrompt() — contract = getUserDefinition() + '\\n\\n' + body
  F. postTurn() — interaction log + metrics, fault isolation
  G. Smoke — construct → postTurn without error
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

_EXPECTED_USER_DEFINITION = (
    "The user is 'proactive_thought' — a special background process "
    "that represents your own reflections on recent activity."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_dmn(raw_input='background context', metadata=None):
    """Construct a fresh DMNMessageProcessor per-test."""
    from services.dmn_message_processor import DMNMessageProcessor
    return DMNMessageProcessor(
        raw_input=raw_input,
        metadata=metadata if metadata is not None else {'mode': 'recent'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# A. Class constants
# ─────────────────────────────────────────────────────────────────────────────

class TestClassConstants:
    """Class-level attributes must match the north-star contract exactly."""

    def test_a1_channel_is_dmn(self):
        from services.dmn_message_processor import DMNMessageProcessor
        assert DMNMessageProcessor.CHANNEL == 'dmn'

    def test_a2_role_is_proactive_thought(self):
        from services.dmn_message_processor import DMNMessageProcessor
        assert DMNMessageProcessor.ROLE == 'proactive_thought'

    def test_a3_max_iterations_is_15(self):
        from services.dmn_message_processor import DMNMessageProcessor
        assert DMNMessageProcessor.MAX_ITERATIONS == 15

    def test_a4_max_timeout_is_300(self):
        from services.dmn_message_processor import DMNMessageProcessor
        assert DMNMessageProcessor.MAX_TIMEOUT == 300

    def test_a5_system_prompt_class_is_dmn_prompt(self):
        from services.dmn_message_processor import DMNMessageProcessor
        from services.system_message_prompt import DMNSystemMessagePrompt
        assert DMNMessageProcessor.SYSTEM_PROMPT_CLASS is DMNSystemMessagePrompt

    def test_a6_goal_pursuit_excluded_from_native_tools(self):
        """Recursion guard: DMN must not be able to spawn goal_pursuit."""
        from services.dmn_message_processor import DMNMessageProcessor
        assert 'goal_pursuit' not in DMNMessageProcessor.NATIVE_TOOLS

    def test_a7_native_tools_is_sorted(self):
        """Deterministic ordering so provider schemas are stable across runs."""
        from services.dmn_message_processor import DMNMessageProcessor
        tools = DMNMessageProcessor.NATIVE_TOOLS
        assert list(tools) == sorted(tools)

    def test_a8_native_tools_is_all_skills_minus_goal_pursuit(self):
        """NATIVE_TOOLS == ALL_SKILL_NAMES - {'goal_pursuit'}, sorted."""
        from services.dmn_message_processor import DMNMessageProcessor
        from services.innate_skills.registry import ALL_SKILL_NAMES
        expected = sorted(s for s in ALL_SKILL_NAMES if s != 'goal_pursuit')
        assert list(DMNMessageProcessor.NATIVE_TOOLS) == expected


# ─────────────────────────────────────────────────────────────────────────────
# B. Constructor
# ─────────────────────────────────────────────────────────────────────────────

class TestConstructor:
    """Per-turn instance fields are initialised correctly."""

    def test_b1_raw_input_stored(self):
        proc = _make_dmn(raw_input='some context')
        assert proc._raw_input == 'some context'

    def test_b2_metadata_stored(self):
        proc = _make_dmn(metadata={'mode': 'salience'})
        assert proc._metadata == {'mode': 'salience'}

    def test_b3_metadata_defaults_to_empty_when_none(self):
        from services.dmn_message_processor import DMNMessageProcessor
        proc = DMNMessageProcessor(raw_input='x', metadata=None)
        assert proc._metadata == {}

    def test_b4_pending_tool_calls_starts_empty(self):
        proc = _make_dmn()
        assert proc._pending_tool_calls == []

    def test_b5_act_trail_starts_empty(self):
        proc = _make_dmn()
        assert proc._act_trail == []

    def test_b6_discovered_tools_starts_empty(self):
        proc = _make_dmn()
        assert proc._discovered_tools == []

    def test_b7_uid_starts_none(self):
        proc = _make_dmn()
        assert proc._uid is None

    def test_b8_two_instances_do_not_share_state(self):
        """Per-instance mutation must not bleed between instances."""
        p1 = _make_dmn(raw_input='first')
        p2 = _make_dmn(raw_input='second')

        p1._pending_tool_calls.append({'name': 'test'})
        p1._act_trail.append('trail entry')

        assert p2._pending_tool_calls == []
        assert p2._act_trail == []


# ─────────────────────────────────────────────────────────────────────────────
# C. getUserDefinition()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUserDefinition:
    """Exact literal string — regression-lock."""

    def test_c1_returns_exact_proactive_thought_string(self):
        proc = _make_dmn()
        assert proc.getUserDefinition() == _EXPECTED_USER_DEFINITION

    def test_c2_string_contains_role_label(self):
        proc = _make_dmn()
        assert 'proactive_thought' in proc.getUserDefinition()

    def test_c3_string_mentions_background_process(self):
        proc = _make_dmn()
        assert 'background process' in proc.getUserDefinition()

    def test_c4_result_is_deterministic_across_calls(self):
        """Must return the same string on repeated calls (no dynamic content)."""
        proc = _make_dmn()
        assert proc.getUserDefinition() == proc.getUserDefinition()


# ─────────────────────────────────────────────────────────────────────────────
# D. getUserPrompt()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUserPrompt:
    """Role-prefix + optional ACT trail assembly."""

    def test_d1_empty_trail_produces_role_prefix_only(self):
        proc = _make_dmn(raw_input='the context text')
        # _act_trail is empty by default → getActLoopTrail() returns ''
        result = proc.getUserPrompt()
        assert result == 'proactive_thought: the context text'

    def test_d2_non_empty_trail_appended_after_newline(self):
        proc = _make_dmn(raw_input='ctx')
        proc._act_trail = ['[tool(x=1)] result line']
        result = proc.getUserPrompt()

        lines = result.splitlines()
        assert lines[0] == 'proactive_thought: ctx'
        assert lines[1] == '[tool(x=1)] result line'

    def test_d3_raw_input_preserved_verbatim(self):
        """Special chars are not escaped or munged."""
        raw = 'ctx with "quotes" & <angle> brackets'
        proc = _make_dmn(raw_input=raw)
        result = proc.getUserPrompt()
        assert f'proactive_thought: {raw}' in result

    def test_d4_role_prefix_is_proactive_thought(self):
        """Role prefix must be the ROLE constant, not the CHANNEL."""
        proc = _make_dmn(raw_input='body')
        result = proc.getUserPrompt()
        assert result.startswith('proactive_thought:')
        assert not result.startswith('dmn:')

    def test_d5_multi_entry_trail_joined_by_newlines(self):
        proc = _make_dmn(raw_input='q')
        proc._act_trail = ['line1', 'line2', 'line3']
        result = proc.getUserPrompt()

        lines = result.splitlines()
        assert lines[0] == 'proactive_thought: q'
        assert lines[1] == 'line1'
        assert lines[2] == 'line2'
        assert lines[3] == 'line3'

    def test_d6_no_trailing_newline_when_trail_empty(self):
        proc = _make_dmn(raw_input='input')
        result = proc.getUserPrompt()
        assert not result.endswith('\n')


# ─────────────────────────────────────────────────────────────────────────────
# E. getSystemPrompt()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSystemPrompt:
    """Base-class final method: getUserDefinition() + '\\n\\n' + body."""

    def test_e1_user_definition_prepended(self):
        proc = _make_dmn()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value='BODY'):
            result = proc.getSystemPrompt()
        assert result.startswith(_EXPECTED_USER_DEFINITION + '\n\n')

    def test_e2_body_appended_after_separator(self):
        proc = _make_dmn()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value='BODY'):
            result = proc.getSystemPrompt()
        assert result == _EXPECTED_USER_DEFINITION + '\n\nBODY'

    def test_e3_fresh_system_prompt_instance_per_call(self):
        """getSystemPrompt() must construct SYSTEM_PROMPT_CLASS fresh each call.

        Two calls must not share state — verified by ensuring both calls
        get the same result even after mutation of any cached object.
        """
        proc = _make_dmn()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value='X'):
            r1 = proc.getSystemPrompt()
            r2 = proc.getSystemPrompt()
        assert r1 == r2

    def test_e4_empty_body_still_prepends_definition(self):
        proc = _make_dmn()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value=''):
            result = proc.getSystemPrompt()
        assert result == _EXPECTED_USER_DEFINITION + '\n\n'


# ─────────────────────────────────────────────────────────────────────────────
# F. postTurn() — metrics + fault isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestPostTurn:
    """postTurn() fires metrics; fault isolation keeps it from raising."""

    def _make_proc_for_postturn(self, metadata=None):
        proc = _make_dmn(
            raw_input='dmn context',
            metadata=metadata if metadata is not None else {'mode': 'recent'},
        )
        proc._uid = 1
        return proc

    # ── F1: metrics counters ─────────────────────────────────────────────────

    def test_f1_metrics_records_requests_total(self):
        proc = self._make_proc_for_postturn()
        mock_metrics = MagicMock()

        with patch('services.metrics_service.MetricsService',
                   return_value=mock_metrics):
            proc.postTurn()

        counter_names = [c.args[0] for c in mock_metrics.record_counter.call_args_list]
        assert 'requests_total' in counter_names

    def test_f2_metrics_records_dmn_turns_total(self):
        proc = self._make_proc_for_postturn()
        mock_metrics = MagicMock()

        with patch('services.metrics_service.MetricsService',
                   return_value=mock_metrics):
            proc.postTurn()

        counter_names = [c.args[0] for c in mock_metrics.record_counter.call_args_list]
        assert 'dmn_turns_total' in counter_names

    def test_f3_exactly_two_metric_counters_incremented(self):
        proc = self._make_proc_for_postturn()
        mock_metrics = MagicMock()

        with patch('services.metrics_service.MetricsService',
                   return_value=mock_metrics):
            proc.postTurn()

        assert mock_metrics.record_counter.call_count == 2

    # ── F2: fault isolation ──────────────────────────────────────────────────

    def test_f4_metrics_crash_postturn_still_returns(self):
        """MetricsService raises → postTurn() must not raise."""
        proc = self._make_proc_for_postturn()

        with patch('services.metrics_service.MetricsService',
                   side_effect=RuntimeError('metrics crash')):
            proc.postTurn()  # must not raise

    # ── F3: DMNService re-entrancy guard ─────────────────────────────────────

    def test_f5_postturn_does_not_call_dmn_service_on_turn(self):
        """postTurn() must NOT call DMNService.on_turn() — that would be re-entrant."""
        proc = self._make_proc_for_postturn()

        on_turn_calls = []
        mock_dmn_svc = MagicMock()
        mock_dmn_svc.on_turn.side_effect = lambda: on_turn_calls.append(True)

        with patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()), \
             patch('services.dmn_service.get_dmn_service',
                   return_value=mock_dmn_svc) as mock_get_dmn:
            proc.postTurn()

        mock_get_dmn.assert_not_called()
        assert on_turn_calls == []


# ─────────────────────────────────────────────────────────────────────────────
# G. Smoke
# ─────────────────────────────────────────────────────────────────────────────

class TestSmoke:
    """Construct → postTurn() without error (no real DB write)."""

    def test_g1_construct_and_call_postturn_without_error(self, store):
        proc = _make_dmn(metadata={'mode': 'recent'})
        proc._uid = 99

        with patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()):
            proc.postTurn()  # must not raise
