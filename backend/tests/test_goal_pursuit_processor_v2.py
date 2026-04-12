# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for GoalPursuitProcessor (Commit 9 rewrite).

Locks the north-star contract for the goal_pursuit-channel MessageProcessor subclass:
  /Volumes/llm/chalie-plans/message-processing.md
  /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md § Commit 9

Test groups:
  A. Class constants — CHANNEL (flat), ROLE, safety caps, SYSTEM_PROMPT_CLASS, NATIVE_TOOLS
  B. Constructor — field initialisation, metadata carries pursuit_id
  C. getUserDefinition() — exact literal string (regression-lock)
  D. getUserPrompt() — role-prefix + trail assembly
  E. getSystemPrompt() — contract = getUserDefinition() + '\\n\\n' + body
  F. postTurn() — interaction log + metrics, payload includes pursuit_id, fault isolation
  G. Smoke — construct → postTurn without error
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

_EXPECTED_USER_DEFINITION = (
    "The user is 'goal_pursuit' — a background process that is "
    "autonomously pursuing a long-running goal on behalf of the user."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_gpp(raw_input='research machine learning techniques', metadata=None):
    """Construct a fresh GoalPursuitProcessor per-test."""
    from services.goal_pursuit_processor import GoalPursuitProcessor
    return GoalPursuitProcessor(
        raw_input=raw_input,
        metadata=metadata if metadata is not None else {'pursuit_id': 'abc123'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# A. Class constants
# ─────────────────────────────────────────────────────────────────────────────

class TestClassConstants:
    """Class-level attributes must match the north-star contract exactly."""

    def test_a1_channel_is_flat_goal_pursuit(self):
        """CHANNEL must be flat 'goal_pursuit', no colon or id suffix."""
        from services.goal_pursuit_processor import GoalPursuitProcessor
        assert GoalPursuitProcessor.CHANNEL == 'goal_pursuit'
        assert ':' not in GoalPursuitProcessor.CHANNEL

    def test_a2_role_is_goal_pursuit(self):
        from services.goal_pursuit_processor import GoalPursuitProcessor
        assert GoalPursuitProcessor.ROLE == 'goal_pursuit'

    def test_a3_max_iterations_is_50(self):
        from services.goal_pursuit_processor import GoalPursuitProcessor
        assert GoalPursuitProcessor.MAX_ITERATIONS == 50

    def test_a4_max_timeout_is_7200(self):
        from services.goal_pursuit_processor import GoalPursuitProcessor
        assert GoalPursuitProcessor.MAX_TIMEOUT == 7200

    def test_a5_system_prompt_class_is_goal_pursuit_prompt(self):
        from services.goal_pursuit_processor import GoalPursuitProcessor
        from services.system_message_prompt import GoalPursuitSystemMessagePrompt
        assert GoalPursuitProcessor.SYSTEM_PROMPT_CLASS is GoalPursuitSystemMessagePrompt

    def test_a6_goal_pursuit_excluded_from_native_tools(self):
        """Recursion guard: goal_pursuit must not be able to spawn itself."""
        from services.goal_pursuit_processor import GoalPursuitProcessor
        assert 'goal_pursuit' not in GoalPursuitProcessor.NATIVE_TOOLS

    def test_a7_native_tools_is_sorted(self):
        """Deterministic ordering so provider schemas are stable across runs."""
        from services.goal_pursuit_processor import GoalPursuitProcessor
        tools = GoalPursuitProcessor.NATIVE_TOOLS
        assert list(tools) == sorted(tools)

    def test_a8_native_tools_is_all_skills_minus_goal_pursuit(self):
        """NATIVE_TOOLS == ALL_SKILL_NAMES - {'goal_pursuit'}, sorted."""
        from services.goal_pursuit_processor import GoalPursuitProcessor
        from services.innate_skills.registry import ALL_SKILL_NAMES
        expected = sorted(s for s in ALL_SKILL_NAMES if s != 'goal_pursuit')
        assert list(GoalPursuitProcessor.NATIVE_TOOLS) == expected


# ─────────────────────────────────────────────────────────────────────────────
# B. Constructor
# ─────────────────────────────────────────────────────────────────────────────

class TestConstructor:
    """Per-turn instance fields are initialised correctly."""

    def test_b1_raw_input_stored(self):
        proc = _make_gpp(raw_input='my goal text')
        assert proc._raw_input == 'my goal text'

    def test_b2_metadata_stored_with_pursuit_id(self):
        proc = _make_gpp(metadata={'pursuit_id': 'hex-id-42'})
        assert proc._metadata == {'pursuit_id': 'hex-id-42'}

    def test_b3_metadata_defaults_to_empty_when_none(self):
        from services.goal_pursuit_processor import GoalPursuitProcessor
        proc = GoalPursuitProcessor(raw_input='x', metadata=None)
        assert proc._metadata == {}

    def test_b4_pending_tool_calls_starts_empty(self):
        proc = _make_gpp()
        assert proc._pending_tool_calls == []

    def test_b5_act_trail_starts_empty(self):
        proc = _make_gpp()
        assert proc._act_trail == []

    def test_b6_uid_starts_none(self):
        proc = _make_gpp()
        assert proc._uid is None

    def test_b7_two_instances_do_not_share_state(self):
        """Per-instance mutation must not bleed between instances."""
        p1 = _make_gpp(raw_input='goal 1')
        p2 = _make_gpp(raw_input='goal 2')

        p1._pending_tool_calls.append({'name': 'test'})
        p1._act_trail.append('trail entry')

        assert p2._pending_tool_calls == []
        assert p2._act_trail == []


# ─────────────────────────────────────────────────────────────────────────────
# C. getUserDefinition()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUserDefinition:
    """Exact literal string — regression-lock."""

    def test_c1_returns_exact_goal_pursuit_string(self):
        proc = _make_gpp()
        assert proc.getUserDefinition() == _EXPECTED_USER_DEFINITION

    def test_c2_string_contains_role_label(self):
        proc = _make_gpp()
        assert 'goal_pursuit' in proc.getUserDefinition()

    def test_c3_string_mentions_background_process(self):
        proc = _make_gpp()
        assert 'background process' in proc.getUserDefinition()

    def test_c4_result_is_deterministic_across_calls(self):
        proc = _make_gpp()
        assert proc.getUserDefinition() == proc.getUserDefinition()


# ─────────────────────────────────────────────────────────────────────────────
# D. getUserPrompt()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUserPrompt:
    """Role-prefix + optional ACT trail assembly."""

    def test_d1_empty_trail_produces_role_prefix_only(self):
        proc = _make_gpp(raw_input='research task')
        result = proc.getUserPrompt()
        assert result == 'goal_pursuit: research task'

    def test_d2_non_empty_trail_appended_after_newline(self):
        proc = _make_gpp(raw_input='goal text')
        proc._act_trail = ['[read(url=example.com)] page content']
        result = proc.getUserPrompt()

        lines = result.splitlines()
        assert lines[0] == 'goal_pursuit: goal text'
        assert lines[1] == '[read(url=example.com)] page content'

    def test_d3_raw_input_preserved_verbatim(self):
        """Goal string is not escaped or truncated."""
        raw = 'Research "quantum computing" & write a summary < 500 words'
        proc = _make_gpp(raw_input=raw)
        result = proc.getUserPrompt()
        assert f'goal_pursuit: {raw}' in result

    def test_d4_role_prefix_is_goal_pursuit(self):
        """Role prefix must be the ROLE constant, not the CHANNEL."""
        proc = _make_gpp(raw_input='the goal')
        result = proc.getUserPrompt()
        assert result.startswith('goal_pursuit:')

    def test_d5_multi_entry_trail_all_present(self):
        proc = _make_gpp(raw_input='q')
        proc._act_trail = ['line1', 'line2', 'line3']
        result = proc.getUserPrompt()

        lines = result.splitlines()
        assert lines[0] == 'goal_pursuit: q'
        assert lines[1] == 'line1'
        assert lines[2] == 'line2'
        assert lines[3] == 'line3'

    def test_d6_no_trailing_newline_when_trail_empty(self):
        proc = _make_gpp(raw_input='input')
        result = proc.getUserPrompt()
        assert not result.endswith('\n')


# ─────────────────────────────────────────────────────────────────────────────
# E. getSystemPrompt()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSystemPrompt:
    """Base-class final method: getUserDefinition() + '\\n\\n' + body."""

    def test_e1_user_definition_prepended(self):
        proc = _make_gpp()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value='BODY'):
            result = proc.getSystemPrompt()
        assert result.startswith(_EXPECTED_USER_DEFINITION + '\n\n')

    def test_e2_body_appended_after_separator(self):
        proc = _make_gpp()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value='BODY'):
            result = proc.getSystemPrompt()
        assert result == _EXPECTED_USER_DEFINITION + '\n\nBODY'

    def test_e3_empty_body_still_prepends_definition(self):
        proc = _make_gpp()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value=''):
            result = proc.getSystemPrompt()
        assert result == _EXPECTED_USER_DEFINITION + '\n\n'


# ─────────────────────────────────────────────────────────────────────────────
# F. postTurn() — metrics + fault isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestPostTurn:
    """postTurn() fires metrics; fault isolation keeps it from raising."""

    def _make_proc_for_postturn(self, metadata=None):
        proc = _make_gpp(
            raw_input='pursuit goal text',
            metadata=metadata if metadata is not None else {'pursuit_id': 'abc-hex'},
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

    def test_f2_metrics_records_goal_pursuit_turns_total(self):
        proc = self._make_proc_for_postturn()
        mock_metrics = MagicMock()

        with patch('services.metrics_service.MetricsService',
                   return_value=mock_metrics):
            proc.postTurn()

        counter_names = [c.args[0] for c in mock_metrics.record_counter.call_args_list]
        assert 'goal_pursuit_turns_total' in counter_names

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


# ─────────────────────────────────────────────────────────────────────────────
# G. Smoke
# ─────────────────────────────────────────────────────────────────────────────

class TestSmoke:
    """Construct → postTurn() without error (no real DB write)."""

    def test_g1_construct_and_call_postturn_without_error(self, store):
        proc = _make_gpp(metadata={'pursuit_id': 'smoke-id'})
        proc._uid = 99

        with patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()):
            proc.postTurn()  # must not raise
