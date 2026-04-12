# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for ScheduledMessageProcessor (Commit 9 rewrite).

Locks the north-star contract for the scheduled-channel MessageProcessor subclass:
  /Volumes/llm/chalie-plans/message-processing.md
  /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md § Commit 9

Test groups:
  A. Class constants — CHANNEL (flat), ROLE, limits (base defaults), SYSTEM_PROMPT_CLASS, NATIVE_TOOLS
  B. Constructor — field initialisation, metadata carries item_id
  C. getUserDefinition() — exact literal string (regression-lock)
  D. getUserPrompt() — role-prefix + trail assembly
  E. getSystemPrompt() — contract = getUserDefinition() + '\\n\\n' + body
  F. postTurn() — interaction log + metrics + _mark_scheduled_item_executed gate
  G. _mark_scheduled_item_executed() — logged no-op, does not raise
  H. Smoke — construct → postTurn without error
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

_EXPECTED_USER_DEFINITION = (
    "The user is 'scheduled' — an automated background process "
    "executing a task that the user scheduled in advance."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_smp(raw_input='send weekly summary report', metadata=None):
    """Construct a fresh ScheduledMessageProcessor per-test."""
    from services.scheduled_message_processor import ScheduledMessageProcessor
    return ScheduledMessageProcessor(
        raw_input=raw_input,
        metadata=metadata if metadata is not None else {'item_id': 'item-42'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# A. Class constants
# ─────────────────────────────────────────────────────────────────────────────

class TestClassConstants:
    """Class-level attributes must match the north-star contract exactly."""

    def test_a1_channel_is_flat_scheduled(self):
        """CHANNEL must be flat 'scheduled', no colon or id suffix."""
        from services.scheduled_message_processor import ScheduledMessageProcessor
        assert ScheduledMessageProcessor.CHANNEL == 'scheduled'
        assert ':' not in ScheduledMessageProcessor.CHANNEL

    def test_a2_role_is_scheduled(self):
        from services.scheduled_message_processor import ScheduledMessageProcessor
        assert ScheduledMessageProcessor.ROLE == 'scheduled'

    def test_a3_inherits_base_max_iterations_30(self):
        """Scheduled uses the base default of 30 iterations (no override)."""
        from services.scheduled_message_processor import ScheduledMessageProcessor
        assert ScheduledMessageProcessor.MAX_ITERATIONS == 30

    def test_a4_inherits_base_max_timeout_900(self):
        """Scheduled uses the base default of 900 seconds (no override)."""
        from services.scheduled_message_processor import ScheduledMessageProcessor
        assert ScheduledMessageProcessor.MAX_TIMEOUT == 900

    def test_a5_system_prompt_class_is_scheduled_prompt(self):
        from services.scheduled_message_processor import ScheduledMessageProcessor
        from services.system_message_prompt import ScheduledSystemMessagePrompt
        assert ScheduledMessageProcessor.SYSTEM_PROMPT_CLASS is ScheduledSystemMessagePrompt

    def test_a6_schedule_excluded_from_native_tools(self):
        """Recursion guard: scheduled must not be able to create new schedules."""
        from services.scheduled_message_processor import ScheduledMessageProcessor
        assert 'schedule' not in ScheduledMessageProcessor.NATIVE_TOOLS

    def test_a7_goal_pursuit_excluded_from_native_tools(self):
        """Recursion guard: scheduled must not be able to spawn background tasks."""
        from services.scheduled_message_processor import ScheduledMessageProcessor
        assert 'goal_pursuit' not in ScheduledMessageProcessor.NATIVE_TOOLS

    def test_a8_native_tools_is_sorted(self):
        """Deterministic ordering so provider schemas are stable across runs."""
        from services.scheduled_message_processor import ScheduledMessageProcessor
        tools = ScheduledMessageProcessor.NATIVE_TOOLS
        assert list(tools) == sorted(tools)

    def test_a9_native_tools_is_all_skills_minus_excluded(self):
        """NATIVE_TOOLS == ALL_SKILL_NAMES - {'schedule', 'goal_pursuit'}, sorted."""
        from services.scheduled_message_processor import ScheduledMessageProcessor
        from services.innate_skills.registry import ALL_SKILL_NAMES
        excluded = {'schedule', 'goal_pursuit'}
        expected = sorted(s for s in ALL_SKILL_NAMES if s not in excluded)
        assert list(ScheduledMessageProcessor.NATIVE_TOOLS) == expected


# ─────────────────────────────────────────────────────────────────────────────
# B. Constructor
# ─────────────────────────────────────────────────────────────────────────────

class TestConstructor:
    """Per-turn instance fields are initialised correctly."""

    def test_b1_raw_input_stored(self):
        proc = _make_smp(raw_input='scheduled prompt text')
        assert proc._raw_input == 'scheduled prompt text'

    def test_b2_metadata_stored_with_item_id(self):
        proc = _make_smp(metadata={'item_id': 'item-99'})
        assert proc._metadata == {'item_id': 'item-99'}

    def test_b3_metadata_defaults_to_empty_when_none(self):
        from services.scheduled_message_processor import ScheduledMessageProcessor
        proc = ScheduledMessageProcessor(raw_input='x', metadata=None)
        assert proc._metadata == {}

    def test_b4_pending_tool_calls_starts_empty(self):
        proc = _make_smp()
        assert proc._pending_tool_calls == []

    def test_b5_act_trail_starts_empty(self):
        proc = _make_smp()
        assert proc._act_trail == []

    def test_b6_uid_starts_none(self):
        proc = _make_smp()
        assert proc._uid is None

    def test_b7_two_instances_do_not_share_state(self):
        p1 = _make_smp(raw_input='prompt 1')
        p2 = _make_smp(raw_input='prompt 2')

        p1._pending_tool_calls.append({'name': 'test'})
        p1._act_trail.append('trail entry')

        assert p2._pending_tool_calls == []
        assert p2._act_trail == []


# ─────────────────────────────────────────────────────────────────────────────
# C. getUserDefinition()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUserDefinition:
    """Exact literal string — regression-lock."""

    def test_c1_returns_exact_scheduled_string(self):
        proc = _make_smp()
        assert proc.getUserDefinition() == _EXPECTED_USER_DEFINITION

    def test_c2_string_contains_role_label(self):
        proc = _make_smp()
        assert 'scheduled' in proc.getUserDefinition()

    def test_c3_string_mentions_automated_background_process(self):
        proc = _make_smp()
        defn = proc.getUserDefinition()
        assert 'automated' in defn or 'background process' in defn

    def test_c4_result_is_deterministic_across_calls(self):
        proc = _make_smp()
        assert proc.getUserDefinition() == proc.getUserDefinition()


# ─────────────────────────────────────────────────────────────────────────────
# D. getUserPrompt()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUserPrompt:
    """Role-prefix + optional ACT trail assembly."""

    def test_d1_empty_trail_produces_role_prefix_only(self):
        proc = _make_smp(raw_input='send daily digest')
        result = proc.getUserPrompt()
        assert result == 'scheduled: send daily digest'

    def test_d2_non_empty_trail_appended_after_newline(self):
        proc = _make_smp(raw_input='the prompt')
        proc._act_trail = ['[memory(radius=0.2)] some memories']
        result = proc.getUserPrompt()

        lines = result.splitlines()
        assert lines[0] == 'scheduled: the prompt'
        assert lines[1] == '[memory(radius=0.2)] some memories'

    def test_d3_raw_input_preserved_verbatim(self):
        raw = 'Run "weekly report" & email to user@example.com'
        proc = _make_smp(raw_input=raw)
        result = proc.getUserPrompt()
        assert f'scheduled: {raw}' in result

    def test_d4_role_prefix_is_scheduled(self):
        proc = _make_smp(raw_input='body')
        result = proc.getUserPrompt()
        assert result.startswith('scheduled:')

    def test_d5_multi_entry_trail_all_present(self):
        proc = _make_smp(raw_input='q')
        proc._act_trail = ['line1', 'line2']
        result = proc.getUserPrompt()

        lines = result.splitlines()
        assert lines[0] == 'scheduled: q'
        assert lines[1] == 'line1'
        assert lines[2] == 'line2'

    def test_d6_no_trailing_newline_when_trail_empty(self):
        proc = _make_smp(raw_input='input')
        result = proc.getUserPrompt()
        assert not result.endswith('\n')


# ─────────────────────────────────────────────────────────────────────────────
# E. getSystemPrompt()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSystemPrompt:
    """Base-class final method: getUserDefinition() + '\\n\\n' + body."""

    def test_e1_user_definition_prepended(self):
        proc = _make_smp()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value='BODY'):
            result = proc.getSystemPrompt()
        assert result.startswith(_EXPECTED_USER_DEFINITION + '\n\n')

    def test_e2_body_appended_after_separator(self):
        proc = _make_smp()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value='BODY'):
            result = proc.getSystemPrompt()
        assert result == _EXPECTED_USER_DEFINITION + '\n\nBODY'

    def test_e3_empty_body_still_prepends_definition(self):
        proc = _make_smp()
        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt', return_value=''):
            result = proc.getSystemPrompt()
        assert result == _EXPECTED_USER_DEFINITION + '\n\n'


# ─────────────────────────────────────────────────────────────────────────────
# F. postTurn() — metrics + mark-executed gate + fault isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestPostTurn:
    """postTurn() fires metrics, conditionally mark-executed, fault-isolated."""

    def _make_proc_for_postturn(self, metadata=None):
        proc = _make_smp(
            raw_input='scheduled prompt text',
            metadata=metadata if metadata is not None else {'item_id': 'item-42'},
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

    def test_f2_metrics_records_scheduled_turns_total(self):
        proc = self._make_proc_for_postturn()
        mock_metrics = MagicMock()

        with patch('services.metrics_service.MetricsService',
                   return_value=mock_metrics):
            proc.postTurn()

        counter_names = [c.args[0] for c in mock_metrics.record_counter.call_args_list]
        assert 'scheduled_turns_total' in counter_names

    def test_f3_exactly_two_metric_counters_incremented(self):
        proc = self._make_proc_for_postturn()
        mock_metrics = MagicMock()

        with patch('services.metrics_service.MetricsService',
                   return_value=mock_metrics):
            proc.postTurn()

        assert mock_metrics.record_counter.call_count == 2

    # ── F2: _mark_scheduled_item_executed gate ───────────────────────────────

    def test_f4_mark_executed_called_when_item_id_present(self):
        proc = self._make_proc_for_postturn(metadata={'item_id': 'item-55'})
        executed_ids = []
        proc._mark_scheduled_item_executed = lambda iid: executed_ids.append(iid)

        with patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()):
            proc.postTurn()

        assert executed_ids == ['item-55']

    def test_f5_mark_executed_not_called_when_item_id_missing(self):
        proc = self._make_proc_for_postturn(metadata={})
        executed_ids = []
        proc._mark_scheduled_item_executed = lambda iid: executed_ids.append(iid)

        with patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()):
            proc.postTurn()

        assert executed_ids == []

    # ── F3: fault isolation ──────────────────────────────────────────────────

    def test_f6_metrics_crash_postturn_still_returns(self):
        proc = self._make_proc_for_postturn()

        with patch('services.metrics_service.MetricsService',
                   side_effect=RuntimeError('metrics crash')):
            proc.postTurn()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# G. _mark_scheduled_item_executed()
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkScheduledItemExecuted:
    """_mark_scheduled_item_executed() is a logged no-op that must never raise."""

    def test_g1_does_not_raise_on_any_item_id(self):
        proc = _make_smp()
        proc._mark_scheduled_item_executed('item-arbitrary-id')  # must not raise

    def test_g2_does_not_raise_on_empty_string(self):
        proc = _make_smp()
        proc._mark_scheduled_item_executed('')  # must not raise

    def test_g3_does_not_write_to_database(self):
        """The no-op must not touch the DB — scheduler_service already marked it."""
        import services.database_service as _db_mod
        original_get = _db_mod.get_shared_db_service

        db_called = []

        def _spy():
            db_called.append(True)
            return original_get()

        _db_mod.get_shared_db_service = _spy
        try:
            proc = _make_smp()
            proc._mark_scheduled_item_executed('item-42')
        finally:
            _db_mod.get_shared_db_service = original_get

        assert db_called == [], (
            "_mark_scheduled_item_executed must not touch the DB "
            "(it is a no-op placeholder; scheduler_service already fired the row)"
        )

    def test_g4_logs_at_debug_level(self, caplog):
        proc = _make_smp()
        with caplog.at_level(logging.DEBUG, logger='services.scheduled_message_processor'):
            proc._mark_scheduled_item_executed('item-debug-42')

        # At least one debug message mentioning the item_id
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any('item-debug-42' in m for m in debug_msgs)


# ─────────────────────────────────────────────────────────────────────────────
# H. Smoke
# ─────────────────────────────────────────────────────────────────────────────

class TestSmoke:
    """Construct → postTurn() without error (no real DB write)."""

    def test_h1_construct_and_call_postturn_without_error(self, store):
        proc = _make_smp(metadata={'item_id': 'smoke-item'})
        proc._uid = 99

        with patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()):
            proc.postTurn()  # must not raise
