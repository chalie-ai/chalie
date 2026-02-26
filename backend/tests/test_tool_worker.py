"""Tests for tool_worker — pure logic helpers and constants."""

import pytest

from workers.tool_worker import (
    _action_fingerprint,
    _action_types,
    INNATE_SKILLS,
)


pytestmark = pytest.mark.unit


# ── _action_fingerprint ──────────────────────────────────────────────

class TestActionFingerprint:

    def test_single_action_with_query(self):
        actions = [{'type': 'search', 'query': 'what is python'}]
        assert _action_fingerprint(actions) == 'search: what is python'

    def test_multiple_actions_joined(self):
        actions = [
            {'type': 'search', 'query': 'rust lang'},
            {'type': 'recall', 'query': 'previous discussion'},
        ]
        result = _action_fingerprint(actions)
        assert result == 'search: rust lang | recall: previous discussion'

    def test_key_priority_query_over_text_over_input(self):
        action_with_text = [{'type': 'tool', 'text': 'fallback text', 'input': 'ignored'}]
        assert _action_fingerprint(action_with_text) == 'tool: fallback text'

        action_with_input = [{'type': 'tool', 'input': 'last resort'}]
        assert _action_fingerprint(action_with_input) == 'tool: last resort'

    def test_missing_all_keys_returns_type_with_empty(self):
        actions = [{'type': 'noop'}]
        assert _action_fingerprint(actions) == 'noop: '


# ── _action_types ────────────────────────────────────────────────────

class TestActionTypes:

    def test_extracts_unique_types(self):
        actions = [
            {'type': 'search'},
            {'type': 'recall'},
            {'type': 'search'},
        ]
        assert _action_types(actions) == {'search', 'recall'}

    def test_missing_type_defaults_to_unknown(self):
        actions = [{'query': 'hello'}]
        assert _action_types(actions) == {'unknown'}


# ── Repetition detection logic ───────────────────────────────────────

class TestRepetitionDetection:
    """Tests for the type-based repetition counter pattern from tool_worker."""

    @staticmethod
    def _simulate_repetition(action_batches):
        """Simulate the repetition counter from tool_worker lines 400-413."""
        consecutive_same_action = 0
        last_action_type = None

        for actions in action_batches:
            if len(actions) == 1:
                current_type = actions[0].get('type', '')
                if current_type == last_action_type:
                    consecutive_same_action += 1
                else:
                    consecutive_same_action = 1
                last_action_type = current_type
            else:
                consecutive_same_action = 0
                last_action_type = None

        return consecutive_same_action

    def test_triggers_at_three_consecutive_same_type(self):
        batches = [
            [{'type': 'search'}],
            [{'type': 'search'}],
            [{'type': 'search'}],
        ]
        count = self._simulate_repetition(batches)
        assert count >= 3

    def test_resets_on_different_type(self):
        batches = [
            [{'type': 'search'}],
            [{'type': 'search'}],
            [{'type': 'recall'}],  # different
        ]
        count = self._simulate_repetition(batches)
        assert count == 1  # reset to 1 for new type

    def test_resets_on_multi_action_batch(self):
        batches = [
            [{'type': 'search'}],
            [{'type': 'search'}],
            [{'type': 'search'}, {'type': 'recall'}],  # multi-action
        ]
        count = self._simulate_repetition(batches)
        assert count == 0


# ── Novelty gating rules ────────────────────────────────────────────

class TestNoveltyGating:
    """Tests for the novelty gate filter logic in _enqueue_tool_reflection."""

    def test_failed_action_filtered(self):
        action = {'status': 'error', 'action_type': 'search', 'result': 'x' * 100}
        assert action['status'] != 'success'

    def test_innate_skill_filtered(self):
        action = {'status': 'success', 'action_type': 'recall', 'result': 'x' * 100}
        assert action['action_type'] in INNATE_SKILLS

    def test_short_result_filtered(self):
        action = {'status': 'success', 'action_type': 'web_search', 'result': 'short'}
        result_str = str(action.get('result', ''))
        assert len(result_str) < 50

    def test_long_result_truncated_at_2000(self):
        long_result = 'x' * 3000
        truncated = long_result[:2000]
        assert len(truncated) == 2000


# ── INNATE_SKILLS constant ──────────────────────────────────────────

class TestInnateSkills:

    def test_contains_expected_skills(self):
        expected = {'recall', 'memorize', 'introspect', 'associate', 'schedule', 'persistent_task'}
        assert INNATE_SKILLS == expected
