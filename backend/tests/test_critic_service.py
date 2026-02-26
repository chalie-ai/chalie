"""Tests for CriticService — post-action verification for the ACT loop."""

import json
import pytest
from unittest.mock import patch, MagicMock

from services.critic_service import (
    CriticService,
    SAFE_ACTIONS,
    READ_ACTIONS,
    DEFAULT_CONFIDENCE_SKIP_THRESHOLD,
    CALIBRATION_EMA_ALPHA,
    CALIBRATION_CORRECTION_RATE_LIMIT,
    CRITIC_FATIGUE_COST,
    MAX_CRITIC_RETRIES,
)


pytestmark = pytest.mark.unit


class TestCriticShouldSkip:
    """Tests for CriticService.should_skip()."""

    def test_failed_status_skips(self):
        """Actions with non-success status should always be skipped."""
        svc = CriticService()
        result = {'status': 'error', 'confidence': 0.99}
        assert svc.should_skip('recall', result) is True

    def test_high_confidence_read_action_skips(self):
        """Read actions (recall, introspect) with high confidence should be skipped."""
        svc = CriticService()
        for action in READ_ACTIONS:
            result = {'status': 'success', 'confidence': 0.95}
            assert svc.should_skip(action, result) is True

    def test_low_confidence_does_not_skip(self):
        """Actions below the confidence threshold should not be skipped."""
        svc = CriticService()
        result = {'status': 'success', 'confidence': 0.3}
        assert svc.should_skip('recall', result) is False

    def test_high_confidence_non_read_action_skips(self):
        """Non-read actions with confidence >= threshold should also be skipped."""
        svc = CriticService()
        result = {'status': 'success', 'confidence': 0.95}
        assert svc.should_skip('schedule', result) is True

    def test_missing_confidence_does_not_skip(self):
        """Missing confidence field defaults to 0.0, which should not skip."""
        svc = CriticService()
        result = {'status': 'success'}
        assert svc.should_skip('recall', result) is False

    def test_skip_increments_skipped_counter(self):
        """Each skip should increment the skipped telemetry counter."""
        svc = CriticService()
        result = {'status': 'success', 'confidence': 0.95}
        svc.should_skip('recall', result)
        svc.should_skip('introspect', result)
        assert svc.skipped == 2


class TestCriticEvaluate:
    """Tests for CriticService.evaluate()."""

    def test_evaluate_parses_json_verdict(self, mock_llm):
        """Clean JSON response should be parsed into a verdict dict."""
        mock_llm.response_text = json.dumps({
            'verified': True,
            'severity': None,
            'issue': None,
        })
        svc = CriticService()
        verdict = svc.evaluate(
            original_request='What time is it?',
            action_type='recall',
            action_intent={'type': 'recall', 'query': 'time'},
            action_result={'status': 'success', 'data': '3pm'},
        )
        assert verdict['verified'] is True

    def test_evaluate_parses_markdown_code_block(self, mock_llm):
        """JSON wrapped in markdown code fences should be parsed correctly."""
        inner = json.dumps({'verified': False, 'severity': 'minor', 'issue': 'wrong date'})
        mock_llm.response_text = f'```json\n{inner}\n```'
        svc = CriticService()
        verdict = svc.evaluate(
            original_request='Schedule meeting',
            action_type='schedule',
            action_intent={'type': 'schedule'},
            action_result={'status': 'success'},
        )
        assert verdict['verified'] is False
        assert verdict['severity'] == 'minor'

    def test_evaluate_text_with_error_words_returns_unverified(self, mock_llm):
        """Plain text containing error words should produce verified=False."""
        mock_llm.response_text = 'The result is incorrect because the date is wrong.'
        svc = CriticService()
        verdict = svc.evaluate(
            original_request='Check schedule',
            action_type='recall',
            action_intent={'type': 'recall'},
            action_result={'status': 'success'},
        )
        assert verdict['verified'] is False
        assert verdict['severity'] == 'minor'

    def test_evaluate_plain_text_no_error_words_returns_verified(self, mock_llm):
        """Plain text without error indicators should default to verified=True."""
        mock_llm.response_text = 'Looks good to me, all checks pass.'
        svc = CriticService()
        verdict = svc.evaluate(
            original_request='Check recall',
            action_type='recall',
            action_intent={'type': 'recall'},
            action_result={'status': 'success'},
        )
        assert verdict['verified'] is True

    def test_evaluate_increments_total_evaluations(self, mock_llm):
        """Each evaluate call should increment total_evaluations counter."""
        mock_llm.response_text = '{"verified": true}'
        svc = CriticService()
        svc.evaluate('req', 'recall', {}, {})
        svc.evaluate('req', 'recall', {}, {})
        assert svc.total_evaluations == 2

    def test_evaluate_correction_increments_corrections_counter(self, mock_llm):
        """A verdict with correction should increment corrections counter."""
        mock_llm.response_text = json.dumps({
            'verified': False,
            'severity': 'minor',
            'issue': 'wrong value',
            'correction': 'use 42 instead',
        })
        svc = CriticService()
        svc.evaluate('req', 'recall', {}, {})
        assert svc.corrections == 1

    def test_evaluate_escalation_increments_escalations_counter(self, mock_llm):
        """A verdict without correction (but unverified) should increment escalations."""
        mock_llm.response_text = json.dumps({
            'verified': False,
            'severity': 'major',
            'issue': 'completely wrong',
            'correction': None,
        })
        svc = CriticService()
        svc.evaluate('req', 'schedule', {}, {})
        assert svc.escalations == 1

    def test_evaluate_llm_failure_returns_verified_true(self, mock_llm):
        """When LLM call fails, evaluate should default to verified=True."""
        mock_llm.send_message.side_effect = Exception('LLM timeout')
        svc = CriticService()
        verdict = svc.evaluate('req', 'recall', {}, {})
        assert verdict['verified'] is True


class TestCriticIsSafeAction:
    """Tests for CriticService.is_safe_action()."""

    def test_recall_is_safe(self):
        """recall should be considered a safe action."""
        svc = CriticService()
        assert svc.is_safe_action('recall') is True

    def test_memorize_is_safe(self):
        """memorize should be considered a safe action."""
        svc = CriticService()
        assert svc.is_safe_action('memorize') is True

    def test_schedule_is_consequential(self):
        """schedule should NOT be a safe action (consequential)."""
        svc = CriticService()
        assert svc.is_safe_action('schedule') is False

    def test_list_is_consequential(self):
        """list should NOT be a safe action."""
        svc = CriticService()
        assert svc.is_safe_action('list') is False

    def test_all_safe_actions_match_module_constant(self):
        """Every item in SAFE_ACTIONS should return True from is_safe_action."""
        svc = CriticService()
        for action in SAFE_ACTIONS:
            assert svc.is_safe_action(action) is True


class TestCriticTelemetry:
    """Tests for telemetry reporting and calibration."""

    def test_get_telemetry_returns_all_keys(self):
        """get_telemetry must return the complete set of telemetry fields."""
        svc = CriticService()
        telemetry = svc.get_telemetry()
        expected_keys = {
            'critic_total_checks',
            'critic_evaluations',
            'critic_skipped',
            'critic_corrections',
            'critic_escalations',
            'critic_oscillation_events',
            'critic_correction_rate',
            'critic_severity_distribution',
            'critic_calibration',
        }
        assert expected_keys == set(telemetry.keys())

    def test_correction_rate_zero_with_no_evaluations(self):
        """Correction rate should be 0.0 when no evaluations have occurred."""
        svc = CriticService()
        assert svc.get_telemetry()['critic_correction_rate'] == 0.0

    def test_calibration_raises_threshold_when_correction_rate_high(self, mock_llm):
        """High correction rate should raise the effective skip threshold."""
        svc = CriticService()
        # Manually set high correction rate for 'recall'
        svc._calibration['recall'] = 0.3  # well above CALIBRATION_CORRECTION_RATE_LIMIT

        threshold = svc._get_calibrated_threshold('recall')
        assert threshold > DEFAULT_CONFIDENCE_SKIP_THRESHOLD

    def test_calibration_uses_default_threshold_when_rate_low(self):
        """Low correction rate should use the default skip threshold."""
        svc = CriticService()
        svc._calibration['recall'] = 0.01
        threshold = svc._get_calibrated_threshold('recall')
        assert threshold == DEFAULT_CONFIDENCE_SKIP_THRESHOLD

    def test_calibration_threshold_capped_at_one(self):
        """Calibrated threshold should never exceed 1.0."""
        svc = CriticService()
        svc._calibration['recall'] = 0.99  # Extreme correction rate
        threshold = svc._get_calibrated_threshold('recall')
        assert threshold <= 1.0

    def test_format_correction_entry_contains_all_parts(self):
        """format_correction_entry should include action type, original, and correction."""
        svc = CriticService()
        entry = svc.format_correction_entry(
            action_type='recall',
            original_result='wrong answer',
            correction='use correct query',
            final_result='right answer',
        )
        assert 'recall' in entry
        assert 'wrong answer' in entry
        assert 'use correct query' in entry
        assert 'right answer' in entry
        assert '[CRITIC CORRECTION]' in entry
