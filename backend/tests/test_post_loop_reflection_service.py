"""Tests for PostLoopReflectionService."""

import json
import pytest
from unittest.mock import patch, MagicMock

from services.post_loop_reflection_service import PostLoopReflectionService


pytestmark = pytest.mark.unit


class TestPostLoopReflectionServiceReflect:

    def test_returns_empty_dict_on_llm_failure(self):
        """reflect() returns {} when LLM call raises."""
        svc = PostLoopReflectionService()
        with patch.object(svc, '_get_llm', side_effect=Exception("LLM down")):
            result = svc.reflect(
                exchange_text='test goal',
                act_history_text='recall -> success',
                termination_reason='no_actions',
                existing_goal_guidance='',
                loop_id='test-id',
                iterations_used=3,
            )
        assert result == {}

    def test_returns_parsed_reflection_on_success(self):
        """reflect() returns parsed dict when LLM returns valid JSON."""
        svc = PostLoopReflectionService()
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(
            text=json.dumps({
                'outcome_quality': 0.8,
                'what_worked': 'recall found relevant data',
                'what_failed': '',
                'lesson': 'check calendar before scheduling',
                'confidence': 0.9,
                'goal_guidance': 'For research goals, start with recall then memorize findings.',
            })
        )
        with patch.object(svc, '_get_llm', return_value=mock_llm):
            result = svc.reflect(
                exchange_text='research best practices',
                act_history_text='recall -> success: found data',
                termination_reason='no_actions',
                existing_goal_guidance='',
                loop_id='test-id',
                iterations_used=3,
            )
        assert result['outcome_quality'] == 0.8
        assert result['goal_guidance'].startswith('For research goals')
        assert result['confidence'] == 0.9

    def test_clamps_outcome_quality_and_confidence(self):
        """Values outside [0, 1] are clamped."""
        svc = PostLoopReflectionService()
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(
            text=json.dumps({
                'outcome_quality': 1.5,
                'what_worked': '',
                'what_failed': '',
                'lesson': None,
                'confidence': -0.3,
                'goal_guidance': 'test',
            })
        )
        with patch.object(svc, '_get_llm', return_value=mock_llm):
            result = svc.reflect(
                exchange_text='test', act_history_text='', termination_reason='',
                existing_goal_guidance='', loop_id='x', iterations_used=1,
            )
        assert result['outcome_quality'] == 1.0
        assert result['confidence'] == 0.0

    def test_handles_json_in_code_fence(self):
        """reflect() parses JSON wrapped in markdown code fence."""
        svc = PostLoopReflectionService()
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(
            text='Here is the reflection:\n```json\n{"outcome_quality": 0.6, "what_worked": "ok", "what_failed": "", "lesson": null, "confidence": 0.7, "goal_guidance": "test guidance"}\n```'
        )
        with patch.object(svc, '_get_llm', return_value=mock_llm):
            result = svc.reflect(
                exchange_text='test', act_history_text='', termination_reason='',
                existing_goal_guidance='', loop_id='x', iterations_used=1,
            )
        assert result['outcome_quality'] == 0.6
        assert result['goal_guidance'] == 'test guidance'

    def test_returns_empty_dict_on_unparseable_response(self):
        """reflect() returns {} when LLM returns gibberish."""
        svc = PostLoopReflectionService()
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='not json at all')
        with patch.object(svc, '_get_llm', return_value=mock_llm):
            result = svc.reflect(
                exchange_text='test', act_history_text='', termination_reason='',
                existing_goal_guidance='', loop_id='x', iterations_used=1,
            )
        assert result == {}

    def test_goal_guidance_field_present(self):
        """Parsed result always has goal_guidance key."""
        svc = PostLoopReflectionService()
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(
            text=json.dumps({
                'outcome_quality': 0.7,
                'what_worked': 'x',
                'what_failed': '',
                'lesson': 'do y',
                'confidence': 0.8,
                'goal_guidance': 'For this type of goal, approach Z works best.',
            })
        )
        with patch.object(svc, '_get_llm', return_value=mock_llm):
            result = svc.reflect(
                exchange_text='test', act_history_text='',
                termination_reason='', existing_goal_guidance='prior guidance',
                loop_id='x', iterations_used=2,
            )
        assert 'goal_guidance' in result
        assert result['goal_guidance'] != ''


class TestPostLoopReflectionServiceBuildPrompt:

    def test_injects_existing_guidance(self):
        """existing_goal_guidance appears verbatim in the built prompt."""
        svc = PostLoopReflectionService()
        svc._prompt_template = 'Goal: {{original_goal}} Guidance: {{existing_goal_guidance}} Iter: {{iterations}}'
        prompt = svc._build_prompt(
            exchange_text='research docker',
            act_history_text='',
            termination_reason='no_actions',
            existing_goal_guidance='Previously, use recall first.',
            iterations_used=3,
        )
        assert 'Previously, use recall first.' in prompt
        assert '3' in prompt

    def test_empty_guidance_replaced_with_placeholder(self):
        """When no existing guidance, a placeholder is injected."""
        svc = PostLoopReflectionService()
        svc._prompt_template = 'Guidance: {{existing_goal_guidance}}'
        prompt = svc._build_prompt(
            exchange_text='test',
            act_history_text='',
            termination_reason='',
            existing_goal_guidance='',
        )
        assert '(none' in prompt

    def test_exchange_text_truncated_to_500(self):
        """exchange_text is truncated to 500 chars in the prompt."""
        svc = PostLoopReflectionService()
        svc._prompt_template = 'Goal: {{original_goal}}'
        long_text = 'x' * 600
        prompt = svc._build_prompt(
            exchange_text=long_text,
            act_history_text='',
            termination_reason='',
            existing_goal_guidance='',
        )
        assert len(prompt) <= len('Goal: ') + 500 + 10  # allow small buffer
