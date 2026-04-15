"""
Unit tests for ThinkingLevelClassifierService.

Covers the v0.9.0 MLP-head architecture:
  - High-confidence predictions return the correct class
  - Low-confidence with prev_level → sticky fallback
  - Low-confidence with prev_level='none' → default 'medium'
  - ONNX service raising → medium fallback without re-raising
  - Invalid prev_level strings normalised to 'none'
  - Missing model: predict returns (None, 0.0) → correct fallback
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.thinking_level_classifier_service import ThinkingLevelClassifierService

pytestmark = pytest.mark.unit


# ── classify() — high-confidence paths ───────────────────────────────────────

class TestClassifyHighConfidence:
    def _mock_svc(self, label, confidence):
        mock = MagicMock()
        mock.predict.return_value = (label, confidence)
        return mock

    def test_label_low_returns_low(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('low', 0.95)):
            result = svc.classify("what's the capital of France?", prev_level='none')
        assert result['level'] == 'low'
        assert result['confidence'] == 0.95
        assert result['fallback'] is False

    def test_label_medium_returns_medium(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('medium', 0.88)):
            result = svc.classify("compare postgres vs mysql", prev_level='none')
        assert result['level'] == 'medium'
        assert result['confidence'] == 0.88
        assert result['fallback'] is False

    def test_label_high_returns_high(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('high', 0.91)):
            result = svc.classify("design a multi-tenant auth system", prev_level='none')
        assert result['level'] == 'high'
        assert result['confidence'] == 0.91
        assert result['fallback'] is False

    def test_exactly_at_threshold_passes(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('medium', 0.70)):
            result = svc.classify("summarize this", prev_level='none')
        assert result['fallback'] is False


# ── classify() — low-confidence sticky fallback ───────────────────────────────

class TestClassifyLowConfidence:
    def _mock_svc(self, label, confidence):
        mock = MagicMock()
        mock.predict.return_value = (label, confidence)
        return mock

    def test_low_confidence_with_prev_returns_prev(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('low', 0.50)):
            result = svc.classify("ok", prev_level='high')
        assert result['level'] == 'high'
        assert result['fallback'] is True

    def test_low_confidence_with_prev_medium_returns_medium(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('medium', 0.40)):
            result = svc.classify("sure", prev_level='medium')
        assert result['level'] == 'medium'
        assert result['fallback'] is True

    def test_low_confidence_with_prev_low_returns_low(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('high', 0.60)):
            result = svc.classify("thanks", prev_level='low')
        assert result['level'] == 'low'
        assert result['fallback'] is True

    def test_low_confidence_prev_none_defaults_to_medium(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('low', 0.55)):
            result = svc.classify("ok", prev_level='none')
        assert result['level'] == 'medium'
        assert result['fallback'] is True

    def test_null_label_triggers_fallback(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc(None, 0.0)):
            result = svc.classify("hmm", prev_level='high')
        assert result['level'] == 'high'
        assert result['fallback'] is True

    def test_just_below_threshold_triggers_fallback(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('high', 0.699)):
            result = svc.classify("go on", prev_level='medium')
        assert result['fallback'] is True
        assert result['level'] == 'medium'


# ── classify() — exception path ───────────────────────────────────────────────

class TestClassifyException:
    def test_onnx_service_raises_returns_medium_no_prev(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   side_effect=RuntimeError("model not loaded")):
            result = svc.classify("hello", prev_level='none')
        assert result['level'] == 'medium'
        assert result['fallback'] is True
        assert result['confidence'] == 0.0

    def test_onnx_service_raises_sticky_fallback_with_prev(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   side_effect=RuntimeError("model not loaded")):
            result = svc.classify("yes", prev_level='high')
        assert result['level'] == 'high'
        assert result['fallback'] is True

    def test_does_not_raise_on_exception(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   side_effect=Exception("boom")):
            result = svc.classify("test", prev_level='none')
        assert isinstance(result, dict)
        assert 'level' in result


# ── Invalid prev_level normalisation ──────────────────────────────────────────

class TestPrevLevelValidation:
    def _mock_svc_high_confidence(self):
        mock = MagicMock()
        mock.predict.return_value = ('medium', 0.90)
        return mock

    def test_invalid_prev_level_normalised_to_none(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc_high_confidence()):
            result = svc.classify("test", prev_level='invalid_value')
        assert result['level'] in ('low', 'medium', 'high')

    def test_empty_prev_level_normalised_to_none(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc_high_confidence()):
            result = svc.classify("test", prev_level='')
        assert result['level'] in ('low', 'medium', 'high')

    def test_invalid_prev_low_confidence_falls_back_to_medium(self):
        mock = MagicMock()
        mock.predict.return_value = ('low', 0.40)
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=mock):
            result = svc.classify("test", prev_level='garbage')
        # normalised to 'none' → low confidence → 'medium'
        assert result['level'] == 'medium'
        assert result['fallback'] is True


# ── Missing model fallback ─────────────────────────────────────────────────────

class TestMissingModelFallback:
    """Simulate a deployment where the thinking_level head is not yet installed."""

    def _mock_svc_unloaded(self):
        mock = MagicMock()
        mock.predict.return_value = (None, 0.0)
        return mock

    def test_missing_model_no_prev_returns_medium(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc_unloaded()):
            result = svc.classify("help me plan my week", prev_level='none')
        assert result['level'] == 'medium'
        assert result['fallback'] is True

    def test_missing_model_with_prev_high_sticks_to_high(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc_unloaded()):
            result = svc.classify("go on", prev_level='high')
        assert result['level'] == 'high'
        assert result['fallback'] is True

    def test_missing_model_with_prev_low_sticks_to_low(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc_unloaded()):
            result = svc.classify("thanks", prev_level='low')
        assert result['level'] == 'low'
        assert result['fallback'] is True

    def test_missing_model_does_not_raise(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc_unloaded()):
            result = svc.classify("anything", prev_level='none')
        assert isinstance(result, dict)
        assert 'level' in result
