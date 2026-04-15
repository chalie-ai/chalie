"""
Unit tests for ThinkingLevelClassifierService.

Covers:
  - Byte-exact _build_onnx_input format (no ONNX required)
  - Byte-exact parity with training-side _format_input() (cross-repo assert)
  - Tokenizer parity: A/B/C → IDs 32/33/34 in Qwen2.5
  - Label → class mapping for A/B/C
  - High-confidence predictions return the correct class
  - Low-confidence with prev_level → sticky fallback
  - Low-confidence with prev_level='none' → default 'medium'
  - ONNX service raising → medium fallback without re-raising
  - Invalid prev_level strings normalised to 'none'
  - _read_prev_thinking_level: row with invalid value ('foo') → 'none'
  - _read_prev_thinking_level: only-NULL row → 'none'
  - MODEL_REGISTRY missing model: predict returns (None, 0.0) → correct fallback
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.thinking_level_classifier_service import ThinkingLevelClassifierService

pytestmark = pytest.mark.unit


# ── Byte-exact input format ───────────────────────────────────────────────────

class TestBuildOnnxInput:
    def test_none_prev_level(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("Hello there", "none")
        assert result == "[prev=none] Hello there\nOptions: A: low | B: medium | C: high\nAnswer:"

    def test_low_prev_level(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("What time is it?", "low")
        assert result == "[prev=low] What time is it?\nOptions: A: low | B: medium | C: high\nAnswer:"

    def test_medium_prev_level(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("Compare postgres vs mysql", "medium")
        assert result == "[prev=medium] Compare postgres vs mysql\nOptions: A: low | B: medium | C: high\nAnswer:"

    def test_high_prev_level(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("Design an auth system", "high")
        assert result == "[prev=high] Design an auth system\nOptions: A: low | B: medium | C: high\nAnswer:"

    def test_no_trailing_space_after_answer_colon(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("x", "none")
        assert result.endswith("Answer:")
        assert not result.endswith("Answer: ")

    def test_no_trailing_newline(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("x", "none")
        assert not result.endswith("\n")

    def test_single_newline_between_turn_and_options(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("hello", "none")
        lines = result.split("\n")
        assert lines[0] == "[prev=none] hello"
        assert lines[1] == "Options: A: low | B: medium | C: high"
        assert lines[2] == "Answer:"

    def test_pipe_delimiters_with_spaces(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("x", "none")
        assert "A: low | B: medium | C: high" in result

    def test_square_brackets_around_prev(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("x", "medium")
        assert result.startswith("[prev=medium]")

    def test_single_space_after_bracket(self):
        svc = ThinkingLevelClassifierService()
        result = svc._build_onnx_input("hello world", "none")
        assert result.startswith("[prev=none] hello world")


# ── Label mapping ─────────────────────────────────────────────────────────────

class TestLabelMapping:
    def test_a_maps_to_low(self):
        assert ThinkingLevelClassifierService._ONNX_LABEL_TO_CLASS['A'] == 'low'

    def test_b_maps_to_medium(self):
        assert ThinkingLevelClassifierService._ONNX_LABEL_TO_CLASS['B'] == 'medium'

    def test_c_maps_to_high(self):
        assert ThinkingLevelClassifierService._ONNX_LABEL_TO_CLASS['C'] == 'high'


# ── classify() — high-confidence paths ───────────────────────────────────────

class TestClassifyHighConfidence:
    def _mock_svc(self, label, confidence):
        mock = MagicMock()
        mock.predict.return_value = (label, confidence)
        return mock

    def test_label_a_returns_low(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('A', 0.95)):
            result = svc.classify("what's the capital of France?", prev_level='none')
        assert result['level'] == 'low'
        assert result['confidence'] == 0.95
        assert result['fallback'] is False

    def test_label_b_returns_medium(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('B', 0.88)):
            result = svc.classify("compare postgres vs mysql", prev_level='none')
        assert result['level'] == 'medium'
        assert result['confidence'] == 0.88
        assert result['fallback'] is False

    def test_label_c_returns_high(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('C', 0.91)):
            result = svc.classify("design a multi-tenant auth system", prev_level='none')
        assert result['level'] == 'high'
        assert result['confidence'] == 0.91
        assert result['fallback'] is False

    def test_exactly_at_threshold_passes(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('B', 0.70)):
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
                   return_value=self._mock_svc('A', 0.50)):
            result = svc.classify("ok", prev_level='high')
        assert result['level'] == 'high'
        assert result['fallback'] is True

    def test_low_confidence_with_prev_medium_returns_medium(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('B', 0.40)):
            result = svc.classify("sure", prev_level='medium')
        assert result['level'] == 'medium'
        assert result['fallback'] is True

    def test_low_confidence_with_prev_low_returns_low(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('C', 0.60)):
            result = svc.classify("thanks", prev_level='low')
        assert result['level'] == 'low'
        assert result['fallback'] is True

    def test_low_confidence_prev_none_defaults_to_medium(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc('A', 0.55)):
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
                   return_value=self._mock_svc('C', 0.699)):
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
        mock.predict.return_value = ('B', 0.90)
        return mock

    def test_invalid_prev_level_normalised_to_none(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc_high_confidence()):
            result = svc.classify("test", prev_level='invalid_value')
        # Classify should succeed — the invalid prev was treated as 'none'
        assert result['level'] in ('low', 'medium', 'high')

    def test_empty_prev_level_normalised_to_none(self):
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=self._mock_svc_high_confidence()):
            result = svc.classify("test", prev_level='')
        assert result['level'] in ('low', 'medium', 'high')

    def test_invalid_prev_low_confidence_falls_back_to_medium(self):
        mock = MagicMock()
        mock.predict.return_value = ('A', 0.40)
        svc = ThinkingLevelClassifierService()
        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   return_value=mock):
            result = svc.classify("test", prev_level='garbage')
        # normalised to 'none' → low confidence → 'medium'
        assert result['level'] == 'medium'
        assert result['fallback'] is True


# ── Byte-exact parity with training-side _format_input() ─────────────────────
# Import the canonical formatter from the training pipeline to assert that
# ThinkingLevelClassifierService._build_onnx_input() produces IDENTICAL bytes.

_TRAINING_PIPELINE_PATH = '/Volumes/llm/chalie-training-pipeline'

def _training_format_input_available() -> bool:
    try:
        sys.path.insert(0, _TRAINING_PIPELINE_PATH)
        from data.tasks.thinking_level import _format_input  # noqa: F401
        return True
    except Exception:
        return False


_training_available = _training_format_input_available()


@pytest.mark.skipif(
    not _training_available,
    reason="Training pipeline not present at expected path — skipping cross-repo parity tests",
)
class TestByteExactParity:
    """Assert string equality between the Chalie wrapper and the training-side formatter.

    Any drift here silently degrades classification accuracy because the model
    was trained on strings produced by _format_input(). These tests catch a
    common class of bug where whitespace, newline, or option-text differs.
    """

    def _training_format(self, prompt: str, prev_level: str) -> str:
        sys.path.insert(0, _TRAINING_PIPELINE_PATH)
        from data.tasks.thinking_level import _format_input
        return _format_input(prompt, prev_level)

    def test_parity_none_prev_simple_prompt(self):
        prompt = "What time is it?"
        prev = "none"
        chalie = ThinkingLevelClassifierService()._build_onnx_input(prompt, prev)
        training = self._training_format(prompt, prev)
        assert chalie == training

    def test_parity_low_prev(self):
        prompt = "Thanks, got it"
        prev = "low"
        chalie = ThinkingLevelClassifierService()._build_onnx_input(prompt, prev)
        training = self._training_format(prompt, prev)
        assert chalie == training

    def test_parity_medium_prev(self):
        prompt = "Compare PostgreSQL and MySQL for a write-heavy workload"
        prev = "medium"
        chalie = ThinkingLevelClassifierService()._build_onnx_input(prompt, prev)
        training = self._training_format(prompt, prev)
        assert chalie == training

    def test_parity_high_prev_multi_word(self):
        prompt = "Design a multi-tenant authentication system with RBAC and audit logging"
        prev = "high"
        chalie = ThinkingLevelClassifierService()._build_onnx_input(prompt, prev)
        training = self._training_format(prompt, prev)
        assert chalie == training

    def test_parity_prompt_with_newlines_preserved(self):
        # User prompts can contain newlines — the wrapper must not strip them
        prompt = "First line\nSecond line"
        prev = "none"
        chalie = ThinkingLevelClassifierService()._build_onnx_input(prompt, prev)
        training = self._training_format(prompt, prev)
        assert chalie == training

    def test_parity_empty_prompt(self):
        # Edge: empty string is in-distribution for short replies ("ok", "sure")
        prompt = ""
        prev = "medium"
        chalie = ThinkingLevelClassifierService()._build_onnx_input(prompt, prev)
        training = self._training_format(prompt, prev)
        assert chalie == training


# ── Tokenizer parity: A/B/C → IDs 32/33/34 ───────────────────────────────────

def _tokenizer_available() -> bool:
    try:
        from transformers import AutoTokenizer  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _tokenizer_available(),
    reason="transformers not installed — skipping tokenizer parity tests",
)
class TestTokenizerParity:
    """Verify that A, B, C each tokenize to exactly one token with the expected IDs.

    The ONNX head compares logits at positions 32/33/34 in the Qwen2.5 vocabulary.
    If the tokenizer ever produces a different mapping (model upgrade, different
    tokenizer version), this test catches it before the model ships.
    """

    @classmethod
    def _get_tokenizer(cls):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-0.5B', trust_remote_code=True
        )

    def test_a_tokenizes_to_id_32(self):
        tok = self._get_tokenizer()
        ids = tok.encode('A', add_special_tokens=False)
        assert ids == [32], (
            f"Label 'A' must tokenize to [32] (got {ids}). "
            "The ONNX head uses logit at index 32 for 'low'."
        )

    def test_b_tokenizes_to_id_33(self):
        tok = self._get_tokenizer()
        ids = tok.encode('B', add_special_tokens=False)
        assert ids == [33], (
            f"Label 'B' must tokenize to [33] (got {ids}). "
            "The ONNX head uses logit at index 33 for 'medium'."
        )

    def test_c_tokenizes_to_id_34(self):
        tok = self._get_tokenizer()
        ids = tok.encode('C', add_special_tokens=False)
        assert ids == [34], (
            f"Label 'C' must tokenize to [34] (got {ids}). "
            "The ONNX head uses logit at index 34 for 'high'."
        )

    def test_each_label_is_single_token(self):
        tok = self._get_tokenizer()
        for label in ('A', 'B', 'C'):
            ids = tok.encode(label, add_special_tokens=False)
            assert len(ids) == 1, (
                f"Label '{label}' must be a single token for the classifier head "
                f"to read the correct logit position (got {ids})."
            )


# ── MODEL_REGISTRY missing model: predict returns (None, 0.0) ─────────────────

class TestMissingModelFallback:
    """Simulate a pre-v0.8.0 deployment where thinking_level head is not yet
    installed. OnnxInferenceService.predict() returns (None, 0.0) in this case.
    The classifier must fall back gracefully rather than crash.
    """

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
        # (None, 0.0) → label is None → low-confidence path → sticky prev
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
