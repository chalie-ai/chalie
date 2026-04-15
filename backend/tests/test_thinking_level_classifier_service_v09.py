"""
Unit tests for ThinkingLevelClassifierService v0.9.0.

Tests the one-hot ordering, sticky fallback, cold-start fallback, direct
label passthrough, and that the raw user turn (not the old Options: prefix)
is passed to predict().
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

logging.disable(logging.CRITICAL)


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_service():
    from services.thinking_level_classifier_service import ThinkingLevelClassifierService
    return ThinkingLevelClassifierService()


def _mock_predict(label, confidence):
    """Return a mock for onnx_inference_service.predict."""
    m = MagicMock(return_value=(label, confidence))
    return m


# ── Test 1: prev_level one-hot ordering ──────────────────────────────────────

@pytest.mark.unit
def test_prev_level_onehot_ordering():
    """Mock predict to capture extra_features; assert one-hot slot fires at index 0/1/2/3."""
    svc = _make_service()

    captured = {}

    def _fake_predict(task_name, prompt, extra_features=None):
        captured["extra_features"] = extra_features
        return ("high", 0.90)

    mock_onnx = MagicMock()
    mock_onnx.predict.side_effect = _fake_predict

    cases = [
        ("none",   0),
        ("low",    1),
        ("medium", 2),
        ("high",   3),
    ]

    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mock_onnx):
        for prev, expected_idx in cases:
            captured.clear()
            svc.classify("test prompt", prev_level=prev)

            ef = captured.get("extra_features")
            assert ef is not None, f"extra_features not passed for prev_level={prev!r}"
            assert ef.shape == (1, 4), f"Expected (1, 4) for prev_level={prev!r}, got {ef.shape}"
            assert ef[0, expected_idx] == 1.0, (
                f"Expected slot {expected_idx} to be 1.0 for prev_level={prev!r}, "
                f"got {ef[0].tolist()}"
            )
            # All other slots must be 0
            for i in range(4):
                if i != expected_idx:
                    assert ef[0, i] == 0.0, (
                        f"Expected slot {i} to be 0.0 for prev_level={prev!r}, "
                        f"got {ef[0, i]}"
                    )


# ── Test 2: sticky fallback inherits prev_level ───────────────────────────────

@pytest.mark.unit
def test_sticky_fallback_inherits_prev_level():
    """Low confidence + prev='high' → result level is 'high', fallback=True."""
    svc = _make_service()

    mock_onnx = MagicMock()
    mock_onnx.predict.return_value = ("low", 0.50)  # below 0.70 threshold

    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mock_onnx):
        result = svc.classify("sure, whatever", prev_level="high")

    assert result["level"] == "high", f"Expected 'high', got {result['level']!r}"
    assert result["fallback"] is True
    assert result["confidence"] == 0.0


# ── Test 3: cold start fallback to medium ─────────────────────────────────────

@pytest.mark.unit
def test_cold_start_fallback_to_medium():
    """Low confidence + prev_level='none' → returns 'medium', fallback=True."""
    svc = _make_service()

    mock_onnx = MagicMock()
    mock_onnx.predict.return_value = (None, 0.0)  # unavailable

    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mock_onnx):
        result = svc.classify("ok go on", prev_level="none")

    assert result["level"] == "medium", f"Expected 'medium', got {result['level']!r}"
    assert result["fallback"] is True


# ── Test 4: direct label passthrough ─────────────────────────────────────────

@pytest.mark.unit
def test_direct_label_passthrough():
    """High-confidence 'high' returned as 'high' (not a letter like 'C')."""
    svc = _make_service()

    mock_onnx = MagicMock()
    mock_onnx.predict.return_value = ("high", 0.92)

    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mock_onnx):
        result = svc.classify("design an auth system for a multi-tenant app")

    assert result["level"] == "high", f"Expected 'high', got {result['level']!r}"
    assert result["fallback"] is False
    assert result["confidence"] == pytest.approx(0.92)


# ── Test 5: raw user turn passed to predict, no Options: substring ────────────

@pytest.mark.unit
def test_no_options_suffix_in_input():
    """The raw user turn text is what gets passed to predict; no 'Options:' anywhere."""
    svc = _make_service()

    captured_prompt = {}

    def _fake_predict(task_name, prompt, extra_features=None):
        captured_prompt["prompt"] = prompt
        return ("medium", 0.85)

    mock_onnx = MagicMock()
    mock_onnx.predict.side_effect = _fake_predict

    raw_turn = "What time does the meeting start?"

    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mock_onnx):
        svc.classify(raw_turn, prev_level="none")

    prompt_sent = captured_prompt.get("prompt", "")
    assert "Options:" not in prompt_sent, (
        f"'Options:' found in prompt sent to predict: {prompt_sent!r}"
    )
    assert "Answer:" not in prompt_sent, (
        f"'Answer:' found in prompt sent to predict: {prompt_sent!r}"
    )
    assert prompt_sent == raw_turn, (
        f"Expected raw turn to be passed as-is. Got: {prompt_sent!r}"
    )
