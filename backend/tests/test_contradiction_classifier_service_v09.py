"""
Unit tests for ContradictionClassifierService v0.9.0.

Tests that _build_onnx_input produces JSON-only output (no Options/Answer suffix)
and that the MLP path passes direct label strings through (not A/B/C/D/E letters).
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

logging.disable(logging.CRITICAL)

_VALID_LABELS = frozenset([
    "temporal_change",
    "true_contradiction",
    "context_dependent",
    "figurative",
    "compatible",
])


def _make_service():
    from services.contradiction_classifier_service import ContradictionClassifierService
    return ContradictionClassifierService()


# ── Test 1: _build_onnx_input is JSON only ────────────────────────────────────

@pytest.mark.unit
def test_build_onnx_input_is_json_only():
    """Output must be a valid JSON object with no 'Options:' or 'Answer:' substring."""
    svc = _make_service()

    output = svc._build_onnx_input(
        text_a="I work at Google",
        text_b="I work at Apple",
        meta_a={"source": "incoming"},
        meta_b={"source": "trait", "reinforcement_count": 5},
    )

    # Must not contain the retired suffix
    assert "Options:" not in output, f"'Options:' found in output: {output!r}"
    assert "Answer:" not in output, f"'Answer:' found in output: {output!r}"

    # Must be valid JSON
    parsed = json.loads(output)

    # Must contain exactly the expected fields
    expected_fields = {
        "text_a", "text_b", "type_a", "type_b",
        "age_a_days", "age_b_days", "established_a", "established_b",
    }
    assert set(parsed.keys()) == expected_fields, (
        f"Unexpected fields. Got: {set(parsed.keys())}, expected: {expected_fields}"
    )

    assert parsed["text_a"] == "I work at Google"
    assert parsed["text_b"] == "I work at Apple"


# ── Test 2: direct label output ───────────────────────────────────────────────

@pytest.mark.unit
def test_direct_label_output():
    """Mocked predict returning 'temporal_change' → classification string is 'temporal_change', not 'A'."""
    svc = _make_service()

    mock_onnx = MagicMock()
    mock_onnx.predict.return_value = ("temporal_change", 0.90)

    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mock_onnx):
        result = svc._classify_pair_onnx(
            text_a="I worked at Google",
            text_b="I work at Apple",
            meta_a={"source": "incoming"},
            meta_b={"source": "trait", "reinforcement_count": 5},
        )

    assert result is not None, "Expected a result dict, got None"
    classification = result.get("classification")
    assert classification == "temporal_change", (
        f"Expected 'temporal_change', got {classification!r}"
    )
    assert classification in _VALID_LABELS, (
        f"Classification {classification!r} is not a valid label (got a letter?)"
    )

    # Ensure we are NOT returning a letter
    assert classification not in ("A", "B", "C", "D", "E"), (
        f"Got a letter label instead of class name: {classification!r}"
    )


# ── Test 3: all valid labels pass through correctly ───────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("label", sorted(_VALID_LABELS))
def test_all_valid_labels_pass_through(label):
    """Each direct label string is returned as-is from _classify_pair_onnx."""
    svc = _make_service()

    mock_onnx = MagicMock()
    mock_onnx.predict.return_value = (label, 0.85)

    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mock_onnx):
        result = svc._classify_pair_onnx(
            text_a="memory a",
            text_b="memory b",
            meta_a={"source": "incoming"},
            meta_b={"source": "trait"},
        )

    assert result is not None, f"Expected result for label={label!r}"
    assert result["classification"] == label, (
        f"Expected classification={label!r}, got {result['classification']!r}"
    )
