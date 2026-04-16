"""
Pure-function unit tests for the classifier stack.

Only covers functions that are:
  - Deterministic
  - Have no IO, no collaborators, no state

Tests that would require mocks belong in the integration suite
(test_classifier_features.py) or in nightly scenarios.
"""

import struct

import numpy as np
import pytest

from services.contradiction_classifier_service import (
    _cosine_similarity,
    _unpack_embedding,
    _is_established,
)
from services.onnx_inference_service import _softmax, _ClassifierHead

pytestmark = pytest.mark.unit


# ── _cosine_similarity ────────────────────────────────────────────────────────


class TestCosimeSimilarity:
    def test_identical_vectors_return_one(self):
        assert abs(_cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_empty_vectors_return_zero(self):
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths_return_zero(self):
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_opposite_vectors_return_minus_one(self):
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-6


# ── _unpack_embedding ─────────────────────────────────────────────────────────


class TestUnpackEmbedding:
    def _pack(self, values: list) -> bytes:
        return struct.pack(f"{len(values)}f", *values)

    def test_bytes_round_trip(self):
        vals = [0.1, 0.2, 0.3]
        unpacked = _unpack_embedding(self._pack(vals))
        assert len(unpacked) == 3
        for a, b in zip(vals, unpacked):
            assert abs(a - b) < 1e-5

    def test_none_returns_none(self):
        assert _unpack_embedding(None) is None

    def test_list_passes_through(self):
        vals = [0.5, 0.6]
        assert _unpack_embedding(vals) == vals

    def test_empty_bytes_returns_none(self):
        assert _unpack_embedding(b"") is None


# ── _is_established ───────────────────────────────────────────────────────────
# Frozen thresholds — changing these requires retraining the contradiction head.


class TestIsEstablished:
    def test_incoming_always_false(self):
        assert _is_established("incoming", {"reinforcement_count": 99}) is False

    def test_episode_always_false(self):
        assert _is_established("episode", {"confidence": 1.0, "access_count": 100}) is False

    def test_trait_below_threshold_is_false(self):
        assert _is_established("trait", {"reinforcement_count": 2}) is False

    def test_trait_at_threshold_is_true(self):
        # Threshold is exactly 3 — burned into trained model
        assert _is_established("trait", {"reinforcement_count": 3}) is True

    def test_trait_above_threshold_is_true(self):
        assert _is_established("trait", {"reinforcement_count": 10}) is True

    def test_trait_missing_key_defaults_below_threshold(self):
        # Default reinforcement_count is 1, below the threshold of 3
        assert _is_established("trait", {}) is False

    def test_concept_high_confidence_is_true(self):
        assert _is_established("concept", {"confidence": 0.8, "access_count": 0}) is True

    def test_concept_just_below_confidence_is_false(self):
        assert _is_established("concept", {"confidence": 0.79, "access_count": 0}) is False

    def test_concept_high_access_count_is_true(self):
        assert _is_established("concept", {"confidence": 0.0, "access_count": 5}) is True

    def test_concept_access_count_below_threshold_is_false(self):
        assert _is_established("concept", {"confidence": 0.0, "access_count": 4}) is False

    def test_unknown_memory_type_returns_false(self):
        assert _is_established("xyzzy", {"confidence": 1.0, "reinforcement_count": 100}) is False


# ── _softmax numerical stability ──────────────────────────────────────────────


class TestSoftmaxStability:
    def test_large_positive_logits_produce_no_nan(self):
        probs = _softmax(np.array([[1e30, 1e30, 1e30]], dtype=np.float32))
        assert not np.any(np.isnan(probs))
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_large_negative_logits_produce_no_nan(self):
        probs = _softmax(np.array([[-1e30, -1e30, -1e30]], dtype=np.float32))
        assert not np.any(np.isnan(probs))
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_dominant_logit_captures_near_all_probability(self):
        probs = _softmax(np.array([[1e30, -1e30, 0.0]], dtype=np.float32))
        assert not np.any(np.isnan(probs))
        assert probs[0, 0] > 0.99


# ── _ClassifierHead forward / weight transpose ────────────────────────────────


class TestClassifierHeadForward:
    def test_output_shape_matches_num_classes(self):
        W1 = np.eye(5, 3, dtype=np.float32)   # (hidden=5, input=3)
        W2 = np.eye(2, 5, dtype=np.float32)   # (classes=2, hidden=5)
        head = _ClassifierHead(
            W1=W1, b1=np.zeros(5, np.float32),
            W2=W2, b2=np.zeros(2, np.float32),
            labels=["a", "b"], activation="relu",
        )
        logits = head.forward(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        assert logits.shape == (1, 2)

    def test_transposed_weight_convention_produces_correct_values(self):
        # W1 stored as (out, in) — nn.Linear convention.
        # features @ W1.T must give correct result, not features @ W1.
        W1 = np.eye(5, 3, dtype=np.float32)
        W2 = np.eye(2, 5, dtype=np.float32)
        head = _ClassifierHead(
            W1=W1, b1=np.zeros(5, np.float32),
            W2=W2, b2=np.zeros(2, np.float32),
            labels=["a", "b"], activation="relu",
        )
        # Input [1,0,0]: relu(h=[1,0,0,0,0]) → out=[1,0]
        logits = head.forward(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        assert abs(logits[0, 0] - 1.0) < 1e-5
        assert abs(logits[0, 1] - 0.0) < 1e-5

    def test_gelu_activation_differs_from_relu(self):
        # GELU is non-zero for small negative inputs; relu is zero.
        W1 = np.ones((4, 2), dtype=np.float32) * -0.5
        W2 = np.ones((1, 4), dtype=np.float32)
        head_gelu = _ClassifierHead(
            W1=W1, b1=np.zeros(4, np.float32),
            W2=W2, b2=np.zeros(1, np.float32),
            labels=["x"], activation="gelu",
        )
        head_relu = _ClassifierHead(
            W1=W1, b1=np.zeros(4, np.float32),
            W2=W2, b2=np.zeros(1, np.float32),
            labels=["x"], activation="relu",
        )
        features = np.array([[1.0, 1.0]], dtype=np.float32)
        gelu_out = head_gelu.forward(features)
        relu_out = head_relu.forward(features)
        # The hidden pre-activations are negative (W1 * -0.5), so gelu ≠ relu
        assert not np.allclose(gelu_out, relu_out)
