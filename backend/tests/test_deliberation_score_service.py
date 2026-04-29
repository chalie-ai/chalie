"""Feature tests for DeliberationScoreService.

Asserts bucketing behaviour and None semantics against the real shipped
deliberation_score head and real encoder. The existing
test_onnx_inference_service.py already covers predict_scalar range/NaN guards
and the empty-string/whitespace None path; those tests are NOT duplicated here.

Skips automatically when the encoder ONNX is absent from disk.

pytestmark applies pytest.mark.integration to every test in this file.
"""

import math
import os

import pytest

from services.deliberation_score_service import DeliberationScoreService
from services.onnx_inference_service import OnnxInferenceService

pytestmark = pytest.mark.integration

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MODELS_DIR = os.path.join(_BACKEND_ROOT, "data", "models")
_DEFAULT_PRETRAINED_DIR = os.path.join(_BACKEND_ROOT, "data", "pre-trained")
_ENCODER_PATH = os.path.join(
    _DEFAULT_MODELS_DIR, "gte-modernbert-base", "onnx", "model.onnx"
)


def _require_encoder():
    if not os.path.exists(_ENCODER_PATH):
        pytest.skip("gte-modernbert-base encoder not on disk — skipping encoder-dependent test")


def _real_svc() -> DeliberationScoreService:
    _require_encoder()
    from services.embedding_service import _get_session_and_tokenizer
    _get_session_and_tokenizer()
    return DeliberationScoreService(
        inference_service=OnnxInferenceService(_DEFAULT_MODELS_DIR, _DEFAULT_PRETRAINED_DIR)
    )


# Bucket thresholds from the shipped meta: high > 0.67, medium > 0.46, else low.
_HIGH_THR = 0.67
_MED_THR = 0.46


class TestDeliberationScoreServiceBucketing:
    """Real-world bucketing: conversational inputs land low, complex prompts land high."""

    def test_short_conversational_input_scores_below_medium_threshold(self):
        """'hi' is pure chit-chat — classifier should score it below the medium bucket."""
        _require_encoder()
        svc = _real_svc()
        result = svc.classify("hi")
        assert result is not None, "classify('hi') returned None — head unavailable?"
        assert math.isfinite(result)
        # Score must be below the medium threshold to land in the 'low' bucket.
        assert result < _MED_THR, (
            f"'hi' scored {result:.4f} — expected < {_MED_THR} (low bucket)"
        )

    def test_complex_multistep_prompt_scores_above_high_threshold(self):
        """A deep, multi-constraint engineering prompt should land in the high bucket."""
        _require_encoder()
        svc = _real_svc()
        prompt = (
            "Design a fault-tolerant multi-region distributed system. "
            "Explain the sharding strategy, failover sequence, consistency model "
            "trade-offs, observability requirements, and how you would test the "
            "full disaster-recovery path end-to-end."
        )
        result = svc.classify(prompt)
        assert result is not None, "classify(<complex>) returned None — head unavailable?"
        assert math.isfinite(result)
        assert result > _HIGH_THR, (
            f"Complex engineering prompt scored {result:.4f} — expected > {_HIGH_THR} (high bucket)"
        )

    def test_simple_factual_question_scores_between_low_and_high(self):
        """A one-step factual question scores somewhere in the low-to-medium range,
        not at the top of the scale — verifies the classifier is not saturating."""
        _require_encoder()
        svc = _real_svc()
        result = svc.classify("What is the capital of France?")
        assert result is not None
        assert math.isfinite(result)
        # Score must be comfortably below the high threshold — not clamped to 1.0.
        assert result < _HIGH_THR, (
            f"Simple factual question scored {result:.4f} — expected < {_HIGH_THR}"
        )

    def test_classify_two_different_inputs_produce_different_scores(self):
        """'hi' and a complex prompt must not return the same scalar — confirms the
        classifier is actually discriminating, not outputting a constant."""
        _require_encoder()
        svc = _real_svc()
        low = svc.classify("thanks")
        high = svc.classify(
            "Analyse the Byzantine Generals problem, prove why 3f+1 nodes are required "
            "for safety under f Byzantine faults, and sketch a practical BFT consensus protocol."
        )
        assert low is not None and high is not None
        # The gap should be substantial — not just floating-point noise.
        assert high - low > 0.2, (
            f"Expected a clear gap: low={low:.4f}, high={high:.4f}, diff={high - low:.4f}"
        )
