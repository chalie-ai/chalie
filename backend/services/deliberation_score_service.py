"""
DeliberationScoreService — scalar regression wrapper for the deliberation-score head.

Thin wrapper over OnnxInferenceService.predict_scalar('deliberation_score', text).
Returns a float in [0.0, 1.0] or None on failure. Never raises.

The head is 768 → 256 → 1 with sigmoid output and extra_feature_dim=0.
No prev_level conditioning — the new head has no one-hot input.
"""

import logging

logger = logging.getLogger(__name__)

LOG_PREFIX = "[DELIBERATION]"


class DeliberationScoreService:
    """Classify deliberation depth required for a user turn as a scalar in [0, 1].

    One instance per call is fine — all state lives in OnnxInferenceService (singleton).
    Constructor accepts an optional inference_service for dependency injection in tests.
    """

    def __init__(self, inference_service=None):
        self._inference_service = inference_service

    def _get_svc(self):
        if self._inference_service is not None:
            return self._inference_service
        from services.onnx_inference_service import get_onnx_inference_service
        return get_onnx_inference_service()

    def classify(self, user_turn: str) -> 'float | None':
        """Return scalar in [0, 1] or None if head unavailable or text is empty.

        Never raises — all exceptions are caught and logged at INFO level.
        """
        if not user_turn or not user_turn.strip():
            return None
        try:
            svc = self._get_svc()
            result = svc.predict_scalar("deliberation_score", user_turn)
            return result
        except Exception as exc:
            logger.info("%s classify failed: %s", LOG_PREFIX, exc)
            return None
