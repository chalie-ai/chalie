
import logging

logger = logging.getLogger(__name__)

LOG_PREFIX = "[DELIBERATION]"


class DeliberationScoreService:

    def __init__(self, inference_service=None):
        self._inference_service = inference_service

    def _get_svc(self):
        if self._inference_service is not None:
            return self._inference_service
        from services.onnx_inference_service import get_onnx_inference_service
        return get_onnx_inference_service()

    def classify(self, user_turn: str) -> 'float | None':
        if not user_turn or not user_turn.strip():
            return None
        try:
            svc = self._get_svc()
            return svc.predict_scalar("deliberation_score", user_turn)
        except Exception as exc:
            logger.info("%s classify failed: %s", LOG_PREFIX, exc)
            return None
