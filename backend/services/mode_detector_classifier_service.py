"""ModeDetectorClassifierService — 8-class multi-label mode classifier for user turns."""

import logging

logger = logging.getLogger(__name__)

LOG_PREFIX = "[MODE-DETECTOR]"


class ModeDetectorClassifierService:

    def predict(self, user_turn: str) -> dict[str, float]:
        try:
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()
            return svc.predict_all_sigmoids("mode_detector", user_turn)
        except Exception as exc:
            logger.warning("%s predict failed: %s", LOG_PREFIX, exc)
            return {}
