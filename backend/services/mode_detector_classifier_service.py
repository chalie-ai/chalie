"""
ModeDetectorClassifierService — 8-class multi-label mode classifier for user turns.

Wraps the mode_detector MLP head (gte-modernbert encoder + 2-layer MLP,
8 sigmoid outputs) to estimate which cognitive modes a user turn activates.

Modes: research, coding, brainstorm, analyze, plan, write, math, converse.

Output contract:
  - Returns a dict of {mode: probability} for all 8 modes.
  - Probabilities are raw sigmoids in [0, 1] — no threshold applied.
  - On any failure returns {} and logs at WARN. Never raises.

See mode_gate_service.py for the state machine that consumes these probabilities.
"""

import logging

logger = logging.getLogger(__name__)

LOG_PREFIX = "[MODE-DETECTOR]"


class ModeDetectorClassifierService:
    """Classify a user turn along 8 cognitive mode dimensions.

    Stateless — no caching, no internal state. A new instance per call is fine.
    The underlying encoder session is shared via OnnxInferenceService.
    """

    def predict(self, user_turn: str) -> dict[str, float]:
        """Return raw sigmoid probabilities for all 8 modes.

        Args:
            user_turn: Raw user input text.

        Returns:
            Dict mapping each mode name to its sigmoid probability in [0, 1].
            Returns ``{}`` when the classifier is unavailable or inference fails.
            Never raises.
        """
        try:
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()
            result = svc.predict_all_sigmoids("mode_detector", user_turn)
            return result
        except Exception as exc:
            logger.warning("%s predict failed: %s", LOG_PREFIX, exc)
            return {}
