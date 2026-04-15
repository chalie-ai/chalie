"""
ThinkingLevelClassifierService — deliberation-depth classifier for user turns.

Wraps the thinking_level ONNX model (Qwen2.5-0.5B fine-tuned, 3-class) to
predict how much deliberation a user turn deserves before Chalie responds.

Classes:
  A → low    (chit-chat, direct lookups, factual Q&A)
  B → medium (bounded research, short synthesis, single-function code)
  C → high   (multi-step reasoning, multi-tool orchestration, planning)

ONNX MODEL CONTRACT — see training/data/tasks/thinking_level/SIGNALS.md.
Any change to the input format, class labels, or prev_level vocabulary
requires retraining the model, not just a meta update.
"""

import logging

logger = logging.getLogger(__name__)

LOG_PREFIX = "[THINKING]"

_VALID_PREV = frozenset(['none', 'low', 'medium', 'high'])


class ThinkingLevelClassifierService:
    """Classify the deliberation depth required for a user turn.

    Primary path: ONNX head inference (sub-millisecond once base is warm).
    Fallback: sticky prev_level, or 'medium' on cold start.
    All exceptions are silently trapped — gate failure must never kill a turn.
    """

    # !! STATIC CONTRACT — must match CLASS_LABELS in
    # !! training/data/tasks/thinking_level/__init__.py
    _ONNX_LABEL_TO_CLASS = {'A': 'low', 'B': 'medium', 'C': 'high'}

    _CONFIDENCE_THRESHOLD = 0.70

    def classify(self, user_turn: str, prev_level: str = 'none') -> dict:
        """Return {'level': str, 'confidence': float, 'fallback': bool}.

        Flow:
          1. Validate prev_level. Invalid → 'none'.
          2. Build byte-exact ONNX input per INTEGRATION.md §2.1.
          3. Call onnx_inference_service.predict("thinking_level", input_text).
          4. If label is None OR confidence < threshold: sticky fallback.
          5. Otherwise: return mapped class + confidence, fallback=False.
        """
        if prev_level not in _VALID_PREV:
            prev_level = 'none'

        try:
            from services.onnx_inference_service import get_onnx_inference_service

            svc = get_onnx_inference_service()
            input_text = self._build_onnx_input(user_turn, prev_level)
            label, confidence = svc.predict("thinking_level", input_text)

            if label is None or confidence < self._CONFIDENCE_THRESHOLD:
                fallback_level = prev_level if prev_level != 'none' else 'medium'
                logger.info(
                    "%s label=%s confidence=%.3f below threshold — sticky fallback: %s "
                    "(prev=%s)",
                    LOG_PREFIX, label, confidence if label is not None else 0.0,
                    fallback_level, prev_level,
                )
                return {'level': fallback_level, 'confidence': 0.0, 'fallback': True}

            level = self._ONNX_LABEL_TO_CLASS.get(label, 'medium')
            logger.info(
                "%s level=%s confidence=%.3f prev=%s fallback=False",
                LOG_PREFIX, level, confidence, prev_level,
            )
            return {'level': level, 'confidence': confidence, 'fallback': False}

        except Exception as exc:
            fallback_level = prev_level if prev_level != 'none' else 'medium'
            logger.info(
                "%s classify failed (%s) — fallback: %s",
                LOG_PREFIX, exc, fallback_level,
            )
            return {'level': fallback_level, 'confidence': 0.0, 'fallback': True}

    def _build_onnx_input(self, user_turn: str, prev_level: str) -> str:
        """Byte-exact per INTEGRATION.md §2.1.

        Format:
            [prev=<x>] <turn>\\nOptions: A: low | B: medium | C: high\\nAnswer:

        No trailing space or newline after 'Answer:'.
        """
        return (
            f"[prev={prev_level}] {user_turn}\n"
            "Options: A: low | B: medium | C: high\n"
            "Answer:"
        )
