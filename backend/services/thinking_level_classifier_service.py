"""
ThinkingLevelClassifierService — deliberation-depth classifier for user turns.

Wraps the thinking_level MLP head (gte-modernbert encoder + 2-layer MLP, 3-class)
to predict how much deliberation a user turn deserves before Chalie responds.

Classes:
  low    — chit-chat, direct lookups, factual Q&A
  medium — bounded research, short synthesis, single-function code
  high   — multi-step reasoning, multi-tool orchestration, planning

Input contract (v0.9.0):
  - Raw user turn goes straight into the encoder. No prefix, no suffix, no
    options list — the old [prev=...] / Options: A..C / Answer: format was
    retired when the Qwen single-token classifier was dropped.
  - prev_level is encoded as a 4-dim one-hot concatenated to the 768-d
    embedding before the MLP head. Index order is FROZEN:
    {none: 0, low: 1, medium: 2, high: 3}

See training/data/tasks/thinking_level/SIGNALS.md for the full signal contract.
Any change to the input format, class labels, or prev_level vocabulary
requires retraining the model, not just a meta update.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

LOG_PREFIX = "[THINKING]"

_VALID_PREV = frozenset(['none', 'low', 'medium', 'high'])

# Index order is burned into the trained head's W1. NEVER reorder.
PREV_LEVEL_TO_IDX = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}

_CONFIDENCE_THRESHOLD = 0.70


class ThinkingLevelClassifierService:
    """Classify the deliberation depth required for a user turn.

    Primary path: gte-modernbert encoder → MLP head inference.
    Fallback: sticky prev_level (or 'medium' on cold start).
    All exceptions are silently trapped — gate failure must never kill a turn.
    """

    def classify(self, user_turn: str, prev_level: str = 'none') -> dict:
        """Return {'level': str, 'confidence': float, 'fallback': bool}.

        Flow:
          1. Validate prev_level. Invalid → 'none'.
          2. Build 4-dim one-hot from PREV_LEVEL_TO_IDX.
          3. Call onnx_inference_service.predict('thinking_level', user_turn, onehot).
          4. If label is None OR confidence < threshold: sticky fallback.
          5. Otherwise: return direct label + confidence, fallback=False.
        """
        if prev_level not in _VALID_PREV:
            prev_level = 'none'

        try:
            from services.onnx_inference_service import get_onnx_inference_service

            svc = get_onnx_inference_service()

            # Build (1, 4) float32 one-hot — order is FROZEN per PREV_LEVEL_TO_IDX
            onehot = np.zeros((1, 4), dtype=np.float32)
            onehot[0, PREV_LEVEL_TO_IDX[prev_level]] = 1.0

            label, confidence = svc.predict("thinking_level", user_turn,
                                            extra_features=onehot)

            if label is None or confidence < _CONFIDENCE_THRESHOLD:
                fallback_level = prev_level if prev_level != 'none' else 'medium'
                logger.info(
                    "%s MLP level=%s confidence=%.3f prev=%s fallback=true",
                    LOG_PREFIX,
                    fallback_level,
                    confidence if label is not None else 0.0,
                    prev_level,
                )
                return {'level': fallback_level, 'confidence': 0.0, 'fallback': True}

            logger.info(
                "%s MLP level=%s confidence=%.3f prev=%s fallback=false",
                LOG_PREFIX, label, confidence, prev_level,
            )
            return {'level': label, 'confidence': confidence, 'fallback': False}

        except Exception as exc:
            fallback_level = prev_level if prev_level != 'none' else 'medium'
            logger.info(
                "%s classify failed (%s) — fallback: %s",
                LOG_PREFIX, exc, fallback_level,
            )
            return {'level': fallback_level, 'confidence': 0.0, 'fallback': True}
