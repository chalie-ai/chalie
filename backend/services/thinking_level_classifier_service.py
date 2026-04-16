"""
ThinkingLevelClassifierService — deliberation-depth classifier for user turns.

Wraps the thinking_level MLP head (gte-modernbert encoder + 2-layer MLP, 3-class)
to predict how much deliberation a user turn deserves before Chalie responds.

Classes:
  low    — chit-chat, direct lookups, factual Q&A
  medium — bounded research, short synthesis, single-function code
  high   — multi-step reasoning, multi-tool orchestration, planning

Input contract:
  - Raw user turn goes straight into the encoder. No prefix, no suffix, no
    options list.
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

# Per-channel consecutive-inherit counter. Prevents the anti-demotion guard
# from permanently locking a channel in 'high'/'medium' once the user has
# actually moved to chit-chat.
#
# Mechanic: each time the guard inherits prev_level on a low-head signal,
# increment the channel's counter. On the SECOND consecutive inherit we
# allow demotion — i.e. the guard protects a single ambiguous follow-up
# ("yes", "go on") but yields to the head after two consecutive low signals.
#
# Reset to 0 whenever:
#   - the head emits a non-low label confidently (above threshold)
#   - the user turn genuinely classifies as 'low' (prev was none/low, no
#     inherit needed)
# Scope: in-memory, keyed by channel. Process restart resets the chain,
# which is the correct behaviour (ambiguous short follow-ups at cold start
# fall back to 'low'/'none' anyway).
_INHERIT_COUNT_BY_CHANNEL: dict[str, int] = {}
_INHERIT_LIMIT = 2


def _reset_inherit_counter(channel: str) -> None:
    _INHERIT_COUNT_BY_CHANNEL.pop(channel, None)


def _bump_inherit_counter(channel: str) -> int:
    _INHERIT_COUNT_BY_CHANNEL[channel] = _INHERIT_COUNT_BY_CHANNEL.get(channel, 0) + 1
    return _INHERIT_COUNT_BY_CHANNEL[channel]


class ThinkingLevelClassifierService:
    """Classify the deliberation depth required for a user turn.

    Primary path: gte-modernbert encoder → MLP head inference.
    Fallback: sticky prev_level (or 'medium' on cold start).
    All exceptions are silently trapped — gate failure must never kill a turn.
    """

    def classify(self, user_turn: str, prev_level: str = 'none',
                 channel: str = '') -> dict:
        """Return {'level': str, 'confidence': float, 'fallback': bool}.

        Flow:
          1. Validate prev_level. Invalid → 'none'.
          2. Build 4-dim one-hot from PREV_LEVEL_TO_IDX.
          3. Call onnx_inference_service.predict('thinking_level', user_turn, onehot).
          4. If label is None OR confidence < threshold: sticky fallback.
          5. Otherwise: return direct label + confidence, fallback=False.

        ``channel`` keys the anti-demotion inherit counter. Omit (default '')
        for callers that don't care about chain-breaking — they simply never
        lock in.
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

            # Anti-demotion guard: when the previous turn was 'medium' or
            # 'high', a confident 'low' classification on a short ambiguous
            # follow-up ("yes go on", "ok", "continue") is almost always a
            # mis-classification on the FIRST such turn.
            #
            # Chain-break rule: after _INHERIT_LIMIT consecutive inherits on
            # the same channel, yield to the head. Prevents the "permanent
            # lock-in" failure mode where a channel that hit 'high' once
            # never escapes back to 'low' even when the user has clearly
            # moved on to chit-chat.
            if label == 'low' and prev_level in ('medium', 'high'):
                inherits = _INHERIT_COUNT_BY_CHANNEL.get(channel, 0)
                if inherits >= _INHERIT_LIMIT:
                    logger.info(
                        "%s anti-demotion chain broken: head=low confidence=%.3f "
                        "prev=%s channel=%s inherits=%d — yielding to head",
                        LOG_PREFIX, confidence, prev_level, channel, inherits,
                    )
                    _reset_inherit_counter(channel)
                    return {'level': 'low', 'confidence': confidence,
                            'fallback': False}
                count = _bump_inherit_counter(channel)
                logger.info(
                    "%s anti-demotion: head=low confidence=%.3f prev=%s "
                    "channel=%s inherits=%d — sticky inherit prev_level",
                    LOG_PREFIX, confidence, prev_level, channel, count,
                )
                return {'level': prev_level, 'confidence': confidence,
                        'fallback': True}

            if label is None or confidence < _CONFIDENCE_THRESHOLD:
                # Cold-start fallback is 'low' — a no-op that lets the LLM
                # answer without the [RULE] "think deeply" tail. 'medium' was
                # the previous default but it is an ACTIVE intervention, not
                # a neutral default; applying it whenever classifier confidence
                # dropped below threshold on the very first turn caused
                # spurious deliberation on simple recall/chit-chat turns in
                # the benchmark suite.
                fallback_level = prev_level if prev_level != 'none' else 'low'
                logger.info(
                    "%s MLP level=%s confidence=%.3f prev=%s fallback=true",
                    LOG_PREFIX,
                    fallback_level,
                    confidence if label is not None else 0.0,
                    prev_level,
                )
                # Low-confidence fallback that inherits a non-low prev_level
                # is the SAME lock-in risk as the anti-demotion path above.
                # Count it too.
                if fallback_level in ('medium', 'high'):
                    _bump_inherit_counter(channel)
                else:
                    _reset_inherit_counter(channel)
                return {'level': fallback_level, 'confidence': 0.0, 'fallback': True}

            # Confident non-low classification — chain broken, reset counter.
            _reset_inherit_counter(channel)
            logger.info(
                "%s MLP level=%s confidence=%.3f prev=%s fallback=false",
                LOG_PREFIX, label, confidence, prev_level,
            )
            return {'level': label, 'confidence': confidence, 'fallback': False}

        except Exception as exc:
            # Same reasoning as the low-confidence fallback above: default to
            # 'low' (no-op) on cold start, not 'medium' (active [RULE] tail).
            fallback_level = prev_level if prev_level != 'none' else 'low'
            logger.info(
                "%s classify failed (%s) — fallback: %s",
                LOG_PREFIX, exc, fallback_level,
            )
            if fallback_level in ('medium', 'high'):
                _bump_inherit_counter(channel)
            else:
                _reset_inherit_counter(channel)
            return {'level': fallback_level, 'confidence': 0.0, 'fallback': True}
