"""
Message Gate Service — deterministic pre-filter replacing CognitiveTriageService.

Two responsibilities:
  1. Empty input guard (deterministic) — silently drops messages with no content
  2. ONNX mode gate → returns route + confidence

No LLM call. No tool selection. No skill selection. No effort estimation.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[GATE]"


@dataclass
class GateResult:
    """Output of the message gate.

    Attributes:
        route: 'respond' | 'act'
        confidence: Float 0-1 confidence in the route decision.
        gate_time_ms: Wall-clock ms for the gate call.
    """
    route: str        # 'respond' | 'act'
    confidence: float # 0-1
    gate_time_ms: float


class MessageGateService:
    """Thin deterministic gate: empty guard → ONNX mode gate."""

    def gate(self, text: str) -> GateResult:
        """Route a message. No LLM, no tool selection."""
        start = time.time()

        # Guard: empty input has no signal — default to act
        if not text or not text.strip():
            return GateResult(
                route='act',
                confidence=0.5,
                gate_time_ms=(time.time() - start) * 1000,
            )

        # ONNX mode gate — binary RESPOND/ACT classifier
        #
        # Qwen2.5-0.5B + sigmoid head. Trained on ~45k samples (14 edge case
        # categories). Eval: 97.9% accuracy, RESPOND F1=0.978 (P=98.9%, R=96.7%).
        # Threshold ≥ 0.85 → RESPOND fast-path. Below that → ACT (safe default).
        route = 'act'
        confidence = 0.5

        try:
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()

            if svc.is_available("mode-gate"):
                # mode-gate is multi_label with single "RESPOND" label.
                # sigmoid > threshold → RESPOND fires; otherwise ACT (safe default).
                predictions = svc.predict_multi_label("mode-gate", text)
                respond_hit = next(
                    ((lbl, c) for lbl, c in predictions if lbl.upper() == 'RESPOND'),
                    None,
                )
                if respond_hit:
                    route = 'respond'
                    confidence = respond_hit[1]
                    logger.info(f"{LOG_PREFIX} ONNX: RESPOND({confidence:.2f}) → route=respond")
                else:
                    route = 'act'
                    confidence = 0.5
                    logger.info(f"{LOG_PREFIX} ONNX: RESPOND below threshold → route=act")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} ONNX gate failed: {e}")

        return GateResult(
            route=route,
            confidence=confidence,
            gate_time_ms=(time.time() - start) * 1000,
        )
