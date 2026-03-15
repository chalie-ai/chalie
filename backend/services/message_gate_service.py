"""
Message Gate Service — deterministic pre-filter replacing CognitiveTriageService.

Three responsibilities:
  1. Empty input guard (deterministic)
  2. CANCEL detection (deterministic keyword match)
  3. ONNX mode gate → returns route + confidence

No LLM call. No tool selection. No skill selection. No effort estimation.
"""

import re
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[GATE]"

# Cancel keywords — user explicitly stopping something
_CANCEL_PATTERNS = re.compile(
    r'^(stop|cancel|nevermind|never mind|forget it|abort|nvm)\b',
    re.IGNORECASE
)


@dataclass
class GateResult:
    """Output of the message gate.

    Attributes:
        route: 'cancel' | 'respond' | 'act'
        confidence: Float 0-1 confidence in the route decision.
        gate_time_ms: Wall-clock ms for the gate call.
    """
    route: str        # 'cancel' | 'respond' | 'act'
    confidence: float # 0-1
    gate_time_ms: float


class MessageGateService:
    """Thin deterministic gate: empty guard → CANCEL → ONNX mode gate."""

    def gate(self, text: str) -> GateResult:
        """Route a message. No LLM, no tool selection."""
        start = time.time()

        # 1. Empty input guard
        if not text or not text.strip():
            return GateResult(
                route='cancel',
                confidence=1.0,
                gate_time_ms=(time.time() - start) * 1000,
            )

        # 2. CANCEL detection (deterministic)
        if _CANCEL_PATTERNS.match(text.strip()):
            return GateResult(
                route='cancel',
                confidence=1.0,
                gate_time_ms=(time.time() - start) * 1000,
            )

        # 3. ONNX mode gate
        route = 'act'  # default: everything goes to ACT
        confidence = 0.5

        try:
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()

            if svc.is_available("mode-tiebreaker"):
                label, conf = svc.predict("mode-tiebreaker", text)
                if label is not None:
                    onnx_mode = label.upper()
                    if onnx_mode == 'RESPOND' and conf >= 0.85:
                        route = 'respond'
                        confidence = conf
                    elif onnx_mode in ('ACT', 'CLARIFY'):
                        route = 'act'
                        confidence = conf
                    else:
                        # IGNORE or unknown → act (Chalie always responds)
                        route = 'act'
                        confidence = conf
                    logger.info(f"{LOG_PREFIX} ONNX: {onnx_mode}({conf:.2f}) → route={route}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} ONNX gate failed: {e}")

        return GateResult(
            route=route,
            confidence=confidence,
            gate_time_ms=(time.time() - start) * 1000,
        )

    def prefilter_skills(self, text: str) -> tuple:
        """Run ONNX skill selector to pre-filter skills and detect tool need.

        Returns:
            (selected_skills: list[str], needs_external_tool: bool)
        """
        selected_skills = []
        needs_external_tool = False

        try:
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()

            if not svc.is_available("skill-selector"):
                return selected_skills, needs_external_tool

            input_text = f"{text}\nSkills:"
            predictions = svc.predict_multi_label("skill-selector", input_text)

            if predictions:
                for skill_name, conf in predictions:
                    if skill_name == 'needs_external_tool':
                        needs_external_tool = True
                    else:
                        selected_skills.append(skill_name)

                logger.info(f"{LOG_PREFIX} ONNX skills: {selected_skills}, ext_tool={needs_external_tool}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Skill pre-filter unavailable: {e}")

        return selected_skills, needs_external_tool
