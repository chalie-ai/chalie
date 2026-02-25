"""
Critic Service — Post-action verification for the ACT loop.

Evaluates action results for correctness after execution.
Catches silent errors (wrong dates, irrelevant recalls) before they compound.

Skip conditions:
  - Simple reads with high-confidence structured results
  - Dispatcher confidence >= 0.9 (subject to calibration)
  - Deterministic actions with no ambiguity

Supervised autonomy:
  - Safe actions (recall, memorize, introspect, associate): silent correction
  - Consequential actions (schedule, list, tools): pause + ask user
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)
LOG_PREFIX = "[CRITIC]"

# Actions that can be silently corrected without user confirmation
SAFE_ACTIONS = {'recall', 'memorize', 'introspect', 'associate', 'autobiography', 'moment'}

# Actions that are simple reads — skip critic entirely when high confidence
READ_ACTIONS = {'recall', 'introspect'}

# Maximum retries per action to prevent oscillation
MAX_CRITIC_RETRIES = 2

# Fixed fatigue cost per critic evaluation
CRITIC_FATIGUE_COST = 0.3

# Confidence threshold above which critic is skipped (subject to calibration)
DEFAULT_CONFIDENCE_SKIP_THRESHOLD = 0.9

# Calibration: if correction rate for high-confidence actions exceeds this,
# lower the effective skip threshold
CALIBRATION_CORRECTION_RATE_LIMIT = 0.15

# Exponential moving average alpha for calibration
CALIBRATION_EMA_ALPHA = 0.1


class CriticService:
    """Post-action verification with skip logic, confidence calibration, and telemetry."""

    def __init__(self):
        self._llm = None
        self._prompt_template = None

        # Per-session telemetry
        self.total_evaluations = 0
        self.corrections = 0
        self.escalations = 0
        self.oscillation_events = 0
        self.skipped = 0
        self.severity_counts = {'minor': 0, 'major': 0}

        # Confidence calibration state (per action type)
        # Tracks EMA of correction rate for actions that reported confidence >= threshold
        self._calibration: Dict[str, float] = {}

    def should_skip(self, action_type: str, result: Dict[str, Any]) -> bool:
        """
        Determine if critic evaluation should be skipped.

        Skip when:
        - Action is a simple read AND confidence >= threshold
        - Dispatcher confidence >= skip threshold (calibrated)
        - Action failed (nothing to verify)
        """
        if result.get('status') != 'success':
            return True

        confidence = result.get('confidence', 0.0)
        effective_threshold = self._get_calibrated_threshold(action_type)

        # Simple read with high confidence → skip
        if action_type in READ_ACTIONS and confidence >= effective_threshold:
            logger.debug(f"{LOG_PREFIX} Skipping critic for {action_type} (read + confidence={confidence:.2f})")
            self.skipped += 1
            return True

        # Any action with very high confidence → skip
        if confidence >= effective_threshold:
            logger.debug(f"{LOG_PREFIX} Skipping critic for {action_type} (confidence={confidence:.2f} >= {effective_threshold:.2f})")
            self.skipped += 1
            return True

        return False

    def evaluate(
        self,
        original_request: str,
        action_type: str,
        action_intent: Dict[str, Any],
        action_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate an action result for correctness.

        Args:
            original_request: The user's original prompt
            action_type: Type of action executed
            action_intent: The action specification that was dispatched
            action_result: The structured result from the dispatcher

        Returns:
            Critic verdict: {verified, severity?, issue?, correction?}
        """
        self.total_evaluations += 1

        try:
            llm = self._get_llm()
            prompt = self._build_prompt(original_request, action_type, action_intent, action_result)

            response_text = llm.send_message("", prompt).text
            verdict = self._parse_verdict(response_text)

            if not verdict.get('verified', True):
                severity = verdict.get('severity', 'minor')
                self.severity_counts[severity] = self.severity_counts.get(severity, 0) + 1

                if verdict.get('correction'):
                    self.corrections += 1
                    self._update_calibration(action_type, corrected=True)
                    logger.info(
                        f"{LOG_PREFIX} Correction for {action_type}: "
                        f"{verdict.get('issue', 'unknown issue')}"
                    )
                else:
                    self.escalations += 1
                    logger.info(
                        f"{LOG_PREFIX} Escalation for {action_type}: "
                        f"{verdict.get('issue', 'unknown issue')}"
                    )
            else:
                self._update_calibration(action_type, corrected=False)

            return verdict

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Evaluation failed: {e}")
            # On failure, default to verified=true (err on the side of accepting)
            return {'verified': True}

    def format_correction_entry(
        self,
        action_type: str,
        original_result: str,
        correction: str,
        final_result: str,
    ) -> str:
        """
        Format a correction for injection into act_history.

        Corrections are additive — the failed attempt is retained so the planner
        avoids repeating the same mistake.
        """
        return (
            f"[CRITIC CORRECTION] Action '{action_type}' result was corrected. "
            f"Original: {original_result}. "
            f"Correction: {correction}. "
            f"Corrected result: {final_result}."
        )

    def is_safe_action(self, action_type: str) -> bool:
        """Check if an action type can be silently corrected without user confirmation."""
        return action_type in SAFE_ACTIONS

    def get_telemetry(self) -> Dict[str, Any]:
        """Return critic telemetry for logging to cortex_iterations."""
        total = self.total_evaluations + self.skipped
        return {
            'critic_total_checks': total,
            'critic_evaluations': self.total_evaluations,
            'critic_skipped': self.skipped,
            'critic_corrections': self.corrections,
            'critic_escalations': self.escalations,
            'critic_oscillation_events': self.oscillation_events,
            'critic_correction_rate': (
                round(self.corrections / self.total_evaluations, 3)
                if self.total_evaluations > 0 else 0.0
            ),
            'critic_severity_distribution': dict(self.severity_counts),
            'critic_calibration': dict(self._calibration),
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _get_calibrated_threshold(self, action_type: str) -> float:
        """
        Get the effective confidence skip threshold for an action type.

        If high-confidence actions of this type are being corrected above
        the calibration limit, the threshold is raised (harder to skip).
        """
        correction_rate = self._calibration.get(action_type, 0.0)
        if correction_rate > CALIBRATION_CORRECTION_RATE_LIMIT:
            # Raise threshold proportionally — more corrections = harder to skip
            adjusted = DEFAULT_CONFIDENCE_SKIP_THRESHOLD + (correction_rate - CALIBRATION_CORRECTION_RATE_LIMIT)
            return min(1.0, adjusted)
        return DEFAULT_CONFIDENCE_SKIP_THRESHOLD

    def _update_calibration(self, action_type: str, corrected: bool):
        """Update EMA correction rate for confidence calibration."""
        current = self._calibration.get(action_type, 0.0)
        value = 1.0 if corrected else 0.0
        self._calibration[action_type] = (
            CALIBRATION_EMA_ALPHA * value + (1 - CALIBRATION_EMA_ALPHA) * current
        )

    def _build_prompt(
        self,
        original_request: str,
        action_type: str,
        action_intent: Dict[str, Any],
        action_result: Dict[str, Any],
    ) -> str:
        """Build the critic evaluation prompt."""
        template = self._load_prompt()

        # Serialize intent and result for the prompt
        intent_str = json.dumps(action_intent, default=str, indent=2)
        result_str = json.dumps(action_result, default=str, indent=2)

        return (
            template
            .replace('{{original_request}}', original_request)
            .replace('{{action_type}}', action_type)
            .replace('{{action_intent}}', intent_str)
            .replace('{{action_result}}', result_str)
        )

    def _parse_verdict(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM verdict response into a structured dict."""
        try:
            # Try direct JSON parse first
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        try:
            import re
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: check for obvious failure signals in text
        lower = response_text.lower()
        if any(w in lower for w in ['incorrect', 'wrong', 'error', 'mismatch']):
            return {
                'verified': False,
                'severity': 'minor',
                'issue': response_text[:200],
                'correction': None,
            }

        # Default: accept
        return {'verified': True}

    def _load_prompt(self) -> str:
        """Load the critic prompt template."""
        if self._prompt_template is None:
            import os
            prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
            path = os.path.join(prompts_dir, 'act-critic.md')
            with open(path, 'r') as f:
                self._prompt_template = f.read()
        return self._prompt_template

    def _get_llm(self):
        """Get or create the LLM service for critic evaluations."""
        if self._llm is None:
            from services.llm_service import create_llm_service
            from services.config_service import ConfigService
            # Reuse the cognitive-triage agent config (lightweight model)
            agent_cfg = ConfigService.resolve_agent_config('cognitive-triage')
            self._llm = create_llm_service(agent_cfg)
        return self._llm
