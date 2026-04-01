"""
Post-loop Reflection Service — single LLM call for learning signal + methodology synthesis.

Replaces CriticService.reflect_on_execution(). Makes exactly one LLM call that produces
both a reflection (outcome_quality, lesson, etc.) and a methodology guidance paragraph.
Never blocks the response path. Never raises.
"""

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)
LOG_PREFIX = "[POST-LOOP REFLECTION]"


class PostLoopReflectionService:
    """One LLM call -> reflection + methodology update."""

    def __init__(self):
        self._llm = None
        self._prompt_template = None

    def reflect(
        self,
        exchange_text: str,
        act_history_text: str,
        termination_reason: str,
        existing_goal_guidance: str,
        loop_id: str,
        iterations_used: int = 0,
    ) -> dict:
        """
        One LLM call -> reflection + methodology update.

        Args:
            exchange_text: The user's original goal/message
            act_history_text: Formatted action history string
            termination_reason: Why the loop ended
            existing_goal_guidance: Current methodology paragraph ('' if none)
            loop_id: For logging
            iterations_used: Number of iterations executed

        Returns:
            Parsed dict with keys: outcome_quality, what_worked, what_failed,
            lesson, confidence, goal_guidance. Returns {} on any failure.
        """
        try:
            llm = self._get_llm()
            prompt = self._build_prompt(
                exchange_text=exchange_text,
                act_history_text=act_history_text,
                termination_reason=termination_reason,
                existing_goal_guidance=existing_goal_guidance,
                iterations_used=iterations_used,
            )
            response = llm.send_message("", prompt)
            reflection = self._parse_reflection(response.text)
            if reflection:
                logger.info(
                    f"{LOG_PREFIX} [{loop_id[:8] if loop_id else '?'}] "
                    f"quality={reflection.get('outcome_quality')}, "
                    f"confidence={reflection.get('confidence')}, "
                    f"guidance_len={len(reflection.get('goal_guidance', ''))}"
                )
            return reflection or {}
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} reflect() failed (non-fatal): {e}")
            return {}

    def _build_prompt(
        self,
        exchange_text: str,
        act_history_text: str,
        termination_reason: str,
        existing_goal_guidance: str,
        iterations_used: int = 0,
    ) -> str:
        template = self._load_prompt_template()

        from services.time_utils import utc_now
        current_datetime = utc_now().strftime('%A, %Y-%m-%d %H:%M UTC')

        return (
            template
            .replace('{{current_datetime}}', current_datetime)
            .replace('{{original_goal}}', exchange_text[:500])
            .replace('{{iterations}}', str(iterations_used))
            .replace('{{termination_reason}}', termination_reason or 'natural_completion')
            .replace('{{actions_summary}}', act_history_text or '(no actions executed)')
            .replace('{{existing_goal_guidance}}', existing_goal_guidance or '(none — write fresh guidance)')
        )

    def _parse_reflection(self, response_text: str) -> Optional[dict]:
        """Parse the LLM reflection response. Returns None if parsing fails."""
        parsed = None

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            pass

        if parsed is None:
            try:
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
            except (json.JSONDecodeError, AttributeError):
                pass

        if not isinstance(parsed, dict):
            return None

        try:
            return {
                'outcome_quality': max(0.0, min(1.0, float(parsed.get('outcome_quality', 0.5)))),
                'what_worked': str(parsed.get('what_worked', '') or ''),
                'what_failed': str(parsed.get('what_failed', '') or ''),
                'lesson': parsed.get('lesson') or None,
                'confidence': max(0.0, min(1.0, float(parsed.get('confidence', 0.5)))),
                'goal_guidance': str(parsed.get('goal_guidance', '') or ''),
            }
        except (TypeError, ValueError):
            return None

    def _load_prompt_template(self) -> str:
        """Load the act-reflection.md prompt template. Cached on first call."""
        if self._prompt_template is None:
            prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
            path = os.path.join(prompts_dir, 'act-reflection.md')
            with open(path, 'r') as f:
                self._prompt_template = f.read()
        return self._prompt_template

    def _get_llm(self):
        """Lazily create LLM service using cognitive-triage agent config."""
        if self._llm is None:
            from services.llm_service import create_llm_service
            from services.config_service import ConfigService
            agent_cfg = ConfigService.resolve_agent_config('cognitive-triage')
            self._llm = create_llm_service(agent_cfg)
        return self._llm
