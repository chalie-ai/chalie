"""
AutonomousExecutionGate — decides whether an action can be auto-executed.

Uses consequence tier (from ConsequenceClassifierService) to answer a single
question: can Chalie act on this without asking the user?

Reversibility is the hard line:
  Tier 0 (Observe):  Always auto-execute — zero side-effects.
  Tier 1 (Organize): Require user approval.
  Tier 2 (Act):      Require user approval.
  Tier 3 (Commit):   Require user approval.

Integration points:
  - ActDispatcherService: gate external actions by tier before execution
  - GoalPursuitProcessor: verify tier before each background cycle
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)
LOG_PREFIX = "[EXEC GATE]"


class AutonomousExecutionGate:
    """
    Decides whether an action can be auto-executed based on consequence tier.

    Tier 0 (Observe) is auto-executed — zero side-effects.
    Every other tier requires explicit user approval.
    """

    _TIER_NAMES = {0: "observe", 1: "organize", 2: "act", 3: "commit"}

    def __init__(self):
        self._consequence_service = None

    def evaluate(self, action_description: str, domain: str) -> dict:
        consequence_result = self._get_consequence_service().classify(action_description)
        tier: int = consequence_result.get("tier", 2)
        tier_name: str = self._TIER_NAMES.get(tier, "act")

        auto_execute = self.should_auto_execute(tier)
        reasoning = self._build_reasoning(tier, tier_name, domain, auto_execute)

        result = {
            "auto_execute": auto_execute,
            "consequence_tier": tier,
            "consequence_name": tier_name,
            "domain": domain,
            "reasoning": reasoning,
        }

        logger.info(
            f"{LOG_PREFIX} Decision: auto_execute={auto_execute} | "
            f"tier={tier_name}({tier}) | domain={domain!r} | "
            f"action={action_description!r:.100}"
        )

        return result

    def should_auto_execute(self, consequence_tier: int) -> bool:
        return consequence_tier == 0

    def _get_consequence_service(self):
        if self._consequence_service is None:
            from services.consequence_classifier_service import (
                get_consequence_classifier_service,
            )
            self._consequence_service = get_consequence_classifier_service()
        return self._consequence_service

    @staticmethod
    def _build_reasoning(tier: int, tier_name: str, domain: str, auto_execute: bool) -> str:
        if tier == 0:
            return "Tier 0 (observe) — no side-effects; always safe to auto-execute."
        return (
            f"Tier {tier} ({tier_name}) in domain '{domain}' — requires explicit user approval."
        )


_instance: Optional[AutonomousExecutionGate] = None
_instance_lock = threading.Lock()


def get_autonomous_execution_gate() -> AutonomousExecutionGate:
    global _instance
    if _instance is not None:
        return _instance

    with _instance_lock:
        if _instance is not None:
            return _instance
        _instance = AutonomousExecutionGate()
        return _instance
