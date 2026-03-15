"""
AutonomousExecutionGate — decides whether an action can be auto-executed.

Combines consequence tier (from ConsequenceClassifierService) with domain
confidence (from DomainConfidenceService) to answer a single question:
can Chalie act on this without asking the user?

Reversibility is the hard line:
  Tier 0 (Observe):  Always auto-execute — zero side-effects.
  Tier 1 (Organize): Auto-execute if domain_confidence >= 0.50.
  Tier 2 (Act):      Auto-execute if domain_confidence >= 0.75.
  Tier 3 (Commit):   Never auto-execute regardless of confidence.

The thresholds are constants here.  They may become configurable later,
but the plan explicitly defers that until there is evidence that per-user
tuning is worth the complexity.

Integration points (Stage 6a):
  - GoalInferenceService: PROPOSED vs auto-accept for inferred goals
  - PlanAction: block Tier-3 tasks from autonomous creation
  - ActDispatcherService: gate external actions by tier before execution
  - PersistentTaskWorker: verify tier before each background cycle

Both dependency services are lazy-imported inside methods to avoid circular
import issues — the same pattern used by FailureAnalysisService._get_llm()
and other services in this codebase.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)
LOG_PREFIX = "[EXEC GATE]"


class AutonomousExecutionGate:
    """
    Decides whether an action can be auto-executed based on consequence tier
    and domain confidence from accumulated memory.

    Reversibility is the hard line:
      - Tier 0 (Observe): Always auto-execute
      - Tier 1 (Organize): Auto-execute if domain_confidence >= 0.5
      - Tier 2 (Act): Auto-execute if domain_confidence >= 0.75
      - Tier 3 (Commit): Never auto-execute

    Usage::

        gate = AutonomousExecutionGate()
        result = gate.evaluate("search the web for Python tutorials", domain="technology")
        if result["auto_execute"]:
            run_action()
        else:
            ask_user()
    """

    # Confidence thresholds keyed by consequence tier.
    # float('inf') for Tier 3 means confidence can never be high enough.
    TIER_THRESHOLDS = {
        0: 0.0,            # Observe: always auto-execute
        1: 0.50,           # Organize: moderate confidence required
        2: 0.75,           # Act: high confidence required
        3: float('inf'),   # Commit: never auto-execute
    }

    _TIER_NAMES = {0: "observe", 1: "organize", 2: "act", 3: "commit"}

    def __init__(self):
        # Lazy-initialised service handles — populated on first use.
        self._consequence_service = None
        self._confidence_service = None

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        action_description: str,
        domain: str,
        account_id: int = 1,
    ) -> dict:
        """
        Evaluate whether an action can be auto-executed.

        Calls ConsequenceClassifierService to determine the action's consequence
        tier, then calls DomainConfidenceService to compute confidence in the
        given domain, and finally applies the tier threshold.

        Every call is logged at INFO level as the audit trail for autonomous
        action decisions.

        Args:
            action_description: Natural-language description of the proposed action.
            domain: Semantic domain (e.g. "scheduling", "finance", "health").
            account_id: Account to compute domain confidence for (default: 1).

        Returns:
            dict with keys:
              - ``auto_execute`` (bool): whether to proceed without asking
              - ``consequence_tier`` (int): 0–3
              - ``consequence_name`` (str): 'observe', 'organize', 'act', or 'commit'
              - ``domain`` (str): the domain used for confidence lookup
              - ``domain_confidence`` (float): 0.0–1.0
              - ``threshold`` (float): the confidence threshold for this tier
              - ``reasoning`` (str): human-readable explanation
        """
        # Classify consequence tier
        consequence_result = self._get_consequence_service().classify(action_description)
        tier: int = consequence_result.get("tier", 2)  # default ACT if something goes wrong
        tier_name: str = self._TIER_NAMES.get(tier, "act")

        # Compute domain confidence
        domain_confidence: float = self._get_domain_confidence(domain, account_id)

        # Apply decision logic
        auto_execute = self.should_auto_execute(tier, domain_confidence)
        threshold = self.TIER_THRESHOLDS.get(tier, float('inf'))

        # Build human-readable reasoning
        reasoning = self._build_reasoning(tier, tier_name, domain, domain_confidence,
                                          threshold, auto_execute)

        result = {
            "auto_execute": auto_execute,
            "consequence_tier": tier,
            "consequence_name": tier_name,
            "domain": domain,
            "domain_confidence": domain_confidence,
            "threshold": threshold,
            "reasoning": reasoning,
        }

        logger.info(
            f"{LOG_PREFIX} Decision: auto_execute={auto_execute} | "
            f"tier={tier_name}({tier}) | domain={domain!r} | "
            f"confidence={domain_confidence:.3f} (threshold={threshold}) | "
            f"action={action_description!r:.100}"
        )

        return result

    def should_auto_execute(self, consequence_tier: int, domain_confidence: float) -> bool:
        """
        Pure logic check — no service calls.

        For use when the consequence tier and domain confidence are already
        known (e.g. inside a tight loop or in tests).

        Args:
            consequence_tier: 0–3 consequence tier integer.
            domain_confidence: 0.0–1.0 confidence score for the domain.

        Returns:
            True if the action should auto-execute, False if user confirmation
            is required.
        """
        # Tier 0 (Observe) is always safe — zero side-effects, always reversible.
        # Check explicitly so that even nonsense confidence values don't block it.
        if consequence_tier == 0:
            return True

        # Tier 3 (Commit) is never safe to auto-execute, no matter the confidence.
        if consequence_tier == 3:
            return False

        threshold = self.TIER_THRESHOLDS.get(consequence_tier)

        if threshold is None:
            # Unknown tier — treat as most restrictive (require manual approval).
            logger.warning(
                f"{LOG_PREFIX} Unknown consequence tier {consequence_tier!r} — "
                "defaulting to manual approval"
            )
            return False

        return domain_confidence >= threshold

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_consequence_service(self):
        """Lazily initialise and return the ConsequenceClassifierService."""
        if self._consequence_service is None:
            from services.consequence_classifier_service import (
                get_consequence_classifier_service,
            )
            self._consequence_service = get_consequence_classifier_service()
        return self._consequence_service

    def _get_domain_confidence(self, domain: str, account_id: int) -> float:
        """
        Lazily initialise DomainConfidenceService and compute domain confidence.

        Returns 0.0 if the service is unavailable (Component 2 not yet deployed)
        so the gate always defaults to the cautious path rather than crashing.
        """
        try:
            if self._confidence_service is None:
                from services.domain_confidence_service import DomainConfidenceService
                self._confidence_service = DomainConfidenceService()
            return float(self._confidence_service.compute_domain_confidence(
                domain, account_id
            ))
        except ImportError:
            # DomainConfidenceService (Component 2) not yet deployed.
            # Return 0.0 — the gate will require manual approval for Tier 1+.
            logger.debug(
                f"{LOG_PREFIX} DomainConfidenceService not available — "
                "confidence=0.0 (Component 2 not deployed)"
            )
            return 0.0
        except Exception as e:
            logger.warning(
                f"{LOG_PREFIX} DomainConfidenceService error for domain {domain!r}: {e} — "
                "falling back to confidence=0.0"
            )
            return 0.0

    @staticmethod
    def _build_reasoning(
        tier: int,
        tier_name: str,
        domain: str,
        domain_confidence: float,
        threshold: float,
        auto_execute: bool,
    ) -> str:
        """Build a human-readable explanation for the gate decision."""
        if tier == 0:
            return (
                f"Tier 0 (observe) — no side-effects; always safe to auto-execute."
            )
        if tier == 3:
            return (
                f"Tier 3 (commit) — irreversible; always requires explicit user approval."
            )
        if auto_execute:
            return (
                f"Tier {tier} ({tier_name}) — domain confidence {domain_confidence:.2f} "
                f">= threshold {threshold:.2f} for domain '{domain}'; auto-executing."
            )
        return (
            f"Tier {tier} ({tier_name}) — domain confidence {domain_confidence:.2f} "
            f"< threshold {threshold:.2f} for domain '{domain}'; requesting user approval."
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

import threading

_instance: Optional[AutonomousExecutionGate] = None
_instance_lock = threading.Lock()


def get_autonomous_execution_gate() -> AutonomousExecutionGate:
    """Get or create the singleton AutonomousExecutionGate."""
    global _instance
    if _instance is not None:
        return _instance

    with _instance_lock:
        if _instance is not None:
            return _instance
        _instance = AutonomousExecutionGate()
        return _instance
