"""
ActionDecisionRouter — Selects the best action for a drift thought.

Iterates registered actions, collects (score, eligible) from each,
picks the highest-scoring eligible action. Ties broken by priority.

All evaluation results (eligible and rejected) are captured and returned
so the caller can feed them into the memory pipeline.
"""

import logging
from typing import List, Optional, Dict, Any

from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ACTION ROUTER]"


class ActionDecisionRouter:
    """Routes drift thoughts to the highest-scoring eligible action."""

    def __init__(self, actions: Optional[List[AutonomousAction]] = None):
        """
        Initialize the router with an optional pre-populated list of actions.

        Args:
            actions: Initial list of AutonomousAction instances to register.
                Additional actions can be added later via :meth:`register`.
                Defaults to an empty list when not provided.
        """
        self.actions: List[AutonomousAction] = actions or []

    def register(self, action: AutonomousAction):
        """
        Register an AutonomousAction with the router.

        Appends the action to the internal list so it will be evaluated on
        subsequent calls to :meth:`decide_and_execute` and :meth:`get_scores`.

        Args:
            action: The AutonomousAction instance to register.
        """
        self.actions.append(action)
        logger.info(f"{LOG_PREFIX} Registered action: {action.name} (priority={action.priority})")

    def decide_and_execute(self, thought: ThoughtContext) -> ActionResult:
        """
        Evaluate all registered actions and execute the best one.

        Returns:
            ActionResult from the winning action, with 'gate_rejections'
            in details capturing why ineligible actions were rejected.
        """
        if not self.actions:
            logger.warning(f"{LOG_PREFIX} No actions registered")
            return ActionResult(action_name='NOTHING', success=True)

        # Collect scores and rejections
        candidates = []
        rejections = []

        for action in self.actions:
            if not action.enabled:
                continue

            try:
                score, eligible = action.should_execute(thought)
                if eligible:
                    candidates.append((action, score))
                    logger.debug(
                        f"{LOG_PREFIX} {action.name}: score={score:.3f}, eligible=True"
                    )
                else:
                    # Capture gate rejection details from the action
                    gate_result = getattr(action, 'last_gate_result', None) or {}
                    rejections.append({
                        'action': action.name,
                        'gate': gate_result.get('gate', 'unknown'),
                        'reason': gate_result.get('reason', 'ineligible'),
                    })
                    logger.debug(
                        f"{LOG_PREFIX} {action.name}: eligible=False "
                        f"(gate={gate_result.get('gate', '?')}, reason={gate_result.get('reason', '?')})"
                    )
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} {action.name}.should_execute() failed: {e}")
                rejections.append({
                    'action': action.name,
                    'gate': 'exception',
                    'reason': str(e)[:100],
                })

        if not candidates:
            logger.info(f"{LOG_PREFIX} No eligible actions, defaulting to NOTHING")
            result = ActionResult(action_name='NOTHING', success=True)
            result.details['gate_rejections'] = rejections
            return result

        # Sort by score descending, then priority descending for ties
        candidates.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
        winner, winning_score = candidates[0]

        logger.info(
            f"{LOG_PREFIX} Selected {winner.name} (score={winning_score:.3f}) "
            f"for thought: [{thought.thought_type}] {thought.thought_content[:60]}..."
        )

        # Execute
        try:
            result = winner.execute(thought)
            result.details['gate_rejections'] = rejections
            return result
        except Exception as e:
            logger.error(f"{LOG_PREFIX} {winner.name}.execute() failed: {e}", exc_info=True)
            result = ActionResult(action_name=winner.name, success=False, details={'error': str(e)})
            result.details['gate_rejections'] = rejections
            return result

    def get_scores(self, thought: ThoughtContext) -> List[Dict[str, Any]]:
        """Get all action scores without executing. Useful for logging."""
        scores = []
        for action in self.actions:
            if not action.enabled:
                scores.append({'action': action.name, 'eligible': False, 'reason': 'disabled'})
                continue
            try:
                score, eligible = action.should_execute(thought)
                entry = {'action': action.name, 'score': score, 'eligible': eligible}
                if not eligible:
                    gate_result = getattr(action, 'last_gate_result', None) or {}
                    entry['gate'] = gate_result.get('gate', 'unknown')
                    entry['reason'] = gate_result.get('reason', 'ineligible')
                scores.append(entry)
            except Exception as e:
                scores.append({'action': action.name, 'eligible': False, 'reason': str(e)})
        return scores
