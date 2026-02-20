"""
ActionDecisionRouter — Selects the best action for a drift thought.

Iterates registered actions, collects (score, eligible) from each,
picks the highest-scoring eligible action. Ties broken by priority.
"""

import logging
from typing import List, Optional, Dict, Any

from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ACTION ROUTER]"


class ActionDecisionRouter:
    """Routes drift thoughts to the highest-scoring eligible action."""

    def __init__(self, actions: Optional[List[AutonomousAction]] = None):
        self.actions: List[AutonomousAction] = actions or []

    def register(self, action: AutonomousAction):
        self.actions.append(action)
        logger.info(f"{LOG_PREFIX} Registered action: {action.name} (priority={action.priority})")

    def decide_and_execute(self, thought: ThoughtContext) -> ActionResult:
        """
        Evaluate all registered actions and execute the best one.

        Returns:
            ActionResult from the winning action.
        """
        if not self.actions:
            logger.warning(f"{LOG_PREFIX} No actions registered")
            return ActionResult(action_name='NOTHING', success=True)

        # Collect scores
        candidates = []
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
                    logger.debug(
                        f"{LOG_PREFIX} {action.name}: eligible=False"
                    )
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} {action.name}.should_execute() failed: {e}")

        if not candidates:
            logger.info(f"{LOG_PREFIX} No eligible actions, defaulting to NOTHING")
            return ActionResult(action_name='NOTHING', success=True)

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
            return result
        except Exception as e:
            logger.error(f"{LOG_PREFIX} {winner.name}.execute() failed: {e}", exc_info=True)
            return ActionResult(action_name=winner.name, success=False, details={'error': str(e)})

    def get_scores(self, thought: ThoughtContext) -> List[Dict[str, Any]]:
        """Get all action scores without executing. Useful for logging."""
        scores = []
        for action in self.actions:
            if not action.enabled:
                scores.append({'action': action.name, 'eligible': False, 'reason': 'disabled'})
                continue
            try:
                score, eligible = action.should_execute(thought)
                scores.append({'action': action.name, 'score': score, 'eligible': eligible})
            except Exception as e:
                scores.append({'action': action.name, 'eligible': False, 'reason': str(e)})
        return scores
