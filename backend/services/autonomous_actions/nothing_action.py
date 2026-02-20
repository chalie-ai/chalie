"""
NothingAction — The default fallback action.

Always eligible, score 0. Does nothing beyond what the drift engine
already does (store the gist in Redis for reactive surfacing).
"""

from .base import AutonomousAction, ActionResult, ThoughtContext


class NothingAction(AutonomousAction):
    """Default action: let the drift gist live in Redis as-is."""

    def __init__(self):
        super().__init__(name='NOTHING', enabled=True, priority=-1)

    def should_execute(self, thought: ThoughtContext) -> tuple:
        return (0.0, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        return ActionResult(action_name='NOTHING', success=True)
