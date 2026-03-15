"""
NothingAction — The default fallback action.

Always eligible, score 0. Does nothing beyond what the drift engine
already does (store the gist in MemoryStore for reactive surfacing).
"""

from .base import AutonomousAction, ActionResult, ThoughtContext


class NothingAction(AutonomousAction):
    """Default action: let the drift gist live in MemoryStore as-is."""

    def __init__(self):
        """Initialize NothingAction with lowest priority (-1) so it never wins a contested decision."""
        super().__init__(name='NOTHING', enabled=True, priority=-1)

    def should_execute(self, thought: ThoughtContext) -> tuple:
        """Always eligible; always scores zero.

        NothingAction is the guaranteed fallback — it scores 0 so that any
        other eligible action will beat it, but it remains eligible at all
        times so the drift engine always has a valid action to take.

        Args:
            thought: The current drift thought context (unused).

        Returns:
            tuple: ``(score: float, eligible: bool)`` — always ``(0.0, True)``.
        """
        return (0.0, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        """Execute the no-op action.

        The drift gist is already stored in MemoryStore by the drift engine
        before actions are evaluated. This method simply acknowledges
        completion with no additional side effects.

        Args:
            thought: The current drift thought context (unused).

        Returns:
            ActionResult: Always successful with ``action_name='NOTHING'``.
        """
        return ActionResult(action_name='NOTHING', success=True)
