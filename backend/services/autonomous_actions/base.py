"""
AutonomousAction — Base class for all autonomous actions.

Every action the drift engine can take implements this interface.
The decision router iterates registered actions, collects scores from
eligible ones, and picks the highest scorer. NOTHING always returns (0, True).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ThoughtContext:
    """Context passed to each action's should_execute and execute methods.

    Extended with event fields for event-bridge-driven autonomous actions.
    When thought_type='event', the event_* fields carry context change data.
    """
    thought_type: str           # 'reflection', 'question', 'hypothesis', 'event'
    thought_content: str        # The synthesized thought text
    activation_energy: float    # Max activation score from spreading activation
    seed_concept: str           # Name of the seed concept
    seed_topic: str             # Topic associated with the seed
    thought_embedding: Optional[list] = None  # Embedding vector of the thought
    drift_gist_id: Optional[str] = None       # ID of the stored drift gist
    drift_gist_ttl: int = 1800                # Original TTL of the drift gist (30min)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result returned by an action's execute method."""
    action_name: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class AutonomousAction(ABC):
    """
    Base class for autonomous actions.

    Every action implements:
      - should_execute(thought, context) -> (score, eligible)
      - execute(thought, context) -> ActionResult
      - on_outcome(result, user_feedback) -> None
    """

    def __init__(self, name: str, enabled: bool = True, priority: int = 0):
        self.name = name
        self.enabled = enabled
        self.priority = priority  # Higher = wins ties

    @abstractmethod
    def should_execute(self, thought: ThoughtContext) -> tuple:
        """
        Evaluate whether this action should run.

        Returns:
            (score: float, eligible: bool)
            Score competes with other actions. Eligible=False means this
            action can't run right now (gated by timing, engagement, etc.).
        """

    @abstractmethod
    def execute(self, thought: ThoughtContext) -> ActionResult:
        """Perform the action. Returns result for logging/feedback."""

    def on_outcome(self, result: ActionResult, user_feedback: Optional[Dict] = None) -> None:
        """Learn from the outcome. Called when feedback arrives. Optional override."""
        pass
