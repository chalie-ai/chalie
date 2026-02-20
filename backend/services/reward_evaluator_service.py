"""
Reward Evaluator Service - Outcome-based reward scoring for actions.

Derives reward signals from:
- Immediate action success/failure
- Downstream user behavior (engagement vs repetition vs topic switch)
- Explicit feedback detection
"""

import logging
from typing import Dict, Any, Optional


class RewardEvaluatorService:
    """Evaluates reward signals from action outcomes and user behavior."""

    # Reward constants
    ACTION_SUCCESS_REWARD = 0.5
    ACTION_FAILURE_PENALTY = -0.3
    ACTION_TIMEOUT_PENALTY = -0.2

    # Behavioral signals
    USER_ENGAGED_REWARD = 0.3        # User continues same topic with new input
    USER_REPEATED_PENALTY = -0.2     # User repeats similar question (confusion signal)
    USER_TOPIC_SWITCH_NEUTRAL = 0.0  # Topic switch is neutral
    EXPLICIT_POSITIVE_REWARD = 0.5   # Detected positive feedback
    EXPLICIT_NEGATIVE_PENALTY = -0.5 # Detected negative feedback

    def __init__(self):
        """Initialize reward evaluator."""
        # Positive/negative feedback markers
        self._positive_markers = [
            'thanks', 'thank you', 'great', 'perfect', 'awesome',
            'exactly', 'that works', 'correct', 'good', 'nice',
            'helpful', 'got it', 'understood'
        ]
        self._negative_markers = [
            'wrong', 'incorrect', 'no that', "that's not", "doesn't work",
            'confused', 'not what i', 'try again', 'still not',
            'misunderstood', 'not helpful'
        ]

    def evaluate_action_outcome(self, action_result: Dict[str, Any]) -> float:
        """
        Evaluate reward from immediate action result.

        Args:
            action_result: Dict with 'status' key (success/error/timeout)

        Returns:
            Reward value (-1.0 to 1.0)
        """
        status = action_result.get('status', 'error')

        if status == 'success':
            return self.ACTION_SUCCESS_REWARD
        elif status == 'timeout':
            return self.ACTION_TIMEOUT_PENALTY
        else:
            return self.ACTION_FAILURE_PENALTY

    def evaluate_user_behavior(
        self,
        current_input: str,
        previous_input: str = None,
        previous_topic: str = None,
        current_topic: str = None
    ) -> float:
        """
        Evaluate reward from downstream user behavior.

        Args:
            current_input: Current user message
            previous_input: Previous user message (if available)
            previous_topic: Previous conversation topic
            current_topic: Current conversation topic

        Returns:
            Reward value (-1.0 to 1.0)
        """
        reward = 0.0

        # Check for explicit feedback
        feedback_reward = self._detect_feedback(current_input)
        if feedback_reward != 0.0:
            return feedback_reward

        # Check for topic switch
        if previous_topic and current_topic and previous_topic != current_topic:
            return self.USER_TOPIC_SWITCH_NEUTRAL

        # Check for repetition (confusion signal)
        if previous_input and self._is_repetition(current_input, previous_input):
            return self.USER_REPEATED_PENALTY

        # Default: engaged (continuing same topic with new content)
        return self.USER_ENGAGED_REWARD

    def _detect_feedback(self, text: str) -> float:
        """
        Detect explicit positive/negative feedback in user text.

        Args:
            text: User message text

        Returns:
            Reward value or 0.0 if no feedback detected
        """
        text_lower = text.lower().strip()

        for marker in self._positive_markers:
            if marker in text_lower:
                logging.debug(f"[REWARD] Detected positive feedback: '{marker}'")
                return self.EXPLICIT_POSITIVE_REWARD

        for marker in self._negative_markers:
            if marker in text_lower:
                logging.debug(f"[REWARD] Detected negative feedback: '{marker}'")
                return self.EXPLICIT_NEGATIVE_PENALTY

        return 0.0

    def _is_repetition(self, current: str, previous: str) -> bool:
        """
        Check if current input is a repetition of previous input.

        Uses simple word overlap ratio.

        Args:
            current: Current user message
            previous: Previous user message

        Returns:
            True if messages are similar enough to indicate repetition
        """
        current_words = set(current.lower().split())
        previous_words = set(previous.lower().split())

        if not current_words or not previous_words:
            return False

        overlap = current_words & previous_words
        union = current_words | previous_words

        # >60% overlap = repetition
        ratio = len(overlap) / len(union) if union else 0
        return ratio > 0.6

    def calculate_composite_reward(
        self,
        action_results: list = None,
        current_input: str = None,
        previous_input: str = None,
        previous_topic: str = None,
        current_topic: str = None
    ) -> float:
        """
        Calculate composite reward from all signals.

        Args:
            action_results: List of action result dicts
            current_input: Current user message
            previous_input: Previous user message
            previous_topic: Previous topic
            current_topic: Current topic

        Returns:
            Composite reward value (-1.0 to 1.0)
        """
        rewards = []

        # Action outcome rewards
        if action_results:
            for result in action_results:
                rewards.append(self.evaluate_action_outcome(result))

        # User behavior reward
        if current_input:
            behavior_reward = self.evaluate_user_behavior(
                current_input, previous_input, previous_topic, current_topic
            )
            rewards.append(behavior_reward)

        if not rewards:
            return 0.0

        # Average all reward signals
        composite = sum(rewards) / len(rewards)
        return max(-1.0, min(1.0, composite))
