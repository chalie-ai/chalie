"""Tests for RewardEvaluatorService — pure logic, no mocks needed."""

import pytest
from services.reward_evaluator_service import RewardEvaluatorService


pytestmark = pytest.mark.unit


class TestRewardEvaluator:

    def setup_method(self):
        self.evaluator = RewardEvaluatorService()

    # ── Explicit feedback ─────────────────────────────────────────

    def test_positive_feedback_reward(self):
        reward = self.evaluator.evaluate_user_behavior("thanks for the help")
        assert reward == 0.5

    def test_negative_feedback_reward(self):
        reward = self.evaluator.evaluate_user_behavior("that's wrong, try again")
        assert reward == -0.5

    def test_no_feedback_neutral(self):
        """Neutral text with no previous input → engaged reward (continuing conversation)."""
        reward = self.evaluator.evaluate_user_behavior("Let me explain the architecture")
        assert reward == 0.3  # USER_ENGAGED_REWARD

    # ── Repetition detection ──────────────────────────────────────

    def test_repetition_penalty(self):
        """>60% word overlap should trigger repetition penalty."""
        reward = self.evaluator.evaluate_user_behavior(
            current_input="what is the capital of france",
            previous_input="what is the capital of france please",
        )
        assert reward == -0.2  # USER_REPEATED_PENALTY

    def test_no_repetition_different_text(self):
        """Different text should not trigger repetition."""
        reward = self.evaluator.evaluate_user_behavior(
            current_input="tell me about python programming",
            previous_input="what is the capital of france",
        )
        assert reward == 0.3  # USER_ENGAGED_REWARD (no feedback, no repetition)

    # ── Action outcome ────────────────────────────────────────────

    def test_action_success_reward(self):
        reward = self.evaluator.evaluate_action_outcome({'status': 'success'})
        assert reward == 0.5

    def test_action_failure_penalty(self):
        reward = self.evaluator.evaluate_action_outcome({'status': 'error'})
        assert reward == -0.3

    def test_action_timeout_penalty(self):
        reward = self.evaluator.evaluate_action_outcome({'status': 'timeout'})
        assert reward == -0.2

    # ── Composite reward ──────────────────────────────────────────

    def test_composite_reward_averages(self):
        """Multiple signals should be averaged."""
        reward = self.evaluator.calculate_composite_reward(
            action_results=[
                {'status': 'success'},  # +0.5
                {'status': 'error'},    # -0.3
            ],
            current_input="thanks",  # +0.5 (positive feedback)
        )
        # Average of [0.5, -0.3, 0.5] = 0.2333...
        assert abs(reward - (0.5 + -0.3 + 0.5) / 3) < 0.001

    def test_reward_clamped_range(self):
        """Result must always be in [-1.0, 1.0]."""
        # All positive signals
        reward_pos = self.evaluator.calculate_composite_reward(
            action_results=[{'status': 'success'}] * 5,
            current_input="thanks perfect awesome great",
        )
        assert -1.0 <= reward_pos <= 1.0

        # All negative signals
        reward_neg = self.evaluator.calculate_composite_reward(
            action_results=[{'status': 'error'}] * 5,
            current_input="wrong incorrect try again",
        )
        assert -1.0 <= reward_neg <= 1.0
