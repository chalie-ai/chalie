"""
Tests for tiered reward assignment in invocation_tracker._compute_reward().

Verifies that external failures (rate limits, timeouts) receive an attenuated
penalty (-0.05) rather than the full internal-failure penalty (-0.2), preventing
unjust weight degradation for tools that failed due to circumstances outside their control.
"""
import pytest
from services.invocation_tracker import _compute_reward


class TestTieredRewards:
    """Unit tests for _compute_reward reward classification."""

    @pytest.mark.unit
    def test_success_reward(self):
        """Successful invocation of an unknown tool records +0.3 reward."""
        reward = _compute_reward("my_tool", True, None)
        assert reward == pytest.approx(0.3), f"Expected 0.3 for success, got {reward}"

    @pytest.mark.unit
    def test_internal_failure_reward(self):
        """Internal tool failure records -0.2 reward."""
        reward = _compute_reward("my_tool", False, "internal")
        assert reward == pytest.approx(-0.2), f"Expected -0.2 for internal failure, got {reward}"

    @pytest.mark.unit
    def test_external_failure_reward_attenuated(self):
        """External failure (rate limit / network) records attenuated -0.05 reward."""
        reward = _compute_reward("my_tool", False, "external")
        assert reward == pytest.approx(-0.05), f"Expected -0.05 for external failure, got {reward}"

    @pytest.mark.unit
    def test_external_penalty_smaller_than_internal(self):
        """External failure penalty must be strictly smaller in magnitude than internal."""
        external_reward = _compute_reward("my_tool", False, "external")
        internal_reward = _compute_reward("my_tool", False, "internal")
        assert external_reward > internal_reward, (
            f"External penalty ({external_reward}) should be less severe than internal ({internal_reward})"
        )

    @pytest.mark.unit
    def test_unknown_failure_class_defaults_to_internal(self):
        """failure_class=None (legacy callers) defaults to internal -0.2 penalty."""
        reward = _compute_reward("my_tool", False, None)
        assert reward == pytest.approx(-0.2), f"Default failure should apply -0.2, got {reward}"

    @pytest.mark.unit
    def test_known_skill_uses_custom_reward(self):
        """Known skills from _REWARD_MAP use their own reward function."""
        reward = _compute_reward("recall", True, None, result="Found 3 matches")
        assert reward == pytest.approx(0.3), f"Expected 0.3 for successful recall, got {reward}"

    @pytest.mark.unit
    def test_recall_no_matches_penalized(self):
        """recall with 'no matches' gets a negative reward."""
        reward = _compute_reward("recall", True, None, result="No matches found")
        assert reward == pytest.approx(-0.1), f"Expected -0.1 for recall with no matches, got {reward}"
