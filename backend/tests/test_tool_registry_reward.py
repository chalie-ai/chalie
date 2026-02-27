"""
Tests for tiered reward assignment in ToolRegistryService._log_outcome().

Verifies that external failures (rate limits, timeouts) receive an attenuated
penalty (-0.05) rather than the full internal-failure penalty (-0.2), preventing
unjust weight degradation for tools that failed due to circumstances outside their control.
"""
import pytest
from unittest.mock import MagicMock, patch, call


class TestToolRegistryTieredRewards:
    """Unit tests for _log_outcome reward classification."""

    def _make_service(self):
        """Return a ToolRegistryService with tools disabled (skips Docker setup)."""
        with patch("services.tool_registry_service.ToolRegistryService._load_tools"):
            from services.tool_registry_service import ToolRegistryService
            svc = ToolRegistryService.__new__(ToolRegistryService)
            svc._enabled = True
            svc.tools = {}
            svc.MAX_OUTPUT_CHARS = 8000
            svc.STRIP_PATTERNS = []
            return svc

    @pytest.mark.unit
    def test_success_reward(self):
        """Successful invocation records +0.3 reward."""
        svc = self._make_service()
        mock_proc = MagicMock()

        with patch("services.tool_registry_service.ProceduralMemoryService", return_value=mock_proc), \
             patch("services.tool_registry_service.get_shared_db_service"):
            svc._log_outcome("my_tool", True, "test_topic", 500)

        mock_proc.record_action_outcome.assert_called_once()
        _, args, kwargs = mock_proc.record_action_outcome.mock_calls[0]
        reward = args[2] if len(args) > 2 else kwargs.get("reward")
        assert reward == pytest.approx(0.3), f"Expected 0.3 for success, got {reward}"

    @pytest.mark.unit
    def test_internal_failure_reward(self):
        """Internal tool failure (container crash) records -0.2 reward."""
        svc = self._make_service()
        mock_proc = MagicMock()

        with patch("services.tool_registry_service.ProceduralMemoryService", return_value=mock_proc), \
             patch("services.tool_registry_service.get_shared_db_service"):
            svc._log_outcome("my_tool", False, "test_topic", 500, failure_class="internal")

        mock_proc.record_action_outcome.assert_called_once()
        _, args, kwargs = mock_proc.record_action_outcome.mock_calls[0]
        reward = args[2] if len(args) > 2 else kwargs.get("reward")
        assert reward == pytest.approx(-0.2), f"Expected -0.2 for internal failure, got {reward}"

    @pytest.mark.unit
    def test_external_failure_reward_attenuated(self):
        """External failure (rate limit / network) records attenuated -0.05 reward."""
        svc = self._make_service()
        mock_proc = MagicMock()

        with patch("services.tool_registry_service.ProceduralMemoryService", return_value=mock_proc), \
             patch("services.tool_registry_service.get_shared_db_service"):
            svc._log_outcome("my_tool", False, "test_topic", 500, failure_class="external")

        mock_proc.record_action_outcome.assert_called_once()
        _, args, kwargs = mock_proc.record_action_outcome.mock_calls[0]
        reward = args[2] if len(args) > 2 else kwargs.get("reward")
        assert reward == pytest.approx(-0.05), f"Expected -0.05 for external failure, got {reward}"

    @pytest.mark.unit
    def test_external_penalty_smaller_than_internal(self):
        """External failure penalty must be strictly smaller in magnitude than internal."""
        svc = self._make_service()
        recorded = {}

        def capture(*args, **kwargs):
            recorded["reward"] = args[2] if len(args) > 2 else kwargs.get("reward")

        mock_proc = MagicMock()
        mock_proc.record_action_outcome.side_effect = capture

        with patch("services.tool_registry_service.ProceduralMemoryService", return_value=mock_proc), \
             patch("services.tool_registry_service.get_shared_db_service"):
            svc._log_outcome("my_tool", False, "test_topic", 500, failure_class="external")
        external_reward = recorded["reward"]

        with patch("services.tool_registry_service.ProceduralMemoryService", return_value=mock_proc), \
             patch("services.tool_registry_service.get_shared_db_service"):
            svc._log_outcome("my_tool", False, "test_topic", 500, failure_class="internal")
        internal_reward = recorded["reward"]

        assert external_reward > internal_reward, (
            f"External penalty ({external_reward}) should be less severe than internal ({internal_reward})"
        )

    @pytest.mark.unit
    def test_unknown_failure_class_defaults_to_internal(self):
        """failure_class=None (legacy callers) defaults to internal -0.2 penalty."""
        svc = self._make_service()
        mock_proc = MagicMock()

        with patch("services.tool_registry_service.ProceduralMemoryService", return_value=mock_proc), \
             patch("services.tool_registry_service.get_shared_db_service"):
            svc._log_outcome("my_tool", False, "test_topic", 500)

        mock_proc.record_action_outcome.assert_called_once()
        _, args, kwargs = mock_proc.record_action_outcome.mock_calls[0]
        reward = args[2] if len(args) > 2 else kwargs.get("reward")
        assert reward == pytest.approx(-0.2), f"Default failure should apply -0.2, got {reward}"

    @pytest.mark.unit
    def test_failure_class_passed_to_procedural_memory(self):
        """failure_class kwarg is forwarded to record_action_outcome for observability."""
        svc = self._make_service()
        mock_proc = MagicMock()

        with patch("services.tool_registry_service.ProceduralMemoryService", return_value=mock_proc), \
             patch("services.tool_registry_service.get_shared_db_service"):
            svc._log_outcome("my_tool", False, "test_topic", 500, failure_class="external")

        call_kwargs = mock_proc.record_action_outcome.call_args.kwargs
        assert call_kwargs.get("failure_class") == "external"
