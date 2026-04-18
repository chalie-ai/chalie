"""
Unit tests for AutonomousExecutionGate.

Tests cover:
  - should_auto_execute() pure logic across all tiers
  - evaluate() with mocked consequence classifier
  - Output shape and type correctness
  - Edge cases: unknown tier, negative tier
"""

import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit


@pytest.fixture
def gate():
    from services.autonomous_execution_gate import AutonomousExecutionGate
    return AutonomousExecutionGate()


class TestShouldAutoExecute:
    def test_tier0_auto_executes(self, gate):
        assert gate.should_auto_execute(0) is True

    def test_tier1_requires_approval(self, gate):
        assert gate.should_auto_execute(1) is False

    def test_tier2_requires_approval(self, gate):
        assert gate.should_auto_execute(2) is False

    def test_tier3_requires_approval(self, gate):
        assert gate.should_auto_execute(3) is False

    def test_unknown_tier_requires_approval(self, gate):
        assert gate.should_auto_execute(99) is False

    def test_negative_tier_requires_approval(self, gate):
        assert gate.should_auto_execute(-1) is False


class TestEvaluateWithMocks:
    def _make_gate_with_mock(self, tier, tier_name):
        from services.autonomous_execution_gate import AutonomousExecutionGate
        gate = AutonomousExecutionGate()
        mock_consequence_svc = MagicMock()
        mock_consequence_svc.classify.return_value = {
            "tier": tier,
            "tier_name": tier_name,
            "confidence": 0.95,
            "scores": {},
            "method": "rule_based",
        }
        gate._consequence_service = mock_consequence_svc
        return gate

    def test_observe_auto_executes(self):
        gate = self._make_gate_with_mock(tier=0, tier_name="observe")
        result = gate.evaluate("search the web", domain="technology")
        assert result["auto_execute"] is True

    def test_organize_requires_approval(self):
        gate = self._make_gate_with_mock(tier=1, tier_name="organize")
        result = gate.evaluate("save this note", domain="productivity")
        assert result["auto_execute"] is False

    def test_act_requires_approval(self):
        gate = self._make_gate_with_mock(tier=2, tier_name="act")
        result = gate.evaluate("send email", domain="communication")
        assert result["auto_execute"] is False

    def test_commit_requires_approval(self):
        gate = self._make_gate_with_mock(tier=3, tier_name="commit")
        result = gate.evaluate("delete files", domain="files")
        assert result["auto_execute"] is False

    def test_evaluate_calls_consequence_service(self):
        gate = self._make_gate_with_mock(tier=0, tier_name="observe")
        gate.evaluate("look up the weather", domain="general")
        gate._consequence_service.classify.assert_called_once_with("look up the weather")


class TestEvaluateOutputShape:
    def _make_gate(self, tier=0, tier_name="observe"):
        from services.autonomous_execution_gate import AutonomousExecutionGate
        gate = AutonomousExecutionGate()
        mock_csvc = MagicMock()
        mock_csvc.classify.return_value = {
            "tier": tier,
            "tier_name": tier_name,
            "confidence": 0.95,
            "scores": {},
            "method": "rule_based",
        }
        gate._consequence_service = mock_csvc
        return gate

    def test_all_required_keys_present(self):
        gate = self._make_gate()
        result = gate.evaluate("search", domain="general")
        required_keys = {
            "auto_execute",
            "consequence_tier",
            "consequence_name",
            "domain",
            "reasoning",
        }
        assert required_keys.issubset(result.keys())

    def test_auto_execute_is_bool(self):
        gate = self._make_gate()
        result = gate.evaluate("search", domain="general")
        assert isinstance(result["auto_execute"], bool)

    def test_consequence_tier_is_int(self):
        gate = self._make_gate()
        result = gate.evaluate("search", domain="general")
        assert isinstance(result["consequence_tier"], int)

    def test_consequence_name_is_str(self):
        gate = self._make_gate()
        result = gate.evaluate("search", domain="general")
        assert isinstance(result["consequence_name"], str)

    def test_domain_passthrough(self):
        gate = self._make_gate()
        result = gate.evaluate("do something", domain="scheduling")
        assert result["domain"] == "scheduling"

    def test_reasoning_is_non_empty_str(self):
        gate = self._make_gate()
        result = gate.evaluate("search", domain="general")
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        from services.autonomous_execution_gate import get_autonomous_execution_gate
        a = get_autonomous_execution_gate()
        b = get_autonomous_execution_gate()
        assert a is b

    def test_singleton_is_correct_type(self):
        from services.autonomous_execution_gate import (
            AutonomousExecutionGate,
            get_autonomous_execution_gate,
        )
        gate = get_autonomous_execution_gate()
        assert isinstance(gate, AutonomousExecutionGate)
