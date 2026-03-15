"""
Unit tests for AutonomousExecutionGate.

Tests cover:
  - should_auto_execute() pure logic across all tier/confidence combinations
  - evaluate() with mocked consequence classifier + domain confidence service
  - Output shape and type correctness
  - Edge cases: unknown tier, negative confidence, confidence > 1.0
"""

import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def gate():
    """AutonomousExecutionGate with no services pre-loaded."""
    from services.autonomous_execution_gate import AutonomousExecutionGate
    return AutonomousExecutionGate()


@pytest.fixture
def mock_consequence_result():
    """Factory for a mock consequence classification result."""
    def _make(tier, tier_name, confidence=0.95):
        return {
            "tier": tier,
            "tier_name": tier_name,
            "confidence": confidence,
            "scores": {},
            "method": "rule_based",
        }
    return _make


# ── should_auto_execute() — pure logic ───────────────────────────────────────


class TestShouldAutoExecutePureLogic:
    """
    Exercises should_auto_execute() with no external service calls.

    All combinations from the spec plus boundary values.
    """

    def test_tier0_any_confidence_returns_true(self, gate):
        """Tier 0 (observe) — always auto-execute regardless of confidence."""
        assert gate.should_auto_execute(0, 0.0) is True

    def test_tier0_zero_confidence_returns_true(self, gate):
        """Tier 0 with zero confidence still auto-executes."""
        assert gate.should_auto_execute(0, 0.0) is True

    def test_tier0_high_confidence_returns_true(self, gate):
        """Tier 0 with 1.0 confidence auto-executes."""
        assert gate.should_auto_execute(0, 1.0) is True

    # Tier 1 (Organize) — threshold 0.50

    def test_tier1_low_confidence_returns_false(self, gate):
        """Tier 1 with 0.3 confidence — below 0.50 threshold."""
        assert gate.should_auto_execute(1, 0.3) is False

    def test_tier1_just_below_threshold_returns_false(self, gate):
        """Tier 1 with 0.499 confidence — strictly below threshold."""
        assert gate.should_auto_execute(1, 0.499) is False

    def test_tier1_at_threshold_returns_true(self, gate):
        """Tier 1 with exactly 0.5 confidence — meets threshold."""
        assert gate.should_auto_execute(1, 0.5) is True

    def test_tier1_above_threshold_returns_true(self, gate):
        """Tier 1 with 0.7 confidence — above threshold."""
        assert gate.should_auto_execute(1, 0.7) is True

    # Tier 2 (Act) — threshold 0.75

    def test_tier2_low_confidence_returns_false(self, gate):
        """Tier 2 with 0.3 confidence — well below 0.75 threshold."""
        assert gate.should_auto_execute(2, 0.3) is False

    def test_tier2_medium_confidence_returns_false(self, gate):
        """Tier 2 with 0.7 confidence — just below 0.75 threshold."""
        assert gate.should_auto_execute(2, 0.7) is False

    def test_tier2_just_below_threshold_returns_false(self, gate):
        """Tier 2 with 0.749 confidence — strictly below threshold."""
        assert gate.should_auto_execute(2, 0.749) is False

    def test_tier2_at_threshold_returns_true(self, gate):
        """Tier 2 with exactly 0.75 confidence — meets threshold."""
        assert gate.should_auto_execute(2, 0.75) is True

    def test_tier2_high_confidence_returns_true(self, gate):
        """Tier 2 with 0.9 confidence — above threshold."""
        assert gate.should_auto_execute(2, 0.9) is True

    # Tier 3 (Commit) — never

    def test_tier3_high_confidence_returns_false(self, gate):
        """Tier 3 (commit) — never auto-execute, even with 1.0 confidence."""
        assert gate.should_auto_execute(3, 1.0) is False

    def test_tier3_zero_confidence_returns_false(self, gate):
        """Tier 3 (commit) with zero confidence — never auto-execute."""
        assert gate.should_auto_execute(3, 0.0) is False

    def test_tier3_mid_confidence_returns_false(self, gate):
        """Tier 3 (commit) with 0.5 confidence — never auto-execute."""
        assert gate.should_auto_execute(3, 0.5) is False


# ── should_auto_execute() — edge cases ───────────────────────────────────────


class TestShouldAutoExecuteEdgeCases:
    """Edge cases that fall outside the normal operating range."""

    def test_unknown_tier_returns_false(self, gate):
        """Unknown tier is treated as most restrictive — require manual approval."""
        assert gate.should_auto_execute(99, 1.0) is False

    def test_unknown_negative_tier_returns_false(self, gate):
        """Negative tier value is treated as unknown — require manual approval."""
        assert gate.should_auto_execute(-1, 1.0) is False

    def test_negative_confidence_tier1_returns_false(self, gate):
        """Negative confidence is below any threshold."""
        assert gate.should_auto_execute(1, -0.5) is False

    def test_negative_confidence_tier0_returns_true(self, gate):
        """Tier 0 always executes even with nonsense confidence value."""
        assert gate.should_auto_execute(0, -1.0) is True

    def test_confidence_above_1_tier2_returns_true(self, gate):
        """Confidence > 1.0 (caller error) still produces a valid decision for Tier 2."""
        assert gate.should_auto_execute(2, 1.5) is True

    def test_confidence_above_1_tier3_returns_false(self, gate):
        """Confidence > 1.0 still can't unlock Tier 3."""
        assert gate.should_auto_execute(3, 999.0) is False


# ── evaluate() — full integration with mocked services ───────────────────────


class TestEvaluateWithMocks:
    """
    Tests for evaluate() using mocked consequence + confidence services.

    The consequence classifier and domain confidence service are both mocked
    so these tests remain unit tests with no SQLite or ONNX dependencies.
    """

    def _make_gate_with_mocks(self, tier, tier_name, domain_confidence):
        """
        Helper: return an AutonomousExecutionGate whose internal service
        references are replaced with pre-configured MagicMocks.
        """
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

        mock_confidence_svc = MagicMock()
        mock_confidence_svc.compute_domain_confidence.return_value = domain_confidence
        gate._confidence_service = mock_confidence_svc

        return gate

    def test_observe_action_always_auto_executes(self):
        """Tier 0 with any confidence auto-executes."""
        gate = self._make_gate_with_mocks(tier=0, tier_name="observe",
                                          domain_confidence=0.0)
        result = gate.evaluate("search the web for Python tutorials",
                               domain="technology")
        assert result["auto_execute"] is True

    def test_organize_low_confidence_requires_approval(self):
        """Tier 1 + confidence 0.3 → manual approval."""
        gate = self._make_gate_with_mocks(tier=1, tier_name="organize",
                                          domain_confidence=0.3)
        result = gate.evaluate("save this note", domain="productivity")
        assert result["auto_execute"] is False

    def test_organize_sufficient_confidence_auto_executes(self):
        """Tier 1 + confidence 0.5 → auto-execute."""
        gate = self._make_gate_with_mocks(tier=1, tier_name="organize",
                                          domain_confidence=0.5)
        result = gate.evaluate("add item to list", domain="productivity")
        assert result["auto_execute"] is True

    def test_act_insufficient_confidence_requires_approval(self):
        """Tier 2 + confidence 0.7 → manual approval."""
        gate = self._make_gate_with_mocks(tier=2, tier_name="act",
                                          domain_confidence=0.7)
        result = gate.evaluate("send email to Alice", domain="communication")
        assert result["auto_execute"] is False

    def test_act_sufficient_confidence_auto_executes(self):
        """Tier 2 + confidence 0.75 → auto-execute."""
        gate = self._make_gate_with_mocks(tier=2, tier_name="act",
                                          domain_confidence=0.75)
        result = gate.evaluate("schedule reminder for tomorrow", domain="scheduling")
        assert result["auto_execute"] is True

    def test_commit_never_auto_executes(self):
        """Tier 3 + confidence 1.0 → always manual approval."""
        gate = self._make_gate_with_mocks(tier=3, tier_name="commit",
                                          domain_confidence=1.0)
        result = gate.evaluate("delete all files", domain="files")
        assert result["auto_execute"] is False

    def test_evaluate_calls_consequence_service(self):
        """evaluate() must call classify() on the consequence service."""
        gate = self._make_gate_with_mocks(tier=0, tier_name="observe",
                                          domain_confidence=0.9)
        gate.evaluate("look up the weather", domain="general")
        gate._consequence_service.classify.assert_called_once_with(
            "look up the weather"
        )

    def test_evaluate_calls_confidence_service_with_correct_args(self):
        """evaluate() must pass domain and account_id to domain confidence service."""
        gate = self._make_gate_with_mocks(tier=1, tier_name="organize",
                                          domain_confidence=0.8)
        gate.evaluate("memorize this fact", domain="general", account_id=42)
        gate._confidence_service.compute_domain_confidence.assert_called_once_with(
            "general", 42
        )


# ── evaluate() — output shape and types ──────────────────────────────────────


class TestEvaluateOutputShape:
    """Verifies the contract of the evaluate() return dict."""

    def _make_gate(self, tier=0, tier_name="observe", domain_confidence=0.5):
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
        mock_dsvc = MagicMock()
        mock_dsvc.compute_domain_confidence.return_value = domain_confidence
        gate._confidence_service = mock_dsvc
        return gate

    def test_all_required_keys_present(self):
        """All seven required keys are present in the result."""
        gate = self._make_gate()
        result = gate.evaluate("search for something", domain="general")

        required_keys = {
            "auto_execute",
            "consequence_tier",
            "consequence_name",
            "domain",
            "domain_confidence",
            "threshold",
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
        """The domain field echoes back the domain argument."""
        gate = self._make_gate()
        result = gate.evaluate("do something", domain="scheduling")
        assert result["domain"] == "scheduling"

    def test_domain_confidence_is_float(self):
        gate = self._make_gate(domain_confidence=0.65)
        result = gate.evaluate("search", domain="general")
        assert isinstance(result["domain_confidence"], float)

    def test_domain_confidence_value_matches(self):
        gate = self._make_gate(domain_confidence=0.65)
        result = gate.evaluate("search", domain="general")
        assert abs(result["domain_confidence"] - 0.65) < 1e-9

    def test_reasoning_is_non_empty_str(self):
        gate = self._make_gate()
        result = gate.evaluate("search", domain="general")
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0

    def test_threshold_matches_tier_constant(self):
        """Threshold in output must equal the TIER_THRESHOLDS constant for the tier."""
        from services.autonomous_execution_gate import AutonomousExecutionGate
        gate = self._make_gate(tier=2, tier_name="act")
        result = gate.evaluate("send email", domain="communication")
        assert result["threshold"] == AutonomousExecutionGate.TIER_THRESHOLDS[2]


# ── evaluate() — domain confidence service not available ─────────────────────


class TestEvaluateDomainConfidenceFallback:
    """
    evaluate() must degrade gracefully when DomainConfidenceService
    (Component 2) is not yet deployed or raises an error.
    """

    def test_import_error_falls_back_to_zero(self):
        """ImportError from DomainConfidenceService results in confidence=0.0."""
        from services.autonomous_execution_gate import AutonomousExecutionGate
        gate = AutonomousExecutionGate()

        mock_csvc = MagicMock()
        mock_csvc.classify.return_value = {
            "tier": 1, "tier_name": "organize", "confidence": 0.9,
            "scores": {}, "method": "rule_based",
        }
        gate._consequence_service = mock_csvc
        # _confidence_service stays None; trigger the lazy-import path

        with patch("builtins.__import__",
                   side_effect=lambda name, *a, **kw: (
                       __import__(name, *a, **kw)
                       if "domain_confidence" not in name
                       else (_ for _ in ()).throw(ImportError("not installed"))
                   )):
            # Resetting cached service so lazy path is re-entered
            gate._confidence_service = None
            confidence = gate._get_domain_confidence("scheduling", 1)

        assert confidence == 0.0

    def test_runtime_error_falls_back_to_zero(self):
        """Exception from compute_domain_confidence returns 0.0 without crashing."""
        from services.autonomous_execution_gate import AutonomousExecutionGate
        gate = AutonomousExecutionGate()

        mock_dsvc = MagicMock()
        mock_dsvc.compute_domain_confidence.side_effect = RuntimeError("db error")
        gate._confidence_service = mock_dsvc

        confidence = gate._get_domain_confidence("scheduling", 1)
        assert confidence == 0.0

    def test_zero_confidence_tier1_requires_approval(self):
        """With confidence=0.0 from fallback, Tier 1 requires user approval."""
        from services.autonomous_execution_gate import AutonomousExecutionGate
        gate = AutonomousExecutionGate()

        mock_csvc = MagicMock()
        mock_csvc.classify.return_value = {
            "tier": 1, "tier_name": "organize", "confidence": 0.9,
            "scores": {}, "method": "rule_based",
        }
        gate._consequence_service = mock_csvc

        mock_dsvc = MagicMock()
        mock_dsvc.compute_domain_confidence.side_effect = RuntimeError("unavailable")
        gate._confidence_service = mock_dsvc

        result = gate.evaluate("add to list", domain="productivity")
        assert result["auto_execute"] is False
        assert result["domain_confidence"] == 0.0


# ── Singleton ─────────────────────────────────────────────────────────────────


class TestSingleton:
    """get_autonomous_execution_gate() must return the same instance."""

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
