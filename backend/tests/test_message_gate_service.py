"""Unit tests for MessageGateService."""
import pytest
from unittest.mock import MagicMock, patch

# Mark all tests as unit (no external dependencies)
pytestmark = pytest.mark.unit


class TestEmptyInputGuard:
    """Test the empty-input guard in gate()."""

    def test_empty_text_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_whitespace_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("   ")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_none_text_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate(None)
        assert result.route == 'cancel'
        assert result.confidence == 1.0


class TestCancelDetection:
    """Test deterministic CANCEL keyword matching."""

    def test_stop_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("stop")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_cancel_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("cancel that")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_nevermind_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("nevermind")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_never_mind_two_words_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("never mind about that")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_forget_it_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("forget it")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_abort_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("abort")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_nvm_returns_cancel(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("nvm")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_cancel_case_insensitive(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("STOP doing that")
        assert result.route == 'cancel'
        assert result.confidence == 1.0

    def test_cancel_not_matched_mid_sentence(self):
        """Cancel keyword only fires when it starts the message."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        # "please cancel" does not start with a cancel keyword
        result = svc.gate("please cancel that")
        # Should not be cancel (no match at start)
        assert result.route in ('act', 'respond')


class TestOnnxModeGate:
    """Test ONNX mode gate routing."""

    def test_onnx_respond_high_confidence_returns_respond(self):
        """ONNX predicts RESPOND with conf >= 0.85 → respond route."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict.return_value = ('RESPOND', 0.92)

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("hello how are you?")

        assert result.route == 'respond'
        assert result.confidence == pytest.approx(0.92)

    def test_onnx_respond_low_confidence_falls_to_act(self):
        """ONNX predicts RESPOND but conf < 0.85 → act route (conservative)."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict.return_value = ('RESPOND', 0.70)

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("what should I do about this?")

        assert result.route == 'act'

    def test_onnx_act_returns_act(self):
        """ONNX predicts ACT → act route."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict.return_value = ('ACT', 0.88)

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("search for the latest Python news")

        assert result.route == 'act'
        assert result.confidence == pytest.approx(0.88)

    def test_onnx_clarify_remapped_to_act(self):
        """ONNX predicts CLARIFY → remapped to act."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict.return_value = ('CLARIFY', 0.75)

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("do the thing")

        assert result.route == 'act'

    def test_onnx_ignore_remapped_to_act(self):
        """ONNX predicts IGNORE → remapped to act (Chalie always responds)."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict.return_value = ('IGNORE', 0.80)

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("hey")

        assert result.route == 'act'

    def test_onnx_unavailable_defaults_to_act(self):
        """When ONNX is unavailable, defaults to act route."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = False

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("search for Python tutorials")

        assert result.route == 'act'
        assert result.confidence == pytest.approx(0.5)

    def test_onnx_exception_defaults_to_act(self):
        """When ONNX raises an exception, defaults to act route gracefully."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   side_effect=RuntimeError("ONNX model not loaded")):
            result = svc.gate("find me some recipes")

        assert result.route == 'act'
        assert result.confidence == pytest.approx(0.5)


class TestGateResultMetadata:
    """Test that GateResult carries timing information."""

    def test_gate_time_ms_is_non_negative(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()
        result = svc.gate("")
        assert result.gate_time_ms >= 0.0

    def test_gate_time_ms_set_for_normal_message(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = False

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("hello there")

        assert result.gate_time_ms >= 0.0


class TestPrefilterSkills:
    """Test ONNX skill pre-filtering."""

    def test_returns_skills_and_ext_tool_flag(self):
        """prefilter_skills returns selected skills and needs_external_tool flag."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict_multi_label.return_value = [
            ('schedule', 0.91),
            ('recall', 0.85),
        ]

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            skills, needs_tool = svc.prefilter_skills("remind me to take my meds at 9am")

        assert 'schedule' in skills
        assert needs_tool is False

    def test_needs_external_tool_flag_detected(self):
        """needs_external_tool flag is set when skill selector predicts it."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict_multi_label.return_value = [
            ('needs_external_tool', 0.88),
            ('recall', 0.80),
        ]

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            skills, needs_tool = svc.prefilter_skills("search for Python news")

        assert needs_tool is True
        assert 'needs_external_tool' not in skills

    def test_onnx_unavailable_returns_empty(self):
        """When ONNX is unavailable, returns empty skills and False for tool flag."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = False

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            skills, needs_tool = svc.prefilter_skills("do something")

        assert skills == []
        assert needs_tool is False

    def test_onnx_exception_returns_empty(self):
        """When ONNX raises, gracefully returns empty skills."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        with patch('services.onnx_inference_service.get_onnx_inference_service',
                   side_effect=RuntimeError("model error")):
            skills, needs_tool = svc.prefilter_skills("test message")

        assert skills == []
        assert needs_tool is False
