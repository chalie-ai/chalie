"""Unit tests for MessageGateService."""
import pytest
from unittest.mock import MagicMock, patch

# Mark all tests as unit (no external dependencies)
pytestmark = pytest.mark.unit


class TestEmptyInputGuard:
    """Empty/None input defaults to act (no signal to classify)."""

    def test_empty_text_returns_act(self):
        from services.message_gate_service import MessageGateService
        result = MessageGateService().gate("")
        assert result.route == 'act'

    def test_whitespace_returns_act(self):
        from services.message_gate_service import MessageGateService
        result = MessageGateService().gate("   ")
        assert result.route == 'act'

    def test_none_text_returns_act(self):
        from services.message_gate_service import MessageGateService
        result = MessageGateService().gate(None)
        assert result.route == 'act'


class TestOnnxModeGate:
    """Test ONNX mode gate routing."""

    def test_onnx_respond_high_confidence_returns_respond(self):
        """mode-gate multi_label fires RESPOND → respond route."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict_multi_label.return_value = [('RESPOND', 0.92)]

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("hello how are you?")

        mock_onnx.is_available.assert_any_call("mode-gate")
        assert result.route == 'respond'
        assert result.confidence == pytest.approx(0.92)

    def test_onnx_respond_below_threshold_falls_to_act(self):
        """mode-gate RESPOND below threshold → empty predictions → act route."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict_multi_label.return_value = []

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("what should I do about this?")

        assert result.route == 'act'

    def test_onnx_no_respond_label_returns_act(self):
        """mode-gate returns no RESPOND label → act route."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict_multi_label.return_value = []

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("search for the latest Python news")

        assert result.route == 'act'
        assert result.confidence == pytest.approx(0.5)

    def test_onnx_unknown_label_defaults_to_act(self):
        """Unknown label from mode-gate → act (safe default)."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = True
        mock_onnx.predict_multi_label.return_value = [('UNKNOWN', 0.80)]

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("hey")

        assert result.route == 'act'

    def test_onnx_unavailable_defaults_to_act(self):
        """When mode-gate model is unavailable, defaults to act route."""
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

    def test_no_mode_gate_model_means_all_act(self):
        """Without mode-gate deployed, every message → act."""
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = False

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            for msg in ["hello", "what's the weather?", "remind me to call mom"]:
                result = svc.gate(msg)
                assert result.route == 'act', f"Expected act for {msg!r}, got {result.route}"


class TestGateResultMetadata:
    """Test that GateResult carries timing information."""

    def test_gate_time_ms_is_non_negative(self):
        from services.message_gate_service import MessageGateService
        result = MessageGateService().gate("")
        assert result.gate_time_ms >= 0.0

    def test_gate_time_ms_set_for_normal_message(self):
        from services.message_gate_service import MessageGateService
        svc = MessageGateService()

        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = False

        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc.gate("hello there")

        assert result.gate_time_ms >= 0.0
