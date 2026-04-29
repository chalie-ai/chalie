"""Tests for services/onnx_session.py — real .onnx files, no ORT mocking.

``_model_fits_coreml`` inspects model initializer dims up-front so Macs can
drop the CoreML EP before ORT partitions the graph across the Metal 2D-texture
ceiling (16384). These tests write real ONNX graphs to tmp_path and call the
real function — no monkey-patching.
"""

import numpy as np
import pytest

from services.onnx_session import (
    METAL_TEXTURE_LIMIT,
    _model_fits_coreml,
    choose_providers,
    CPU_PROVIDER,
)


def _write_dummy_onnx(path, init_dims):
    """Write a minimal ONNX model whose single initializer has the given shape."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = np.zeros(init_dims, dtype=np.float32)
    init = numpy_helper.from_array(weight, name="weight")

    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "dummy", [inp], [out], initializer=[init])
    # ai.onnx opset 13: Identity has identical semantics across opsets; version
    # chosen only to keep the ONNX checker happy on import.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


@pytest.mark.unit
class TestModelFitsCoreML:
    """Real-file tests on the Metal texture-limit check."""

    def test_returns_false_for_oversized_initializer(self, tmp_path):
        """ModernBERT-sized vocab embedding (50368×768) must trigger the strip."""
        p = tmp_path / "big.onnx"
        _write_dummy_onnx(p, (50368, 768))
        assert _model_fits_coreml(p) is False

    def test_returns_true_at_exact_limit(self, tmp_path):
        """Dim equal to the 16384 limit is still safe — the check is strict >."""
        p = tmp_path / "edge.onnx"
        _write_dummy_onnx(p, (METAL_TEXTURE_LIMIT, 768))
        assert _model_fits_coreml(p) is True

    def test_returns_true_for_small_initializer(self, tmp_path):
        """Small models (all dims within limit) should keep CoreML."""
        p = tmp_path / "small.onnx"
        _write_dummy_onnx(p, (512, 768))
        assert _model_fits_coreml(p) is True

    def test_returns_true_on_missing_file_fail_open(self, tmp_path):
        """Missing file → fail open: return True so ORT makes the final call."""
        assert _model_fits_coreml(tmp_path / "does-not-exist.onnx") is True

    def test_returns_true_on_corrupt_file_fail_open(self, tmp_path):
        """Unparseable bytes → fail open (log debug, caller keeps CoreML)."""
        p = tmp_path / "corrupt.onnx"
        p.write_bytes(b"not a valid protobuf")
        assert _model_fits_coreml(p) is True

    def test_custom_limit_argument(self, tmp_path):
        """Caller-supplied limit overrides the default for tuning/tests."""
        p = tmp_path / "mid.onnx"
        _write_dummy_onnx(p, (1024, 768))
        assert _model_fits_coreml(p, limit=512) is False
        assert _model_fits_coreml(p, limit=2048) is True


@pytest.mark.unit
class TestChooseProviders:
    """``choose_providers`` always returns CPU as a safety net."""

    def test_cpu_provider_always_present(self):
        """Whatever the installed wheel exposes, CPU must be in the returned list."""
        providers = choose_providers()
        assert CPU_PROVIDER in providers

    def test_coreml_stripped_for_oversized_model(self, tmp_path):
        """When CoreML is available and the model would trip Metal, CoreML is dropped."""
        import onnxruntime as ort
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CoreML EP not available on this host.")

        p = tmp_path / "big.onnx"
        _write_dummy_onnx(p, (50368, 768))
        providers = choose_providers(p)
        assert "CoreMLExecutionProvider" not in providers
        assert CPU_PROVIDER in providers
