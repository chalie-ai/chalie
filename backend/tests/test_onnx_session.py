from pathlib import Path

import numpy as np
import pytest

from services.onnx_session import (
    METAL_TEXTURE_LIMIT,
    _model_fits_coreml,
    choose_providers,
    CPU_PROVIDER,
)


def _write_dummy_onnx(path: Path, init_dims: tuple[int, int]) -> None:
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

    def test_returns_false_for_oversized_initializer(self, tmp_path: Path) -> None:
        p = tmp_path / "big.onnx"
        _write_dummy_onnx(p, (50368, 768))
        assert _model_fits_coreml(p) is False

    def test_returns_true_at_exact_limit(self, tmp_path: Path) -> None:
        p = tmp_path / "edge.onnx"
        _write_dummy_onnx(p, (METAL_TEXTURE_LIMIT, 768))
        assert _model_fits_coreml(p) is True

    def test_returns_true_for_small_initializer(self, tmp_path: Path) -> None:
        p = tmp_path / "small.onnx"
        _write_dummy_onnx(p, (512, 768))
        assert _model_fits_coreml(p) is True

    def test_returns_true_on_missing_file_fail_open(self, tmp_path: Path) -> None:
        assert _model_fits_coreml(tmp_path / "does-not-exist.onnx") is True

    def test_returns_true_on_corrupt_file_fail_open(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupt.onnx"
        p.write_bytes(b"not a valid protobuf")
        assert _model_fits_coreml(p) is True

    def test_custom_limit_argument(self, tmp_path: Path) -> None:
        p = tmp_path / "mid.onnx"
        _write_dummy_onnx(p, (1024, 768))
        assert _model_fits_coreml(p, limit=512) is False
        assert _model_fits_coreml(p, limit=2048) is True


@pytest.mark.unit
class TestChooseProviders:

    def test_cpu_provider_always_present(self) -> None:
        providers = choose_providers()
        assert CPU_PROVIDER in providers

    def test_coreml_stripped_for_oversized_model(self, tmp_path: Path) -> None:
        import onnxruntime as ort
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CoreML EP not available on this host.")

        p = tmp_path / "big.onnx"
        _write_dummy_onnx(p, (50368, 768))
        providers = choose_providers(p)
        assert "CoreMLExecutionProvider" not in providers
        assert CPU_PROVIDER in providers
