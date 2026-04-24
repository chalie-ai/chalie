"""Tests for services/embedding_service.py — real ONNX model, no mocking.

These tests trigger model download on first run (~300MB, cached after that at
backend/data/models/gte-modernbert-base/onnx/model.onnx). They exist to catch
regressions in the ONNX pipeline end-to-end: load, tokenize, infer, normalize.
"""

import numpy as np
import onnxruntime as ort
import pytest

from services.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    _build_session,
    _get_session_and_tokenizer,
    _model_dir,
    _supports_coreml,
    _COMPILING_EPS,
    _METAL_TEXTURE_LIMIT,
)


class TestEmbeddingServiceONNX:
    """Verify gte-modernbert-base ONNX pipeline loads and produces correct output."""

    @classmethod
    def setup_class(cls):
        """Load the ONNX session once for all tests in this class."""
        _get_session_and_tokenizer()

    # ── model load ──────────────────────────────────────────────────────────

    def test_session_is_loaded(self):
        """Module-level _session must be non-None after load."""
        from services.embedding_service import _session
        assert _session is not None

    def test_tokenizer_is_loaded(self):
        """Module-level _tokenizer must be non-None after load."""
        from services.embedding_service import _tokenizer
        assert _tokenizer is not None

    # ── output shape and dtype ───────────────────────────────────────────────

    def test_generate_embedding_returns_768d_list(self):
        """Single embedding should be a 768-element list."""
        svc = EmbeddingService()
        vec = svc.generate_embedding("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 768

    def test_generate_embedding_np_returns_float32_array(self):
        """generate_embedding_np should return a (768,) float32 numpy array."""
        svc = EmbeddingService()
        vec = svc.generate_embedding_np("numpy test")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (768,)

    def test_batch_embeddings_shape(self):
        """Batch of N texts should return N embeddings, each (768,)."""
        svc = EmbeddingService()
        vecs = svc.generate_embeddings_batch(["first", "second", "third"])
        assert len(vecs) == 3
        for v in vecs:
            assert hasattr(v, "shape")
            assert v.shape == (768,)

    def test_empty_batch_returns_empty_list(self):
        """Empty input to batch embeddings must return an empty list."""
        svc = EmbeddingService()
        assert svc.generate_embeddings_batch([]) == []

    # ── L2 normalisation ─────────────────────────────────────────────────────

    def test_embedding_is_l2_normalized(self):
        """Single embedding norm must be 1.0 (±1e-5)."""
        svc = EmbeddingService()
        vec = np.array(svc.generate_embedding("normalize this"))
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_batch_embeddings_are_l2_normalized(self):
        """Every vector in a batch must have norm 1.0 (±1e-5)."""
        svc = EmbeddingService()
        vecs = svc.generate_embeddings_batch(["alpha", "beta", "gamma"])
        for v in vecs:
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    # ── semantic quality ──────────────────────────────────────────────────────

    def test_semantically_similar_texts_score_higher(self):
        """Cosine similarity of related words must exceed unrelated pair."""
        svc = EmbeddingService()
        dog = np.array(svc.generate_embedding("dog"))
        puppy = np.array(svc.generate_embedding("puppy"))
        quantum = np.array(svc.generate_embedding("quantum entanglement"))
        assert np.dot(dog, puppy) > np.dot(dog, quantum)

    def test_identical_texts_have_similarity_one(self):
        """Same text encoded twice should have cosine similarity ≈ 1.0."""
        svc = EmbeddingService()
        a = np.array(svc.generate_embedding("the quick brown fox"))
        b = np.array(svc.generate_embedding("the quick brown fox"))
        assert abs(np.dot(a, b) - 1.0) < 1e-4

    # ── singleton ─────────────────────────────────────────────────────────────

    def test_get_embedding_service_returns_singleton(self):
        """get_embedding_service() must return the same instance on repeated calls."""
        assert get_embedding_service() is get_embedding_service()


class TestCompilingEpCachePrime:
    """Compiling EPs (CoreML/CUDA/TRT/ROCm) cannot co-exist with graph serialization —
    ``_build_session`` must prime the optimized cache via a CPU-only pass first,
    then open the real session from the written graph."""

    @pytest.mark.skipif(
        not any(ep in _COMPILING_EPS for ep in ort.get_available_providers()),
        reason="No compiling EP available — prime-pass code path is unreachable here.",
    )
    def test_prime_pass_writes_cache_and_loads_session(self):
        """Wipe the optimized cache, rebuild on a host with a compiling EP, and confirm
        the prime-then-load flow completes without the 'compiled nodes' serializer crash."""
        onnx_path = _model_dir() / "onnx" / "model.onnx"
        ort_ver = ort.__version__.replace(".", "_")
        optimized_path = _model_dir() / "onnx" / f"model.optimized.{ort_ver}.onnx"

        if not onnx_path.exists():
            pytest.skip("Base model.onnx not cached — skip to avoid 300MB download.")

        backup_path = optimized_path.with_suffix(".onnx.primebackup")
        # Restart from a clean slate so the prime branch actually fires.
        if optimized_path.exists():
            if backup_path.exists():
                backup_path.unlink()
            optimized_path.rename(backup_path)

        try:
            session, _ = _build_session()
            try:
                assert optimized_path.exists(), (
                    "prime pass should have written the optimized graph to disk"
                )
                # Session must come up — historically crashed mid-construction on Mac.
                assert session.get_providers(), "session loaded with at least one provider"
            finally:
                del session
        finally:
            # Drop the freshly-produced cache and restore the original (if any).
            if optimized_path.exists():
                optimized_path.unlink()
            if backup_path.exists():
                backup_path.rename(optimized_path)


def _write_dummy_onnx(path, init_dims):
    """Write a minimal ONNX model with a single initializer of the given shape.

    The graph itself is a no-op (identity on a float tensor); we only care
    about ``graph.initializer`` dims since that's what ``_supports_coreml``
    inspects.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = np.zeros(init_dims, dtype=np.float32)
    init = numpy_helper.from_array(weight, name="weight")

    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "dummy", [inp], [out], initializer=[init])
    # ai.onnx opset 13 is widely supported; the actual op (Identity) has no
    # opset-specific semantics beyond v1, the version just keeps checkers happy.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


@pytest.mark.unit
class TestSupportsCoreML:
    """``_supports_coreml`` inspects model initializer dims up-front so Macs
    can drop the CoreML EP before ORT partitions the graph across the Metal
    2D-texture ceiling (16384 px). False means the model will trip the limit."""

    def test_returns_false_for_oversized_initializer(self, tmp_path):
        """ModernBERT-sized vocab embedding (50368×768) must trigger the strip."""
        p = tmp_path / "big.onnx"
        _write_dummy_onnx(p, (50368, 768))
        assert _supports_coreml(p) is False

    def test_returns_true_at_exact_limit(self, tmp_path):
        """Dim equal to the 16384 limit is still safe — the check is strict >."""
        p = tmp_path / "edge.onnx"
        _write_dummy_onnx(p, (_METAL_TEXTURE_LIMIT, 768))
        assert _supports_coreml(p) is True

    def test_returns_true_for_small_initializer(self, tmp_path):
        """Small models (all dims within limit) should keep CoreML."""
        p = tmp_path / "small.onnx"
        _write_dummy_onnx(p, (512, 768))
        assert _supports_coreml(p) is True

    def test_returns_true_on_missing_file_fail_open(self, tmp_path):
        """Missing file → fail open: return True so ORT makes the final call."""
        assert _supports_coreml(tmp_path / "does-not-exist.onnx") is True

    def test_returns_true_on_corrupt_file_fail_open(self, tmp_path):
        """Unparseable bytes → fail open (log warning, caller keeps CoreML)."""
        p = tmp_path / "corrupt.onnx"
        p.write_bytes(b"not a valid protobuf")
        assert _supports_coreml(p) is True

    def test_custom_limit_argument(self, tmp_path):
        """Caller-supplied limit overrides the default for tuning/tests."""
        p = tmp_path / "mid.onnx"
        _write_dummy_onnx(p, (1024, 768))
        assert _supports_coreml(p, limit=512) is False
        assert _supports_coreml(p, limit=2048) is True


@pytest.mark.unit
class TestBuildSessionProviderStrip:
    """When auto-detecting providers, ``_build_session`` must strip CoreML if
    ``_supports_coreml`` rejects the model. Explicit ``providers=`` overrides
    bypass the check — operators get what they asked for."""

    def test_coreml_stripped_when_model_exceeds_limit(self, monkeypatch, caplog, tmp_path):
        """Auto-detect path: CoreML present + big model → CoreML dropped, CPU retained."""
        import services.embedding_service as es

        onnx_path = tmp_path / "onnx" / "model.onnx"
        onnx_path.parent.mkdir(parents=True)
        _write_dummy_onnx(onnx_path, (50368, 768))
        # Skip prime-pass branch: pre-create the optimized cache sentinel so the
        # fake InferenceSession only fires once, on the real session construction.
        ort_ver = ort.__version__.replace(".", "_")
        (onnx_path.parent / f"model.optimized.{ort_ver}.onnx").touch()

        captured = {}

        def fake_inference_session(path, sess_options=None, providers=None):
            captured["providers"] = list(providers or [])
            raise RuntimeError("stop-after-provider-selection")

        monkeypatch.setattr(es, "_model_dir", lambda: tmp_path)
        monkeypatch.setattr(
            ort, "get_available_providers",
            lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        monkeypatch.setattr(ort, "InferenceSession", fake_inference_session)

        with caplog.at_level("INFO", logger="services.embedding_service"):
            with pytest.raises(RuntimeError, match="stop-after-provider-selection"):
                _build_session()

        assert "CoreMLExecutionProvider" not in captured["providers"]
        assert "CPUExecutionProvider" in captured["providers"]
        assert any("Dropped CoreMLExecutionProvider" in r.message for r in caplog.records)

    def test_coreml_retained_when_model_within_limit(self, monkeypatch, tmp_path):
        """Auto-detect path: CoreML present + small model → CoreML kept as-is."""
        import services.embedding_service as es

        onnx_path = tmp_path / "onnx" / "model.onnx"
        onnx_path.parent.mkdir(parents=True)
        _write_dummy_onnx(onnx_path, (_METAL_TEXTURE_LIMIT, 768))
        ort_ver = ort.__version__.replace(".", "_")
        (onnx_path.parent / f"model.optimized.{ort_ver}.onnx").touch()

        captured = {}

        def fake_inference_session(path, sess_options=None, providers=None):
            captured["providers"] = list(providers or [])
            raise RuntimeError("stop-after-provider-selection")

        monkeypatch.setattr(es, "_model_dir", lambda: tmp_path)
        monkeypatch.setattr(
            ort, "get_available_providers",
            lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        monkeypatch.setattr(ort, "InferenceSession", fake_inference_session)

        with pytest.raises(RuntimeError, match="stop-after-provider-selection"):
            _build_session()

        assert "CoreMLExecutionProvider" in captured["providers"]

    def test_explicit_providers_bypass_shape_check(self, monkeypatch, tmp_path):
        """Explicit override: caller asked for CoreML on a big model — honour it."""
        import services.embedding_service as es

        onnx_path = tmp_path / "onnx" / "model.onnx"
        onnx_path.parent.mkdir(parents=True)
        _write_dummy_onnx(onnx_path, (50368, 768))
        ort_ver = ort.__version__.replace(".", "_")
        (onnx_path.parent / f"model.optimized.{ort_ver}.onnx").touch()

        captured = {}
        probe_calls = {"count": 0}

        def fake_inference_session(path, sess_options=None, providers=None):
            captured["providers"] = list(providers or [])
            raise RuntimeError("stop-after-provider-selection")

        def spy_supports_coreml(*args, **kwargs):
            probe_calls["count"] += 1
            return False

        monkeypatch.setattr(es, "_model_dir", lambda: tmp_path)
        monkeypatch.setattr(es, "_supports_coreml", spy_supports_coreml)
        monkeypatch.setattr(ort, "InferenceSession", fake_inference_session)

        with pytest.raises(RuntimeError, match="stop-after-provider-selection"):
            _build_session(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])

        assert captured["providers"] == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert probe_calls["count"] == 0, "shape check must be skipped on explicit override"

    def test_non_coreml_providers_untouched(self, monkeypatch, tmp_path):
        """CUDA/DirectML/ROCm/TRT all handle dims up to INT_MAX — only CoreML is at risk."""
        import services.embedding_service as es

        onnx_path = tmp_path / "onnx" / "model.onnx"
        onnx_path.parent.mkdir(parents=True)
        _write_dummy_onnx(onnx_path, (50368, 768))
        ort_ver = ort.__version__.replace(".", "_")
        (onnx_path.parent / f"model.optimized.{ort_ver}.onnx").touch()

        captured = {}

        def fake_inference_session(path, sess_options=None, providers=None):
            captured["providers"] = list(providers or [])
            raise RuntimeError("stop-after-provider-selection")

        monkeypatch.setattr(es, "_model_dir", lambda: tmp_path)
        monkeypatch.setattr(
            ort, "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        monkeypatch.setattr(ort, "InferenceSession", fake_inference_session)

        with pytest.raises(RuntimeError, match="stop-after-provider-selection"):
            _build_session()

        assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
