"""Tests for services/embedding_service.py — real ONNX model, no mocking.

These tests trigger model download on first run (~300MB, cached after that at
data/models/gte-modernbert-base/onnx/model.onnx). They exist to catch
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
    _COMPILING_EPS,
)


class TestEmbeddingServiceONNX:
    @classmethod
    def setup_class(cls):
        _get_session_and_tokenizer()

    # ── model load ──────────────────────────────────────────────────────────

    def test_session_is_loaded(self):
        from services.embedding_service import _session
        assert _session is not None

    def test_tokenizer_is_loaded(self):
        from services.embedding_service import _tokenizer
        assert _tokenizer is not None

    # ── output shape and dtype ───────────────────────────────────────────────

    def test_generate_embedding_returns_768d_list(self):
        svc = EmbeddingService()
        vec = svc.generate_embedding("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 768

    def test_generate_embedding_np_returns_float32_array(self):
        svc = EmbeddingService()
        vec = svc.generate_embedding_np("numpy test")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (768,)

    def test_batch_embeddings_shape(self):
        svc = EmbeddingService()
        vecs = svc.generate_embeddings_batch(["first", "second", "third"])
        assert len(vecs) == 3
        for v in vecs:
            assert hasattr(v, "shape")
            assert v.shape == (768,)

    def test_empty_batch_returns_empty_list(self):
        svc = EmbeddingService()
        assert svc.generate_embeddings_batch([]) == []

    # ── L2 normalisation ─────────────────────────────────────────────────────

    def test_embedding_is_l2_normalized(self):
        svc = EmbeddingService()
        vec = np.array(svc.generate_embedding("normalize this"))
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_batch_embeddings_are_l2_normalized(self):
        svc = EmbeddingService()
        vecs = svc.generate_embeddings_batch(["alpha", "beta", "gamma"])
        for v in vecs:
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    # ── semantic quality ──────────────────────────────────────────────────────

    def test_semantically_similar_texts_score_higher(self):
        svc = EmbeddingService()
        dog = np.array(svc.generate_embedding("dog"))
        puppy = np.array(svc.generate_embedding("puppy"))
        quantum = np.array(svc.generate_embedding("quantum entanglement"))
        assert np.dot(dog, puppy) > np.dot(dog, quantum)

    def test_identical_texts_have_similarity_one(self):
        svc = EmbeddingService()
        a = np.array(svc.generate_embedding("the quick brown fox"))
        b = np.array(svc.generate_embedding("the quick brown fox"))
        assert abs(np.dot(a, b) - 1.0) < 1e-4

    # ── singleton ─────────────────────────────────────────────────────────────

    def test_get_embedding_service_returns_singleton(self):
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
