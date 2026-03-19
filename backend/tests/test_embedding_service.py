"""Tests for services/embedding_service.py — real ONNX model, no mocking.

These tests trigger model download on first run (~300MB, cached after that at
backend/data/models/gte-modernbert-base/onnx/model.onnx). They exist to catch
regressions in the ONNX pipeline end-to-end: load, tokenize, infer, normalize.
"""

import numpy as np
import pytest

from services.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    _get_session_and_tokenizer,
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
