"""Tests for services/embedding_service.py — real ONNX model, no mocking.

These tests trigger model download on first run (~300MB, cached after that at
data/models/gte-modernbert-base/onnx/model.onnx). They exist to catch
regressions in the ONNX pipeline end-to-end: load, tokenize, infer, normalize.
"""

import numpy as np

from services.embedding_service import EmbeddingService


class TestEmbeddingServiceONNX:
    @classmethod
    def setup_class(cls) -> None:
        EmbeddingService._get_session_and_tokenizer()

    # ── output shape and dtype ───────────────────────────────────────────────

    def test_generate_embedding_returns_768d_list(self) -> None:
        svc = EmbeddingService()
        vec = svc.generate_embedding("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 768

    def test_generate_embedding_np_returns_float32_array(self) -> None:
        svc = EmbeddingService()
        vec = svc.generate_embedding_np("numpy test")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (768,)

    def test_batch_embeddings_shape(self) -> None:
        svc = EmbeddingService()
        vecs = svc.generate_embeddings_batch(["first", "second", "third"])
        assert len(vecs) == 3
        for v in vecs:
            assert hasattr(v, "shape")
            assert v.shape == (768,)

    def test_empty_batch_returns_empty_list(self) -> None:
        svc = EmbeddingService()
        assert svc.generate_embeddings_batch([]) == []

    # ── L2 normalisation ─────────────────────────────────────────────────────

    def test_embedding_is_l2_normalized(self) -> None:
        svc = EmbeddingService()
        vec = np.array(svc.generate_embedding("normalize this"))
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_batch_embeddings_are_l2_normalized(self) -> None:
        svc = EmbeddingService()
        vecs = svc.generate_embeddings_batch(["alpha", "beta", "gamma"])
        for v in vecs:
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    # ── semantic quality ─────────────────────────────────────────────────────

    def test_semantically_similar_texts_score_higher(self) -> None:
        svc = EmbeddingService()
        dog = np.array(svc.generate_embedding("dog"))
        puppy = np.array(svc.generate_embedding("puppy"))
        quantum = np.array(svc.generate_embedding("quantum entanglement"))
        assert np.dot(dog, puppy) > np.dot(dog, quantum)

    def test_identical_texts_have_similarity_one(self) -> None:
        svc = EmbeddingService()
        a = np.array(svc.generate_embedding("the quick brown fox"))
        b = np.array(svc.generate_embedding("the quick brown fox"))
        assert abs(np.dot(a, b) - 1.0) < 1e-4
