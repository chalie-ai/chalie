"""
Unified Embedding Service — Single source of truth for all vector embeddings.

Uses sentence-transformers (all-mpnet-base-v2) — pure Python, no external service required.
All outputs are L2-normalized for cosine similarity via dot product.

Model downloads automatically from HuggingFace on first run (~438MB, cached locally).
"""

import logging
import numpy as np
from typing import List

from services.config_service import ConfigService

logger = logging.getLogger(__name__)

# Singleton model (lazy loaded)
_st_model = None


def _get_st_model(model_name: str = 'all-mpnet-base-v2'):
    """Get or create the sentence-transformers model (singleton)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[EMBEDDING] Loading sentence-transformers model '{model_name}' (first run may download ~438MB)...")
        _st_model = SentenceTransformer(model_name)
        logger.info(f"[EMBEDDING] Model '{model_name}' ready")
    return _st_model


class EmbeddingService:
    """Unified embedding service using sentence-transformers (no external service required)."""

    def __init__(self, config: dict = None):
        self.config = config or ConfigService.resolve_agent_config("semantic-memory")
        self.embedding_dimensions = self.config.get('embedding_dimensions', 768)
        self.model_name = self.config.get('embedding_model', 'all-mpnet-base-v2')

    def generate_embedding(self, text: str) -> list:
        """Single embedding → list (for PostgreSQL storage). L2-normalized."""
        try:
            model = _get_st_model(self.model_name)
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()

        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embedding_np(self, text: str) -> np.ndarray:
        """Single embedding → numpy array (for cosine similarity math). L2-normalized."""
        try:
            model = _get_st_model(self.model_name)
            embedding = model.encode(text, normalize_embeddings=True)
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed → list of numpy arrays. L2-normalized."""
        if not texts:
            return []

        try:
            model = _get_st_model(self.model_name)
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        except Exception as e:
            logger.error(f"[EMBEDDING] Batch generation failed: {e}")
            raise
