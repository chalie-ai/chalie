"""
Unified Embedding Service — Single source of truth for all vector embeddings.

Uses sentence-transformers (all-mpnet-base-v2) — pure Python, no external service required.
All outputs are L2-normalized for cosine similarity via dot product.

Model downloads automatically from HuggingFace on first run (~438MB, cached locally).
"""

import logging
import threading
import numpy as np
from typing import List

from services.config_service import ConfigService

logger = logging.getLogger(__name__)

# Singleton model (lazy loaded, thread-safe)
_st_model = None
_st_model_lock = threading.Lock()


def _get_st_model(model_name: str = 'all-mpnet-base-v2'):
    """Return the shared sentence-transformers model, loading it on first call.

    Uses a double-checked locking pattern so only one thread triggers the
    (potentially expensive) model load.

    Args:
        model_name: HuggingFace model identifier
            (default ``'all-mpnet-base-v2'``).

    Returns:
        A loaded :class:`~sentence_transformers.SentenceTransformer` instance.
    """
    global _st_model
    if _st_model is not None:
        return _st_model
    with _st_model_lock:
        # Double-check after acquiring lock
        if _st_model is not None:
            return _st_model
        from sentence_transformers import SentenceTransformer
        # Try loading from local cache first to avoid HuggingFace revision checks on
        # every startup. Falls back to normal load (with network) only on first run.
        try:
            _st_model = SentenceTransformer(model_name, local_files_only=True)
            logger.info(f"[EMBEDDING] Model '{model_name}' ready (loaded from cache)")
        except Exception:
            logger.info(f"[EMBEDDING] Loading sentence-transformers model '{model_name}' (first run may download ~438MB)...")
            _st_model = SentenceTransformer(model_name)
            logger.info(f"[EMBEDDING] Model '{model_name}' ready")
    return _st_model


# Singleton EmbeddingService instance
_embedding_service_instance = None


def get_embedding_service() -> 'EmbeddingService':
    """Return the process-wide EmbeddingService singleton, creating it if needed.

    Returns:
        The singleton :class:`EmbeddingService` instance.
    """
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance


class EmbeddingService:
    """Unified embedding service using sentence-transformers (no external service required)."""

    def __init__(self, config: dict = None):
        """Initialize the embedding service.

        Args:
            config: Optional configuration dict. When ``None`` the config is
                resolved from the ``'semantic-memory'`` agent entry. Supported
                keys: ``embedding_dimensions`` (int, default 768) and
                ``embedding_model`` (str, default ``'all-mpnet-base-v2'``).
        """
        self.config = config or ConfigService.resolve_agent_config("semantic-memory")
        self.embedding_dimensions = self.config.get('embedding_dimensions', 768)
        self.model_name = self.config.get('embedding_model', 'all-mpnet-base-v2')

    def generate_embedding(self, text: str) -> list:
        """Generate a single L2-normalized embedding vector as a list.

        Args:
            text: Text string to embed.

        Returns:
            Embedding as a plain Python list of floats suitable for SQLite storage.

        Raises:
            Exception: Propagates embedding generation errors to the caller.
        """
        try:
            model = _get_st_model(self.model_name)
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()

        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embedding_np(self, text: str) -> np.ndarray:
        """Generate a single L2-normalized embedding vector as a numpy array.

        Args:
            text: Text string to embed.

        Returns:
            Embedding as a ``float32`` numpy array for cosine similarity math.

        Raises:
            Exception: Propagates embedding generation errors to the caller.
        """
        try:
            model = _get_st_model(self.model_name)
            embedding = model.encode(text, normalize_embeddings=True)
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate L2-normalized embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of ``float32`` numpy arrays, one per input text.

        Raises:
            Exception: Propagates batch embedding errors to the caller.
        """
        if not texts:
            return []

        try:
            model = _get_st_model(self.model_name)
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        except Exception as e:
            logger.error(f"[EMBEDDING] Batch generation failed: {e}")
            raise
