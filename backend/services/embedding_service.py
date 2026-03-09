"""
Unified Embedding Service — Single source of truth for all vector embeddings.

Uses sentence-transformers (all-mpnet-base-v2) — pure Python, no external service required.
All outputs are L2-normalized for cosine similarity via dot product.

Embeddings are cached in MemoryStore (1h TTL) keyed by text hash. Identical text
never hits the model twice within the TTL window, regardless of which service
requests it (reflex, topic classifier, context assembly, etc.).

Model downloads automatically from HuggingFace on first run (~438MB, cached locally).
"""

import hashlib
import json
import logging
import threading
import numpy as np
from typing import List, Optional

from services.config_service import ConfigService

logger = logging.getLogger(__name__)

# Singleton model (lazy loaded, thread-safe)
_st_model = None
_st_model_lock = threading.Lock()

# Cache TTL — 1 hour covers all request-scoped reuse and short-term repeats
_CACHE_TTL = 3600
_CACHE_PREFIX = 'emb:'


def _get_st_model(model_name: str = 'all-mpnet-base-v2'):
    """Get or create the sentence-transformers model (singleton, thread-safe)."""
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


def _cache_key(text: str) -> str:
    """Deterministic cache key from text content."""
    return _CACHE_PREFIX + hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _get_store():
    """Lazy import to avoid circular imports at module load time."""
    from services.memory_client import MemoryClientService
    return MemoryClientService.create_connection()


# Singleton EmbeddingService instance
_embedding_service_instance = None


def get_embedding_service() -> 'EmbeddingService':
    """Get or create the EmbeddingService singleton."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance


class EmbeddingService:
    """Unified embedding service using sentence-transformers (no external service required).

    All single-text methods check MemoryStore before computing. Cache is keyed by
    sha256(text)[:16] with a 1-hour TTL. Batch embeddings bypass the cache (bulk
    operations like document chunking don't benefit from per-text caching).
    """

    def __init__(self, config: dict = None):
        self.config = config or ConfigService.resolve_agent_config("semantic-memory")
        self.embedding_dimensions = self.config.get('embedding_dimensions', 768)
        self.model_name = self.config.get('embedding_model', 'all-mpnet-base-v2')

    def _cache_get(self, text: str) -> Optional[list]:
        """Check MemoryStore for a cached embedding. Returns list or None."""
        try:
            store = _get_store()
            raw = store.get(_cache_key(text))
            if raw is not None:
                return json.loads(raw)
        except Exception:
            pass  # Cache miss — compute normally
        return None

    def _cache_put(self, text: str, embedding: list) -> None:
        """Store embedding in MemoryStore with TTL."""
        try:
            store = _get_store()
            store.set(_cache_key(text), json.dumps(embedding), ex=_CACHE_TTL)
        except Exception:
            pass  # Non-fatal — next call will just recompute

    def generate_embedding(self, text: str) -> list:
        """Single embedding → list (for SQLite storage). L2-normalized. Cached."""
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        try:
            model = _get_st_model(self.model_name)
            embedding = model.encode(text, normalize_embeddings=True).tolist()
            self._cache_put(text, embedding)
            return embedding

        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embedding_np(self, text: str) -> np.ndarray:
        """Single embedding → numpy array (for cosine similarity math). L2-normalized. Cached."""
        cached = self._cache_get(text)
        if cached is not None:
            return np.array(cached, dtype=np.float32)

        try:
            model = _get_st_model(self.model_name)
            embedding = model.encode(text, normalize_embeddings=True)
            embedding_list = embedding.tolist()
            self._cache_put(text, embedding_list)
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed → list of numpy arrays. L2-normalized.

        Batch operations (document chunking, bulk consolidation) bypass per-text
        caching — the overhead of N cache lookups + JSON serialization would negate
        the benefit for large batches.
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
