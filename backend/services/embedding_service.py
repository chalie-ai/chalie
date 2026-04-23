"""
Unified Embedding Service — Single source of truth for all vector embeddings.

Uses gte-modernbert-base via ONNX Runtime — no PyTorch required.
All outputs are L2-normalized for cosine similarity via dot product.

Model: Alibaba-NLP/gte-modernbert-base
  - 768 dimensions (drop-in compatible with existing vector store)
  - ~300MB model file (smaller than previous all-mpnet-base-v2 at 438MB)
  - MTEB 64.38 (vs 57.8 for all-mpnet-base-v2)
  - 8192 token context window (vs 512 previously)
  - ONNX Runtime inference, CPU-friendly, no PyTorch dependency

Model downloads automatically from HuggingFace on first run (~300MB, cached
at backend/data/models/gte-modernbert-base/onnx/model.onnx).
"""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from services.config_service import ConfigService

logger = logging.getLogger(__name__)

# Model config
_MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
_MODEL_SUBDIR = "gte-modernbert-base"
_ONNX_FILENAME = "onnx/model.onnx"

# Hard ceiling = ModernBERT's positional-embedding limit. The tokenizer's own
# model_max_length is the HuggingFace "no limit" sentinel (~1e30), so we must
# enforce the architectural cap here to guard against pathological inputs.
# Tokeniser uses padding=True so short inputs pay no extra compute for a high
# ceiling — the tensor is sized to the actual token count, not to this value.
_MODEL_MAX_TOKENS = 8192

# Singleton (lazy loaded, thread-safe)
_session = None
_tokenizer = None
_output_names: List[str] = []
_input_names: List[str] = []
_model_lock = threading.Lock()

# Cache TTL — 1 hour covers all request-scoped reuse and short-term repeats
_CACHE_TTL = 3600
_CACHE_PREFIX = 'emb:'


def _model_dir() -> Path:
    """Return path to local model cache directory, creating it if needed."""
    base = Path(__file__).parent.parent / "data" / "models" / _MODEL_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_thread_count() -> int:
    """Pick ``intra_op_num_threads``. Env override wins, else ``min(4, max(2, cpu//2))``.

    The hardcoded ``2`` left cycles on the table on 8+ core hosts. Cap of 4 avoids
    oversubscribing shared workers (memory pipeline, DMN, goal pursuit all share the
    same CPU). Env override (``CHALIE_ORT_INTRA_THREADS``) is the escape hatch for
    container CPU limits that ``os.cpu_count()`` cannot see.
    """
    override = os.environ.get("CHALIE_ORT_INTRA_THREADS")
    if override:
        try:
            n = int(override)
            if n >= 1:
                return n
        except ValueError:
            logger.warning(f"[EMBEDDING] Ignoring non-integer CHALIE_ORT_INTRA_THREADS={override!r}")
    cpu = os.cpu_count() or 2
    return min(4, max(2, cpu // 2))


def _build_session(providers: Optional[List[str]] = None):
    """Construct an ONNX InferenceSession with the chosen providers.

    If ``providers`` is None, uses ``ort.get_available_providers()`` — the
    runtime returns accelerators (CUDA/CoreML/ROCm/…) first and CPU last, so
    whatever is installed gets picked automatically.

    The pre-optimized graph is cached with the ORT version baked into the filename
    (``model.optimized.<ort_version>.onnx``). Optimized graphs are not forward-
    compatible across ORT upgrades — a stale ``.optimized.onnx`` from an older ORT
    can load but silently degrade, or worse, fail to run specific op kernels. Pin
    the version so every upgrade forces a fresh optimize pass.
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_dir = _model_dir()
    onnx_path = model_dir / "onnx" / "model.onnx"
    ort_ver = ort.__version__.replace(".", "_")
    optimized_path = model_dir / "onnx" / f"model.optimized.{ort_ver}.onnx"

    if not onnx_path.exists():
        logger.info("[EMBEDDING] Downloading gte-modernbert-base (~300MB, first run)...")
        try:
            hf_hub_download(repo_id=_MODEL_ID, filename=_ONNX_FILENAME, local_dir=str(model_dir))
            logger.info(f"[EMBEDDING] Model saved to {onnx_path}")
        except Exception as e:
            logger.error(f"[EMBEDDING] Failed to download model: {e}")
            raise

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _resolve_thread_count()
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    if optimized_path.exists():
        load_path = optimized_path
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        logger.info(f"[EMBEDDING] Loading pre-optimized model (ORT {ort.__version__})")
    else:
        load_path = onnx_path
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        opts.optimized_model_filepath = str(optimized_path)

    chosen = providers if providers is not None else ort.get_available_providers()
    session = ort.InferenceSession(str(load_path), sess_options=opts, providers=chosen)
    logger.info(
        f"[EMBEDDING] Providers: {session.get_providers()}, "
        f"intra_op_num_threads={opts.intra_op_num_threads}"
    )
    return session, onnx_path


def _rebuild_session_cpu_only():
    """Rebuild the module-level session as CPU-only. Called when an accelerator fails at runtime."""
    global _session
    with _model_lock:
        session, _ = _build_session(providers=["CPUExecutionProvider"])
        _session = session
        return session


def _get_session_and_tokenizer():
    """Return (ort.InferenceSession, AutoTokenizer), loading on first call.

    Uses double-checked locking so only one thread triggers the model load.
    Some providers (notably CoreML on ModernBERT variants) init cleanly but
    fail on certain runtime shapes/token-ids. The inference path in
    _encode_batch catches that and calls _rebuild_session_cpu_only.
    """
    global _session, _tokenizer, _output_names, _input_names

    if _session is not None and _tokenizer is not None:
        return _session, _tokenizer

    with _model_lock:
        if _session is not None and _tokenizer is not None:
            return _session, _tokenizer

        from transformers import AutoTokenizer

        session, onnx_path = _build_session()

        _output_names = [o.name for o in session.get_outputs()]
        _input_names = [i.name for i in session.get_inputs()]
        logger.debug(f"[EMBEDDING] Inputs: {_input_names}, outputs: {_output_names}")

        # Load tokenizer — cached in HF default cache after first download
        try:
            tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, local_files_only=True)
            logger.info("[EMBEDDING] Tokenizer loaded from cache")
        except Exception:
            logger.info("[EMBEDDING] Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)

        _session = session
        _tokenizer = tokenizer

        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        logger.info(f"[EMBEDDING] gte-modernbert-base ready ({size_mb:.0f}MB, ONNX)")
        return _session, _tokenizer


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pool token embeddings, excluding padding tokens."""
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    sum_emb = (last_hidden_state * mask).sum(axis=1)
    sum_mask = mask.sum(axis=1).clip(min=1e-9)
    return sum_emb / sum_mask


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings along the last axis."""
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True).clip(min=1e-9)
    return embeddings / norms


def _encode_batch(texts: List[str]) -> np.ndarray:
    """Tokenize and embed a batch of texts. Returns (N, 768) float32, L2-normalized.

    Sequence length is set dynamically by the tokenizer (padding=True pads each
    batch to its longest item). _MODEL_MAX_TOKENS only bites for truly
    oversized inputs — it prevents position_ids from exceeding the model's
    positional-embedding range.
    """
    session, tokenizer = _get_session_and_tokenizer()

    encoded = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=_MODEL_MAX_TOKENS,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in _input_names:
        feed["token_type_ids"] = np.zeros_like(input_ids)

    try:
        outputs = session.run(None, feed)
    except Exception as e:
        # Accelerated providers (CoreML, CUDA, etc.) can init cleanly but fail
        # on specific runtime shapes/tokens. Rebuild once as CPU-only and retry.
        if session.get_providers() != ["CPUExecutionProvider"]:
            logger.warning(
                f"[EMBEDDING] Inference failed on {session.get_providers()}: {e}. "
                f"Rebuilding session as CPU-only for the rest of this process."
            )
            session = _rebuild_session_cpu_only()
            outputs = session.run(None, feed)
        else:
            raise

    # Use pre-pooled output if available, otherwise mean pool last_hidden_state
    if "sentence_embedding" in _output_names:
        pooled = outputs[_output_names.index("sentence_embedding")]
    else:
        last_hidden = outputs[_output_names.index("last_hidden_state")]
        pooled = _mean_pool(last_hidden, attention_mask)

    return _l2_normalize(pooled).astype(np.float32)


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
    """Return the process-wide EmbeddingService singleton, creating it if needed."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance


class EmbeddingService:
    """Unified embedding service using gte-modernbert-base via ONNX Runtime.

    All single-text methods check MemoryStore before computing. Cache is keyed by
    sha256(text)[:16] with a 1-hour TTL. Batch embeddings bypass the cache (bulk
    operations like document chunking don't benefit from per-text caching).
    """

    def __init__(self, config: dict = None):
        if config is None:
            try:
                config = ConfigService.resolve_agent_config("semantic-memory")
            except FileNotFoundError:
                config = {}
        self.config = config
        self.embedding_dimensions = self.config.get('embedding_dimensions', 768)

    def _cache_get(self, text: str) -> Optional[list]:
        """Check MemoryStore for a cached embedding. Returns list or None."""
        try:
            store = _get_store()
            raw = store.get(_cache_key(text))
            if raw is not None:
                return json.loads(raw)
        except Exception:
            pass
        return None

    def _cache_put(self, text: str, embedding: list) -> None:
        """Store embedding in MemoryStore with TTL."""
        try:
            store = _get_store()
            store.set(_cache_key(text), json.dumps(embedding), ex=_CACHE_TTL)
        except Exception:
            pass

    def generate_embedding(self, text: str) -> list:
        """Generate a single L2-normalized embedding vector as a list. Cached.

        Returns:
            Embedding as a plain Python list of floats suitable for SQLite storage.
        """
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        try:
            embedding = _encode_batch([text])[0].tolist()
            self._cache_put(text, embedding)
            return embedding
        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embedding_np(self, text: str) -> np.ndarray:
        """Generate a single L2-normalized embedding vector as a numpy array. Cached.

        Returns:
            Embedding as a float32 numpy array for cosine similarity math.
        """
        cached = self._cache_get(text)
        if cached is not None:
            return np.array(cached, dtype=np.float32)

        try:
            embedding = _encode_batch([text])[0]
            self._cache_put(text, embedding.tolist())
            return embedding
        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate L2-normalized embeddings for a batch of texts.

        Batch operations (document chunking, bulk consolidation) bypass per-text
        caching — the overhead of N cache lookups + JSON serialization would negate
        the benefit for large batches.

        Returns:
            List of float32 numpy arrays, one per input text.
        """
        if not texts:
            return []

        try:
            results = []
            # Process in chunks of 32 to bound memory usage
            chunk_size = 32
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                embeddings = _encode_batch(chunk)
                results.extend(embeddings)
            return results
        except Exception as e:
            logger.error(f"[EMBEDDING] Batch generation failed: {e}")
            raise
