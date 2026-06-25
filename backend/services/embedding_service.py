
import concurrent.futures
import hashlib
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, cast

import numpy as np

from services.onnx_session import CPU_PROVIDER, build_session, choose_providers

if TYPE_CHECKING:
    from services.memory_store import MemoryStore

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

# Execution providers that compile graph subgraphs into backend-specific binaries
# and therefore cannot round-trip back to ONNX — the optimized-graph serializer
# bails out on any compiled node. When one of these is in the chosen EP list,
# the optimized cache must be written via a CPU-only prime pass instead.
_COMPILING_EPS = frozenset({
    "CoreMLExecutionProvider",
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "ROCMExecutionProvider",
})


def _model_dir() -> Path:
    from services.file_mapper_service import FileMapperService
    base = FileMapperService.get_models_path(_MODEL_SUBDIR)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_thread_count() -> int:
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


def _build_session(providers: Optional[List[str]] = None) -> tuple[object, Path]:
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

    chosen = list(providers) if providers is not None else choose_providers(onnx_path)

    # ORT refuses to serialize a graph once a compiling EP (CoreML/CUDA/TRT/ROCm)
    # has claimed nodes — session construction crashes mid-way when
    # ``optimized_model_filepath`` is set. Prime the cache with a throwaway
    # CPU-only session first, then open the real session from the written graph.
    # One-off cost on first boot per ORT version; no-op on CPU-only hosts.
    if not optimized_path.exists() and any(ep in _COMPILING_EPS for ep in chosen):
        logger.info(
            f"[EMBEDDING] Priming optimized graph via CPU-only pass (ORT {ort.__version__})"
        )
        prime_opts = ort.SessionOptions()
        prime_opts.intra_op_num_threads = _resolve_thread_count()
        prime_opts.inter_op_num_threads = 1
        prime_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        prime_opts.optimized_model_filepath = str(optimized_path)
        prime_sess = ort.InferenceSession(
            str(onnx_path), sess_options=prime_opts, providers=[CPU_PROVIDER]
        )
        del prime_sess

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

    session = build_session(
        load_path, sess_options=opts, providers=chosen, log_prefix="[EMBEDDING]"
    )
    return session, onnx_path


def _rebuild_session_cpu_only() -> object:
    global _session
    with _model_lock:
        session, _ = _build_session(providers=[CPU_PROVIDER])
        _session = session
        return session


def _get_session_and_tokenizer() -> tuple[object, object]:
    global _session, _tokenizer, _output_names, _input_names

    if _session is not None and _tokenizer is not None:
        return _session, _tokenizer

    with _model_lock:
        if _session is not None and _tokenizer is not None:
            return _session, _tokenizer

        from transformers import AutoTokenizer

        session, onnx_path = _build_session()

        from onnxruntime import InferenceSession as _IS  # noqa: PLC0415
        _output_names = [o.name for o in cast(_IS, session).get_outputs()]
        _input_names = [i.name for i in cast(_IS, session).get_inputs()]
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
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    sum_emb = (last_hidden_state * mask).sum(axis=1)
    sum_mask = mask.sum(axis=1).clip(min=1e-9)
    return cast(np.ndarray, sum_emb / sum_mask)


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True).clip(min=1e-9)
    return cast(np.ndarray, embeddings / norms)


def _encode_batch(texts: List[str]) -> np.ndarray:
    from onnxruntime import InferenceSession as _IS  # noqa: PLC0415
    from transformers import PreTrainedTokenizerBase as _Tok  # noqa: PLC0415
    session, tokenizer = _get_session_and_tokenizer()

    encoded = cast(_Tok, tokenizer)(
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
        outputs = cast(_IS, session).run(None, feed)
    except Exception as e:
        # Accelerated providers (CoreML, CUDA, etc.) can init cleanly but fail
        # on specific runtime shapes/tokens. Rebuild once as CPU-only and retry.
        if cast(_IS, session).get_providers() != ["CPUExecutionProvider"]:
            logger.warning(
                f"[EMBEDDING] Inference failed on {cast(_IS, session).get_providers()}: {e}. "
                f"Rebuilding session as CPU-only for the rest of this process."
            )
            session = _rebuild_session_cpu_only()
            outputs = cast(_IS, session).run(None, feed)
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
    return _CACHE_PREFIX + hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _get_store() -> object:
    from services.memory_client import MemoryClientService
    return MemoryClientService.create_connection()


# ── Inference queue ─────────────────────────────────────────────────────────
#
# A single daemon worker serializes ALL ORT inference calls.  Multiple
# EmbeddingService instances (list_service, search router, …)
# all share this one queue via the module-level globals, preventing concurrent
# session.run() calls that otherwise allocate 500 MB+ each and OOM under bulk
# ingestion.
#
# Worker is started lazily on the first job submission so that importing this
# module in tests never spawns a real ONNX thread.

_embedding_queue: queue.Queue[tuple[List[str], concurrent.futures.Future[np.ndarray]]] = queue.Queue()
_embedding_worker_started = threading.Lock()
_embedding_worker_running = False


def _run_embedding_worker() -> None:
    while True:
        texts, future = _embedding_queue.get()
        try:
            result = _encode_batch(texts)
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)


def _ensure_worker_started() -> None:
    global _embedding_worker_running
    if _embedding_worker_running:
        return
    with _embedding_worker_started:
        if _embedding_worker_running:
            return
        t = threading.Thread(target=_run_embedding_worker, name="embedding-worker", daemon=True)
        t.start()
        _embedding_worker_running = True


def _submit_for_inference(texts: List[str], mp: object = None) -> np.ndarray:
    _ensure_worker_started()
    future: concurrent.futures.Future[np.ndarray] = concurrent.futures.Future()
    _embedding_queue.put((texts, future))
    metrics = getattr(mp, '_metrics', None) if mp is not None else None
    if metrics is None:
        return future.result()
    with metrics.stage('embedding_wait'):
        return future.result()


# Singleton EmbeddingService instance
_embedding_service_instance = None
_embedding_service_lock = threading.Lock()


def get_embedding_service() -> 'EmbeddingService':
    global _embedding_service_instance
    if _embedding_service_instance is None:
        with _embedding_service_lock:
            if _embedding_service_instance is None:
                _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance


class EmbeddingService:

    def __init__(self, config: Optional[dict[str, object]] = None) -> None:
        self.config = config or {}
        self.embedding_dimensions = self.config.get('embedding_dimensions', 768)

    def _cache_get(self, text: str) -> Optional[list[float]]:
        try:
            store = _get_store()
            raw = cast("MemoryStore", store).get(_cache_key(text))
            if raw is not None:
                return cast(list[float], json.loads(raw))
        except Exception:
            pass
        return None

    def _cache_put(self, text: str, embedding: list[float]) -> None:
        try:
            store = _get_store()
            cast("MemoryStore", store).set(_cache_key(text), json.dumps(embedding), ex=_CACHE_TTL)
        except Exception:
            pass

    def generate_embedding(self, text: str, mp: object = None) -> list[float]:
        # Cache check FIRST — bypass the queue entirely on a hit.
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        try:
            embedding = cast(list[float], _submit_for_inference([text], mp)[0].tolist())
            self._cache_put(text, embedding)
            return embedding
        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embedding_np(self, text: str, mp: object = None) -> np.ndarray:
        # Cache check FIRST — bypass the queue entirely on a hit.
        cached = self._cache_get(text)
        if cached is not None:
            return np.array(cached, dtype=np.float32)

        try:
            embedding = cast(np.ndarray, _submit_for_inference([text], mp)[0])
            self._cache_put(text, embedding.tolist())
            return embedding
        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str], mp: object = None) -> List[np.ndarray]:
        if not texts:
            return []

        try:
            results: list[np.ndarray] = []
            # Process in chunks of 32 to bound memory usage per job.
            chunk_size = 32
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                embeddings = _submit_for_inference(chunk, mp)
                results.extend(embeddings)
            return results
        except Exception as e:
            logger.error(f"[EMBEDDING] Batch generation failed: {e}")
            raise
