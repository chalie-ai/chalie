
import concurrent.futures
import hashlib
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import List, Optional, cast

import numpy as np

from services.onnx_session import build_session

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thread-safe singleton facade for gte-modernbert-base ONNX embeddings.

    Enforces a single ORT session and a single serialized inference worker so
    that concurrent callers never fork a second 500MB session — the OOM guard.
    Construct via ``EmbeddingService()`` or ``get_embedding_service()``; both
    return the same instance.

    # final: __new__ — singleton invariant must not be overridden.
    """

    # ── Class constants ──────────────────────────────────────────────────────

    _MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
    _MODEL_SUBDIR = "gte-modernbert-base"
    _ONNX_FILENAME = "onnx/model.onnx"

    # Hard ceiling = ModernBERT's positional-embedding limit. The tokenizer's own
    # model_max_length is the HuggingFace "no limit" sentinel (~1e30), so we must
    # enforce the architectural cap here to guard against pathological inputs.
    # Tokeniser uses padding=True so short inputs pay no extra compute for a high
    # ceiling — the tensor is sized to the actual token count, not to this value.
    _MODEL_MAX_TOKENS = 8192

    # Cache TTL — 1 hour covers all request-scoped reuse and short-term repeats
    _CACHE_TTL = 3600
    _CACHE_PREFIX = "emb:"

    _BATCH_CHUNK = 32

    # ── Singleton enforcement ────────────────────────────────────────────────

    _instance: Optional["EmbeddingService"] = None
    _instance_lock = threading.Lock()

    # Instance state (built in __new__, declared here for typing).
    _session: object
    _tokenizer: object
    _output_names: List[str]
    _input_names: List[str]
    _model_lock: threading.Lock
    _queue: "queue.Queue[tuple[List[str], concurrent.futures.Future[np.ndarray]]]"
    _worker_lock: threading.Lock
    _worker_running: bool

    def __new__(cls) -> "EmbeddingService":
        # State is built here, inside the lock, so the single-session invariant
        # holds even when concurrent boot threads construct simultaneously.
        # (Python would re-run __init__ on every call, racing the guard — so
        # there is no __init__; the instance is fully formed before publication.)
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._session = None
                    inst._tokenizer = None
                    inst._output_names = []
                    inst._input_names = []
                    inst._model_lock = threading.Lock()
                    inst._queue = queue.Queue()
                    inst._worker_lock = threading.Lock()
                    inst._worker_running = False
                    cls._instance = inst
        return cls._instance

    # ── Path / config helpers ────────────────────────────────────────────────

    def _model_dir(self) -> Path:
        from services.file_mapper_service import FileMapperService
        base = FileMapperService.get_models_path(self._MODEL_SUBDIR)
        base.mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def _resolve_thread_count() -> int:
        override = os.environ.get("CHALIE_ORT_INTRA_THREADS")
        if override:
            try:
                n = int(override)
                if n >= 1:
                    return n
            except ValueError:
                logger.warning(
                    f"[EMBEDDING] Ignoring non-integer CHALIE_ORT_INTRA_THREADS={override!r}"
                )
        cpu = os.cpu_count() or 2
        return min(4, max(2, cpu // 2))

    # ── Model loading ────────────────────────────────────────────────────────

    def _build_session(self) -> tuple[object, Path]:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_dir = self._model_dir()
        onnx_path = model_dir / "onnx" / "model.onnx"
        ort_ver = ort.__version__.replace(".", "_")
        optimized_path = model_dir / "onnx" / f"model.optimized.{ort_ver}.onnx"

        if not onnx_path.exists():
            logger.info("[EMBEDDING] Downloading gte-modernbert-base (~300MB, first run)...")
            try:
                hf_hub_download(
                    repo_id=self._MODEL_ID,
                    filename=self._ONNX_FILENAME,
                    local_dir=str(model_dir),
                )
                logger.info(f"[EMBEDDING] Model saved to {onnx_path}")
            except Exception as e:
                logger.error(f"[EMBEDDING] Failed to download model: {e}")
                raise

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._resolve_thread_count()
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

        return build_session(load_path, sess_options=opts, log_prefix="[EMBEDDING]"), onnx_path

    def _ensure_loaded(self) -> None:
        """Load session + tokenizer (double-checked, thread-safe)."""
        if self._session is not None and self._tokenizer is not None:
            return
        with self._model_lock:
            if self._session is not None and self._tokenizer is not None:
                return

            from onnxruntime import InferenceSession as _IS  # noqa: PLC0415
            from transformers import AutoTokenizer

            session, onnx_path = self._build_session()

            self._output_names = [o.name for o in cast(_IS, session).get_outputs()]
            self._input_names = [i.name for i in cast(_IS, session).get_inputs()]
            logger.debug(f"[EMBEDDING] Inputs: {self._input_names}, outputs: {self._output_names}")

            try:
                tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID, local_files_only=True)
                logger.info("[EMBEDDING] Tokenizer loaded from cache")
            except Exception:
                logger.info("[EMBEDDING] Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID)

            self._session = session
            self._tokenizer = tokenizer

            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"[EMBEDDING] gte-modernbert-base ready ({size_mb:.0f}MB, ONNX)")

    # ── Inference ────────────────────────────────────────────────────────────

    @staticmethod
    def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask[..., np.newaxis].astype(np.float32)
        sum_emb = (last_hidden_state * mask).sum(axis=1)
        sum_mask = mask.sum(axis=1).clip(min=1e-9)
        return cast(np.ndarray, sum_emb / sum_mask)

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True).clip(min=1e-9)
        return cast(np.ndarray, embeddings / norms)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        from onnxruntime import InferenceSession as _IS  # noqa: PLC0415
        from transformers import PreTrainedTokenizerBase as _Tok  # noqa: PLC0415

        self._ensure_loaded()

        encoded = cast(_Tok, self._tokenizer)(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self._MODEL_MAX_TOKENS,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = cast(_IS, self._session).run(None, feed)

        if "sentence_embedding" in self._output_names:
            pooled = outputs[self._output_names.index("sentence_embedding")]
        else:
            last_hidden = outputs[self._output_names.index("last_hidden_state")]
            pooled = self._mean_pool(last_hidden, attention_mask)

        return self._l2_normalize(pooled).astype(np.float32)

    # ── Inference queue (OOM guard) ──────────────────────────────────────────

    def _run_worker(self) -> None:
        while True:
            texts, future = self._queue.get()
            try:
                future.set_result(self._encode_batch(texts))
            except Exception as exc:
                future.set_exception(exc)

    def _ensure_worker(self) -> None:
        if self._worker_running:
            return
        with self._worker_lock:
            if self._worker_running:
                return
            threading.Thread(
                target=self._run_worker, name="embedding-worker", daemon=True
            ).start()
            self._worker_running = True

    def _submit(self, texts: List[str], mp: object = None) -> np.ndarray:
        self._ensure_worker()
        future: concurrent.futures.Future[np.ndarray] = concurrent.futures.Future()
        self._queue.put((texts, future))
        metrics = getattr(mp, "_metrics", None) if mp is not None else None
        if metrics is None:
            return future.result()
        with metrics.stage("embedding_wait"):
            return future.result()

    # ── Cache helpers ────────────────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        return self._CACHE_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _cache_get(self, text: str) -> Optional[list[float]]:
        try:
            from services.memory_client import MemoryClientService
            raw = MemoryClientService.create_connection().get(self._cache_key(text))
            if raw is not None:
                return cast(list[float], json.loads(raw))
        except Exception as e:
            logger.debug(f"[EMBEDDING] Cache get failed: {e}")
        return None

    def _cache_put(self, text: str, embedding: list[float]) -> None:
        try:
            from services.memory_client import MemoryClientService
            MemoryClientService.create_connection().set(
                self._cache_key(text), json.dumps(embedding), ex=self._CACHE_TTL
            )
        except Exception as e:
            logger.debug(f"[EMBEDDING] Cache put failed: {e}")

    # ── Public facade ────────────────────────────────────────────────────────

    def ensure_loaded(self) -> None:
        """Load the ONNX session and tokenizer (idempotent warmup entrypoint)."""
        self._ensure_loaded()

    @property
    def is_loaded(self) -> bool:
        """True once the ONNX session has been loaded."""
        return self._session is not None

    def loaded_encoder(self) -> tuple[object, object, list[str], list[str]]:
        """Return (session, tokenizer, output_names, input_names) after loading.

        Tight interface for onnx_inference_service which borrows the same model.
        """
        self._ensure_loaded()
        return self._session, self._tokenizer, self._output_names, self._input_names

    def generate_embedding(self, text: str, mp: object = None) -> list[float]:
        """Embed *text* as a 768-float L2-normalised vector (cache-backed)."""
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        try:
            embedding = cast(list[float], self._submit([text], mp)[0].tolist())
            self._cache_put(text, embedding)
            return embedding
        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embedding_np(self, text: str, mp: object = None) -> np.ndarray:
        """Embed *text* as a (768,) float32 ndarray (cache-backed)."""
        cached = self._cache_get(text)
        if cached is not None:
            return np.array(cached, dtype=np.float32)
        try:
            embedding = cast(np.ndarray, self._submit([text], mp)[0])
            self._cache_put(text, embedding.tolist())
            return embedding
        except Exception as e:
            logger.error(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embeddings_batch(
        self, texts: List[str], mp: object = None
    ) -> List[np.ndarray]:
        """Embed multiple texts in chunks, returning one (768,) array per text."""
        if not texts:
            return []
        try:
            results: list[np.ndarray] = []
            for i in range(0, len(texts), self._BATCH_CHUNK):
                results.extend(self._submit(texts[i : i + self._BATCH_CHUNK], mp))
            return results
        except Exception as e:
            logger.error(f"[EMBEDDING] Batch generation failed: {e}")
            raise


def get_embedding_service() -> EmbeddingService:
    """Canonical accessor — always returns the singleton."""
    return EmbeddingService()
