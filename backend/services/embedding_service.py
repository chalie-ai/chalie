
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

from services.onnx_session import build_session

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
_session: object | None = None
_tokenizer: object | None = None
_output_names: List[str] = []
_input_names: List[str] = []
_model_lock = threading.Lock()

# Cache TTL — 1 hour covers all request-scoped reuse and short-term repeats
_CACHE_TTL = 3600
_CACHE_PREFIX = 'emb:'


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


class EmbeddingService:

    @staticmethod
    def _model_dir() -> Path:
        from services.file_mapper_service import FileMapperService
        base = FileMapperService.get_models_path(_MODEL_SUBDIR)
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
                logger.warning(f"[EMBEDDING] Ignoring non-integer CHALIE_ORT_INTRA_THREADS={override!r}")
        cpu = os.cpu_count() or 2
        return min(4, max(2, cpu // 2))

    @staticmethod
    def _build_session() -> tuple[object, Path]:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_dir = EmbeddingService._model_dir()
        onnx_path = model_dir / "onnx" / "model.onnx"
        ort_ver = ort.__version__.replace(".", "_")
        optimized_path = model_dir / "onnx" / f"model.optimized.{ort_ver}.onnx"

        if not onnx_path.exists():
            logger.info("[EMBEDDING] Downloading gte-modernbert-base (~300MB, first run)...")
            try:
                hf_hub_download(repo_id=_MODEL_ID, filename=_ONNX_FILENAME, local_dir=str(model_dir))
                logger.info(f"[EMBEDDING] Model saved to {onnx_path}")
            except Exception as e:
                logger.exception(f"[EMBEDDING] Failed to download model: {e}")
                raise

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = EmbeddingService._resolve_thread_count()
        opts.inter_op_num_threads = 1
        opts.enable_mem_pattern = True
        # Arena OFF: ORT's BFCArena never returns scratch to the OS — it grows to
        # the largest activation ever allocated and holds it for the process
        # lifetime. Batch shape here is (32 x longest text in the batch), so one
        # long web page permanently raises the floor for every later batch.
        # Measured on gte-modernbert-base over an alternating short/long workload:
        # steady state 3,482 MiB with the arena on vs 773 MiB with it off.
        opts.enable_cpu_mem_arena = False

        if optimized_path.exists():
            load_path = optimized_path
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            logger.info(f"[EMBEDDING] Loading pre-optimized model (ORT {ort.__version__})")
        else:
            load_path = onnx_path
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            opts.optimized_model_filepath = str(optimized_path)

        session = build_session(
            load_path, sess_options=opts, log_prefix="[EMBEDDING]"
        )
        return session, onnx_path

    @staticmethod
    def _get_session_and_tokenizer() -> tuple[object, object]:
        global _session, _tokenizer, _output_names, _input_names

        if _session is not None and _tokenizer is not None:
            return _session, _tokenizer

        with _model_lock:
            if _session is not None and _tokenizer is not None:
                return _session, _tokenizer

            from transformers import AutoTokenizer

            session, onnx_path = EmbeddingService._build_session()

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

    @staticmethod
    def _encode_batch(texts: List[str]) -> np.ndarray:
        from onnxruntime import InferenceSession as _IS  # noqa: PLC0415
        from transformers import PreTrainedTokenizerBase as _Tok  # noqa: PLC0415
        session, tokenizer = EmbeddingService._get_session_and_tokenizer()

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

        outputs = cast(_IS, session).run(None, feed)

        # Use pre-pooled output if available, otherwise mean pool last_hidden_state
        if "sentence_embedding" in _output_names:
            pooled = outputs[_output_names.index("sentence_embedding")]
        else:
            last_hidden = outputs[_output_names.index("last_hidden_state")]
            pooled = EmbeddingService._mean_pool(last_hidden, attention_mask)

        return EmbeddingService._l2_normalize(pooled).astype(np.float32)

    @staticmethod
    def _cache_key(text: str) -> str:
        return _CACHE_PREFIX + hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    @staticmethod
    def _get_store() -> object:
        from services.memory_client import MemoryClientService
        return MemoryClientService.create_connection()

    @staticmethod
    def _run_embedding_worker() -> None:
        while True:
            texts, future = _embedding_queue.get()
            try:
                result = EmbeddingService._encode_batch(texts)
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)

    @classmethod
    def _ensure_worker_started(cls) -> None:
        global _embedding_worker_running
        if _embedding_worker_running:
            return
        with _embedding_worker_started:
            if _embedding_worker_running:
                return
            t = threading.Thread(target=cls._run_embedding_worker, name="embedding-worker", daemon=True)
            t.start()
            _embedding_worker_running = True

    @classmethod
    def _submit_for_inference(cls, texts: List[str], mp: object = None) -> np.ndarray:
        cls._ensure_worker_started()
        future: concurrent.futures.Future[np.ndarray] = concurrent.futures.Future()
        _embedding_queue.put((texts, future))
        metrics = getattr(mp, '_metrics', None) if mp is not None else None
        if metrics is None:
            return future.result()
        with metrics.stage('embedding_wait'):
            return future.result()

    def _cache_get(self, text: str) -> Optional[list[float]]:
        try:
            store = self._get_store()
            raw = cast("MemoryStore", store).get(self._cache_key(text))
            if raw is not None:
                return cast(list[float], json.loads(raw))
        except Exception:
            pass
        return None

    def _cache_put(self, text: str, embedding: list[float]) -> None:
        try:
            store = self._get_store()
            cast("MemoryStore", store).set(self._cache_key(text), json.dumps(embedding), ex=_CACHE_TTL)
        except Exception:
            pass

    def generate_embedding(self, text: str, mp: object = None) -> list[float]:
        # Cache check FIRST — bypass the queue entirely on a hit.
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        try:
            embedding = cast(list[float], self._submit_for_inference([text], mp)[0].tolist())
            self._cache_put(text, embedding)
            return embedding
        except Exception as e:
            logger.exception(f"[EMBEDDING] Generation failed: {e}")
            raise

    def generate_embedding_np(self, text: str, mp: object = None) -> np.ndarray:
        # Cache check FIRST — bypass the queue entirely on a hit.
        cached = self._cache_get(text)
        if cached is not None:
            return np.array(cached, dtype=np.float32)

        try:
            embedding = cast(np.ndarray, self._submit_for_inference([text], mp)[0])
            self._cache_put(text, embedding.tolist())
            return embedding
        except Exception as e:
            logger.exception(f"[EMBEDDING] Generation failed: {e}")
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
                embeddings = self._submit_for_inference(chunk, mp)
                results.extend(embeddings)
            return results
        except Exception as e:
            logger.exception(f"[EMBEDDING] Batch generation failed: {e}")
            raise


# Singleton EmbeddingService instance
_embedding_service_instance: EmbeddingService | None = None


def get_embedding_service() -> 'EmbeddingService':
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
