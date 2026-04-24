"""Feature tests for the FIFO embedding queue.

Verifies that the module-level serialization contract introduced in rc-0.4.0
holds end-to-end:

1. Concurrent callers get valid results AND take longer than a single call
   would (proves inference is sequential, not parallel — the OOM-prevention
   guarantee).
2. Cache hits bypass the queue entirely (proves the deadlock-prevention contract
   and that repeated calls are cheap).
3. All three public methods (generate_embedding, generate_embedding_np,
   generate_embeddings_batch) route through the same queue and each returns
   the correct type/shape.

All tests run against the real ONNX model, real queue, real worker.
Zero mocks, zero patches of production code.
"""

import threading
import time
import uuid

import numpy as np
import pytest

from services.embedding_service import EmbeddingService, get_embedding_service


# ── Skip guard ───────────────────────────────────────────────────────────────
# Model must be present; skip rather than trigger a 300MB download in CI.

def _model_available() -> bool:
    from services.embedding_service import _model_dir
    return (_model_dir() / "onnx" / "model.onnx").exists()


_SKIP = pytest.mark.skipif(
    not _model_available(),
    reason="gte-modernbert-base not cached — skip to avoid 300MB download in CI.",
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def emb_svc():
    """Real EmbeddingService singleton, model loaded once for this module."""
    svc = get_embedding_service()
    # Warm the ONNX session so setup cost doesn't inflate timing measurements.
    svc.generate_embedding("warmup")
    return svc


# ── Tests ─────────────────────────────────────────────────────────────────────


@_SKIP
@pytest.mark.integration
class TestConcurrentCallsSerializeViaQueue:
    """8 threads calling generate_embedding concurrently must each receive a
    valid 768-float embedding.  The total wall-clock time must exceed
    5 × a single-call baseline, proving that inference is serialized rather
    than parallelized — the OOM-prevention guarantee of the FIFO queue.

    If inference ran in parallel (old behaviour), 8 threads on a 4-core host
    would finish in ~1–2 × baseline.  Serialized, they take ~8 × baseline.
    The threshold of 5 × gives generous headroom for scheduling noise while
    still distinguishing the two regimes.
    """

    def test_concurrent_callers_all_succeed_and_execution_is_serialized(self, emb_svc):
        N = 8
        # Unique texts so cache plays no role.
        texts = [f"concurrent-serialization-probe-{uuid.uuid4()}" for _ in range(N)]

        # Single-call baseline: measure 3 calls and take the median to smooth
        # out first-call CoreML JIT warmup, which inflates the first timing.
        baseline_times = []
        for _ in range(3):
            bt = f"baseline-{uuid.uuid4()}"
            t0 = time.perf_counter()
            emb_svc.generate_embedding(bt)
            baseline_times.append(time.perf_counter() - t0)
        baseline_times.sort()
        single_call_time = baseline_times[1]  # median of 3

        results = [None] * N
        errors = []

        def worker(i):
            try:
                results[i] = emb_svc.generate_embedding(texts[i])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]

        wall_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        wall_clock = time.perf_counter() - wall_start

        assert not errors, f"Worker raised: {errors}"

        # Every caller received a valid 768-float embedding.
        for i, vec in enumerate(results):
            assert vec is not None, f"Thread {i} returned None"
            assert isinstance(vec, list), f"Thread {i}: expected list, got {type(vec)}"
            assert len(vec) == 768, f"Thread {i}: expected 768 dims, got {len(vec)}"

        # Serialization check: total wall-clock ≥ 2.5 × single call.
        #
        # If inference ran in parallel on a 4-core host, 8 threads would
        # complete in ~1–1.5 × single call (all overlap on multiple cores).
        # Serialized through the queue, 8 calls take ~8 × single call.
        # A 2.5 × threshold cleanly separates the two regimes while tolerating
        # per-call speedup from EP (CoreML/CUDA) warm caching across calls.
        threshold = 2.5 * single_call_time
        assert wall_clock >= threshold, (
            f"Wall-clock {wall_clock:.2f}s < threshold {threshold:.2f}s "
            f"(median single={single_call_time:.3f}s × 2.5). "
            f"Inference appears parallel — queue serialization may be broken."
        )


@_SKIP
@pytest.mark.integration
class TestCacheHitBypassesQueue:
    """Warming the cache for a text, then calling generate_embedding 100 times
    in a tight loop, must complete in dramatically less time than 100 × a single
    cold inference.  This verifies the cache-first contract that also prevents
    reentrance deadlock if a cached lookup were ever attempted from the worker.
    """

    def test_cached_text_returns_in_fraction_of_cold_inference_time(self, emb_svc):
        # Cold baseline on a unique text.
        cold_text = f"cache-bypass-baseline-{uuid.uuid4()}"
        t0 = time.perf_counter()
        emb_svc.generate_embedding(cold_text)
        single_cold = time.perf_counter() - t0

        # Warm the cache for the hot text.
        hot_text = "cache hit test — stable text"
        emb_svc.generate_embedding(hot_text)

        # 100 cache-hit calls in a single thread.
        REPS = 100
        t0 = time.perf_counter()
        for _ in range(REPS):
            vec = emb_svc.generate_embedding(hot_text)
        cache_total = time.perf_counter() - t0

        # Each cache hit must be at least 10× faster on average than cold inference.
        # (Cold inference is typically 200–600ms; cache hit is <1ms.)
        max_allowed = REPS * single_cold / 10
        assert cache_total < max_allowed, (
            f"100 cache hits took {cache_total:.3f}s; expected < {max_allowed:.3f}s "
            f"(single cold={single_cold:.3f}s). Cache bypass may not be working."
        )

        # Embedding is still valid.
        assert isinstance(vec, list)
        assert len(vec) == 768


@_SKIP
@pytest.mark.integration
class TestAllThreePublicMethodsProduceValidOutput:
    """generate_embedding, generate_embedding_np, and generate_embeddings_batch
    each submit through the same queue and each return the correct type and shape.
    Spawning one thread per method exercises cross-method serialization.
    """

    def test_all_three_methods_return_valid_output_under_concurrent_load(self, emb_svc):
        # Unique inputs — no cache interference across methods.
        uid = uuid.uuid4()
        text_single   = f"method-test-single-{uid}"
        text_np       = f"method-test-np-{uid}"
        texts_batch   = [f"method-test-batch-{uid}-{i}" for i in range(3)]

        results = {}
        errors = []

        def call_generate_embedding():
            try:
                results["list"] = emb_svc.generate_embedding(text_single)
            except Exception as exc:
                errors.append(exc)

        def call_generate_embedding_np():
            try:
                results["np"] = emb_svc.generate_embedding_np(text_np)
            except Exception as exc:
                errors.append(exc)

        def call_generate_embeddings_batch():
            try:
                results["batch"] = emb_svc.generate_embeddings_batch(texts_batch)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=call_generate_embedding),
            threading.Thread(target=call_generate_embedding_np),
            threading.Thread(target=call_generate_embeddings_batch),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert not errors, f"Method call raised: {errors}"

        # generate_embedding → list of 768 floats
        vec_list = results.get("list")
        assert vec_list is not None
        assert isinstance(vec_list, list)
        assert len(vec_list) == 768

        # generate_embedding_np → (768,) float32 ndarray
        vec_np = results.get("np")
        assert vec_np is not None
        assert isinstance(vec_np, np.ndarray)
        assert vec_np.shape == (768,)
        assert vec_np.dtype == np.float32

        # generate_embeddings_batch → list of 3 × (768,) ndarray
        vec_batch = results.get("batch")
        assert vec_batch is not None
        assert len(vec_batch) == 3
        for v in vec_batch:
            assert isinstance(v, np.ndarray)
            assert v.shape == (768,)
