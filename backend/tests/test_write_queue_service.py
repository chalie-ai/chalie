"""Behavioural tests for WriteQueueService — exercises submit / submit_sync / get_stats against the real production stack."""

from collections.abc import Iterator

import pytest

from services.write_queue_service import WriteQueueService


def _raise(exc: BaseException) -> None:
    raise exc

pytestmark = pytest.mark.unit


# ── submit_sync ─────────────────────────────────────────────────────────────

@pytest.fixture
def wq() -> Iterator[WriteQueueService]:
    svc = WriteQueueService()
    yield svc


class TestSubmitSync:
    def test_round_trip_returns_callable_result(self, wq: WriteQueueService) -> None:
        result = wq.submit_sync(lambda a, b: a + b, 3, 4)
        assert result == 7

    def test_exception_propagates_to_caller(self, wq: WriteQueueService) -> None:
        with pytest.raises(ValueError, match="boom"):
            wq.submit_sync(lambda: _raise(ValueError("boom")))

    def test_preserves_fifo_ordering(self, wq: WriteQueueService) -> None:
        results = []
        for i in range(5):
            results.append(wq.submit_sync(lambda x=i: x))
        assert results == [0, 1, 2, 3, 4]


# ── submit (fire-and-forget) ────────────────────────────────────────────────

class TestSubmit:
    def test_side_effect_applied_asynchronously(self, wq: WriteQueueService) -> None:
        side = []
        wq.submit(lambda: side.append("fired"))
        # Drain enough to let the worker thread process it — use sync to flush.
        wq.submit_sync(lambda: None)
        assert side == ["fired"]

    def test_exception_does_not_propagate(self, wq: WriteQueueService) -> None:
        # Fire-and-forget exceptions are logged but NOT stored (no result
        # container), so they cannot be counted in stats.  They must never
        # propagate to the caller — that's the contract under test.
        wq.submit(lambda: _raise(RuntimeError("silent")))
        # Flush to ensure the item was processed.
        wq.submit_sync(lambda: None)
        # Not asserting errors count — fire-and-forget has no result tracking.


# ── get_stats ───────────────────────────────────────────────────────────────

class TestGetStats:
    def test_initial_stats_all_zero(self, wq: WriteQueueService) -> None:
        stats = wq.get_stats()
        assert stats == {"queue_size": 0, "processed": 0, "errors": 0}

    def test_stats_reflect_processed_count(self, wq: WriteQueueService) -> None:
        for _ in range(3):
            wq.submit_sync(lambda: None)
        assert wq.get_stats()["processed"] == 3

    def test_errors_incremented_on_sync_failure(self, wq: WriteQueueService) -> None:
        with pytest.raises(RuntimeError, match="e"):
            wq.submit_sync(lambda: _raise(RuntimeError("e")))
        stats = wq.get_stats()
        assert stats["errors"] == 1


