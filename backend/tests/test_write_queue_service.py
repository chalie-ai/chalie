"""
Behavioural tests for WriteQueueService — real queue, real threading, real DB.

WriteQueueService serialises SQLite writes through a stdlib queue.Queue backed
by a daemon thread.  These tests exercise submit / submit_sync / get_stats
against the real production stack with the test DB fixture.
"""

import pytest

from services.write_queue_service import WriteQueueService

pytestmark = pytest.mark.unit


# ── submit_sync ─────────────────────────────────────────────────────────────

@pytest.fixture
def wq():
    """Fresh WriteQueueService with its own daemon thread."""
    svc = WriteQueueService()
    yield svc
    # Daemon thread exits with process; no explicit shutdown needed.


class TestSubmitSync:
    def test_round_trip_returns_callable_result(self, wq):
        result = wq.submit_sync(lambda a, b: a + b, 3, 4)
        assert result == 7

    def test_exception_propagates_to_caller(self, wq):
        with pytest.raises(ValueError, match="boom"):
            wq.submit_sync(lambda: (_ for _ in ()).throw(ValueError("boom")))

    def test_preserves_fifo_ordering(self, wq):
        results = []
        for i in range(5):
            results.append(wq.submit_sync(lambda x=i: x))
        assert results == [0, 1, 2, 3, 4]


# ── submit (fire-and-forget) ────────────────────────────────────────────────

class TestSubmit:
    def test_side_effect_applied_asynchronously(self, wq):
        side = []
        wq.submit(lambda: side.append("fired"))
        # Drain enough to let the worker thread process it — use sync to flush.
        wq.submit_sync(lambda: None)
        assert side == ["fired"]

    def test_exception_does_not_propagate(self, wq):
        # Fire-and-forget exceptions are logged but NOT stored (no result
        # container), so they cannot be counted in stats.  They must never
        # propagate to the caller — that's the contract under test.
        wq.submit(lambda: (_ for _ in ()).throw(RuntimeError("silent")))
        # Flush to ensure the item was processed.
        wq.submit_sync(lambda: None)
        # Not asserting errors count — fire-and-forget has no result tracking.


# ── get_stats ───────────────────────────────────────────────────────────────

class TestGetStats:
    def test_initial_stats_all_zero(self, wq):
        stats = wq.get_stats()
        assert stats == {"queue_size": 0, "processed": 0, "errors": 0}

    def test_stats_reflect_processed_count(self, wq):
        for _ in range(3):
            wq.submit_sync(lambda: None)
        assert wq.get_stats()["processed"] == 3

    def test_errors_incremented_on_sync_failure(self, wq):
        with pytest.raises(RuntimeError, match="e"):
            wq.submit_sync(lambda: (_ for _ in ()).throw(RuntimeError("e")))
        stats = wq.get_stats()
        assert stats["errors"] == 1


# ── submit_transaction ──────────────────────────────────────────────────────

class TestSubmitTransaction:
    def test_writes_land_in_real_db(self, wq, db):
        """Two INSERTs in a transaction either both land or neither do.

        The ``db`` fixture installs a test-scoped DatabaseService as the
        process-wide singleton, so :meth:`submit_transaction` follows it
        automatically (no env override needed).
        """
        db.execute("CREATE TABLE IF NOT EXISTS wq_test (id INTEGER PRIMARY KEY, val TEXT)")
        db.commit()

        def insert_a(conn):
            conn.execute("INSERT INTO wq_test (val) VALUES ('a')")

        def insert_b(conn):
            conn.execute("INSERT INTO wq_test (val) VALUES ('b')")

        wq.submit_transaction([insert_a, insert_b])

        rows = db.execute("SELECT val FROM wq_test ORDER BY id").fetchall()
        assert [r[0] for r in rows] == ["a", "b"]

    def test_transaction_rollback_on_failure(self, wq, db):
        """If one callable raises, neither row persists."""
        db.execute("CREATE TABLE IF NOT EXISTS wq_test2 (id INTEGER PRIMARY KEY, val TEXT)")
        db.commit()

        def insert_ok(conn):
            conn.execute("INSERT INTO wq_test2 (val) VALUES ('ok')")

        def insert_broken(conn):
            conn.execute("INSERT INTO wq_test2 (val) VALUES ('doomed')")
            raise RuntimeError("forced failure")

        with pytest.raises(RuntimeError, match="forced failure"):
            wq.submit_transaction([insert_ok, insert_broken])

        rows = db.execute("SELECT val FROM wq_test2").fetchall()
        assert rows == [], "rollback must leave no rows"
