"""
Tests for user_summary_worker._should_synthesise() and the lazy-fallback
concurrency guard in UserMessageProcessor.getUserDefinition().

All deterministic tests: @pytest.mark.unit
Integration test (real LLM): @pytest.mark.integration
"""

import threading
import time

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _insert_trait(db, key, value, last_confirmed_at):
    """Insert a user_specific row with an explicit last_confirmed_at."""
    from services.time_utils import utc_now

    now = utc_now().isoformat()
    db.execute(
        """
        INSERT INTO data_graph
            (kind, key, value, active, deleted_at, retrieval_weight,
             storage_strength, evidence_count, first_seen_at, last_confirmed_at,
             source)
        VALUES ('user_specific', ?, ?, 1, NULL, 1.0, 0.5, 1, ?, ?, 'test')
        """,
        (key, value, now, last_confirmed_at),
    )
    db.commit()


def _insert_summary(db, value, last_confirmed_at):
    """Insert a kind='system' key='user_summary' row with explicit last_confirmed_at."""
    from services.time_utils import utc_now

    now = utc_now().isoformat()
    db.execute(
        """
        INSERT INTO data_graph
            (kind, key, value, active, deleted_at, retrieval_weight,
             storage_strength, evidence_count, first_seen_at, last_confirmed_at,
             source)
        VALUES ('system', 'user_summary', ?, 1, NULL, 0.7, 0.5, 1, ?, ?, 'user_summary_processor')
        """,
        (value, now, last_confirmed_at),
    )
    db.commit()


# ── TestShouldSynthesise ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestShouldSynthesise:

    def test_returns_false_when_latest_trait_older_than_summary(self, db):
        """T1 trait < T2 summary → no re-synthesis needed."""
        from workers.user_summary_worker import _should_synthesise

        _insert_trait(db, 'name', 'Alice', '2026-01-01T10:00:00+00:00')
        _insert_summary(db, 'Alice is a person', '2026-01-01T12:00:00+00:00')

        assert not _should_synthesise()

    def test_returns_true_on_new_trait_after_summary(self, db):
        """Summary at T1, new trait at T2 > T1 → re-synthesis needed."""
        from workers.user_summary_worker import _should_synthesise

        _insert_summary(db, 'old summary', '2026-01-01T10:00:00+00:00')
        _insert_trait(db, 'city', 'Valletta', '2026-01-01T12:00:00+00:00')

        assert _should_synthesise()

    def test_returns_true_when_summary_missing_but_traits_exist(self, db):
        """No summary row, one trait → synthesis needed."""
        from workers.user_summary_worker import _should_synthesise

        _insert_trait(db, 'hobby', 'cycling', '2026-01-01T10:00:00+00:00')

        assert _should_synthesise()

    def test_returns_false_when_both_missing(self, db):
        """No traits at all → False (nothing to synthesise)."""
        from workers.user_summary_worker import _should_synthesise

        assert not _should_synthesise()

    def test_returns_false_when_trait_equal_to_summary_ts(self, db):
        """Same timestamp → trait is NOT newer than summary → False."""
        from workers.user_summary_worker import _should_synthesise

        ts = '2026-01-01T10:00:00+00:00'
        _insert_trait(db, 'name', 'Bob', ts)
        _insert_summary(db, 'Bob is a person', ts)

        assert not _should_synthesise()

    def test_does_not_count_summary_row_itself_in_max_query(self, db):
        """The user_summary row (kind='system') must never be picked up by the
        MAX(last_confirmed_at) query — that would cause an infinite loop.

        Scenario: summary written at T2, no user_specific rows at all.
        _should_synthesise() MUST return False (not True from the summary's
        own timestamp fooling the check).
        """
        from workers.user_summary_worker import _should_synthesise

        _insert_summary(db, 'a summary with no traits', '2026-06-01T10:00:00+00:00')

        assert not _should_synthesise(), (
            "_should_synthesise() returned True even though no user_specific rows exist. "
            "The kind='user_specific' filter on the MAX query is broken."
        )


# ── TestLazyFallbackConcurrencyGuard ──────────────────────────────────────────

@pytest.mark.unit
class TestLazyFallbackConcurrencyGuard:

    def setup_method(self):
        """Reset the module-level in-flight flag before each test."""
        import services.user_message_processor as mod
        mod._lazy_fire_in_flight = False

    def test_lazy_fallback_concurrency_guard_single_daemon(self, db):
        """Two concurrent calls to _fire_lazy_synthesis spawn only ONE daemon.

        Real-stack (no mocks): swap the module's UserSummaryProcessor class
        with a counting subclass whose send() blocks on a hold_event so the
        in-flight flag stays True while the second thread tries to enter.
        """
        import services.user_message_processor as mod
        import services.user_summary_processor as proc_mod

        daemon_count = {'n': 0}
        start_barrier = threading.Barrier(2, timeout=5)
        hold_event = threading.Event()

        class CountingProc(proc_mod.UserSummaryProcessor):
            def send(self, request_id: str | None = None):  # noqa: D401
                # Test double, blocks intentionally to keep in-flight flag True.
                del request_id
                daemon_count['n'] += 1
                hold_event.wait(timeout=5)

        errors = []

        def fire():
            try:
                start_barrier.wait()
                mod._fire_lazy_synthesis()
            except Exception as exc:
                errors.append(exc)

        original_cls = proc_mod.UserSummaryProcessor
        proc_mod.UserSummaryProcessor = CountingProc
        try:
            t1 = threading.Thread(target=fire)
            t2 = threading.Thread(target=fire)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)
            hold_event.set()
        finally:
            proc_mod.UserSummaryProcessor = original_cls

        assert not errors, f"Threads raised: {errors}"
        assert daemon_count['n'] == 1, (
            f"Expected exactly 1 synthesis daemon call, got {daemon_count['n']}"
        )


# ── Integration ────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestLazyFallbackIntegration:

    def test_lazy_fallback_spawns_daemon_when_row_missing_and_traits_exist(self, db):
        """Cold start: missing user_summary + one trait → lazy synthesis fires once.

        Waits up to 30s for the daemon to write the row, then checks:
          1. Both calls returned the generic fallback string.
          2. Exactly one user_summary row was written by user_summary_processor.
        """
        import services.user_message_processor as mod
        from services.data_graph_service import get_data_graph_service

        # Reset in-flight guard
        mod._lazy_fire_in_flight = False

        _insert_trait(db, 'name', 'Integration Tester', '2026-01-01T10:00:00+00:00')

        results = []

        def call():
            proc = mod.UserMessageProcessor(raw_input='hello')
            proc._user_definition_cached = None
            results.append(proc.getUserDefinition())

        t1 = threading.Thread(target=call)
        t2 = threading.Thread(target=call)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        fallback = "The user is a real human. Treat this conversation as peer-to-peer dialogue."
        for r in results:
            assert r == fallback, f"Expected fallback for this turn, got: {r!r}"

        # Wait for daemon to complete (max 30s)
        deadline = time.time() + 30
        summary_rows = []
        while time.time() < deadline:
            summary_rows = [
                r for r in get_data_graph_service().fetch(kinds=['system'])
                if r.get('key') == 'user_summary'
                and r.get('source') == 'user_summary_processor'
            ]
            if summary_rows:
                break
            time.sleep(0.5)

        assert summary_rows, "Lazy synthesis daemon did not write user_summary row within 30s"
        assert len(summary_rows) == 1, (
            f"Expected exactly 1 user_summary row, found {len(summary_rows)}"
        )
