
import time
import pytest

from services.memory_store import MemoryStore
from services.wrapper_rate_limiter import WrapperRateLimiter, _KEY_PREFIX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_limiter(limit=10, window=60, store=None):
    if store is None:
        store = MemoryStore()
    return WrapperRateLimiter(limit=limit, window_seconds=window, store=store), store


# ---------------------------------------------------------------------------
# Basic allow / deny
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWrapperRateLimiterBasic:
    def test_first_signal_is_allowed(self):
        limiter, _ = _make_limiter(limit=5)
        assert limiter.is_allowed("wrp_test") is True

    def test_signals_up_to_limit_are_allowed(self):
        limiter, _ = _make_limiter(limit=3)
        for _ in range(3):
            assert limiter.is_allowed("wrp_test") is True

    def test_signal_over_limit_is_denied(self):
        limiter, _ = _make_limiter(limit=3)
        for _ in range(3):
            limiter.is_allowed("wrp_test")
        # 4th should be denied
        assert limiter.is_allowed("wrp_test") is False

    def test_deny_does_not_consume_slot(self):
        limiter, store = _make_limiter(limit=2)
        limiter.is_allowed("wrp_x")
        limiter.is_allowed("wrp_x")
        # Now at limit — two denied calls
        assert limiter.is_allowed("wrp_x") is False
        assert limiter.is_allowed("wrp_x") is False
        # Count still == 2 (denied calls added nothing)
        key = f"{_KEY_PREFIX}wrp_x"
        assert store.zcard(key) == 2

    def test_different_wrappers_have_independent_counters(self):
        limiter, _ = _make_limiter(limit=2)
        limiter.is_allowed("wrp_a")
        limiter.is_allowed("wrp_a")
        # wrp_a is at limit
        assert limiter.is_allowed("wrp_a") is False
        # wrp_b has a fresh counter
        assert limiter.is_allowed("wrp_b") is True



# ---------------------------------------------------------------------------
# Sliding window expiry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWrapperRateLimiterSlidingWindow:
    def test_old_entries_slide_out_of_window(self):
        """Entries older than the window should not count."""
        store = MemoryStore()
        limiter = WrapperRateLimiter(limit=2, window_seconds=1, store=store)
        key = f"{_KEY_PREFIX}wrp_slide"

        # Manually insert two entries with a past timestamp (2 seconds ago)
        past = time.time() - 2
        store.zadd(key, {f"{past}:aaa": past, f"{past}:bbb": past})

        # Both old entries should be pruned → counter is 0 → new call is allowed
        assert limiter.is_allowed("wrp_slide") is True

    def test_fresh_entries_within_window_are_counted(self):
        """Entries within the window must still block new calls."""
        store = MemoryStore()
        limiter = WrapperRateLimiter(limit=2, window_seconds=60, store=store)
        key = f"{_KEY_PREFIX}wrp_fresh"

        now = time.time()
        store.zadd(key, {f"{now}:x": now, f"{now}:y": now})

        # Window is 60s and entries are ~0s old — should be denied
        assert limiter.is_allowed("wrp_fresh") is False

    def test_mixed_old_and_fresh_entries(self):
        """Only entries within the window count toward the limit."""
        store = MemoryStore()
        limiter = WrapperRateLimiter(limit=2, window_seconds=10, store=store)
        key = f"{_KEY_PREFIX}wrp_mix"

        now = time.time()
        old = now - 20  # outside 10s window
        store.zadd(key, {
            f"{old}:expired": old,
            f"{now}:fresh1": now,
        })

        # 1 fresh entry (old is pruned) → below limit of 2 → allowed
        assert limiter.is_allowed("wrp_mix") is True
        # Now 2 entries in window → at limit → denied
        assert limiter.is_allowed("wrp_mix") is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWrapperRateLimiterEdgeCases:
    def test_limit_of_one_allows_exactly_one(self):
        limiter, _ = _make_limiter(limit=1)
        assert limiter.is_allowed("wrp_one") is True
        assert limiter.is_allowed("wrp_one") is False

    def test_unique_members_prevent_collision(self):
        """Two calls at the exact same time should each record a distinct member."""
        store = MemoryStore()
        limiter = WrapperRateLimiter(limit=5, window_seconds=60, store=store)
        # Call twice in rapid succession — both should be recorded
        limiter.is_allowed("wrp_unique")
        limiter.is_allowed("wrp_unique")
        key = f"{_KEY_PREFIX}wrp_unique"
        assert store.zcard(key) == 2
