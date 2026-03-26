"""
Wrapper Rate Limiter — sliding window rate limiter for external wrapper signals.

Uses MemoryStore sorted sets to implement a sliding window counter.
Each signal timestamp is stored as a member in a ZSET keyed by wrapper_id.
On each check we:
  1. Remove timestamps older than the window.
  2. Count remaining entries.
  3. If under the limit, add the new timestamp and allow.
  4. If at or over the limit, deny.

Default: 100 signals/minute per wrapper.
"""

import time
import logging
import uuid

from services.memory_store import get_shared_store

logger = logging.getLogger(__name__)

# MemoryStore key pattern for rate-limit windows
_KEY_PREFIX = "wrapper_rate:"

# Default limits
DEFAULT_LIMIT = 100       # signals
DEFAULT_WINDOW = 60       # seconds


class WrapperRateLimiter:
    """Sliding window rate limiter backed by MemoryStore sorted sets.

    Each wrapper's signal timestamps are stored in a ZSET.  On every call to
    :meth:`is_allowed` we:

    1. Prune entries older than ``window_seconds`` using ``zremrangebyscore``.
    2. Count remaining entries with ``zcard``.
    3. If count < limit, record the current timestamp and return ``True``.
    4. Otherwise return ``False`` without recording.

    Args:
        limit: Maximum number of signals allowed within ``window_seconds``.
        window_seconds: Width of the sliding window in seconds.
        store: Optional pre-built MemoryStore connection (injected in tests).
    """

    def __init__(
        self,
        limit: int = DEFAULT_LIMIT,
        window_seconds: int = DEFAULT_WINDOW,
        store=None,
    ):
        """Initialise the rate limiter.

        Args:
            limit: Maximum number of allowed signals per window.  Defaults to
                :data:`DEFAULT_LIMIT` (100).
            window_seconds: Window width in seconds.  Defaults to
                :data:`DEFAULT_WINDOW` (60).
            store: Optional MemoryStore connection.  When ``None`` the shared
                singleton is obtained via ``get_shared_store()``.
        """
        self._limit = limit
        self._window = window_seconds
        self._store = store

    def _get_store(self):
        """Return the MemoryStore connection, lazily creating it if needed."""
        if self._store is not None:
            return self._store
        return get_shared_store()

    def is_allowed(self, wrapper_id: str) -> bool:
        """Check whether ``wrapper_id`` may emit another signal right now.

        Slides the window forward, prunes stale entries, and conditionally
        records the new timestamp.

        Args:
            wrapper_id: The stable identifier of the calling wrapper.  Use
                ``'__chat_ui__'`` for cookie-authenticated (non-wrapper) callers.

        Returns:
            ``True`` if the signal is within the rate limit and has been
            recorded.  ``False`` if the limit is exceeded — the signal is
            **not** recorded in this case.
        """
        store = self._get_store()
        key = f"{_KEY_PREFIX}{wrapper_id}"
        now = time.time()
        window_start = now - self._window

        # 1. Prune entries outside the current window
        store.zremrangebyscore(key, '-inf', window_start)

        # 2. Count remaining entries
        count = store.zcard(key)

        # 3. Allow or deny
        if count >= self._limit:
            logger.debug(
                "[WrapperRateLimiter] Rate limit exceeded for %s (%d/%d in %ds window)",
                wrapper_id, count, self._limit, self._window,
            )
            return False

        # 4. Record this signal's timestamp (use uuid suffix for uniqueness
        #    when two signals arrive at exactly the same instant)
        member = f"{now}:{uuid.uuid4().hex}"
        store.zadd(key, {member: now})
        # Keep the ZSET alive for the window duration plus a small buffer
        store.expire(key, self._window + 10)

        return True

    def remaining(self, wrapper_id: str) -> int:
        """Return the number of signal slots remaining in the current window.

        This is a read-only check — it does **not** consume a slot.

        Args:
            wrapper_id: The stable identifier of the calling wrapper.

        Returns:
            Non-negative integer: ``max(0, limit - count_in_window)``.
        """
        store = self._get_store()
        key = f"{_KEY_PREFIX}{wrapper_id}"
        now = time.time()
        window_start = now - self._window

        store.zremrangebyscore(key, '-inf', window_start)
        count = store.zcard(key)
        return max(0, self._limit - count)
