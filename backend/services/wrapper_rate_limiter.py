import logging
import time
import uuid
from typing import Optional, cast

from services.memory_client import MemoryClientService
from services.memory_store import MemoryStore

logger = logging.getLogger(__name__)

# MemoryStore key pattern for rate-limit windows
_KEY_PREFIX = "wrapper_rate:"

# Default limits
DEFAULT_LIMIT = 100       # signals
DEFAULT_WINDOW = 60       # seconds


class WrapperRateLimiter:
    def __init__(
        self,
        limit: int = DEFAULT_LIMIT,
        window_seconds: int = DEFAULT_WINDOW,
        store: Optional[MemoryStore] = None,
    ):
        self._limit = limit
        self._window = window_seconds
        self._store = store

    def _get_store(self) -> MemoryStore:
        if self._store is not None:
            return self._store
        return MemoryClientService.create_connection()

    def is_allowed(self, wrapper_id: str) -> bool:
        """Use `'__chat_ui__'` for cookie-authenticated (non-wrapper) callers."""
        store = self._get_store()
        key = f"{_KEY_PREFIX}{wrapper_id}"
        now = time.time()
        window_start = now - self._window

        # 1. Prune entries outside the current window
        store.zremrangebyscore(key, cast(float, '-inf'), window_start)

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
