"""Persistent hook dedup — flags survive process restarts.

Uses SQLite as the primary store with MemoryStore as a read-through
cache.  Replaces the MemoryStore-only ``store.get()``/``store.setex()``
pattern that loses all dedup flags on restart.
"""

import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

_TABLE = "hook_dedup"
_ENSURED = False


def _ensure_table():
    """Create the hook_dedup table if it doesn't exist yet."""
    global _ENSURED
    if _ENSURED:
        return
    from services.database_service import get_shared_db_service
    with get_shared_db_service().connection() as conn:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
            "  key TEXT PRIMARY KEY,"
            "  expires_at TEXT NOT NULL"
            ")"
        )
    _ENSURED = True


def is_fired(key: str) -> bool:
    """Check whether a dedup flag is set and not expired.

    Fast path: MemoryStore cache.  Slow path: SQLite.
    """
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        if store.get(key):
            return True
    except Exception:
        pass

    try:
        _ensure_table()
        from services.database_service import get_shared_db_service
        from services.time_utils import utc_now
        now_iso = utc_now().isoformat()
        with get_shared_db_service().connection() as conn:
            row = conn.execute(
                f"SELECT 1 FROM {_TABLE} WHERE key = ? AND expires_at > ?",
                (key, now_iso),
            ).fetchone()
        if row:
            try:
                store = MemoryClientService.create_connection()
                store.setex(key, 300, "1")
            except Exception:
                pass
            return True
    except Exception as exc:
        logger.debug("hook_dedup.is_fired(%s): %s", key, exc)
    return False


def mark_fired(key: str, ttl_seconds: int) -> None:
    """Mark a dedup flag in both SQLite (persistent) and MemoryStore (cache)."""
    from services.time_utils import utc_now

    expires_at = (utc_now() + timedelta(seconds=ttl_seconds)).isoformat()

    try:
        _ensure_table()
        from services.database_service import get_shared_db_service
        with get_shared_db_service().connection() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {_TABLE} (key, expires_at) "
                "VALUES (?, ?)",
                (key, expires_at),
            )
            conn.execute(
                f"DELETE FROM {_TABLE} WHERE expires_at <= ?",
                (utc_now().isoformat(),),
            )
    except Exception as exc:
        logger.debug("hook_dedup.mark_fired(%s) SQLite: %s", key, exc)

    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        store.setex(key, ttl_seconds, "1")
    except Exception as exc:
        logger.debug("hook_dedup.mark_fired(%s) MemoryStore: %s", key, exc)
