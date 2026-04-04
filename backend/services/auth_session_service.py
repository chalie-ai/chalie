"""
Cookie-based session management.
Sessions are stored in BOTH MemoryStore (hot cache) and SQLite (durable).
MemoryStore is checked first for speed; SQLite is the fallback that survives
container restarts.

Cookie name: chalie_session (HTTP-only, SameSite=Lax)
"""
import os
import secrets
import logging

from services.time_utils import utc_now

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = 'chalie_session'
SESSION_TTL = 30 * 24 * 60 * 60  # 30 days in seconds
SESSION_KEY_PREFIX = 'auth_session:'


def _persist_session_to_sqlite(token: str):
    """Write session to SQLite so it survives MemoryStore wipes (restart)."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        expires_at = utc_now().isoformat()
        db.execute(
            """INSERT OR REPLACE INTO auth_sessions (token, created_at, expires_at)
               VALUES (?, ?, datetime('now', '+30 days'))""",
            (token, utc_now().isoformat())
        )
    except Exception as e:
        logger.error(f"[Session] SQLite persist failed: {e}")


def _delete_session_from_sqlite(token: str):
    """Remove session from SQLite."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        db.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
    except Exception as e:
        logger.debug(f"[Session] SQLite delete failed: {e}")


def _validate_session_in_sqlite(token: str) -> bool:
    """Check SQLite for a valid (non-expired) session and rehydrate MemoryStore."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        rows = db.fetch_all(
            """SELECT token FROM auth_sessions
               WHERE token = ? AND expires_at > datetime('now')
               LIMIT 1""",
            (token,)
        )
        if rows:
            # Rehydrate MemoryStore so subsequent requests are fast
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            store.setex(f"{SESSION_KEY_PREFIX}{token}", SESSION_TTL, "1")
            logger.info("[Session] Rehydrated session from SQLite (post-restart)")
            return True
        return False
    except Exception as e:
        logger.error(f"[Session] SQLite validate failed: {e}")
        return False


def create_session(response) -> str:
    """Create a new session, set cookie on response, return token.

    Generates a cryptographically secure random token, stores it in both
    MemoryStore (fast path) and SQLite (durable), and attaches the
    ``chalie_session`` HTTP-only cookie to the given response object.

    Args:
        response: Flask (or compatible) response object on which the session
            cookie will be set.

    Returns:
        The newly created session token string.
    """
    from services.memory_client import MemoryClientService

    token = secrets.token_urlsafe(32)
    store = MemoryClientService.create_connection()
    store.setex(f"{SESSION_KEY_PREFIX}{token}", SESSION_TTL, "1")

    # Persist to SQLite so session survives restart
    _persist_session_to_sqlite(token)

    secure = os.environ.get('COOKIE_SECURE', 'false').lower() == 'true'
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=SESSION_TTL,
        httponly=True,
        samesite='Lax',
        secure=secure,
    )
    logger.info("[Session] Created new session")
    return token


def validate_session(request) -> bool:
    """Return True if the request carries a valid session cookie.

    Checks MemoryStore first (fast path). If MemoryStore misses (e.g. after
    restart), falls back to SQLite and rehydrates MemoryStore on hit.

    Args:
        request: Flask (or compatible) request object providing access to
            cookies.

    Returns:
        True if the session token is present and valid, False otherwise.
    """
    from services.memory_client import MemoryClientService

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return False

    # Fast path: check MemoryStore
    store = MemoryClientService.create_connection()
    key = f"{SESSION_KEY_PREFIX}{token}"
    exists = store.exists(key)
    if exists:
        store.expire(key, SESSION_TTL)  # Slide the TTL
        return True

    # Slow path: check SQLite (handles post-restart scenario)
    return _validate_session_in_sqlite(token)


def destroy_session(request, response):
    """Invalidate the session and clear the cookie.

    Deletes the session from both MemoryStore and SQLite, then instructs
    the response to delete the ``chalie_session`` cookie from the client.

    Args:
        request: Flask (or compatible) request object providing access to
            cookies.
        response: Flask (or compatible) response object on which the cookie
            deletion will be applied.
    """
    from services.memory_client import MemoryClientService

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        store = MemoryClientService.create_connection()
        store.delete(f"{SESSION_KEY_PREFIX}{token}")
        _delete_session_from_sqlite(token)
    response.delete_cookie(SESSION_COOKIE_NAME)
    logger.info("[Session] Destroyed session")


def cleanup_expired_sessions():
    """Delete expired sessions from SQLite. Called periodically or on startup."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        db.execute("DELETE FROM auth_sessions WHERE expires_at <= datetime('now')")
    except Exception as e:
        logger.debug(f"[Session] Expired session cleanup failed: {e}")