"""Cookie-based session management.

Sessions are stored in both MemoryStore (hot cache) and SQLite (durable).
MemoryStore is checked first for speed; SQLite is the fallback that
survives container restarts. Cookie: ``chalie_session`` (HTTP-only,
SameSite=Lax).
"""
import hashlib
import logging
import secrets

from flask import Request, Response

from services.time_utils import utc_now

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = 'chalie_session'
SESSION_TTL = 30 * 24 * 60 * 60  # 30 days in seconds
SESSION_KEY_PREFIX = 'auth_session:'


def _hash_session_token(raw_token: str) -> str:
    """SHA-256 of the raw session token; the SQLite row stores only the hash."""
    return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()


def _persist_session_to_sqlite(token: str) -> None:
    """Write session to SQLite so it survives MemoryStore wipes (restart).

    Only the SHA-256 hash of the token is persisted — never the raw token —
    mirroring the wrapper-auth path so a DB leak cannot yield live tokens.
    """
    try:
        from services.database import Database
        with Database.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO auth_sessions (token, created_at, expires_at)
                   VALUES (?, ?, datetime('now', '+30 days'))""",
                (_hash_session_token(token), utc_now().isoformat())
            )
    except Exception as e:
        logger.error(f"[Session] SQLite persist failed: {e}")


def _delete_session_from_sqlite(token: str) -> None:
    """Remove session from SQLite."""
    try:
        from services.database import Database
        with Database.transaction() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE token = ?", (_hash_session_token(token),))
    except Exception as e:
        logger.debug(f"[Session] SQLite delete failed: {e}")


def _validate_session_in_sqlite(token: str) -> bool:
    """Check SQLite for a valid (non-expired) session and rehydrate MemoryStore."""
    try:
        from services.database import Database
        rows = Database.conn().execute(
            """SELECT token FROM auth_sessions
               WHERE token = ? AND expires_at > datetime('now')
               LIMIT 1""",
            (_hash_session_token(token),)
        ).fetchall()
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


def _cookie_secure() -> bool:
    """Mark the session cookie ``Secure`` only when the deployment serves HTTPS.

    Sourced from the ``ssl_enabled`` DB setting (the same flag that drives TLS
    serving) so cookie scope tracks the wire scheme without any environment var.
    """
    from models.setting import Setting
    return Setting.get_bool(Setting.SSL_ENABLED)


def create_session(response: Response) -> str:
    """Generates a cryptographically secure random token, stores it in both
    MemoryStore (fast path) and SQLite (durable), and attaches the
    ``chalie_session`` HTTP-only cookie to the given response object."""
    from services.memory_client import MemoryClientService

    token = secrets.token_urlsafe(32)
    store = MemoryClientService.create_connection()
    store.setex(f"{SESSION_KEY_PREFIX}{token}", SESSION_TTL, "1")

    _persist_session_to_sqlite(token)

    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=SESSION_TTL,
        httponly=True,
        samesite='Lax',
        secure=_cookie_secure(),
    )
    logger.info("[Session] Created new session")
    return token


def validate_session(request: Request) -> bool:
    """Checks MemoryStore first (fast path). If MemoryStore misses (e.g.
    after restart), falls back to SQLite and rehydrates MemoryStore on hit."""
    from services.memory_client import MemoryClientService

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return False

    store = MemoryClientService.create_connection()
    key = f"{SESSION_KEY_PREFIX}{token}"
    exists = store.exists(key)
    if exists:
        store.expire(key, SESSION_TTL)  # Slide the TTL
        return True

    return _validate_session_in_sqlite(token)


def destroy_session(request: Request, response: Response) -> None:
    """Deletes the session from both MemoryStore and SQLite, then deletes
    the ``chalie_session`` cookie from the client."""
    from services.memory_client import MemoryClientService

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        store = MemoryClientService.create_connection()
        store.delete(f"{SESSION_KEY_PREFIX}{token}")
        _delete_session_from_sqlite(token)
    response.delete_cookie(SESSION_COOKIE_NAME)
    logger.info("[Session] Destroyed session")


def cleanup_expired_sessions() -> None:
    """Delete expired sessions from SQLite. Called periodically or on startup."""
    try:
        from services.database import Database
        with Database.transaction() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE expires_at <= datetime('now')")
    except Exception as e:
        logger.debug(f"[Session] Expired session cleanup failed: {e}")