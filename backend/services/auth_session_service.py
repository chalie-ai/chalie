"""
Cookie-based session management.
Sessions are stored in Redis with a 30-day TTL.
Cookie name: chalie_session (HTTP-only, SameSite=Lax)
"""
import os
import secrets
import logging

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = 'chalie_session'
SESSION_TTL = 30 * 24 * 60 * 60  # 30 days in seconds
SESSION_KEY_PREFIX = 'auth_session:'


def create_session(response) -> str:
    """Create a new session, set cookie on response, return token."""
    from services.redis_client import RedisClientService

    token = secrets.token_urlsafe(32)
    redis = RedisClientService.create_connection()
    redis.setex(f"{SESSION_KEY_PREFIX}{token}", SESSION_TTL, "1")

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
    """Return True if the request carries a valid session cookie."""
    from services.redis_client import RedisClientService

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return False
    redis = RedisClientService.create_connection()
    key = f"{SESSION_KEY_PREFIX}{token}"
    exists = redis.exists(key)
    if exists:
        redis.expire(key, SESSION_TTL)  # Slide the TTL
    return bool(exists)


def destroy_session(request, response):
    """Invalidate the session and clear the cookie."""
    from services.redis_client import RedisClientService

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        redis = RedisClientService.create_connection()
        redis.delete(f"{SESSION_KEY_PREFIX}{token}")
    response.delete_cookie(SESSION_COOKIE_NAME)
    logger.info("[Session] Destroyed session")