"""
Session-based authentication middleware with bearer token support.

Decorator @require_auth checks for a valid chalie_session cookie first,
then falls back to a Bearer token via WrapperAuthService.
Sessions are stored in MemoryStore via services.auth_session_service.

``require_session`` is kept as a backward-compatible alias for
``require_auth``.
"""

import logging
from functools import wraps
from flask import request, jsonify, g

logger = logging.getLogger(__name__)


def require_auth(f):
    """Decorator that enforces authentication via cookie session or bearer token.

    Tries the cookie session first (existing path).  If that fails, tries
    validating a ``Bearer <token>`` header via ``WrapperAuthService``.  On
    bearer success, ``g.wrapper_id`` is set to the wrapper's stable
    identifier.  For cookie auth, ``g.wrapper_id`` is ``None``.

    Returns 401 only when both methods fail.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        from services.auth_session_service import validate_session

        # Try cookie session first
        if validate_session(request):
            g.wrapper_id = None
            return f(*args, **kwargs)

        # Try bearer token
        try:
            from services.wrapper_auth_service import WrapperAuthService
            from services.database_service import get_shared_db_service
            svc = WrapperAuthService(get_shared_db_service())
            wrapper_id = svc.validate_bearer(request)
            if wrapper_id:
                g.wrapper_id = wrapper_id
                return f(*args, **kwargs)
        except Exception as e:
            logger.debug("[Auth] Bearer token validation failed: %s", e)

        return jsonify({"error": "Authentication required"}), 401

    return decorated


# Backward-compatible alias
require_session = require_auth
