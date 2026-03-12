"""
Session-based authentication middleware.

Decorator @require_session checks for a valid chalie_session cookie.
Sessions are stored in MemoryStore via services.auth_session_service.
"""

import logging
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger(__name__)


def require_session(f):
    """Decorator that enforces cookie session auth."""
    @wraps(f)
    def decorated(*args, **kwargs):
        """Validate the request session cookie before calling the wrapped view.

        Reads the ``chalie_session`` cookie from the current request, delegates
        validation to ``auth_session_service.validate_session``, and returns a
        401 JSON error response if the session is absent or invalid.

        Args:
            *args: Positional arguments forwarded to the wrapped view function.
            **kwargs: Keyword arguments forwarded to the wrapped view function.

        Returns:
            The return value of the wrapped view function ``f`` on success, or
            a ``({"error": "Authentication required"}, 401)`` JSON response if
            the session is invalid.
        """
        from services.auth_session_service import validate_session

        if not validate_session(request):
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)

    return decorated
