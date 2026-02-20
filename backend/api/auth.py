"""
Session-based authentication middleware.

Decorator @require_session checks for a valid chalie_session cookie.
Sessions are stored in Redis via services.auth_session_service.
"""

import logging
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger(__name__)


def require_session(f):
    """Decorator that enforces cookie session auth."""
    @wraps(f)
    def decorated(*args, **kwargs):
        from services.auth_session_service import validate_session

        if not validate_session(request):
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)

    return decorated
