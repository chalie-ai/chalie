"""
Session-based authentication middleware with bearer token support.

Decorator @require_auth checks for a valid chalie_session cookie first,
then falls back to a Bearer token via WrapperAuthService.
Sessions are stored in MemoryStore via services.auth_session_service.

``require_session`` is kept as a backward-compatible alias for
``require_auth``.
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

from flask import request, g, abort

from services.feature_flags import internal_dev_enabled

if TYPE_CHECKING:
    from flask.typing import ResponseReturnValue

logger = logging.getLogger(__name__)


def internal_only(f: Callable[..., "ResponseReturnValue"]) -> Callable[..., "ResponseReturnValue"]:
    """Hide a route unless in-development features are enabled for this process.

    Answers 404 (not 403) when disabled, so a gated feature is indistinguishable
    from one that does not exist. Outermost decorator, above ``require_auth``, so
    the gate closes before authentication even runs.
    """
    @wraps(f)
    def decorated(*args: object, **kwargs: object) -> "ResponseReturnValue":
        if not internal_dev_enabled():
            abort(404)
        return f(*args, **kwargs)

    return decorated


def require_auth(f: Callable[..., "ResponseReturnValue"]) -> Callable[..., "ResponseReturnValue"]:
    """Decorator that enforces authentication via cookie session or bearer token.

    Tries the cookie session first (existing path).  If that fails, tries
    validating a ``Bearer <token>`` header via ``WrapperAuthService``.  On
    bearer success, ``g.wrapper_id`` is set to the wrapper's stable
    identifier.  For cookie auth, ``g.wrapper_id`` is ``None``.

    Returns 401 only when both methods fail.
    """
    @wraps(f)
    def decorated(*args: object, **kwargs: object) -> "ResponseReturnValue":
        from services.auth_session_service import AuthSessionService

        # Try cookie session first
        if AuthSessionService.validate_session(request):
            g.wrapper_id = None
            return f(*args, **kwargs)

        # Try bearer token
        try:
            from services.wrapper_auth_service import WrapperAuthService
            svc = WrapperAuthService()
            wrapper_id = svc.validate_bearer(request)
            if wrapper_id:
                g.wrapper_id = wrapper_id
                return f(*args, **kwargs)
        except Exception as e:
            logger.debug("[Auth] Bearer token validation failed: %s", e)

        return {"error": "Authentication required"}, 401

    return decorated


def _cookie_only(f: Callable[..., "ResponseReturnValue"]) -> Callable[..., "ResponseReturnValue"]:
    """Restrict a route to cookie-authenticated requests (no bearer tokens).

    Stacked under ``@require_auth``: once a request has authenticated, a bearer
    wrapper has ``g.wrapper_id`` set while a human cookie session leaves it
    ``None``. Used where only a human dashboard session may act — minting
    wrapper tokens, reading the master login username for device pairing —
    never a wrapper acting on its own behalf.
    """
    @wraps(f)
    def decorated(*args: object, **kwargs: object) -> "ResponseReturnValue":
        if getattr(g, "wrapper_id", None) is not None:
            return {"error": "This action requires cookie session auth"}, 403
        return f(*args, **kwargs)

    return decorated


# Backward-compatible alias
require_session = require_auth
