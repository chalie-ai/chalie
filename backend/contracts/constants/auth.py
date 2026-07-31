"""Canonical authentication constants — the one home for the session cookie,
its TTL and store key prefix, and the login rate limit.

Both the session layer (``services.auth_session_service``) and the login API
(``api.user_auth``) read from this module so there is exactly one source of
truth for these values. Model-free by design: ``contracts`` is a low-level
package that never imports from ``services`` or ``api`` (avoids import cycles).
"""

from __future__ import annotations

SESSION_COOKIE_NAME = 'chalie_session'
SESSION_TTL = 30 * 24 * 60 * 60  # 30 days in seconds
SESSION_KEY_PREFIX = 'auth_session:'  # MemoryStore key namespace for live sessions

LOGIN_RATE_LIMIT = 10           # attempts
LOGIN_RATE_WINDOW_SECONDS = 60  # window the attempt count is measured over
