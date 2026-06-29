"""Feature test: the /voice/* endpoints enforce authentication.

The server binds 0.0.0.0, so before the ``@require_auth`` guard was added these
routes were reachable by any host on the LAN — free STT/TTS compute and a
``/voice/health`` response that leaked absolute model-file paths. Each test
drives the real Flask app built by ``create_app()`` through the real
``require_auth`` decorator (cookie-session → bearer fallback) and, where a
logged-in caller is needed, a real bearer token minted into the test DB. No
mocks, no auth bypass — the unauthenticated cases hit the production guard for
real, exactly as a LAN attacker would.
"""

import io
import sqlite3

import pytest
from flask.testing import FlaskClient

from services.wrapper_auth_service import WrapperAuthService


pytestmark = pytest.mark.unit


def _make_client() -> FlaskClient:
    # Real, unauthenticated app — require_auth runs its real cookie+bearer path
    # (the authed_client conftest fixture patches validate_session, which would
    # hide the very guard these tests exist to prove).
    from api import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app.test_client()


# ── The security contract: no session, no voice ─────────────────────────────
# Each of these FAILED before the guard existed — the routes answered 200/503,
# never 401 — and now returns the canonical require_auth rejection.

class TestVoiceEndpointsRejectUnauthenticated:
    def test_health_requires_auth(self, db: sqlite3.Connection) -> None:
        resp = _make_client().get('/api/voice/health')
        assert resp.status_code == 401
        assert resp.get_json() == {"error": "Authentication required"}

    def test_synthesize_requires_auth(self, db: sqlite3.Connection) -> None:
        resp = _make_client().post('/api/voice/synthesize', json={"text": "hello"})
        assert resp.status_code == 401
        assert resp.get_json() == {"error": "Authentication required"}

    def test_transcribe_requires_auth(self, db: sqlite3.Connection) -> None:
        # A would-be attacker POSTs audio; the guard rejects before the body runs.
        resp = _make_client().post(
            '/api/voice/transcribe',
            data={"file": (io.BytesIO(b"RIFF....WAVE"), "clip.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 401
        assert resp.get_json() == {"error": "Authentication required"}


# ── The other half of the contract: a valid session still gets through ───────
# Proves the guard lets legitimate traffic reach the handler (so the real
# frontend mic/TTS feature keeps working) rather than blocking everyone.

class TestVoiceEndpointsAllowAuthenticated:
    def test_health_with_bearer_reaches_handler(self, db: sqlite3.Connection) -> None:
        raw_token, _wrapper_id = WrapperAuthService().create_token(
            name="voice-auth-test",
        )
        resp = _make_client().get(
            '/api/voice/health', headers={"Authorization": f"Bearer {raw_token}"},
        )
        assert resp.status_code == 200
        # The real handler ran: it always reports a concrete status.
        assert resp.get_json()["status"] in ("ok", "loading", "unavailable")
