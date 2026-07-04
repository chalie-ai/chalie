"""Feature test: GET /auth/username returns the master LOGIN username
(master_account.username) to a human dashboard cookie session ONLY.

The QR pairing payload embeds this username so the mobile UnlockVault screen
needs only a password; the username is the WHERE filter POST /auth/login uses.
It is write-only at the API boundary today (no endpoint returns it), so this
endpoint is new. It is cookie-session-only: a wrapper bearer must not read the
master login credential.

Real production path, zero mocks (same shape as test_voice_auth.py):
  * real create_app() Flask app + the real require_auth cookie→bearer guard;
  * the real db SQLite fixture (patches the DB singleton);
  * a real wrapper token minted by the real WrapperAuthService.create_token —
    the exact call POST /api/wrappers makes.

Against the pre-impl tree there is no /auth/username route, so Flask answers
404 and every assertion fails RED. All pass GREEN once the endpoint lands.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient

from services.wrapper_auth_service import WrapperAuthService

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _enable_internal_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    # /auth/username is a native-pairing surface gated behind CHALIE_INTERNAL_DEV;
    # this suite exercises the feature, so it runs with the gate open.
    monkeypatch.setenv("CHALIE_INTERNAL_DEV", "1")


def _make_client() -> FlaskClient:
    # Real, unauthenticated app — require_auth runs its real cookie+bearer
    # path (the authed_client fixture patches validate_session, which would
    # hide the guard the bearer/unauth arms exist to prove).
    from api import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app.test_client()


class TestAuthUsername:
    def test_cookie_session_returns_master_username(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        # authed_client patches validate_session → cookie auth; require_auth
        # sets g.wrapper_id=None so _cookie_only passes and the handler runs.
        client, db_conn, _store = authed_client
        db_conn.execute(
            "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
            ("alice", "x"),
        )
        db_conn.commit()

        resp = client.get('/api/auth/username')
        assert resp.status_code == 200
        assert resp.get_json() == {"username": "alice"}

    def test_bearer_wrapper_is_forbidden(self, db: sqlite3.Connection) -> None:
        # A real wrapper bearer authenticates require_auth (sets g.wrapper_id),
        # so _cookie_only rejects it 403 — the master login username is NOT
        # readable by a wrapper token. (No master_account row needed: the guard
        # rejects before the handler's SELECT.)
        raw_token, _wrapper_id = WrapperAuthService().create_token(
            name="auth-username-test",
        )
        resp = _make_client().get(
            '/api/auth/username', headers={"Authorization": f"Bearer {raw_token}"},
        )
        assert resp.status_code == 403

    def test_unauthenticated_is_rejected(self, db: sqlite3.Connection) -> None:
        resp = _make_client().get('/api/auth/username')
        assert resp.status_code == 401
        assert resp.get_json() == {"error": "Authentication required"}
