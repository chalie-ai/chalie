"""Feature test: GET /api/auth/status does NOT mask has_session when the vault
is locked (#1878 part 2).

A backend restart seals the vault (the DEK lives only in process memory), but
the session cookie is still valid. Previously the status endpoint collapsed
those two distinct states — force-setting ``has_session = False`` whenever the
vault was locked — so the frontend always hard-redirected to /login/ instead of
showing the in-place UnlockVault overlay. Now ``has_session`` truthfully
reflects cookie validity, and ``vault_state`` carries the seal state separately.

Real production path, zero mocks on the handler itself:
  * real create_app() Flask app;
  * the real db SQLite fixture;
  * ``validate_session`` is patched True (via the authed_client fixture) and
    ``_get_vault_state`` is patched to ``"locked"`` — the exact two inputs the
    masking line used to combine.

Against the masking tree this test fails RED (has_session comes back False);
it passes GREEN once the masking is removed.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient
from unittest.mock import patch

pytestmark = pytest.mark.unit


class TestAuthStatusVaultLocked:
    def test_valid_session_with_locked_vault_keeps_has_session_true(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, db_conn, _store = authed_client
        # A master account is required for the frontend to treat a locked vault
        # as "re-seal" rather than "first run"; seed one so the payload is
        # realistic.
        db_conn.execute(
            "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
            ("alice", "x"),
        )
        db_conn.commit()

        # authed_client patches validate_session → True; force the vault sealed
        # (the post-restart state) — the combination the masking line collapsed.
        with patch("api.user_auth._get_vault_state", return_value="locked"):
            resp = client.get('/api/auth/status')

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["has_session"] is True, (
            "has_session must stay True when the cookie is valid but the vault is "
            "locked — the frontend distinguishes re-seal (unseal in place) from "
            "session expiry (redirect to /login/) on this field."
        )
        assert body["vault_state"] == "locked"
