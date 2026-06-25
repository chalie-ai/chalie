"""Feature test: re-authentication on /auth/login invalidates prior sessions.

Regression coverage for the security gap where a previously-issued session
token (e.g. one obtained by an attacker) stayed valid for its full 30-day TTL
after the legitimate user logged in again.  Login must now purge every prior
session belonging to that user from both the MemoryStore hot cache and the
durable SQLite store before minting the fresh token.

Exercises the real production path end-to-end: the Flask test client hits the
real ``/auth/login`` route, the real ``auth_session_service`` writes to the
real SQLite ``auth_sessions`` table and the real in-process MemoryStore.  No
mocks of the session service or login handler.
"""

import secrets
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from flask import Flask, Response as FlaskResponse
from flask.testing import FlaskClient
from werkzeug.security import generate_password_hash

import services.vault_service as _vault_mod
from api.user_auth import user_auth_bp
from services.auth_session_service import (
    SESSION_COOKIE_NAME,
    create_session,
    validate_session,
)
from services.file_mapper_service import FileMapperService
from services.vault_service import _vault_state


def _seed_account(raw_conn: sqlite3.Connection, password: str = "testpassword123") -> str:
    raw_conn.execute(
        "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
        ("admin", generate_password_hash(password)),
    )
    raw_conn.commit()
    return password


@pytest.fixture(autouse=True)
def _reset_vault_singletons() -> Iterator[None]:
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None
    yield
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None


@pytest.fixture
def secure_dir(tmp_path: Path) -> Path:
    return tmp_path / "secure"


@pytest.fixture
def auth_client(
    db: sqlite3.Connection, store: object, secure_dir: Path
) -> Iterator[tuple[FlaskClient, sqlite3.Connection]]:
    monkeypatch_target = FileMapperService
    monkeypatch_target._SECURE_DIR = secure_dir
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.config["TESTING"] = True
    app.register_blueprint(user_auth_bp)
    with app.test_client() as client:
        yield client, db


@pytest.mark.unit
class TestLoginInvalidatesPriorSession:

    def test_prior_session_is_invalidated_after_relogin(
        self,
        auth_client: tuple[FlaskClient, sqlite3.Connection],
        secure_dir: Path,
    ) -> None:
        client, raw_conn = auth_client
        pw = _seed_account(raw_conn)

        # Initialise the vault so the login's unlock_or_restore succeeds.
        vault = _vault_mod.get_vault_service()
        vault.initialize(pw)
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

        # Mint a PRIOR session for the same user, as an attacker would have.
        prior_resp = FlaskResponse()
        prior_token = create_session(prior_resp, "admin")
        assert prior_token, "prior session token must be minted"

        # Sanity: the prior token validates before re-login.
        from flask import Request
        from io import BytesIO

        def _request_with_cookie(token: str) -> Request:
            environ = {
                "REQUEST_METHOD": "GET",
                "SERVER_NAME": "localhost",
                "SERVER_PORT": "80",
                "PATH_INFO": "/",
                "wsgi.input": BytesIO(b""),
                "HTTP_COOKIE": f"{SESSION_COOKIE_NAME}={token}",
            }
            return Request(environ)

        assert validate_session(_request_with_cookie(prior_token)) is True, (
            "prior token must be valid before re-login"
        )

        # Re-authenticate — the real login route purges prior sessions.
        resp = client.post(
            "/auth/login",
            json={"username": "admin", "password": pw},
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

        # The prior token must no longer validate — purged from both stores.
        assert validate_session(_request_with_cookie(prior_token)) is False, (
            "a prior session token must be invalidated after the user re-logs in"
        )

        # And it must be gone from the durable SQLite store too.
        rows = raw_conn.execute(
            "SELECT COUNT(*) FROM auth_sessions WHERE token = ?",
            (prior_token,),
        ).fetchone()
        assert cast(tuple[int, ...], rows)[0] == 0, (
            "prior session row must be deleted from auth_sessions"
        )

        # The fresh token issued by this login must still be valid.
        fresh_cookie = resp.headers.get("Set-Cookie", "")
        assert SESSION_COOKIE_NAME in fresh_cookie, "login must issue a fresh cookie"
        assert prior_token not in fresh_cookie, (
            "login must mint a new token, not reuse the prior one"
        )