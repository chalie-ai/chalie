"""Feature tests for the per-request pre-flight login.

An install carrying a ``credentials.json`` at its root logs itself in on the
first request that arrives, so no login page is ever shown. Everything here
runs the real ``create_app()`` with the real before/after request hooks, a real
SQLite database and a real vault — the only redirection is the credentials-file
path, pointed at ``tmp_path`` so the developer's own file is never read.
"""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from flask.testing import FlaskClient
from werkzeug.security import generate_password_hash

import services.vault_service as _vault_mod
from contracts.constants.auth import SESSION_COOKIE_NAME
from services.file_mapper_service import FileMapperService
from services.vault_service import _vault_state

PASSWORD = "correct-horse-battery-staple"


@pytest.fixture(autouse=True)
def _reset_vault_singletons() -> Iterator[None]:
    """Drop the cached VaultService so it is rebuilt against the per-test DB."""
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None
    yield
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None


@pytest.fixture
def credentials_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the credentials-file lookup at a temp path (absent by default)."""
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(
        FileMapperService, "get_dev_credentials_path", classmethod(lambda cls: path)
    )
    return path


@pytest.fixture
def client(db: sqlite3.Connection, store: object, credentials_path: Path) -> Iterator[FlaskClient]:
    """Real app + real DB, with a master account and an initialised vault."""
    from api import create_app

    db.execute(
        "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
        ("admin", generate_password_hash(PASSWORD)),
    )
    db.commit()

    _vault_mod.get_vault_service().initialize(PASSWORD)
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _write_credentials(path: Path, username: str, password: str) -> None:
    path.write_text(json.dumps({"username": username, "password": password}), encoding="utf-8")


@pytest.mark.unit
def test_no_credentials_file_leaves_the_request_unauthenticated(client: FlaskClient) -> None:
    """The production case: no file, no auto-login, vault stays sealed."""
    resp = client.get("/api/auth/status")

    assert resp.status_code == 200
    assert resp.get_json()["has_session"] is False
    assert resp.get_json()["vault_state"] == "locked"
    assert SESSION_COOKIE_NAME not in resp.headers.get("Set-Cookie", "")


@pytest.mark.unit
def test_credentials_file_logs_in_and_unlocks_on_the_very_first_request(
    client: FlaskClient, credentials_path: Path
) -> None:
    """The first request both authenticates itself and carries the cookie out.

    The session cookie can only be attached on the way out, so the request that
    triggers the login must still report itself as authenticated — otherwise the
    frontend's first status poll would bounce the user to the login page.
    """
    _write_credentials(credentials_path, "admin", PASSWORD)

    resp = client.get("/api/auth/status")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["has_session"] is True, "the request that logged in must not report 401"
    assert body["vault_state"] == "unlocked", "login must unlock the vault, not just authenticate"
    assert SESSION_COOKIE_NAME in resp.headers.get("Set-Cookie", "")


@pytest.mark.unit
def test_the_issued_cookie_authenticates_the_next_request(
    client: FlaskClient, credentials_path: Path
) -> None:
    """A follow-up request is authenticated by the real cookie, not the flag."""
    _write_credentials(credentials_path, "admin", PASSWORD)
    client.get("/api/auth/status")

    # Remove the file: only the previously issued cookie can carry this request.
    credentials_path.unlink()
    resp = client.get("/api/auth/status")

    assert resp.get_json()["has_session"] is True


@pytest.mark.unit
def test_wrong_password_in_the_file_never_authenticates(
    client: FlaskClient, credentials_path: Path, db: sqlite3.Connection
) -> None:
    """A bad credentials file leaves the vault sealed and the account intact."""
    _write_credentials(credentials_path, "admin", "not-the-password")

    resp = client.get("/api/auth/status")

    body = resp.get_json()
    assert body["has_session"] is False
    assert body["vault_state"] == "locked"
    assert SESSION_COOKIE_NAME not in resp.headers.get("Set-Cookie", "")
    assert db.execute("SELECT COUNT(*) FROM master_account").fetchone()[0] == 1


@pytest.mark.unit
def test_unknown_username_in_the_file_never_authenticates(
    client: FlaskClient, credentials_path: Path
) -> None:
    """A username with no master_account row is refused like any bad login."""
    _write_credentials(credentials_path, "nobody", PASSWORD)

    body = client.get("/api/auth/status").get_json()

    assert body["has_session"] is False
    assert body["vault_state"] == "locked"


@pytest.mark.unit
def test_a_vault_that_will_not_open_never_mints_a_session(
    client: FlaskClient, credentials_path: Path, db: sqlite3.Connection
) -> None:
    """Right password, dead vault: no cookie, and no auth_sessions row — ever.

    ``login`` reports "locked" when the credentials match but the vault will not
    open. That state never resolves on its own (the account is never deleted),
    so if the pre-flight treated it as success it would mint a fresh session on
    every cookie-less request forever, growing ``auth_sessions`` without bound.
    """
    _write_credentials(credentials_path, "admin", PASSWORD)
    # Drop the config row AND the key backups: with a backup left in place the
    # vault recovers from it and legitimately unlocks.
    db.execute("DELETE FROM vault_config")
    db.commit()
    for backup in FileMapperService.list_vault_backups():
        backup.unlink()
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None

    for _ in range(3):
        resp = client.get("/api/auth/status")
        assert SESSION_COOKIE_NAME not in resp.headers.get("Set-Cookie", "")
        assert resp.get_json()["has_session"] is False

    rows = db.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0]
    assert rows == 0, f"a vault that cannot open minted {rows} session rows"


@pytest.mark.unit
def test_malformed_credentials_file_does_not_break_the_endpoint(
    client: FlaskClient, credentials_path: Path
) -> None:
    """Unparseable JSON is logged and skipped — it must not 500 every route."""
    credentials_path.write_text("{not json", encoding="utf-8")

    resp = client.get("/api/auth/status")

    assert resp.status_code == 200
    assert resp.get_json()["has_session"] is False


@pytest.mark.unit
def test_an_authenticated_request_never_consults_the_credentials_file(
    client: FlaskClient, credentials_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlocked vault + live session is checked first; the file plays no part.

    Once the client is logged in, the pre-flight must short-circuit before
    ``try_login`` — so a file lookup here would blow the request up, proving
    the order rather than assuming it.
    """
    _write_credentials(credentials_path, "admin", PASSWORD)
    client.get("/api/auth/status")

    def _exploding_path() -> Path:
        raise AssertionError("pre-flight consulted credentials.json on an authenticated request")

    monkeypatch.setattr(
        FileMapperService, "get_dev_credentials_path", classmethod(lambda cls: _exploding_path())
    )
    resp = client.get("/api/auth/status")

    assert resp.status_code == 200
    assert resp.get_json()["has_session"] is True


@pytest.mark.unit
def test_a_client_holding_a_session_is_never_handed_a_second_cookie(
    client: FlaskClient, credentials_path: Path
) -> None:
    """One client, one session: repeat requests must not mint more rows."""
    _write_credentials(credentials_path, "admin", PASSWORD)
    first = client.get("/api/auth/status")
    assert SESSION_COOKIE_NAME in first.headers.get("Set-Cookie", "")

    second = client.get("/api/auth/status")

    assert second.get_json()["has_session"] is True
    assert SESSION_COOKIE_NAME not in second.headers.get("Set-Cookie", "")
