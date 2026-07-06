import sqlite3

import pytest
from flask import Flask, request

from services.wrapper_auth_service import WrapperAuthService, _hash_token

pytestmark = pytest.mark.unit


@pytest.fixture
def svc(db: sqlite3.Connection) -> WrapperAuthService:
    """WrapperAuthService against the real DB fixture (which points the Database gateway at this test's SQLite file)."""
    return WrapperAuthService()


@pytest.fixture(scope="session")
def app() -> Flask:
    """The real production Flask app (``create_app``). Its request context yields a
    genuine Werkzeug ``Request``, so ``validate_bearer`` parses real headers off the
    real production object — no mock standing in for Flask's header API."""
    from api import create_app
    return create_app()


class TestCreateToken:
    def test_returns_unique_raw_token_and_wrapper_id(self, svc: WrapperAuthService) -> None:
        raw1, wid1 = svc.create_token("W1")
        raw2, wid2 = svc.create_token("W2")
        assert isinstance(raw1, str) and len(raw1) > 20
        assert wid1.startswith("wrp_")
        assert wid1 != wid2
        assert raw1 != raw2

    def test_hash_is_stored_not_raw_token(self, svc: WrapperAuthService, db: sqlite3.Connection) -> None:
        raw, wid = svc.create_token("W1")
        row = db.execute(
            "SELECT token_hash FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
        ).fetchone()
        assert row is not None
        assert row[0] == _hash_token(raw)
        assert row[0] != raw


class TestValidateBearer:
    def test_valid_token_returns_wrapper_id_and_slides_last_seen(
        self, svc: WrapperAuthService, app: Flask, db: sqlite3.Connection
    ) -> None:
        raw, wid = svc.create_token("W1")
        with app.test_request_context(headers={"Authorization": f"Bearer {raw}"}):
            assert svc.validate_bearer(request) == wid
        # Downstream effect: a successful validation slides last_seen_at off NULL.
        row = db.execute(
            "SELECT last_seen_at FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
        ).fetchone()
        assert row[0] is not None

    def test_revoked_token_returns_none(self, svc: WrapperAuthService, app: Flask) -> None:
        raw, wid = svc.create_token("W1")
        svc.revoke(wid)
        with app.test_request_context(headers={"Authorization": f"Bearer {raw}"}):
            assert svc.validate_bearer(request) is None


class TestListAndGet:
    def test_get_returns_wrapper(self, svc: WrapperAuthService) -> None:
        _, wid = svc.create_token("MyWrapper")
        result = svc.get_wrapper(wid)
        assert result is not None
        assert result["name"] == "MyWrapper"
