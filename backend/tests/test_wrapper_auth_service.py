import json
import sqlite3
from typing import cast
import pytest
from flask import Flask, request

from services.wrapper_auth_service import WrapperAuthService, _hash_token

pytestmark = pytest.mark.unit


@pytest.fixture
def svc(db: sqlite3.Connection) -> WrapperAuthService:
    """WrapperAuthService using the real DB fixture (no arg = uses get_shared_db_service)."""
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

    def test_capabilities_round_trip(self, svc: WrapperAuthService, db: sqlite3.Connection) -> None:
        caps: dict[str, object] = {"signals": ["context_change", "location_change"]}
        _, wid = svc.create_token("W1", capabilities=caps)
        row = db.execute(
            "SELECT capabilities FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
        ).fetchone()
        assert json.loads(row[0]) == caps


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


class TestCheckPermission:
    def test_listed_resource_granted(self, svc: WrapperAuthService) -> None:
        perms = {"query": ["memory", "threads"], "update": ["memory"], "broadcast": True}
        _, wid = svc.create_token("W1", permissions=perms)
        assert svc.check_permission(wid, "query", "memory") is True
        assert svc.check_permission(wid, "broadcast", "") is True

    def test_unlisted_resource_denied(self, svc: WrapperAuthService) -> None:
        perms = {"query": ["memory"], "update": [], "broadcast": False}
        _, wid = svc.create_token("W1", permissions=perms)
        assert svc.check_permission(wid, "query", "tools") is False
        assert svc.check_permission(wid, "broadcast", "") is False

    def test_missing_and_revoked_wrapper_returns_false(self, svc: WrapperAuthService) -> None:
        assert svc.check_permission("nonexistent", "query", "memory") is False
        _, wid = svc.create_token("W1")
        svc.revoke(wid)
        assert svc.check_permission(wid, "query", "memory") is False


class TestListAndGet:
    def test_get_returns_wrapper(self, svc: WrapperAuthService) -> None:
        caps: dict[str, object] = {"signals": ["x"]}
        _, wid = svc.create_token("MyWrapper", capabilities=caps)
        result = svc.get_wrapper(wid)
        assert result is not None
        assert result["name"] == "MyWrapper"
        assert result["capabilities"] == caps


class TestUpdateCapabilities:
    def test_update_persists_new_capabilities(self, svc: WrapperAuthService) -> None:
        _, wid = svc.create_token("W1", capabilities=cast(dict[str, object], {"signals": ["old"]}))
        new_caps: dict[str, object] = {"signals": ["new1", "new2", "new3"]}
        svc.update_capabilities(wid, new_caps)
        result = svc.get_wrapper(wid)
        assert cast(dict[str, object], result)["capabilities"] == new_caps



