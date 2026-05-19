"""Tests for WrapperAuthService — bearer token auth lifecycle."""

import json

import pytest
from unittest.mock import MagicMock

from services.wrapper_auth_service import WrapperAuthService, _hash_token

pytestmark = pytest.mark.unit


@pytest.fixture
def svc(db):
    """WrapperAuthService using the real DB fixture (no arg = uses get_shared_db_service)."""
    return WrapperAuthService()


def _make_flask_request(authorization_header: str = "") -> MagicMock:
    req = MagicMock()
    headers_mock = MagicMock()
    headers_mock.get = lambda key, default="": (
        authorization_header if key == "Authorization" else default
    )
    req.headers = headers_mock
    return req


class TestCreateToken:
    def test_returns_unique_raw_token_and_wrapper_id(self, svc):
        raw1, wid1 = svc.create_token("W1")
        raw2, wid2 = svc.create_token("W2")
        assert isinstance(raw1, str) and len(raw1) > 20
        assert wid1.startswith("wrp_")
        assert wid1 != wid2
        assert raw1 != raw2

    def test_hash_is_stored_not_raw_token(self, svc, db):
        raw, wid = svc.create_token("W1")
        row = db.execute(
            "SELECT token_hash FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
        ).fetchone()
        assert row is not None
        assert row[0] == _hash_token(raw)
        assert row[0] != raw

    def test_capabilities_round_trip(self, svc, db):
        caps = {"signals": ["context_change"], "intents": ["read_memory"]}
        _, wid = svc.create_token("W1", capabilities=caps)
        row = db.execute(
            "SELECT capabilities FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
        ).fetchone()
        assert json.loads(row[0]) == caps


class TestValidateBearer:
    def test_valid_token_returns_wrapper_id(self, svc):
        raw, wid = svc.create_token("W1")
        req = _make_flask_request(f"Bearer {raw}")
        assert svc.validate_bearer(req) == wid


    def test_revoked_token_returns_none(self, svc):
        raw, wid = svc.create_token("W1")
        svc.revoke(wid)
        req = _make_flask_request(f"Bearer {raw}")
        assert svc.validate_bearer(req) is None


class TestCheckPermission:
    def test_listed_resource_granted(self, svc):
        perms = {"query": ["memory", "threads"], "update": ["memory"], "broadcast": True}
        _, wid = svc.create_token("W1", permissions=perms)
        assert svc.check_permission(wid, "query", "memory") is True
        assert svc.check_permission(wid, "broadcast", "") is True

    def test_unlisted_resource_denied(self, svc):
        perms = {"query": ["memory"], "update": [], "broadcast": False}
        _, wid = svc.create_token("W1", permissions=perms)
        assert svc.check_permission(wid, "query", "tools") is False
        assert svc.check_permission(wid, "broadcast", "") is False

    def test_missing_and_revoked_wrapper_returns_false(self, svc):
        assert svc.check_permission("nonexistent", "query", "memory") is False
        _, wid = svc.create_token("W1")
        svc.revoke(wid)
        assert svc.check_permission(wid, "query", "memory") is False


class TestListAndGet:
    def test_get_returns_wrapper(self, svc):
        caps = {"signals": ["x"]}
        _, wid = svc.create_token("MyWrapper", capabilities=caps)
        result = svc.get_wrapper(wid)
        assert result is not None
        assert result["name"] == "MyWrapper"
        assert result["capabilities"] == caps


class TestUpdateCapabilities:
    def test_update_persists_new_capabilities(self, svc):
        _, wid = svc.create_token("W1", capabilities={"signals": ["old"]})
        new_caps = {"signals": ["new1", "new2"], "intents": ["write"]}
        svc.update_capabilities(wid, new_caps)
        result = svc.get_wrapper(wid)
        assert result["capabilities"] == new_caps



