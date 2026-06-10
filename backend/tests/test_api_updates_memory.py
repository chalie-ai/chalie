"""
Feature tests for POST /api/updates/memory.

Regression guard for commit 361466b8: handle_store() returns a ToolResult, not
a string.  The pre-fix code called .split() on the ToolResult object → AttributeError
→ broad except → HTTP 500 on every call.  The fix checks result.status instead.

These tests drive the real Flask entry point with real bearer-token auth against
a real SQLite database (via the ``db`` conftest fixture), and assert downstream
DB effects.  No collaborators are mocked.
"""

import pytest
from unittest.mock import patch

from services.wrapper_auth_service import WrapperAuthService


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helper — build a real, unauthenticated Flask test client.
# The ``authed_client`` conftest fixture short-circuits validate_session with a
# patch, which means bearer tokens are never evaluated.  We build the app
# ourselves so the real require_auth decorator runs the real bearer path.
# ---------------------------------------------------------------------------

def _make_client():
    """Return a real Flask test client with no session patches applied.

    The only infra patch is ``_get_or_generate_session_secret`` → it reads from
    disk (or generates a file) during app construction, which is irrelevant to
    bearer-auth tests and would fail if the file path is not writable in CI.
    """
    with patch('api._get_or_generate_session_secret', return_value='test-secret-updates'):
        from api import create_app
        app = create_app()
        app.config['TESTING'] = True
        return app.test_client()


# ---------------------------------------------------------------------------
# Test 1 — success path: HTTP 200 + data_graph row written
#
# Regression assertion: pre-fix, handle_store() returned a ToolResult whose
# .split() call raised AttributeError, was caught by ``except Exception``,
# and the endpoint returned 500.  This test would have FAILED (expected 200,
# got 500) on the pre-fix code and PASSES on HEAD.
# ---------------------------------------------------------------------------

class TestUpdateMemorySuccess:
    def test_success_returns_200_and_writes_data_graph_row(self, db):
        svc = WrapperAuthService()
        raw_token, _wrapper_id = svc.create_token(
            name="test-wrapper",
            permissions={"update": ["memory"]},
        )

        client = _make_client()
        resp = client.post(
            '/api/updates/memory',
            json={"content": "Alex prefers tea over coffee", "topic": "preferences"},
            headers={"Authorization": f"Bearer {raw_token}"},
        )

        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}

        # Downstream: a real row was written to data_graph by the store primitive.
        row = db.execute(
            "SELECT kind, value, source, active FROM data_graph "
            "WHERE kind = 'misc' AND value = ? AND source = ?",
            ("Alex prefers tea over coffee", "skill:memory:store:preferences"),
        ).fetchone()
        assert row is not None, "data_graph row was not written"
        assert row["kind"] == "misc"
        assert row["value"] == "Alex prefers tea over coffee"
        assert row["source"] == "skill:memory:store:preferences"
        assert row["active"] == 1


# ---------------------------------------------------------------------------
# Test 2 — genuine store failure: HTTP 422 with structured error body
#
# Dropping the data_graph table causes DataGraphService.store() to raise
# internally, which it catches and returns None → handle_store returns
# ToolResult.err(code="invalid-kind") → result.status == "error" → 422.
# This proves the result.status branch (the actual fix) fires correctly.
# ---------------------------------------------------------------------------

class TestUpdateMemoryStoreFailure:
    def test_genuine_store_failure_returns_422(self, db):
        svc = WrapperAuthService()
        raw_token, _wrapper_id = svc.create_token(
            name="test-wrapper-fail",
            permissions={"update": ["memory"]},
        )

        # Destroy the backing table so store() fails for real — no mocks.
        db.execute("DROP TABLE data_graph")
        db.commit()

        client = _make_client()
        resp = client.post(
            '/api/updates/memory',
            json={"content": "some content that cannot be stored"},
            headers={"Authorization": f"Bearer {raw_token}"},
        )

        assert resp.status_code == 422
        assert resp.get_json() == {"error": "Memory encoding failed"}


# ---------------------------------------------------------------------------
# Test 3 — no token → 401
#
# Guards against accidentally re-introducing a validate_session bypass that
# would let unauthenticated callers reach the store primitive.
# ---------------------------------------------------------------------------

class TestUpdateMemoryNoToken:
    def test_missing_auth_returns_401(self, db):
        client = _make_client()
        resp = client.post(
            '/api/updates/memory',
            json={"content": "should not be stored"},
        )

        assert resp.status_code == 401
