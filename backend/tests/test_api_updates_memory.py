# Regression guard for commit 361466b8: handle_store() returns a ToolResult, not
# a string.  The pre-fix code called .split() on the ToolResult object → AttributeError
# → broad except → HTTP 500 on every call.  The fix checks result.status instead.

import pytest

from services.wrapper_auth_service import WrapperAuthService


pytestmark = pytest.mark.unit


# Build a real, unauthenticated Flask test client — the authed_client conftest
# fixture short-circuits validate_session with a patch, so we build the app
# ourselves so require_auth runs the real bearer path.
def _make_client():
    from api import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app.test_client()


# --- Regression assertion: pre-fix, handle_store() returned a ToolResult,
# .split() raised AttributeError → caught by except Exception → HTTP 500.
# This would have FAILED (expected 200, got 500) on the pre-fix code.
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


# --- Dropping data_graph causes DataGraphService.store() to raise, returning
# None → handle_store returns ToolResult.err(code="invalid-kind") → 422.
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


# --- Missing auth → 401, guarding against a validate_session bypass.
class TestUpdateMemoryNoToken:
    def test_missing_auth_returns_401(self, db):
        client = _make_client()
        resp = client.post(
            '/api/updates/memory',
            json={"content": "should not be stored"},
        )

        assert resp.status_code == 401
