"""
Unit tests for the /api/updates blueprint.

All tests mock external services and session validation so no real database
or authentication back-end is required.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from flask import Flask

from api.updates import updates_bp
from services.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app():
    """Create a minimal Flask app with the updates blueprint registered."""
    app = Flask(__name__)
    app.register_blueprint(updates_bp)
    app.config["TESTING"] = True
    app.secret_key = "test-secret"
    return app


@pytest.fixture
def cookie_client():
    """Test client authenticated via cookie session (g.wrapper_id = None)."""
    app = _make_app()
    with patch("services.auth_session_service.validate_session", return_value=True):
        with app.test_client() as client:
            yield client


def _bearer_client(wrapper_id="wrp_test", permissions=None):
    """Return a context manager yielding a bearer-authenticated test client.

    Args:
        wrapper_id: The wrapper_id that validate_bearer will return.
        permissions: Permissions dict stored on the wrapper token.  If not
            provided, defaults to empty (no update permissions).
    """
    app = _make_app()
    permissions = permissions or {}

    auth_svc_mock = MagicMock()
    auth_svc_mock.validate_bearer.return_value = wrapper_id
    auth_svc_mock.check_permission.side_effect = (
        lambda wid, op, res: res in permissions.get(op, [])
    )

    return (
        app,
        patch("services.auth_session_service.validate_session", return_value=False),
        patch("services.wrapper_auth_service.WrapperAuthService", return_value=auth_svc_mock),
    )


@pytest.fixture
def unauthed_client():
    """Test client with no valid auth."""
    app = _make_app()
    with patch("services.auth_session_service.validate_session", return_value=False), \
         patch("services.wrapper_auth_service.WrapperAuthService") as mock_cls:
        mock_cls.return_value.validate_bearer.return_value = None
        with app.test_client() as client:
            yield client


# ---------------------------------------------------------------------------
# POST /api/updates/belief
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpdateBelief:
    def test_cookie_auth_stores_trait(self, cookie_client):
        svc_mock = MagicMock()
        svc_mock.store.return_value = {'id': 1, 'key': 'risk_tolerance', 'value': 'conservative'}
        with patch("services.data_graph_service.get_data_graph_service", return_value=svc_mock):
            resp = cookie_client.post(
                "/api/updates/belief",
                json={"key": "risk_tolerance", "value": "conservative", "confidence": 0.8},
                content_type="application/json",
            )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_missing_key_returns_400(self, cookie_client):
        resp = cookie_client.post(
            "/api/updates/belief",
            json={"value": "conservative"},
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "key" in resp.get_json()["error"]

    def test_missing_value_returns_400(self, cookie_client):
        resp = cookie_client.post(
            "/api/updates/belief",
            json={"key": "risk_tolerance"},
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "value" in resp.get_json()["error"]

    def test_empty_body_returns_400(self, cookie_client):
        resp = cookie_client.post(
            "/api/updates/belief",
            json={},
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_invalid_category_defaults_to_preference(self, cookie_client):
        """An unrecognised category should be silently coerced to preference."""
        svc_mock = MagicMock()
        svc_mock.store.return_value = {'id': 1, 'key': 'language', 'value': 'English'}
        with patch("services.data_graph_service.get_data_graph_service", return_value=svc_mock):
            resp = cookie_client.post(
                "/api/updates/belief",
                json={"key": "language", "value": "English", "category": "gibberish"},
                content_type="application/json",
            )
        assert resp.status_code == 200
        # DataGraphService.store is called with kind='user_specific'
        assert svc_mock.store.call_args.kwargs['kind'] == 'user_specific'

    def test_trait_validation_failure_returns_422(self, cookie_client):
        """DataGraphService.store() returning None (validation rejection) yields 422."""
        svc_mock = MagicMock()
        svc_mock.store.return_value = None
        with patch("services.data_graph_service.get_data_graph_service", return_value=svc_mock):
            resp = cookie_client.post(
                "/api/updates/belief",
                json={"key": "x", "value": "y"},
                content_type="application/json",
            )
        assert resp.status_code == 422

    def test_unauthenticated_returns_401(self, unauthed_client):
        resp = unauthed_client.post(
            "/api/updates/belief",
            json={"key": "x", "value": "y"},
            content_type="application/json",
        )
        assert resp.status_code == 401

    def test_bearer_without_permission_returns_403(self):
        """A bearer token without update/belief permission gets a 403."""
        app, patch_session, patch_auth = _bearer_client(
            permissions={}  # no update permissions
        )
        with patch_session, patch_auth:
            with app.test_client() as client:
                resp = client.post(
                    "/api/updates/belief",
                    json={"key": "risk_tolerance", "value": "aggressive"},
                    headers={"Authorization": "Bearer faketoken"},
                    content_type="application/json",
                )
        assert resp.status_code == 403

    def test_bearer_with_permission_allowed(self):
        """A bearer token with update/belief permission proceeds."""
        app, patch_session, patch_auth = _bearer_client(
            permissions={"update": ["belief"]}
        )
        svc_mock = MagicMock()
        svc_mock.store.return_value = {'id': 1, 'key': 'risk_tolerance', 'value': 'moderate'}
        with patch_session, patch_auth, \
             patch("services.data_graph_service.get_data_graph_service", return_value=svc_mock):
            with app.test_client() as client:
                resp = client.post(
                    "/api/updates/belief",
                    json={"key": "risk_tolerance", "value": "moderate", "confidence": 0.9},
                    headers={"Authorization": "Bearer faketoken"},
                    content_type="application/json",
                )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/updates/memory
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpdateMemory:
    def test_cookie_auth_stores_memory(self, cookie_client):
        with patch("services.innate_skills.memory_skill.handle_memory",
                   return_value="[MEMORIZE] Stored 1 trait(s) for topic 'work'.") as mock_mem:
            resp = cookie_client.post(
                "/api/updates/memory",
                json={"content": "User met with Sarah about Q3 roadmap", "topic": "work", "salience": 5},
                content_type="application/json",
            )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        mock_mem.assert_called_once()

    def test_missing_content_returns_400(self, cookie_client):
        resp = cookie_client.post(
            "/api/updates/memory",
            json={"topic": "work"},
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "content" in resp.get_json()["error"]

    def test_empty_content_returns_400(self, cookie_client):
        resp = cookie_client.post(
            "/api/updates/memory",
            json={"content": "   "},
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_memorize_error_returns_422(self, cookie_client):
        with patch("services.innate_skills.memory_skill.handle_memory",
                   return_value="[MEMORIZE] Error: no traits specified."):
            resp = cookie_client.post(
                "/api/updates/memory",
                json={"content": "Something"},
                content_type="application/json",
            )
        assert resp.status_code == 422

    def test_default_topic_is_general(self, cookie_client):
        with patch("services.innate_skills.memory_skill.handle_memory",
                   return_value="[MEMORIZE] Stored 1 trait(s) for topic 'general'.") as mock_mem:
            cookie_client.post(
                "/api/updates/memory",
                json={"content": "Something important happened today"},
                content_type="application/json",
            )
        call_args = mock_mem.call_args
        assert call_args.kwargs.get("topic") == "general" or call_args.args[0] == "general"

    def test_salience_clamped(self, cookie_client):
        """Salience values outside 1–10 should be clamped silently."""
        with patch("services.innate_skills.memory_skill.handle_memory",
                   return_value="[MEMORIZE] Stored 1 trait(s) for topic 'test'."):
            resp = cookie_client.post(
                "/api/updates/memory",
                json={"content": "Very important note", "salience": 999},
                content_type="application/json",
            )
        assert resp.status_code == 200

    def test_unauthenticated_returns_401(self, unauthed_client):
        resp = unauthed_client.post(
            "/api/updates/memory",
            json={"content": "Remember this"},
            content_type="application/json",
        )
        assert resp.status_code == 401

    def test_bearer_without_permission_returns_403(self):
        app, patch_session, patch_auth = _bearer_client(permissions={})
        with patch_session, patch_auth:
            with app.test_client() as client:
                resp = client.post(
                    "/api/updates/memory",
                    json={"content": "Something"},
                    headers={"Authorization": "Bearer faketoken"},
                    content_type="application/json",
                )
        assert resp.status_code == 403

    def test_bearer_with_permission_allowed(self):
        app, patch_session, patch_auth = _bearer_client(
            permissions={"update": ["memory"]}
        )
        with patch_session, patch_auth, \
             patch("services.innate_skills.memory_skill.handle_memory",
                   return_value="[MEMORIZE] Stored 1 trait(s) for topic 'general'."):
            with app.test_client() as client:
                resp = client.post(
                    "/api/updates/memory",
                    json={"content": "Meeting with Sarah went well"},
                    headers={"Authorization": "Bearer faketoken"},
                    content_type="application/json",
                )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/updates/feedback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpdateFeedback:
    def test_stores_feedback_record(self, cookie_client):
        store = MemoryStore()
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store), \
             patch("services.time_utils.utc_now") as mock_now:
            from datetime import datetime, timezone
            mock_now.return_value = datetime(2026, 3, 16, 10, 0, 0, tzinfo=timezone.utc)
            resp = cookie_client.post(
                "/api/updates/feedback",
                json={
                    "intent_id": "abc123",
                    "outcome": "success",
                    "details": "PR created successfully",
                },
                content_type="application/json",
            )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

        # Verify the feedback key and record shape via real store state.
        raw = store.get("intent_feedback:abc123")
        assert raw is not None, "Expected feedback record at 'intent_feedback:abc123'"
        record = json.loads(raw)
        assert record["intent_id"] == "abc123"
        assert record["outcome"] == "success"
        assert record["details"] == "PR created successfully"
        assert "recorded_at" in record

    def test_missing_intent_id_returns_400(self, cookie_client):
        resp = cookie_client.post(
            "/api/updates/feedback",
            json={"outcome": "success"},
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "intent_id" in resp.get_json()["error"]

    def test_missing_outcome_returns_400(self, cookie_client):
        resp = cookie_client.post(
            "/api/updates/feedback",
            json={"intent_id": "abc123"},
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "outcome" in resp.get_json()["error"]

    def test_empty_body_returns_400(self, cookie_client):
        resp = cookie_client.post(
            "/api/updates/feedback",
            json={},
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_details_optional(self, cookie_client):
        """details field is optional — omitting it should still succeed."""
        store = MemoryStore()
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store):
            resp = cookie_client.post(
                "/api/updates/feedback",
                json={"intent_id": "xyz789", "outcome": "failure"},
                content_type="application/json",
            )
        assert resp.status_code == 200
        record = json.loads(store.get("intent_feedback:xyz789"))
        assert record["details"] == ""

    def test_unauthenticated_returns_401(self, unauthed_client):
        resp = unauthed_client.post(
            "/api/updates/feedback",
            json={"intent_id": "x", "outcome": "success"},
            content_type="application/json",
        )
        assert resp.status_code == 401

    def test_bearer_without_permission_returns_403(self):
        app, patch_session, patch_auth = _bearer_client(permissions={})
        with patch_session, patch_auth:
            with app.test_client() as client:
                resp = client.post(
                    "/api/updates/feedback",
                    json={"intent_id": "abc", "outcome": "success"},
                    headers={"Authorization": "Bearer faketoken"},
                    content_type="application/json",
                )
        assert resp.status_code == 403

    def test_bearer_with_permission_allowed(self):
        app, patch_session, patch_auth = _bearer_client(
            permissions={"update": ["feedback"]}
        )
        store = MemoryStore()
        with patch_session, patch_auth, \
             patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store):
            with app.test_client() as client:
                resp = client.post(
                    "/api/updates/feedback",
                    json={"intent_id": "def456", "outcome": "success", "details": "All good"},
                    headers={"Authorization": "Bearer faketoken"},
                    content_type="application/json",
                )
        assert resp.status_code == 200

    def test_feedback_ttl_is_24h(self, cookie_client):
        """Feedback records should be stored with a 24-hour TTL."""
        store = MemoryStore()
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store):
            cookie_client.post(
                "/api/updates/feedback",
                json={"intent_id": "ttl_test", "outcome": "success"},
                content_type="application/json",
            )
        # Verify TTL was set by reading the remaining TTL from the real store.
        remaining_ttl = store.ttl("intent_feedback:ttl_test")
        assert 0 < remaining_ttl <= 86400

    def test_cookie_auth_bypasses_permission_check(self, cookie_client):
        """Cookie-authenticated users can always submit feedback."""
        store = MemoryStore()
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store):
            resp = cookie_client.post(
                "/api/updates/feedback",
                json={"intent_id": "perm_test", "outcome": "success"},
                content_type="application/json",
            )
        assert resp.status_code == 200
