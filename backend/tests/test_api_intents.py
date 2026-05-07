"""
Tests for the Intents REST API — /api/intents.

All endpoints require auth.  Tests patch ``validate_session`` to simulate
a cookie-authenticated chat-UI caller (wrapper_id = '__chat_ui__').
"""

import secrets

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

from api.intents import intents_bp

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a minimal Flask app with the intents blueprint."""
    a = Flask(__name__)
    a.secret_key = secrets.token_hex(16)
    a.config["TESTING"] = True
    a.register_blueprint(intents_bp)
    return a


@pytest.fixture
def client(app):
    """Return a Flask test client with session auth always valid."""
    with patch("services.auth_session_service.validate_session", return_value=True):
        yield app.test_client()


def _make_intent_dict(
    intent_id="abc-123",
    status="pending",
    intent_type="execute",
    target_wrapper="wrp_test",
    expires_at=None,
):
    """Build a minimal intent dict as IntentService would return it."""
    from services.time_utils import utc_now
    return {
        "intent_id": intent_id,
        "intent_type": intent_type,
        "target_wrapper": target_wrapper,
        "payload": {"action": "test"},
        "urgency": "normal",
        "confidence": 0.7,
        "expires_at": expires_at,
        "requires_ack": False,
        "created_at": utc_now().isoformat(),
        "status": status,
    }


# ---------------------------------------------------------------------------
# GET /api/intents — list pending intents
# ---------------------------------------------------------------------------

class TestListIntents:

    def test_returns_empty_list_when_no_intents(self, client):
        mock_svc = MagicMock()
        mock_svc.get_pending.return_value = []

        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.get("/api/intents")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["intents"] == []
        assert data["wrapper_id"] == "__chat_ui__"

    def test_returns_pending_intents(self, client):
        intent = _make_intent_dict()
        mock_svc = MagicMock()
        mock_svc.get_pending.return_value = [intent]

        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.get("/api/intents")

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["intents"]) == 1
        assert data["intents"][0]["intent_id"] == "abc-123"

    def test_passes_limit_param(self, client):
        mock_svc = MagicMock()
        mock_svc.get_pending.return_value = []

        with patch("api.intents.IntentService", return_value=mock_svc):
            client.get("/api/intents?limit=5")

        mock_svc.get_pending.assert_called_once_with("__chat_ui__", limit=5)

    def test_invalid_limit_returns_400(self, client):
        resp = client.get("/api/intents?limit=abc")
        assert resp.status_code == 400
        assert "limit" in resp.get_json()["error"]

    def test_limit_clamped_to_100(self, client):
        mock_svc = MagicMock()
        mock_svc.get_pending.return_value = []

        with patch("api.intents.IntentService", return_value=mock_svc):
            client.get("/api/intents?limit=9999")

        mock_svc.get_pending.assert_called_once_with("__chat_ui__", limit=100)

    def test_requires_auth(self, app):
        """Unauthenticated request should receive 401."""
        c = app.test_client()
        with patch("services.auth_session_service.validate_session", return_value=False):
            resp = c.get("/api/intents")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/intents/<id> — get single intent
# ---------------------------------------------------------------------------

class TestGetIntent:

    def test_returns_intent_when_found(self, client):
        intent = _make_intent_dict()
        mock_svc = MagicMock()
        mock_svc.get_intent.return_value = intent

        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.get("/api/intents/abc-123")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["intent"]["intent_id"] == "abc-123"

    def test_returns_404_when_not_found(self, client):
        mock_svc = MagicMock()
        mock_svc.get_intent.return_value = None

        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.get("/api/intents/does-not-exist")

        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# POST /api/intents/<id>/ack — acknowledge intent
# ---------------------------------------------------------------------------

class TestAcknowledgeIntent:

    def test_ack_success(self, client):
        mock_svc = MagicMock()
        mock_svc.acknowledge.return_value = True

        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.post("/api/intents/abc-123/ack")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["intent_id"] == "abc-123"

    def test_ack_not_found_returns_404(self, client):
        mock_svc = MagicMock()
        mock_svc.acknowledge.return_value = False

        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.post("/api/intents/missing/ack")

        assert resp.status_code == 404

    def test_ack_passes_wrapper_id(self, client):
        mock_svc = MagicMock()
        mock_svc.acknowledge.return_value = True

        with patch("api.intents.IntentService", return_value=mock_svc):
            client.post("/api/intents/abc-123/ack")

        mock_svc.acknowledge.assert_called_once_with("abc-123", "__chat_ui__")


# ---------------------------------------------------------------------------
# POST /api/intents/<id>/resolve — report execution result
# ---------------------------------------------------------------------------

class TestResolveIntent:

    def test_resolve_executed(self, client):
        mock_svc = MagicMock()
        mock_svc.resolve.return_value = True

        body = {"status": "executed", "result": {"pr_url": "https://github.com/..."}}
        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.post(
                "/api/intents/abc-123/resolve",
                json=body,
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["status"] == "executed"

    def test_resolve_failed(self, client):
        mock_svc = MagicMock()
        mock_svc.resolve.return_value = True

        body = {"status": "failed", "error": "permission denied"}
        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.post("/api/intents/abc-123/resolve", json=body)

        assert resp.status_code == 200
        assert resp.get_json()["status"] == "failed"

    def test_resolve_skipped(self, client):
        mock_svc = MagicMock()
        mock_svc.resolve.return_value = True

        body = {"status": "skipped", "reason": "user declined"}
        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.post("/api/intents/abc-123/resolve", json=body)

        assert resp.status_code == 200
        assert resp.get_json()["status"] == "skipped"

    def test_resolve_not_found_returns_404(self, client):
        mock_svc = MagicMock()
        mock_svc.resolve.return_value = False

        with patch("api.intents.IntentService", return_value=mock_svc):
            resp = client.post(
                "/api/intents/missing/resolve",
                json={"status": "executed"},
            )

        assert resp.status_code == 404

    def test_resolve_invalid_status_returns_400(self, client):
        resp = client.post(
            "/api/intents/abc-123/resolve",
            json={"status": "exploded"},
        )
        assert resp.status_code == 400
        assert "status" in resp.get_json()["error"]

    def test_resolve_missing_body_returns_400(self, client):
        resp = client.post(
            "/api/intents/abc-123/resolve",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_resolve_passes_full_body_to_service(self, client):
        mock_svc = MagicMock()
        mock_svc.resolve.return_value = True

        body = {"status": "executed", "result": {"output": "done"}}
        with patch("api.intents.IntentService", return_value=mock_svc):
            client.post("/api/intents/abc-123/resolve", json=body)

        mock_svc.resolve.assert_called_once_with("abc-123", body)
