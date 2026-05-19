"""
Unit tests for the /api/wrappers blueprint.

All tests mock WrapperAuthService and cookie session validation so no real
database or authentication back-end is required.
"""

import secrets

import pytest
from unittest.mock import MagicMock, patch

from flask import Flask

from api.wrappers import wrappers_bp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_service_mock(
    create_return=("raw_token_abc", "wrp_testid"),
    list_return=None,
    get_return=None,
    update_return=True,
    revoke_return=True,
):
    """Build a fully-wired WrapperAuthService mock.

    Args:
        create_return: (raw_token, wrapper_id) tuple.
        list_return: List of wrapper dicts returned by list_wrappers().
        get_return: Single wrapper dict (or None) returned by get_wrapper().
        update_return: Bool returned by update_capabilities().
        revoke_return: Bool returned by revoke().

    Returns:
        MagicMock configured with all method return values.
    """
    svc = MagicMock()
    svc.create_token.return_value = create_return
    svc.list_wrappers.return_value = list_return or []
    svc.get_wrapper.return_value = get_return
    svc.update_capabilities.return_value = update_return
    svc.revoke.return_value = revoke_return
    return svc


def _make_wrapper_dict(wrapper_id="wrp_test", name="Test Wrapper"):
    return {
        "id": "rec_abc123",
        "wrapper_id": wrapper_id,
        "name": name,
        "capabilities": {"signals": ["ctx"]},
        "permissions": {"query": ["memory"], "update": [], "broadcast": False},
        "metadata": {"version": "1.0"},
        "last_seen_at": None,
        "created_at": "2026-03-16T10:00:00+00:00",
    }


@pytest.fixture
def cookie_client():
    """Flask test client with cookie session auth already validated (g.wrapper_id = None)."""
    app = Flask(__name__)
    app.register_blueprint(wrappers_bp)
    app.config["TESTING"] = True
    app.secret_key = secrets.token_hex(16)

    with patch("services.auth_session_service.validate_session", return_value=True):
        with app.test_client() as client:
            yield client


@pytest.fixture
def bearer_client():
    """Flask test client authenticated via a bearer token (g.wrapper_id is set)."""
    app = Flask(__name__)
    app.register_blueprint(wrappers_bp)
    app.config["TESTING"] = True
    app.secret_key = secrets.token_hex(16)

    def _make_bearer_app(wrapper_id="wrp_caller"):
        """Return a test client whose requests resolve to a given wrapper_id."""
        # Cookie session fails, bearer token succeeds
        svc_mock = MagicMock()
        svc_mock.validate_bearer.return_value = wrapper_id

        with patch("services.auth_session_service.validate_session", return_value=False), \
             patch("services.wrapper_auth_service.WrapperAuthService", return_value=svc_mock):
            with app.test_client() as c:
                yield c

    return _make_bearer_app


@pytest.fixture
def unauthed_client():
    """Flask test client with no valid auth."""
    app = Flask(__name__)
    app.register_blueprint(wrappers_bp)
    app.config["TESTING"] = True
    app.secret_key = secrets.token_hex(16)

    with patch("services.auth_session_service.validate_session", return_value=False), \
         patch("services.wrapper_auth_service.WrapperAuthService") as mock_cls:
        mock_cls.return_value.validate_bearer.return_value = None
        with app.test_client() as client:
            yield client


# ---------------------------------------------------------------------------
# POST /api/wrappers — create
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateWrapper:
    def test_create_returns_201_and_token(self, cookie_client):
        svc = _make_service_mock()
        with patch("api.wrappers._get_service", return_value=svc):
            resp = cookie_client.post(
                "/api/wrappers",
                json={"name": "IDE Wrapper"},
                content_type="application/json",
            )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["wrapper_id"] == "wrp_testid"
        assert data["token"] == "raw_token_abc"

    def test_create_missing_name_returns_400(self, cookie_client):
        svc = _make_service_mock()
        with patch("api.wrappers._get_service", return_value=svc):
            resp = cookie_client.post(
                "/api/wrappers",
                json={},
                content_type="application/json",
            )
        assert resp.status_code == 400
        assert "name" in resp.get_json()["error"]

    def test_create_via_bearer_returns_403(self):
        """Bearer-authenticated requests must NOT be able to create tokens."""
        app = Flask(__name__)
        app.register_blueprint(wrappers_bp)
        app.config["TESTING"] = True
        app.secret_key = secrets.token_hex(16)

        bearer_svc = MagicMock()
        bearer_svc.validate_bearer.return_value = "wrp_caller"

        with patch("services.auth_session_service.validate_session", return_value=False), \
             patch("services.wrapper_auth_service.WrapperAuthService", return_value=bearer_svc):
            with app.test_client() as client:
                resp = client.post(
                    "/api/wrappers",
                    json={"name": "Sneaky Wrapper"},
                    headers={"Authorization": "Bearer fake"},
                    content_type="application/json",
                )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/wrappers — list
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestListWrappers:
    def test_list_returns_200_with_wrappers(self, cookie_client):
        w = _make_wrapper_dict()
        svc = _make_service_mock(list_return=[w])
        with patch("api.wrappers._get_service", return_value=svc):
            resp = cookie_client.get("/api/wrappers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "wrappers" in data
        assert len(data["wrappers"]) == 1
        assert data["wrappers"][0]["wrapper_id"] == "wrp_test"



# ---------------------------------------------------------------------------
# GET /api/wrappers/<id> — get
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetWrapper:
    def test_get_returns_200_with_wrapper(self, cookie_client):
        w = _make_wrapper_dict(wrapper_id="wrp_abc")
        svc = _make_service_mock(get_return=w)
        with patch("api.wrappers._get_service", return_value=svc):
            resp = cookie_client.get("/api/wrappers/wrp_abc")
        assert resp.status_code == 200
        assert resp.get_json()["wrapper_id"] == "wrp_abc"

    def test_get_not_found_returns_404(self, cookie_client):
        svc = _make_service_mock(get_return=None)
        with patch("api.wrappers._get_service", return_value=svc):
            resp = cookie_client.get("/api/wrappers/wrp_missing")
        assert resp.status_code == 404




# ---------------------------------------------------------------------------
# PUT /api/wrappers/<id>/capabilities — update
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpdateCapabilities:
    def test_update_cookie_auth_returns_200(self, cookie_client):
        svc = _make_service_mock(update_return=True)
        with patch("api.wrappers._get_service", return_value=svc):
            resp = cookie_client.put(
                "/api/wrappers/wrp_abc/capabilities",
                json={"capabilities": {"signals": ["x"]}},
                content_type="application/json",
            )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_update_missing_capabilities_returns_400(self, cookie_client):
        svc = _make_service_mock()
        with patch("api.wrappers._get_service", return_value=svc):
            resp = cookie_client.put(
                "/api/wrappers/wrp_abc/capabilities",
                json={},
                content_type="application/json",
            )
        assert resp.status_code == 400

    def test_update_not_found_returns_404(self, cookie_client):
        svc = _make_service_mock(update_return=False)
        with patch("api.wrappers._get_service", return_value=svc):
            resp = cookie_client.put(
                "/api/wrappers/wrp_missing/capabilities",
                json={"capabilities": {}},
                content_type="application/json",
            )
        assert resp.status_code == 404

    def test_bearer_can_update_own_wrapper(self):
        """A bearer-authenticated request may update its own wrapper."""
        app = Flask(__name__)
        app.register_blueprint(wrappers_bp)
        app.config["TESTING"] = True
        app.secret_key = secrets.token_hex(16)

        bearer_svc = MagicMock()
        bearer_svc.validate_bearer.return_value = "wrp_own"
        bearer_svc.update_capabilities.return_value = True

        with patch("services.auth_session_service.validate_session", return_value=False), \
             patch("services.wrapper_auth_service.WrapperAuthService", return_value=bearer_svc), \
             patch("api.wrappers._get_service", return_value=bearer_svc):
            with app.test_client() as client:
                resp = client.put(
                    "/api/wrappers/wrp_own/capabilities",
                    json={"capabilities": {"signals": ["x"]}},
                    headers={"Authorization": "Bearer faketoken"},
                    content_type="application/json",
                )
        assert resp.status_code == 200

    def test_bearer_cannot_update_other_wrapper(self):
        """A bearer token may NOT update a different wrapper's capabilities."""
        app = Flask(__name__)
        app.register_blueprint(wrappers_bp)
        app.config["TESTING"] = True
        app.secret_key = secrets.token_hex(16)

        bearer_svc = MagicMock()
        bearer_svc.validate_bearer.return_value = "wrp_caller"

        with patch("services.auth_session_service.validate_session", return_value=False), \
             patch("services.wrapper_auth_service.WrapperAuthService", return_value=bearer_svc):
            with app.test_client() as client:
                resp = client.put(
                    "/api/wrappers/wrp_other/capabilities",
                    json={"capabilities": {}},
                    headers={"Authorization": "Bearer faketoken"},
                    content_type="application/json",
                )
        assert resp.status_code == 403




# ---------------------------------------------------------------------------
# DELETE /api/wrappers/<id> — revoke
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRevokeWrapper:
    def test_revoke_returns_200(self, cookie_client):
        svc = _make_service_mock(revoke_return=True)
        with patch("api.wrappers._get_service", return_value=svc):
            resp = cookie_client.delete("/api/wrappers/wrp_abc")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True


