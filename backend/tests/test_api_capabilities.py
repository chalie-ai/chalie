"""
Unit tests for backend/api/capabilities.py.

All tests use a lightweight Flask test client with the capabilities blueprint
registered in isolation.  No real CalDAV connections, database I/O, or
external services are exercised — every dependency is replaced with a
:class:`unittest.mock.MagicMock`.

Authentication is bypassed globally for this test class via an ``autouse``
fixture.  The single test that verifies auth enforcement temporarily
re-patches ``validate_session`` to ``False`` inside the test body, which
overrides the autouse fixture's active patch (inner context managers win).
"""

import pytest
from unittest.mock import patch, MagicMock, call

from flask import Flask
from api.capabilities import capabilities_bp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_cap(
    cap_id: str = "caldav",
    connected: bool = True,
    configure_raises: Exception | None = None,
    connect_returns: bool = True,
) -> MagicMock:
    """Build a mock :class:`~capabilities.base.AbstractCapability` for injection.

    Args:
        cap_id: The capability identifier returned by ``get_id()`` and
            ``get_manifest()["id"]``.
        connected: Value returned by ``is_connected()``.
        configure_raises: If supplied, ``configure()`` will raise this
            exception instead of returning ``None``.
        connect_returns: Value returned by ``connect()``.

    Returns:
        MagicMock: A pre-configured mock that satisfies the
        :class:`~capabilities.base.AbstractCapability` interface.
    """
    cap = MagicMock()
    cap.get_id.return_value = cap_id
    cap.get_manifest.return_value = {
        "id": cap_id,
        "name": "CalDAV",
        "version": "1.0.0",
        "entry_class": "CaldavCapability",
        "providers": ["google", "apple", "fastmail"],
    }
    cap.is_connected.return_value = connected
    if configure_raises is not None:
        cap.configure.side_effect = configure_raises
    else:
        cap.configure.return_value = None
    cap.connect.return_value = connect_returns
    cap.disconnect.return_value = None
    cap.get_tools.return_value = [
        {
            "name": "caldav_list_events",
            "handler": MagicMock(),
            "description": "List calendar events in a date range",
        },
        {
            "name": "caldav_get_event",
            "handler": MagicMock(),
            "description": "Retrieve a single calendar event by UID",
        },
    ]
    return cap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCapabilitiesAPI:
    """Tests for every route in the ``/api/capabilities`` blueprint."""

    @pytest.fixture
    def client(self):
        """Flask test client with only the capabilities blueprint registered.

        Yields:
            FlaskClient: An isolated test client; no other blueprints are
            mounted, which prevents unintended cross-blueprint side-effects.
        """
        app = Flask(__name__)
        app.register_blueprint(capabilities_bp)
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass cookie-session authentication for every test in this class.

        Patches ``validate_session`` to return ``True`` so that all requests
        are treated as authenticated unless the test overrides this patch
        internally (see :meth:`test_all_endpoints_require_auth`).

        Yields:
            None
        """
        with patch("services.auth_session_service.validate_session", return_value=True):
            yield

    # ------------------------------------------------------------------
    # 1 — GET /api/capabilities → 200 with JSON list
    # ------------------------------------------------------------------

    def test_list_capabilities_returns_200(self, client):
        """GET /api/capabilities returns HTTP 200 with a ``capabilities`` array.

        Verifies that:
        - The status code is 200.
        - The response body contains a ``capabilities`` key holding a list.
        - Each element describes a discovered capability (id, connected, etc.).

        Args:
            client: Pytest fixture — isolated Flask test client.
        """
        mock_cap = _make_mock_cap()
        with patch("api.capabilities._load_caps", return_value={"caldav": mock_cap}), \
             patch("api.capabilities._get_last_sync_at", return_value=None):
            resp = client.get("/api/capabilities")

        assert resp.status_code == 200
        body = resp.get_json()
        assert "capabilities" in body
        assert isinstance(body["capabilities"], list)
        assert len(body["capabilities"]) == 1
        entry = body["capabilities"][0]
        assert entry["id"] == "caldav"
        assert entry["name"] == "CalDAV"
        assert "connected" in entry
        assert "last_sync_at" in entry

    # ------------------------------------------------------------------
    # 2 — POST /api/capabilities/nonexistent/setup → 404
    # ------------------------------------------------------------------

    def test_setup_unknown_capability_returns_404(self, client):
        """POST setup for an unknown cap_id returns HTTP 404 with an error body.

        Args:
            client: Pytest fixture — isolated Flask test client.
        """
        with patch("api.capabilities._load_caps", return_value={}):
            resp = client.post(
                "/api/capabilities/nonexistent/setup",
                json={"provider": "google", "username": "u", "password": "p"},
            )

        assert resp.status_code == 404
        body = resp.get_json()
        assert "error" in body
        assert "nonexistent" in body["error"]

    # ------------------------------------------------------------------
    # 3 — POST setup with valid credentials → {status: connected}
    # ------------------------------------------------------------------

    def test_setup_valid_capability_returns_connected(self, client):
        """POST setup when configure() and connect() both succeed returns connected.

        Also verifies that:
        - ``configure()`` is called exactly once with the request body.
        - ``connect()`` is called exactly once.
        - Dynamic tools are registered via :func:`register_tool`.

        Args:
            client: Pytest fixture — isolated Flask test client.
        """
        mock_cap = _make_mock_cap(connect_returns=True)
        credentials = {
            "provider": "google",
            "username": "user@gmail.com",
            "password": "app-password-1234",
        }

        with patch("api.capabilities._load_caps", return_value={"caldav": mock_cap}), \
             patch("services.tool_library_service.register_tool") as mock_register:
            resp = client.post(
                "/api/capabilities/caldav/setup",
                json=credentials,
            )

        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("status") == "connected"
        mock_cap.configure.assert_called_once_with(credentials)
        mock_cap.connect.assert_called_once()
        # Each tool definition should trigger a register_tool call
        assert mock_register.call_count == len(mock_cap.get_tools.return_value)

    # ------------------------------------------------------------------
    # 4 — POST setup when configure() raises ValueError → 400
    # ------------------------------------------------------------------

    def test_setup_invalid_credentials_returns_400(self, client):
        """POST setup when configure() raises ValueError returns 400 with error key.

        Args:
            client: Pytest fixture — isolated Flask test client.
        """
        mock_cap = _make_mock_cap(configure_raises=ValueError("Invalid app password — check Google account settings"))

        with patch("api.capabilities._load_caps", return_value={"caldav": mock_cap}):
            resp = client.post(
                "/api/capabilities/caldav/setup",
                json={"provider": "google", "username": "user@gmail.com", "password": "wrong"},
            )

        assert resp.status_code == 400
        body = resp.get_json()
        assert "error" in body
        assert "Invalid app password" in body["error"]
        # connect() must not be called after a configure() failure
        mock_cap.connect.assert_not_called()

    # ------------------------------------------------------------------
    # 5 — POST /api/capabilities/nonexistent/disconnect → 404
    # ------------------------------------------------------------------

    def test_disconnect_unknown_capability_returns_404(self, client):
        """POST disconnect for an unknown cap_id returns HTTP 404.

        Args:
            client: Pytest fixture — isolated Flask test client.
        """
        with patch("api.capabilities._load_caps", return_value={}):
            resp = client.post("/api/capabilities/nonexistent/disconnect")

        assert resp.status_code == 404
        body = resp.get_json()
        assert "error" in body

    # ------------------------------------------------------------------
    # 6 — POST disconnect for a connected capability → {status: disconnected}
    # ------------------------------------------------------------------

    def test_disconnect_connected_capability_returns_disconnected(self, client):
        """POST disconnect for a known connected cap returns {status: 'disconnected'}.

        Also verifies that:
        - Each tool name is unregistered via :func:`unregister_tool`.
        - ``disconnect()`` is called exactly once.

        Args:
            client: Pytest fixture — isolated Flask test client.
        """
        mock_cap = _make_mock_cap(connected=True)

        with patch("api.capabilities._load_caps", return_value={"caldav": mock_cap}), \
             patch("services.tool_library_service.unregister_tool") as mock_unregister:
            resp = client.post("/api/capabilities/caldav/disconnect")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("status") == "disconnected"
        mock_cap.disconnect.assert_called_once()
        expected_tool_names = {t["name"] for t in mock_cap.get_tools.return_value}
        unregistered_names = {c.args[0] for c in mock_unregister.call_args_list}
        assert unregistered_names == expected_tool_names

    # ------------------------------------------------------------------
    # 7 — GET /api/capabilities/<id>/status → 200 with connected + last_sync_at
    # ------------------------------------------------------------------

    def test_status_returns_connection_state(self, client):
        """GET status for a known capability returns 200 with required keys.

        The response body must contain ``id``, ``connected``, ``last_sync_at``,
        and ``error_count``.

        Args:
            client: Pytest fixture — isolated Flask test client.
        """
        mock_cap = _make_mock_cap(connected=True)
        last_sync = "2026-03-27T12:00:00Z"

        mock_db = MagicMock()
        mock_tool_config_svc = MagicMock()
        mock_tool_config_svc.get_tool_config.return_value = {
            "caldav:last_sync_at": last_sync,
            "caldav:error_count": "0",
        }

        with patch("api.capabilities._load_caps", return_value={"caldav": mock_cap}), \
             patch("api.capabilities._get_last_sync_at", return_value=last_sync), \
             patch("services.database_service.get_shared_db_service", return_value=mock_db), \
             patch("services.tool_config_service.ToolConfigService", return_value=mock_tool_config_svc):
            resp = client.get("/api/capabilities/caldav/status")

        assert resp.status_code == 200
        body = resp.get_json()
        assert "connected" in body
        assert "last_sync_at" in body
        assert body["id"] == "caldav"
        assert body["connected"] is True
        assert body["last_sync_at"] == last_sync

    # ------------------------------------------------------------------
    # 8 — All endpoints require authentication
    # ------------------------------------------------------------------

    def test_all_endpoints_require_auth(self, client):
        """Without valid auth credentials, all 4 endpoint patterns return 401.

        This test temporarily overrides the class-level ``bypass_auth``
        autouse fixture by re-patching ``validate_session`` to ``False`` inside
        the test body.  The inner patch takes precedence over the outer one.

        The bearer-token fallback path is also neutralised by mocking
        :class:`~services.wrapper_auth_service.WrapperAuthService` so that
        ``validate_bearer()`` returns ``None``.

        Args:
            client: Pytest fixture — isolated Flask test client.
        """
        with patch("services.auth_session_service.validate_session", return_value=False), \
             patch("services.wrapper_auth_service.WrapperAuthService") as mock_was:
            # Bearer token check returns None → no wrapper authenticated
            mock_was.return_value.validate_bearer.return_value = None

            # GET /api/capabilities
            resp = client.get("/api/capabilities")
            assert resp.status_code == 401, (
                f"GET /api/capabilities expected 401, got {resp.status_code}"
            )

            # POST /api/capabilities/<id>/setup
            resp = client.post("/api/capabilities/caldav/setup", json={})
            assert resp.status_code == 401, (
                f"POST /api/capabilities/caldav/setup expected 401, got {resp.status_code}"
            )

            # POST /api/capabilities/<id>/disconnect
            resp = client.post("/api/capabilities/caldav/disconnect")
            assert resp.status_code == 401, (
                f"POST /api/capabilities/caldav/disconnect expected 401, got {resp.status_code}"
            )

            # GET /api/capabilities/<id>/status
            resp = client.get("/api/capabilities/caldav/status")
            assert resp.status_code == 401, (
                f"GET /api/capabilities/caldav/status expected 401, got {resp.status_code}"
            )
