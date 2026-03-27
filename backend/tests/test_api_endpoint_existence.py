"""
API endpoint existence tests.

Absorbs nightly scenarios 070, 074, 075 — those scenarios validated
endpoint availability against a live running server.  Route-registration tests
catch the same regressions on every commit without requiring a running instance.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestAPIEndpointExistence:
    """Verify critical API routes are registered in the Flask app.

    Absorbs nightly scenarios:
      070 — /system/status endpoint exists and returns 200 with a status field
      074 — /system/observability/tasks endpoint exists and accepts GET
      075 — /system/observability/autobiography endpoint exists and accepts GET
    """

    @pytest.fixture
    def registered_routes(self, db):
        """Build the Flask app and return a mapping of route -> allowed methods.

        The ``db`` fixture patches ``get_shared_db_service`` at the module level,
        so blueprint registration runs without infrastructure dependencies.  Only
        MemoryStore and the dashboard gateway still need stubs.
        """
        from api import create_app

        mock_store = MagicMock()

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=mock_store), \
             patch('api._init_dashboard_gateway'):
            app = create_app()
            return {rule.rule: rule.methods for rule in app.url_map.iter_rules()}

    # ------------------------------------------------------------------
    # scenario 070 — /system/status
    # ------------------------------------------------------------------

    def test_system_status_route_registered(self, registered_routes):
        """Absorbs scenario 070: /system/status must be a registered GET route."""
        assert '/system/status' in registered_routes, (
            "/system/status is not registered in Flask's URL map"
        )
        assert 'GET' in registered_routes['/system/status'], (
            "/system/status does not accept GET requests"
        )

    def test_system_status_response_structure(self, authed_client):
        """Absorbs scenario 070 (response-structure half): endpoint returns JSON with a 'status' key."""
        client, _db, mock_store = authed_client

        # MemoryStore calls still need safe defaults
        mock_store.llen.return_value = 0
        mock_store.get.return_value = None

        response = client.get('/system/status')

        assert response.status_code == 200, (
            f"/system/status returned {response.status_code}, expected 200"
        )

        data = response.get_json()
        assert data is not None, "/system/status did not return JSON"
        assert 'status' in data, (
            f"/system/status JSON missing 'status' key; got keys: {list(data.keys())}"
        )

    # ------------------------------------------------------------------
    # scenario 074 — /system/observability/tasks
    # ------------------------------------------------------------------

    def test_observability_tasks_route_registered(self, registered_routes):
        """Absorbs scenario 074: /system/observability/tasks must be a registered GET route."""
        assert '/system/observability/tasks' in registered_routes, (
            "/system/observability/tasks is not registered in Flask's URL map"
        )
        assert 'GET' in registered_routes['/system/observability/tasks'], (
            "/system/observability/tasks does not accept GET requests"
        )

    def test_observability_tasks_endpoint_reachable(self, authed_client):
        """Absorbs scenario 074 (reachability): endpoint must not 404."""
        client, _db, _mock_store = authed_client

        response = client.get('/system/observability/tasks')

        assert response.status_code != 404, (
            "/system/observability/tasks returned 404 — blueprint not registered"
        )

    # ------------------------------------------------------------------
    # scenario 075 — /system/observability/autobiography
    # ------------------------------------------------------------------

    def test_observability_autobiography_route_registered(self, registered_routes):
        """Absorbs scenario 075: /system/observability/autobiography must be a registered GET route."""
        assert '/system/observability/autobiography' in registered_routes, (
            "/system/observability/autobiography is not registered in Flask's URL map"
        )
        assert 'GET' in registered_routes['/system/observability/autobiography'], (
            "/system/observability/autobiography does not accept GET requests"
        )

    def test_observability_autobiography_endpoint_reachable(self, authed_client):
        """Absorbs scenario 075 (reachability): endpoint must not 404."""
        client, _db, _mock_store = authed_client

        response = client.get('/system/observability/autobiography')

        assert response.status_code != 404, (
            "/system/observability/autobiography returned 404 — blueprint not registered"
        )
