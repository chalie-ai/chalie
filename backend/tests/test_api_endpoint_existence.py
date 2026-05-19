"""
API endpoint existence tests.

Absorbs nightly scenario 070 — that scenario validated endpoint
availability against a live running server.  Route-registration tests
catch the same regressions on every commit without requiring a running instance.
"""

import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestAPIEndpointExistence:
    """Verify critical API routes are registered in the Flask app.

    Absorbs nightly scenarios:
      070 — /system/status endpoint exists and returns 200 with a status field
    """

    @pytest.fixture
    def registered_routes(self, db):
        """Build the Flask app and return a mapping of route -> allowed methods.

        The ``db`` fixture patches ``get_shared_db_service`` at the module level,
        so blueprint registration runs without infrastructure dependencies.  A
        real ``MemoryStore`` is used so that any store access during app
        initialisation behaves correctly without artificial stubs.

        Returns:
            dict[str, frozenset[str]]: mapping of URL rule string to the set of
            HTTP methods registered for that route.
        """
        from api import create_app
        from services.memory_store import MemoryStore

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('api._init_dashboard_gateway'), \
             patch('api._get_or_generate_session_secret', return_value='test-secret'):
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
        client, _db, _store = authed_client

        response = client.get('/system/status')

        assert response.status_code == 200, (
            f"/system/status returned {response.status_code}, expected 200"
        )

        data = response.get_json()
        assert data is not None, "/system/status did not return JSON"
        assert 'status' in data, (
            f"/system/status JSON missing 'status' key; got keys: {list(data.keys())}"
        )

