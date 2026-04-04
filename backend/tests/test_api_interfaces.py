"""Tests for Interface API blueprint."""

import pytest
from unittest.mock import patch


@pytest.fixture
def app():
    """Flask test app with interfaces blueprint."""
    from unittest.mock import patch
    with patch('api._init_dashboard_gateway'):
        from api import create_app
        app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def authed_client(client):
    """Client with mocked auth session."""
    with patch('services.auth_session_service.validate_session', return_value=True):
        yield client


class TestPairingKeyEndpoint:
    @pytest.mark.unit
    def test_generate_pairing_key(self, authed_client):
        with patch('services.auth_session_service.validate_session', return_value=True), \
             patch('services.interface_registry_service.InterfaceRegistryService') as MockSvc:
            instance = MockSvc.return_value
            instance.generate_pairing_key.return_value = {
                "pairing_key": "test-key",
                "host": "localhost",
                "port": 8081,
                "expires_in_seconds": 600,
            }
            resp = authed_client.post('/api/interfaces/pairing-key')

        assert resp.status_code == 200
        data = resp.get_json()
        assert "pairing_key" in data

    @pytest.mark.unit
    def test_requires_auth(self, client):
        with patch('services.auth_session_service.validate_session', return_value=False):
            resp = client.post('/api/interfaces/pairing-key')
        assert resp.status_code == 401


class TestPairEndpoint:
    @pytest.mark.unit
    def test_successful_pair(self, client):
        with patch('services.interface_registry_service.InterfaceRegistryService') as MockSvc:
            instance = MockSvc.return_value
            instance.pair.return_value = {
                "interface_id": "abc-123",
                "signal_token": "tok_xxx",
                "status": "online",
                "capabilities": ["create_event", "query_events"],
            }
            resp = client.post('/api/interfaces/pair', json={
                "pairing_key": "test-key",
                "name": "Calendar",
                "host": "192.168.1.50",
                "port": 3100,
            })

        assert resp.status_code == 201
        data = resp.get_json()
        assert data["status"] == "online"

    @pytest.mark.unit
    def test_missing_fields(self, client):
        resp = client.post('/api/interfaces/pair', json={"pairing_key": "key"})
        assert resp.status_code == 400

    @pytest.mark.unit
    def test_invalid_key(self, client):
        with patch('services.interface_registry_service.InterfaceRegistryService') as MockSvc:
            instance = MockSvc.return_value
            instance.pair.side_effect = ValueError("Invalid pairing key")
            resp = client.post('/api/interfaces/pair', json={
                "pairing_key": "bad-key",
                "name": "Test",
                "host": "localhost",
                "port": 3100,
            })

        assert resp.status_code == 400


class TestListEndpoint:
    @pytest.mark.unit
    def test_list_interfaces(self, authed_client):
        with patch('services.auth_session_service.validate_session', return_value=True), \
             patch('services.interface_registry_service.InterfaceRegistryService') as MockSvc:
            instance = MockSvc.return_value
            instance.list_interfaces.return_value = [
                {"id": "1", "name": "Calendar", "status": "online", "tool_count": 5}
            ]
            resp = authed_client.get('/api/interfaces')

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1


class TestRemoveEndpoint:
    @pytest.mark.unit
    def test_remove_interface(self, authed_client):
        with patch('services.auth_session_service.validate_session', return_value=True), \
             patch('services.interface_registry_service.InterfaceRegistryService') as MockSvc:
            instance = MockSvc.return_value
            instance.get_interface.return_value = {"id": "1", "name": "Calendar"}
            resp = authed_client.delete('/api/interfaces/1')

        assert resp.status_code == 200

    @pytest.mark.unit
    def test_remove_nonexistent(self, authed_client):
        with patch('services.auth_session_service.validate_session', return_value=True), \
             patch('services.interface_registry_service.InterfaceRegistryService') as MockSvc:
            instance = MockSvc.return_value
            instance.get_interface.return_value = None
            resp = authed_client.delete('/api/interfaces/nonexistent')

        assert resp.status_code == 404
