"""Feature tests — the voice-settings singleton on the Endpoint base contract.

Real app + real SQLite via ``authed_client``; zero mocks. Only the disable
path is exercised on POST — enabling triggers a real background dependency
install, which has no place in a unit-tier test.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient

from services.memory_store import MemoryStore

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestVoiceSettingsAPI:
    def test_get_all_returns_singleton_listing(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.get('/api/voice-settings/all')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert len(data['result']) == 1, (
            f"voice-settings is a singleton — expected exactly 1 item, got {len(data['result'])}"
        )
        item = data['result'][0]
        assert isinstance(item['enabled'], bool)
        assert isinstance(item['install_status'], str)

    def test_post_disable_returns_disabled_state(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.post(
            '/api/voice-settings/-1',
            json={'enabled': False},
            content_type='application/json',
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert data['result'] == {'enabled': False, 'status': 'disabled', 'message': None}

        get_resp = client.get('/api/voice-settings/all')
        assert get_resp.get_json()['result'][0]['enabled'] is False

    def test_post_real_id_is_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.post(
            '/api/voice-settings/5',
            json={'enabled': False},
            content_type='application/json',
        )

        assert resp.status_code == 404, (
            f"No per-id voice-settings resources exist — expected 404, got {resp.status_code}"
        )
        assert resp.get_json()['success'] is False
