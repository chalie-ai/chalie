"""Feature tests — the capabilities endpoint group on the Endpoint/Action contract.

All tests are @pytest.mark.unit — real app via the ``authed_client`` fixture,
real capability registry (in-process integration plugins). No test ever runs a
real setup: connecting a capability requires live credentials and fires an
initial sync on a daemon thread. Coverage here is the read view plus the
unknown-id contract on every route.

Contract notes vs the legacy namespace this suite replaces:
- GET /api/capabilities → GET /api/capabilities/all, bare list → listing
  envelope with pagination.
- Verb routes reshaped: /<id>/status → /status/<id>, /<id>/setup →
  /setup/<id>, /<id>/disconnect → /disconnect/<id>.
- Unknown ids answer 404 with the uniform failure envelope on all four
  id-addressed routes.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient

from services.memory_store import MemoryStore

pytestmark = pytest.mark.unit

_MISSING = 'no-such-capability'


@pytest.mark.unit
class TestCapabilitiesList:
    def test_get_all_returns_listing_envelope(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.get('/api/capabilities/all')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert isinstance(data['result'], list)
        assert 'pagination' in data
        for item in data['result']:
            for key in ('id', 'name', 'version', 'connected', 'last_sync_at', 'providers'):
                assert key in item, f"Listing item missing '{key}': {item!r}"
            assert 'password' not in item, "Listing must never carry credentials"


@pytest.mark.unit
class TestCapabilitiesUnknownId:
    def test_detail_unknown_id_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.get(f'/api/capabilities/{_MISSING}')
        assert resp.status_code == 404
        data = resp.get_json()
        assert data['success'] is False

    def test_status_unknown_id_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.get(f'/api/capabilities/status/{_MISSING}')
        assert resp.status_code == 404
        assert resp.get_json()['success'] is False

    def test_setup_unknown_id_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.post(
            f'/api/capabilities/setup/{_MISSING}',
            json={},
            content_type='application/json',
        )
        assert resp.status_code == 404
        assert resp.get_json()['success'] is False

    def test_disconnect_unknown_id_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.post(
            f'/api/capabilities/disconnect/{_MISSING}',
            json={},
            content_type='application/json',
        )
        assert resp.status_code == 404
        assert resp.get_json()['success'] is False
