"""Feature tests — GET /api/memory/search on the Action base contract.

Real app + real SQLite via ``authed_client``; zero mocks. The route fans out
to both memory stores and tolerates per-store failure by design, so a valid
query against an empty database must still answer 200 with an empty listing.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient

from services.memory_store import MemoryStore

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestMemorySearchAPI:
    def test_missing_q_is_400(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.get('/api/memory/search')

        assert resp.status_code == 400, (
            f"Expected 400 for missing q, got {resp.status_code}"
        )
        assert resp.get_json()['success'] is False

    def test_blank_q_is_400(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.get('/api/memory/search?q=%20%20')

        assert resp.status_code == 400, (
            f"Expected 400 for whitespace-only q, got {resp.status_code}"
        )

    def test_valid_q_returns_listing_envelope(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.get('/api/memory/search?q=anything')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert isinstance(data['result'], list)
        assert 'pagination' in data
