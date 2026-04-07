"""
Tests for backend/api/conversation.py — conversation blueprint.

Covers /conversation/recent endpoint.

Note: The /chat endpoint was replaced by the WebSocket handler in
api/websocket.py (Phase 4). WebSocket tests live separately.
"""

import pytest
from unittest.mock import patch
from flask import Flask

from api.conversation import conversation_bp


@pytest.mark.unit
class TestConversationAPI:
    """Test conversation API endpoints — uses real SQLite db + MemoryStore."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with conversation blueprint."""
        app = Flask(__name__)
        app.register_blueprint(conversation_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for all tests."""
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    # ------------------------------------------------------------------
    # GET /conversation/recent
    # ------------------------------------------------------------------

    def test_recent_returns_exchanges(self, client, db, store):
        """GET /conversation/recent returns exchanges from real transcript rows."""
        channel = 'web:default:1'
        store.set('active_channel:default', channel)

        # Seed two transcript entries (one user, one assistant) in the real DB
        db.execute(
            "INSERT INTO transcript (channel, role, content, internal, created_at) VALUES (?, 'user', ?, 0, '2026-01-01T00:00:00+00:00')",
            (channel, 'Hello there'),
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, internal, created_at) VALUES (?, 'assistant', ?, 0, '2026-01-01T00:00:01+00:00')",
            (channel, 'Hi, how can I help?'),
        )
        db.commit()

        response = client.get('/conversation/recent')

        assert response.status_code == 200
        data = response.get_json()
        assert "exchanges" in data
        assert data["thread_id"] == channel
        assert data["total"] == 2
        assert len(data["exchanges"]) == 1  # one user+assistant pair
        assert data["exchanges"][0]["prompt"] == "Hello there"

    def test_recent_no_entries_returns_empty(self, client, db, store):
        """GET /conversation/recent with no transcript entries returns empty."""
        channel = 'web:default:1'
        store.set('active_channel:default', channel)
        # No rows seeded — real DB is empty for this channel

        response = client.get('/conversation/recent')

        assert response.status_code == 200
        data = response.get_json()
        assert data["exchanges"] == []
        assert data["total"] == 0

    def test_recent_uses_default_channel_when_store_empty(self, client, db, store):
        """When active_channel key is absent, falls back to 'web:default:1'."""
        # No store key set — store.get returns None, endpoint should use 'web:default:1'
        response = client.get('/conversation/recent')
        assert response.status_code == 200
        data = response.get_json()
        assert data["thread_id"] == "web:default:1"

    def test_recent_has_more_when_offset_within_total(self, client, db, store):
        """has_more is True when there are more records beyond the current page."""
        channel = 'web:default:1'
        store.set('active_channel:default', channel)

        # Seed 3 user+assistant pairs (6 rows total)
        for i in range(3):
            db.execute(
                "INSERT INTO transcript (channel, role, content, internal, created_at) VALUES (?, 'user', ?, 0, '2026-01-01T00:00:00+00:00')",
                (channel, f'Question {i}'),
            )
            db.execute(
                "INSERT INTO transcript (channel, role, content, internal, created_at) VALUES (?, 'assistant', ?, 0, '2026-01-01T00:00:01+00:00')",
                (channel, f'Answer {i}'),
            )
        db.commit()

        # Request only 1 entry — has_more should be True
        response = client.get('/conversation/recent?limit=2')

        assert response.status_code == 200
        data = response.get_json()
        assert data["total"] == 6
        assert data["has_more"] is True

    def test_recent_internal_rows_excluded(self, client, db, store):
        """Internal transcript rows (internal=1) are not included in results."""
        channel = 'web:default:1'
        store.set('active_channel:default', channel)

        # Seed one internal row and one normal user row
        db.execute(
            "INSERT INTO transcript (channel, role, content, internal, created_at) VALUES (?, 'user', 'internal message', 1, '2026-01-01T00:00:00+00:00')",
            (channel,),
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, internal, created_at) VALUES (?, 'user', 'real message', 0, '2026-01-01T00:00:02+00:00')",
            (channel,),
        )
        db.commit()

        response = client.get('/conversation/recent')

        assert response.status_code == 200
        data = response.get_json()
        # Only the non-internal user row counts; total = 1, no assistant pair so 0 exchanges
        assert data["total"] == 1
