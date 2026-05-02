"""Tests for backend/api/memory.py — memory blueprint."""

import pytest

pytestmark = pytest.mark.unit


class TestMemorySearch:
    def test_missing_query_returns_400(self, authed_client):
        client, db_conn, store = authed_client
        response = client.get('/memory/search')
        assert response.status_code == 400
        data = response.get_json()
        assert data.get('error')

    def test_search_returns_results_from_data_graph(self, authed_client):
        client, db_conn, store = authed_client
        db_conn.execute(
            "INSERT INTO data_graph (kind, key, value, storage_strength, retrieval_weight) "
            "VALUES ('user_specific', 'coffee_preference', 'likes espresso', 0.5, 0.9)"
        )
        db_conn.commit()

        response = client.get('/memory/search?q=coffee')
        assert response.status_code == 200
        data = response.get_json()
        assert 'results' in data
        assert isinstance(data['results'], list)

    def test_search_empty_query_returns_400(self, authed_client):
        client, db_conn, store = authed_client
        response = client.get('/memory/search?q=')
        assert response.status_code == 400
