"""Tests for moments API — data_graph-backed pin/list/search/forget.

Coverage:
  POST /moments              — happy path, 404 missing, 400 user role, idempotency
  GET  /moments              — list active, exclude deleted
  POST /moments/<id>/forget  — soft-delete
  GET  /moments/search       — semantic search
"""

import pytest
from flask import Flask

from api.moments import moments_bp
from services.data_graph_service import get_data_graph_service
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


def _insert_transcript(db, role, content='Test content', channel='user'):
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, role, content),
    )
    row_id = cursor.lastrowid
    cursor.close()
    return row_id


def _raw_data_graph(db, key):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM data_graph WHERE key = ?", (key,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


@pytest.fixture
def dgs(db):
    """DataGraphService backed by the real schema, embeddings disabled."""
    svc = get_data_graph_service()
    svc._generate_embedding = lambda text: None
    return svc


@pytest.fixture
def client(db, dgs):
    """Flask test client with moments_bp, auth bypassed."""
    app = Flask(__name__)
    app.register_blueprint(moments_bp)
    app.config['TESTING'] = True

    import services.auth_session_service
    original = services.auth_session_service.validate_session
    services.auth_session_service.validate_session = lambda token: True

    with app.test_client() as c:
        yield c

    services.auth_session_service.validate_session = original


class TestPostMoments:

    def test_happy_path_stores_moment_in_data_graph(self, client, db):
        tid = _insert_transcript(db, role='assistant', content='Great insight here')

        resp = client.post(
            '/moments',
            json={'transcript_id': tid},
            content_type='application/json',
        )

        assert resp.status_code == 201
        body = resp.get_json()
        assert 'item' in body
        assert body['item']['key'] == f'moment_{tid}'
        assert body['item']['value'] == 'Great insight here'

        row = _raw_data_graph(db, f'moment_{tid}')
        assert row is not None
        assert row['kind'] == 'moment'
        assert row['source'] == 'pin'

    def test_404_when_transcript_id_not_found(self, client):
        resp = client.post(
            '/moments',
            json={'transcript_id': 99999},
            content_type='application/json',
        )
        assert resp.status_code == 404
        assert 'error' in resp.get_json()

    def test_400_when_transcript_row_is_user_role(self, client, db):
        tid = _insert_transcript(db, role='user', content='User message')

        resp = client.post(
            '/moments',
            json={'transcript_id': tid},
            content_type='application/json',
        )
        assert resp.status_code == 400

    def test_idempotent_pin_same_id_twice_no_error(self, client, db):
        tid = _insert_transcript(db, role='assistant', content='Reusable insight')

        resp1 = client.post('/moments', json={'transcript_id': tid}, content_type='application/json')
        resp2 = client.post('/moments', json={'transcript_id': tid}, content_type='application/json')

        assert resp1.status_code == 201
        assert resp2.status_code == 200

        cursor = db.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM data_graph WHERE kind='moment' AND key=? AND deleted_at IS NULL",
            (f'moment_{tid}',),
        )
        count = cursor.fetchone()[0]
        cursor.close()
        assert count == 1

    def test_400_when_content_type_is_not_json(self, client):
        resp = client.post('/moments', data='transcript_id=1')
        assert resp.status_code == 400


class TestGetMoments:

    def test_returns_all_active_moment_rows(self, client, db, dgs):
        t1 = _insert_transcript(db, role='assistant', content='First insight')
        t2 = _insert_transcript(db, role='assistant', content='Second insight')

        for tid in (t1, t2):
            dgs.store(kind='moment', key=f'moment_{tid}', value=f'Content {tid}', source='pin')

        resp = client.get('/moments')
        assert resp.status_code == 200
        body = resp.get_json()
        assert 'items' in body
        assert len(body['items']) == 2

    def test_excludes_deleted_rows(self, client, db, dgs):
        t1 = _insert_transcript(db, role='assistant', content='Kept')
        t2 = _insert_transcript(db, role='assistant', content='Deleted')

        dgs.store(kind='moment', key=f'moment_{t1}', value='Kept', source='pin')
        dgs.store(kind='moment', key=f'moment_{t2}', value='Deleted', source='pin')

        db.execute(
            "UPDATE data_graph SET deleted_at = ? WHERE key = ?",
            (utc_now().isoformat(), f'moment_{t2}'),
        )

        resp = client.get('/moments')
        assert resp.status_code == 200
        items = resp.get_json()['items']
        assert len(items) == 1
        assert items[0]['key'] == f'moment_{t1}'

    def test_returns_empty_list_when_no_moments(self, client):
        resp = client.get('/moments')
        assert resp.status_code == 200
        assert resp.get_json()['items'] == []


class TestForgetMoment:

    def _seed_moment(self, db, dgs, content='Insightful response'):
        tid = _insert_transcript(db, role='assistant', content=content)
        dgs.store(kind='moment', key=f'moment_{tid}', value=content, source='pin')
        return tid

    def test_soft_deletes_moment_row(self, client, db, dgs):
        tid = self._seed_moment(db, dgs)

        resp = client.post(f'/moments/{tid}/forget')
        assert resp.status_code == 200
        assert resp.get_json()['ok'] is True

        row = _raw_data_graph(db, f'moment_{tid}')
        assert row is not None
        assert row['deleted_at'] is not None

    def test_forget_nonexistent_moment_returns_404(self, client):
        resp = client.post('/moments/99999/forget')
        assert resp.status_code == 404
        assert 'error' in resp.get_json()

    def test_forgotten_moment_absent_from_list(self, client, db, dgs):
        tid = self._seed_moment(db, dgs)
        client.post(f'/moments/{tid}/forget')

        resp = client.get('/moments')
        items = resp.get_json()['items']
        assert all(item['key'] != f'moment_{tid}' for item in items)

    def test_second_forget_of_same_moment_returns_404(self, client, db, dgs):
        tid = self._seed_moment(db, dgs)
        client.post(f'/moments/{tid}/forget')
        resp2 = client.post(f'/moments/{tid}/forget')
        assert resp2.status_code == 404


class TestSearchMoments:

    def _seed_moment(self, db, dgs, content, transcript_id=None):
        if transcript_id is None:
            transcript_id = _insert_transcript(db, role='assistant', content=content)
        moment_key = f'moment_{transcript_id}'
        dgs.store(kind='moment', key=moment_key, value=content, source='pin')
        return transcript_id, moment_key

    def test_missing_query_parameter_returns_400(self, client):
        resp = client.get('/moments/search')
        assert resp.status_code == 400

    def test_empty_query_returns_400(self, client):
        resp = client.get('/moments/search?q=')
        assert resp.status_code == 400

    def test_search_with_no_moments_returns_empty(self, client):
        resp = client.get('/moments/search?q=cooking')
        assert resp.status_code == 200
        assert resp.get_json()['items'] == []

    def test_search_returns_200_with_moments_present(self, client, db, dgs):
        self._seed_moment(db, dgs, 'Memorable response about cooking')
        resp = client.get('/moments/search?q=cooking')
        assert resp.status_code == 200
        assert 'items' in resp.get_json()
