"""Tests for api/system.py — /health, /metrics, /system/status, /system/observability/* endpoints."""

import pytest
from unittest.mock import patch, MagicMock

from flask import Flask
from api.system import system_bp
from services.memory_store import MemoryStore


@pytest.mark.unit
class TestSystemAPI:

    @pytest.fixture
    def client(self):
        app = Flask(__name__)
        app.register_blueprint(system_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    # GET /health

    def test_get_health_returns_ok_and_version(self, client):
        """GET /health returns status 'ok' and the current APP_VERSION."""
        with patch('consumer.APP_VERSION', '2.5.0'):
            resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert data['version'] == '2.5.0'

    # GET /system/status

    def test_system_status_returns_expected_keys(self, client, db):
        """GET /system/status returns status, memory, storage top-level keys."""
        store = MemoryStore()
        # Seed 2 keys for each memory namespace so counts == 2.
        for ns in ('working_memory', 'gist_index', 'fact_index'):
            store.set(f'{ns}:a', 'x')
            store.set(f'{ns}:b', 'x')
        # Seed DMN delivery ZSET with a recent entry.
        store.zadd('dmn:deliveries', {'test-delivery': 1711500000.0})

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            resp = client.get('/system/status')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert 'memory' in data
        assert 'storage' in data
        # Memory keys should reflect store.keys() calls (3 calls: working_memory, gist, fact)
        assert data['memory']['working_memory_keys'] == 2
        assert data['memory']['gist_keys'] == 2
        assert data['memory']['fact_keys'] == 2

    def test_system_status_degraded_when_store_fails(self, client, db):
        """GET /system/status reports 'degraded' when MemoryStore ping raises."""
        # Category C (error-path): keep MagicMock to simulate a broken store.
        broken_store = MagicMock()
        broken_store.ping.side_effect = ConnectionError('store unreachable')
        broken_store.llen.return_value = 0
        broken_store.get.return_value = None

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=broken_store):
            resp = client.get('/system/status')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'degraded'
        assert 'memory_store_error' in data

    # GET /system/observability/records

    def test_records_episodes_source_returns_gist_and_location(self, client, db):
        """Episodes source returns value=gist + location, ordered by last_accessed DESC (NULLs last)."""
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel, created_at, last_accessed_at, location_name) "
            "VALUES ('ep-a', 'first gist', 5, 'user', '2026-01-01T00:00:00', '2026-01-03T00:00:00', 'Valletta, Malta')"
        )
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel, created_at, last_accessed_at, location_name) "
            "VALUES ('ep-b', 'second gist', 5, 'user', '2026-01-02T00:00:00', '2026-01-04T00:00:00', NULL)"
        )
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel, created_at, last_accessed_at) "
            "VALUES ('ep-c', 'null access', 5, 'user', '2026-01-03T00:00:00', NULL)"
        )
        db.commit()

        resp = client.get('/system/observability/records?source=episodes')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['source'] == 'episodes'
        assert data['returned'] == 3
        values = [r['value'] for r in data['rows']]
        assert values[-1] == 'null access'
        assert values.index('second gist') < values.index('first gist')
        row = next(r for r in data['rows'] if r['value'] == 'first gist')
        assert row['location'] == 'Valletta, Malta'
        assert 'key' not in row
        row_null = next(r for r in data['rows'] if r['value'] == 'second gist')
        assert row_null['location'] == ''

    def test_records_search_filters_by_like(self, client, db):
        """Search param filters gist (episodes) and key/value (data_graph)."""
        now_iso = '2026-01-01T00:00:00+00:00'
        db.execute(
            "INSERT INTO data_graph (kind, key, value, first_seen_at, last_confirmed_at) "
            "VALUES ('user_specific', 'favorite_color', 'blue', ?, ?)",
            (now_iso, now_iso),
        )
        db.execute(
            "INSERT INTO data_graph (kind, key, value, first_seen_at, last_confirmed_at) "
            "VALUES ('user_specific', 'pet_name', 'Rex', ?, ?)",
            (now_iso, now_iso),
        )
        db.commit()

        resp = client.get('/system/observability/records?source=user&q=blue')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['returned'] == 1
        assert data['rows'][0]['key'] == 'favorite_color'

    def test_records_pagination_offset_works(self, client, db):
        """First page returns 250 rows with has_more=true; offset=250 returns remainder."""
        now_iso = '2026-01-01T00:00:00+00:00'
        for i in range(260):
            db.execute(
                "INSERT INTO episodes (id, gist, salience, channel, created_at) "
                "VALUES (?, ?, 5, 'user', ?)",
                (f'ep-{i:04d}', f'gist {i}', now_iso),
            )
        db.commit()

        resp1 = client.get('/system/observability/records?source=episodes')
        assert resp1.status_code == 200
        data1 = resp1.get_json()
        assert data1['returned'] == 250
        assert data1['has_more'] is True

        resp2 = client.get('/system/observability/records?source=episodes&offset=250')
        assert resp2.status_code == 200
        data2 = resp2.get_json()
        assert data2['returned'] == 10
        assert data2['has_more'] is False

    def test_records_invalid_source_400(self, client, db):
        """Unknown source returns 400 with error payload."""
        resp = client.get('/system/observability/records?source=bogus')
        assert resp.status_code == 400
        assert resp.get_json()['error'] == 'invalid source'

    # GET /system/observability/tools

    def test_observability_tools_returns_stats(self, client, db):
        """GET /system/observability/tools returns per-tool usage counts from tool_calls.

        Verifies: aggregation (COUNT, MAX), ORDER BY last_used_at DESC, and
        exclusion of pseudo-tool audit rows (compaction/thinking).
        """
        # Seed a transcript row so FK constraint is satisfied
        db.execute(
            "INSERT INTO transcript (role, content, channel) VALUES ('user', 'hi', 'test')"
        )
        transcript_id = db.execute(
            "SELECT id FROM transcript ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        # weather: 2 calls
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'weather', 'sunny', '2025-01-01T00:00:00'),
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'weather', 'cloudy', '2025-01-03T00:00:00'),
        )
        # code_exec: 1 call
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'code_exec', 'ok', '2025-01-02T00:00:00'),
        )
        # Pseudo-tool rows must NOT surface in the panel
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'compaction', '{}', '2025-01-04T00:00:00'),
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'thinking', '{}', '2025-01-05T00:00:00'),
        )
        db.commit()

        resp = client.get('/system/observability/tools')

        assert resp.status_code == 200
        data = resp.get_json()
        assert 'generated_at' in data
        tools = data['tools']

        # Pseudo-tool rows excluded
        names = [t['tool_name'] for t in tools]
        assert 'compaction' not in names
        assert 'thinking' not in names

        # Two real tools surfaced
        assert {'weather', 'code_exec'} == set(names)

        weather = next(t for t in tools if t['tool_name'] == 'weather')
        code_exec = next(t for t in tools if t['tool_name'] == 'code_exec')

        # COUNT aggregation
        assert weather['count'] == 2
        assert code_exec['count'] == 1

        # MAX(last_used_at) — weather's latest is 2025-01-03
        assert weather['last_used_at'] == '2025-01-03T00:00:00'
        assert code_exec['last_used_at'] == '2025-01-02T00:00:00'

        # ORDER BY last_used_at DESC — weather (Jan 3) before code_exec (Jan 2)
        assert names.index('weather') < names.index('code_exec')

        # Response shape: only tool_name, count, last_used_at
        for tool in tools:
            assert 'tool_name' in tool
            assert 'count' in tool
            assert 'last_used_at' in tool



    # GET /ready

    def _ready_patches(self, db_ok=True, store_ok=True):
        """Build patch context for /ready — database and store can be individually broken.

        Args:
            db_ok (bool): When False, patches get_shared_db_service() to raise.
            store_ok (bool): When True, uses a real MemoryStore (Category A). When False,
                uses a broken_store MagicMock whose ping() raises (Category C).

        Returns:
            dict: Mapping of patch target strings to mock/real values for use with
                ``unittest.mock.patch``.
        """
        if store_ok:
            # Category A: real MemoryStore — ping() succeeds, no pre-population needed.
            store = MemoryStore()
        else:
            # Category C: error-path only — keep a MagicMock so ping() can raise.
            broken_store = MagicMock()
            broken_store.ping.side_effect = Exception('store down')
            store = broken_store

        # Load the real ONNX embedding model so the /ready endpoint sees _session
        # as non-None and reports 'ok'. No mock — we verify the real model works.
        from services.embedding_service import _get_session_and_tokenizer
        _get_session_and_tokenizer()

        # Patch ONNX service to report ready
        mock_onnx_svc = MagicMock()
        mock_onnx_svc.ready = True

        patches = {
            'services.memory_client.MemoryClientService.create_connection': MagicMock(return_value=store),
            'services.onnx_inference_service.get_onnx_inference_service': MagicMock(return_value=mock_onnx_svc),
        }

        if db_ok:
            # When db_ok, the real db fixture is active and get_shared_db_service()
            # returns the test DatabaseService — no patching needed.
            pass
        else:
            # Force the database check to fail by patching get_shared_db_service
            mock_db = MagicMock()
            mock_db.connection.side_effect = Exception('db down')
            patches['services.database_service.get_shared_db_service'] = MagicMock(return_value=mock_db)

        return patches

    def test_ready_all_ok_returns_200_with_component_status(self, client, db):
        """/ready with all components healthy returns 200 and structured component objects."""
        patches = self._ready_patches()
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target, mock_val in patches.items():
                stack.enter_context(patch(target, mock_val))
            resp = client.get('/ready')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['ready'] is True
        assert data['database'] == {'status': 'ok', 'connected': True}
        assert data['memory_store'] == {'status': 'ok'}

    def test_ready_db_failure_returns_503(self, client, db):
        """/ready with database down returns 503 and error status in database component."""
        patches = self._ready_patches(db_ok=False)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target, mock_val in patches.items():
                stack.enter_context(patch(target, mock_val))
            resp = client.get('/ready')

        assert resp.status_code == 503
        data = resp.get_json()
        assert data['ready'] is False
        assert data['database']['status'] == 'error'
        assert data['database']['connected'] is False
        assert 'message' in data['database']
        assert data['memory_store']['status'] == 'ok'


