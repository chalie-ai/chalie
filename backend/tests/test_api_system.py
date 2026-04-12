"""Tests for api/system.py — /health, /metrics, /system/status, /system/observability/* endpoints."""

from datetime import datetime

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

    # ────────────────────────────────────────────
    # GET /health
    # ────────────────────────────────────────────

    def test_get_health_returns_ok_and_version(self, client):
        """GET /health returns status 'ok' and the current APP_VERSION."""
        with patch('consumer.APP_VERSION', '2.5.0'):
            resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert data['version'] == '2.5.0'
        # GET variant does not include 'attention' field
        assert 'attention' not in data

    # ────────────────────────────────────────────
    # POST /health
    # ────────────────────────────────────────────

    def test_post_health_saves_context_and_returns_attention(self, client):
        """POST /health saves client context and returns inferred attention."""
        mock_ctx_svc = MagicMock()
        mock_ambient_svc = MagicMock()
        mock_ambient_svc.infer.return_value = {'attention': 'focused'}

        with patch('consumer.APP_VERSION', '2.5.0'), \
             patch('services.client_context_service.ClientContextService', return_value=mock_ctx_svc), \
             patch('services.ambient_inference_service.AmbientInferenceService', return_value=mock_ambient_svc):
            resp = client.post('/health', json={'battery': 80, 'screen': 'on'})

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert data['version'] == '2.5.0'
        assert data['attention'] == 'focused'
        mock_ctx_svc.save.assert_called_once_with({'battery': 80, 'screen': 'on'})
        mock_ambient_svc.infer.assert_called_once_with({'battery': 80, 'screen': 'on'})

    def test_post_health_empty_body_returns_ok(self, client):
        """POST /health with an empty JSON body still returns 200 ok with attention=None."""
        with patch('consumer.APP_VERSION', '1.0.0'):
            resp = client.post('/health', data='{}', content_type='application/json')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert data['attention'] is None

    # ────────────────────────────────────────────
    # GET /metrics
    # ────────────────────────────────────────────

    def test_get_metrics_returns_dashboard_data(self, client):
        """GET /metrics proxies MetricsService.get_dashboard_data()."""
        mock_svc = MagicMock()
        mock_svc.get_dashboard_data.return_value = {'requests_per_min': 42, 'uptime': 3600}

        with patch('services.metrics_service.MetricsService', return_value=mock_svc):
            resp = client.get('/metrics')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['requests_per_min'] == 42
        assert data['uptime'] == 3600

    def test_get_metrics_returns_500_on_service_error(self, client):
        """GET /metrics returns 500 when MetricsService raises."""
        with patch('services.metrics_service.MetricsService', side_effect=RuntimeError('db down')):
            resp = client.get('/metrics')

        assert resp.status_code == 500
        data = resp.get_json()
        assert 'error' in data

    # ────────────────────────────────────────────
    # GET /system/status
    # ────────────────────────────────────────────

    def test_system_status_returns_expected_keys(self, client, db):
        """GET /system/status returns status, memory, storage, queues top-level keys."""
        store = MemoryStore()
        # Seed 2 keys for each memory namespace so counts == 2.
        for ns in ('working_memory', 'gist_index', 'fact_index'):
            store.set(f'{ns}:a', 'x')
            store.set(f'{ns}:b', 'x')
        # Seed output-queue with 5 items so llen == 5.
        for _ in range(5):
            store.rpush('output-queue', 'x')
        # Seed DMN delivery ZSET with a recent entry.
        store.zadd('dmn:deliveries', {'test-delivery': 1711500000.0})

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            resp = client.get('/system/status')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert 'memory' in data
        assert 'storage' in data
        assert 'queues' in data
        # Memory keys should reflect store.keys() calls (3 calls: working_memory, gist, fact)
        assert data['memory']['working_memory_keys'] == 2
        assert data['memory']['gist_keys'] == 2
        assert data['memory']['fact_keys'] == 2
        # Queue depth
        assert data['queues']['output-queue'] == 5
        assert 'prompt-queue' not in data['queues']

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

    # ────────────────────────────────────────────
    # GET /system/observability/memory
    # ────────────────────────────────────────────

    def test_observability_memory_returns_all_layers(self, client, db):
        """GET /system/observability/memory returns flat counts from SQLite and MemoryStore."""
        now_iso = '2026-01-01T00:00:00+00:00'

        # Seed episodes
        for i in range(42):
            db.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, "
                "gist, salience, channel) "
                "VALUES (?, '{}', '{}', 'a', '{}', 'ok', 'g', 5, 't')",
                (f'ep-{i}',),
            )
        # Seed user_specific data_graph entries — these count as both concepts AND traits
        # The endpoint counts data_graph WHERE kind='user_specific' for both.
        for i in range(8):
            db.execute(
                "INSERT INTO data_graph (kind, key, value, retrieval_weight, "
                "first_seen_at, last_confirmed_at) "
                "VALUES ('user_specific', ?, 'val', 0.6234, ?, ?)",
                (f'trait-{i}', now_iso, now_iso),
            )
        db.commit()

        store = MemoryStore()
        for _ in range(3):
            store.rpush('working_memory:t1', 'x')
        for _ in range(5):
            store.rpush('working_memory:t2', 'x')
        # No facts:* keys → falls back to traits count (8).

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            resp = client.get('/system/observability/memory')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['episodes'] == 42
        # concepts and traits both count data_graph user_specific rows
        assert data['concepts'] == 8
        assert data['traits'] == 8
        assert data['facts'] == 8  # falls back to traits when no facts:* keys
        assert data['avg_episode_activation'] == 1.0
        assert data['avg_trait_strength'] == pytest.approx(0.6234, abs=0.001)
        assert data['working_memory'] == 8
        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # GET /system/observability/tools
    # ────────────────────────────────────────────

    def test_observability_tools_returns_stats(self, client, db):
        """GET /system/observability/tools returns per-tool performance stats."""
        # Seed tool_capability_profiles rows
        db.execute(
            "INSERT INTO tool_capability_profiles "
            "(id, tool_name, tool_type, short_summary, full_profile, domain, effort, "
            "reliability_score, cost_tier, avg_latency_ms, enrichment_count, "
            "triage_triggers, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ('tcp-1', 'weather', 'docker', 'Search', 'Full search profile',
             'Search', 'low', 0.9, 'free', 1200, 1, '[]', '2025-01-01T00:00:00'),
        )
        db.execute(
            "INSERT INTO tool_capability_profiles "
            "(id, tool_name, tool_type, short_summary, full_profile, domain, effort, "
            "reliability_score, cost_tier, avg_latency_ms, enrichment_count, "
            "triage_triggers, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ('tcp-2', 'code_exec', 'docker', 'Execute', 'Full exec profile',
             'Dev', 'moderate', 0.95, 'free', 800, 1, '[]', '2025-01-01T00:00:00'),
        )
        db.commit()

        mock_perf = MagicMock()
        mock_perf.get_all_tool_stats.return_value = []

        # Mock the tool registry to include our two tool names so the WHERE IN filter matches
        mock_registry = MagicMock()
        mock_registry.tools = {'weather': {}, 'code_exec': {}}

        with patch('services.tool_performance_service.ToolPerformanceService', return_value=mock_perf), \
             patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            resp = client.get('/system/observability/tools')

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data['tools']) == 2
        assert data['tools'][0]['tool_name'] in ('weather', 'code_exec')
        assert 'generated_at' in data



    # ────────────────────────────────────────────
    # GET /system/observability/tasks
    # ────────────────────────────────────────────

    def test_observability_tasks_returns_goal_ecology_stats(self, client):
        """GET /system/observability/tasks returns goal_ecology_stats."""
        resp = client.get('/system/observability/tasks')

        assert resp.status_code == 200
        data = resp.get_json()
        assert 'generated_at' in data
        assert 'goal_ecology_stats' in data

    def test_observability_tasks_handles_store_failures(self, client):
        """GET /system/observability/tasks returns 200 even if store query fails."""
        with patch('services.memory_client.MemoryClientService.create_connection', side_effect=RuntimeError('store down')):
            resp = client.get('/system/observability/tasks')

        assert resp.status_code == 200
        data = resp.get_json()
        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # GET /system/observability/traits
    # ────────────────────────────────────────────

    def test_observability_traits_returns_categories(self, client, db):
        """GET /system/observability/traits returns traits from data_graph."""
        # Seed data_graph rows (endpoint now reads data_graph, not knowledge)
        now_iso = '2026-02-25T00:00:00+00:00'
        db.execute(
            "INSERT INTO data_graph (kind, key, value, retrieval_weight, "
            "evidence_count, first_seen_at, last_confirmed_at) "
            "VALUES ('user_specific', 'favorite_drink', 'coffee', 0.92, 3, ?, ?)",
            (now_iso, now_iso)
        )
        db.execute(
            "INSERT INTO data_graph (kind, key, value, retrieval_weight, "
            "evidence_count, first_seen_at, last_confirmed_at) "
            "VALUES ('user_specific', 'name', 'Dylan', 0.99, 5, ?, ?)",
            (now_iso, now_iso)
        )
        db.commit()

        resp = client.get('/system/observability/traits')

        assert resp.status_code == 200
        data = resp.get_json()
        categories = data['categories']
        # All user_specific rows go under 'general' in current implementation
        assert 'general' in categories
        assert len(categories['general']) == 2

        keys = {r['key'] for r in categories['general']}
        assert 'favorite_drink' in keys
        assert 'name' in keys

        drink = next(r for r in categories['general'] if r['key'] == 'favorite_drink')
        assert drink['value'] == 'coffee'
        assert drink['confidence'] == pytest.approx(0.92, abs=0.01)
        assert drink['reinforcement_count'] == 3

        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # DELETE /system/observability/traits/<trait_key>
    # ────────────────────────────────────────────

    def test_delete_trait_returns_200(self, client, db):
        """DELETE /system/observability/traits/<key> returns 200 when row is deleted."""
        # Seed a trait to be deleted
        db.execute(
            "INSERT INTO knowledge (kind, entity, key, value, confidence) "
            "VALUES ('trait', 'user', 'favorite_drink', 'coffee', 0.9)"
        )
        db.commit()

        resp = client.delete('/system/observability/traits/favorite_drink')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['ok'] is True
        assert data['deleted'] == 'favorite_drink'

    def test_delete_trait_returns_404_when_not_found(self, client, db):
        """DELETE /system/observability/traits/<key> returns 404 when trait does not exist."""
        resp = client.delete('/system/observability/traits/nonexistent')

        assert resp.status_code == 404
        data = resp.get_json()
        assert data['error'] == 'Trait not found'

    # ────────────────────────────────────────────
    # Service constructor receives db_service
    # (regression guards against missing arg 500s)
    # ────────────────────────────────────────────

    def test_observability_memory_returns_flat_structure(self, client, db):
        """Memory endpoint returns flat counts (episodes, concepts, traits, etc.)."""
        # Seed episodes
        for i in range(10):
            db.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, "
                "gist, salience, channel) "
                "VALUES (?, '{}', '{}', 'a', '{}', 'ok', 'g', 5, 't')",
                (f'ep-flat-{i}',),
            )
        # Seed concepts
        for i in range(5):
            db.execute(
                "INSERT INTO knowledge (kind, entity, key, value, confidence) "
                "VALUES ('concept', 'system', ?, 'val', 0.5)",
                (f'concept-flat-{i}',),
            )
        # Seed traits
        for i in range(3):
            db.execute(
                "INSERT INTO knowledge (kind, entity, key, value, confidence) "
                "VALUES ('trait', 'user', ?, 'val', 0.4)",
                (f'trait-flat-{i}',),
            )
        db.commit()

        # A fresh MemoryStore naturally returns [] for keys() and 0 for llen() —
        # no pre-population needed for this structural smoke-test.
        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            resp = client.get('/system/observability/memory')

        assert resp.status_code == 200
        data = resp.get_json()
        assert 'episodes' in data
        assert 'concepts' in data
        assert 'traits' in data
        assert 'avg_episode_activation' in data
        assert 'avg_trait_strength' in data
        assert 'working_memory' in data

    # ────────────────────────────────────────────
    # generated_at field on all observability endpoints
    # ────────────────────────────────────────────

    @pytest.mark.parametrize('path,patches', [
        (
            '/system/observability/tools',
            {
                'services.tool_performance_service.ToolPerformanceService': MagicMock(
                    return_value=MagicMock(get_all_tool_stats=MagicMock(return_value=[]))
                ),
                'services.tool_registry_service.ToolRegistryService': MagicMock(
                    return_value=MagicMock(tools={})
                ),
            },
        ),
    ], ids=['tools'])
    def test_observability_endpoints_include_generated_at(self, client, db, path, patches):
        """All observability endpoints include a generated_at ISO timestamp."""
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target, mock_val in patches.items():
                stack.enter_context(patch(target, mock_val))
            resp = client.get(path)

        assert resp.status_code == 200
        data = resp.get_json()
        assert 'generated_at' in data
        # Should be a valid ISO 8601 string
        parsed = datetime.fromisoformat(data['generated_at'])
        assert parsed.tzinfo is not None

    # ────────────────────────────────────────────
    # GET /ready
    # ────────────────────────────────────────────

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
        assert 'workers' not in data

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

    def test_ready_store_failure_returns_503(self, client, db):
        """/ready with memory store down returns 503 and error in memory_store component."""
        patches = self._ready_patches(store_ok=False)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target, mock_val in patches.items():
                stack.enter_context(patch(target, mock_val))
            resp = client.get('/ready')

        assert resp.status_code == 503
        data = resp.get_json()
        assert data['ready'] is False
        assert data['database']['status'] == 'ok'
        assert data['memory_store']['status'] == 'error'
        assert 'message' in data['memory_store']

    def test_ready_no_checks_key_in_response(self, client, db):
        """Response must not include the legacy 'checks' key — components are top-level."""
        patches = self._ready_patches()
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target, mock_val in patches.items():
                stack.enter_context(patch(target, mock_val))
            resp = client.get('/ready')

        data = resp.get_json()
        assert 'checks' not in data
