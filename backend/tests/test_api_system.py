"""Tests for api/system.py — /health, /metrics, /system/status, /system/observability/* endpoints."""

import json
from datetime import datetime, timezone

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

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
        # Seed queues with 5 items each so llen == 5.
        for _ in range(5):
            store.rpush('prompt-queue', 'x')
            store.rpush('output-queue', 'x')
        # Seed the last-run timestamp so get() returns a non-None value.
        store.set('cognitive_drift:last_run', '2026-02-26T10:00:00')

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
        # Queue depths
        for q in ['prompt-queue', 'output-queue']:
            assert data['queues'][q] == 5

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
        # Seed episodes
        for i in range(42):
            db.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, "
                "gist, salience, freshness, topic, activation_score) "
                "VALUES (?, '{}', '{}', 'a', '{}', 'ok', 'g', 5, 5, 't', ?)",
                (f'ep-{i}', 0.7123),
            )
        # Seed concept knowledge entries
        for i in range(15):
            db.execute(
                "INSERT INTO knowledge (kind, entity, key, value, confidence) "
                "VALUES ('concept', 'system', ?, 'val', 0.5)",
                (f'concept-{i}',),
            )
        # Seed trait/preference knowledge entries
        for i in range(8):
            db.execute(
                "INSERT INTO knowledge (kind, entity, key, value, confidence) "
                "VALUES (?, 'user', ?, 'val', ?)",
                ('trait' if i < 4 else 'preference', f'trait-{i}', 0.6234),
            )
        db.commit()

        store = MemoryStore()
        # Seed 2 working-memory lists matching the 'working_memory:*' pattern,
        # with 3 and 5 entries respectively (total 8 turns).
        for _ in range(3):
            store.rpush('working_memory:t1', 'x')
        for _ in range(5):
            store.rpush('working_memory:t2', 'x')
        # No facts:* keys → service falls back to traits count (8).

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            resp = client.get('/system/observability/memory')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['episodes'] == 42
        assert data['concepts'] == 15
        assert data['traits'] == 8
        assert data['facts'] == 8  # falls back to traits when no facts:* keys
        assert data['avg_episode_activation'] == 0.7123
        assert data['avg_trait_strength'] == 0.6234
        assert data['working_memory'] == 8  # 3 + 5 turns across two threads
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
            ('tcp-1', 'web_search', 'docker', 'Search', 'Full search profile',
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
        mock_registry.tools = {'web_search': {}, 'code_exec': {}}

        with patch('services.tool_performance_service.ToolPerformanceService', return_value=mock_perf), \
             patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            resp = client.get('/system/observability/tools')

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data['tools']) == 2
        assert data['tools'][0]['tool_name'] in ('web_search', 'code_exec')
        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # GET /system/observability/identity
    # ────────────────────────────────────────────

    def test_observability_identity_returns_vector_states(self, client, db):
        """GET /system/observability/identity returns identity vector breakdown."""
        mock_svc = MagicMock()
        mock_svc.get_vectors.return_value = {
            'warmth': {
                'baseline_weight': 0.6,
                'current_activation': 0.7,
                'plasticity_rate': 0.02,
                'inertia_rate': 0.01,
                'reinforcement_count': 15,
                'min_cap': 0.1,
                'max_cap': 0.9,
            },
        }

        with patch('services.identity_service.IdentityService', return_value=mock_svc):
            resp = client.get('/system/observability/identity')

        assert resp.status_code == 200
        data = resp.get_json()
        warmth = data['vectors']['warmth']
        assert warmth['baseline'] == 0.6
        assert warmth['activation'] == 0.7
        assert warmth['plasticity'] == 0.02
        assert warmth['inertia'] == 0.01
        assert warmth['reinforcements'] == 15
        assert warmth['min'] == 0.1
        assert warmth['max'] == 0.9
        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # GET /system/observability/tasks
    # ────────────────────────────────────────────

    def test_observability_tasks_returns_all_sections(self, client, db):
        """GET /system/observability/tasks returns persistent_tasks."""
        # Seed a master_account row so the endpoint's account_id query works
        db.execute(
            "INSERT INTO master_account (id, username, password_hash) VALUES (1, 'admin', 'hash')"
        )
        db.commit()

        mock_pt_svc = MagicMock()
        mock_pt_svc.get_active_tasks.return_value = [{'id': 1, 'goal': 'research X', 'state': 'active'}]

        with patch('services.persistent_task_service.PersistentTaskService', return_value=mock_pt_svc):
            resp = client.get('/system/observability/tasks')

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data['persistent_tasks']) == 1
        assert data['persistent_tasks'][0]['goal'] == 'research X'
        assert 'generated_at' in data

    def test_observability_tasks_handles_sub_service_failures(self, client):
        """GET /system/observability/tasks gracefully handles individual sub-service failures."""
        with patch('services.persistent_task_service.PersistentTaskService', side_effect=RuntimeError('pt down')):
            resp = client.get('/system/observability/tasks')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['persistent_tasks'] == []
        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # GET /system/observability/autobiography
    # ────────────────────────────────────────────

    def test_observability_autobiography_returns_narrative(self, client):
        """GET /system/observability/autobiography returns narrative data with deltas."""
        mock_auto_svc = MagicMock()
        mock_auto_svc.get_current_narrative.return_value = {
            'narrative': 'User likes coffee and coding.',
            'version': 5,
            'episodes_since': 12,
            'created_at': datetime(2026, 2, 20, 10, 0, 0),
        }

        mock_delta_svc = MagicMock()
        mock_delta_svc.get_changed_sections.return_value = {
            'changed': ['preferences', 'habits'],
            'unchanged': ['identity'],
            'from_version': 4,
            'to_version': 5,
        }

        with patch('services.autobiography_service.AutobiographyService', return_value=mock_auto_svc), \
             patch('services.autobiography_delta_service.AutobiographyDeltaService', return_value=mock_delta_svc):
            resp = client.get('/system/observability/autobiography')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['narrative'] == 'User likes coffee and coding.'
        assert data['version'] == 5
        assert data['episodes_since'] == 12
        assert data['created_at'] == '2026-02-20 10:00:00'
        assert data['delta']['changed'] == ['preferences', 'habits']
        assert data['delta']['from_version'] == 4
        assert data['delta']['to_version'] == 5
        assert 'generated_at' in data

    def test_observability_autobiography_no_narrative(self, client):
        """GET /system/observability/autobiography returns nulls when no narrative exists."""
        mock_auto_svc = MagicMock()
        mock_auto_svc.get_current_narrative.return_value = None

        mock_delta_svc = MagicMock()
        mock_delta_svc.get_changed_sections.return_value = None

        with patch('services.autobiography_service.AutobiographyService', return_value=mock_auto_svc), \
             patch('services.autobiography_delta_service.AutobiographyDeltaService', return_value=mock_delta_svc):
            resp = client.get('/system/observability/autobiography')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['narrative'] is None
        assert data['version'] is None
        assert data['episodes_since'] is None
        assert data['created_at'] is None
        assert data['delta'] is None
        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # GET /system/observability/traits
    # ────────────────────────────────────────────

    def test_observability_traits_returns_categories(self, client, db):
        """GET /system/observability/traits returns traits grouped by category."""
        # Seed knowledge rows with trait/preference kinds and category in data JSON
        db.execute(
            "INSERT INTO knowledge (kind, entity, key, value, confidence, "
            "evidence_count, data, updated_at) "
            "VALUES ('trait', 'user', 'favorite_drink', 'coffee', 0.92, 3, "
            "'{\"category\": \"preferences\"}', '2026-02-25 00:00:00')"
        )
        db.execute(
            "INSERT INTO knowledge (kind, entity, key, value, confidence, "
            "evidence_count, data, updated_at) "
            "VALUES ('trait', 'user', 'name', 'Dylan', 0.99, 5, "
            "'{\"category\": \"identity\"}', '2026-02-20 00:00:00')"
        )
        db.execute(
            "INSERT INTO knowledge (kind, entity, key, value, confidence, "
            "evidence_count, data, updated_at) "
            "VALUES ('preference', 'user', 'language', 'english', 0.85, 1, "
            "'{\"category\": \"preferences\"}', '2026-02-18')"
        )
        db.commit()

        resp = client.get('/system/observability/traits')

        assert resp.status_code == 200
        data = resp.get_json()
        categories = data['categories']
        assert 'preferences' in categories
        assert 'identity' in categories
        assert len(categories['preferences']) == 2
        assert len(categories['identity']) == 1

        pref0 = categories['preferences'][0]
        assert pref0['key'] == 'favorite_drink'
        assert pref0['value'] == 'coffee'
        assert pref0['confidence'] == 0.92
        assert pref0['reinforcement_count'] == 3
        assert pref0['updated_at'] == '2026-02-25 00:00:00'

        ident0 = categories['identity'][0]
        assert ident0['key'] == 'name'

        # String updated_at should be left as-is
        pref1 = categories['preferences'][1]
        assert pref1['updated_at'] == '2026-02-18'

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

    def test_observability_identity_passes_db_to_service(self, client, db):
        """IdentityService must be constructed with get_shared_db_service()."""
        mock_svc = MagicMock()
        mock_svc.get_vectors.return_value = {}
        mock_cls = MagicMock(return_value=mock_svc)

        with patch('services.identity_service.IdentityService', mock_cls):
            resp = client.get('/system/observability/identity')

        assert resp.status_code == 200
        # Verify the service was called with the real db_service singleton
        mock_cls.assert_called_once()
        from services.database_service import get_shared_db_service
        assert mock_cls.call_args[0][0] is get_shared_db_service()

    def test_observability_tasks_passes_db_to_persistent_task_service(self, client, db):
        """PersistentTaskService must be constructed with get_shared_db_service()."""
        mock_pt_svc = MagicMock()
        mock_pt_svc.get_active_tasks.return_value = []
        mock_pt_cls = MagicMock(return_value=mock_pt_svc)

        with patch('services.persistent_task_service.PersistentTaskService', mock_pt_cls):
            resp = client.get('/system/observability/tasks')

        assert resp.status_code == 200
        from services.database_service import get_shared_db_service
        mock_pt_cls.assert_called_once_with(get_shared_db_service())

    def test_observability_autobiography_passes_db_to_both_services(self, client, db):
        """AutobiographyService and AutobiographyDeltaService must receive get_shared_db_service()."""
        mock_auto_svc = MagicMock()
        mock_auto_svc.get_current_narrative.return_value = None
        mock_auto_cls = MagicMock(return_value=mock_auto_svc)

        mock_delta_svc = MagicMock()
        mock_delta_svc.get_changed_sections.return_value = None
        mock_delta_cls = MagicMock(return_value=mock_delta_svc)

        with patch('services.autobiography_service.AutobiographyService', mock_auto_cls), \
             patch('services.autobiography_delta_service.AutobiographyDeltaService', mock_delta_cls):
            resp = client.get('/system/observability/autobiography')

        assert resp.status_code == 200
        from services.database_service import get_shared_db_service
        db_svc = get_shared_db_service()
        mock_auto_cls.assert_called_once_with(db_svc)
        mock_delta_cls.assert_called_once_with(db_svc)

    def test_observability_memory_returns_flat_structure(self, client, db):
        """Memory endpoint returns flat counts (episodes, concepts, traits, etc.)."""
        # Seed episodes
        for i in range(10):
            db.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, "
                "gist, salience, freshness, topic, activation_score) "
                "VALUES (?, '{}', '{}', 'a', '{}', 'ok', 'g', 5, 5, 't', 0.5)",
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
        (
            '/system/observability/identity',
            {'services.identity_service.IdentityService': MagicMock(
                return_value=MagicMock(get_vectors=MagicMock(return_value={}))
            )},
        ),
    ], ids=['tools', 'identity'])
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

    def _ready_patches(self, db_ok=True, store_ok=True, worker_ok=True):
        """Build patch context for /ready — all three components can be individually broken.

        Args:
            db_ok (bool): When False, patches get_shared_db_service() to raise.
            store_ok (bool): When True, uses a real MemoryStore (Category A). When False,
                uses a broken_store MagicMock whose ping() raises (Category C).
            worker_ok (bool): When False, patches PromptQueue to raise ImportError.

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

        if not worker_ok:
            patches['services.prompt_queue.PromptQueue'] = MagicMock(side_effect=ImportError('no queue'))
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
        assert data['workers'] == {'status': 'ok'}

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
        assert data['workers']['status'] == 'ok'

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
