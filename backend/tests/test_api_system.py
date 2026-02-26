"""Tests for api/system.py — /health, /metrics, /system/status, /system/observability/* endpoints."""

import json
from datetime import datetime, timezone

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from flask import Flask
from api.system import system_bp


def _make_db_mock(execute_side_effects=None):
    """Build a mock db service whose connection() context manager yields a mock conn.

    ``execute_side_effects`` is an optional list of return values; each call to
    ``conn.execute(...)`` will pop the next one.  When the list contains a simple
    tuple it is wrapped so that ``.fetchone()`` returns that tuple and
    ``.fetchall()`` returns ``[tuple]``.
    """
    mock_conn = MagicMock()

    if execute_side_effects is not None:
        results = []
        for effect in execute_side_effects:
            mock_result = MagicMock()
            if isinstance(effect, tuple):
                mock_result.fetchone.return_value = effect
                mock_result.fetchall.return_value = [effect]
            elif isinstance(effect, list):
                mock_result.fetchone.return_value = effect[0] if effect else None
                mock_result.fetchall.return_value = effect
            else:
                # Allow passing a fully configured MagicMock
                mock_result = effect
            results.append(mock_result)
        mock_conn.execute.side_effect = results

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    mock_db = MagicMock()
    mock_db.connection.return_value = mock_ctx
    return mock_db, mock_conn


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

    def test_system_status_returns_expected_keys(self, client):
        """GET /system/status returns status, memory, storage, queues top-level keys."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.keys.return_value = ['k1', 'k2']
        mock_redis.llen.return_value = 5
        mock_redis.get.return_value = '2026-02-26T10:00:00'

        mock_db, mock_conn = _make_db_mock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (42,)
        mock_conn.execute.return_value = mock_result

        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_redis), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            resp = client.get('/system/status')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert 'memory' in data
        assert 'storage' in data
        assert 'queues' in data
        # Memory keys should reflect redis.keys() calls (3 calls: working_memory, gist, fact)
        assert data['memory']['working_memory_keys'] == 2
        assert data['memory']['gist_keys'] == 2
        assert data['memory']['fact_keys'] == 2
        # Queue depths
        for q in ['prompt-queue', 'output-queue', 'memory-chunker-queue']:
            assert data['queues'][q] == 5

    def test_system_status_degraded_when_redis_fails(self, client):
        """GET /system/status reports 'degraded' when Redis ping raises."""
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = ConnectionError('redis unreachable')
        mock_redis.llen.return_value = 0
        mock_redis.get.return_value = None

        mock_db, mock_conn = _make_db_mock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (0,)
        mock_conn.execute.return_value = mock_result

        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_redis), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            resp = client.get('/system/status')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'degraded'
        assert 'redis_error' in data

    # ────────────────────────────────────────────
    # GET /system/observability/routing
    # ────────────────────────────────────────────

    def test_observability_routing_returns_distribution(self, client):
        """GET /system/observability/routing returns distribution, tiebreaker_rate, etc."""
        mock_svc = MagicMock()
        mock_svc.get_mode_distribution.return_value = {'RESPOND': 60, 'ACT': 25, 'CLARIFY': 15}
        mock_svc.get_tiebreaker_rate.return_value = 0.1234
        mock_svc.get_recent_decisions.return_value = [
            {'selected_mode': 'RESPOND', 'router_confidence': 0.95, 'topic': 'chat', 'created_at': datetime(2026, 2, 26)},
            {'selected_mode': 'ACT', 'router_confidence': 0.80, 'topic': 'task', 'created_at': datetime(2026, 2, 26)},
        ]

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc):
            resp = client.get('/system/observability/routing')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['distribution'] == {'RESPOND': 60, 'ACT': 25, 'CLARIFY': 15}
        assert data['tiebreaker_rate_24h'] == 0.1234
        assert data['total_decisions_24h'] == 2
        assert data['avg_confidence_24h'] == 0.875
        assert len(data['recent']) == 2
        assert data['recent'][0]['mode'] == 'RESPOND'
        assert 'generated_at' in data

    def test_observability_routing_returns_500_on_error(self, client):
        """GET /system/observability/routing returns 500 when service fails."""
        with patch('services.routing_decision_service.RoutingDecisionService', side_effect=RuntimeError('boom')):
            resp = client.get('/system/observability/routing')

        assert resp.status_code == 500
        data = resp.get_json()
        assert 'error' in data

    # ────────────────────────────────────────────
    # GET /system/observability/memory
    # ────────────────────────────────────────────

    def test_observability_memory_returns_all_layers(self, client):
        """GET /system/observability/memory returns episode, concept, trait counts and Redis counts."""
        mock_redis = MagicMock()
        mock_redis.keys.side_effect = lambda pattern: {
            'working_memory:*': ['wm1'],
            'gist_index:*': ['g1', 'g2'],
            'fact_index:*': ['f1', 'f2', 'f3'],
        }.get(pattern, [])
        mock_redis.llen.return_value = 0

        # Three SQL calls: episodes (count+avg), semantic_concepts (count), user_traits (count+avg)
        episodes_result = MagicMock()
        episodes_result.fetchone.return_value = (100, 0.72)
        concepts_result = MagicMock()
        concepts_result.fetchone.return_value = (50,)
        traits_result = MagicMock()
        traits_result.fetchone.return_value = (30, 0.85)

        mock_db, mock_conn = _make_db_mock()
        mock_conn.execute.side_effect = [episodes_result, concepts_result, traits_result]

        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_redis), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            resp = client.get('/system/observability/memory')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['episodes'] == 100
        assert data['avg_episode_activation'] == 0.72
        assert data['concepts'] == 50
        assert data['traits'] == 30
        assert data['avg_trait_strength'] == 0.85
        assert data['working_memory'] == 1
        assert data['gists'] == 2
        assert data['facts'] == 3
        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # GET /system/observability/tools
    # ────────────────────────────────────────────

    def test_observability_tools_returns_stats(self, client):
        """GET /system/observability/tools returns per-tool performance stats."""
        mock_svc = MagicMock()
        mock_svc.get_all_tool_stats.return_value = [
            {'tool': 'web_search', 'success_rate': 0.9, 'avg_latency': 1.2},
            {'tool': 'code_exec', 'success_rate': 0.95, 'avg_latency': 0.8},
        ]

        with patch('services.tool_performance_service.ToolPerformanceService', return_value=mock_svc):
            resp = client.get('/system/observability/tools')

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data['tools']) == 2
        assert data['tools'][0]['tool'] == 'web_search'
        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # GET /system/observability/identity
    # ────────────────────────────────────────────

    def test_observability_identity_returns_vector_states(self, client):
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

    def test_observability_tasks_returns_all_sections(self, client):
        """GET /system/observability/tasks returns persistent_tasks, curiosity_threads, calibration."""
        mock_pt_svc = MagicMock()
        mock_pt_svc.get_active_tasks.return_value = [{'id': 1, 'goal': 'research X', 'state': 'active'}]

        mock_ct_svc = MagicMock()
        mock_ct_svc.get_active_threads.return_value = [
            {'id': 2, 'question': 'Why is the sky blue?', 'last_explored_at': datetime(2026, 2, 26), 'created_at': '2026-02-25', 'last_surfaced_at': None},
        ]

        mock_tc_svc = MagicMock()
        mock_tc_svc.get_calibration_stats.return_value = {'accuracy': 0.88, 'total_scored': 200}

        with patch('services.persistent_task_service.PersistentTaskService', return_value=mock_pt_svc), \
             patch('services.curiosity_thread_service.CuriosityThreadService', return_value=mock_ct_svc), \
             patch('services.triage_calibration_service.TriageCalibrationService', return_value=mock_tc_svc):
            resp = client.get('/system/observability/tasks')

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data['persistent_tasks']) == 1
        assert data['persistent_tasks'][0]['goal'] == 'research X'
        assert len(data['curiosity_threads']) == 1
        # datetime objects should be stringified
        assert data['curiosity_threads'][0]['last_explored_at'] == '2026-02-26 00:00:00'
        # Already a string, should be left as-is
        assert data['curiosity_threads'][0]['created_at'] == '2026-02-25'
        assert data['calibration']['accuracy'] == 0.88
        assert 'generated_at' in data

    def test_observability_tasks_handles_sub_service_failures(self, client):
        """GET /system/observability/tasks gracefully handles individual sub-service failures."""
        with patch('services.persistent_task_service.PersistentTaskService', side_effect=RuntimeError('pt down')), \
             patch('services.curiosity_thread_service.CuriosityThreadService', side_effect=RuntimeError('ct down')), \
             patch('services.triage_calibration_service.TriageCalibrationService', side_effect=RuntimeError('tc down')):
            resp = client.get('/system/observability/tasks')

        assert resp.status_code == 200
        data = resp.get_json()
        # All sections fallback to empty defaults
        assert data['persistent_tasks'] == []
        assert data['curiosity_threads'] == []
        assert data['calibration'] == {}
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

    def test_observability_traits_returns_categories(self, client):
        """GET /system/observability/traits returns traits grouped by category."""
        rows = [
            ('favorite_drink', 'coffee', 0.92, 'preferences', 'inferred', 3, False, datetime(2026, 2, 25)),
            ('name', 'Dylan', 0.99, 'identity', 'explicit', 5, True, datetime(2026, 2, 20)),
            ('language', 'english', 0.85, 'preferences', 'inferred', 1, False, '2026-02-18'),
        ]

        mock_db, mock_conn = _make_db_mock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = rows
        mock_conn.execute.return_value = mock_result

        with patch('services.database_service.get_shared_db_service', return_value=mock_db):
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
        assert pref0['source'] == 'inferred'
        assert pref0['reinforcement_count'] == 3
        assert pref0['is_literal'] is False
        # datetime object should be stringified
        assert pref0['updated_at'] == '2026-02-25 00:00:00'

        ident0 = categories['identity'][0]
        assert ident0['key'] == 'name'
        assert ident0['is_literal'] is True

        # String updated_at should be left as-is
        pref1 = categories['preferences'][1]
        assert pref1['updated_at'] == '2026-02-18'

        assert 'generated_at' in data

    # ────────────────────────────────────────────
    # DELETE /system/observability/traits/<trait_key>
    # ────────────────────────────────────────────

    def test_delete_trait_returns_200(self, client):
        """DELETE /system/observability/traits/<key> returns 200 when row is deleted."""
        mock_db, mock_conn = _make_db_mock()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_conn.execute.return_value = mock_result

        with patch('services.database_service.get_shared_db_service', return_value=mock_db):
            resp = client.delete('/system/observability/traits/favorite_drink')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['ok'] is True
        assert data['deleted'] == 'favorite_drink'

    def test_delete_trait_returns_404_when_not_found(self, client):
        """DELETE /system/observability/traits/<key> returns 404 when trait does not exist."""
        mock_db, mock_conn = _make_db_mock()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_conn.execute.return_value = mock_result

        with patch('services.database_service.get_shared_db_service', return_value=mock_db):
            resp = client.delete('/system/observability/traits/nonexistent')

        assert resp.status_code == 404
        data = resp.get_json()
        assert data['error'] == 'Trait not found'

    # ────────────────────────────────────────────
    # generated_at field on all observability endpoints
    # ────────────────────────────────────────────

    @pytest.mark.parametrize('path,patches', [
        (
            '/system/observability/routing',
            {'services.routing_decision_service.RoutingDecisionService': MagicMock(
                return_value=MagicMock(
                    get_mode_distribution=MagicMock(return_value={}),
                    get_tiebreaker_rate=MagicMock(return_value=0.0),
                    get_recent_decisions=MagicMock(return_value=[]),
                )
            )},
        ),
        (
            '/system/observability/tools',
            {'services.tool_performance_service.ToolPerformanceService': MagicMock(
                return_value=MagicMock(get_all_tool_stats=MagicMock(return_value=[]))
            )},
        ),
        (
            '/system/observability/identity',
            {'services.identity_service.IdentityService': MagicMock(
                return_value=MagicMock(get_vectors=MagicMock(return_value={}))
            )},
        ),
    ], ids=['routing', 'tools', 'identity'])
    def test_observability_endpoints_include_generated_at(self, client, path, patches):
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
