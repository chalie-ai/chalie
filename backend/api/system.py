"""
System blueprint — /health, /metrics, /system/status, /system/observability/* endpoints.
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

system_bp = Blueprint('system', __name__)


@system_bp.route('/health', methods=['GET', 'POST'])
def health_check():
    """Health check endpoint (no auth required). POST saves client context."""
    if request.method == 'POST':
        attention = None
        try:
            from services.client_context_service import ClientContextService
            data = request.get_json() or {}
            if data:
                svc = ClientContextService()
                svc.save(data)
                # Run ambient inference on the saved context
                from services.ambient_inference_service import AmbientInferenceService
                inference = AmbientInferenceService().infer(data)
                attention = inference.get('attention') if inference else None
        except Exception as e:
            logger.warning(f"[HEALTH] Failed to save client context: {e}")
        from consumer import APP_VERSION
        return jsonify({"status": "ok", "version": APP_VERSION, "attention": attention}), 200
    from consumer import APP_VERSION
    return jsonify({"status": "ok", "version": APP_VERSION}), 200


@system_bp.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness probe — true only when SQLite, MemoryStore, and prompt-queue worker are all available."""
    components = {}

    # SQLite
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            cursor.close()
        components['database'] = {'status': 'ok', 'connected': True}
    except Exception as e:
        logger.debug(f'[READY] database not ready: {e}')
        components['database'] = {'status': 'error', 'connected': False, 'message': str(e)}

    # MemoryStore
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        store.ping()
        components['memory_store'] = {'status': 'ok'}
    except Exception as e:
        logger.debug(f'[READY] memory store not ready: {e}')
        components['memory_store'] = {'status': 'error', 'message': str(e)}

    # prompt-queue worker (PromptQueue is an in-process thread dispatcher — always available
    # once the module is importable; _locks is lazily populated on first enqueue so checking
    # it causes a false 503 on every cold boot before the first message arrives)
    try:
        from services.prompt_queue import PromptQueue  # noqa: F401 — import-only check
        components['workers'] = {'status': 'ok'}
    except Exception as e:
        logger.debug(f'[READY] worker check failed: {e}')
        components['workers'] = {'status': 'error', 'message': str(e)}

    # Embedding model — preloaded in background thread on boot. Not ready until
    # the model is loaded AND the first inference pass has warmed the graph.
    try:
        from services.embedding_service import _st_model
        if _st_model is not None:
            components['embeddings'] = {'status': 'ok'}
        else:
            components['embeddings'] = {'status': 'loading'}
    except Exception as e:
        logger.debug(f'[READY] embedding model not ready: {e}')
        components['embeddings'] = {'status': 'error', 'message': str(e)}

    ready = all(c.get('status') == 'ok' for c in components.values())
    return jsonify({'ready': ready, **components}), (200 if ready else 503)


@system_bp.route('/metrics', methods=['GET'])
@require_session
def metrics_endpoint():
    """Metrics dashboard endpoint."""
    try:
        from services.metrics_service import MetricsService
        metrics = MetricsService()
        data = metrics.get_dashboard_data()
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"[REST API] Metrics error: {e}")
        return jsonify({"error": "Failed to retrieve metrics"}), 500


@system_bp.route('/system/status', methods=['GET'])
@require_session
def system_status():
    """Comprehensive system health and diagnostics."""
    try:
        from services.memory_client import MemoryClientService
        from services.database_service import get_shared_db_service

        store = MemoryClientService.create_connection()
        result = {"status": "ok", "memory": {}, "storage": {}, "queues": {}}

        # MemoryStore health
        try:
            store.ping()
            # Count memory store keys
            result["memory"]["working_memory_keys"] = len(store.keys("working_memory:*"))
            result["memory"]["gist_keys"] = len(store.keys("gist_index:*"))
            result["memory"]["fact_keys"] = len(store.keys("fact_index:*"))
        except Exception as e:
            result["status"] = "degraded"
            result["memory_store_error"] = str(e)

        # SQLite counts
        try:
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                for table in ["episodes", "semantic_concepts", "user_traits"]:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        row = cursor.fetchone()
                        result["storage"][table] = row[0] if row else 0
                    except Exception:
                        result["storage"][table] = -1
        except Exception as e:
            result["status"] = "degraded"
            result["database_error"] = str(e)

        # Queue depths
        for queue_name in ["prompt-queue", "output-queue"]:
            try:
                result["queues"][queue_name] = store.llen(queue_name)
            except Exception:
                result["queues"][queue_name] = -1

        # Last proactive drift run
        try:
            last_run = store.get("cognitive_drift:last_run")
            result["last_proactive_run"] = last_run if last_run else None
        except Exception:
            pass

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] System status error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# Observability — cognitive legibility endpoints
# ─────────────────────────────────────────────

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


@system_bp.route('/system/observability/routing', methods=['GET'])
@require_session
def observability_routing():
    """Mode router decision distribution and recent activity."""
    try:
        from services.routing_decision_service import RoutingDecisionService
        from services.database_service import get_shared_db_service

        svc = RoutingDecisionService(get_shared_db_service())
        distribution = svc.get_mode_distribution(168)
        tiebreaker_rate = svc.get_tiebreaker_rate(24)
        recent = svc.get_recent_decisions(24, 20)

        # Compute 24h totals and avg confidence from recent decisions
        total_24h = len(recent)
        avg_confidence = 0.0
        if total_24h:
            avg_confidence = sum(d.get('router_confidence', 0) or 0 for d in recent) / total_24h

        # Compact recent: mode, confidence, topic, created_at only
        compact_recent = []
        for d in recent:
            compact_recent.append({
                'mode': d.get('selected_mode', ''),
                'confidence': round(d.get('router_confidence', 0) or 0, 3),
                'topic': d.get('topic', ''),
                'created_at': str(d['created_at']) if d.get('created_at') else None,
            })

        return jsonify({
            'generated_at': _now_iso(),
            'distribution': distribution,
            'tiebreaker_rate_24h': round(tiebreaker_rate, 4),
            'avg_confidence_24h': round(avg_confidence, 3),
            'total_decisions_24h': total_24h,
            'recent': compact_recent,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/routing error: {e}")
        return jsonify({"error": "Failed to retrieve routing data"}), 500


@system_bp.route('/system/observability/memory', methods=['GET'])
@require_session
def observability_memory():
    """Memory layer counts and health indicators."""
    try:
        from services.memory_client import MemoryClientService
        from services.self_model_service import SelfModelService

        # SelfModelService holds a cached (sub-ms) snapshot with the canonical
        # nested structure: operational.memory_pressure.{episode_count, ...}
        # Use it directly to avoid redundant DB queries and ensure the response
        # shape matches what consumers (e.g. get_memory_richness()) expect.
        snapshot = SelfModelService().get_snapshot()

        result = {
            'generated_at': _now_iso(),
            'operational': snapshot.get('operational', {}),
            'epistemic': snapshot.get('epistemic', {}),
            'noteworthy': snapshot.get('noteworthy', []),
            'working_memory': 0,
            'gists': 0,
            'facts': 0,
            'queues': {},
        }

        # MemoryStore counts (not in the snapshot — add alongside)
        try:
            store = MemoryClientService.create_connection()
            result['working_memory'] = len(store.keys("working_memory:*"))

            for q in ["prompt-queue", "output-queue"]:
                depth = store.llen(q)
                if depth:
                    result['queues'][q] = depth
        except Exception as e:
            logger.warning(f"[OBS] memory store error: {e}")

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"[REST API] observability/memory error: {e}")
        return jsonify({"error": "Failed to retrieve memory data"}), 500


@system_bp.route('/system/observability/tools', methods=['GET'])
@require_session
def observability_tools():
    """Tool capability profiles with effort annotations and performance stats."""
    try:
        from services.database_service import get_shared_db_service
        from services.tool_performance_service import ToolPerformanceService

        db = get_shared_db_service()
        rows = db.fetch_all(
            "SELECT tool_name, tool_type, short_summary, domain, effort, "
            "reliability_score, cost_tier, avg_latency_ms, enrichment_count, "
            "triage_triggers, updated_at "
            "FROM tool_capability_profiles ORDER BY domain, tool_name"
        )

        # Index performance stats by tool name for merging
        perf_by_name = {}
        try:
            perf_stats = ToolPerformanceService().get_all_tool_stats(30)
            for s in (perf_stats or []):
                perf_by_name[s.get('tool_name', '')] = s
        except Exception:
            pass

        import json as _json
        tools = []
        for r in (rows or []):
            triggers = r.get('triage_triggers') or []
            if isinstance(triggers, str):
                try:
                    triggers = _json.loads(triggers)
                except Exception:
                    triggers = []
            entry = {
                'tool_name': r['tool_name'],
                'tool_type': r.get('tool_type', 'tool'),
                'summary': f"{r.get('short_summary', '')} (effort: {r.get('effort') or 'moderate'})",
                'domain': r.get('domain') or 'Other',
                'effort': r.get('effort') or 'moderate',
                'reliability_score': r.get('reliability_score', 1.0),
                'cost_tier': r.get('cost_tier', 'free'),
                'avg_latency_ms': r.get('avg_latency_ms', 0),
                'enrichment_count': r.get('enrichment_count', 0),
                'triage_triggers': triggers,
                'updated_at': r.get('updated_at'),
            }
            if r['tool_name'] in perf_by_name:
                entry['performance'] = perf_by_name[r['tool_name']]
            tools.append(entry)

        return jsonify({
            'generated_at': _now_iso(),
            'tools': tools,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/tools error: {e}")
        return jsonify({"error": "Failed to retrieve tool data"}), 500


@system_bp.route('/system/observability/identity', methods=['GET'])
@require_session
def observability_identity():
    """Identity vector states."""
    try:
        from services.identity_service import IdentityService
        from services.database_service import get_shared_db_service

        svc = IdentityService(get_shared_db_service())
        raw = svc.get_vectors()

        vectors = {}
        for name, state in raw.items():
            vectors[name] = {
                'baseline': state.get('baseline_weight', 0.5),
                'activation': state.get('current_activation', 0.5),
                'plasticity': state.get('plasticity_rate', 0),
                'inertia': state.get('inertia_rate', 0),
                'reinforcements': state.get('reinforcement_count', 0),
                'min': state.get('min_cap', 0),
                'max': state.get('max_cap', 1),
            }

        return jsonify({
            'generated_at': _now_iso(),
            'vectors': vectors,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/identity error: {e}")
        return jsonify({"error": "Failed to retrieve identity data"}), 500


@system_bp.route('/system/observability/tasks', methods=['GET'])
@require_session
def observability_tasks():
    """Active persistent tasks and curiosity threads."""
    try:
        result = {
            'generated_at': _now_iso(),
            'persistent_tasks': [],
            'curiosity_threads': [],
        }

        # Persistent tasks
        try:
            from services.persistent_task_service import PersistentTaskService
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            svc = PersistentTaskService(db)
            with db.connection() as conn:
                row = conn.execute("SELECT id FROM master_account LIMIT 1").fetchone()
            account_id = row[0] if row else 1
            result['persistent_tasks'] = svc.get_active_tasks(account_id)
        except Exception as e:
            logger.warning(f"[OBS] persistent tasks error: {e}")

        # Curiosity threads — datetime fields need str() conversion
        try:
            from services.curiosity_thread_service import CuriosityThreadService
            svc = CuriosityThreadService()
            threads = svc.get_active_threads()
            for t in threads:
                for key in ('last_explored_at', 'created_at', 'last_surfaced_at'):
                    if key in t and t[key] is not None and not isinstance(t[key], str):
                        t[key] = str(t[key])
            result['curiosity_threads'] = threads
        except Exception as e:
            logger.warning(f"[OBS] curiosity threads error: {e}")

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"[REST API] observability/tasks error: {e}")
        return jsonify({"error": "Failed to retrieve task data"}), 500


@system_bp.route('/system/observability/tasks/<int:task_id>', methods=['DELETE'])
@require_session
def cancel_persistent_task(task_id):
    """Cancel (dismiss) a persistent background task."""
    try:
        from services.persistent_task_service import PersistentTaskService
        from services.database_service import get_shared_db_service
        svc = PersistentTaskService(get_shared_db_service())
        ok, msg = svc.transition(task_id, 'cancelled')
        if ok:
            return jsonify({"status": "cancelled"}), 200
        return jsonify({"error": msg}), 400
    except Exception as e:
        logger.error(f"[REST API] cancel task error: {e}")
        return jsonify({"error": "Failed to cancel task"}), 500


@system_bp.route('/system/observability/autobiography', methods=['GET'])
@require_session
def observability_autobiography():
    """Current autobiography narrative with delta information."""
    try:
        from services.autobiography_service import AutobiographyService
        from services.autobiography_delta_service import AutobiographyDeltaService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        narrative_data = AutobiographyService(db).get_current_narrative()
        delta_data = AutobiographyDeltaService(db).get_changed_sections()

        result = {
            'generated_at': _now_iso(),
            'narrative': None,
            'version': None,
            'episodes_since': None,
            'created_at': None,
            'delta': None,
        }

        if narrative_data:
            result['narrative'] = narrative_data.get('narrative')
            result['version'] = narrative_data.get('version')
            result['episodes_since'] = narrative_data.get('episodes_since')
            created = narrative_data.get('created_at')
            result['created_at'] = str(created) if created and not isinstance(created, str) else created

        if delta_data:
            result['delta'] = {
                'changed': delta_data.get('changed', []),
                'unchanged': delta_data.get('unchanged', []),
                'from_version': delta_data.get('from_version'),
                'to_version': delta_data.get('to_version'),
            }

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"[REST API] observability/autobiography error: {e}")
        return jsonify({"error": "Failed to retrieve autobiography data"}), 500


@system_bp.route('/system/observability/traits', methods=['GET'])
@require_session
def observability_traits():
    """User traits grouped by category."""
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        categories = {}

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT trait_key, trait_value, confidence, category, "
                "reinforcement_count, updated_at "
                "FROM user_traits "
                "ORDER BY category, confidence DESC"
            )
            rows = cursor.fetchall()

            for row in rows:
                cat = row[3] or 'general'
                if cat not in categories:
                    categories[cat] = []
                updated = row[5]
                categories[cat].append({
                    'key': row[0],
                    'value': row[1],
                    'confidence': round(float(row[2] or 0), 3),
                    'reinforcement_count': row[4] or 0,
                    'updated_at': str(updated) if updated and not isinstance(updated, str) else updated,
                })

        return jsonify({
            'generated_at': _now_iso(),
            'categories': categories,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/traits error: {e}")
        return jsonify({"error": "Failed to retrieve traits data"}), 500


@system_bp.route('/system/observability/traits/<trait_key>', methods=['DELETE'])
@require_session
def observability_delete_trait(trait_key):
    """Delete a specific user trait by key."""
    try:
        from services.database_service import get_shared_db_service
        from services.user_trait_service import UserTraitService

        db = get_shared_db_service()
        svc = UserTraitService(db)
        deleted = svc.delete_trait(trait_key)

        if deleted:
            return jsonify({'ok': True, 'deleted': trait_key}), 200
        else:
            return jsonify({'error': 'Trait not found'}), 404
    except Exception as e:
        logger.error(f"[REST API] observability/traits DELETE error: {e}")
        return jsonify({"error": "Failed to delete trait"}), 500


@system_bp.route('/system/observability/temporal', methods=['GET'])
@require_session
def observability_temporal():
    """Temporal pattern mining stats and prediction availability."""
    try:
        from services.temporal_pattern_service import TemporalPatternService
        from services.database_service import get_shared_db_service
        from services.memory_client import MemoryClientService

        db = get_shared_db_service()
        service = TemporalPatternService(db)
        stats = service.get_observation_stats()

        # Add last mining run time from MemoryStore
        store = MemoryClientService.create_connection()
        last_run = store.get("temporal:last_mining_run")
        stats['mining_last_run'] = last_run if last_run else None

        return jsonify({
            'generated_at': _now_iso(),
            **stats,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/temporal error: {e}")
        return jsonify({"error": "Failed to retrieve temporal data"}), 500


@system_bp.route('/system/observability/temporal/mine', methods=['POST'])
@require_session
def observability_temporal_mine():
    """Trigger on-demand temporal pattern mining."""
    try:
        from services.temporal_pattern_service import TemporalPatternService, observation_buffer
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()

        # Flush pending observations first
        observation_buffer.flush(db)

        # Run mining
        service = TemporalPatternService(db)
        patterns = service.mine_patterns()

        return jsonify({
            'patterns_mined': len(patterns),
            'mining_duration_seconds': round(service._last_mining_duration, 3),
            'patterns': [{'key': p['key'], 'value': p['value'],
                          'confidence': p['confidence']} for p in patterns],
        }), 200
    except Exception as e:
        logger.error(f"[REST API] temporal/mine error: {e}")
        return jsonify({"error": "Mining failed"}), 500


@system_bp.route('/system/observability/self-model', methods=['GET'])
@require_session
def observability_self_model():
    """Self-model snapshot: epistemic, operational, capability state."""
    try:
        from services.self_model_service import SelfModelService
        snapshot = SelfModelService().get_snapshot()
        return jsonify(snapshot), 200
    except Exception as e:
        logger.error(f"[REST API] observability/self-model error: {e}")
        return jsonify({"error": "Failed to retrieve self-model"}), 500


@system_bp.route('/system/observability/capability-gaps', methods=['GET'])
@require_session
def observability_capability_gaps():
    """Capability gaps — things users ask for that Chalie cannot do."""
    try:
        from services.self_model_service import SelfModelService
        service = SelfModelService()
        gaps = service.get_frequent_gaps(min_occurrences=1, limit=20)
        return jsonify({
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_unresolved': len(gaps),
            'gaps': gaps,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/capability-gaps error: {e}")
        return jsonify({"error": "Failed to retrieve capability gaps"}), 500


@system_bp.route('/system/activity', methods=['GET'])
@require_session
def activity_feed():
    """Unified activity feed — what Chalie did autonomously."""
    try:
        from services.interaction_log_service import InteractionLogService

        since_hours = request.args.get('since_hours', 24, type=int)
        limit = min(max(1, request.args.get('limit', 50, type=int)), 200)
        offset = max(0, request.args.get('offset', 0, type=int))

        # Clamp since_hours to reasonable range (1h to 7 days)
        since_hours = max(1, min(since_hours, 168))

        log_service = InteractionLogService()
        feed = log_service.get_activity_feed(
            since_hours=since_hours, limit=limit, offset=offset
        )
        feed['generated_at'] = datetime.now(timezone.utc).isoformat()
        return jsonify(feed), 200

    except Exception as e:
        logger.error(f"[REST API] activity feed error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve activity feed"}), 500


# ──────────────────────────────────────────────
# Provider health — model vs job tier matching
# ──────────────────────────────────────────────

_JOB_RECOMMENDED_TIER = {
    'autobiography': 'pro', 'frontal-cortex': 'pro', 'frontal-cortex-act': 'pro',
    'plan-decomposition': 'pro', 'frontal-cortex-respond': 'pro',
    'cognitive-drift': 'pro', 'episodic-memory': 'pro',
    'frontal-cortex-clarify': 'pro', 'frontal-cortex-proactive': 'pro',
    'mode-reflection': 'pro', 'semantic-memory': 'pro',
    'cognitive-triage': 'lite', 'experience-assimilation': 'lite',
    'fact-store': 'lite', 'autonomous-nurture': 'lite',
    'autonomous-ambient-tool': 'lite', 'autonomous-suggest': 'lite',
    'frontal-cortex-reflexive': 'lite', 'frontal-cortex-scheduled-tool': 'lite',
    'trait-extraction': 'lite', 'moment-enrichment': 'lite',
    'document-synthesis': 'lite', 'document-classification': 'lite',
    'document-ocr': 'lite',
}

_TIER_ORDER = {'frontier': 3, 'pro': 2, 'lite': 1, 'unknown': 0}


def _detect_model_tier(model: str) -> str:
    """Classify a model string into a capability tier."""
    m = model.lower()
    if 'flash-lite' in m or 'flash_lite' in m:
        return 'lite'
    if any(k in m for k in ('opus', 'sonnet', 'gpt-4o', 'gpt-4.1', 'o3', 'o4',
                             'gemini-2.5-pro', 'gemini-3-pro', 'gemini-3.1-pro')):
        return 'frontier'
    if any(k in m for k in ('haiku', 'gpt-4o-mini', 'gemini-2.5-flash', 'gemini-3-flash',
                             'gemini-3.1-flash', 'deepseek',
                             ':32b', ':30b', ':14b', ':70b')):
        return 'pro'
    if any(k in m for k in (':8b', ':4b', ':7b', ':3b', ':1b', 'phi', 'mistral')):
        return 'lite'
    return 'unknown'


@system_bp.route('/system/observability/provider-health', methods=['GET'])
@require_session
def observability_provider_health():
    """Job health: does the assigned model meet the job's recommended tier?"""
    try:
        from services.database_service import get_shared_db_service
        from services.provider_db_service import ProviderDbService

        db = get_shared_db_service()
        svc = ProviderDbService(db)
        job_assignments = {a['job_name']: a['provider_id'] for a in svc.get_all_job_assignments()}
        providers_by_id = {p['id']: p for p in svc.list_providers_summary()}

        jobs = []
        for job_id, rec_tier in _JOB_RECOMMENDED_TIER.items():
            pid = job_assignments.get(job_id)
            if not pid or pid not in providers_by_id:
                jobs.append({
                    'job_id': job_id, 'provider_id': None, 'provider_name': None,
                    'model': None, 'model_tier': None,
                    'recommended_tier': rec_tier, 'health': 'red',
                    'tooltip': 'No provider assigned',
                })
                continue

            p = providers_by_id[pid]
            model = p.get('model', '')
            model_tier = _detect_model_tier(model)
            if _TIER_ORDER.get(model_tier, 0) >= _TIER_ORDER.get(rec_tier, 0):
                health = 'green'
                tooltip = 'Model meets recommendation'
            else:
                health = 'yellow'
                tooltip = f'Recommended: {rec_tier}+ \u00b7 Assigned: {model_tier} ({model})'

            jobs.append({
                'job_id': job_id, 'provider_id': pid,
                'provider_name': p.get('name'), 'model': model,
                'model_tier': model_tier, 'recommended_tier': rec_tier,
                'health': health, 'tooltip': tooltip,
            })

        return jsonify({'generated_at': _now_iso(), 'jobs': jobs}), 200
    except Exception as e:
        logger.error(f"[REST API] observability/provider-health error: {e}")
        return jsonify({"error": "Failed to retrieve provider health"}), 500


# ──────────────────────────────────────────────
# In-place update endpoints
# ──────────────────────────────────────────────

@system_bp.route('/system/update/check', methods=['GET'])
@require_session
def update_check():
    """Check GitHub for a newer Chalie release."""
    try:
        from services.app_update_service import AppUpdateService
        info = AppUpdateService().check_for_update()
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"[REST API] update/check error: {e}")
        return jsonify({"error": "Failed to check for updates"}), 500


@system_bp.route('/system/update/apply', methods=['POST'])
@require_session
def update_apply():
    """Apply an in-place update (installed mode only)."""
    try:
        from services.app_update_service import AppUpdateService
        data = request.get_json(silent=True) or {}
        tag = data.get('tag')
        if not tag:
            return jsonify({"ok": False, "message": "Missing 'tag' parameter"}), 400

        svc = AppUpdateService()
        result = svc.apply_update(tag)

        if result.get('ok'):
            svc.request_restart()

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"[REST API] update/apply error: {e}")
        return jsonify({"ok": False, "message": f"Update failed: {e}"}), 500


# ──────────────────────────────────────────────
# Settings endpoints
# ──────────────────────────────────────────────

@system_bp.route('/system/settings/<key>', methods=['GET'])
@require_session
def get_setting(key):
    """Get a single setting value."""
    from services.settings_service import SettingsService
    from services.database_service import get_shared_db_service
    try:
        svc = SettingsService(get_shared_db_service())
        value = svc.get(key)
        return jsonify({"key": key, "value": value})
    except Exception as e:
        logger.error(f"[REST API] get setting error: {e}")
        return jsonify({"error": "Failed to get setting"}), 500


@system_bp.route('/system/settings/<key>', methods=['PUT'])
@require_session
def set_setting(key):
    """Set a single setting value."""
    from services.settings_service import SettingsService
    from services.database_service import get_shared_db_service
    data = request.get_json(silent=True) or {}
    value = data.get('value', '')
    try:
        svc = SettingsService(get_shared_db_service())
        if not value:
            svc.delete(key)
        else:
            svc.set(key, str(value))
        return jsonify({"key": key, "value": value or None})
    except Exception as e:
        logger.error(f"[REST API] set setting error: {e}")
        return jsonify({"error": "Failed to save setting"}), 500
