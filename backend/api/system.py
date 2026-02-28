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
        from services.redis_client import RedisClientService
        from services.database_service import get_shared_db_service

        redis = RedisClientService.create_connection()
        result = {"status": "ok", "memory": {}, "storage": {}, "queues": {}}

        # Redis health
        try:
            redis.ping()
            # Count memory store keys
            result["memory"]["working_memory_keys"] = len(redis.keys("working_memory:*"))
            result["memory"]["gist_keys"] = len(redis.keys("gist_index:*"))
            result["memory"]["fact_keys"] = len(redis.keys("fact_index:*"))
        except Exception as e:
            result["status"] = "degraded"
            result["redis_error"] = str(e)

        # PostgreSQL counts
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
            result["postgres_error"] = str(e)

        # Queue depths
        for queue_name in ["prompt-queue", "output-queue", "memory-chunker-queue"]:
            try:
                result["queues"][queue_name] = redis.llen(queue_name)
            except Exception:
                result["queues"][queue_name] = -1

        # Last proactive drift run
        try:
            last_run = redis.get("cognitive_drift:last_run")
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
        from services.redis_client import RedisClientService
        from services.database_service import get_shared_db_service

        result = {
            'generated_at': _now_iso(),
            'episodes': 0,
            'concepts': 0,
            'traits': 0,
            'avg_episode_activation': 0.0,
            'avg_trait_strength': 0.0,
            'working_memory': 0,
            'gists': 0,
            'facts': 0,
            'queues': {},
        }

        # PostgreSQL counts + averages
        try:
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*), AVG(activation_score) FROM episodes")
                row = cursor.fetchone()
                if row:
                    result['episodes'] = row[0] or 0
                    result['avg_episode_activation'] = round(float(row[1] or 0), 3)

                cursor.execute("SELECT COUNT(*) FROM semantic_concepts")
                row = cursor.fetchone()
                if row:
                    result['concepts'] = row[0] or 0

                cursor.execute("SELECT COUNT(*), AVG(confidence) FROM user_traits")
                row = cursor.fetchone()
                if row:
                    result['traits'] = row[0] or 0
                    result['avg_trait_strength'] = round(float(row[1] or 0), 3)
        except Exception as e:
            logger.warning(f"[OBS] memory postgres error: {e}")

        # Redis counts
        try:
            redis = RedisClientService.create_connection()
            result['working_memory'] = len(redis.keys("working_memory:*"))
            result['gists'] = len(redis.keys("gist_index:*"))
            result['facts'] = len(redis.keys("fact_index:*"))

            # Queue depths (only include non-zero)
            for q in ["prompt-queue", "output-queue", "memory-chunker-queue"]:
                depth = redis.llen(q)
                if depth:
                    result['queues'][q] = depth
        except Exception as e:
            logger.warning(f"[OBS] memory redis error: {e}")

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"[REST API] observability/memory error: {e}")
        return jsonify({"error": "Failed to retrieve memory data"}), 500


@system_bp.route('/system/observability/tools', methods=['GET'])
@require_session
def observability_tools():
    """Tool performance stats across all tools."""
    try:
        from services.tool_performance_service import ToolPerformanceService

        svc = ToolPerformanceService()
        stats = svc.get_all_tool_stats(30)

        return jsonify({
            'generated_at': _now_iso(),
            'tools': stats,
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
    """Active persistent tasks, curiosity threads, and triage calibration."""
    try:
        result = {
            'generated_at': _now_iso(),
            'persistent_tasks': [],
            'curiosity_threads': [],
            'calibration': {},
        }

        # Persistent tasks
        try:
            from services.persistent_task_service import PersistentTaskService
            from services.database_service import get_shared_db_service
            svc = PersistentTaskService(get_shared_db_service())
            result['persistent_tasks'] = svc.get_active_tasks(1)
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

        # Triage calibration stats
        try:
            from services.triage_calibration_service import TriageCalibrationService
            svc = TriageCalibrationService()
            result['calibration'] = svc.get_calibration_stats()
        except Exception as e:
            logger.warning(f"[OBS] triage calibration error: {e}")

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
                "SELECT trait_key, trait_value, confidence, category, source, "
                "reinforcement_count, is_literal, updated_at "
                "FROM user_traits WHERE user_id = 'primary' "
                "ORDER BY category, confidence DESC"
            )
            rows = cursor.fetchall()

            for row in rows:
                cat = row[3] or 'general'
                if cat not in categories:
                    categories[cat] = []
                updated = row[7]
                categories[cat].append({
                    'key': row[0],
                    'value': row[1],
                    'confidence': round(float(row[2] or 0), 3),
                    'source': row[4],
                    'reinforcement_count': row[5] or 0,
                    'is_literal': bool(row[6]),
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

        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM user_traits WHERE user_id = 'primary' AND trait_key = %s",
                (trait_key,)
            )
            deleted = cursor.rowcount

        if deleted:
            return jsonify({'ok': True, 'deleted': trait_key}), 200
        else:
            return jsonify({'error': 'Trait not found'}), 404
    except Exception as e:
        logger.error(f"[REST API] observability/traits DELETE error: {e}")
        return jsonify({"error": "Failed to delete trait"}), 500


@system_bp.route('/system/observability/reflexes', methods=['GET'])
@require_session
def observability_reflexes():
    """Cognitive reflex cluster stats and activation rates."""
    try:
        from services.cognitive_reflex_service import CognitiveReflexService

        svc = CognitiveReflexService()
        stats = svc.get_stats()

        return jsonify({
            'generated_at': _now_iso(),
            **stats,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/reflexes error: {e}")
        return jsonify({"error": "Failed to retrieve reflex data"}), 500


@system_bp.route('/system/activity', methods=['GET'])
@require_session
def activity_feed():
    """Unified activity feed — what Chalie did autonomously."""
    try:
        from services.interaction_log_service import InteractionLogService

        since_hours = request.args.get('since_hours', 24, type=int)
        limit = min(request.args.get('limit', 50, type=int), 200)
        offset = request.args.get('offset', 0, type=int)

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
