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
    """Readiness probe — true only when SQLite, MemoryStore, embeddings, and ONNX are ready."""
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

    # Embedding model — lazy-loaded on first use. Ready once the ONNX session
    # and tokenizer are initialised.
    try:
        from services.embedding_service import _session
        if _session is not None:
            components['embeddings'] = {'status': 'ok'}
        else:
            components['embeddings'] = {'status': 'loading'}
    except Exception as e:
        logger.debug(f'[READY] embedding model not ready: {e}')
        components['embeddings'] = {'status': 'error', 'message': str(e)}

    # ONNX models — preloaded in background thread on boot. Not ready until
    # ensure_models() + warmup inference have completed.
    try:
        from services.onnx_inference_service import get_onnx_inference_service
        onnx_svc = get_onnx_inference_service()
        if onnx_svc.ready:
            components['onnx'] = {'status': 'ok'}
        else:
            components['onnx'] = {'status': 'loading'}
    except Exception as e:
        logger.debug(f'[READY] ONNX not ready: {e}')
        components['onnx'] = {'status': 'error', 'message': str(e)}

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
                for table_label, query in [
                    ("episodes", "SELECT COUNT(*) FROM episodes"),
                    ("concepts", "SELECT COUNT(*) FROM data_graph WHERE kind = 'user_specific' AND deleted_at IS NULL AND active=1"),
                    ("traits", "SELECT COUNT(*) FROM data_graph WHERE kind = 'user_specific' AND deleted_at IS NULL AND active=1"),
                ]:
                    try:
                        cursor.execute(query)
                        row = cursor.fetchone()
                        result["storage"][table_label] = row[0] if row else 0
                    except Exception as e:
                        logger.warning(f"[SYSTEM] Count query failed for '{table_label}': {e}")
                        result["storage"][table_label] = -1
        except Exception as e:
            result["status"] = "degraded"
            result["database_error"] = str(e)

        # Queue depths
        for queue_name in ["output-queue"]:
            try:
                result["queues"][queue_name] = store.llen(queue_name)
            except Exception as e:
                logger.warning(f"[SYSTEM] Queue depth check failed for '{queue_name}': {e}")
                result["queues"][queue_name] = -1

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] System status error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# Observability — cognitive legibility endpoints
# ─────────────────────────────────────────────

def _now_iso():
    return datetime.now(timezone.utc).isoformat()



@system_bp.route('/system/observability/memory', methods=['GET'])
@require_session
def observability_memory():
    """Memory layer counts and health indicators.

    Returns flat fields consumed by the brain admin frontend:
      episodes, concepts, traits, facts,
      avg_episode_activation, avg_trait_strength,
      working_memory, queues
    """
    try:
        from services.database_service import get_shared_db_service
        from services.memory_client import MemoryClientService

        db = get_shared_db_service()

        # ── Long-term counts from SQLite ──
        episode_row = db.fetch_all(
            "SELECT COUNT(*) AS cnt, COALESCE(AVG(retrieval_weight), 0) AS avg_act "
            "FROM episodes WHERE deleted_at IS NULL"
        )
        episodes = episode_row[0]['cnt'] if episode_row else 0
        avg_episode_activation = episode_row[0]['avg_act'] if episode_row else 0.0

        concept_row = db.fetch_all(
            "SELECT COUNT(*) AS cnt FROM data_graph "
            "WHERE kind = 'user_specific' AND deleted_at IS NULL AND active=1"
        )
        concepts = concept_row[0]['cnt'] if concept_row else 0

        trait_row = db.fetch_all(
            "SELECT COUNT(*) AS cnt, COALESCE(AVG(retrieval_weight), 0) AS avg_conf "
            "FROM data_graph "
            "WHERE kind = 'user_specific' AND deleted_at IS NULL AND active=1"
        )
        traits = trait_row[0]['cnt'] if trait_row else 0
        avg_trait_strength = trait_row[0]['avg_conf'] if trait_row else 0.0

        # ── Short-term counts from MemoryStore ──
        working_memory_turns = 0
        facts = 0
        queues = {}

        try:
            store = MemoryClientService.create_connection()

            # Count total buffered turns across all active working-memory keys.
            # Each key is a list of JSON turn entries; sum their lengths.
            wm_keys = store.keys("working_memory:*")
            for key in wm_keys:
                working_memory_turns += store.llen(key)

            # "Facts" = short-term atomic assertions stored in MemoryStore
            # under the "facts:*" namespace (if any), otherwise fall back to
            # user_traits count which already covers persistent facts.
            fact_keys = store.keys("facts:*")
            facts = len(fact_keys) if fact_keys else traits

            for q in ["output-queue"]:
                depth = store.llen(q)
                if depth:
                    queues[q] = depth
        except Exception as e:
            logger.warning(f"[OBS] memory store error: {e}")

        result = {
            'generated_at': _now_iso(),
            'episodes': episodes,
            'concepts': concepts,
            'traits': traits,
            'facts': facts,
            'avg_episode_activation': round(avg_episode_activation, 4),
            'avg_trait_strength': round(avg_trait_strength, 4),
            'working_memory': working_memory_turns,
            'queues': queues,
        }

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
        from services.innate_skills.registry import ALL_SKILL_NAMES

        db = get_shared_db_service()

        # Build the set of currently valid names (registered tools + innate skills)
        valid_names: set | None = None
        try:
            from services.tool_registry_service import ToolRegistryService
            valid_names = set(ToolRegistryService().tools.keys()) | ALL_SKILL_NAMES
        except Exception as e:
            logger.warning(f"[OBS] Tool registry unavailable, showing all profiles: {e}")

        if valid_names is not None:
            placeholders = ','.join('?' * len(valid_names))
            rows = db.fetch_all(
                "SELECT tool_name, tool_type, short_summary, effort, "
                "reliability_score, cost_tier, avg_latency_ms, enrichment_count, updated_at "
                f"FROM tool_capability_profiles WHERE tool_name IN ({placeholders}) "
                "ORDER BY tool_name",
                list(valid_names),
            )
        else:
            rows = db.fetch_all(
                "SELECT tool_name, tool_type, short_summary, effort, "
                "reliability_score, cost_tier, avg_latency_ms, enrichment_count, updated_at "
                "FROM tool_capability_profiles ORDER BY tool_name"
            )

        # Index performance stats by tool name for merging
        perf_by_name = {}
        try:
            perf_stats = ToolPerformanceService().get_all_tool_stats(30)
            for s in (perf_stats or []):
                perf_by_name[s.get('tool_name', '')] = s
        except Exception as e:
            logger.warning(f"[OBS] Tool performance stats unavailable: {e}")

        tools = []
        for r in (rows or []):
            entry = {
                'tool_name': r['tool_name'],
                'tool_type': r.get('tool_type', 'tool'),
                'summary': f"{r.get('short_summary', '')} (effort: {r.get('effort') or 'moderate'})",
                'effort': r.get('effort') or 'moderate',
                'reliability_score': r.get('reliability_score', 1.0),
                'cost_tier': r.get('cost_tier', 'free'),
                'avg_latency_ms': r.get('avg_latency_ms', 0),
                'enrichment_count': r.get('enrichment_count', 0),
                'updated_at': r.get('updated_at'),
            }
            if r['tool_name'] in perf_by_name:
                entry.update(perf_by_name[r['tool_name']])
            tools.append(entry)

        return jsonify({
            'generated_at': _now_iso(),
            'tools': tools,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/tools error: {e}")
        return jsonify({"error": "Failed to retrieve tool data"}), 500



@system_bp.route('/system/observability/tasks', methods=['GET'])
@require_session
def observability_tasks():
    """Goal ecology stats."""
    try:
        result = {
            'generated_at': _now_iso(),
        }

        # Goal ecology stats
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            result['goal_ecology_stats'] = {
                'last_run': store.get('goal_ecology:last_run'),
                'cycles_total': int(store.get('goal_ecology:cycles_total') or 0),
                'goals_decayed_total': int(store.get('goal_ecology:goals_decayed_total') or 0),
                'goals_created_total': int(store.get('goal_ecology:goals_created_total') or 0),
                'proactive_attempts_total': int(store.get('goal_ecology:proactive_attempts_total') or 0),
                'unmatched_signals': len(store.keys('goal:unmatched:*')),
            }
        except Exception as e:
            logger.warning(f"[OBS] goal ecology stats error: {e}")

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"[REST API] observability/tasks error: {e}")
        return jsonify({"error": "Failed to retrieve task data"}), 500


@system_bp.route('/system/observability/traits', methods=['GET'])
@require_session
def observability_traits():
    """User traits grouped by kind."""
    try:
        from services.data_graph_service import get_data_graph_service

        rows = get_data_graph_service().fetch(
            kinds=['user_specific'],
            order_by='retrieval_weight DESC',
        )
        categories = {}
        for row in rows:
            cat = 'general'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                'key': row.get('key'),
                'value': row.get('value'),
                'confidence': round(float(row.get('retrieval_weight') or 0), 3),
                'reinforcement_count': row.get('evidence_count') or 0,
                'updated_at': row.get('last_confirmed_at'),
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
        from services.data_graph_service import get_data_graph_service
        dgs = get_data_graph_service()
        rows = dgs.fetch(kinds=['user_specific'])
        match = next((r for r in rows if r.get('key') == trait_key), None)
        if not match:
            return jsonify({'error': 'Trait not found'}), 404
        dgs.soft_delete_by_id(match['id'])
        return jsonify({'ok': True, 'deleted': trait_key}), 200
    except Exception as e:
        logger.error(f"[REST API] observability/traits DELETE error: {e}")
        return jsonify({"error": "Failed to delete trait"}), 500


@system_bp.route('/system/observability/world-state', methods=['GET'])
@require_session
def observability_world_state():
    """World state as seen by the ACT loop — same data the LLM receives."""
    try:
        from services.world_state_service import WorldStateService

        svc = WorldStateService()
        summary = svc.get_world_model_summary()
        formatted = svc.get_world_state(topic='', message_embedding=None)

        return jsonify({
            'generated_at': _now_iso(),
            'summary': summary,
            'formatted': formatted,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/world-state error: {e}")
        return jsonify({"error": "Failed to retrieve world state"}), 500


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


@system_bp.route('/system/observability/pipeline-health', methods=['GET'])
@require_session
def observability_pipeline_health():
    """Pipeline health checks — subset of self-model noteworthy items."""
    try:
        from services.self_model_service import SelfModelService
        snapshot = SelfModelService().get_snapshot()
        noteworthy = snapshot.get('noteworthy', [])
        pipeline_items = [n for n in noteworthy if n.get('severity', 0) >= 0.3]
        return jsonify({'ok': True, 'checks': pipeline_items}), 200
    except Exception as e:
        logger.error(f"[REST API] observability/pipeline-health error: {e}")
        return jsonify({'ok': False, 'error': 'Failed to retrieve pipeline health'}), 500



@system_bp.route('/system/observability/situation', methods=['GET'])
@require_session
def observability_situation():
    """Returns the full situation model state for debugging/observability."""
    try:
        from services.situation_model_service import get_situation_model_service
        svc = get_situation_model_service()
        state = svc.get_current()
        directive = svc.get_directive()
        return jsonify({
            'ok': True,
            'situation': state,
            'directive': directive,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/situation error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@system_bp.route('/system/observability/write-queue', methods=['GET'])
@require_session
def observability_write_queue():
    """Write queue runtime statistics.

    Returns a JSON snapshot of the :class:`~services.write_queue_service.WriteQueueService`
    singleton covering current backlog depth, completed writes, and error
    count since process start.

    Responses:
        200: JSON object with keys ``queue_size`` (int), ``processed`` (int),
             and ``errors`` (int).
        500: JSON error object if the write-queue service is unavailable.
    """
    try:
        from services.write_queue_service import get_write_queue
        stats = get_write_queue().get_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"[REST API] observability/write-queue error: {e}")
        return jsonify({"error": "Failed to retrieve write queue stats"}), 500


@system_bp.route('/system/observability/telemetry', methods=['GET'])
@require_session
def observability_telemetry():
    """Telemetry event summary across all tracked event types.

    Returns a per-event-type breakdown produced by the process-level
    :class:`~services.telemetry_service.TelemetryCollector` singleton.
    Each key in the response body corresponds to one of the seven canonical
    telemetry event types (e.g. ``memory_recall``, ``act_loop_complete``).
    All types are present even when no events have been recorded yet.

    The response is wrapped with a ``generated_at`` ISO 8601 timestamp so
    callers can detect a stale/cached response.

    Responses:
        200: JSON object structured as::

                {
                    "generated_at": "<ISO-8601>",
                    "memory_recall": {"count": 42, "recent": [...]},
                    "context_assembly": {"count": 7, "recent": [...]},
                    ...
                }

        500: JSON error object if the telemetry collector is unavailable.
    """
    try:
        from services.telemetry_service import get_telemetry_collector
        summary = get_telemetry_collector().get_summary()
        return jsonify({"generated_at": _now_iso(), **summary}), 200
    except Exception as e:
        logger.error(f"[REST API] observability/telemetry error: {e}")
        return jsonify({"error": "Failed to retrieve telemetry summary"}), 500


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
# Thread reset (test harness support)
# ──────────────────────────────────────────────

@system_bp.route('/system/reset-thread', methods=['POST'])
@require_session
def reset_thread():
    """Clear conversation context and advance the thread ID.

    Does two things:
    1. Deletes all transcript rows (and their linked tool_calls + compaction watermark)
       for the given processor channel (default 'user'). This empties the Previous
       Messages block so the next turn starts genuinely fresh — recall must come from
       the memory pipeline, not conversation context.
    2. Advances the active_channel MemoryStore key (web:default:N → N+1) so postTurn
       services (adaptive directives, phase, situation model) scope their writes to the
       new thread_id.

    In the persistent-context architecture, getPreviousMessages() reads from the
    transcript table filtered by the hardcoded CHANNEL value ('user'). The MemoryStore
    active_channel key is separate — it scopes postTurn service state, not transcript
    retrieval. Both must be updated for a correct reset.

    Body (optional):
        {"channel": "user"}  — processor CHANNEL to clear (default: "user")
    """
    try:
        from services.database_service import get_shared_db_service
        from services.memory_client import MemoryClientService

        data = request.get_json(silent=True) or {}
        channel = data.get('channel', 'user')

        # ── 1. Clear transcript context ────────────────────────────────────────
        db = get_shared_db_service()
        with db.connection() as conn:
            # Compaction watermark has FK → transcript.id; delete it first
            conn.execute('DELETE FROM compactions WHERE channel = ?', (channel,))

            ids = [r[0] for r in conn.execute(
                'SELECT id FROM transcript WHERE channel = ?', (channel,)
            ).fetchall()]

            cleared = 0
            if ids:
                placeholders = ','.join('?' * len(ids))
                conn.execute(
                    f'DELETE FROM tool_calls WHERE transcript_id IN ({placeholders})',
                    ids,
                )
                conn.execute('DELETE FROM transcript WHERE channel = ?', (channel,))
                cleared = len(ids)

            conn.commit()

        # ── 2. Advance MemoryStore thread_id ──────────────────────────────────
        store = MemoryClientService.create_connection()
        current = store.get('active_channel:default')
        if isinstance(current, bytes):
            current = current.decode()

        new_thread = None
        if current:
            parts = current.rsplit(':', 1)
            try:
                seq = int(parts[-1]) + 1
            except (ValueError, IndexError):
                seq = 1
            new_thread = f"{parts[0]}:{seq}" if len(parts) > 1 else f"web:default:{seq}"
            store.set('active_channel:default', new_thread, ex=604800)

        logger.info(
            f"[RESET-THREAD] channel='{channel}': cleared {cleared} transcript rows, "
            f"thread {current!r} → {new_thread!r}"
        )
        return jsonify({
            'ok': True,
            'channel': channel,
            'transcript_rows_cleared': cleared,
            'new_thread': new_thread,
        }), 200

    except Exception as e:
        logger.error(f"[REST API] reset-thread error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


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
