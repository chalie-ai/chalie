"""
System blueprint — /health, /metrics, /system/status, /system/observability/* endpoints.
"""

import json
import logging
import os

from flask import Blueprint, jsonify, request

from .auth import require_session
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

system_bp = Blueprint('system', __name__)

# Signal source for telemetry signals absorbed by WorldState from health pings.
_SIGNAL_SOURCE_HEALTH = '/health'


def _ok_response():
    """Standard health response body — used by both GET and POST."""
    from consumer import APP_VERSION
    return jsonify({"status": "ok", "version": APP_VERSION}), 200


def _mirror_telemetry_to_world_state(svc, data: dict) -> None:
    """Mirror persisted client telemetry into WorldState as Signals."""
    from services.world_state import world_state, Signal
    world_state.set("telemetry", svc.get() or data)
    world_state.absorb(Signal(source=_SIGNAL_SOURCE_HEALTH, kind='heartbeat', payload=data))
    device_class = data.get('device_class') or (data.get('device') or {}).get('class')
    if device_class:
        world_state.absorb(Signal(source=_SIGNAL_SOURCE_HEALTH, kind='device', payload={'device_class': device_class}))
    local_time = data.get('local_time')
    if local_time:
        world_state.absorb(Signal(source=_SIGNAL_SOURCE_HEALTH, kind='local_time', payload={'local_time': local_time}))


def _persist_heartbeat(data: dict) -> None:
    """Persist client context + mirror to WorldState. Each step is independently logged on failure."""
    from services.client_context_service import ClientContextService
    svc = ClientContextService()
    svc.save(data)
    try:
        _mirror_telemetry_to_world_state(svc, data)
    except Exception as ws_err:
        logger.warning(f"[HEALTH] Failed to mirror telemetry to WorldState: {ws_err}")


@system_bp.route('/health', methods=['GET', 'POST'])
def health_check():
    """Health check endpoint (no auth required). POST saves client context."""
    if request.method != 'POST':
        return _ok_response()
    try:
        data = request.get_json() or {}
        if data:
            _persist_heartbeat(data)
    except Exception as e:
        logger.warning(f"[HEALTH] Failed to save client context: {e}")
    return _ok_response()


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
        elif onnx_svc.degraded:
            # Registration completed but one or more tasks failed (e.g. sha256
            # gate refused on encoder mismatch). Surface this loudly — the boot
            # gate exists precisely to catch these cases. Silent fallback would
            # let the system run with a wrong/incomplete classifier.
            failed = onnx_svc.failed_registrations
            components['onnx'] = {
                'status': 'degraded',
                'failed_tasks': [t for t, _ in failed],
                'message': '; '.join(f'{t}: {err}' for t, err in failed),
            }
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
        result = {"status": "ok", "memory": {}, "storage": {}}

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

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] System status error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# Observability — cognitive legibility endpoints
# ─────────────────────────────────────────────

def _now_iso():
    return utc_now().isoformat()


_RECORDS_LIMIT = 250
_VALID_SOURCES = {'episodes', 'user', 'system'}


@system_bp.route('/system/observability/records', methods=['GET'])
@require_session
def observability_records():
    """Paginated record browser for episodes, user, and system memory sources."""
    try:
        source = request.args.get('source', '')
        if source not in _VALID_SOURCES:
            return jsonify({"error": "invalid source"}), 400

        raw_offset = request.args.get('offset', '0')
        try:
            offset = int(raw_offset)
        except (ValueError, TypeError):
            return jsonify({"error": "invalid offset"}), 400
        if offset < 0:
            return jsonify({"error": "invalid offset"}), 400

        q = (request.args.get('q', '') or '')[:200]

        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        if source == 'episodes':
            rows = db.fetch_all(
                "SELECT created_at AS created, last_accessed_at AS last_accessed, "
                "gist AS value, location_name "
                "FROM episodes "
                "WHERE deleted_at IS NULL "
                "AND (? = '' OR gist LIKE ?) "
                "ORDER BY last_accessed_at IS NULL, last_accessed_at DESC, created_at DESC "
                "LIMIT ? OFFSET ?",
                (q, f"%{q}%", _RECORDS_LIMIT, offset),
            )
        else:
            kind = 'user_specific' if source == 'user' else 'system'
            rows = db.fetch_all(
                "SELECT first_seen_at AS created, last_accessed_at AS last_accessed, "
                "key, value "
                "FROM data_graph "
                "WHERE kind = ? AND active = 1 AND deleted_at IS NULL "
                "AND (? = '' OR key LIKE ? OR value LIKE ?) "
                "ORDER BY last_accessed_at IS NULL, last_accessed_at DESC, first_seen_at DESC "
                "LIMIT ? OFFSET ?",
                (kind, q, f"%{q}%", f"%{q}%", _RECORDS_LIMIT, offset),
            )

        rows = rows or []
        serialised = []
        for r in rows:
            row = {
                'created': r['created'],
                'last_accessed': r['last_accessed'],
                'value': r['value'],
            }
            if source == 'episodes':
                row['location'] = r.get('location_name') or ''
            else:
                row['key'] = r['key']
            serialised.append(row)
        return jsonify({
            'generated_at': _now_iso(),
            'source': source,
            'rows': serialised,
            'offset': offset,
            'limit': _RECORDS_LIMIT,
            'returned': len(rows),
            'has_more': len(rows) == _RECORDS_LIMIT,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] observability/records error: {e}")
        return jsonify({"error": "Failed to retrieve records"}), 500


@system_bp.route('/system/observability/tools', methods=['GET'])
@require_session
def observability_tools():
    """Tool usage counts from tool_calls — count + last_used per tool."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        rows = db.fetch_all(
            "SELECT tool_name, COUNT(*) AS count, MAX(created_at) AS last_used_at "
            "FROM tool_calls "
            "WHERE tool_name NOT IN ('compaction', 'tool_compaction', 'trail_compaction', "
            "'chat_history_compactor', 'tool_chain_compactor', 'thinking') "
            "GROUP BY tool_name "
            "ORDER BY last_used_at DESC"
        )
        tools = [
            {
                'tool_name': r['tool_name'],
                'count': r['count'],
                'last_used_at': r['last_used_at'],
            }
            for r in (rows or [])
        ]
        return jsonify({'generated_at': _now_iso(), 'tools': tools}), 200
    except Exception as e:
        logger.error(f"[REST API] observability/tools error: {e}")
        return jsonify({"error": "Failed to retrieve tool data"}), 500



@system_bp.route('/system/observability/token-usage', methods=['GET'])
@require_session
def observability_token_usage():
    """Token usage aggregated by time window, model, provider, and usage class.

    Query params:
        window: hour | day | week | month | lifetime (default: day)
        usage_class: chat | subagent | subconscious (optional filter)
    """
    from services.llm_call_log_service import get_token_usage, VALID_WINDOWS
    window = request.args.get('window', 'day')
    if window not in VALID_WINDOWS:
        return jsonify({'error': f"Invalid window '{window}'. Use: {', '.join(sorted(VALID_WINDOWS))}"}), 400
    usage_class = request.args.get('usage_class') or None
    try:
        data = get_token_usage(window=window, usage_class=usage_class)
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"[REST API] observability/token-usage error: {e}")
        return jsonify({"error": "Failed to retrieve token usage data"}), 500



@system_bp.route('/system/observability/world-state', methods=['GET'])
@require_session
def observability_world_state():
    """World state as seen by the ACT loop — rendered block + raw inputs."""
    from services.world_state import world_state, _fetch_schedule_rows
    from services.heartbeat_service import heartbeat_service

    return jsonify({
        "rendered": world_state.render(),
        "inputs": {
            "telemetry": heartbeat_service.read(),
            "signals": world_state.get("signals"),
            "schedule": _fetch_schedule_rows(),
        },
    }), 200


@system_bp.route('/system/observability/compaction', methods=['GET'])
@require_session
def observability_compaction():
    """Continuity-compaction synthesis for the chat ('user') channel.

    Returns the durable summary the ACT loop carries forward after a
    context-overflow compaction — the same text prepended to the
    UserMessageProcessor prompt in place of the older turns. Read-only.
    Returns ``{"compaction": null}`` when no compaction has run yet.
    """
    try:
        from services import compaction_persistence, locale_service
        record = compaction_persistence.get_compaction('user')
        if not record:
            return jsonify({'compaction': None}), 200
        return jsonify({
            'compaction': {
                'summary': record['compacted_text'],
                'compacted_up_to_id': record['compacted_up_to_id'],
                'compacted_at': locale_service.format_date(record['created_at'], for_ui=True),
            },
        }), 200
    except Exception:
        logger.exception("[REST API] observability/compaction error")
        return jsonify({"error": "Failed to retrieve compaction summary"}), 500


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


_LOG_FILE_PATH = "/tmp/chalie.log"  # Read-only; written exclusively by utils/logger.py in the same process
_LOG_TAIL_BYTES = 256 * 1024   # 256 KB tail read — never loads the full file
_ERROR_LEVELS = frozenset({"ERROR", "CRITICAL"})
_ERROR_CAP = 200


def _tail_error_lines() -> list[dict]:
    """Read the last ~256 KB of /tmp/chalie.log and return ERROR/CRITICAL entries newest-first.

    Written by utils.logger.Logger.start() — see backend/utils/logger.py FileHandler.
    Skips malformed JSON lines silently. Capped at _ERROR_CAP entries.
    """
    try:
        size = os.path.getsize(_LOG_FILE_PATH)
    except OSError:
        return []

    errors: list[dict] = []
    try:
        fh_cm = open(_LOG_FILE_PATH, "r", errors="replace")
    except OSError:
        return []

    with fh_cm as fh:
        if size > _LOG_TAIL_BYTES:
            # Absolute seek — text-mode files only allow 0-byte end-relative seeks.
            fh.seek(size - _LOG_TAIL_BYTES)
            fh.readline()   # discard partial first line
        for raw in fh:
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if entry.get("level") in _ERROR_LEVELS:
                errors.append({"timestamp": entry.get("timestamp", ""), "message": entry.get("message", "")})

    errors.reverse()
    return errors[:_ERROR_CAP]


@system_bp.route('/system/observability/errors', methods=['GET'])
@require_session
def observability_errors():
    """Recent ERROR and CRITICAL log lines from /tmp/chalie.log, newest first.

    Reads only the last ~256 KB of the log file to avoid loading unbounded content.
    Written by utils.logger.Logger.start() (backend/utils/logger.py).
    """
    try:
        return jsonify({"generated_at": _now_iso(), "errors": _tail_error_lines()}), 200
    except Exception as e:
        logger.error(f"[REST API] observability/errors error: {e}")
        return jsonify({"error": "Failed to retrieve error log"}), 500


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


# Settings endpoints
# ──────────────────────────────────────────────

@system_bp.route('/system/context-usage', methods=['GET'])
@require_session
def get_context_usage():
    """Last user-turn request size + context window for the composer indicator.

    ``last_request_tokens`` is the provider-reported ``tokens_input`` of the most
    recent main-conversation call (``job_name='user:user'`` — NOT every
    usage_class='chat' row, which would also include the thinking pre-pass and
    each web_search/web_browse delegate iteration, making the indicator
    oscillate); ``context_window`` is the selected provider's ``max_tokens``.
    Either is null when unknown — the endpoint never raises so a transient miss
    can't break the composer.
    """
    from services.llm_call_log_service import get_last_chat_request_tokens
    from services.provider_cache_service import ProviderCacheService
    last = get_last_chat_request_tokens()
    selected = ProviderCacheService.get_selected_provider() or {}
    return jsonify({
        "last_request_tokens": last,
        "context_window": selected.get('max_tokens'),
    })


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
