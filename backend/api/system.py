"""
System blueprint — /health, /metrics, /system/status endpoints.
"""

import logging
from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

system_bp = Blueprint('system', __name__)


@system_bp.route('/health', methods=['GET', 'POST'])
def health_check():
    """Health check endpoint (no auth required). POST saves client context."""
    if request.method == 'POST':
        try:
            from services.client_context_service import ClientContextService
            data = request.get_json() or {}
            if data:
                ClientContextService().save(data)
        except Exception as e:
            logger.warning(f"[HEALTH] Failed to save client context: {e}")
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
                for table in ["episodes", "semantic_concepts", "user_traits"]:
                    try:
                        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
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


@system_bp.route('/system/voice-config', methods=['GET'])
@require_session
def get_voice_config():
    """Return TTS/STT server endpoints stored in settings."""
    try:
        from services.database_service import get_shared_db_service
        from services.settings_service import SettingsService

        db = get_shared_db_service()
        settings = SettingsService(db)

        return jsonify({
            'tts_endpoint': settings.get('tts_endpoint') or '',
            'stt_endpoint': settings.get('stt_endpoint') or '',
        }), 200
    except Exception as e:
        logger.error(f"[REST API] Failed to get voice config: {e}")
        return jsonify({"error": "Failed to retrieve voice config"}), 500


@system_bp.route('/system/voice-config', methods=['PUT'])
@require_session
def update_voice_config():
    """Update TTS/STT server endpoints."""
    try:
        from services.database_service import get_shared_db_service
        from services.settings_service import SettingsService

        db = get_shared_db_service()
        settings = SettingsService(db)
        data = request.get_json() or {}

        if 'tts_endpoint' in data:
            settings.set('tts_endpoint', data['tts_endpoint'], description='TTS server URL')
        if 'stt_endpoint' in data:
            settings.set('stt_endpoint', data['stt_endpoint'], description='STT server URL')

        return jsonify({'ok': True}), 200
    except Exception as e:
        logger.error(f"[REST API] Failed to update voice config: {e}")
        return jsonify({"error": "Failed to update voice config"}), 500
