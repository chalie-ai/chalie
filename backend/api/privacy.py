"""
Privacy blueprint — /privacy/data-summary, /privacy/export, /privacy/delete-all.
"""

import logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from .auth import require_session

logger = logging.getLogger(__name__)

privacy_bp = Blueprint('privacy', __name__)


def _serialize_row(row: dict) -> dict:
    """Convert a database row dict to JSON-serializable form."""
    import uuid
    from decimal import Decimal
    result = {}
    for k, v in row.items():
        if v is None:
            result[k] = None
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, uuid.UUID):
            result[k] = str(v)
        elif isinstance(v, Decimal):
            result[k] = float(v)
        elif isinstance(v, (bytes, bytearray, memoryview)):
            result[k] = None  # Skip binary/encrypted columns (embeddings, encrypted keys)
        elif isinstance(v, (dict, list)):
            result[k] = v  # JSONB columns are already dicts/lists
        else:
            result[k] = v
    return result


@privacy_bp.route('/privacy/data-summary', methods=['GET'])
@require_session
def data_summary():
    """Overview of all stored data — counts by type."""
    try:
        from services.database_service import get_shared_db_service
        from services.redis_client import RedisClientService

        db = get_shared_db_service()
        redis = RedisClientService.create_connection()

        result = {}

        # PostgreSQL table counts — all user-data tables
        with db.connection() as conn:
            for table in [
                "episodes", "semantic_concepts", "user_traits", "threads",
                "autobiography", "moments", "scheduled_items", "persistent_tasks",
                "lists", "list_items", "identity_vectors", "place_fingerprints",
                "cognitive_reflexes", "interaction_log", "cortex_iterations",
                "curiosity_threads",
            ]:
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row = cursor.fetchone()
                    result[table] = row[0] if row else 0
                except Exception:
                    result[table] = 0

            # Oldest and newest memory timestamps
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM episodes")
                row = cursor.fetchone()
                if row and row[0]:
                    result["oldest_memory"] = str(row[0].date()) if hasattr(row[0], 'date') else str(row[0])
                    result["newest_memory"] = str(row[1].date()) if hasattr(row[1], 'date') else str(row[1])
            except Exception:
                pass

        # Redis fact count
        try:
            result["facts"] = len(redis.keys("fact_index:*"))
        except Exception:
            result["facts"] = 0

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] privacy/data-summary error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve data summary"}), 500


@privacy_bp.route('/privacy/export', methods=['GET'])
@require_session
def export_data():
    """Export all user data as JSON."""
    try:
        from services.database_service import get_shared_db_service
        from services.redis_client import RedisClientService

        db = get_shared_db_service()
        redis_conn = RedisClientService.create_connection()
        export = {"exported_at": datetime.now(timezone.utc).isoformat(), "tables": {}, "redis": {}}

        # ── PostgreSQL tables ──
        user_data_tables = [
            "episodes", "semantic_concepts", "semantic_relationships",
            "user_traits", "threads", "autobiography", "moments",
            "scheduled_items", "persistent_tasks", "lists", "list_items",
            "list_events", "identity_vectors", "identity_events",
            "place_fingerprints", "cognitive_reflexes", "curiosity_threads",
            "interaction_log", "cortex_iterations", "routing_decisions",
            "procedural_memory", "topics", "user_tool_preferences",
        ]

        with db.connection() as conn:
            for table in user_data_tables:
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT * FROM {table}")
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    export["tables"][table] = {
                        "count": len(rows),
                        "columns": columns,
                        "rows": [_serialize_row(dict(zip(columns, row))) for row in rows],
                    }
                except Exception:
                    export["tables"][table] = {"count": 0, "error": "table not found or empty"}

        # ── Redis keys (user-data patterns only) ──
        redis_patterns = [
            "working_memory:*", "gist:*", "fact:*",
            "identity_state:*", "spark_state:*", "focus_session:*",
        ]
        for pattern in redis_patterns:
            keys = redis_conn.keys(pattern)
            if keys:
                section = {}
                for key in keys:
                    key_str = key if isinstance(key, str) else key.decode()
                    key_type = redis_conn.type(key)
                    key_type = key_type if isinstance(key_type, str) else key_type.decode()
                    try:
                        if key_type == "string":
                            val = redis_conn.get(key)
                            section[key_str] = val if isinstance(val, str) else val.decode() if val else None
                        elif key_type == "list":
                            section[key_str] = [v.decode() if isinstance(v, bytes) else v for v in redis_conn.lrange(key, 0, -1)]
                        elif key_type == "hash":
                            raw = redis_conn.hgetall(key)
                            section[key_str] = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items()}
                        elif key_type == "zset":
                            section[key_str] = [(v.decode() if isinstance(v, bytes) else v) for v in redis_conn.zrange(key, 0, -1)]
                        elif key_type == "set":
                            section[key_str] = [(v.decode() if isinstance(v, bytes) else v) for v in redis_conn.smembers(key)]
                    except Exception:
                        section[key_str] = "<unreadable>"
                export["redis"][pattern] = section

        response = jsonify(export)
        response.headers["Content-Disposition"] = "attachment; filename=chalie-export.json"
        return response, 200

    except Exception as e:
        logger.error(f"[REST API] privacy/export error: {e}", exc_info=True)
        return jsonify({"error": "Failed to export data"}), 500


@privacy_bp.route('/privacy/delete-all', methods=['DELETE'])
@require_session
def delete_all():
    """Nuclear option — clear all stored user data."""
    confirm = request.headers.get("X-Confirm-Delete", "")
    if confirm != "yes":
        return jsonify({"error": "Requires X-Confirm-Delete: yes header"}), 400

    try:
        from services.redis_client import RedisClientService
        from services.database_service import get_shared_db_service

        # Clear Redis — all user-data patterns
        redis = RedisClientService.create_connection()
        for pattern in [
            # Memory layer
            "working_memory:*", "gist:*", "gist_index:*",
            "fact:*", "fact_index:*", "world_state:*",

            # Threads
            "thread:*", "active_thread:*",
            "thread_conv:*", "thread_conv_index:*",

            # Identity & context
            "identity_state:*",
            "spark_state:*",
            "focus_session:*",
            "client_context:*",
            "ambient:*",

            # Autonomous action state
            "proactive:*",
            "spark_nurture:*",
            "spark_suggest:*",
            "reflection:*",
            "plan:*",

            # Cognitive systems
            "cognitive_drift_state",
            "cognitive_drift_concept_cooldowns",
            "cognitive_drift_activations",
            "drift:*",
            "experience_assimilation_state",
            "experience_assimilation_cooldowns",
            "tool_reflection:pending",
            "semantic_consolidation:*",
            "reflex:*",
            "adaptive_boundary:*",
            "adaptive_fork_*",
            "adaptive_growth_*",

            # Topic & routing
            "recent_topic:*", "recent_topic",
            "routing_reflection_last_batch",

            # Output & notifications
            "notifications:recent",
            "output:*", "output-queue",

            # Sessions (invalidate all — current session still valid for this request)
            "auth_session:*",

            # Tool state
            "tool_state:*",
            "tool_triage_summaries",

            # Metrics (contain user-derived data)
            "metrics:timing:*",
            "metrics:counter:*",

            # Misc
            "fok:*",
            "curiosity:*",
            "bg_llm:*",
        ]:
            keys = redis.keys(pattern)
            if keys:
                redis.delete(*keys)

        # Truncate PostgreSQL — all user-data tables
        # NOTE: lists CASCADE handles list_items and list_events via FK relationships
        # NOTE: interaction_log is truncated here; the audit entry below is written after
        db = get_shared_db_service()
        with db.connection() as conn:
            for table in [
                # User-personal data (critical)
                "episodes",
                "semantic_concepts",
                "semantic_relationships",
                "user_traits",
                "threads",
                "autobiography",
                "moments",
                "scheduled_items",
                "persistent_tasks",
                "lists",              # CASCADE will handle list_items, list_events
                "identity_vectors",
                "identity_events",
                "place_fingerprints",
                "cognitive_reflexes",

                # Behavioral/derived data
                "interaction_log",
                "cortex_iterations",
                "message_cycles",
                "routing_decisions",
                "procedural_memory",
                "topics",
                "semantic_schemas",
                "triage_calibration_events",
                "tool_performance_metrics",
                "user_tool_preferences",
                "curiosity_threads",
            ]:
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"TRUNCATE TABLE {table} CASCADE")
                except Exception:
                    pass
            conn.commit()

        # Audit trail — log the deletion event AFTER truncation so it persists
        try:
            from services.interaction_log_service import InteractionLogService
            log_service = InteractionLogService()
            log_service.log_event(
                event_type="privacy_delete_all",
                payload={"timestamp": datetime.now(timezone.utc).isoformat()}
            )
        except Exception:
            pass

        ts = datetime.now(timezone.utc).isoformat()
        return jsonify({"deleted": True, "timestamp": ts}), 200

    except Exception as e:
        logger.error(f"[REST API] privacy/delete-all error: {e}", exc_info=True)
        return jsonify({"error": "Failed to delete data"}), 500
