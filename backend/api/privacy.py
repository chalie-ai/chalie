"""
Privacy blueprint — /privacy/data-summary, /privacy/export, /privacy/delete-all.
"""

import json
import logging
from datetime import datetime, timezone
from flask import Blueprint, Response, request, jsonify, stream_with_context

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
        from services.memory_client import MemoryClientService

        db = get_shared_db_service()
        store = MemoryClientService.create_connection()

        result = {}

        # SQLite table counts — all user-data tables
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

        # MemoryStore fact count
        try:
            result["facts"] = sum(1 for _ in store.scan_iter(match="fact_index:*", count=100))
        except Exception:
            result["facts"] = 0

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] privacy/data-summary error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve data summary"}), 500


@privacy_bp.route('/privacy/export', methods=['GET'])
@require_session
def export_data():
    """Export all user data as a streaming JSON download."""

    user_data_tables = [
        "episodes", "semantic_concepts", "semantic_relationships",
        "user_traits", "threads", "autobiography", "moments",
        "scheduled_items", "persistent_tasks", "lists", "list_items",
        "list_events", "identity_vectors", "identity_events",
        "place_fingerprints", "cognitive_reflexes", "curiosity_threads",
        "interaction_log", "cortex_iterations", "routing_decisions",
        "procedural_memory", "topics", "user_tool_preferences",
    ]

    store_patterns = [
        "working_memory:*", "gist:*", "fact:*",
        "identity_state:*", "spark_state:*", "focus_session:*",
    ]

    MAX_EXPORT_ROWS = 10000
    FETCH_BATCH = 500  # Rows fetched per iteration — keeps memory bounded

    def generate():
        from services.database_service import get_shared_db_service
        from services.memory_client import MemoryClientService

        db = get_shared_db_service()
        store = MemoryClientService.create_connection()

        exported_at = datetime.now(timezone.utc).isoformat()
        yield f'{{"exported_at": {json.dumps(exported_at)}, "tables": {{'

        first_table = True
        for table in user_data_tables:
            if not first_table:
                yield ','
            first_table = False

            yield json.dumps(table) + ': '
            try:
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    total_count = cursor.fetchone()[0]
                    cursor.execute(f"SELECT * FROM {table} LIMIT {MAX_EXPORT_ROWS}")
                    columns = [desc[0] for desc in cursor.description]

                    yield (
                        f'{{"count": {total_count}, '
                        f'"columns": {json.dumps(columns)}, '
                        f'"rows": ['
                    )

                    first_row = True
                    exported = 0
                    while True:
                        batch = cursor.fetchmany(FETCH_BATCH)
                        if not batch:
                            break
                        for row in batch:
                            if not first_row:
                                yield ','
                            first_row = False
                            yield json.dumps(_serialize_row(dict(zip(columns, row))))
                            exported += 1

                    suffix = ']'
                    if total_count > MAX_EXPORT_ROWS:
                        suffix += f', "truncated": true, "exported_rows": {exported}'
                    yield suffix + '}'
                    cursor.close()

            except Exception:
                yield '{"count": 0, "error": "table not found or empty"}'

        # ── MemoryStore keys ──
        yield '}, "memory_store": {'
        first_pattern = True
        for pattern in store_patterns:
            keys = store.keys(pattern)
            if not keys:
                continue
            if not first_pattern:
                yield ','
            first_pattern = False

            section = {}
            for key in keys:
                key_str = key if isinstance(key, str) else key.decode()
                key_type = store.type(key)
                key_type = key_type if isinstance(key_type, str) else key_type.decode()
                try:
                    if key_type == "string":
                        val = store.get(key)
                        section[key_str] = val if isinstance(val, str) else val.decode() if val else None
                    elif key_type == "list":
                        section[key_str] = [v.decode() if isinstance(v, bytes) else v for v in store.lrange(key, 0, -1)]
                    elif key_type == "hash":
                        raw = store.hgetall(key)
                        section[key_str] = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items()}
                    elif key_type == "zset":
                        section[key_str] = [(v.decode() if isinstance(v, bytes) else v) for v in store.zrange(key, 0, -1)]
                    elif key_type == "set":
                        section[key_str] = [(v.decode() if isinstance(v, bytes) else v) for v in store.smembers(key)]
                except Exception:
                    section[key_str] = "<unreadable>"

            yield json.dumps(pattern) + ': ' + json.dumps(section)

        yield '}}'

    response = Response(
        stream_with_context(generate()),
        mimetype='application/json',
    )
    response.headers["Content-Disposition"] = "attachment; filename=chalie-export.json"
    return response


@privacy_bp.route('/privacy/delete-all', methods=['DELETE'])
@require_session
def delete_all():
    """Nuclear option — clear all stored user data."""
    confirm = request.headers.get("X-Confirm-Delete", "")
    if confirm != "yes":
        return jsonify({"error": "Requires X-Confirm-Delete: yes header"}), 400

    try:
        from services.memory_client import MemoryClientService
        from services.database_service import get_shared_db_service

        # Clear MemoryStore — all user-data patterns
        store = MemoryClientService.create_connection()
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
            keys = store.keys(pattern)
            if keys:
                store.delete(*keys)

        # Truncate SQLite — all user-data tables
        # NOTE: lists CASCADE handles list_items and list_events via FK relationships
        # NOTE: interaction_log is truncated here; the audit entry below is written after
        db = get_shared_db_service()
        truncate_failures = []
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
                except Exception as e:
                    logger.warning(f"[REST API] Failed to truncate {table}: {e}")
                    truncate_failures.append(table)

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
        result = {"deleted": True, "timestamp": ts}
        if truncate_failures:
            result["warnings"] = f"Failed to truncate: {', '.join(truncate_failures)}"
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] privacy/delete-all error: {e}", exc_info=True)
        return jsonify({"error": "Failed to delete data"}), 500
