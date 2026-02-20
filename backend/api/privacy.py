"""
Privacy blueprint — /privacy/data-summary, /privacy/delete-all.
"""

import logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from .auth import require_session

logger = logging.getLogger(__name__)

privacy_bp = Blueprint('privacy', __name__)


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

        # PostgreSQL table counts
        with db.connection() as conn:
            for table in ["episodes", "semantic_concepts", "user_traits", "threads"]:
                try:
                    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                    result[table] = row[0] if row else 0
                except Exception:
                    result[table] = 0

            # Oldest and newest memory timestamps
            try:
                row = conn.execute("SELECT MIN(created_at), MAX(created_at) FROM episodes").fetchone()
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


@privacy_bp.route('/privacy/delete-all', methods=['DELETE'])
@require_session
def delete_all():
    """Nuclear option — clear all stored data."""
    confirm = request.headers.get("X-Confirm-Delete", "")
    if confirm != "yes":
        return jsonify({"error": "Requires X-Confirm-Delete: yes header"}), 400

    try:
        from services.redis_client import RedisClientService
        from services.database_service import get_shared_db_service

        # Clear Redis
        redis = RedisClientService.create_connection()
        for pattern in ["working_memory:*", "gist:*", "gist_index:*", "fact:*", "fact_index:*",
                        "world_state:*", "thread:*", "active_thread:*", "thread_conv:*", "thread_conv_index:*"]:
            keys = redis.keys(pattern)
            if keys:
                redis.delete(*keys)

        # Truncate PostgreSQL
        db = get_shared_db_service()
        with db.connection() as conn:
            for table in ["episodes", "semantic_concepts", "semantic_relationships", "user_traits", "threads"]:
                try:
                    conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                except Exception:
                    pass
            conn.commit()

        # Audit trail (non-deletable)
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
