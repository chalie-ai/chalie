"""
Privacy blueprint — /privacy/data-summary, /privacy/export, /privacy/delete-all.

User data tables covered: episodes, transcript, tool_calls, compactions,
goal_evidence, list_items, list_events, data_graph_edges, data_graph, goals,
lists, scheduled_items, documents, watched_folders, place_fingerprints,
user_tool_preferences, tool_performance_metrics, memory_recall_log, llm_call_log,
concept_lut_misses, browser_snapshots, browser_credentials.
"""

import json
import logging
from datetime import datetime, timezone
from flask import Blueprint, Response, request, jsonify, stream_with_context

from services.time_utils import utc_now

from .auth import require_session

logger = logging.getLogger(__name__)

privacy_bp = Blueprint('privacy', __name__)

# Ordered list of user-data tables for the nuclear delete operation.
# Children must appear before parents to satisfy FK constraints.
# System / auth / config tables are deliberately excluded.
_DELETE_ALL_TABLES = (
    # ── FK children first ─────────────────────────────────────────────────
    "tool_calls",          # FK → transcript(id)
    "compactions",         # FK → transcript(id)
    "goal_evidence",       # FK → goals(id) ON DELETE CASCADE
    "list_items",          # FK → lists(id)
    "list_events",         # FK → lists(id)
    "data_graph_edges",    # FK → data_graph(id) ON DELETE CASCADE
    # ── Parents / independents ────────────────────────────────────────────
    "transcript",
    "episodes",
    "data_graph",
    "goals",
    "lists",
    "scheduled_items",
    "documents",
    "watched_folders",
    "place_fingerprints",
    "user_tool_preferences",
    "tool_performance_metrics",
    "memory_recall_log",
    "llm_call_log",
    "concept_lut_misses",
    "browser_snapshots",
    "browser_credentials",
)

# MemoryStore key patterns that belong to the user and must be cleared.
_DELETE_ALL_STORE_PATTERNS = (
    "working_memory:*",
    "mode_gate:*",
)


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
        MemoryClientService.create_connection()

        result = {}

        # SQLite table counts — all user-data tables
        with db.connection() as conn:
            for table in [
                "episodes", "transcript",
                "scheduled_items",
                "lists", "list_items", "place_fingerprints",
                "documents",
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

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] privacy/data-summary error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve data summary"}), 500


@privacy_bp.route('/privacy/export', methods=['GET'])
@require_session
def export_data():
    """Export all user data as a streaming JSON download."""

    user_data_tables = [
        "episodes",
        "transcript",
        "scheduled_items", "lists", "list_items",
        "list_events",
        "place_fingerprints",
        "user_tool_preferences",
        "documents", "watched_folders",
    ]

    store_patterns = [
        "working_memory:*",
    ]

    MAX_EXPORT_ROWS = 10000
    FETCH_BATCH = 500  # Rows fetched per iteration — keeps memory bounded

    def generate():
        """Stream all user data as a single JSON object to the HTTP response.

        Yields successive chunks of JSON text covering every SQLite table listed
        in ``user_data_tables`` and every MemoryStore key prefix listed in
        ``store_patterns``.  Rows are fetched in batches of ``FETCH_BATCH`` to
        keep memory usage bounded, and tables that exceed ``MAX_EXPORT_ROWS``
        rows are truncated with a ``"truncated": true`` marker in the output.

        Yields:
            str: Raw JSON text fragments that, when concatenated, form a valid
            JSON object with ``"exported_at"``, ``"tables"``, and
            ``"memory_store"`` top-level keys.
        """
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

        # ── MemoryStore keys — single-pass export via export_matching ──
        # export_matching() scans each keyspace once and applies all patterns
        # simultaneously, returning {key: {type, value}} in O(n) time.
        # This replaces the previous O(n×m×5) pattern of keys() + type() per key.
        yield '}, "memory_store": '
        try:
            import re as _re
            all_entries = store.export_matching(store_patterns)

            # Group results by pattern for the same JSON shape as before
            compiled = [(p, _re.compile(p.replace("*", ".*").replace("?", "."))) for p in store_patterns]
            by_pattern = {p: {} for p in store_patterns}
            for key, entry in all_entries.items():
                for pattern, rx in compiled:
                    if rx.fullmatch(key):
                        by_pattern[pattern][key] = entry["value"]
                        break

            yield json.dumps(by_pattern)
        except Exception:
            yield '{}'

        yield '}'

    response = Response(
        stream_with_context(generate()),
        mimetype='application/json',
    )
    response.headers["Content-Disposition"] = "attachment; filename=chalie-export.json"
    return response


@privacy_bp.route('/privacy/delete-all', methods=['DELETE'])
@require_session
def delete_all():
    """Nuclear option — clear all stored user data.

    Wipes every user-owned table (episodes, transcript, tool_calls,
    compactions, goal_evidence, list_items, list_events, data_graph_edges,
    data_graph, goals, lists, scheduled_items, documents,
    watched_folders, place_fingerprints, user_tool_preferences,
    tool_performance_metrics, memory_recall_log, llm_call_log,
    concept_lut_misses, browser_snapshots, browser_credentials)
    and clears MemoryStore working_memory keys.

    System / auth / config tables are deliberately excluded.
    """
    confirm = request.headers.get("X-Confirm-Delete", "")
    if confirm != "yes":
        return jsonify({"error": "Requires X-Confirm-Delete: yes header"}), 400

    try:
        from services.database_service import get_shared_db_service
        from services.memory_client import MemoryClientService

        db = get_shared_db_service()
        truncate_failures = []

        with db.connection() as conn:
            cursor = conn.cursor()
            for table in _DELETE_ALL_TABLES:
                try:
                    cursor.execute(f"DELETE FROM {table}")
                except Exception as e:
                    logger.warning(f"[REST API] Failed to delete from {table}: {e}")
                    truncate_failures.append(table)
            cursor.close()

        # Clear user-owned MemoryStore keys
        try:
            store = MemoryClientService.create_connection()
            for pattern in _DELETE_ALL_STORE_PATTERNS:
                matched = store.keys(pattern)
                if matched:
                    store.delete(*matched)
        except Exception as e:
            logger.warning(f"[REST API] Failed to clear memory store: {e}")

        result = {"deleted": True, "timestamp": utc_now().isoformat()}
        if truncate_failures:
            result["warnings"] = f"Failed to truncate: {', '.join(truncate_failures)}"
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] privacy/delete-all error: {e}", exc_info=True)
        return jsonify({"error": "Failed to delete data"}), 500
