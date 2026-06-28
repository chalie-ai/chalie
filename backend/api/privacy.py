"""
Privacy namespace — /privacy/data-summary, /privacy/export, /privacy/delete-all.

User data tables covered: episodes, transcript, tool_calls,
list_items, data_graph_edges, data_graph,
lists, scheduled_items, documents, watched_folders,
user_tool_preferences, memory_recall_log, llm_call_log,
concept_lut_misses.
"""

import collections.abc
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from flask import Response, request, stream_with_context
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.time_utils import utc_now

from .auth import require_session

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

privacy_ns = Namespace('privacy', description='Privacy data controls', path='/privacy')

# Ordered list of user-data tables for the nuclear delete operation.
# Children must appear before parents to satisfy FK constraints.
# System / auth / config tables are deliberately excluded.
_DELETE_ALL_TABLES = (
    # ── FK children first ─────────────────────────────────────────────────
    "tool_calls",          # FK → transcript(id)
    "list_items",          # FK → lists(id)
    "data_graph_edges",    # FK → data_graph(id) ON DELETE CASCADE
    # ── Parents / independents ────────────────────────────────────────────
    "transcript",
    "episodes",
    "data_graph",
    "moments",
    "lists",
    "scheduled_items",
    "documents",
    "watched_folders",
    "user_tool_preferences",
    "memory_recall_log",
    "llm_call_log",
    "concept_lut_misses",
)

# Companion index tables wiped alongside ``moments``. ``moments_fts`` is an
# external-content FTS5 table, so its postings are cleared with the FTS5
# ``'delete-all'`` command (a plain DELETE would strand them and corrupt the
# next pin); ``moments_vec`` is a standalone vec0 table cleared by plain DELETE.
_DELETE_ALL_MOMENTS_FTS = "moments_fts"
_DELETE_ALL_MOMENTS_VEC = "moments_vec"

# MemoryStore key patterns that belong to the user and must be cleared.
_DELETE_ALL_STORE_PATTERNS = (
    "working_memory:*",
    "deliberation_score:*",
)


def _serialize_row(row: dict[str, object]) -> dict[str, object]:
    import uuid
    from decimal import Decimal
    result: dict[str, object] = {}
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


@privacy_ns.route('/data-summary')
class DataSummaryResource(Resource):
    @require_session
    @privacy_ns.response(200, "Data summary")
    @privacy_ns.response(500, "Failed to retrieve data summary")
    def get(self) -> ResponseReturnValue:
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
                    "lists", "list_items",
                    "documents",
                    "data_graph",
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

            return result, 200

        except Exception as e:
            logger.exception(f"[REST API] privacy/data-summary error: {e}")
            return {"error": "Failed to retrieve data summary"}, 500


@privacy_ns.route('/export')
class ExportDataResource(Resource):
    @require_session
    @privacy_ns.response(200, "Data export (streaming JSON)")
    def get(self) -> ResponseReturnValue:

        user_data_tables = [
            "episodes",
            "transcript",
            "scheduled_items", "lists", "list_items",
            "user_tool_preferences",
            "documents", "watched_folders",
            "data_graph",
        ]

        store_patterns = [
            "working_memory:*",
        ]

        MAX_EXPORT_ROWS = 10000
        FETCH_BATCH = 500  # Rows fetched per iteration — keeps memory bounded

        def generate() -> "collections.abc.Generator[str, None, None]":
            from services.database_service import get_shared_db_service
            from services.memory_client import MemoryClientService

            db = get_shared_db_service()
            store = MemoryClientService.create_connection()

            from services.time_utils import utc_now
            exported_at = utc_now().isoformat()
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
                by_pattern: dict[str, dict[str, object]] = {p: {} for p in store_patterns}
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


@privacy_ns.route('/delete-all')
class DeleteAllResource(Resource):
    @require_session
    @privacy_ns.response(200, "Data deleted")
    @privacy_ns.response(400, "Confirmation header required")
    @privacy_ns.response(500, "Failed to delete data")
    def delete(self) -> ResponseReturnValue:
        confirm = request.headers.get("X-Confirm-Delete", "")
        if confirm != "yes":
            return {"error": "Requires X-Confirm-Delete: yes header"}, 400

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
                # The moments base table is cleared above; its external-content FTS
                # index needs the FTS5 'delete-all' command (a plain DELETE strands
                # postings), and its vec0 shadow is cleared by rowid-free DELETE.
                try:
                    cursor.execute(
                        f"INSERT INTO {_DELETE_ALL_MOMENTS_FTS}({_DELETE_ALL_MOMENTS_FTS}) "
                        "VALUES ('delete-all')"
                    )
                    cursor.execute(f"DELETE FROM {_DELETE_ALL_MOMENTS_VEC}")
                except Exception as e:
                    logger.warning(f"[REST API] Failed to clear moments indexes: {e}")
                    truncate_failures.append(_DELETE_ALL_MOMENTS_FTS)
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
            return result, 200

        except Exception as e:
            logger.exception(f"[REST API] privacy/delete-all error: {e}")
            return {"error": "Failed to delete data"}, 500