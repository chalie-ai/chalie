"""Privacy namespace — /api/privacy/data-summary, /api/privacy/export, /api/privacy/delete-all.

Not CRUD: a summary read (raw passthrough), a streaming JSON export download
(raw Flask Response), and a single irreversible nuclear delete whose result is
the :class:`DeleteAllResult` DTO. The row counts, export stream, and ``DELETE
FROM`` loop are inline SQL against the shared DB plus MemoryStore — kept verbatim
(they are real correctness logic, not serialization). The ``X-Confirm-Delete``
header gates the irreversible delete.

User data tables covered: episodes, transcript, tool_calls, list_items,
data_graph_edges, data_graph, lists, scheduled_items, documents, watched_folders,
user_tool_preferences, memory_recall_log, llm_call_log, concept_lut_misses.
"""

import collections.abc
import fnmatch
import json
import logging
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from flask import Response, stream_with_context
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.database import Database
from services.time_utils import utc_now
from .auth import require_session
from .dto import Error, register_dto, responds
from .dto.boundary import error
from .dto.privacy import DeleteAllResult

if TYPE_CHECKING:
    from services.memory_store import MemoryStore

logger = logging.getLogger(__name__)

_ERR_SUMMARY = "Failed to retrieve data summary"
_ERR_DELETE = "Failed to delete data"

privacy_ns = Namespace("privacy", description="Privacy data controls", path="/api/privacy")

register_dto(privacy_ns, DeleteAllResult, Error)

_P = privacy_ns.models

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
    "lists",
    "scheduled_items",
    "documents",
    "watched_folders",
    "user_tool_preferences",
    "memory_recall_log",
    "llm_call_log",
    "concept_lut_misses",
)

# MemoryStore key patterns that belong to the user and must be cleared.
_DELETE_ALL_STORE_PATTERNS = (
    "working_memory:*",
    "deliberation_score:*",
)

# Tables + MemoryStore patterns streamed by the full export (/export).
_EXPORT_TABLES = [
    "episodes",
    "transcript",
    "scheduled_items", "lists", "list_items",
    "user_tool_preferences",
    "documents", "watched_folders",
    "data_graph",
]
_EXPORT_STORE_PATTERNS = ["working_memory:*"]
_EXPORT_MAX_ROWS = 10000
_EXPORT_FETCH_BATCH = 500  # Rows fetched per iteration — keeps memory bounded


def _serialize_row(row: dict[str, object]) -> dict[str, object]:
    """Serialize an arbitrary DB row for the raw export stream.

    This serializes arbitrary DB column values inside the raw streaming body (not
    a DTO field), so it stays here — UUIDs/Decimal/bytes/datetime are normalized
    for JSON, independent of the foundation DTO serializer.
    """
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


def _stream_row_batches(
    cursor: sqlite3.Cursor, columns: list[str], fetch_batch: int
) -> "collections.abc.Generator[str, None, int]":
    """Yield comma-joined serialized JSON rows in batches; returns the total rows yielded."""
    first_row = True
    exported = 0
    while True:
        batch = cursor.fetchmany(fetch_batch)
        if not batch:
            return exported
        for row in batch:
            if not first_row:
                yield ","
            first_row = False
            yield json.dumps(_serialize_row(dict(zip(columns, row))))
            exported += 1


def _stream_table_export(
    table: str, max_rows: int, fetch_batch: int
) -> "collections.abc.Generator[str, None, None]":
    """Yield one table's export body — ``{"count", "columns", "rows": [...]}`` — or an error stub."""
    try:
        conn = Database.conn()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total_count = cursor.fetchone()[0]
        cursor.execute(f"SELECT * FROM {table} LIMIT {max_rows}")
        columns = [desc[0] for desc in cursor.description]

        yield f'{{"count": {total_count}, "columns": {json.dumps(columns)}, "rows": ['
        exported = yield from _stream_row_batches(cursor, columns, fetch_batch)

        suffix = "]"
        if total_count > max_rows:
            suffix += f', "truncated": true, "exported_rows": {exported}'
        yield suffix + "}"
        cursor.close()
    except Exception:
        yield '{"count": 0, "error": "table not found or empty"}'


def _stream_tables_export(
    tables: list[str], max_rows: int, fetch_batch: int
) -> "collections.abc.Generator[str, None, None]":
    """Yield the comma-joined ``"table": {...}`` entries for every table (body of the ``tables`` object)."""
    first_table = True
    for table in tables:
        if not first_table:
            yield ","
        first_table = False
        yield json.dumps(table) + ": "
        yield from _stream_table_export(table, max_rows, fetch_batch)


def _export_memory_store_by_pattern(
    store: "MemoryStore", patterns: list[str]
) -> dict[str, dict[str, object]]:
    """Group MemoryStore entries matching ``patterns`` by pattern, single-pass via ``export_matching``.

    export_matching() scans each keyspace once and applies all patterns simultaneously,
    returning {key: {type, value}} in O(n) time — avoiding an O(n×m) keys()+type() scan
    per pattern.
    """
    import re as _re
    all_entries = store.export_matching(patterns)
    compiled = [(p, _re.compile(fnmatch.translate(p))) for p in patterns]
    by_pattern: dict[str, dict[str, object]] = {p: {} for p in patterns}
    for key, entry in all_entries.items():
        for pattern, rx in compiled:
            if rx.fullmatch(key):
                by_pattern[pattern][key] = entry["value"]
                break
    return by_pattern


def _stream_memory_store_export(
    store: "MemoryStore", patterns: list[str]
) -> "collections.abc.Generator[str, None, None]":
    """Yield the ``memory_store`` value body, or ``{}`` on any failure."""
    try:
        yield json.dumps(_export_memory_store_by_pattern(store, patterns))
    except Exception:
        yield "{}"


@privacy_ns.route("/data-summary")
class DataSummaryResource(Resource):
    @require_session
    @privacy_ns.response(200, "Data summary (raw table-count map)")
    @privacy_ns.response(500, _ERR_SUMMARY, model=_P["Error"])
    @responds(code=200)
    def get(self) -> ResponseReturnValue:
        """Row counts per user-data table plus oldest/newest memory dates (raw passthrough)."""
        try:
            from services.memory_client import MemoryClientService

            MemoryClientService.create_connection()

            result: dict[str, object] = {}

            # SQLite table counts — all user-data tables
            conn = Database.conn()
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
                except Exception as exc:
                    logger.warning("[REST API] privacy/data-summary count failed for %s: %s", table, exc)
                    result[table] = 0

            # Oldest and newest memory timestamps
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM episodes")
                row = cursor.fetchone()
                if row and row[0]:
                    result["oldest_memory"] = str(row[0].date()) if hasattr(row[0], "date") else str(row[0])
                    result["newest_memory"] = str(row[1].date()) if hasattr(row[1], "date") else str(row[1])
            except Exception as exc:
                logger.warning("[REST API] privacy/data-summary timestamp query failed: %s", exc)

            return result
        except Exception as e:
            logger.exception(f"[REST API] privacy/data-summary error: {e}")
            return error(_ERR_SUMMARY, 500)


@privacy_ns.route("/export")
@privacy_ns.produces(["application/json"])
@privacy_ns.doc(description="Full data export as a streaming JSON attachment (not a marshalled body).")
class ExportDataResource(Resource):
    @require_session
    @privacy_ns.response(200, "Streaming JSON attachment")
    @privacy_ns.response(500, "Export failed", model=_P["Error"])
    @responds(code=200)
    def get(self) -> ResponseReturnValue:
        """Stream a complete JSON export of all user-data tables + memory store as an attachment."""

        def generate() -> "collections.abc.Generator[str, None, None]":
            from services.memory_client import MemoryClientService

            store = MemoryClientService.create_connection()

            exported_at = utc_now().isoformat()
            yield f'{{"exported_at": {json.dumps(exported_at)}, "tables": {{'
            yield from _stream_tables_export(_EXPORT_TABLES, _EXPORT_MAX_ROWS, _EXPORT_FETCH_BATCH)
            yield '}, "memory_store": '
            yield from _stream_memory_store_export(store, _EXPORT_STORE_PATTERNS)
            yield "}"

        response = Response(
            stream_with_context(generate()),
            mimetype="application/json",
        )
        response.headers["Content-Disposition"] = "attachment; filename=chalie-export.json"
        return response


@privacy_ns.route("/delete-all")
class DeleteAllResource(Resource):
    @require_session
    @privacy_ns.param("X-Confirm-Delete", "Must be literally 'yes' to confirm the irreversible wipe.", _in="header", required=True)
    @privacy_ns.response(200, "Data deleted", model=_P["DeleteAllResult"])
    @privacy_ns.response(422, "Missing/invalid X-Confirm-Delete header", model=_P["Error"])
    @privacy_ns.response(500, _ERR_DELETE, model=_P["Error"])
    @responds(DeleteAllResult, code=200)
    def delete(self) -> DeleteAllResult | ResponseReturnValue:
        """Irreversibly delete ALL user data — gated by the ``X-Confirm-Delete: yes`` header."""
        from flask import request
        if request.headers.get("X-Confirm-Delete", "") != "yes":
            return error("Requires X-Confirm-Delete: yes header", 422)

        try:
            from services.memory_client import MemoryClientService

            truncate_failures: list[str] = []

            with Database.transaction() as conn:
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

            return DeleteAllResult(
                deleted=True,
                timestamp=utc_now(),
                warnings=f"Failed to truncate: {', '.join(truncate_failures)}" if truncate_failures else None,
            )
        except Exception as e:
            logger.exception(f"[REST API] privacy/delete-all error: {e}")
            return error(_ERR_DELETE, 500)