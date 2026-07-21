"""Privacy export action — GET /api/privacy/export.

Streams a complete user-data export as a downloadable JSON attachment. The
success body is a raw streaming JSON attachment (not the JSON envelope).

The streaming helpers serialize arbitrary DB column values inside the raw
body (not DTO fields), independent of the foundation DTO serializer.
"""

from __future__ import annotations

import collections.abc
import fnmatch
import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from flask import Response, stream_with_context
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from services.database import Database

if TYPE_CHECKING:
    from services.memory_store import MemoryStore

# Tables + MemoryStore patterns streamed by the full export (/export).
_EXPORT_TABLES = [
    "episodes",
    "transcript",
    "scheduled_items", "lists", "list_items",
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


class PrivacyExportAction(Action):
    """Export the entire instance as a streaming JSON attachment."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get"})
    response_dto = {"get": DocumentedResponse(not_found=False)}

    def slug(self) -> str:
        return "privacy"

    def verb(self) -> str:
        return "export"

    def get(self, id: int | str) -> ResponseReturnValue:
        """Stream a complete JSON export of all user-data tables + memory store as an attachment.

        Returns a raw JSON stream (not the JSON envelope). The ``id`` argument
        is ignored — the export covers the entire user dataset.
        """
        from services.memory_client import MemoryClientService
        from services.time_utils import utc_now

        def generate() -> "collections.abc.Generator[str, None, None]":
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
