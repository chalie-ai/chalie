"""Dedicated runtime DB for the mcp_tools tool index.

The ``mcp_tools.sqlite`` file is separate from chalie.db (Rule 5 / §4.1
dedicated-file exception). This module is the single source of truth for
the table schema and for opening / ensuring the DB is ready for queries.

Both :class:`~models.mcp_tool.McpTool` (reads via the model base) and
:class:`~services.mcp_client_service.McpClientService` (writes via the
service) reach this module — never ``Database.conn`` directly, so the
schema is always guaranteed before any query runs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from services.database import Database
from services.file_mapper_service import FileMapperService

# ── Schema ───────────────────────────────────────────────────────────────────

_TOOLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS mcp_tools (
    id          INTEGER PRIMARY KEY,
    server_id   TEXT NOT NULL,
    tool_name   TEXT NOT NULL UNIQUE,
    summary     TEXT NOT NULL DEFAULT '',
    raw_schema  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_server
    ON mcp_tools(server_id);
"""

# ── Gateway ──────────────────────────────────────────────────────────────────


def get_tools_connection() -> sqlite3.Connection:
    """Open the mcp_tools DB and guarantee the schema exists.

    This is the single chokepoint for every open of mcp_tools.sqlite.
    Callers MUST NOT close the returned connection — the ``Database``
    gateway's per-thread registry owns the lifecycle (closing would
    poison that registry).

    The ``executescript`` is idempotent: ``CREATE TABLE IF NOT EXISTS``
    and ``CREATE INDEX IF NOT EXISTS`` are no-ops once the schema is
    in place, so calling this on every access is cheap and safe.
    """
    db_path: Path = FileMapperService.get_mcp_tools_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = Database.conn(str(db_path))
    conn.executescript(_TOOLS_SCHEMA)
    return conn
