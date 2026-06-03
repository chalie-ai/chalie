"""
McpClientService — outbound MCP client connection manager.

Chalie connects OUT to remote MCP servers via the MCP streamable-HTTP
transport.  This service owns:
  - CRUD for mcp_client_servers rows (chalie.db)
  - per-server ping + tool-list sync (written to data/mcp_tools.sqlite)
  - dynamic policy-row management in policy_rules (ask by default)
  - dispatch of _mcp_<server>_<tool> calls
  - heartbeat loop orchestration

Design notes:
  - All MCP client calls are async (mcp library).  Chalie's ability system
    and workers are synchronous.  The bridge is asyncio.run() per call —
    safe because Chalie's worker threads never carry a running event loop.
  - mcp_tools.sqlite is a separate gitignored DB so build_ability_db
    rebuilds never destroy the dynamically-synced rows.
  - Tool names use the scheme: _mcp_<sanitized_server_name>_<remote_tool>.
    Server name is sanitized to [a-z0-9_] to make a valid Python identifier.
"""

import asyncio
import json
import logging
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from services.database_service import get_shared_db_service
from services.file_mapper_service import FileMapperService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[MCP CLIENT]"

# Status strings — these exact values are checked by the nightly scenario.
_STATUS_UNKNOWN = "unknown"
_STATUS_ONLINE = "online"
_STATUS_OFFLINE = "offline"

# Name-sanitization pattern: keep lowercase alpha, digits, underscore.
_SANITIZE_RE = re.compile(r"[^a-z0-9_]")


def _sanitize_name(name: str) -> str:
    """Convert a server name to a safe [a-z0-9_] identifier fragment."""
    lowered = name.lower().strip()
    # Replace runs of non-identifier characters with a single underscore.
    sanitized = _SANITIZE_RE.sub("_", lowered)
    # Strip leading/trailing underscores produced by the replacement.
    return sanitized.strip("_") or "server"


def _tool_name(server_name: str, remote_tool: str) -> str:
    """Build the prefixed tool name for a remote MCP tool.

    Format: _mcp_<sanitized_server_name>_<remote_tool_name>
    Example: server='taskie', tool='create_document' → '_mcp_taskie_create_document'
    """
    return f"_mcp_{_sanitize_name(server_name)}_{remote_tool}"


def _normalize_host(host: str) -> str:
    """Canonical dedup key for a remote MCP endpoint.

    One connection per endpoint: lowercases scheme + hostname, drops default
    ports (443/https, 80/http), trims surrounding whitespace, and strips a
    single trailing slash from the path.  A bare host ("mcp.example.com/mcp")
    is parsed as if https.  The original ``host`` is still stored verbatim for
    display — this value is only the key used to detect re-adds of the same
    endpoint.
    """
    raw = (host or "").strip()
    parts = urlsplit(raw)
    if not parts.scheme and not parts.netloc:
        # Bare host like "mcp.example.com/mcp" — re-parse with a scheme so the
        # hostname/port split correctly instead of landing in ``path``.
        parts = urlsplit("https://" + raw)
    scheme = (parts.scheme or "https").lower()
    hostname = (parts.hostname or "").lower()
    port = parts.port
    is_default_port = (
        (scheme == "https" and port == 443)
        or (scheme == "http" and port == 80)
    )
    netloc = hostname if (port is None or is_default_port) else f"{hostname}:{port}"
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo = f"{userinfo}:{parts.password}"
        netloc = f"{userinfo}@{netloc}"
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parts.query, ""))


# ── mcp_tools.sqlite schema helpers ─────────────────────────────────────────

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

_TOOLS_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS mcp_tools_fts USING fts5(
    tool_name,
    summary,
    content='mcp_tools',
    content_rowid='id'
);
"""


def _open_tools_db() -> sqlite3.Connection:
    """Open (and initialize if necessary) the mcp_tools.sqlite runtime DB."""
    db_path: Path = FileMapperService.get_mcp_tools_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_TOOLS_SCHEMA)
    try:
        conn.executescript(_TOOLS_FTS_SCHEMA)
    except sqlite3.OperationalError:
        pass  # FTS5 may not be available in all environments
    conn.commit()
    return conn


# ── Async MCP helpers ────────────────────────────────────────────────────────

async def _async_list_tools(host: str, headers: dict) -> list[dict]:
    """Connect to a remote MCP server and return its tool list.

    Each returned dict has 'name', 'description', and 'inputSchema' keys
    matching the MCP protocol ToolDescription shape.
    """
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession

    async with streamablehttp_client(host, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": (
                        t.inputSchema if isinstance(t.inputSchema, dict) else {}
                    ),
                }
                for t in result.tools
            ]


async def _async_call_tool(
    host: str, headers: dict, remote_tool: str, params: dict
) -> Any:
    """Connect to a remote MCP server and call a single tool.

    Returns the raw tool result content (may be a list of content blocks or
    a plain value depending on the server).
    """
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession

    async with streamablehttp_client(host, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(remote_tool, params)
            return result.content


# ── McpClientService ──────────────────────────────────────────────────────────


class McpClientService:
    """CRUD, sync, and dispatch for outbound MCP client connections.

    All public methods are synchronous.  Async MCP calls are bridged via
    asyncio.run() — safe in Chalie's sync worker/ability context because those
    threads never carry a running event loop.
    """

    def __init__(self) -> None:
        self._db = get_shared_db_service()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def list_servers(self) -> list[dict]:
        """Return all mcp_client_servers rows as dicts."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id, name, host, headers, enabled, status, "
                "last_pinged_at, created_at, updated_at "
                "FROM mcp_client_servers ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def add_server(self, name: str, host: str, headers: dict, enabled: bool) -> dict:
        """Add a remote server, deduping by normalized host.

        If an existing server resolves to the same endpoint (see
        ``_normalize_host``), this is an idempotent upsert: the existing row's
        name/host/headers are updated and it is re-enabled — no duplicate is
        created.  Otherwise a new row is inserted.  Succeeds even if the host
        is unreachable (new rows start with status 'unknown').
        """
        existing = self._find_by_normalized_host(host)
        if existing is not None:
            return self._upsert_existing(existing, name, host, headers)

        server_id = str(uuid.uuid4())
        now = utc_now().isoformat()
        headers_json = json.dumps(headers or {})
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO mcp_client_servers "
                "(id, name, host, headers, enabled, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (server_id, name, host, headers_json,
                 1 if enabled else 0, _STATUS_UNKNOWN, now, now),
            )
            conn.commit()
        logger.info("%s Added server %r (id=%s)", _LOG_PREFIX, name, server_id)
        return self._get_server(server_id)

    def _find_by_normalized_host(self, host: str) -> dict | None:
        """Return an existing server resolving to the same endpoint, else None."""
        key = _normalize_host(host)
        for server in self.list_servers():
            if _normalize_host(server["host"]) == key:
                return server
        return None

    def _upsert_existing(self, existing: dict, name: str, host: str,
                         headers: dict) -> dict:
        """Re-add of a known endpoint: update fields and re-enable.

        The local tool prefix is derived from the server name, so when the name
        changes we purge the old ``_mcp_<oldname>_*`` tool + policy rows first —
        otherwise the post-add re-sync would leave orphaned entries under the
        stale prefix.
        """
        old_prefix = f"_mcp_{_sanitize_name(existing['name'])}_"
        new_prefix = f"_mcp_{_sanitize_name(name)}_"
        if old_prefix != new_prefix:
            self._delete_tools_for_server(existing["id"])
            self._delete_policy_rows(old_prefix)
        logger.info(
            "%s Re-add of existing endpoint %r → upsert id=%s",
            _LOG_PREFIX, host, existing["id"],
        )
        return self.update_server(existing["id"], {
            "name": name,
            "host": host,
            "headers": headers or {},
            "enabled": True,
        })

    def get_server(self, server_id: str) -> dict | None:
        """Return a single server dict, or None if not found."""
        try:
            return self._get_server(server_id)
        except LookupError:
            return None

    def update_server(self, server_id: str, updates: dict) -> dict:
        """Apply a partial update to a server row.

        Accepts: name, host, headers (dict or JSON string), enabled (bool|int).
        Returns the updated server dict.  Raises LookupError if not found.
        """
        allowed = {"name", "host", "headers", "enabled"}
        cols, vals = [], []
        for key, val in updates.items():
            if key not in allowed:
                continue
            if key == "headers" and isinstance(val, dict):
                val = json.dumps(val)
            if key == "enabled":
                val = 1 if val else 0
            cols.append(f"{key} = ?")
            vals.append(val)
        if not cols:
            return self._get_server(server_id)
        now = utc_now().isoformat()
        cols.append("updated_at = ?")
        vals.extend([now, server_id])
        with self._db.connection() as conn:
            cur = conn.execute(
                f"UPDATE mcp_client_servers SET {', '.join(cols)} WHERE id = ?",
                vals,
            )
            conn.commit()
            if cur.rowcount == 0:
                raise LookupError(f"Server not found: {server_id}")
        return self._get_server(server_id)

    def delete_server(self, server_id: str) -> None:
        """Delete a server row and purge its tools + policy rows."""
        server = self.get_server(server_id)
        if server is None:
            raise LookupError(f"Server not found: {server_id}")
        sanitized = _sanitize_name(server["name"])
        prefix = f"_mcp_{sanitized}_"
        self._delete_tools_for_server(server_id)
        self._delete_policy_rows(prefix)
        with self._db.connection() as conn:
            conn.execute(
                "DELETE FROM mcp_client_servers WHERE id = ?", (server_id,)
            )
            conn.commit()
        logger.info("%s Deleted server %r (id=%s)", _LOG_PREFIX, server["name"], server_id)

    # ── Ping + sync ───────────────────────────────────────────────────────────

    def ping_and_sync(self, server_id: str) -> dict:
        """Connect to the server, sync its tool list, and update status.

        Returns {status, tool_count, reachable}.  Never raises — errors are
        caught and reflected as status='offline'.
        """
        server = self.get_server(server_id)
        if server is None:
            return {"status": _STATUS_OFFLINE, "tool_count": 0, "reachable": False}

        host = server["host"]
        headers = server.get("headers") or {}
        if isinstance(headers, str):
            headers = json.loads(headers) if headers else {}

        try:
            tools = asyncio.run(_async_list_tools(host, headers))
            self._write_tools(server_id, server["name"], tools)
            self._update_status(server_id, _STATUS_ONLINE)
            logger.info(
                "%s Server %r online — %d tools synced",
                _LOG_PREFIX, server["name"], len(tools),
            )
            return {"status": _STATUS_ONLINE, "tool_count": len(tools), "reachable": True}
        except Exception as exc:
            logger.warning(
                "%s Server %r unreachable: %s", _LOG_PREFIX, server["name"], exc
            )
            self._update_status(server_id, _STATUS_OFFLINE)
            return {"status": _STATUS_OFFLINE, "tool_count": 0, "reachable": False}

    def get_server_tools(self, server_id: str) -> list[dict]:
        """Return the raw synced tool inventory for a single server."""
        conn = _open_tools_db()
        try:
            rows = conn.execute(
                "SELECT tool_name, summary, raw_schema FROM mcp_tools "
                "WHERE server_id = ? ORDER BY tool_name",
                (server_id,),
            ).fetchall()
            return [
                {
                    "tool_name": r["tool_name"],
                    "summary": r["summary"],
                    "schema": json.loads(r["raw_schema"]) if r["raw_schema"] else {},
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_tool_schema(self, tool_name: str) -> dict | None:
        """Return the LLM tool spec for a single _mcp_* tool, or None if unknown.

        Reads mcp_tools.sqlite by exact tool_name match and returns the
        {name, description, input_schema} shape AbilityRegistry.build_tools
        expects, so an _mcp_* name appended to mp.active_tools (by find_tools)
        resolves to a full schema for the next ACT iteration's provider call.
        """
        conn = _open_tools_db()
        try:
            row = conn.execute(
                "SELECT summary, raw_schema FROM mcp_tools WHERE tool_name = ?",
                (tool_name,),
            ).fetchone()
            if row is None:
                return None
            try:
                input_schema = json.loads(row["raw_schema"]) if row["raw_schema"] else {}
            except json.JSONDecodeError:
                input_schema = {}
            return {
                "name": tool_name,
                "description": row["summary"] or "",
                "input_schema": input_schema,
            }
        finally:
            conn.close()

    def get_online_mcp_tool_names(self) -> list[str]:
        """Return _mcp_* tool names for servers that are enabled AND online.

        Used by find_tools to gate discoverability and by the /discoverable
        API endpoint.  Disabled or offline servers' tools never appear.
        """
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id FROM mcp_client_servers "
                "WHERE enabled = 1 AND status = ?",
                (_STATUS_ONLINE,),
            ).fetchall()
        if not rows:
            return []
        online_ids = {r[0] for r in rows}
        conn = _open_tools_db()
        try:
            all_rows = conn.execute(
                "SELECT server_id, tool_name FROM mcp_tools"
            ).fetchall()
            return [r["tool_name"] for r in all_rows if r["server_id"] in online_ids]
        finally:
            conn.close()

    def get_online_mcp_tools_index(self) -> list[tuple[str, str]]:
        """Return (call_name, display_name) pairs for enabled+online tools.

        ``call_name`` is the prefixed ``_mcp_<server>_<tool>`` identifier the
        model must invoke; ``display_name`` is the bare server-reported
        ``tool.name`` (e.g. ``list_tickets``).  Used by find_tools to list MCP
        tools in the discoverability hint by their native names without losing
        the prefixed call target.
        """
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id, name FROM mcp_client_servers "
                "WHERE enabled = 1 AND status = ?",
                (_STATUS_ONLINE,),
            ).fetchall()
        if not rows:
            return []
        server_name_by_id = {r[0]: r[1] for r in rows}
        conn = _open_tools_db()
        try:
            all_rows = conn.execute(
                "SELECT server_id, tool_name FROM mcp_tools"
            ).fetchall()
        finally:
            conn.close()
        index: list[tuple[str, str]] = []
        for r in all_rows:
            server_id = r["server_id"]
            if server_id not in server_name_by_id:
                continue
            call_name = r["tool_name"]
            prefix = f"_mcp_{_sanitize_name(server_name_by_id[server_id])}_"
            display = call_name[len(prefix):] if call_name.startswith(prefix) else call_name
            index.append((call_name, display))
        return index

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch_mcp_tool(self, tool_name: str, params: dict) -> dict:
        """Dispatch a _mcp_* tool call to the appropriate remote server.

        Parses the tool_name to find the server and remote tool, opens a
        fresh MCP session, calls the tool, and returns a result dict
        compatible with ActDispatcherService._build_success_result().

        Raises ValueError if the tool is unknown or the server is offline.
        """
        server, remote_tool = self._resolve_tool(tool_name)
        host = server["host"]
        headers = server.get("headers") or {}
        if isinstance(headers, str):
            headers = json.loads(headers) if headers else {}

        # Strip dispatcher-internal keys before forwarding to the remote server.
        clean_params = {
            k: v for k, v in params.items()
            if k not in ("type", "exchange_id", "_rich_media_ordinal")
        }

        try:
            content = asyncio.run(
                _async_call_tool(host, headers, remote_tool, clean_params)
            )
            result_text = self._format_tool_result(content)
            logger.info(
                "%s Dispatched %r → server %r", _LOG_PREFIX, tool_name, server["name"]
            )
            return {"text": result_text}
        except Exception as exc:
            logger.warning("%s Tool dispatch failed for %r: %s", _LOG_PREFIX, tool_name, exc)
            raise RuntimeError(
                f"MCP tool {tool_name!r} failed: {exc}"
            ) from exc

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    def run_heartbeat(self) -> None:
        """Ping every enabled server and update status + tool index.

        Called by mcp_client_worker on a 15-minute loop.  Errors per server
        are caught and logged; the loop continues to the next server.
        """
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id, name FROM mcp_client_servers WHERE enabled = 1"
            ).fetchall()
        server_ids = [(r[0], r[1]) for r in rows]
        logger.info(
            "%s Heartbeat — pinging %d enabled server(s)", _LOG_PREFIX, len(server_ids)
        )
        for server_id, name in server_ids:
            try:
                self.ping_and_sync(server_id)
            except Exception as exc:
                logger.error(
                    "%s Heartbeat error for %r (id=%s): %s",
                    _LOG_PREFIX, name, server_id, exc,
                )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_server(self, server_id: str) -> dict:
        """Fetch a single server row; raise LookupError if missing."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT id, name, host, headers, enabled, status, "
                "last_pinged_at, created_at, updated_at "
                "FROM mcp_client_servers WHERE id = ?",
                (server_id,),
            ).fetchone()
        if row is None:
            raise LookupError(f"Server not found: {server_id}")
        return self._row_to_dict(row)

    @staticmethod
    def _row_to_dict(row) -> dict:
        """Convert a DB row (tuple or sqlite3.Row) to a plain dict."""
        (
            server_id, name, host, headers_raw, enabled,
            status, last_pinged_at, created_at, updated_at,
        ) = row[:9]
        try:
            headers = json.loads(headers_raw) if headers_raw else {}
        except json.JSONDecodeError:
            headers = {}
        return {
            "id": server_id,
            "name": name,
            "host": host,
            "headers": headers,
            "enabled": bool(enabled),
            "status": status,
            "last_pinged_at": last_pinged_at,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def _update_status(self, server_id: str, status: str) -> None:
        """Write the new status and last_pinged_at timestamp."""
        now = utc_now().isoformat()
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE mcp_client_servers "
                "SET status = ?, last_pinged_at = ?, updated_at = ? "
                "WHERE id = ?",
                (status, now, now, server_id),
            )
            conn.commit()

    def _write_tools(self, server_id: str, server_name: str, tools: list[dict]) -> None:
        """Replace the tool index for one server in mcp_tools.sqlite."""
        conn = _open_tools_db()
        try:
            conn.execute(
                "DELETE FROM mcp_tools WHERE server_id = ?", (server_id,)
            )
            for t in tools:
                name = _tool_name(server_name, t["name"])
                summary = t.get("description", "") or ""
                schema_json = json.dumps(t.get("inputSchema") or {})
                conn.execute(
                    "INSERT OR REPLACE INTO mcp_tools "
                    "(server_id, tool_name, summary, raw_schema) "
                    "VALUES (?, ?, ?, ?)",
                    (server_id, name, summary, schema_json),
                )
            # Rebuild FTS index for this server.
            try:
                conn.execute(
                    "INSERT INTO mcp_tools_fts(mcp_tools_fts) VALUES('rebuild')"
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
        finally:
            conn.close()

    def _delete_tools_for_server(self, server_id: str) -> None:
        """Remove all tool rows for a server from mcp_tools.sqlite."""
        conn = _open_tools_db()
        try:
            conn.execute(
                "DELETE FROM mcp_tools WHERE server_id = ?", (server_id,)
            )
            try:
                conn.execute(
                    "INSERT INTO mcp_tools_fts(mcp_tools_fts) VALUES('rebuild')"
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
        finally:
            conn.close()

    def _delete_policy_rows(self, tool_name_prefix: str) -> None:
        """Remove all policy_rules rows whose action_id starts with prefix."""
        with self._db.connection() as conn:
            conn.execute(
                "DELETE FROM policy_rules WHERE action_id LIKE ?",
                (f"{tool_name_prefix}%",),
            )
            conn.commit()

    def _resolve_tool(self, prefixed_name: str) -> tuple[dict, str]:
        """Map a _mcp_<server>_<tool> name to (server_dict, remote_tool_name).

        Strategy: iterate all known servers, check whether their sanitized name
        is a prefix of the tool name, pick the longest match.
        Raises ValueError if no server matches.
        """
        if not prefixed_name.startswith("_mcp_"):
            raise ValueError(f"Not an MCP tool name: {prefixed_name!r}")
        remainder = prefixed_name[5:]  # strip '_mcp_'

        servers = self.list_servers()
        best: tuple[dict, str] | None = None
        best_len = 0

        for srv in servers:
            prefix = _sanitize_name(srv["name"]) + "_"
            if remainder.startswith(prefix) and len(prefix) > best_len:
                best = (srv, remainder[len(prefix):])
                best_len = len(prefix)

        if best is None:
            raise ValueError(
                f"No registered MCP server matches tool {prefixed_name!r}"
            )
        server, remote_tool = best
        if not server.get("enabled"):
            raise ValueError(
                f"MCP server {server['name']!r} is disabled; "
                f"cannot dispatch {prefixed_name!r}"
            )
        return server, remote_tool

    @staticmethod
    def _format_tool_result(content: Any) -> str:
        """Serialize MCP tool result content to a string for the ACT trail."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict):
                    parts.append(block.get("text", json.dumps(block)))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return json.dumps(content, default=str)
