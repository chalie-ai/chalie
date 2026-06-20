"""
McpClientService — outbound MCP client connection manager.

Chalie connects OUT to remote MCP servers via the MCP streamable-HTTP transport.
All MCP client calls bridge to asyncio.run() per dispatch (safe because Chalie's
worker threads carry no running loop). Tool names use scheme: _mcp_<sanitized_server>_<remote>.

Each server's tool index is synced via ping_and_sync and stored in mcp_tools.sqlite;
vector embeddings are optional — queries degrade gracefully without sqlite_vec.
"""

import asyncio
import json
import logging
import re
import sqlite3
import uuid
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from services.database_service import get_shared_db_service
from services.file_mapper_service import FileMapperService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[MCP CLIENT]"

# Status strings — these exact values are part of the public status contract.
_STATUS_UNKNOWN = "unknown"
_STATUS_ONLINE = "online"
_STATUS_OFFLINE = "offline"

# Name-sanitization pattern: keep lowercase alpha, digits, underscore.
_SANITIZE_RE = re.compile(r"[^a-z0-9_]")


class McpToolUnknown(Exception):
    """An ``_mcp_*`` name resolves to no enabled/registered server.

    Distinct from a reachable-but-failing call: the tool name itself cannot be
    routed (no matching server, or the matching server is disabled), so retrying
    is pointless until the server is (re-)added/enabled.
    """


class McpServerUnreachable(Exception):
    """The remote MCP server could not be reached (transport/connect/timeout).

    Carries the human-facing ``server_name`` so the proxy can NAME the failing
    endpoint in its error envelope instead of leaking a transport stack trace.
    """

    def __init__(self, server_name: str, detail: str) -> None:
        super().__init__(f"MCP server {server_name!r} is unreachable: {detail}")
        self.server_name = server_name
        self.detail = detail


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


# Short words that stay lowercase mid-label when humanizing a tool name.
_MINOR_LABEL_WORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in",
    "of", "on", "or", "the", "to", "with",
})


def humanize_segment(segment: str) -> str:
    """Turn an underscore identifier into a Title-Cased label.

    ``add_comment_to_pending_review`` -> ``Add Comment to Pending Review``.
    The first word is always capitalized; minor words elsewhere stay lowercase.
    """
    words = [w for w in segment.split("_") if w]
    out = []
    for i, w in enumerate(words):
        out.append(w if (i and w.lower() in _MINOR_LABEL_WORDS) else w[:1].upper() + w[1:])
    return " ".join(out)


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

# Vector embedding tables — keyed by tool_name (stable across _write_tools churn).
# mcp_tools_vec.rowid == mcp_tool_vectors.rowid for the JOIN.
# Created in the same _open_tools_db() call so callers get a unified DB handle.
_TOOLS_VECTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS mcp_tool_vectors (
    rowid     INTEGER PRIMARY KEY,
    tool_name TEXT UNIQUE NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS mcp_tools_vec USING vec0(embedding float[768]);
"""


def _open_tools_db() -> sqlite3.Connection:
    """Open (and initialize if necessary) the mcp_tools.sqlite runtime DB.

    Creates the FTS table and, when sqlite_vec is available, the vector
    embedding tables.  If sqlite_vec cannot be loaded the vec tables are
    skipped and the DB remains FTS-only — queries degrade gracefully.
    """
    db_path: Path = FileMapperService.get_mcp_tools_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_TOOLS_SCHEMA)
    try:
        conn.executescript(_TOOLS_FTS_SCHEMA)
    except sqlite3.OperationalError:
        pass  # FTS5 may not be available in all environments
    try:
        conn.enable_load_extension(True)
        try:
            import sqlite_vec  # noqa: PLC0415
            sqlite_vec.load(conn)
        except Exception as exc:
            logger.debug("%s sqlite_vec module load failed, trying vec0: %s", _LOG_PREFIX, exc)
            conn.load_extension("vec0")
        conn.executescript(_TOOLS_VECTOR_SCHEMA)
    except Exception as exc:
        logger.warning(
            "%s sqlite_vec unavailable — vec tables skipped, FTS-only mode: %s",
            _LOG_PREFIX, exc,
        )
    conn.commit()
    return conn


# ── Async MCP helpers ────────────────────────────────────────────────────────

async def _async_list_tools(host: str, headers: dict[str, str]) -> list[dict[str, object]]:
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
    host: str, headers: dict[str, str], remote_tool: str, params: dict[str, object]
) -> object:
    """Connect to a remote MCP server and call a single tool.

    Returns the full ``CallToolResult`` so the caller can read ``isError`` and
    ``structuredContent`` alongside the content blocks — an MCP tool signalling
    failure via ``isError=true`` must NOT be mistaken for success, and structured
    JSON must be preserved as structure rather than collapsed into a blob.
    """
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession

    async with streamablehttp_client(host, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(remote_tool, params)


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

    def list_servers(self) -> list[dict[str, object]]:
        """Return all mcp_client_servers rows as dicts."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id, name, host, headers, enabled, status, "
                "last_pinged_at, created_at, updated_at "
                "FROM mcp_client_servers ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def add_server(self, name: str, host: str, headers: dict[str, str], enabled: bool) -> dict[str, object]:
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

    def _find_by_normalized_host(self, host: str) -> dict[str, object] | None:
        """Return an existing server resolving to the same endpoint, else None."""
        key = _normalize_host(host)
        for server in self.list_servers():
            if _normalize_host(cast(str, server["host"])) == key:
                return server
        return None

    def _upsert_existing(self, existing: dict[str, object], name: str, host: str,
                         headers: dict[str, str]) -> dict[str, object]:
        """Re-add of a known endpoint: update fields and re-enable.

        Always purge the existing toolset (rows + vector embeddings) before the
        caller's re-sync repopulates it.  This reads the *current* tool_names off
        ``mcp_tools`` while they are still present, so a tool the server has since
        removed can never leave an orphaned vector row — the post-add
        ``ping_and_sync`` then writes back only the live toolset.  The tool prefix
        is derived from the server name, so a name change additionally invalidates
        the old ``_mcp_<oldname>_*`` policy rows.
        """
        old_prefix = f"_mcp_{_sanitize_name(cast(str, existing['name']))}_"
        new_prefix = f"_mcp_{_sanitize_name(name)}_"
        if old_prefix != new_prefix:
            self._delete_policy_rows(cast(str, existing["id"]))
        self._delete_tools_for_server(cast(str, existing["id"]))
        logger.info(
            "%s Re-add of existing endpoint %r → upsert id=%s",
            _LOG_PREFIX, host, existing["id"],
        )
        return self.update_server(cast(str, existing["id"]), {
            "name": name,
            "host": host,
            "headers": headers or {},
            "enabled": True,
        })

    def get_server(self, server_id: str) -> dict[str, object] | None:
        """Return a single server dict, or None if not found."""
        try:
            return self._get_server(server_id)
        except LookupError:
            return None

    def update_server(self, server_id: str, updates: dict[str, object]) -> dict[str, object]:
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
        self._delete_policy_rows(server_id)
        self._delete_tools_for_server(server_id)
        with self._db.connection() as conn:
            conn.execute(
                "DELETE FROM mcp_client_servers WHERE id = ?", (server_id,)
            )
            conn.commit()
        logger.info("%s Deleted server %r (id=%s)", _LOG_PREFIX, server["name"], server_id)

    # ── Ping + sync ───────────────────────────────────────────────────────────

    def ping_and_sync(self, server_id: str) -> dict[str, object]:
        """Connect to the server, sync its tool list, and update status.

        Returns ``{status, tool_count, reachable, error}``.  Never raises —
        connect/transport failures are caught and reflected as status='offline'.
        ``error`` is ``None`` on success and the stringified exception on failure,
        so callers can classify the failure (e.g. a 401/403 auth rejection vs a
        plain unreachable host) instead of losing the detail.
        """
        server = self.get_server(server_id)
        if server is None:
            return {
                "status": _STATUS_OFFLINE,
                "tool_count": 0,
                "reachable": False,
                "error": "server not found",
            }

        host = cast(str, server["host"])
        headers = server.get("headers") or {}
        if isinstance(headers, str):
            headers = json.loads(headers) if headers else {}

        try:
            tools = asyncio.run(_async_list_tools(host, cast(dict[str, str], headers)))
            self._write_tools(server_id, cast(str, server["name"]), tools)
            self._update_status(server_id, _STATUS_ONLINE)
            logger.info(
                "%s Server %r online — %d tools synced",
                _LOG_PREFIX, server["name"], len(tools),
            )
            return {
                "status": _STATUS_ONLINE,
                "tool_count": len(tools),
                "reachable": True,
                "error": None,
            }
        except Exception as exc:
            logger.warning(
                "%s Server %r unreachable: %s", _LOG_PREFIX, server["name"], exc
            )
            self._update_status(server_id, _STATUS_OFFLINE)
            return {
                "status": _STATUS_OFFLINE,
                "tool_count": 0,
                "reachable": False,
                "error": str(exc),
            }

    def get_server_tools(self, server_id: str) -> list[dict[str, object]]:
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

    def get_tool_schema(self, tool_name: str) -> dict[str, object] | None:
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

    def label_mcp_permissions(self, permissions: list[str]) -> dict[str, dict[str, str]]:
        """Map ``_mcp_*`` policy permissions to display ``{group, label}``.

        ``group`` is the configured server title (e.g. ``GitHub``); ``label`` is
        the remote tool name humanized (``_mcp_github_add_comment_to_pending_review``
        -> ``Add Comment to Pending Review``).  Non-MCP permissions and orphan rows
        (no matching server) are omitted, so callers fall back to the raw string.

        Disabled/offline servers ARE included — the Brain policies UI must group
        their rows too — unlike ``_resolve_tool`` which gates on enabled status.
        One ``list_servers`` read serves the whole batch.
        """
        servers = sorted(
            ({"slug": _sanitize_name(cast(str, s["name"])), "title": cast(str, s["name"])}
             for s in self.list_servers()),
            key=lambda s: len(s["slug"]), reverse=True,  # longest slug wins the prefix race
        )
        out: dict[str, dict[str, str]] = {}
        for perm in permissions:
            if not perm.startswith("_mcp_"):
                continue
            base, _, action = perm.partition(".")
            for s in servers:
                prefix = f"_mcp_{s['slug']}_"
                if base.startswith(prefix):
                    label = humanize_segment(base[len(prefix):])
                    if action:
                        label += f" — {humanize_segment(action)}"
                    out[perm] = {"group": s["title"], "label": label}
                    break
        return out

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch_mcp_tool(self, tool_name: str, params: dict[str, object]) -> dict[str, object]:
        """Dispatch a ``_mcp_*`` tool call to the appropriate remote server.

        Resolves the server + remote tool, opens a fresh MCP session, calls the
        tool, and returns a SHAPED result the synthetic ``_MCPAbility`` proxy maps
        straight onto the ``ToolResult`` contract::

            {"server": <server name>, "is_error": <bool>, "body": <str|dict|list>}

        ``body`` is a ``str`` when the remote returned prose (text content blocks
        joined) and a ``dict``/``list`` when the remote returned structured JSON
        (preserved as structure, never re-serialised into a string). ``is_error``
        carries the remote's ``isError`` flag so an MCP-level tool failure is
        surfaced as an error, not dressed up as success.

        Raises :class:`McpToolUnknown` when the name routes to no enabled server,
        and :class:`McpServerUnreachable` (naming the server) on any transport /
        connect / timeout failure.
        """
        try:
            server, remote_tool = self._resolve_tool(tool_name)
        except ValueError as exc:
            raise McpToolUnknown(str(exc)) from exc

        host = cast(str, server["host"])
        headers = server.get("headers") or {}
        if isinstance(headers, str):
            headers = json.loads(headers) if headers else {}

        try:
            result = asyncio.run(
                _async_call_tool(host, cast(dict[str, str], headers), remote_tool, params)
            )
        except Exception as exc:
            logger.warning("%s Tool dispatch failed for %r: %s", _LOG_PREFIX, tool_name, exc)
            raise McpServerUnreachable(cast(str, server["name"]), str(exc)) from exc

        logger.info(
            "%s Dispatched %r → server %r", _LOG_PREFIX, tool_name, server["name"]
        )
        return {
            "server": server["name"],
            "is_error": bool(getattr(result, "isError", False)),
            "body": self._shape_tool_result(result),
        }

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

    def embed_server_tools(self, server_id: str) -> None:
        """Generate and store vector embeddings for one server's tools.

        Called only on add (never on heartbeat/enable) so the 15-min sync path
        has zero embedding overhead.  Embeddings are keyed by the stable
        tool_name (not by mcp_tools.id, which is reassigned on every _write_tools
        call).  Vectors for the server's *current* toolset are replaced in place,
        so re-embedding is idempotent.  Orphan-free re-adds (where the server has
        dropped a tool) are guaranteed upstream: ``_upsert_existing`` purges the
        prior toolset via ``_delete_tools_for_server`` before ``ping_and_sync``
        repopulates only the live tools.

        Lazy-imports EmbeddingService (avoids import-time cost and circular refs).
        Wraps the entire body so a failure leaves the server FTS-searchable
        without surfacing an error to the caller.

        Depends on: services.embedding_service.EmbeddingService (offline ONNX,
        no provider/mp required) and services.embedding_utils.pack_embedding.
        McpClientService is the only add-time embedding trigger — do not call
        this from _write_tools or run_heartbeat.
        """
        try:
            # Lazy imports — avoid circular import chain and expensive ONNX load
            # at module initialisation time.  EmbeddingService depends on
            # McpClientService indirectly (via find_tools); importing at call
            # site breaks the cycle cleanly.
            from services.embedding_service import EmbeddingService  # noqa: PLC0415
            from services.embedding_utils import pack_embedding  # noqa: PLC0415

            server = self.get_server(server_id)
            if server is None:
                logger.warning("%s embed_server_tools: server %r not found", _LOG_PREFIX, server_id)
                return

            conn = _open_tools_db()
            try:
                rows = conn.execute(
                    "SELECT tool_name, summary FROM mcp_tools WHERE server_id = ?",
                    (server_id,),
                ).fetchall()
            finally:
                conn.close()

            if not rows:
                logger.debug("%s embed_server_tools: no tools for server %r", _LOG_PREFIX, server_id)
                return

            server_name = cast(str, server["name"])
            prefix = f"_mcp_{_sanitize_name(server_name)}_"
            tool_names = [r["tool_name"] for r in rows]
            texts = []
            for r in rows:
                tool_name = r["tool_name"]
                display = tool_name[len(prefix):] if tool_name.startswith(prefix) else tool_name
                display_words = display.replace("_", " ")
                summary = r["summary"] or ""
                texts.append(f"{display_words}. {summary}" if summary else display_words)

            embeddings = EmbeddingService().generate_embeddings_batch(texts)

            conn = _open_tools_db()
            try:
                # Replace-all-for-server: purge existing vec rows for these tool_names.
                placeholders = ",".join("?" * len(tool_names))
                existing = conn.execute(
                    f"SELECT rowid FROM mcp_tool_vectors WHERE tool_name IN ({placeholders})",
                    tool_names,
                ).fetchall()
                if existing:
                    vec_rowids = [r[0] for r in existing]
                    rowid_placeholders = ",".join("?" * len(vec_rowids))
                    conn.execute(
                        f"DELETE FROM mcp_tools_vec WHERE rowid IN ({rowid_placeholders})",
                        vec_rowids,
                    )
                    conn.execute(
                        f"DELETE FROM mcp_tool_vectors WHERE tool_name IN ({placeholders})",
                        tool_names,
                    )

                # Insert fresh rows — mcp_tool_vectors assigns a new rowid.
                for tool_name, embedding in zip(tool_names, embeddings):
                    conn.execute(
                        "INSERT INTO mcp_tool_vectors (tool_name) VALUES (?)",
                        (tool_name,),
                    )
                    rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    blob = pack_embedding(embedding)
                    conn.execute(
                        "INSERT INTO mcp_tools_vec (rowid, embedding) VALUES (?, ?)",
                        (rowid, blob),
                    )
                conn.commit()
                logger.info(
                    "%s Embedded %d tools for server %r",
                    _LOG_PREFIX, len(tool_names), server_name,
                )
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(
                "%s embed_server_tools failed for server %r — FTS-only fallback: %s",
                _LOG_PREFIX, server_id, exc,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_server(self, server_id: str) -> dict[str, object]:
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
    def _row_to_dict(row: "sqlite3.Row | tuple[object, ...]") -> dict[str, object]:
        """Convert a DB row (tuple or sqlite3.Row) to a plain dict."""
        (
            server_id, name, host, headers_raw, enabled,
            status, last_pinged_at, created_at, updated_at,
        ) = row[:9]
        try:
            headers = json.loads(cast(str, headers_raw)) if headers_raw else {}
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

    def _write_tools(self, server_id: str, server_name: str, tools: list[dict[str, object]]) -> None:
        """Replace the tool index for one server in mcp_tools.sqlite."""
        conn = _open_tools_db()
        try:
            conn.execute(
                "DELETE FROM mcp_tools WHERE server_id = ?", (server_id,)
            )
            for t in tools:
                name = _tool_name(server_name, cast(str, t["name"]))
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
        """Remove all tool rows (including vector embeddings) for a server.

        Purges mcp_tool_vectors + mcp_tools_vec BEFORE deleting mcp_tools so
        we can still read the tool_names for the vec lookup.  This single
        chokepoint covers both delete_server and the _upsert_existing name-change
        path — neither needs to repeat the purge logic.
        """
        conn = _open_tools_db()
        try:
            tool_rows = conn.execute(
                "SELECT tool_name FROM mcp_tools WHERE server_id = ?", (server_id,)
            ).fetchall()
            tool_names = [r["tool_name"] for r in tool_rows]

            if tool_names:
                placeholders = ",".join("?" * len(tool_names))
                try:
                    vec_rows = conn.execute(
                        f"SELECT rowid FROM mcp_tool_vectors WHERE tool_name IN ({placeholders})",
                        tool_names,
                    ).fetchall()
                    if vec_rows:
                        vec_rowids = [r[0] for r in vec_rows]
                        rowid_placeholders = ",".join("?" * len(vec_rowids))
                        conn.execute(
                            f"DELETE FROM mcp_tools_vec WHERE rowid IN ({rowid_placeholders})",
                            vec_rowids,
                        )
                    conn.execute(
                        f"DELETE FROM mcp_tool_vectors WHERE tool_name IN ({placeholders})",
                        tool_names,
                    )
                except sqlite3.OperationalError as exc:
                    if "no such table" in str(exc).lower():
                        # Vec tables absent (sqlite_vec unavailable) — expected.
                        logger.debug("%s Vec purge skipped (tables absent): %s", _LOG_PREFIX, exc)
                    else:
                        # Any other operational error (locked/corrupt DB) is real —
                        # surface it instead of masking it behind the absent-table case.
                        logger.warning("%s Vec purge failed unexpectedly: %s", _LOG_PREFIX, exc)

            conn.execute("DELETE FROM mcp_tools WHERE server_id = ?", (server_id,))
            try:
                conn.execute("INSERT INTO mcp_tools_fts(mcp_tools_fts) VALUES('rebuild')")
            except sqlite3.OperationalError:
                pass
            conn.commit()
        finally:
            conn.close()

    def _delete_policy_rows(self, server_id: str) -> None:
        """Clear one server's lazily-provisioned policy rows — one exact-match
        delete per tool, keyed off the tool_names in mcp_tools.

        Exact matches (never a LIKE prefix) so a server can never delete a
        colliding server's rows.  Reads mcp_tools, so callers MUST invoke this
        before _delete_tools_for_server purges that index.
        """
        tools_db = _open_tools_db()
        try:
            tool_names = [r["tool_name"] for r in tools_db.execute(
                "SELECT tool_name FROM mcp_tools WHERE server_id = ?", (server_id,)
            ).fetchall()]
        finally:
            tools_db.close()
        with self._db.connection() as conn:
            for tool_name in tool_names:
                conn.execute("DELETE FROM policy WHERE permission = ?", (tool_name,))
            conn.commit()

    def _resolve_tool(self, prefixed_name: str) -> tuple[dict[str, object], str]:
        """Map a _mcp_<server>_<tool> name to (server_dict, remote_tool_name).

        Strategy: iterate all known servers, check whether their sanitized name
        is a prefix of the tool name, pick the longest match.
        Raises ValueError if no server matches.
        """
        if not prefixed_name.startswith("_mcp_"):
            raise ValueError(f"Not an MCP tool name: {prefixed_name!r}")
        remainder = prefixed_name[5:]  # strip '_mcp_'

        servers = self.list_servers()
        best: tuple[dict[str, object], str] | None = None
        best_len = 0

        for srv in servers:
            prefix = _sanitize_name(cast(str, srv["name"])) + "_"
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
    def _shape_tool_result(result: object) -> str | dict[str, object] | list[object]:
        """Shape an MCP ``CallToolResult`` into a ``str`` (prose) or ``dict``/``list``
        (structured) body for the ToolResult contract.

        The text content blocks are the reliable, consistent channel every MCP
        server populates (``structuredContent`` is an optional parallel field that
        servers fill inconsistently — e.g. FastMCP wraps every scalar return in
        ``{"result": …}`` noise — so it is NOT used as the body source). The blocks
        are joined; if that joined text parses cleanly as a JSON object/array it is
        returned as the parsed structure (so a server that emits text-encoded JSON
        still yields a structured body the dispatcher renders as compact JSON —
        never a JSON-string wrapped inside another JSON string). Otherwise the
        joined text is returned verbatim as prose.

        A bare value (no ``content`` attribute) degrades to its string form.
        """
        content = getattr(result, "content", result)
        if content is None:
            return ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict):
                    parts.append(block.get("text", json.dumps(block)))
                else:
                    parts.append(str(block))
            text = "\n".join(parts)
        else:
            return str(content)

        stripped = text.strip()
        if stripped[:1] in ("{", "["):
            try:
                parsed = json.loads(stripped)
            except ValueError:
                return text
            if isinstance(parsed, (dict, list)):
                return parsed
        return text
