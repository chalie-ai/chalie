"""
McpClientService — outbound MCP client connection manager.

Chalie connects OUT to remote MCP servers via the MCP streamable-HTTP transport.
All MCP client calls bridge to asyncio.run() per dispatch (safe because Chalie's
worker threads carry no running loop). Tool names use scheme: _mcp_<sanitized_server>_<remote>.

Each server's tool index is synced via ping_and_sync and stored in mcp_tools.sqlite;
the DB is a plain tool/schema store with no search index.
"""

import asyncio
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from exceptions import McpServerUnreachable, McpToolUnknown
from models.mcp_client_server import McpClientServer
from models.mcp_tool import McpTool
from services.database import Database
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


def _open_tools_db() -> sqlite3.Connection:
    """Open (and initialize if necessary) the mcp_tools.sqlite runtime DB."""
    db_path: Path = FileMapperService.get_mcp_tools_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = Database.conn(str(db_path))
    conn.executescript(_TOOLS_SCHEMA)
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

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def list_servers(self) -> list[dict[str, object]]:
        """Return all mcp_client_servers rows as dicts."""
        servers = McpClientServer.order_by("created_at").get()
        return [self._server_to_dict(s) for s in servers]

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
            return self._upsert_existing(cast(str, existing["id"]), cast(str, existing["name"]), name, host, headers)

        server = McpClientServer(
            name=name,
            host=host,
            headers=json.dumps(headers or {}),
            enabled=1 if enabled else 0,
            status=_STATUS_UNKNOWN,
            created_at=utc_now().isoformat(),
            updated_at=utc_now().isoformat(),
        )
        server.save()
        server_id = cast(str, server.id)
        logger.info("%s Added server %r (id=%s)", _LOG_PREFIX, name, server_id)
        # Re-read the row: columns left unset so their SQL defaults fire
        # (last_pinged_at) exist only on a hydrated instance, and
        # _server_to_dict projects every column.
        saved = McpClientServer.get(server_id)
        if saved is None:
            raise RuntimeError(f"mcp server row vanished after insert: {server_id}")
        return self._server_to_dict(saved)

    def _find_by_normalized_host(self, host: str) -> dict[str, object] | None:
        """Return an existing server resolving to the same endpoint, else None."""
        key = _normalize_host(host)
        for server in self.list_servers():
            if _normalize_host(cast(str, server["host"])) == key:
                return server
        return None

    def _upsert_existing(self, existing_id: str, existing_name: str, name: str, host: str,
                         headers: dict[str, str]) -> dict[str, object]:
        """Re-add of a known endpoint: update fields and re-enable.

        Always purge the existing toolset before the caller's re-sync
        repopulates it.  This reads the *current* tool_names off ``mcp_tools``
        while they are still present, so a tool the server has since removed
        can never leave an orphaned row — the post-add ``ping_and_sync`` then
        writes back only the live toolset.  The tool prefix is derived from the
        server name, so a name change additionally invalidates the old
        ``_mcp_<oldname>_*`` policy rows.
        """
        old_prefix = f"_mcp_{_sanitize_name(existing_name)}_"
        new_prefix = f"_mcp_{_sanitize_name(name)}_"
        if old_prefix != new_prefix:
            self._delete_policy_rows(existing_id)
        self._delete_tools_for_server(existing_id)
        logger.info(
            "%s Re-add of existing endpoint %r → upsert id=%s",
            _LOG_PREFIX, host, existing_id,
        )
        return self.update_server(existing_id, {
            "name": name,
            "host": host,
            "headers": headers or {},
            "enabled": True,
        })

    def get_server(self, server_id: str) -> dict[str, object] | None:
        """Return a single server dict, or None if not found."""
        server = McpClientServer.get(server_id)
        if server is None:
            return None
        return self._server_to_dict(server)

    def update_server(self, server_id: str, updates: dict[str, object]) -> dict[str, object]:
        """Apply a partial update to a server row.

        Accepts: name, host, headers (dict or JSON string), enabled (bool|int).
        Returns the updated server dict.  Raises LookupError if not found.
        """
        allowed = {"name", "host", "headers", "enabled"}
        server = McpClientServer.get(server_id)
        if server is None:
            raise LookupError(f"Server not found: {server_id}")
        for key, val in updates.items():
            if key not in allowed:
                continue
            if key == "headers" and isinstance(val, dict):
                val = json.dumps(val)
            if key == "enabled":
                val = 1 if val else 0
            setattr(server, key, val)
        server.updated_at = utc_now().isoformat()
        server.save()
        return self._server_to_dict(server)

    def delete_server(self, server_id: str) -> None:
        """Delete a server row and purge its tools + policy rows."""
        server = McpClientServer.get(server_id)
        if server is None:
            raise LookupError(f"Server not found: {server_id}")
        self._delete_policy_rows(server_id)
        self._delete_tools_for_server(server_id)
        server.delete()
        logger.info("%s Deleted server %r (id=%s)", _LOG_PREFIX, server.name, server_id)

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
        rows = McpTool.filter("server_id", server_id).order_by("tool_name").select(
            "tool_name", "summary", "raw_schema"
        )
        return [
            {
                "tool_name": r["tool_name"],
                "summary": r["summary"],
                "schema": json.loads(cast(str, r["raw_schema"])) if r["raw_schema"] else {},
            }
            for r in rows
        ]

    def get_tool_schema(self, tool_name: str) -> dict[str, object] | None:
        """Return the LLM tool spec for a single _mcp_* tool, or None if unknown.

        Reads mcp_tools.sqlite by exact tool_name match and returns the
        {name, description, input_schema} shape AbilityRegistry.build_tools
        expects, so an _mcp_* name appended to mp.active_tools (by mcp_tools)
        resolves to a full schema for the next ACT iteration's provider call.
        """
        row = McpTool.filter("tool_name", tool_name).first()
        if row is None:
            return None
        try:
            input_schema = json.loads(row.raw_schema) if row.raw_schema else {}
        except json.JSONDecodeError:
            input_schema = {}
        return {
            "name": tool_name,
            "description": row.summary or "",
            "input_schema": input_schema,
        }

    def get_online_mcp_tool_names(self) -> list[str]:
        """Return _mcp_* tool names for servers that are enabled AND online.

        Used by mcp_tools to validate activation and by the /discoverable
        API endpoint.  Disabled or offline servers' tools never appear.
        """
        online_ids = cast(
            "list[str]",
            [
                s.id
                for s in McpClientServer.filter("enabled", 1).filter("status", _STATUS_ONLINE).get()
                if s.id is not None
            ],
        )
        if not online_ids:
            return []
        return cast("list[str]", McpTool.filter_in("server_id", online_ids).pluck("tool_name"))

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
        servers = McpClientServer.filter("enabled", 1).get()
        server_ids = [(s.id, s.name) for s in servers]
        logger.info(
            "%s Heartbeat — pinging %d enabled server(s)", _LOG_PREFIX, len(server_ids)
        )
        for server_id, name in server_ids:
            if server_id is None:
                continue
            try:
                self.ping_and_sync(server_id)
            except Exception as exc:
                logger.exception(
                    "%s Heartbeat error for %r (id=%s): %s",
                    _LOG_PREFIX, name, server_id, exc,
                )


    # ── Internal helpers ──────────────────────────────────────────────────────

    def _server_to_dict(self, server: McpClientServer) -> dict[str, object]:
        """Project a McpClientServer instance to the service's dict shape."""
        try:
            headers = json.loads(server.headers) if server.headers else {}
        except json.JSONDecodeError:
            headers = {}
        return {
            "id": server.id,
            "name": server.name,
            "host": server.host,
            "headers": headers,
            "enabled": bool(server.enabled),
            "status": server.status,
            "last_pinged_at": server.last_pinged_at,
            "created_at": server.created_at,
            "updated_at": server.updated_at,
        }

    def _update_status(self, server_id: str, status: str) -> None:
        """Write the new status and last_pinged_at timestamp."""
        now = utc_now().isoformat()
        McpClientServer.filter("id", server_id).update(
            status=status,
            last_pinged_at=now,
            updated_at=now,
        )

    def _write_tools(self, server_id: str, server_name: str, tools: list[dict[str, object]]) -> None:
        """Replace the tool index for one server in mcp_tools.sqlite."""
        _open_tools_db()
        with Database.transaction(str(FileMapperService.get_mcp_tools_db_path())):
            McpTool.filter("server_id", server_id).delete()
            for t in tools:
                name = _tool_name(server_name, cast(str, t["name"]))
                summary = t.get("description", "") or ""
                schema_json = json.dumps(t.get("inputSchema") or {})
                tool = McpTool(
                    server_id=server_id,
                    tool_name=name,
                    summary=summary,
                    raw_schema=schema_json,
                )
                tool.save()

    def _delete_tools_for_server(self, server_id: str) -> None:
        """Remove all tool rows for a server.

        This single chokepoint covers both delete_server and the
        _upsert_existing name-change path — neither needs to repeat the purge
        logic.
        """
        _open_tools_db()
        with Database.transaction(str(FileMapperService.get_mcp_tools_db_path())):
            McpTool.filter("server_id", server_id).delete()

    def _delete_policy_rows(self, server_id: str) -> None:
        """Clear one server's lazily-provisioned policy rows — one exact-match
        delete per tool, keyed off the tool_names in mcp_tools.

        Exact matches (never a LIKE prefix) so a server can never delete a
        colliding server's rows.  Reads mcp_tools, so callers MUST invoke this
        before _delete_tools_for_server purges that index.
        """
        tool_names = McpTool.filter("server_id", server_id).pluck("tool_name")
        with Database.transaction() as conn:
            for tool_name in tool_names:
                conn.execute("DELETE FROM policy WHERE permission = ?", (tool_name,))

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
