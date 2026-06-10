"""
MCP Manager Ability — manage outbound MCP client server connections.

Actions: list | add | enable | disable

Policy tier:
  SYSTEM = True   — always-allowed, never shown in Policy Manager.
  DISCOVERABLE    — listed in each channel's ``ProcessorConfig.discoverable``
                    (the default surface is ``configs.channels._common.
                    DEFAULT_DISCOVERABLE``) so find_tools can surface it when
                    the user asks to connect/manage remote MCP servers.

When `add` runs, it creates the server row AND immediately calls
ping_and_sync() so remote tools are indexed before the LLM's next turn.

Why a separate MCP manager ability instead of a REST-only workflow:
the LLM needs to be able to set up an MCP connection when a user asks
conversationally ("connect to the taskie MCP server at …"), without the
user having to open the Brain UI.  This ability handles that path.
"""

import logging
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[MCP MANAGER]"


class McpManagerAbility(Ability):
    """Manage outbound MCP client server connections (add/list/enable/disable)."""

    def get_name(self) -> str:
        return "mcp_manager"

    def get_summary(self) -> str:
        return (
            "Connect Chalie to a remote MCP server so its tools become available. "
            "Use to add, list, enable, or disable outbound MCP server connections."
        )

    def get_examples(self) -> list[str]:
        return [
            "connect to the MCP server at https://mcp.example.com/mcp",
            "add an MCP connection named taskie at https://mcp.example.com/mcp",
            "list all connected MCP servers",
            "enable the taskie MCP server",
            "disable the weather MCP server",
            "show me which external tools are available from remote MCP servers",
            "set up a connection to an external agent via MCP",
            "what remote tools can you access through MCP?",
        ]

    def get_search_tooltip(self) -> str:
        return "connect to a remote MCP server"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "add", "enable", "disable"],
                "description": (
                    "list: show all configured MCP servers and their status. "
                    "add: register a new remote MCP server. "
                    "enable: re-enable a previously disabled server. "
                    "disable: temporarily disable a server (keeps the row, "
                    "hides its tools from discovery)."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "For add/enable/disable: human-readable server label "
                    "(e.g. 'taskie', 'home-assistant')."
                ),
            },
            "host": {
                "type": "string",
                "description": (
                    "For add: full URL including port, e.g. "
                    "'https://mcp.example.com/mcp'."
                ),
            },
            "headers": {
                "type": "object",
                "description": (
                    "For add: optional extra HTTP headers as a JSON object "
                    "(e.g. {'Authorization': 'Bearer …'})."
                ),
            },
            "server_id": {
                "type": "string",
                "description": "For enable/disable: the server UUID from list.",
            },
        },
        "required": ["action"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    # SYSTEM: always-allowed, hidden from Policy Manager (same pattern as memory).
    # Management operations are Chalie self-configuration, not
    # user-data writes — no per-action policy gate is appropriate.
    SYSTEM = True

    def run(self, params: dict) -> ToolResult:
        """Dispatch to the appropriate sub-action handler."""
        action = params.get("action", "").strip()
        dispatch = {
            "list": self._do_list,
            "add": self._do_add,
            "enable": self._do_enable,
            "disable": self._do_disable,
        }
        handler = dispatch.get(action)
        if handler is None:
            return ToolResult.ok(
                f"Unknown action: {action!r}. Must be one of: list, add, enable, disable."
            )
        return handler(params)

    # ── Sub-action handlers ───────────────────────────────────────────────────

    def _do_list(self, params: dict) -> ToolResult:
        """List all configured servers with their current status."""
        from services.mcp_client_service import McpClientService
        servers = McpClientService().list_servers()
        if not servers:
            return ToolResult.ok("No MCP servers configured. Use action=add to register one.")
        lines = []
        for s in servers:
            state = "enabled" if s["enabled"] else "disabled"
            lines.append(
                f"• {s['name']} ({s['host']}) — {s['status']}, {state} [id={s['id']}]"
            )
        return ToolResult.ok("Configured MCP servers:\n" + "\n".join(lines))

    def _do_add(self, params: dict) -> ToolResult:
        """Register a new server and immediately ping it."""
        from services.mcp_client_service import McpClientService
        name = (params.get("name") or "").strip()
        host = (params.get("host") or "").strip()
        if not name:
            return ToolResult.ok("Error: name is required to add a server.")
        if not host:
            return ToolResult.ok("Error: host is required to add a server.")
        headers = params.get("headers") or {}
        enabled = True  # new servers are enabled by default

        svc = McpClientService()
        server = svc.add_server(name=name, host=host, headers=headers, enabled=enabled)
        server_id = server["id"]

        # Trigger immediate sync so tools are discoverable in this turn.
        sync_result = svc.ping_and_sync(server_id)
        status = sync_result["status"]
        tool_count = sync_result["tool_count"]

        if sync_result["reachable"]:
            # Build vector embeddings for the newly-synced tools so semantic
            # queries can reach them immediately.  Add-only — never called on
            # heartbeat or enable so the 15-min sync path stays zero-cost.
            svc.embed_server_tools(server_id)
            msg = (
                f"Connected to MCP server {name!r} at {host}. "
                f"Synced {tool_count} tool(s). Server is now online."
            )
        else:
            msg = (
                f"Registered MCP server {name!r} at {host} (id={server_id}). "
                f"Server is currently unreachable (status={status}). "
                "Tools will be indexed when the server comes online."
            )
        logger.info("%s Added server %r — status=%s tools=%d", _LOG_PREFIX, name, status, tool_count)
        return ToolResult.ok(msg)

    def _do_enable(self, params: dict) -> ToolResult:
        """Enable a previously disabled server and re-sync its tools."""
        from services.mcp_client_service import McpClientService
        server_id = self._resolve_server_id(params)
        if server_id is None:
            return ToolResult.ok("Error: provide server_id (from list) or name to enable.")
        svc = McpClientService()
        try:
            svc.update_server(server_id, {"enabled": True})
        except LookupError:
            return ToolResult.ok(f"No server found with id={server_id!r}.")
        sync = svc.ping_and_sync(server_id)
        return ToolResult.ok(
            f"Server enabled and synced — "
            f"status={sync['status']}, tools={sync['tool_count']}."
        )

    def _do_disable(self, params: dict) -> ToolResult:
        """Disable a server so its tools disappear from discovery."""
        from services.mcp_client_service import McpClientService
        server_id = self._resolve_server_id(params)
        if server_id is None:
            return ToolResult.ok("Error: provide server_id (from list) or name to disable.")
        svc = McpClientService()
        try:
            server = svc.update_server(server_id, {"enabled": False})
        except LookupError:
            return ToolResult.ok(f"No server found with id={server_id!r}.")
        return ToolResult.ok(f"Server {server['name']!r} disabled. Its tools are no longer discoverable.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_server_id(params: dict) -> str | None:
        """Return a server_id from params, resolving by name if needed."""
        server_id = (params.get("server_id") or "").strip()
        if server_id:
            return server_id
        name = (params.get("name") or "").strip()
        if not name:
            return None
        from services.mcp_client_service import McpClientService
        servers = McpClientService().list_servers()
        for s in servers:
            if s["name"].lower() == name.lower():
                return s["id"]
        return None
