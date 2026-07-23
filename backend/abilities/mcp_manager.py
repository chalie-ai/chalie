"""
MCP Manager Ability — manage outbound MCP client server connections.

Actions: list | add | enable | disable

Policy tier:
  SYSTEM = True   — always-allowed, never shown in Policy Manager.
  DISCOVERABLE    — the ability leaves ``Ability.DISCOVERABLE`` at its True
                    default, so it is in the global find_tools roster and any
                    discovery-capable channel can surface it when the user asks
                    to connect/manage remote MCP servers.

When `add` runs, it creates the server row AND immediately calls
ping_and_sync() so remote tools are indexed before the LLM's next turn.

Why a separate MCP manager ability instead of a REST-only workflow:
the LLM needs to be able to set up an MCP connection when a user asks
conversationally ("connect to the remote MCP server at …"), without the
user having to open the Brain UI.  This ability handles that path.

Result contract:
  ``run()`` returns ONLY ``ToolResult.ok``/``ToolResult.err`` — the dispatcher
  renders the wire envelope.  Every failure carries a stable kebab-case code so
  a weak model can self-correct without re-reading the schema:

    * unknown action / missing add params → handled by the dispatcher's
      ACTION_REQUIRED pre-gate (``unknown-action`` / ``missing-params``).
    * an unreachable host on add/enable → ``code=mcp-unreachable`` — the row IS
      registered (existing service behaviour); the error is the failed connect
      test, surfaced loudly so the model never treats a dead server as connected.
    * an auth rejection (401/403) on add/enable → ``code=auth-failed``.
    * an unknown server_id/name on enable/disable → ``code=not-found``.

  The error vocabulary (``mcp-unreachable``) is shared verbatim with the
  ``_mcp_*`` passthrough proxy (``abilities/_mcp_ability.py``) so one failure has
  one spelling across the whole MCP surface.  ``list``/``add`` success bodies are
  structured JSON rows, never prose.  This tool has no rich-media card.
"""

import logging
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys

if TYPE_CHECKING:
    from services.mcp_client_service import McpClientService as _McpClientService
from contracts.params.mcp_manager_params_bag import (
    McpManagerAddParams,
    McpManagerDisableParams,
    McpManagerEnableParams,
    McpManagerListParams,
    McpManagerParamsBag,
)
from contracts.params.param_bag import ParamBag

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[MCP MANAGER]"

# Substrings (case-insensitive) in a sync error that mark an auth rejection
# rather than a plain unreachable host — a 401/403 means the endpoint answered
# but refused the credentials, which is a different fix (the token) than a dead
# host (wait for it to come online).
_AUTH_MARKERS = ("401", "403", "unauthorized", "forbidden")


def _classify_sync_error(error: str) -> str:
    lowered = (error or "").lower()
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return "auth-failed"
    return "mcp-unreachable"


class McpManagerAbility(Ability[McpManagerParamsBag]):
    # The typed input contract: the dispatch seam builds the bag via
    # McpManagerParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = McpManagerParamsBag

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
            "add an MCP connection named weather at https://mcp.example.com/mcp",
            "list all connected MCP servers",
            "enable the weather MCP server",
            "disable the weather MCP server",
            "show me which external tools are available from remote MCP servers",
            "set up a connection to an external agent via MCP",
            "what remote tools can you access through MCP?",
        ]

    def get_search_tooltip(self) -> str:
        return "connect to a remote MCP server"

    def get_follow_up(self, tr: ToolResult) -> str:
        """Steer a just-connected server's remote tools into context via find_tools."""
        body = tr.body if isinstance(tr.body, dict) else {}
        if body.get("status") != "online":
            return ""
        name = body.get("name")
        return (
            f"`{name}` is now available. Call `find_tools` with `{name}` and the "
            "action you want to perform to get its tools in context."
        )

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["list", "add", "enable", "disable"],
                "description": (
                    "list: show all configured MCP servers and their status. "
                    "add: register a new remote MCP server (requires name + host). "
                    "enable: re-enable a previously disabled server (by name or "
                    "server_id). "
                    "disable: temporarily disable a server (keeps the row, hides "
                    "its tools from discovery; by name or server_id)."
                ),
            },
            Keys.name_: {
                "type": "string",
                "description": (
                    "For add: required human-readable server label "
                    "(e.g. 'weather', 'home-assistant'). For enable/disable: the "
                    "label to resolve the server by when no server_id is given."
                ),
            },
            Keys.host: {
                "type": "string",
                "description": (
                    "For add: required full URL including port, e.g. "
                    "'https://mcp.example.com/mcp'."
                ),
            },
            Keys.headers: {
                "type": "object",
                "description": (
                    "For add: optional extra HTTP headers as a JSON object "
                    "(e.g. {'Authorization': 'Bearer …'})."
                ),
            },
            Keys.server_id: {
                "type": "string",
                "description": (
                    "For enable/disable: the server UUID from list (takes "
                    "precedence over name)."
                ),
            },
        },
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    # SYSTEM: always-allowed, hidden from Policy Manager (same pattern as memory).
    # Management operations are Chalie self-configuration, not
    # user-data writes — no per-action policy gate is appropriate.
    SYSTEM = True

    # ACTION_REQUIRED drives the dispatcher's pre-gate (BEFORE run()): an unknown
    # action → code=unknown-action with valid=<these keys>; a known action whose
    # required params are missing/blank → ONE code=missing-params naming them all.
    # ALL four actions are keyed — a non-empty map must cover every action or a
    # known action falls through the unknown-action branch (all actions must be mapped).
    # The pre-gate is truthiness-based, which is correct for add: name/host are blank-
    # invalid strings.  The bag's require_str strips and rejects whitespace-only
    # residue ("  " is truthy here but blank after the strip) under the same
    # missing-params code.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "list": (),
        "add": (Keys.name_, Keys.host),
        "enable": (),
        "disable": (),
    }

    def run(self, params: McpManagerParamsBag) -> ToolResult:
        # The router factory only ever yields the four leaves; the isinstance
        # chain is the narrowing that hands each handler its exact leaf type.
        # The trailing branch catches only a hand-built foreign subclass (or
        # the bare router) — loudly.
        if isinstance(params, McpManagerListParams):
            return self._do_list(params)
        if isinstance(params, McpManagerAddParams):
            return self._do_add(params)
        if isinstance(params, McpManagerEnableParams):
            return self._do_enable(params)
        if isinstance(params, McpManagerDisableParams):
            return self._do_disable(params)
        return ToolResult.err(
            f"Unknown mcp_manager params bag: {type(params).__name__}.",
            code="unknown-action",
            valid=("list", "add", "enable", "disable"),
        )

    # ── Sub-action handlers ───────────────────────────────────────────────────

    def _do_list(self, params: McpManagerListParams) -> ToolResult:
        """Empty inventory is SUCCESS (``[]``, ``count=0``) — never an error."""
        from services.mcp_client_service import McpClientService  # noqa: PLC0415
        svc = McpClientService()
        rows = [self._server_row(svc, s) for s in svc.list_servers()]
        return ToolResult.ok(rows, count=len(rows))

    def _do_add(self, params: McpManagerAddParams) -> ToolResult:
        """The row is registered BEFORE the connect test — an unreachable host still
        persists the connection, but a failed ping is surfaced as an ERROR."""
        from services.mcp_client_service import McpClientService  # noqa: PLC0415
        name = params.name
        host = params.host
        headers = params.headers

        svc = McpClientService()
        server = svc.add_server(name=name, host=host, headers=headers, enabled=True)
        server_id = cast("str", server["id"])

        # Trigger an immediate sync so tools are discoverable in this turn.
        sync = svc.ping_and_sync(server_id)
        if sync["reachable"]:
            # Build vector embeddings for the newly-synced tools so semantic
            # queries can reach them immediately.  Add-only — never called on
            # heartbeat or enable so the 15-min sync path stays zero-cost.
            svc.embed_server_tools(server_id)
            logger.info(
                "%s Added server %r — online, %d tools",
                _LOG_PREFIX, name, sync["tool_count"],
            )
            return ToolResult.ok(
                {
                    "id": server_id,
                    "name": name,
                    "url": host,
                    "status": "online",
                    "tool_count": sync["tool_count"],
                },
                count=sync["tool_count"],
            )

        code = _classify_sync_error(cast("str", sync.get("error") or ""))
        logger.info(
            "%s Added server %r — registered but %s (%s)",
            _LOG_PREFIX, name, code, sync.get("error"),
        )
        return self._unreachable_error(code, name)

    def _do_enable(self, params: McpManagerEnableParams) -> ToolResult:
        """Enabling a dead server is reported as an error (mcp-unreachable /
        auth-failed): the row IS enabled, but the model must know the tools are
        not yet available so it never assumes a live connection."""
        from services.mcp_client_service import McpClientService  # noqa: PLC0415
        svc = McpClientService()
        resolved = self._resolve(svc, params.server_id, params.name)
        if isinstance(resolved, ToolResult):
            return resolved
        server_id, name = resolved

        svc.update_server(server_id, {"enabled": True})
        sync = svc.ping_and_sync(server_id)
        if sync["reachable"]:
            logger.info("%s Enabled server %r — online, %d tools",
                        _LOG_PREFIX, name, sync["tool_count"])
            return ToolResult.ok({
                "id": server_id,
                "name": name,
                "status": "online",
                "tool_count": sync["tool_count"],
            })

        code = _classify_sync_error(cast("str", sync.get("error") or ""))
        logger.info("%s Enabled server %r — %s (%s)",
                    _LOG_PREFIX, name, code, sync.get("error"))
        return self._unreachable_error(code, name, enabled=True)

    def _do_disable(self, params: McpManagerDisableParams) -> ToolResult:
        from services.mcp_client_service import McpClientService  # noqa: PLC0415
        svc = McpClientService()
        resolved = self._resolve(svc, params.server_id, params.name)
        if isinstance(resolved, ToolResult):
            return resolved
        server_id, name = resolved

        svc.update_server(server_id, {"enabled": False})
        logger.info("%s Disabled server %r", _LOG_PREFIX, name)
        return ToolResult.ok({"id": server_id, "name": name, "enabled": False})

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _server_row(svc: "_McpClientService", server: dict[str, object]) -> dict[str, object]:
        server_id = cast("str", server["id"])
        return {
            "id": server_id,
            "name": server["name"],
            "url": server["host"],
            "status": server["status"],
            "enabled": server["enabled"],
            "tool_count": len(svc.get_server_tools(server_id)),
        }

    @staticmethod
    def _resolve(svc: "_McpClientService", server_id: str, name: str) -> "tuple[str, str] | ToolResult":
        """Returns an error ToolResult when no target was given
        (``missing-params``) or the target does not exist (``not-found``), so the
        caller only proceeds on a real server. ``server_id`` and ``name`` are
        already stripped by the bag; empty strings mean "not provided"."""
        if not server_id and not name:
            return ToolResult.err(
                "Provide a server to act on.",
                code="missing-params",
                hint="provide server_id (from list) or name",
            )

        if server_id:
            server = svc.get_server(server_id)
            if server is not None:
                return cast("str", server["id"]), cast("str", server["name"])
            return ToolResult.err(
                f"No MCP server with id {server_id!r}.",
                code="not-found",
                hint="use action=list to see configured servers",
            )

        for s in svc.list_servers():
            if cast("str", s["name"]).lower() == name.lower():
                return cast("str", s["id"]), cast("str", s["name"])
        return ToolResult.err(
            f"No MCP server named {name!r}.",
            code="not-found",
            hint="use action=list to see configured servers",
        )

    @staticmethod
    def _unreachable_error(code: str, name: str, *, enabled: bool = False) -> ToolResult:
        """The hint states the row WAS persisted (and, for enable, that it IS
        enabled) so the model knows the connection exists and the tools will sync
        once the server is reachable."""
        registered_note = (
            f"the connection to {name!r} is "
            f"{'registered and enabled' if enabled else 'registered'}; "
            "tools will sync when it comes online"
        )
        if code == "auth-failed":
            hint = (
                f"check the headers/Authorization token for server {name!r} — "
                f"{registered_note}"
            )
            message = (
                f"MCP server {name!r} rejected the connection (authentication "
                f"failed). {registered_note.capitalize()}."
            )
        else:
            hint = f"the MCP server {name!r} did not respond; {registered_note}"
            message = (
                f"MCP server {name!r} is unreachable; the connect test failed. "
                f"{registered_note.capitalize()}."
            )
        return ToolResult.err(message, code=code, hint=hint, server=name)
