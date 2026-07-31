"""
MCP Manager Ability — manage outbound MCP client server connections.

Actions: list | add | enable | disable

Policy tier:
  SYSTEM = True        — always-allowed, never shown in Policy Manager.
  DISCOVERABLE = False — pinned directly on every discovery-capable channel via
                         ``DEFAULT_ALWAYS_AVAILABLE``, so it is always already in
                         context. ``find_tools`` is itself pinned only through
                         that same roster, which means discovery could never
                         reach a channel this tool is absent from — leaving it
                         discoverable would only ever offer the model a tool it
                         can already call.

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

    NAME: ClassVar[str] = "mcp_manager"

    # Pinned, never discovered — see the Policy tier note in the module docstring.
    # Neither CATEGORY nor SEARCHABLE_AS follows from that, and both are dropped:
    # a category is the heading a tool renders under in the find_tools menu, and
    # SEARCHABLE_AS feeds AbilityRegistry.discovery_aliases(), which only walks
    # the discoverable roster. This tool is in neither, so both would be dead.
    DISCOVERABLE: ClassVar[bool] = False

    # The static half of the description. This is the ONLY text rendered
    # off-spine (``self.mp is None`` — introspection / search-index build), so the
    # indexed summary stays byte-stable regardless of what this box has connected.
    _BASE_SUMMARY: ClassVar[str] = (
        "Connect Chalie to a remote MCP server so its tools become available. "
        "Use to add, list, enable, or disable outbound MCP server connections."
    )

    # Stated explicitly rather than by omission: an empty inventory is the answer
    # to "what am I connected to?", so the model never spends a `list` call to
    # learn there is nothing there.
    _NONE_CONNECTED: ClassVar[str] = "No MCP servers are connected."

    def get_summary(self) -> str:
        inventory = self._inventory_line()
        return f"{self._BASE_SUMMARY}\n{inventory}" if inventory else self._BASE_SUMMARY

    def _inventory_line(self) -> str:
        """Server NAMES only — the inventory answers "what am I connected to?";
        ``mcp_tools`` answers "what can each one do?" per server. Listing tools
        here would put the entire remote surface into every single request.

        Disabled servers are excluded: ``disable`` hides a server's tools from
        discovery, so the row is registered but not connected. Order is
        ``list_servers``' ``created_at`` order, so this line and the ``list``
        action can never disagree about the inventory.
        """
        if self.mp is None:
            return ""  # off-spine: deterministic base text only (see Ability docstring)

        from services.mcp_client_service import McpClientService  # noqa: PLC0415

        try:
            names = [str(s["name"]) for s in McpClientService().list_servers() if s["enabled"]]
        except Exception as exc:
            # This getter runs inside the descriptor assembler on EVERY step, so a
            # DB hiccup must degrade to the static text rather than fail the turn
            # (same containment as FindToolsAbility._summary_for). Returning ""
            # rather than _NONE_CONNECTED: unknown is not the same as none.
            logger.warning("%s inventory lookup failed: %s: %s", _LOG_PREFIX, type(exc).__name__, exc)
            return ""

        if not names:
            return self._NONE_CONNECTED
        return f"Connected MCP Servers: {', '.join(names)}"

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
        """Steer a just-connected server's remote tools into context via mcp_tools."""
        from abilities.mcp_tools import McpToolsAbility  # noqa: PLC0415

        body = tr.body if isinstance(tr.body, dict) else {}
        if body.get("status") != "online":
            return ""
        name = body.get("name")
        return (
            f"`{name}` is now available. Call `{McpToolsAbility.NAME}` with action `list` to "
            "see its tools, then `activate` the ones you need."
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
        """Empty inventory is a loud ``code=no-results`` — never a quiet success
        with zero rows."""
        from services.mcp_client_service import McpClientService  # noqa: PLC0415
        svc = McpClientService()
        rows = [self._server_row(svc, s) for s in svc.list_servers()]
        if not rows:
            return ToolResult.no_results()
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
