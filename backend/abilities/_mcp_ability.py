"""Synthetic ability proxy for MCP (``_mcp_*``) tool calls.

An ``_mcp_<server>_<tool>`` name has no native Ability in the registry, so it is
wrapped in a synthetic ``_MCPAbility`` whose ``run()`` forwards to
``McpClientService``. Routing it through an Ability subclass means MCP calls take
the exact same dispatch path (match → policy gate → execute → record) as native
tools instead of a parallel one.

The dispatcher returns an ``_MCPAbility`` for any ``_mcp_*`` name; nothing else
constructs it.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, ClassVar, cast

# Ability is the shared base every dispatchable tool subclasses; _MCPAbility is
# a synthetic member of that hierarchy so MCP calls reuse the one gate AND the
# one schema assembler (get_input_schema → framework-field injection).
from abilities._ability import Ability
from abilities._result import ToolResult
from exceptions import McpServerUnreachable, McpToolUnknown

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


def _dispatch_mcp(tool_name: str, params: "dict[str, object]") -> ToolResult:
    """Route an ``_mcp_<server>_<tool>`` call through ``McpClientService`` and map
    the remote outcome onto the ``ToolResult`` contract.

    Policy is enforced by ``PolicyManager.wrap()`` in the dispatcher (via the
    ``_MCPAbility`` proxy) BEFORE this runs; this only performs the MCP call.

    The shaping is the service's job (``dispatch_mcp_tool`` returns
    ``{"server", "is_error", "body"}`` with ``body`` already str-or-structured);
    this proxy only chooses ``ok``/``err`` and the stable code:

    * ``McpServerUnreachable`` (transport/connect/timeout) → ``code=mcp-unreachable``
      NAMING the server, so the model knows which endpoint is down.
    * ``McpToolUnknown`` (no enabled server matches the name) → ``code=mcp-unknown-tool``.
    * a remote ``isError=true`` → ``code=mcp-tool-error`` with the error content as
      the message — an MCP-level failure is never dressed up as success.
    * otherwise → ``ToolResult.ok(body)`` with ``server=<name>`` flat meta.

    No exceptions are swallowed beyond the two anticipated MCP failure modes; any
    other exception propagates to the dispatcher's ``_run`` guard
    (``code=unhandled-exception``) rather than being masked here.
    """
    from services.mcp_client_service import McpClientService  # noqa: PLC0415

    try:
        result = McpClientService().dispatch_mcp_tool(tool_name, params)
    except McpServerUnreachable as exc:
        logger.warning("[_dispatch_mcp] %s", exc)
        return ToolResult.err(
            str(exc),
            code="mcp-unreachable",
            hint=f"The MCP server {exc.server_name!r} did not respond; "
                 "check it is online before retrying.",
            server=exc.server_name,
        )
    except McpToolUnknown as exc:
        logger.warning("[_dispatch_mcp] %s", exc)
        return ToolResult.err(
            str(exc),
            code="mcp-unknown-tool",
            hint="No enabled MCP server exposes this tool; "
                 "re-discover available tools instead of retrying this name.",
        )

    body = result["body"]
    server = cast(str, result["server"])
    if result["is_error"]:
        message = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        return ToolResult.err(
            message,
            code="mcp-tool-error",
            hint=f"The remote tool on server {server!r} reported an error; "
                 "adjust the arguments or report the failure.",
            server=server,
        )
    return ToolResult.ok(cast("str | dict[str, object] | list[object]", body), server=server)


class _MCPAbility(Ability):
    """Synthetic proxy for an _mcp_* tool so MCP calls flow through
    match → wrap → execute AND name → description → input_schema assembly
    exactly like a native ability — both gated and schema-built for the first
    time through the single path.

    Its metadata getters source the description / parameters from the remote
    tool schema (lazily fetched + cached via McpClientService); the search-facing
    getters return empty because a synthetic proxy is never indexed.

    _SYNTHETIC=True exempts it from __init_subclass__ metadata validation (no
    EXAMPLES/SEARCH_TOOLTIP shape check) and from the registry's boot-time
    instantiation (_all_concrete_subclasses skips it — it takes a tool_name).
    """

    _SYNTHETIC: ClassVar[bool] = True

    def __init__(self, tool_name: str, mp: "MessageProcessor | None" = None) -> None:
        super().__init__(mp)
        self._tool_name = tool_name
        self._remote: "dict[str, object] | None" = None
        self._fetched = False

    def remote_schema(self) -> "dict[str, object] | None":
        """The remote MCP tool schema (``{'name','description','input_schema'}``),
        fetched once and cached. None when the server exposes no such tool —
        build_tools treats that as 'skip this tool'."""
        if not self._fetched:
            from services.mcp_client_service import McpClientService  # noqa: PLC0415
            self._remote = McpClientService().get_tool_schema(self._tool_name)
            self._fetched = True
        return self._remote

    def get_name(self) -> str:
        return self._tool_name

    def get_summary(self) -> str:
        remote = self.remote_schema()
        return cast(str, (remote or {}).get("description", ""))

    def get_examples(self) -> list[str]:
        return []

    def get_search_tooltip(self) -> str:
        return ""

    def get_parameters(self) -> "dict[str, object]":
        remote = self.remote_schema()
        return cast("dict[str, object]", (remote or {}).get("input_schema", {}))

    def run(self, params: "dict[str, object]") -> ToolResult:
        return _dispatch_mcp(self._tool_name, params)
