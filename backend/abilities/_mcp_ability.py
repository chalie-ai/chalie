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

import logging
from typing import ClassVar

# Ability is the shared base every dispatchable tool subclasses; _MCPAbility is
# a synthetic member of that hierarchy so MCP calls reuse the one gate AND the
# one schema assembler (get_input_schema → framework-field injection).
from abilities._ability import Ability
from abilities._result import ToolResult

logger = logging.getLogger(__name__)


def _dispatch_mcp(tool_name: str, params: dict) -> ToolResult:
    """Route an _mcp_<server>_<tool> call through McpClientService.

    Policy is enforced by PolicyManager.wrap() in the dispatcher (via the
    _MCPAbility proxy) BEFORE this runs; this only performs the MCP call.
    Mechanical ToolResult wrap (TKT-895 redesigns the structured-content mapping).
    """
    try:
        from services.mcp_client_service import McpClientService  # noqa: PLC0415
        raw = McpClientService().dispatch_mcp_tool(tool_name, params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[_dispatch_mcp] MCP tool %r failed: %s", tool_name, exc)
        return ToolResult.err(f"MCP tool error: {exc}", code="mcp-error")

    # McpClientService.dispatch_mcp_tool returns {'text': ...}
    if isinstance(raw, dict) and "text" in raw:
        return ToolResult.ok(raw["text"])
    return ToolResult.ok(str(raw))


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

    def __init__(self, tool_name: str, mp: "object | None" = None) -> None:
        super().__init__(mp)
        self._tool_name = tool_name
        self._remote: "dict | None" = None
        self._fetched = False

    def remote_schema(self) -> "dict | None":
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
        return (remote or {}).get("description", "")

    def get_examples(self) -> list[str]:
        return []

    def get_search_tooltip(self) -> str:
        return ""

    def get_parameters(self) -> dict:
        remote = self.remote_schema()
        return (remote or {}).get("input_schema", {})

    def run(self, params: dict) -> ToolResult:
        return _dispatch_mcp(self._tool_name, params)
