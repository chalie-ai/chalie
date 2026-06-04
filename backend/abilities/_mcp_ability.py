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
# a synthetic member of that hierarchy so MCP calls reuse the one gate.
from abilities._base import Ability

logger = logging.getLogger(__name__)


def _dispatch_mcp(tool_name: str, params: dict) -> dict:
    """Route an _mcp_<server>_<tool> call through McpClientService.

    Policy is enforced by PolicyManager.wrap() in the dispatcher (via the
    _MCPAbility proxy) BEFORE this runs; this only performs the MCP call.
    """
    try:
        from services.mcp_client_service import McpClientService  # noqa: PLC0415
        raw = McpClientService().dispatch_mcp_tool(tool_name, params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[_dispatch_mcp] MCP tool %r failed: %s", tool_name, exc)
        return {"status": "error", "result": f"MCP tool error: {exc}"}

    # McpClientService.dispatch_mcp_tool returns {'text': ...}
    if isinstance(raw, dict) and "text" in raw:
        return {"status": "success", "result": raw["text"]}
    return {"status": "success", "result": str(raw)}


class _MCPAbility(Ability):
    """Synthetic proxy for an _mcp_* tool so MCP calls flow through
    match → wrap → execute exactly like a native ability — gated for the
    first time.

    _SYNTHETIC=True exempts it from __init_subclass__ validation (no
    EXAMPLES/SEARCH_TOOLTIP) and from the registry's boot-time instantiation
    (_all_concrete_subclasses skips it — it cannot be built with no args).
    """

    _SYNTHETIC: ClassVar[bool] = True

    def __init__(self, tool_name: str) -> None:
        self.NAME = tool_name

    def run(self, params: dict) -> dict:
        return _dispatch_mcp(self.NAME, params)
