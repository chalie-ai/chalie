"""McpToolsAbility — list and activate tools from connected MCP servers."""

import logging
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.mcp_tools_params_bag import McpToolsParamsBag
from contracts.params.param_bag import ParamBag

logger = logging.getLogger(__name__)


class McpToolsAbility(Ability):
    """Gateway to MCP server tools: list available tools or activate them for the turn.

    Use action ``"list"`` to see every connected MCP server and its tools, then
    action ``"activate"`` with exact tool names to enable them for this turn."""

    PARAMS: ClassVar[type[ParamBag] | None] = McpToolsParamsBag

    DISCOVERABLE: ClassVar[bool] = True

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["list", "activate"],
                "description": "Either 'list' to see all MCP tools or 'activate' to enable specific ones.",
            },
            Keys.tools: {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tool names to activate (required when action is 'activate').",
            },
        },
        "required": [Keys.action],
    }

    def get_name(self) -> str:
        return "mcp_tools"

    def get_summary(self) -> str:
        return (
            "List and activate tools from connected MCP servers. Use action 'list' "
            "to see all available tools, then 'activate' with exact tool names to "
            "enable them for this turn."
        )

    def get_examples(self) -> list[str]:
        return [
            "Show me all available MCP tools",
            "List the tools from my GitHub server",
            "Activate the _mcp_github_list_issues tool",
            "Show me what tools are connected",
            "Enable the _mcp_github_add_comment tool for me",
            "List all MCP servers and their tools",
            "Activate _mcp_github_create_issue and _mcp_github_list_issues",
            "What MCP tools can I use right now",
        ]

    def get_search_tooltip(self) -> str:
        return "list and activate tools from connected MCP servers"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def get_follow_up(self, tr: ToolResult) -> str:
        """Announce the freshly-activated tools by name — live for this turn."""
        body = tr.body
        if not isinstance(body, dict):
            return ""
        activated = body.get("activated")
        if not isinstance(activated, list):
            return ""
        names: list[str] = []
        for n in activated:
            if isinstance(n, dict):
                name = cast("dict[str, str]", n).get("name")
                if isinstance(name, str):
                    names.append(name)
        if not names:
            return ""
        joined = ", ".join(f"`{n}`" for n in names)
        verb = "is" if len(names) == 1 else "are"
        return f"{joined} {verb} now available and can be called."

    def run(self, params: McpToolsParamsBag) -> ToolResult:
        """Execute the mcp_tools action: list or activate MCP server tools."""
        if params.action == "list":
            return self._handle_list()
        return self._handle_activate(params.tools)

    def _handle_list(self) -> ToolResult:
        """Return every configured MCP server and its tools."""
        try:
            from services.mcp_client_service import McpClientService
            mcp_service = McpClientService()

            servers = mcp_service.list_servers()
            server_dicts: list[dict[str, object]] = []
            total_tools = 0

            for server in servers:
                server_info: dict[str, object] = {
                    "name": server["name"],
                    "status": server["status"],
                }

                # Only include tools for enabled + online servers
                if server.get("enabled") and server["status"] == "online":
                    server_id = cast(str, server["id"])
                    tools = mcp_service.get_server_tools(server_id)
                    tool_rows = [
                        {"name": t["tool_name"], "summary": t["summary"]}
                        for t in tools
                    ]
                    server_info["tools"] = tool_rows
                    total_tools += len(tool_rows)

                server_dicts.append(server_info)

            body = {"servers": server_dicts}
            return ToolResult.ok(
                body,
                server_count=len(server_dicts),
                tool_count=total_tools,
            )

        except Exception as exc:
            logger.error("[MCP_TOOLS] List failed: %s", exc)
            return ToolResult.err(
                f"Failed to list MCP tools: {exc}",
                code="internal-error",
            )

    def _handle_activate(self, tool_names: list[str]) -> ToolResult:
        """Activate specific MCP tools by appending them to mp.active_tools."""
        try:
            from services.mcp_client_service import McpClientService
            mcp_service = McpClientService()

            activated = []
            not_found = []

            # Get all online MCP tool names for validation
            online_tools = mcp_service.get_online_mcp_tool_names()

            for tool_name in tool_names:
                if tool_name in online_tools:
                    # Append to active_tools
                    self._append_active([tool_name])

                    # Get tool summary
                    tool_schema = mcp_service.get_tool_schema(tool_name)
                    summary = tool_schema.get("description", "") if tool_schema else ""

                    activated.append({"name": tool_name, "summary": summary})
                else:
                    not_found.append(
                        f"'{tool_name}' is not an online MCP tool. "
                        "Use action 'list' to see the exact tool names."
                    )

            body = {
                "activated": activated,
                "not_found": not_found,
            }

            return ToolResult.ok(
                body,
                activated_count=len(activated),
                not_found_count=len(not_found),
            )

        except Exception as exc:
            logger.error("[MCP_TOOLS] Activate failed: %s", exc)
            return ToolResult.err(
                f"Failed to activate MCP tools: {exc}",
                code="internal-error",
            )
