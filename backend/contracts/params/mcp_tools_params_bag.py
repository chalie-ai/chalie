"""McpToolsParamsBag — the typed input contract of the ``mcp_tools`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class McpToolsParamsBag(ParamBag):
    """``action`` — ``"list"`` or ``"activate"``; ``tools`` — list of tool names
    (required non-empty when action is ``"activate"``, else empty list)."""

    action: str
    tools: list[str]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        action = cls.require_str(params, Keys.action)
        if isinstance(action, ToolResult):
            return action
        if action not in ("list", "activate"):
            return ToolResult.err(
                f"'{Keys.action}' must be 'list' or 'activate'.",
                code="invalid-param",
                hint=f"pass '{Keys.action}' as 'list' or 'activate'.",
                valid=("list", "activate"),
            )
        if action == "activate":
            tools_result = cls.require_str_list(params, Keys.tools)
            if isinstance(tools_result, ToolResult):
                return tools_result
            tools = tools_result
        else:
            tools = []
        return cls(action=action, tools=tools)