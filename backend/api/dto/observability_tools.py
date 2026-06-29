"""DTOs for ``GET /system/observability/tools`` — tool usage counts."""

from __future__ import annotations

from datetime import datetime

from .base import DTO


class ToolUsage(DTO):
    """One tool's usage count + last-used timestamp."""

    tool_name: str
    count: int
    last_used_at: str


class ToolUsageList(DTO):
    """Tool usage summary across all tracked tools."""

    generated_at: datetime
    tools: list[ToolUsage]