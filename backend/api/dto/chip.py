"""DTO for a tool-call chip rendered under an assistant message."""

from __future__ import annotations

from .base import DTO


class Chip(DTO):
    """Per-row tool chip: the ability name and its persisted act-summary."""

    tool_name: str
    summary: str
