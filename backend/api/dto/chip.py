"""DTO for a tool-call chip rendered under an assistant message."""

from __future__ import annotations

from .base import DTO


class DelegateRef(DTO):
    """The delegate turn a tool call spawned, addressed the way the threads
    read takes it: its full channel plus its ``turn_id``."""

    channel: str
    turn_id: int


class Chip(DTO):
    """Per-row tool chip: the ability name, its persisted act-summary, and the
    persisted lifecycle ``state`` (started/done/error) + ``ended_at`` so a
    refetched turn re-renders the true pill state — an error stays red on reload
    rather than downgrading to a neutral chip. ``delegate`` names the delegate
    turn the call spawned, when it wrote a transcript, so the pill opens it."""

    tool_name: str
    summary: str
    state: str
    ended_at: str | None = None
    delegate: DelegateRef | None = None
