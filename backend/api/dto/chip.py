"""DTO for a tool-call chip rendered under an assistant message."""

from __future__ import annotations

from .base import DTO


class DelegateRef(DTO):
    """The delegate turn a tool call spawned, addressed the way the threads
    read takes it: its full channel plus its ``turn_id``. ``working`` says
    whether that turn is still running as of this read — a subtask outlives
    the call that started it, so the call's own state cannot say."""

    channel: str
    turn_id: int
    working: bool


class Chip(DTO):
    """Per-row tool chip: the ability name, the ``params`` it was called with
    (the same parsed object the live frame carries), its persisted
    act-summary, and the persisted lifecycle ``state`` (started/done/error) +
    ``ended_at`` so a refetched turn re-renders the true row state — an error
    stays red on reload rather than downgrading to a neutral chip. ``delegate``
    names the delegate turn the call spawned, when it wrote a transcript, so
    its row opens it. ``id`` is the tool_calls row id — the id the live
    tool-call frame keys its row on, and the one the per-call read takes for
    the call's result."""

    id: int
    tool_name: str
    params: dict[str, object]
    summary: str
    state: str
    ended_at: str | None = None
    delegate: DelegateRef | None = None
