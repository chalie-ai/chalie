"""Outbound DTO for the tool-calls endpoint group."""

from __future__ import annotations

from api.dto.base import DTO
from api.dto.chip import DelegateRef

from .response import Response


class Page(DTO):
    """One page a browsing call reached: its url, and its title (``""`` when
    the call's result gives none)."""

    url: str
    title: str


class ToolCallDetail(Response):
    """One tool call's full record — the per-call read behind an opened row. ``params``
    is the object the call was made with. ``result`` is the stored result text
    as the model read it: ``""`` while the call is still running, None only
    when none was stored. ``delegate`` names the child turn a delegate call
    spawned, as on the chip. ``pages`` lists, in order, the pages a browsing
    call reached — a ``browser`` call's own page, every page of a
    ``web_browse`` call's child turn — and is empty for every other tool."""

    id: int
    tool_name: str
    params: dict[str, object]
    result: str | None
    summary: str
    state: str
    created_at: str
    ended_at: str | None = None
    delegate: DelegateRef | None = None
    pages: list[Page]
