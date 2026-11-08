"""Response DTOs for the browser's live tabs."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .policies import PermissionOriginResponse
from .response import Response


class BrowserHandoffResponse(Response):
    """A browsing run waiting on the user to finish something on its tab — the
    fields of the ``browser_handoff`` socket frame, so a client restoring over
    REST shows the same prompt it missed."""

    id: str
    reason: str
    origin: PermissionOriginResponse
    asked_at: str = Field(description="When the run handed off, ISO-8601 UTC with its offset (e.g. '2026-10-07T06:32:11+00:00').")


class BrowserTabResponse(Response):
    """One open tab of the browser."""

    id: str
    url: str = Field(description="The page's url; 'about:blank' while a tab has nothing loaded.")
    title: str
    opener: Literal["chalie", "user"] = Field(description="'chalie' for a browsing run's page, 'user' for a tab the user opened or a popup kept as its own tab.")
    handoff: BrowserHandoffResponse | None = None


class BrowserTabsResponse(Response):
    """Every open tab, in the order they opened."""

    tabs: list[BrowserTabResponse]


class BrowserTabCreatedResponse(Response):
    """The tab a create just opened."""

    tab: BrowserTabResponse
