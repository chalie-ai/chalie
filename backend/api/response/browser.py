"""Response DTOs for the browser's live tabs."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .policies import PermissionOriginResponse
from .response import Response


class BrowserCertificateErrorResponse(Response):
    """A page the browser refused for its certificate; the tab stays on the page it was on."""

    host: str
    port: int
    error: str = Field(description="The browser's reason, e.g. 'ERR_CERT_AUTHORITY_INVALID'.")
    url: str = Field(description="The url refused: where proceeding goes.")


class BrowserTabResponse(Response):
    """One open tab of the browser."""

    id: str
    url: str = Field(description="The page's url; 'about:blank' while a tab has nothing loaded.")
    title: str
    opener: Literal["chalie", "user"] = Field(description="'chalie' for a browsing run's page, 'user' for a tab the user opened or a popup kept as its own tab.")
    origin: PermissionOriginResponse | None = Field(description="The interactive turn of the browsing run the tab belongs to: the run's own page, or a popup kept from it. Null for the user's own tabs, and for a run on no conversation the user sees.")
    certificate_error: BrowserCertificateErrorResponse | None = Field(default=None, description="The page the tab was sent to, refused for its certificate, until the tab loads another page or goes back. Null otherwise.")


class BrowserTabsResponse(Response):
    """Every open tab, in the order they opened."""

    tabs: list[BrowserTabResponse]


class BrowserTabCreatedResponse(Response):
    """The tab a create just opened."""

    tab: BrowserTabResponse
