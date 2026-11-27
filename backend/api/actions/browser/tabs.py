"""Browser tabs action — the live view's list of open tabs.

Covers:
- GET    /api/browser/tabs       → every open tab; never starts the browser
- POST   /api/browser/tabs       → open a user tab, on ``url`` when given
- DELETE /api/browser/tabs/<id>  → close a user tab (a browsing run's tab
  closes with its run, so closing one here is a 409)
"""

from __future__ import annotations

import logging
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.browser import BrowserOpenTabRequest
from api.response.browser import BrowserTabCreatedResponse, BrowserTabResponse, BrowserTabsResponse
from configs.enums.param_key import Keys
from contracts.params.browser_params_bag import BrowserOpenParams
from exceptions import ConflictError, EndpointError, NotFoundError
from tools.browser import session, tabs

logger = logging.getLogger(__name__)


class BrowserTabs(Action):
    """Action listing, opening and closing the browser's tabs."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get", "post", "delete"})
    request_dto: ClassVar[type[Request] | None] = BrowserOpenTabRequest
    _post_may_create: ClassVar[bool] = True
    response_dto = {
        "get": DocumentedResponse(BrowserTabsResponse),
        "post": DocumentedResponse(BrowserTabCreatedResponse),
        "delete": DocumentedResponse(extras=((409, "A browsing run's tab closes when its run ends"),)),
    }

    def get(self, id: int | str) -> ResponseReturnValue:
        """List every open tab, in the order they opened; empty while the browser is down."""
        self.require_create_sentinel(id)
        return BrowserTabsResponse.model_validate({"tabs": tabs.listing()}).single()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Open a user tab — blank, or loading ``url`` — starting the browser if it is down."""
        self.require_create_sentinel(id)
        url = cast(BrowserOpenTabRequest, data).url
        if url is not None:
            params = BrowserOpenParams.from_params({Keys.url: url})
            if not isinstance(params, BrowserOpenParams):
                raise EndpointError("url must not be blank")
            url = params.url
        tab = BrowserTabResponse.model_validate(session.open_tab(url))
        # The page asked for: the tab is listed before the browser reports it there.
        logger.info("[BROWSER] panel opened tab=%s url=%s", tab.id, url or tab.url)
        return BrowserTabCreatedResponse(tab=tab).single(), 201

    def delete(self, id: int | str) -> ResponseReturnValue:
        """Close the user tab *id*."""
        tab = tabs.get(str(id))
        if tab is not None and tab.opener == "chalie":
            logger.info("[BROWSER] panel close tab=%s url=%s outcome=conflict", tab.id, tab.url)
            raise ConflictError("A browsing run's tab closes when its run ends")
        if tab is not None:
            try:
                session.close_tab(tab.id)
            except LookupError:
                tab = None  # it closed since it was looked up
        logger.info("[BROWSER] panel close tab=%s url=%s outcome=%s", id, tab.url if tab else "-",
                    "ok" if tab else "not-found")
        if tab is None:
            raise NotFoundError("Not found")
        return "", 204
