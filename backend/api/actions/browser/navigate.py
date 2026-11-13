"""Browser navigate action — ``POST /api/browser/navigate/<id>`` points a tab at a url.

Returns once the browser has the request, not once the page loads; the tab's
``meta`` frames and ``browser_tabs_changed`` report where it lands.
"""

from __future__ import annotations

import logging
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.browser import BrowserNavigateRequest
from configs.enums.param_key import Keys
from contracts.params.browser_params_bag import BrowserOpenParams
from exceptions import EndpointError, NotFoundError
from tools.browser import session

logger = logging.getLogger(__name__)


class BrowserNavigate(Action):
    """Action pointing an open tab at a url."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    request_dto: ClassVar[type[Request] | None] = BrowserNavigateRequest
    response_dto = {"post": DocumentedResponse(extras=((404, "No open tab"),))}

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Point the tab *id* at ``url``."""
        if self.is_create(id):
            raise NotFoundError("Not found")
        params = BrowserOpenParams.from_params({Keys.url: cast(BrowserNavigateRequest, data).url})
        if not isinstance(params, BrowserOpenParams):
            raise EndpointError("url must not be blank")
        try:
            session.navigate_tab(str(id), params.url)
        except LookupError as e:
            logger.info("[BROWSER] panel navigate tab=%s url=%s outcome=not-found", id, params.url)
            raise NotFoundError("Not found") from e
        logger.info("[BROWSER] panel navigate tab=%s url=%s outcome=ok", id, params.url)
        return "", 204
