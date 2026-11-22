"""Browser proceed action — ``POST /api/browser/proceed/<id>`` is the user's choice on a certificate warning.

The tab's page was refused for its certificate. Proceeding lets the browser
load that site — its host and port — despite the certificate, in every tab,
until the browser closes, and sends the tab to the url refused. Returns once
the browser has the request, not once the page loads.

The body names the site the warning showed: a tab whose refusal has changed
since — another site, or none — is a 409, and nothing is allowed. Only a
signed-in user's session reaches this; no run's verb does.
"""

from __future__ import annotations

import logging
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.browser import BrowserProceedRequest
from exceptions import ConflictError, NotFoundError
from tools.browser import session

logger = logging.getLogger(__name__)


class BrowserProceed(Action):
    """Action proceeding to a site despite its certificate, on the user's word."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    request_dto: ClassVar[type[Request] | None] = BrowserProceedRequest
    response_dto = {"post": DocumentedResponse(extras=(
        (404, "No open tab"),
        (409, "The tab no longer shows a certificate warning for that site"),
    ))}

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Proceed to the site the tab *id*'s certificate warning names."""
        if self.is_create(id):
            raise NotFoundError("Not found")
        body = cast(BrowserProceedRequest, data)
        try:
            proceeded: bool | None = session.proceed_tab(str(id), body.host, body.port)
        except LookupError:
            proceeded = None
        outcome = "not-found" if proceeded is None else "ok" if proceeded else "no-warning"
        logger.info("[BROWSER] panel proceed tab=%s site=%s:%s outcome=%s", id, body.host, body.port, outcome)
        if proceeded is None:
            raise NotFoundError("Not found")
        if not proceeded:
            raise ConflictError("The tab no longer shows a certificate warning for that site")
        return "", 204
