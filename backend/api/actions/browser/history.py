"""Browser history action — ``POST /api/browser/history/<id>`` goes back, forward or reloads.

Returns once the browser has the request, not once the page loads. Back with
nothing behind (or forward with nothing ahead) is a no-op.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.browser import BrowserHistoryRequest
from exceptions import NotFoundError
from tools.browser import session


class BrowserHistory(Action):
    """Action moving an open tab through its history."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    request_dto: ClassVar[type[Request] | None] = BrowserHistoryRequest
    response_dto = {"post": DocumentedResponse(extras=((404, "No open tab"),))}

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Go ``back``, ``forward`` or ``reload`` in the tab *id*."""
        if self.is_create(id):
            raise NotFoundError("Not found")
        try:
            session.history_tab(str(id), cast(BrowserHistoryRequest, data).action)
        except LookupError as e:
            raise NotFoundError("Not found") from e
        return "", 204
