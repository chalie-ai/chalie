"""Open asks — ``GET /api/asks/open`` lists every ask still waiting on the user."""

from __future__ import annotations

from typing import ClassVar

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.asks import OpenAskResponse
from api.response.response import Response
from services.ask_user import AskUser


class AsksOpen(Action):
    """Action listing the asks still waiting for a reply."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get"})
    # id is ignored (lists every open ask, no per-id lookup to miss), so get
    # can never emit 404.
    response_dto = {
        "get": DocumentedResponse(OpenAskResponse, listing=True, not_found=False),
    }

    def get(self, id: int | str) -> ResponseReturnValue:
        """List every open ask, oldest first.

        Each entry carries the fields of its ``ask_opened`` frame, so a client
        rebuilds its asks from here on every connect and the ``ask_opened`` /
        ``ask_closed`` frames keep it current after that.
        """
        dtos = [OpenAskResponse.model_validate(ask.to_dict()) for ask in AskUser.open_asks()]
        total = len(dtos)
        return Response.listing(dtos, page=1, limit=max(total, 1), total=total)
