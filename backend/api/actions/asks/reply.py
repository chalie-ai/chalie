"""Reply to an ask — ``POST /api/asks/reply/<ask_id>`` answers it; the first reply wins."""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.asks import AskReplyRequest
from exceptions import AskReplyInvalid, EndpointError, NotFoundError
from services.ask_user import AskUser


class AsksReply(Action):
    """Action answering an open ask."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    request_dto = AskReplyRequest
    response_dto = {"post": DocumentedResponse(extras=((404, "No open ask"),))}

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Answer ask *id* with the body's value: a bool for a permission ask, null for a hand-off.

        An ask that is not open — already answered, dismissed or ended — is a
        404. A value wrong for the ask's kind is a 400 and leaves it open.
        """
        if self.is_create(id):
            raise NotFoundError("Not found")
        dto = cast(AskReplyRequest, data)
        try:
            replied = AskUser.reply(str(id), dto.value)
        except AskReplyInvalid as exc:
            raise EndpointError(str(exc)) from exc
        if not replied:
            raise NotFoundError("Not found")
        return "", 204
