"""Dismiss an ask — ``POST /api/asks/dismiss/<ask_id>`` closes it unanswered."""

from __future__ import annotations

from typing import ClassVar

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from exceptions import NotFoundError
from services.ask_user import AskUser


class AsksDismiss(Action):
    """Action closing an open ask without an answer."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    response_dto = {"post": DocumentedResponse(extras=((404, "No open ask"),))}

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Dismiss ask *id*: its caller reads it as unanswered.

        An ask that is not open — already answered, dismissed or ended — is a 404.
        """
        if self.is_create(id) or not AskUser.cancel(str(id)):
            raise NotFoundError("Not found")
        return "", 204
