"""Browser hand-offs action — ``POST /api/browser/handoffs/<id>`` says the user is done.

The browsing run parked on the hand-off resumes and reads the page. An unknown
or already-finished hand-off is a 404.
"""

from __future__ import annotations

from typing import ClassVar

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from exceptions import NotFoundError
from tools.browser import tabs


class BrowserHandoffs(Action):
    """Action finishing a hand-off."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    response_dto = {"post": DocumentedResponse(extras=((404, "No open hand-off"),))}

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Finish the hand-off *id*, resuming its run."""
        if self.is_create(id) or not tabs.finish_handoff(str(id)):
            raise NotFoundError("Not found")
        return "", 204
