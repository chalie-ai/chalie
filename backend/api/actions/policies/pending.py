"""Pending permission prompts — the ask-gates parked right now."""

from __future__ import annotations

from typing import cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.policies import PermissionOriginResponse, PermissionRequestResponse
from api.response.response import Response
from services.policy_manager import PolicyManager


class PendingAction(Action):
    """Action to list the permission prompts still waiting for an answer."""

    # id is ignored (lists every parked gate, no per-id lookup to miss), so
    # get can never emit 404.
    response_dto = {
        "get": DocumentedResponse(PermissionRequestResponse, listing=True, not_found=False),
    }

    def _service(self) -> PolicyManager:
        return PolicyManager()

    def get(self, id: int | str) -> ResponseReturnValue:
        """List every parked ask-gate, oldest first.

        Each entry is the ``permission_request`` frame the socket broadcast for
        it, so a client that missed the frame (reload, reconnect, no tab open)
        restores the same card from here; ``POST /api/policies/respond``
        answers it.
        """
        dtos = [
            PermissionRequestResponse(
                request_id=cast(str, frame["request_id"]),
                action_id=cast(str, frame["action_id"]),
                summary=cast(str, frame["summary"]),
                origin=PermissionOriginResponse.model_validate(frame["origin"]),
            )
            for frame in self._service().pending()
        ]
        total = len(dtos)
        return Response.listing(dtos, page=1, limit=max(total, 1), total=total)
