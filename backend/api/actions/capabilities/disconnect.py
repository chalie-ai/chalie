"""Capability disconnect action — disconnect a capability.

Covers:
- POST /api/capabilities/disconnect/<id>  → post

URL reshaped from the legacy ``/api/capabilities/<id>/disconnect`` — the same
contract reshape as every other migrated verb.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.endpoints.capabilities import CapabilitiesEndpoint
from api.request import Request

if TYPE_CHECKING:
    from api.endpoints.capabilities import _Capability


class CapabilitiesDisconnect(Action):
    """Action disconnecting a capability."""

    response_dto = {
        "post": DocumentedResponse(extras=((404, "Capability not found"),)),
    }

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Disconnect a capability.

        Raises:
            NotFoundError: capability id not found.
        """
        cap_id = cast(str, id)
        caps = CapabilitiesEndpoint._load_caps()
        if cap_id not in caps:
            raise NotFoundError(f"Capability not found: {cap_id}")
        cast("_Capability", caps[cap_id]).disconnect()
        return "", 204
