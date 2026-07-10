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
from api.endpoint import DocumentedResponse, NotFoundError
from api.endpoints.capabilities import _load_caps
from api.request import Request

if TYPE_CHECKING:
    from api.endpoints.capabilities import _Capability


class CapabilitiesDisconnect(Action):
    """Action disconnecting a capability."""

    response_dto = {
        "post": DocumentedResponse(extras=((404, "Capability not found"),)),
    }

    def slug(self) -> str:
        return "capabilities"

    def verb(self) -> str:
        return "disconnect"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Disconnect a capability.

        Raises:
            NotFoundError: capability id not found.
        """
        cap_id = cast(str, id)
        caps = _load_caps()
        if cap_id not in caps:
            raise NotFoundError(f"Capability not found: {cap_id}")
        cast("_Capability", caps[cap_id]).disconnect()
        return "", 204
