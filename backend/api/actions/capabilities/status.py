"""Capability status action — connectivity status for a single capability.

Covers:
- GET /api/capabilities/status/<id>  → get

URL reshaped from the legacy ``/api/capabilities/<id>/status`` — the same
contract reshape as every other migrated verb; zero callers depended on the
old path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.endpoints.capabilities import _get_last_sync_at, _load_caps
from api.response.capabilities import CapabilityStatusResponse

if TYPE_CHECKING:
    from api.endpoints.capabilities import _Capability


class CapabilitiesStatus(Action):
    """Action reading a single capability's connectivity status."""

    response_dto = {
        "get": DocumentedResponse(CapabilityStatusResponse),
    }

    def get(self, id: int | str) -> ResponseReturnValue:
        """Get connectivity status for a capability.

        Raises:
            NotFoundError: capability id not found.
        """
        cap_id = cast(str, id)
        caps = _load_caps()
        if cap_id not in caps:
            raise NotFoundError(f"Capability not found: {cap_id}")

        cap = cast("_Capability", caps[cap_id])
        return CapabilityStatusResponse(
            id=cap_id,
            connected=cap.is_connected(),
            last_sync_at=_get_last_sync_at(cap_id),
        ).single()
