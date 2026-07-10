"""Capability setup action — connect a capability with credentials.

Covers:
- POST /api/capabilities/setup/<id>  → post

URL reshaped from the legacy ``/api/capabilities/<id>/setup`` — the same
contract reshape as every other migrated verb. Mirrors the legacy handler's
semantics exactly: invalid credentials during ``configure()`` and a
still-disconnected capability afterward both fail with 422; a locked vault
fails with 403 (raised by ``_load_caps``/``configure``/``is_connected``,
caught around the whole handler body, matching the legacy module). On
success an initial sync runs on a daemon thread — the response returns
before the sync completes.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse, NotFoundError
from api.endpoints.capabilities import _load_caps
from api.request import Request
from api.request.capabilities import CapabilitySetupRequest
from api.response.response import Response
from services.vault_service import VaultLockedError

if TYPE_CHECKING:
    from api.endpoints.capabilities import _Capability

logger = logging.getLogger(__name__)

_ERR_VAULT = "Vault is locked — please log out and log back in"
_ERR_CONNECTION = "Connection failed — check credentials and try again"


class CapabilitiesSetup(Action):
    """Action connecting a capability with credentials."""

    request_dto: ClassVar[type[Request] | None] = CapabilitySetupRequest
    response_dto = {
        "post": DocumentedResponse(
            extras=(
                (403, _ERR_VAULT),
                (404, "Capability not found"),
                (422, "Invalid capability configuration or connection failed"),
            ),
        ),
    }

    def slug(self) -> str:
        return "capabilities"

    def verb(self) -> str:
        return "setup"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Connect a capability with credentials; fires an initial sync on a daemon thread.

        Raises:
            NotFoundError: capability id not found.

        Returns:
            "", 204 on success; 422 on invalid configuration or a
            still-disconnected capability after ``configure()``; 403 if the
            vault is locked.
        """
        cap_id = cast(str, id)
        dto = cast(CapabilitySetupRequest, data)
        try:
            caps = _load_caps()
            if cap_id not in caps:
                raise NotFoundError(f"Capability not found: {cap_id}")

            cap = cast("_Capability", caps[cap_id])
            try:
                cap.configure(dto.model_dump(exclude_none=True))
            except ValueError as exc:
                logger.warning("[capabilities] setup '%s' configure failed: %s", cap_id, exc)
                return Response.failure("Invalid capability configuration"), 422

            if not cap.is_connected():
                return Response.failure(_ERR_CONNECTION), 422
        except VaultLockedError:
            logger.warning("[capabilities] setup '%s' failed: vault is locked", cap_id)
            return Response.failure(_ERR_VAULT), 403

        def _first_sync() -> None:
            try:
                cap.monitor()
                logger.info("[capabilities] initial sync complete for '%s'", cap_id)
            except Exception as sync_exc:  # noqa: BLE001
                logger.warning("[capabilities] initial sync failed for '%s': %s", cap_id, sync_exc)

        threading.Thread(target=_first_sync, daemon=True, name=f"cap-init-{cap_id}").start()
        return "", 204
