"""Capability setup action — connect a capability with credentials.

Covers:
- POST /api/capabilities/setup/<id>  → post

URL reshaped from the legacy ``/api/capabilities/<id>/setup`` — the same
contract reshape as every other migrated verb. Mirrors the legacy handler's
semantics exactly: invalid credentials during ``configure()`` and a
capability neither connected nor still confirming its setup afterward both
fail with 422 — a setup still confirming reports through the status action
how it ends. A capability that cannot work where Chalie runs is refused with
422 and its reason before ``configure()`` is called; a locked vault
fails with 403 (raised by ``_load_caps``/``configure``/``is_connected``,
caught around the whole handler body, matching the legacy module). On
success an initial sync runs on a daemon thread — the response returns
before the sync completes.

A ``configure()`` failure carries a plain ``reason`` the Brain words for the
reader, read from the request error the capability raised it from:
``credentials_rejected`` (the server answered 401/403),
``certificate_untrusted`` (its certificate failed verification),
``unreachable`` (no answer, or no secure connection), or
``invalid_configuration`` (anything else).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import requests
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.endpoints.capabilities import _load_caps
from api.request import Request
from api.request.capabilities import CapabilitySetupRequest
from api.response.response import Response
from exceptions import NotFoundError, VaultLockedError

if TYPE_CHECKING:
    from api.endpoints.capabilities import _Capability

logger = logging.getLogger(__name__)

_ERR_VAULT = "Vault is locked — please log out and log back in"
_ERR_CONNECTION = "Connection failed — check credentials and try again"

SetupFailureReason = Literal[
    "credentials_rejected",
    "certificate_untrusted",
    "unreachable",
    "invalid_configuration",
]


def _failure_reason(exc: ValueError) -> SetupFailureReason:
    """Why ``configure()`` failed, from the request error it was raised from."""
    cause = exc.__cause__
    if isinstance(cause, requests.HTTPError) and cause.response is not None and cause.response.status_code in (401, 403):
        return "credentials_rejected"
    if isinstance(cause, requests.exceptions.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(cause):
        return "certificate_untrusted"
    if isinstance(cause, requests.ConnectionError | requests.Timeout):
        return "unreachable"
    return "invalid_configuration"


class CapabilitiesSetup(Action):
    """Action connecting a capability with credentials."""

    request_dto: ClassVar[type[Request] | None] = CapabilitySetupRequest
    response_dto = {
        "post": DocumentedResponse(
            extras=(
                (403, _ERR_VAULT),
                (404, "Capability not found"),
                (422, "Invalid capability configuration, connection failed, or the capability is unavailable here"),
            ),
        ),
    }

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Connect a capability with credentials; fires an initial sync on a daemon thread.

        Raises:
            NotFoundError: capability id not found.

        Returns:
            "", 204 on success or while the setup is still being confirmed;
            422 on a capability unavailable where Chalie runs, on invalid
            configuration (with its ``reason``), or on a capability neither
            connected nor confirming after ``configure()``; 403 if the vault
            is locked.
        """
        cap_id = cast(str, id)
        dto = cast(CapabilitySetupRequest, data)
        try:
            caps = _load_caps()
            if cap_id not in caps:
                raise NotFoundError(f"Capability not found: {cap_id}")

            cap = cast("_Capability", caps[cap_id])
            unavailable_reason = cap.unavailable_reason()
            if unavailable_reason is not None:
                logger.warning("[capabilities] setup '%s' refused: %s", cap_id, unavailable_reason)
                return Response.failure(unavailable_reason), 422
            try:
                cap.configure(dto.model_dump(exclude_none=True))
            except ValueError as exc:
                logger.warning("[capabilities] setup '%s' configure failed: %s", cap_id, exc)
                return Response.failure("Invalid capability configuration", _failure_reason(exc)), 422

            if not cap.is_connected() and not cap.setup_pending():
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
