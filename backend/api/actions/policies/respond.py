"""Respond to a permission gate with allow/deny."""

import logging
import threading
from typing import cast

from api.action import Action
from api.request import Request
from flask.typing import ResponseReturnValue

from api.request.policies import PolicyRespondRequest
from services.policy_manager import PolicyManager, _permission_gates

logger = logging.getLogger(__name__)


class RespondAction(Action):
    """Action to resolve a permission gate with the user's allow/deny decision."""

    request_dto = PolicyRespondRequest

    def slug(self) -> str:
        return "policies"

    def verb(self) -> str:
        return "respond"

    def _service(self) -> PolicyManager:
        return PolicyManager()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Approve or deny a permission gate.

        An unknown or already-resolved ``request_id`` is a graceful 204 no-op,
        NOT a not-found error — the gate-absent path is by design.
        """
        dto = cast(PolicyRespondRequest, data)
        gate = _permission_gates.get(dto.request_id)
        if gate is None:
            logger.warning("No gate found for request_id=%s", dto.request_id)
            return "", 204
        gate["result"] = "approved" if dto.approved else "denied"
        cast(threading.Event, gate["event"]).set()
        return "", 204
