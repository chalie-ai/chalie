"""Reset all policies to their default values."""

from api.action import Action
from api.request import Request
from flask.typing import ResponseReturnValue

from services.policy_manager import PolicyManager


class ResetAction(Action):
    """Action to reset all policies to their default values."""

    def _service(self) -> PolicyManager:
        return PolicyManager()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Re-apply the static seed (wipe + reseed)."""
        self._service().reset_to_defaults()
        return "", 204
