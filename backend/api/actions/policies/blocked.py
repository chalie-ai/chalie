"""Blocked log action — read and clear the blocked-action audit log."""

from __future__ import annotations

from typing import cast

from flask import request
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse, EndpointError
from api.response.policies import BlockedEntryResponse
from api.response.response import Response
from services.policy_manager import PolicyManager
from services.time_utils import parse_utc


class BlockedAction(Action):
    """Action to read or clear the blocked-action log."""

    # id is ignored on both verbs (reads/clears the whole log, no per-id
    # lookup to miss), so neither can ever emit 404.
    response_dto = {
        "get": DocumentedResponse(BlockedEntryResponse, listing=True, not_found=False),
        "delete": DocumentedResponse(not_found=False),
    }

    def slug(self) -> str:
        return "policies"

    def verb(self) -> str:
        return "blocked"

    def _service(self) -> PolicyManager:
        return PolicyManager()

    def get(self, id: int | str) -> ResponseReturnValue:
        """List recent blocked-action log entries.

        Accepts a `limit` query parameter (1-500, default 50), matching the
        legacy BlockedQuery DTO validation.
        """
        limit_str = request.args.get("limit", "50")
        try:
            limit = int(limit_str)
        except ValueError:
            raise EndpointError("Invalid limit parameter")
        limit = max(1, min(limit, 500))

        entries = self._service().get_blocked_log(limit=limit)
        dtos = [
            BlockedEntryResponse(
                action_id=cast(int, entry["action_id"]),
                context=entry["context"],
                reason=entry["reason"],
                created_at=parse_utc(entry["created_at"]),
            )
            for entry in entries
        ]
        total = len(dtos)
        return Response.listing(dtos, page=1, limit=max(total, 1), total=total)

    def delete(self, id: int | str) -> ResponseReturnValue:
        """Clear all entries from the blocked log."""
        self._service().clear_blocked_log()
        return "", 204
