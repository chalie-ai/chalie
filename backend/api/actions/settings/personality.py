"""Personality action — read and write the 5-axis personality tuple.

Covers:
- GET  /api/settings/personality  → get  (read the tuple + derived voice)
- POST /api/settings/personality  → post (write the tuple, legacy PUT → POST)

The personality tuple is a process-wide singleton, sourced from
``services.personality.personality_service`` with no per-resource id, so
both handlers ignore ``id``.
"""

from __future__ import annotations

import logging
from typing import cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.settings import PersonalityRequest
from api.response.settings import PersonalityResponse
from services.personality.personality_service import personality_service

logger = logging.getLogger(__name__)


class PersonalityAction(Action):
    """Action for reading/writing the personality tuple — URL unchanged from legacy."""

    request_dto = PersonalityRequest
    # id is ignored on both verbs (singleton tuple, no per-resource lookup to
    # miss), so `get` can never emit a structural 404.
    response_dto = {
        "get": DocumentedResponse(PersonalityResponse, not_found=False),
        "post": DocumentedResponse(
            PersonalityResponse,
            extras=((422, "Invalid personality tuple"),),
        ),
    }

    def get(self, id: int | str) -> ResponseReturnValue:
        """Read the current personality tuple and its derived voice."""
        tup = personality_service.get_tuple()
        return PersonalityResponse(
            tuple=list(cast("tuple[int, int, int, int, int]", tup)),
            voice=personality_service.get_voice(),
        ).single()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Write the personality tuple; id is ignored (singleton)."""
        dto = cast(PersonalityRequest, data)
        try:
            voice = personality_service.set_tuple(
                cast("tuple[int, int, int, int, int]", tuple(dto.tuple_))
            )
        except ValueError as exc:
            logger.warning("[SETTINGS API] Personality validation error: %s", exc)
            return PersonalityResponse.failure(str(exc)), 422
        return PersonalityResponse(tuple=dto.tuple_, voice=voice).single()
