"""Personality namespace — read and write the 5-axis personality tuple."""

from __future__ import annotations

import logging
from typing import cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.personality import Personality, PersonalityUpdate

logger = logging.getLogger(__name__)

personality_ns = Namespace('personality', description='Personality tuple', path='/api/settings')

register_dto(personality_ns, Personality, PersonalityUpdate, Error)

_P = personality_ns.models


@personality_ns.route('/personality')
class PersonalityResource(Resource):
    @require_session
    @personality_ns.response(200, "Personality tuple", model=_P["Personality"])
    @personality_ns.response(500, "Failed to get personality", model=_P["Error"])
    @responds(Personality, code=200)
    def get(self) -> Personality | ResponseReturnValue:
        try:
            from services.personality.personality_service import personality_service

            tup = personality_service.get_tuple()
            return Personality(
                tuple=list(cast("tuple[int, int, int, int, int]", tup)),
                voice=personality_service.get_voice(),
            )
        except Exception as exc:
            logger.error("[REST API] Failed to get personality: %s", exc)
            return error("Failed to get personality", 500)

    @require_session
    @personality_ns.expect(_P["PersonalityUpdate"])
    @personality_ns.response(200, "Personality updated", model=_P["Personality"])
    @personality_ns.response(422, "Validation failed", model=_P["Error"])
    @personality_ns.response(500, "Failed to set personality", model=_P["Error"])
    @responds(Personality, code=200)
    @expects(PersonalityUpdate)
    def put(self, dto: PersonalityUpdate) -> Personality | ResponseReturnValue:
        try:
            from services.personality.personality_service import personality_service

            voice = personality_service.set_tuple(cast("tuple[int, int, int, int, int]", tuple(dto.tuple_)))
            return Personality(tuple=dto.tuple_, voice=voice)
        except ValueError as exc:
            logger.warning("[REST API] Personality validation error: %s", exc)
            return error(str(exc), 422)
        except Exception as exc:
            logger.error("[REST API] Failed to set personality: %s", exc)
            return error("Failed to set personality", 500)
