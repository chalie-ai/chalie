"""
Personality namespace — read and write the 5-axis personality tuple.
"""

import logging
from typing import TYPE_CHECKING, cast

from flask import request
from flask_restx import Namespace, Resource

from .auth import require_session

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

personality_bp = Namespace('personality', description='Personality tuple', path='/settings')


@personality_bp.route('/personality')
class PersonalityResource(Resource):
    @require_session
    @personality_bp.response(200, "Personality tuple")
    @personality_bp.response(500, "Failed to get personality")
    def get(self):
        try:
            from services.personality.personality_service import personality_service

            tup = personality_service.get_tuple()
            return {"tuple": list(cast("tuple[int, int, int, int, int]", tup)), "voice": personality_service.get_voice()}, 200
        except Exception as exc:
            logger.error("[REST API] Failed to get personality: %s", exc)
            return {"error": "Failed to get personality"}, 500

    @require_session
    @personality_bp.response(200, "Personality updated")
    @personality_bp.response(400, "Validation error")
    @personality_bp.response(500, "Failed to set personality")
    def put(self):
        try:
            data = request.get_json(silent=True) or {}
            raw = data.get('tuple')

            if raw is None:
                return {"error": "Missing required field: tuple"}, 400
            if not isinstance(raw, list) or len(raw) != 5:
                return {"error": "Field 'tuple' must be a list of exactly 5 integers"}, 400
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in raw):
                return {"error": "All tuple elements must be integers"}, 400

            from services.personality.personality_service import personality_service

            voice = personality_service.set_tuple(tuple(raw))
            return {"tuple": raw, "voice": voice}, 200
        except ValueError as exc:
            logger.warning("[REST API] Personality validation error: %s", exc)
            return {"error": str(exc)}, 400
        except Exception as exc:
            logger.error("[REST API] Failed to set personality: %s", exc)
            return {"error": "Failed to set personality"}, 500