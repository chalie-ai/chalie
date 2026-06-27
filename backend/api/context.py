"""Context API -- returns ambient context (location, timezone, device data)."""

import logging
from typing import TYPE_CHECKING, cast

from flask_restx import Namespace, Resource

from api.auth import require_auth

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

context_bp = Namespace("context", description="Ambient context", path="/api/context")


@context_bp.route("")
class ContextResource(Resource):
    @require_auth
    @context_bp.response(200, "Context")
    def get(self):
        """Location is city-level, never raw GPS."""
        result: dict[str, object] = {}

        try:
            from services.locale_service import (
                get_timezone_name, get_location, format_date,
            )
            from services.time_utils import utc_now

            # Location (city-level, never raw GPS)
            location = get_location()
            if location.get("lat") is not None:
                result["location"] = location

            # Timezone
            tz_name = get_timezone_name()
            if tz_name != "UTC":
                result["timezone"] = {
                    "timezone": tz_name,
                    "local_time": format_date(utc_now(), "%Y-%m-%dT%H:%M:%S", for_ui=True),
                }

            # Device (non-locale — reads from telemetry directly)
            from services.client_context_service import ClientContextService
            raw_ctx = ClientContextService().get()
            if raw_ctx:
                device = cast("dict[str, object]", raw_ctx.get("device")) or {}
                device_class = device.get("class")
                platform = device.get("platform")
                if device_class or platform:
                    result["device"] = {
                        "class": device_class or "desktop",
                        "platform": platform or "",
                    }

        except Exception as e:
            logger.warning("[Context API] Failed to build context: %s", e)

        return result, 200