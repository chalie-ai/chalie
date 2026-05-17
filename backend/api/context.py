"""
Context API — returns the user's current ambient context for gateway proxy.

This endpoint provides location, timezone, and device data that the dashboard
gateway filters by approved scopes before serving to interface daemons.
"""

import logging

from flask import Blueprint, jsonify

from api.auth import require_auth

logger = logging.getLogger(__name__)

context_bp = Blueprint("context", __name__)


@context_bp.route("/api/context", methods=["GET"])
@require_auth
def get_context():
    """Return the user's current ambient context.

    Response fields (all optional — absent if data unavailable):
        location: {lat, lon, name}
        timezone: {timezone, local_time}
        device: {class, platform}

    Auth: Bearer token or session cookie.
    """
    result = {}

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
            device = raw_ctx.get("device") or {}
            device_class = device.get("class")
            platform = device.get("platform")
            if device_class or platform:
                result["device"] = {
                    "class": device_class or "desktop",
                    "platform": platform or "",
                }

    except Exception as e:
        logger.warning("[Context API] Failed to build context: %s", e)

    return jsonify(result), 200
