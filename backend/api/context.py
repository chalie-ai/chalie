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
        from services.client_context_service import ClientContextService

        ctx_svc = ClientContextService()
        raw_ctx = ctx_svc.get()

        if raw_ctx:
            # Location (city-level, never raw GPS)
            location = raw_ctx.get("location")
            if location and isinstance(location, dict):
                result["location"] = {
                    "lat": location.get("lat"),
                    "lon": location.get("lon"),
                    "name": location.get("name", location.get("display_name", "")),
                }

            # Timezone
            tz = raw_ctx.get("timezone")
            if tz:
                from services.time_utils import utc_now
                result["timezone"] = {
                    "timezone": tz,
                    "local_time": utc_now().isoformat(),
                }

            # Device
            device_class = raw_ctx.get("device_class")
            platform = raw_ctx.get("platform")
            if device_class or platform:
                result["device"] = {
                    "class": device_class or "desktop",
                    "platform": platform or "",
                }

    except Exception as e:
        logger.warning("[Context API] Failed to build context: %s", e)

    return jsonify(result), 200
