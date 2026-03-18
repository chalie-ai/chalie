"""
Context API — returns the user's current ambient context for gateway proxy.

This endpoint provides location, timezone, device, energy, and attention
data that the dashboard gateway filters by approved scopes before serving
to interface daemons.
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
        energy: float 0–1
        attention: "focused" | "ambient" | "distracted"

    Auth: Bearer token or session cookie.
    """
    result = {}

    try:
        from services.client_context_service import ClientContextService
        from services.ambient_inference_service import AmbientInferenceService

        ctx_svc = ClientContextService()
        raw_ctx = ctx_svc.get()

        if raw_ctx:
            # Run ambient inference for attention, energy, etc.
            ambient_svc = AmbientInferenceService()
            inferences = ambient_svc.infer(raw_ctx)

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

            # Energy and attention from ambient inference
            if inferences:
                energy = inferences.get("energy")
                if energy is not None:
                    # Convert label to numeric if needed
                    energy_map = {"low": 0.25, "moderate": 0.5, "high": 0.75}
                    if isinstance(energy, str):
                        result["energy"] = energy_map.get(energy, 0.5)
                    else:
                        result["energy"] = float(energy)

                attention = inferences.get("attention")
                if attention:
                    # Normalize to SDK-expected values
                    attention_map = {
                        "deep_focus": "focused",
                        "casual": "ambient",
                        "distracted": "distracted",
                        "away": "distracted",
                    }
                    result["attention"] = attention_map.get(attention, "ambient")

    except Exception as e:
        logger.warning("[Context API] Failed to build context: %s", e)

    return jsonify(result), 200
