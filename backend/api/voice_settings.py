"""Voice Settings API — enable/disable voice and check installation status.

Routes:
  GET  /api/voice-settings          — get voice enabled state + install status
  PUT  /api/voice-settings          — enable or disable voice
"""

import logging

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

voice_settings_bp = Blueprint("voice_settings", __name__, url_prefix="/api/voice-settings")


def _get_services():
    from services.database_service import get_shared_db_service
    from services.settings_service import SettingsService

    db = get_shared_db_service()
    return SettingsService(db)


@voice_settings_bp.route("", methods=["GET"])
@require_session
def get_voice_settings():
    from services.runtime_deps_service import RuntimeDepsService

    settings = _get_services()
    enabled = settings.get("voice_enabled") == "true"
    status = RuntimeDepsService.get_status()["voice"]

    return jsonify({
        "enabled": enabled,
        "install_status": status["status"],
        "error": status["error"],
    })


@voice_settings_bp.route("", methods=["PUT"])
@require_session
def update_voice_settings():
    from services.runtime_deps_service import RuntimeDepsService

    body = request.get_json(force=True)
    enabled = body.get("enabled")
    if enabled is None:
        return jsonify({"error": "enabled field required"}), 400

    settings = _get_services()
    settings.set("voice_enabled", "true" if enabled else "false")

    if enabled:
        result = RuntimeDepsService.enable_voice()
        logger.info("[VoiceSettings] Voice enabled — installation triggered")
        return jsonify({"enabled": True, **result})
    else:
        logger.info("[VoiceSettings] Voice disabled")
        return jsonify({"enabled": False, "status": "disabled"})
