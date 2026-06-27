import logging
from typing import TYPE_CHECKING

from flask import request
from flask_restx import Namespace, Resource

from .auth import require_session

if TYPE_CHECKING:
    from services.settings_service import SettingsService

logger = logging.getLogger(__name__)

voice_settings_bp = Namespace("voice_settings", description="Voice settings", url_prefix="/api/voice-settings")


def _get_services() -> "SettingsService":
    from services.database_service import get_shared_db_service
    from services.settings_service import SettingsService

    db = get_shared_db_service()
    return SettingsService(db)


@voice_settings_bp.route("")
class VoiceSettingsResource(Resource):
    @require_session
    @voice_settings_bp.response(200, "Voice settings")
    def get(self):
        from services.runtime_deps_service import RuntimeDepsService

        settings = _get_services()
        enabled = settings.get("voice_enabled") == "true"
        status = RuntimeDepsService.get_status()["voice"]

        return {
            "enabled": enabled,
            "install_status": status["status"],
            "error": status["error"],
        }

    @require_session
    @voice_settings_bp.response(200, "Voice settings updated")
    @voice_settings_bp.response(400, "Missing enabled field")
    def put(self):
        from services.runtime_deps_service import RuntimeDepsService

        body = request.get_json(force=True)
        enabled = body.get("enabled")
        if enabled is None:
            return {"error": "enabled field required"}, 400

        settings = _get_services()
        settings.set("voice_enabled", "true" if enabled else "false")

        if enabled:
            result = RuntimeDepsService.enable_voice()
            logger.info("[VoiceSettings] Voice enabled — installation triggered")
            return {"enabled": True, **result}
        else:
            logger.info("[VoiceSettings] Voice disabled")
            return {"enabled": False, "status": "disabled"}