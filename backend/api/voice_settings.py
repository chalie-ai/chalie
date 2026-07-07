"""Voice-settings namespace — derived state read plus a toggle that triggers a background install."""

from __future__ import annotations

import logging
from typing import cast

from flask_restx import Namespace, Resource

from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.voice_settings import VoiceSettings, VoiceSettingsState, VoiceSettingsUpdate

logger = logging.getLogger(__name__)

voice_settings_ns = Namespace("voice_settings", description="Voice settings", path="/api/voice-settings")

register_dto(voice_settings_ns, VoiceSettings, VoiceSettingsUpdate, VoiceSettingsState, Error)

_VS = voice_settings_ns.models


@voice_settings_ns.route("")
class VoiceSettingsResource(Resource):
    @require_session
    @voice_settings_ns.response(200, "Voice settings", model=_VS["VoiceSettings"])
    @voice_settings_ns.response(500, "Server error", model=_VS["Error"])
    @responds(VoiceSettings, code=200)
    def get(self) -> VoiceSettings:
        from models.setting import Setting
        from services.runtime_deps_service import RuntimeDepsService

        status = RuntimeDepsService.get_status()["voice"]
        return VoiceSettings(
            enabled=Setting.get("voice_enabled") == "true",
            install_status=cast(str, status["status"]),
            error=status["error"],
        )

    @require_session
    @voice_settings_ns.expect(_VS["VoiceSettingsUpdate"])
    @voice_settings_ns.response(200, "Voice settings updated", model=_VS["VoiceSettingsState"])
    @voice_settings_ns.response(422, "Validation failed", model=_VS["Error"])
    @voice_settings_ns.response(500, "Server error", model=_VS["Error"])
    @responds(VoiceSettingsState, code=200)
    @expects(VoiceSettingsUpdate)
    def put(self, dto: VoiceSettingsUpdate) -> VoiceSettingsState:
        from models.setting import Setting
        from services.runtime_deps_service import RuntimeDepsService

        Setting.set("voice_enabled", "true" if dto.enabled else "false")

        if dto.enabled:
            result = RuntimeDepsService.enable_voice()
            logger.info("[VoiceSettings] Voice enabled — installation triggered")
            return VoiceSettingsState(
                enabled=True,
                status=result["status"],
                message=result["message"],
            )
        logger.info("[VoiceSettings] Voice disabled")
        return VoiceSettingsState(enabled=False, status="disabled")