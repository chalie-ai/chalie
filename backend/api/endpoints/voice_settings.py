"""Voice-settings endpoint — migrated from the legacy ``voice_settings`` namespace.

Voice settings are a SINGLETON: no per-id resources exist, so the only
id-addressed calls this endpoint accepts are the create sentinel (``-1``),
matching the legacy module's single ``/api/voice-settings`` resource.

- GET  /api/voice-settings/all  → get_all (a one-item listing: the singleton state)
- POST /api/voice-settings/-1   → post (toggle; the legacy PUT reshaped to POST)

The legacy PUT /api/voice-settings is mapped to POST /api/voice-settings/-1
(the same contract reshape as every other migrated endpoint group) — this
POST never creates a new resource, it flips the one that always exists, so
``_post_may_create`` stays ``False`` and swagger never documents a 201.
"""

from __future__ import annotations

import logging
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from exceptions import NotFoundError
from api.request import Request
from api.request.voice_settings import VoiceSettingsRequest
from api.response.voice_settings import VoiceSettingsResponse, VoiceSettingsStateResponse
from models.setting import Setting
from services.runtime_deps_service import RuntimeDepsService

logger = logging.getLogger(__name__)


class VoiceSettingsEndpoint(Endpoint):
    """Singleton endpoint for the voice on/off toggle and its live install status."""

    request_dto: ClassVar[type[Request] | None] = VoiceSettingsRequest
    _post_may_create: ClassVar[bool] = False
    """This POST toggles the singleton in place — it never creates a new
    resource, so swagger must never document a 201 here."""

    response_dto = {
        "get_all": DocumentedResponse(VoiceSettingsResponse, listing=True),
        "post": DocumentedResponse(
            VoiceSettingsStateResponse,
            extras=((404, "No such voice-settings resource"),),
        ),
    }

    def slug(self) -> str:
        return "voice-settings"

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """Return the singleton voice settings state as a one-item listing."""
        status = RuntimeDepsService.get_status()["voice"]
        state = VoiceSettingsResponse(
            enabled=Setting.get("voice_enabled") == "true",
            install_status=cast(str, status["status"]),
            error=status["error"],
        )
        return VoiceSettingsResponse.listing([state], page=1, limit=1, total=1)

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Toggle voice on/off, triggering a background install when enabling.

        Only the create sentinel (``-1``) addresses the singleton — no
        per-id voice-settings resources exist.

        Raises:
            NotFoundError: id is not the create sentinel.

        Returns:
            VoiceSettingsStateResponse — the post-toggle state.
        """
        if not self.is_create(id):
            raise NotFoundError("Not found")

        dto = cast(VoiceSettingsRequest, data)
        Setting.set("voice_enabled", "true" if dto.enabled else "false")

        if dto.enabled:
            result = RuntimeDepsService.enable_voice()
            logger.info("[VoiceSettings] Voice enabled — installation triggered")
            return VoiceSettingsStateResponse(
                enabled=True,
                status=result["status"],
                message=result["message"],
            ).single()

        logger.info("[VoiceSettings] Voice disabled")
        return VoiceSettingsStateResponse(enabled=False, status="disabled").single()
