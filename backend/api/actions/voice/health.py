"""Voice readiness action — nested resource under /api/voice/health.

Covers:
- GET /api/voice/health  → get (always 200; readiness is body-encoded)
"""

from __future__ import annotations

import threading

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.voice import VoiceHealthResponse
from services.speech_to_text_service import get_service as stt_service
from services.voice_transcript_service import get_service as tts_service

_MODELS_MISSING_HINT = "Voice models are missing from this install. Reinstall Chalie to restore them."


class VoiceHealthAction(Action):
    """Readiness probe for the two on-device voice models (Kokoro TTS, Moonshine STT).

    Always answers 200 — readiness lives in the body, never the status code, so
    a client polling this never has to distinguish "not ready" from "call
    failed". A probe that finds both models present but cold starts the load in
    the background and reports ``loading``, so the first probe warms the service
    instead of merely reporting it idle.
    """

    response_dto = {"get": DocumentedResponse(VoiceHealthResponse)}

    def get(self, id: int | str) -> ResponseReturnValue:
        if not self.is_create(id):
            return self.not_allowed()

        stt = stt_service()
        if not stt.available:
            return VoiceHealthResponse(
                status="unavailable",
                reason="deps_missing",
                missing=list(stt.missing_modules),
                hint=stt.install_hint,
            ).single()

        tts = tts_service()
        if tts.is_loaded and stt.is_loaded:
            return VoiceHealthResponse(status="ok").single()
        if tts.is_loading or stt.is_loading:
            return VoiceHealthResponse(status="loading").single()

        missing = tts.missing_model_files() + stt.missing_model_files()
        if missing:
            return VoiceHealthResponse(
                status="unavailable",
                reason="models_missing",
                missing=missing,
                hint=_MODELS_MISSING_HINT,
            ).single()

        threading.Thread(target=self._warm, daemon=True).start()
        return VoiceHealthResponse(status="loading").single()

    @staticmethod
    def _warm() -> None:
        """Load both models off the request thread so the probe stays non-blocking."""
        tts_service().ensure_loaded()
        stt_service().ensure_loaded()
