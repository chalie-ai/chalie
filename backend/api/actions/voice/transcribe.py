"""Voice transcription action — nested resource under /api/voice/transcribe.

Covers:
- POST /api/voice/transcribe  → post (one uploaded WAV → text)
"""

from __future__ import annotations

import logging

from flask import jsonify, make_response, request
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.response.response import Response
from api.response.voice import TranscriptionResponse
from services.speech_to_text_service import MAX_AUDIO_SECONDS
from services.speech_to_text_service import get_service as stt_service
from services.voice_transcript_service import get_service as tts_service

logger = logging.getLogger(__name__)

_MODELS_MISSING_HINT = "Voice models are missing from this install. Reinstall Chalie to restore them."


class VoiceTranscribeAction(Action):
    """Transcribes a single uploaded WAV file into text.

    The upload arrives as multipart form data, so this action declares no
    ``request_dto`` and reads ``request.files`` directly — the base's JSON body
    reader has nothing to validate here.
    """

    response_dto = {
        "post": DocumentedResponse(
            TranscriptionResponse,
            extras=(
                (422, "Invalid upload (missing/empty/malformed/too-long WAV)"),
                (503, "Voice unavailable (deps or models missing/loading)"),
                (500, "Transcription failed"),
            ),
        )
    }

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if not self.is_create(id):
            return self.not_allowed()

        stt = stt_service()
        if not stt.available:
            return self._unavailable(
                "Voice dependencies are not installed", stt.install_hint,
            )

        if "file" not in request.files:
            return Response.failure("No file uploaded"), 422

        audio = request.files["file"].read()
        if not audio:
            return Response.failure("Empty file"), 422

        duration = stt.wav_duration_seconds(audio)
        if duration <= 0.0:
            return Response.failure("Malformed or unsupported WAV file"), 422
        if duration > MAX_AUDIO_SECONDS:
            return Response.failure(
                f"Audio exceeds {MAX_AUDIO_SECONDS}s limit ({duration:.1f}s)"
            ), 422

        if not stt.ensure_loaded():
            return self._not_ready()

        try:
            return TranscriptionResponse(text=stt.transcribe(audio)).single()
        except Exception as e:
            logger.error("[Voice] STT error: %s", e)
            return Response.failure("Transcription failed"), 500

    def _not_ready(self) -> ResponseReturnValue:
        """503 distinguishing a missing install from a transient cold start.

        Missing model files are terminal, so that answer carries no
        ``Retry-After``; a still-loading model does, letting the client retry
        without polling blind.
        """
        missing = tts_service().missing_model_files() + stt_service().missing_model_files()
        if missing:
            return self._unavailable(
                f"Voice models not installed ({', '.join(missing)})", _MODELS_MISSING_HINT,
            )
        response = make_response(jsonify(Response.failure("Models still loading")), 503)
        response.headers["Retry-After"] = "3"
        return response

    @staticmethod
    def _unavailable(message: str, hint: str) -> ResponseReturnValue:
        """Terminal 503 — the hint rides inside the message because the error
        envelope carries a single string, and the hint is what the user acts on."""
        return Response.failure(f"{message} — {hint}"), 503
