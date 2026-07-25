"""Voice playback action — nested resource under /api/voice/transcript.

Covers:
- GET /api/voice/transcript/<transcript_id>  → get (pre-synthesized speech)
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar, cast

from flask import Response as FlaskResponse
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.voice import VoiceStateResponse
from exceptions import NotFoundError
from models.transcript import Transcript
from models.voice_transcript import VoiceTranscript
from services.file_mapper_service import FileMapperService
from services.voice_transcript_service import get_service

logger = logging.getLogger(__name__)


class VoiceTranscriptAction(Action):
    """Plays back one transcript row's pre-synthesized speech.

    Reads the stored outcome — it never re-synthesizes a row the pipeline
    already carried to a terminal state. The one exception is a row with no
    outcome at all (history predating pre-synthesis, or a job lost to a
    restart): that starts the pipeline once and answers 202, so there is no
    backfill sweep and no stuck state.
    """

    id_type: ClassVar[type[int] | type[str]] = int
    response_dto = {
        "get": DocumentedResponse(
            VoiceStateResponse,
            extras=(
                (200, "Pre-synthesized speech (audio/wav)"),
                (202, "Synthesis started or still running"),
                (409, "Synthesis failed for this transcript row"),
            ),
        )
    }

    def get(self, id: int | str) -> ResponseReturnValue:
        if self.is_create(id):
            return self.not_allowed()

        transcript_id = cast("int", id)
        row = VoiceTranscript.filter("transcript_id", transcript_id).first()

        if row is not None and row.file_path is not None:
            path = FileMapperService.get_voice_path(row.file_path)
            if not path.exists():
                logger.error(
                    "[Voice] transcript %d claims audio at %s but the file is gone",
                    transcript_id, path,
                )
                return VoiceStateResponse(state="failed").single(), 409
            wav_bytes = path.read_bytes()
            # Audio is a non-JSON payload, so it leaves as a raw werkzeug
            # response rather than through an envelope builder.
            return FlaskResponse(
                wav_bytes,
                mimetype="audio/wav",
                headers={
                    "Content-Length": str(len(wav_bytes)),
                    # The audio for a transcript row never changes — the row is
                    # immutable once written, and a deleted row takes the file
                    # with it — so the browser may hold it indefinitely.
                    "Cache-Control": "public, max-age=31536000, immutable",
                },
            )

        if row is not None:
            return VoiceStateResponse(state="failed").single(), 409

        transcript = Transcript.get(transcript_id)
        if transcript is None or transcript.role != "assistant" or not transcript.settled:
            raise NotFoundError("No speech for this message")

        threading.Thread(
            target=get_service().synthesize_settled,
            args=(transcript_id,),
            daemon=True,
        ).start()
        return VoiceStateResponse(state="pending").single(), 202
