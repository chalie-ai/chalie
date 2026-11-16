"""TextToSpeechAbility — turn a text into a spoken audio clip on disk.

Speaks with the same Kokoro voice as the chat's speaker button
(:class:`VoiceTranscriptService`) and writes the clip as an MP3 in Chalie's
temp folder. The result is the clip's full path, which a messaging tool can
send on as a voice note.

At most :data:`MAX_CLIPS` clips are kept: before a new one is written, the
oldest are deleted until there is room. The hourly temp sweep removes any
clip older than a day as well.
"""

from __future__ import annotations

import glob
import os
import threading
import uuid
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.text_to_speech_params_bag import TextToSpeechParamsBag
from services.tmp_storage import new_tmp_path
from services.voice_transcript_service import VoiceTranscriptService

MAX_CLIPS = 25

# One writer at a time, so two calls cannot both count 24 clips and leave 26.
_clips_lock = threading.Lock()


def _clip_age_key(path: str) -> tuple[float, str]:
    """Oldest first by modification time; a clip deleted meanwhile sorts first
    and its unlink is skipped."""
    try:
        return (os.path.getmtime(path), path)
    except FileNotFoundError:
        return (float("-inf"), path)


def _make_room() -> None:
    """Delete the oldest clips until fewer than :data:`MAX_CLIPS` remain."""
    clips = sorted(glob.glob(glob.escape(new_tmp_path("tts_")) + "*.mp3"), key=_clip_age_key)
    for path in clips[: max(0, len(clips) - (MAX_CLIPS - 1))]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


class TextToSpeechAbility(Ability[TextToSpeechParamsBag]):
    NAME: ClassVar[str] = "text_to_speech"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.MEDIA
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("tts", "speak", "voice note", "read aloud")

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.message,)}
    PARAMS: ClassVar[type[ParamBag] | None] = TextToSpeechParamsBag

    def get_summary(self) -> str:
        return (
            "Turn a text into a spoken audio clip (MP3) in Chalie's voice and get "
            "back the clip's full path. Use it to make a voice note to send with a "
            "messaging tool."
        )

    def get_examples(self) -> list[str]:
        return [
            "send me a voice note saying good morning",
            "read this summary out loud as an audio file",
            "make an audio clip of the reminder",
            "turn this message into speech",
            "record a voice message for me",
        ]

    def get_search_tooltip(self) -> str:
        return "Turn text into a spoken audio clip"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.message: {
                "type": "string",
                "description": "The text to speak, as plain sentences.",
            },
        },
        "required": [Keys.message],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: TextToSpeechParamsBag) -> ToolResult:
        svc = VoiceTranscriptService.instance()
        if not svc.ensure_loaded():
            return self._not_ready(svc)
        try:
            samples, sample_rate, audio_bytes = svc.synthesize(params.message)
        except ValueError:
            return ToolResult.err(
                "There is nothing to speak in that text.",
                code="invalid-param",
                hint=f"pass '{Keys.message}' as the words to say.",
            )
        if not svc.speech_is_audible(samples, sample_rate, audio_bytes):
            return ToolResult.err(
                "The speech came out silent or cut short, so no clip was kept.",
                code="tts-failed",
            )
        path = new_tmp_path(f"tts_{uuid.uuid4().hex}.mp3")
        with _clips_lock:
            _make_room()
            with open(path, "wb") as f:
                f.write(audio_bytes)
        return ToolResult.ok(path)

    @staticmethod
    def _not_ready(svc: VoiceTranscriptService) -> ToolResult:
        """Why the voice engine is not ready, told apart the way the voice
        endpoints tell it: not installed, still loading, or failed to load."""
        if VoiceTranscriptService.missing_model_files():
            return ToolResult.err(
                "Text to speech is not available: the voice models are not installed.",
                code="tts-unavailable",
            )
        if svc.is_loading:
            return ToolResult.err(
                "The voice engine is still loading.",
                code="tts-unavailable",
                hint="try again in a moment.",
            )
        return ToolResult.err(
            "The voice engine could not be loaded.",
            code="tts-unavailable",
        )
