"""Response DTOs for the voice actions — readiness, playback state, transcription.

Grouped in one file because they form one cohesive namespace shape set: the
always-200 readiness probe, the playback outcome for a transcript row that has
no audio to hand back yet, and the transcribe result.
"""

from __future__ import annotations

from api.response.response import Response


class VoiceHealthResponse(Response):
    """Readiness probe body.

    ``status`` is one of ``ok`` / ``loading`` / ``unavailable``. The remediation
    triple is populated only when voice is not ready, so a ready probe carries
    ``status`` alone and null siblings.
    """

    status: str
    reason: str | None = None
    missing: list[str] | None = None
    hint: str | None = None


class VoiceStateResponse(Response):
    """Playback state for a transcript row with no audio to return — the row's
    pipeline state (``pending`` while synthesis runs, ``failed`` once it gave up)."""

    state: str


class TranscriptionResponse(Response):
    """A successful speech-to-text result."""

    text: str
