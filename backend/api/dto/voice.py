"""DTOs for the voice namespace — the TTS/STT request and response contract.

Voice is not CRUD: there is no resource and no DB row. These four shapes type
the inbound TTS body, the transcribe result, the always-200 readiness probe, and
the unavailable payload that the frozen in-module helpers already emit. Grouped
in one file because they form one cohesive namespace shape set.
"""

from __future__ import annotations

from pydantic import Field

from .base import DTO


class TtsRequest(DTO):
    """Inbound body for text-to-speech synthesis."""

    text: str = Field(..., min_length=1)


class Transcription(DTO):
    """Read shape for a successful transcription."""

    text: str


class VoiceHealth(DTO):
    """Read shape for the always-200 readiness probe.

    The optional fields carry the remediation triple only when voice is not
    ready; a ready probe is serialized as just ``{"status": "ok"}`` (the handler
    drops ``None`` fields so the wire shape never grows null keys).
    """

    status: str
    reason: str | None = None
    missing: list[str] | None = None
    hint: str | None = None


class VoiceUnavailable(DTO):
    """503 body emitted when voice deps or models are missing or still loading."""

    error: str
    reason: str | None = None
    missing: list[str] | None = None
    hint: str | None = None
