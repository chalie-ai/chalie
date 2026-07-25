"""DTOs for the voice namespace — the TTS/STT response contract.

These four shapes type the playback outcome for a transcript row's
pre-synthesized speech, the transcribe result, the always-200 readiness probe,
and the unavailable payload that the frozen in-module helpers already emit.
Grouped in one file because they form one cohesive namespace shape set.
"""

from __future__ import annotations

from .base import DTO


class VoiceState(DTO):
    """Read shape for a playback request that has no audio to return — the
    transcript row's pipeline state (``pending`` while synthesis runs,
    ``failed`` once it gave up)."""

    state: str


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
