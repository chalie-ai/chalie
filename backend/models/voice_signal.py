"""Transient WS frames the voice pre-synthesis pipeline emits (§4.1).

One lean kind with no persisted counterpart of its own: the pipeline
broadcasts a state transition for a settled transcript row's audio —
``pending`` (synthesis started, fade the speaker button), ``ready`` (stored
WAV available, enable it) or ``failed`` (all attempts exhausted, turn it red).
Fixed wire shape the frontend router discriminates on ``status``:
``{status: "voice_transcript", transcript_id, state}``. Subclasses
:class:`~models.ws_message.WsMessage` — kwargs field storage plus the
all-public-fields ``to_dict`` it inherits, no ``save``, no service reach; the
owning service hands one to ``Websocket.broadcast`` (Rule 9).
"""

from __future__ import annotations

from typing import Self

from models.ws_message import WsMessage


class VoiceSignal(WsMessage):
    """A lean transient voice-pipeline frame: ``pending``, ``ready`` or ``failed``."""

    @classmethod
    def pending(cls, transcript_id: int) -> Self:
        """Synthesis started for this row — the speaker button fades."""
        return cls(status="voice_transcript", transcript_id=transcript_id, state="pending")

    @classmethod
    def ready(cls, transcript_id: int) -> Self:
        """Stored audio is available — the speaker button plays it."""
        return cls(status="voice_transcript", transcript_id=transcript_id, state="ready")

    @classmethod
    def failed(cls, transcript_id: int) -> Self:
        """Every attempt exhausted — the speaker button turns red, disabled."""
        return cls(status="voice_transcript", transcript_id=transcript_id, state="failed")
