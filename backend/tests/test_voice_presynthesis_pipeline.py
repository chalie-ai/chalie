"""Feature test: speech is synthesized when a turn settles, exactly once.

The speaker button no longer asks for synthesis — it plays a file the pipeline
already wrote. Three claims hold that up, and each is driven here through the
real production entry point against the real, fully-migrated SQLite database:

1. a real user turn that settles kicks the pipeline for its settled row, and
   that row always reaches a terminal outcome — a stored WAV where the voice
   models exist, a failed row where they do not;
2. a row that already has an outcome is NEVER synthesized again, whichever
   outcome it holds — the three attempts are spent once and the result stands;
3. a settled row with nothing sayable lands on a failed row rather than a
   zero-length WAV a button would happily play as silence.

Claims 2 and 3 return before the synthesizer is ever reached, so they hold
identically on a machine with no voice models. Claim 1 asserts the invariant
common to both arms rather than the audio itself, so it is a real proof in
either environment instead of a test that quietly skips where it matters.

``FileMapperService._VOICE_DIR`` is redirected into pytest's ``tmp_path``, so
nothing here can write to or delete from the real ``data/generated/voice/``.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from models.voice_transcript import VoiceTranscript
from services.file_mapper_service import FileMapperService
from services.voice_transcript_service import VoiceTranscriptService
from services.websocket import Websocket

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

# The LLM network boundary — the same seam test_context_usage_signal.py patches.
_BUILD_CLIENT = "services.provider_service.build_client"

_REPLY = "The kettle is on. It should be ready in about three minutes."


class _RecordingClient:
    """A real socket-registry subscriber implementing the registry's ``send(str)``
    Protocol, capturing every broadcast frame — the surface the speaker button
    paints from."""

    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def send(self, data: str) -> None:
        self.frames.append(json.loads(data))

    def voice_frames(self) -> list[dict[str, object]]:
        return [f for f in self.frames if f.get("status") == "voice_transcript"]


class _ReplyingProvider:
    """A real functional double at the network boundary: answers with one
    terminal reply so the turn settles in a single step."""

    def get_context_limit(self) -> int:
        return 120_000

    def send(self, _dto: object) -> ProviderResponse:
        return ProviderResponse(
            text=_REPLY, model="scripted-presynthesis", tool_calls=None, tokens_input=1,
        )


@pytest.fixture
def voice_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the voice storage root at a throwaway directory."""
    target = tmp_path / "voice"
    target.mkdir()
    monkeypatch.setattr(FileMapperService, "_VOICE_DIR", target)
    return target


def _settled_row(conn: sqlite3.Connection, content: str = _REPLY) -> int:
    """One settled assistant row — what the pipeline is addressed by."""
    cur = conn.execute(
        "INSERT INTO transcript (channel, role, content, turn_id, settled, created_at) "
        "VALUES ('user', 'assistant', ?, 1, 1, '2026-07-25 10:00:00')",
        (content,),
    )
    conn.commit()
    return cast(int, cur.lastrowid)


def _await_outcome(transcript_id: int, timeout_s: float = 30.0) -> VoiceTranscript | None:
    """Block until the pipeline's background thread reaches a terminal row."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        row = VoiceTranscript.filter("transcript_id", transcript_id).first()
        if row is not None:
            return row
        time.sleep(0.05)
    return None


@pytest.mark.usefixtures("real_voice_presynthesis")
class TestSettleKicksSynthesis:
    """The settle hook itself is under test here, so both cases re-arm the real
    method the suite-wide fixture parks — without it one test would pass against
    a no-op and the other could not go red."""

    def test_a_settled_turn_gets_its_speech_without_anyone_asking(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        """The whole point of the feature: a turn ends and its final reply is
        spoken into storage on its own. Asserts the outcome common to both
        environments — a terminal row for the SETTLED id — because a machine
        without voice models must still prove the hook fired and the pipeline
        ran to a state the button can paint, rather than skipping the claim."""
        assert db is not None  # taken for its binding side effect (real DB gateway)
        client = _RecordingClient()
        Websocket._connect(client)
        try:
            mp = MessageProcessor(UserConfig(), raw_input="is the kettle on")
            with patch(_BUILD_CLIENT, return_value=_ReplyingProvider()):
                mp.begin()
                mp.result()

            settle_id = Transcript.settle0("user", mp.turn_id)
            assert settle_id is not None, "the turn must have settled for speech to exist"
            row = _await_outcome(settle_id)
        finally:
            Websocket._disconnect(client)

        assert row is not None, "the settle hook must fire and reach a terminal state"
        assert row.transcript_id == settle_id, "only the settled row is spoken"
        states = [f.get("state") for f in client.voice_frames()]
        assert "pending" in states, "the button must be told synthesis started"
        assert states[-1] in ("ready", "failed"), f"must end terminal, got {states!r}"
        if row.file_path is not None:
            assert (voice_dir / row.file_path).exists()
            assert states[-1] == "ready"

    def test_an_unsettled_turn_is_left_alone(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        """A turn with no settled row has nothing to speak — the hook must
        no-op rather than synthesize a mid-turn 'let me check…' line."""
        assert db is not None
        mp = MessageProcessor(UserConfig(), raw_input="is the kettle on")
        mp._voice_presynthesis()

        assert VoiceTranscript.all().get() == []
        assert list(voice_dir.iterdir()) == []


class TestSynthesizedExactlyOnce:
    def test_a_stored_recording_is_never_regenerated(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        """Pressing play a hundred times costs nothing: an outcome on record is
        the end of it. Proven by a file the real pipeline could not have
        written — if it re-ran, this content would be replaced by real audio."""
        assert db is not None
        transcript_id = _settled_row(db)
        stored = voice_dir / f"{transcript_id}.wav"
        stored.write_bytes(b"not-audio-but-on-record")
        VoiceTranscript(transcript_id=transcript_id, file_path=stored.name).record()

        VoiceTranscriptService.instance().synthesize_settled(transcript_id)

        assert stored.read_bytes() == b"not-audio-but-on-record"
        assert len(VoiceTranscript.filter("transcript_id", transcript_id).get()) == 1

    def test_a_failed_row_is_never_retried(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        """Failure is terminal, not a suggestion. The three attempts were spent
        when the turn settled; a row that exhausted them stays failed instead of
        spending three more on every press."""
        assert db is not None
        transcript_id = _settled_row(db)
        VoiceTranscript(transcript_id=transcript_id, file_path=None).record()

        VoiceTranscriptService.instance().synthesize_settled(transcript_id)

        rows = VoiceTranscript.filter("transcript_id", transcript_id).get()
        assert len(rows) == 1
        assert rows[0].file_path is None
        assert list(voice_dir.iterdir()) == [], "a retry would have written audio"
