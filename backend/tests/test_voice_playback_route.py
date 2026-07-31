"""Feature test: ``GET /api/voice/transcript/<id>`` serves pre-synthesized speech.

Speech is synthesized in the background the moment a turn's final reply settles,
so playback is a pure read of whatever the pipeline already stored. This test
drives the real Flask app against a real SQLite database and real files on disk
— no mocks — and pins the four outcomes the frontend's speaker button depends
on: the stored audio, the two terminal-failure shapes, and the lazy start that
covers rows predating the feature.

The voice directory is redirected at ``FileMapperService._VOICE_DIR`` so the
test writes and deletes only inside pytest's ``tmp_path`` — never the real
``data/generated/voice/``.
"""

import sqlite3
import time
from pathlib import Path
from typing import cast

import pytest
from flask.testing import FlaskClient

from models.voice_transcript import VoiceTranscript
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit


def _insert_transcript(
    conn: sqlite3.Connection, role: str, settled: int, content: str = "Hello there.",
) -> int:
    cur = conn.execute(
        "INSERT INTO transcript (channel, role, content, turn_id, settled, created_at) "
        "VALUES ('user', ?, ?, 1, ?, '2026-07-25 10:00:00')",
        (role, content, settled),
    )
    conn.commit()
    return cast(int, cur.lastrowid)


def _turn_messages() -> list[dict[str, object]]:
    """The seeded turn's rows as the thread-expand endpoint projects them."""
    from services.turn_serializer_service import get_service as _serializer

    return cast("list[dict[str, object]]", _serializer().serialize("user", 1)["messages"])


@pytest.fixture
def voice_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the voice storage root at a throwaway directory."""
    target = tmp_path / "voice"
    target.mkdir()
    monkeypatch.setattr(FileMapperService, "_VOICE_DIR", target)
    return target


class TestStoredAudio:
    def test_ready_row_serves_the_wav_with_an_immutable_cache(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        voice_dir: Path,
    ) -> None:
        client, conn, _store = authed_client
        transcript_id = _insert_transcript(conn, "assistant", 1)

        wav = b"RIFF$\x00\x00\x00WAVEfmt not-really-audio"
        (voice_dir / f"{transcript_id}.wav").write_bytes(wav)
        VoiceTranscript(
            transcript_id=transcript_id, file_path=f"{transcript_id}.wav",
        ).record()

        resp = client.get(f"/api/voice/transcript/{transcript_id}")

        assert resp.status_code == 200
        assert resp.mimetype == "audio/wav"
        assert resp.get_data() == wav
        # A row's audio never changes, so the browser may hold it forever —
        # a re-press must not cost a second round trip.
        assert "immutable" in resp.headers["Cache-Control"]

    def test_ready_row_whose_file_vanished_reports_failed(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        voice_dir: Path,
    ) -> None:
        """A row claiming audio that is gone must not 500 or re-synthesize."""
        client, conn, _store = authed_client
        transcript_id = _insert_transcript(conn, "assistant", 1)
        VoiceTranscript(
            transcript_id=transcript_id, file_path=f"{transcript_id}.wav",
        ).record()

        resp = client.get(f"/api/voice/transcript/{transcript_id}")

        assert resp.status_code == 409
        assert resp.get_json() == {"success": True, "result": {"state": "failed"}}


class TestTerminalFailure:
    def test_null_file_path_reports_failed(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        voice_dir: Path,
    ) -> None:
        """The pipeline exhausted its attempts — the button stays red forever."""
        client, conn, _store = authed_client
        transcript_id = _insert_transcript(conn, "assistant", 1)
        VoiceTranscript(transcript_id=transcript_id, file_path=None).record()

        resp = client.get(f"/api/voice/transcript/{transcript_id}")

        assert resp.status_code == 409
        assert resp.get_json() == {"success": True, "result": {"state": "failed"}}


class TestRowsThatCanNeverHaveSpeech:
    def test_unknown_transcript_is_404(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        voice_dir: Path,
    ) -> None:
        client, _conn, _store = authed_client
        assert client.get("/api/voice/transcript/999999").status_code == 404

    def test_unsettled_assistant_row_is_404(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        voice_dir: Path,
    ) -> None:
        """Mid-turn "let me check…" rows are never spoken — only the final reply is."""
        client, conn, _store = authed_client
        transcript_id = _insert_transcript(conn, "assistant", 0)
        assert client.get(f"/api/voice/transcript/{transcript_id}").status_code == 404

    def test_user_row_is_404(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        voice_dir: Path,
    ) -> None:
        client, conn, _store = authed_client
        transcript_id = _insert_transcript(conn, "user", 1)
        assert client.get(f"/api/voice/transcript/{transcript_id}").status_code == 404


class TestThreadProjection:
    """The reload path: a refetched turn must carry enough state to redraw the
    speaker button without replaying any websocket frame."""

    def test_settled_row_carries_its_voice_state_and_mid_turn_rows_do_not(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        interim = _insert_transcript(db, "assistant", 0, "Let me check…")
        final = _insert_transcript(db, "assistant", 1, "Here is the answer.")
        VoiceTranscript(transcript_id=final, file_path=f"{final}.wav").record()

        by_id = {m["id"]: m for m in _turn_messages()}

        assert by_id[str(final)]["settled"] is True
        assert by_id[str(final)]["voice_state"] == "ready"
        # An interim row gets no button at all — there is no audio for it.
        assert by_id[str(interim)]["settled"] is False
        assert by_id[str(interim)]["voice_state"] is None

    def test_settled_row_with_no_outcome_yet_reports_none(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        """Pre-feature history: the button renders normally and the first press
        starts the pipeline through the playback route."""
        final = _insert_transcript(db, "assistant", 1)

        message = next(m for m in _turn_messages() if m["id"] == str(final))
        assert message["settled"] is True
        assert message["voice_state"] is None

    def test_failed_row_reports_failed(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        final = _insert_transcript(db, "assistant", 1)
        VoiceTranscript(transcript_id=final, file_path=None).record()

        message = next(m for m in _turn_messages() if m["id"] == str(final))
        assert message["voice_state"] == "failed"


class TestLazyStart:
    def test_settled_row_with_no_outcome_starts_the_pipeline(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        voice_dir: Path,
    ) -> None:
        """History predating pre-synthesis: the first press starts the pipeline
        and answers 202, and the pipeline always lands on a terminal row — a
        stored WAV where the models exist, a failed row where they do not.
        Either way the button never sits pending forever."""
        client, conn, _store = authed_client
        transcript_id = _insert_transcript(conn, "assistant", 1)

        resp = client.get(f"/api/voice/transcript/{transcript_id}")

        assert resp.status_code == 202
        assert resp.get_json() == {"success": True, "result": {"state": "pending"}}

        deadline = time.monotonic() + 30
        row = None
        while time.monotonic() < deadline:
            row = VoiceTranscript.filter("transcript_id", transcript_id).first()
            if row is not None:
                break
            time.sleep(0.05)

        assert row is not None, "the pipeline must reach a terminal state, not hang"
        if row.file_path is not None:
            assert FileMapperService.get_voice_path(row.file_path).exists()
