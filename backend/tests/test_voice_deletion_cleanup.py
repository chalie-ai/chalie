"""Feature test: deleting a transcript deletes its stored speech.

A ``voice_transcript`` row cascades away with its transcript, but the WAV it
names is invisible to SQLite — nothing in the database can reach it once the
row is gone. Both delete paths therefore have to unlink the file *while* the
row still exists, and this test proves they do by driving the real sweep and
the real privacy endpoint against real files on disk. No mocks.

``FileMapperService._VOICE_DIR`` is redirected into pytest's ``tmp_path``, so
these deletions can only ever touch throwaway files — never the real
``data/generated/voice/``.
"""

import sqlite3
from pathlib import Path
from typing import cast

import pytest
from flask.testing import FlaskClient

from models.voice_transcript import VoiceTranscript
from services.file_mapper_service import FileMapperService
from services.garbage_collector_service import GarbageCollectorService

pytestmark = pytest.mark.unit

_WAV = b"RIFF$\x00\x00\x00WAVEfmt stand-in-for-real-audio"


@pytest.fixture
def voice_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the voice storage root at a throwaway directory."""
    target = tmp_path / "voice"
    target.mkdir()
    monkeypatch.setattr(FileMapperService, "_VOICE_DIR", target)
    return target


def _spoken_turn(conn: sqlite3.Connection, voice_dir: Path, turn_id: int, age_days: int) -> int:
    """Seed one complete turn of the given age with speech stored for its
    settled reply. Returns that reply's transcript id."""
    conn.execute(
        "INSERT INTO transcript (channel, role, content, turn_id, settled, created_at) "
        "VALUES ('user', 'user', 'question', ?, 0, datetime('now', ?))",
        (turn_id, f"-{age_days} days"),
    )
    cur = conn.execute(
        "INSERT INTO transcript (channel, role, content, turn_id, settled, created_at) "
        "VALUES ('user', 'assistant', 'answer', ?, 1, datetime('now', ?))",
        (turn_id, f"-{age_days} days"),
    )
    transcript_id = cast(int, cur.lastrowid)
    conn.commit()

    (voice_dir / f"{transcript_id}.wav").write_bytes(_WAV)
    VoiceTranscript(transcript_id=transcript_id, file_path=f"{transcript_id}.wav").record()
    return transcript_id


class TestRetentionSweep:
    def test_swept_transcript_takes_its_audio_with_it(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        """A turn old enough to be collected leaves nothing behind — no row,
        no file. Before the sweep knew about voice, the WAV survived forever
        with no row left pointing at it."""
        transcript_id = _spoken_turn(db, voice_dir, turn_id=1, age_days=200)
        wav = voice_dir / f"{transcript_id}.wav"
        assert wav.exists()

        GarbageCollectorService().run()

        assert not wav.exists(), "the swept transcript's audio must be unlinked"
        assert VoiceTranscript.filter("transcript_id", transcript_id).first() is None
        assert db.execute(
            "SELECT COUNT(*) FROM transcript WHERE id = ?", (transcript_id,),
        ).fetchone()[0] == 0

    def test_retained_transcript_keeps_its_audio(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        """The other half of the contract: a turn inside the retention window
        is untouched, so pressing play on recent history still works."""
        transcript_id = _spoken_turn(db, voice_dir, turn_id=2, age_days=1)

        GarbageCollectorService().run()

        assert (voice_dir / f"{transcript_id}.wav").exists()
        row = VoiceTranscript.filter("transcript_id", transcript_id).first()
        assert row is not None
        assert row.file_path == f"{transcript_id}.wav"

    def test_sweep_survives_an_already_missing_file(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        """A row whose file was removed out of band must not abort the sweep —
        retention would stall on one stray row forever."""
        transcript_id = _spoken_turn(db, voice_dir, turn_id=3, age_days=200)
        (voice_dir / f"{transcript_id}.wav").unlink()

        GarbageCollectorService().run()

        assert VoiceTranscript.filter("transcript_id", transcript_id).first() is None

    def test_sweep_clears_a_failed_row_with_no_file(
        self, db: sqlite3.Connection, voice_dir: Path,
    ) -> None:
        """A terminal-failure row (NULL ``file_path``) names no file at all;
        it still has to go when its transcript does."""
        db.execute(
            "INSERT INTO transcript (channel, role, content, turn_id, settled, created_at) "
            "VALUES ('user', 'assistant', 'answer', 4, 1, datetime('now', '-200 days'))",
        )
        db.commit()
        transcript_id = cast(
            int, db.execute("SELECT MAX(id) FROM transcript").fetchone()[0],
        )
        VoiceTranscript(transcript_id=transcript_id, file_path=None).record()

        GarbageCollectorService().run()

        assert VoiceTranscript.filter("transcript_id", transcript_id).first() is None


class TestPrivacyWipe:
    def test_delete_all_removes_every_stored_voice_file_and_row(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        voice_dir: Path,
    ) -> None:
        """"Delete all my data" means the audio too — stored speech is a
        verbatim recording of the conversation."""
        client, db, _store = authed_client
        _spoken_turn(db, voice_dir, turn_id=5, age_days=1)
        _spoken_turn(db, voice_dir, turn_id=6, age_days=1)
        assert len(list(voice_dir.iterdir())) == 2

        response = client.delete(
            "/api/privacy/delete-all", headers={"X-Confirm-Delete": "yes"},
        )

        assert response.status_code == 200
        assert list(voice_dir.iterdir()) == [], "no recording may survive the wipe"
        assert db.execute("SELECT COUNT(*) FROM voice_transcript").fetchone()[0] == 0
        # The directory itself stays — the next synthesis writes straight into it.
        assert voice_dir.is_dir()
