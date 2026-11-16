"""Feature tests: ``text_to_speech`` speaks a text into an MP3 clip and answers
with the clip's full path, keeping at most 25 clips.

Every call goes through the real dispatcher and policy gate into the real
ability and the real Kokoro voice engine. Chalie's temp folder is the one thing
redirected, into pytest's ``tmp_path``, so no test can write to or delete from
the real temp folder.

Where the voice models are installed, each clip is decoded back from disk and
checked as speech, and the folder is counted after every call. Where they are
not, the tool must say so and write nothing, and every clip already in the
folder must be left alone. Each test asserts what holds in the environment it
runs in, so it is a real proof in both rather than a test that skips where it
matters.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import services.tmp_storage as tmp_storage
from abilities.text_to_speech import MAX_CLIPS
from configs.channels.external_agent import EAMPConfig
from configs.channels.skill_suggestion import SkillSuggestionConfig
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.voice_transcript_service import VoiceTranscriptService

pytestmark = pytest.mark.unit

_NOT_INSTALLED = "Text to speech is not available: the voice models are not installed."
_SPEECH = "Good morning. Your backup finished overnight and nothing needs your attention."


@pytest.fixture
def temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chalie's temp folder, redirected so the test owns every file in it."""
    folder = tmp_path / "temp"
    folder.mkdir()
    monkeypatch.setattr(tmp_storage, "TMP_PATH_PREFIX", f"{folder}{os.sep}{tmp_storage.TMP_PREFIX}")
    return folder


def _models_installed() -> bool:
    return not VoiceTranscriptService.missing_model_files()


def _speak(message: object, config: ProcessorConfig | None = None) -> str:
    """Make the call the model makes on *config*'s channel (a chat turn by
    default) and return the result the model reads."""
    mp = MessageProcessor(config or UserConfig(), raw_input="send me a voice note")
    return mp.dispatch_service.dispatch("text_to_speech", {"message": message})


def _head(result: str) -> str:
    return result.split("\n", 1)[0]


def _body(result: str) -> str:
    """The lines between the result's head and its ``[end:…]`` line."""
    lines = result.split("\n")
    assert lines[-1] == "[end:text_to_speech]", result
    return "\n".join(lines[1:-1])


def _clips(folder: Path) -> list[Path]:
    return sorted(folder.glob("chalie_tts_*.mp3"))


def _old_clips(folder: Path, count: int) -> list[Path]:
    """*count* clips already in the folder, oldest first, a minute apart."""
    paths = []
    for i in range(count):
        path = folder / f"chalie_tts_old{i:02d}.mp3"
        path.write_bytes(b"an earlier clip")
        stamp = 1_700_000_000 + 60 * i
        os.utime(path, (stamp, stamp))
        paths.append(path)
    return paths


def _assert_spoken_clip(result: str, folder: Path) -> Path:
    """The result is the full path of a new MP3 in *folder* that decodes to
    speech of a plausible length for :data:`_SPEECH`."""
    assert "status=success" in _head(result), result
    path = Path(_body(result))
    assert path.is_absolute(), result
    assert path.parent == folder, result
    assert path.name.startswith("chalie_tts_") and path.suffix == ".mp3", result
    samples, rate = sf.read(path, dtype="float32")
    seconds = len(samples) / rate
    assert 2.0 < seconds < 20.0, seconds
    assert float(np.max(np.abs(samples))) > VoiceTranscriptService.SILENCE_PEAK_THRESHOLD
    return path


def _assert_not_installed(result: str) -> None:
    assert "status=error" in _head(result) and "code=tts-unavailable" in _head(result), result
    assert _NOT_INSTALLED in result, result


def test_a_chat_turn_gets_back_the_full_path_of_a_spoken_mp3(
    db: sqlite3.Connection, temp_dir: Path,
) -> None:
    result = _speak(_SPEECH)

    if _models_installed():
        path = _assert_spoken_clip(result, temp_dir)
        assert _clips(temp_dir) == [path]
    else:
        _assert_not_installed(result)
        assert list(temp_dir.iterdir()) == []


@pytest.mark.parametrize(("already_there", "removed"), [(24, 0), (25, 1), (27, 3)])
def test_a_new_clip_deletes_the_oldest_until_25_remain(
    db: sqlite3.Connection, temp_dir: Path, already_there: int, removed: int,
) -> None:
    old = _old_clips(temp_dir, already_there)
    # Other temp files, and an MP3 this tool did not make, are never counted or touched.
    upload = temp_dir / "chalie_upload_report.pdf"
    upload.write_bytes(b"%PDF")
    other_mp3 = temp_dir / "chalie_song.mp3"
    other_mp3.write_bytes(b"not a clip")

    result = _speak(_SPEECH)

    if _models_installed():
        new = _assert_spoken_clip(result, temp_dir)
        assert len(_clips(temp_dir)) == MAX_CLIPS == 25
        assert sorted(_clips(temp_dir)) == sorted([*old[removed:], new])
    else:
        _assert_not_installed(result)
        assert _clips(temp_dir) == old
    assert upload.read_bytes() == b"%PDF" and other_mp3.read_bytes() == b"not a clip"


@pytest.mark.parametrize("message", ["<!-- draft -->", "![chart](https://example.com/chart.png)"])
def test_a_text_with_nothing_to_say_is_refused_and_no_clip_is_made(
    db: sqlite3.Connection, temp_dir: Path, message: str,
) -> None:
    old = _old_clips(temp_dir, MAX_CLIPS)

    result = _speak(message)

    if _models_installed():
        assert "status=error" in _head(result) and "code=invalid-param" in _head(result), result
        assert "There is nothing to speak in that text." in result, result
    else:
        _assert_not_installed(result)
    assert _clips(temp_dir) == old


def test_a_message_that_is_not_text_is_refused(db: sqlite3.Connection, temp_dir: Path) -> None:
    result = _speak(["good", "morning"])

    assert "status=error" in _head(result) and "code=invalid-param" in _head(result), result
    assert "'message' must be text." in result, result
    assert list(temp_dir.iterdir()) == []


def test_outside_agents_and_the_background_mind_may_make_a_clip_too(
    db: sqlite3.Connection, temp_dir: Path,
) -> None:
    for config in (EAMPConfig("reviewer", "sample-project", False), SkillSuggestionConfig()):
        result = _speak(_SPEECH, config)
        if _models_installed():
            _assert_spoken_clip(result, temp_dir)
        else:
            _assert_not_installed(result)

    assert db.execute("SELECT COUNT(*) FROM policy_blocked_log").fetchone()[0] == 0
