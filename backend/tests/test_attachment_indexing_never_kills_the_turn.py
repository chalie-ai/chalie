# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: an attachment that cannot be indexed must not take the turn
down with it.

Attaching a file makes the turn index it for search before the first send. That
indexing opens and writes the file-index database, so it can fail for reasons
that have nothing to do with the file — a failing disk surfaces as
``sqlite3.OperationalError: disk I/O error`` out of ``Database._open``. The
guard around it caught only the ``ValueError`` that content extraction raises,
so a database error escaped it, escaped the thread pool, escaped
``_seed_turn_zero`` and killed the whole turn: attaching an image was enough to
lose everything the user typed with it, with only a crashed execution row to
show for it.

The failure here is real, not simulated: the file-index database is pointed at
a file that is not a database, exactly as ``conftest`` points the main database
at a temp path, and ``sqlite3`` raises for itself. ``_seed_upload_attachment``,
``FileParserService.index``, ``FileIndexService`` and ``Database.conn`` all run
unmodified. No mocks.
"""

import sqlite3
from pathlib import Path

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.transcript import Transcript
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

_CHANNEL = "user"


def test_an_unopenable_file_index_does_not_kill_the_turn(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported crash, reproduced through the real indexing path: the turn
    survives it, keeps its input row, and the failure is reported rather than
    swallowed."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)

    corrupt_index = tmp_path / "file_index.db"
    corrupt_index.write_bytes(b"this is not a database")
    monkeypatch.setattr(FileMapperService, "get_file_index_db_path", lambda *_: corrupt_index)

    attachment = tmp_path / "notes.txt"
    attachment.write_text("the quick brown fox")

    # sanity: the redirect really does make the index unopenable, so a passing
    # test can never mean "nothing went wrong in the first place".
    with pytest.raises(sqlite3.DatabaseError):
        from services.file_index_service import FileIndexService  # noqa: PLC0415

        FileIndexService()

    mp = MessageProcessor(UserConfig(), raw_input="What do you make of this?")  # inert (I2)
    mp.turn_id = mp.transcript_service.allocate_turn()
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    mp.current_transcript_id = mp.uid

    mp._seed_upload_attachment(str(attachment))  # must not raise

    row = Transcript.filter("id", mp.uid).first()
    assert row is not None  # the user's message survived the indexing failure
    assert Transcript.filter("channel", _CHANNEL).filter("turn_id", mp.turn_id).exists()


def test_the_indexing_failure_is_reported_not_swallowed(
    db: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Surviving the failure must not mean hiding it — the turn continues, but
    the reason is logged at ERROR with its traceback, so a failing disk is
    still findable rather than silently degrading every attachment."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)

    corrupt_index = tmp_path / "file_index.db"
    corrupt_index.write_bytes(b"this is not a database")
    monkeypatch.setattr(FileMapperService, "get_file_index_db_path", lambda *_: corrupt_index)

    attachment = tmp_path / "notes.txt"
    attachment.write_text("the quick brown fox")

    mp = MessageProcessor(UserConfig(), raw_input="What do you make of this?")  # inert (I2)
    mp.turn_id = mp.transcript_service.allocate_turn()
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    mp.current_transcript_id = mp.uid

    with caplog.at_level("ERROR", logger="controllers.message_processor"):
        mp._seed_upload_attachment(str(attachment))

    failures = [r for r in caplog.records if "indexing" in r.getMessage()]
    assert failures, "the indexing failure was swallowed without a trace"
    assert failures[0].levelname == "ERROR"
    assert failures[0].exc_info is not None  # the traceback rides along
