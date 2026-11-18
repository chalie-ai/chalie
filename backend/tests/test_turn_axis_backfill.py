"""Feature tests: a 1.3.1 database keeps its checkpoints and thread labels on upgrade.

1.3.1 found a checkpoint and a thread label by turn number: ``compactions``
carried ``for_turn_id`` and ``compacted_up_to`` (a turn number on the spine,
a row id in a thread), and ``thread_gist`` was keyed by ``(channel, turn_id)``.
The running release finds both by transcript row id. Copy-forward computes the
new columns from the old database's transcript, so the first turn after the
upgrade sees the conversation 1.3.1 showed.

Each old database is built the way 1.3.1 built its own: the real
``provision()`` under the old version, then ``compactions`` and
``thread_gist`` recreated with 1.3.1's DDL (frozen history, copied verbatim
from that release's ``schema.sql``) and filled with raw rows. The upgrade is
the real ``provision()`` under the running version, and every assertion reads
through the production readers: the history read, the checkpoint envelope and
the thread label lookup. The only redirection is ``FileMapperService``'s
paths, into this test's temp directory.

The expected views are written out by hand from 1.3.1's read rule, not derived
from the code under test. The spine showed the non-memory rows of the turns
after the checkpoint's turn, each turn cut at its first settled answer, and no
row without a turn; a thread showed its own rows after its checkpoint's row.
"""

import logging
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.compaction import Compaction
from models.scheduled_item import ScheduledItem
from models.thread_gist import ThreadGist
from services.app_version import get_version, version_sort_key
from services.database import Database
from services.file_mapper_service import FileMapperService
from services.snapshot_service import SnapshotService
from services.versioned_database_service import VersionedDatabaseService
from tests.test_versioned_database_service import _connect, _lineage

pytestmark = pytest.mark.unit

# Read before any test redirects the VERSION file.
_RUNNING_VERSION = get_version()
_LEGACY_VERSION = "1.3.1-beta"

_LEGACY_THREAD_GIST = """
CREATE TABLE IF NOT EXISTS thread_gist (
    channel     TEXT NOT NULL,
    turn_id     INTEGER NOT NULL,
    gist        TEXT,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (channel, turn_id)
);
"""
_LEGACY_COMPACTIONS = """
CREATE TABLE IF NOT EXISTS compactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel         TEXT NOT NULL,
    for_turn_id     INTEGER,
    compacted_up_to INTEGER NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_compactions_scope ON compactions(channel, for_turn_id, id);
"""

_CHAT = UserConfig().channel
_SCHEDULE = ScheduledItem.SCHEDULE_CHANNEL
_LABELLED_AT = "2026-04-01 09:00:00"

_SPINE_SUMMARY = "The person is planning spring in Lisbon and packing light."
_THREAD_SUMMARY = "A short day plan for Sintra."
_ORPHAN_SUMMARY = "A thread with no rows on this channel."

Row = tuple[int, str, int | None, str, int, str]

# (id, channel, turn_id, role, settled, content) as 1.3.1 left them.
_TRANSCRIPT: list[Row] = [
    (1, _CHAT, 1, "user", 0, "Where should I go in spring?"),
    (2, _CHAT, 1, "assistant", 1, "Lisbon is lovely in April."),
    (3, _CHAT, 2, "user", 0, "What should I pack?"),
    # The memory step that followed turn 1 opened a turn of its own while
    # turn 2 was still working, so its row sits below turn 2's answer.
    (4, _CHAT, 3, "memory", 0, "Recall what the person said about Lisbon."),
    (5, _CHAT, 2, "assistant", 1, "Light layers and good shoes."),
    (6, _CHAT, 4, "user", 0, "Any restaurant tips?"),
    (7, _CHAT, 4, "assistant", 1, "Try the custard tarts in Belem."),
    (8, _CHAT, None, "user", 0, "A row that belongs to no turn."),
    (9, _CHAT, 5, "memory", 0, "Recall what the person said about food."),
    (10, _CHAT, 6, "user", 0, "Plan a day trip to Sintra."),
    (11, _CHAT, 6, "assistant", 1, "Here is a Sintra plan."),
    # Replies into turn 6: the thread, past the turn's first settled answer.
    (12, _CHAT, 6, "user", 0, "Make it shorter."),
    (13, _CHAT, 6, "assistant", 1, "A shorter Sintra plan."),
    (14, _CHAT, 6, "user", 0, "Add a lunch stop."),
    (15, _CHAT, 6, "assistant", 1, "The Sintra plan with lunch."),
    (16, _CHAT, 7, "user", 0, "And what about Porto?"),
    (17, _SCHEDULE, 3, "user", 0, "Morning briefing."),
    (18, _SCHEDULE, 3, "assistant", 1, "Here is today's briefing."),
]

# (id, channel, for_turn_id, compacted_up_to, content): the spine folded
# through turn 2, turn 6's thread folded through row 13, and turn 40 has no
# rows on the channel at all.
_COMPACTIONS: list[tuple[int, str, int | None, int, str]] = [
    (1, _CHAT, None, 2, _SPINE_SUMMARY),
    (2, _CHAT, 6, 13, _THREAD_SUMMARY),
    (3, _CHAT, 40, 15, _ORPHAN_SUMMARY),
]

# (channel, turn_id, gist)
_GISTS: list[tuple[str, int, str]] = [
    (_CHAT, 1, "Spring in Lisbon"),
    (_CHAT, 6, "Sintra Day Trip"),
    (_SCHEDULE, 3, "Morning Briefing"),
]

# 1.3.1's spine read with its checkpoint at turn 2: turns 4, 6 and 7 (3 and 5
# hold only memory rows), turn 6 cut at its first settled answer (row 11),
# turn 7 never settled and so read whole, and the row with no turn left out.
_SPINE_AS_1_3_1_SHOWED = [
    "Any restaurant tips?",
    "Try the custard tarts in Belem.",
    "Plan a day trip to Sintra.",
    "Here is a Sintra plan.",
    "And what about Porto?",
]
# 1.3.1's thread read with its checkpoint at row 13: the thread's rows after it.
_THREAD_AS_1_3_1_SHOWED = ["Add a lunch stop.", "The Sintra plan with lunch."]


@pytest.fixture(autouse=True)
def _close_gateway() -> Iterator[None]:
    """Drop this thread's connections to the test's files once the test ends."""
    yield
    Database.close()


def _point_instance_at(monkeypatch: pytest.MonkeyPatch, data: Path) -> None:
    """Point every path provisioning and a snapshot can reach into *data*.

    ``get_db_path`` stays real: it resolves through the VERSION file written
    here, so the running version picks its own file name as a live install
    does.
    """
    data.mkdir(parents=True)
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", data)
    monkeypatch.setattr(FileMapperService, "_SECURE_DIR", data / "secure")
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", data / "docs")
    monkeypatch.setattr(FileMapperService, "_USER_SKILLS_DIR", data / "skills" / "user")
    monkeypatch.setattr(FileMapperService, "get_version_path", lambda *_: data / "VERSION")
    monkeypatch.setattr(FileMapperService, "get_skills_db_path", lambda *_: data / "unused-skills.sqlite")
    monkeypatch.setattr(
        FileMapperService, "get_mcp_tools_db_path", lambda *_: data / "unused-mcp-tools.sqlite"
    )


def _boot(version: str) -> Path:
    """Boot *version*'s provisioning and return the file it opens."""
    FileMapperService.get_version_path().write_text(version + "\n", encoding="utf-8")
    VersionedDatabaseService().provision()
    return FileMapperService.get_db_path()


def _insert_gists(conn: sqlite3.Connection, gists: list[tuple[str, int, str]]) -> None:
    conn.executemany(
        "INSERT INTO thread_gist (channel, turn_id, gist, created_at) VALUES (?, ?, ?, ?)",
        [(channel, turn_id, gist, _LABELLED_AT) for channel, turn_id, gist in gists],
    )


def _legacy_install(
    transcript: list[Row],
    compactions: list[tuple[int, str, int | None, int, str]],
    gists: list[tuple[str, int, str]],
) -> Path:
    """A 1.3.1 install's database holding *transcript*, *compactions* and *gists*."""
    path = _boot(_LEGACY_VERSION)
    conn = _connect(path)
    try:
        conn.executescript(
            "DROP TABLE thread_gist; DROP TABLE compactions;"
            + _LEGACY_THREAD_GIST
            + _LEGACY_COMPACTIONS
        )
        conn.executemany(
            "INSERT INTO transcript (id, channel, turn_id, role, settled, content, xml_migrated) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            transcript,
        )
        conn.executemany(
            "INSERT INTO compactions (id, channel, for_turn_id, compacted_up_to, content) "
            "VALUES (?, ?, ?, ?, ?)",
            compactions,
        )
        _insert_gists(conn, gists)
    finally:
        conn.close()
    return path


def _open(target: Path) -> None:
    """Bind the model gateway to *target*, the boot step after provisioning."""
    assert FileMapperService.get_db_path() == target
    Database.close()
    Database().bind()


def _history(mp: MessageProcessor) -> list[str]:
    return [r.content for r in mp.transcript_service.read()]


def _sources(target: Path) -> list[tuple[str, str | None]]:
    """``(version, source file)`` of every lineage row *target* carries."""
    return [(version, source) for version, source, _completed, _failed in _lineage(target)]


def _assert_conversation_reads_as_1_3_1_showed() -> None:
    spine = MessageProcessor(UserConfig(), raw_input="next")
    body = spine.compaction_service.checkpoint("next")
    assert _SPINE_SUMMARY in body, f"the first turn after the upgrade lost the spine's checkpoint: {body!r}"
    assert _THREAD_SUMMARY not in body and _ORPHAN_SUMMARY not in body, (
        f"a thread's checkpoint was read as the spine's: {body!r}"
    )
    assert _history(spine) == _SPINE_AS_1_3_1_SHOWED, "the spine does not show what 1.3.1 showed"

    reply = MessageProcessor(UserConfig(), turn_id=6, raw_input="reply")
    body = reply.compaction_service.checkpoint("reply")
    assert _THREAD_SUMMARY in body, f"a reply into the thread lost the thread's checkpoint: {body!r}"
    assert _SPINE_SUMMARY not in body, f"a reply into the thread read the spine's checkpoint: {body!r}"
    assert _history(reply) == _THREAD_AS_1_3_1_SHOWED, "the thread does not show what 1.3.1 showed"

    assert ThreadGist.bulk_get(_CHAT, [1, 3, 6]) == {1: "Spring in Lisbon", 6: "Sintra Day Trip"}, (
        "the chat threads lost their labels, or one leaked onto a turn of another channel"
    )
    assert ThreadGist.bulk_get(_SCHEDULE, [3]) == {3: "Morning Briefing"}, (
        "the schedule thread lost its label"
    )

    orphan = Compaction.filter("content", _ORPHAN_SUMMARY).first()
    assert orphan is not None, "the checkpoint of a thread with no rows was not carried forward"
    assert (orphan.opening_id, orphan.compacted_up_to_id) == (None, None), (
        "a checkpoint whose thread has no rows must keep neither column, or it reads as the spine's"
    )


def test_the_first_turn_after_upgrading_from_1_3_1_sees_what_1_3_1_showed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spine, a thread and both channels' labels read as they did in 1.3.1."""
    _point_instance_at(monkeypatch, tmp_path / "data")
    _legacy_install(_TRANSCRIPT, _COMPACTIONS, _GISTS)

    _open(_boot(_RUNNING_VERSION))

    _assert_conversation_reads_as_1_3_1_showed()


def test_where_1_3_1_turns_overlapped_a_folded_row_shows_again_and_none_is_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn 2 opened before turn 1 answered and the spine folded turn 1. A
    single row-id cut cannot fold turn 1's answer without hiding turn 2's
    question, so the answer shows again and every row 1.3.1 showed still shows."""
    _point_instance_at(monkeypatch, tmp_path / "data")
    _legacy_install(
        [
            (1, _CHAT, 1, "user", 0, "First question."),
            (2, _CHAT, 2, "user", 0, "Second question."),
            (3, _CHAT, 1, "assistant", 1, "First answer."),
            (4, _CHAT, 2, "assistant", 1, "Second answer."),
        ],
        [(1, _CHAT, None, 1, _SPINE_SUMMARY)],
        [],
    )

    _open(_boot(_RUNNING_VERSION))

    spine = MessageProcessor(UserConfig(), raw_input="next")
    history = _history(spine)
    assert _SPINE_SUMMARY in spine.compaction_service.checkpoint("next")
    assert [row for row in history if row != "First answer."] == ["Second question.", "Second answer."], (
        f"a row 1.3.1 showed is hidden after the upgrade: {history}"
    )
    assert "First answer." in history, f"a folded turn's row above the cut is hidden: {history}"


def test_a_1_3_1_snapshot_restored_onto_a_fresh_install_reads_as_1_3_1_showed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export from a 1.3.1 install, restore onto a fresh install of the running
    release, boot: the restored conversation reads as it did in 1.3.1."""
    _point_instance_at(monkeypatch, tmp_path / "old-install" / "data")
    legacy = _legacy_install(_TRANSCRIPT, _COMPACTIONS, _GISTS)
    zip_path = SnapshotService().export(password=None)
    try:
        _point_instance_at(monkeypatch, tmp_path / "new-install" / "data")
        fresh = _boot(_RUNNING_VERSION)
        assert _sources(fresh) == [(_RUNNING_VERSION, None)], "the new install was not a fresh one"

        SnapshotService().stage_import(zip_path, password=None)
        SnapshotService.apply_pending()
        target = _boot(_RUNNING_VERSION)
    finally:
        shutil.rmtree(zip_path.parent)

    assert (_RUNNING_VERSION, legacy.name) in _sources(target), (
        "the restored 1.3.1 database was not the source of the running release's file"
    )
    _open(target)
    _assert_conversation_reads_as_1_3_1_showed()


def test_a_label_table_the_upgrade_cannot_read_stops_boot_and_the_next_boot_rebuilds_from_the_untouched_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The old label table holds turn numbers but not the channel they belong
    to, so its labels cannot be matched to a row. Boot stops and says the old
    database is intact; it is. Once the table is whole again the next boot
    sets the half-built file aside and carries everything forward."""
    _point_instance_at(monkeypatch, tmp_path / "data")
    source = _legacy_install(_TRANSCRIPT, _COMPACTIONS, [])
    conn = _connect(source)
    try:
        conn.executescript(
            "DROP TABLE thread_gist;"
            "CREATE TABLE thread_gist (turn_id INTEGER NOT NULL, gist TEXT, created_at TEXT NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO thread_gist (turn_id, gist, created_at) VALUES (?, ?, ?)",
            [(turn_id, gist, _LABELLED_AT) for _channel, turn_id, gist in _GISTS],
        )
        before = list(conn.iterdump())
    finally:
        conn.close()

    with (
        caplog.at_level(logging.ERROR, logger="services.versioned_database_service"),
        pytest.raises(RuntimeError),
    ):
        _boot(_RUNNING_VERSION)

    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("thread_gist" in m and source.name in m and "intact" in m for m in errors), (
        f"the log does not name the table and say the old database is intact: {errors}"
    )
    conn = _connect(source)
    try:
        assert list(conn.iterdump()) == before, "the failed upgrade changed the old database"
    finally:
        conn.close()
    target = FileMapperService.get_db_path()
    assert target.exists(), "the failed boot left no file to set aside"
    assert [s for s in _sources(target) if s[0] == _RUNNING_VERSION] == [], (
        "a file whose upgrade failed carries a lineage row, so no boot would rebuild it"
    )

    conn = _connect(source)
    try:
        conn.executescript("DROP TABLE thread_gist;" + _LEGACY_THREAD_GIST)
        _insert_gists(conn, _GISTS)
    finally:
        conn.close()

    _boot(_RUNNING_VERSION)

    asides = [
        p for p in target.parent.glob(f"{target.name}.incomplete-*")
        if not p.name.endswith(("-wal", "-shm"))
    ]
    assert len(asides) == 1, f"the half-built file was not set aside exactly once: {asides}"
    assert [s for s in _sources(target) if s[0] == _RUNNING_VERSION] == [(_RUNNING_VERSION, source.name)], (
        "the rebuilt file was not carried forward from the 1.3.1 database"
    )
    _open(target)
    _assert_conversation_reads_as_1_3_1_showed()


def test_a_database_already_carried_forward_carries_forward_again_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release after this one upgrades from a file whose checkpoints and
    labels are already keyed by row id: they copy as they are and read the same."""
    next_release = f"{_RUNNING_VERSION}.1"
    assert version_sort_key(next_release) > version_sort_key(_RUNNING_VERSION)
    _point_instance_at(monkeypatch, tmp_path / "data")
    _legacy_install(_TRANSCRIPT, _COMPACTIONS, _GISTS)
    carried = _boot(_RUNNING_VERSION)

    target = _boot(next_release)

    assert (next_release, carried.name) in _sources(target)
    _open(target)
    _assert_conversation_reads_as_1_3_1_showed()
