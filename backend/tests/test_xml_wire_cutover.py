"""Phase C wire-format regression tests.

 These tests guard the specific regression Phase C introduced:

 1. Every transcript write path sets xml_migrated=1 on every new INSERT.

 All tests are @pytest.mark.unit: real SQLite (via db fixture), no mocks
 of production code.
 """

import sqlite3

import pytest


@pytest.mark.unit
class TestTranscriptWritesSetXmlMigrated:
    """Every new transcript row must land with xml_migrated=1.

    Regression: if a future INSERT site forgets the column, the Phase A
    boot migration would re-process those rows on next start and could
    corrupt already-correct XML content.
    """

    def test_write_input_row_sets_xml_migrated_1(self, db: sqlite3.Connection) -> None:
        from services.transcript_service import Transcript

        rowid = Transcript.write_input_row("ch-xml", "user", "Input content")
        assert rowid is not None

        row = db.execute(
            "SELECT xml_migrated FROM transcript WHERE id = ?", (rowid,)
        ).fetchone()
        assert row is not None
        assert row[0] == 1, "write_input_row() must write xml_migrated=1"

    def test_write_assistant_row_sets_xml_migrated_1(self, db: sqlite3.Connection) -> None:
        from services.transcript_service import Transcript

        rowid = Transcript.write_assistant_row("ch-xml", "<p>Response</p>")
        assert rowid is not None

        row = db.execute(
            "SELECT xml_migrated FROM transcript WHERE id = ?", (rowid,)
        ).fetchone()
        assert row is not None
        assert row[0] == 1, "write_assistant_row() must write xml_migrated=1"
