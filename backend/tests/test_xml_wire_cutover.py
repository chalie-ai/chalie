"""Phase C wire-format regression tests.
 
 These tests guard the specific regressions Phase C introduced:
 
 1. Transcript.append() sets xml_migrated=1 on every new INSERT.
 2. api/conversation.get_recent_history() returns content per message and
    does NOT return a blocks key (blocks array was the old wire format).
 
 All tests are @pytest.mark.unit: real SQLite (via db fixture), no mocks
 of production code.
 """

import sqlite3
from typing import cast

import pytest


@pytest.mark.unit
class TestTranscriptAppendSetsXmlMigrated:
    """Every new transcript row must land with xml_migrated=1.

    Regression: if a future INSERT site forgets the column, the Phase A
    boot migration would re-process those rows on next start and could
    corrupt already-correct XML content.
    """

    def test_append_sets_xml_migrated_1(self, db: sqlite3.Connection) -> None:
        from services.transcript_service import Transcript

        rowid = Transcript.append("ch-xml", "user", "Hello world")
        assert rowid is not None

        row = db.execute(
            "SELECT xml_migrated FROM transcript WHERE id = ?", (rowid,)
        ).fetchone()
        assert row is not None
        assert row[0] == 1, "append() must write xml_migrated=1"

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



@pytest.mark.unit
class TestConversationHistoryXmlWireFormat:
    """get_recent_history() returns content strings, never blocks arrays.

    Regression: the old implementation returned `blocks: [...]`. If someone
    adds blocks back (e.g. copying old code), the FE chat boot breaks because
    chat.js now expects `content` only.
    """

    def test_messages_have_content_key_not_blocks(self, db: sqlite3.Connection) -> None:
        from api.conversation import get_recent_history

        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES ('user', 'assistant', '<p>Hi</p>')"
        )
        db.commit()

        messages, _, _ = get_recent_history(limit=12, offset=0)
        assert len(messages) >= 1
        for msg in messages:
            assert "content" in msg, "Each message must have a 'content' key"
            assert "blocks" not in msg, (
                "Message must NOT have a 'blocks' key — blocks array is the old wire format"
            )

    def test_content_is_passed_through_verbatim(self, db: sqlite3.Connection) -> None:
        from api.conversation import get_recent_history

        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES ('user', 'assistant', '<p>Verbatim</p>')"
        )
        db.commit()

        messages, _, _ = get_recent_history(limit=12, offset=0)
        target = next((m for m in messages if "<p>Verbatim</p>" in cast(str, m.get("content", ""))), None)
        assert target is not None, "Content must be returned verbatim, not re-serialized"
