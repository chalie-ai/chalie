"""Phase C wire-format regression tests.

These tests guard the specific regressions Phase C introduced:

1. transcript_service.append() sets xml_migrated=1 on every new INSERT.
2. output_service.enqueue_text() pipes the response through sanitize() and
   the frame contains `content`, not `blocks`. Mixed plain-text + HTML
   responses keep their tags intact (no entity-escaping heuristic).
3. api/conversation.get_recent_history() returns `content` per message and
   does NOT return a `blocks` key — blocks array was the old wire format.

All tests are @pytest.mark.unit: real SQLite (via `db` fixture), real
MemoryStore (via `store` fixture), no mocks of production code.
"""

import json

import pytest


@pytest.mark.unit
class TestTranscriptAppendSetsXmlMigrated:
    """Every new transcript row must land with xml_migrated=1.

    Regression: if a future INSERT site forgets the column, the Phase A
    boot migration would re-process those rows on next start and could
    corrupt already-correct XML content.
    """

    def test_append_sets_xml_migrated_1(self, db):
        from services.transcript_service import append

        rowid = append("ch-xml", "user", "Hello world")
        assert rowid is not None

        row = db.execute(
            "SELECT xml_migrated FROM transcript WHERE id = ?", (rowid,)
        ).fetchone()
        assert row is not None
        assert row[0] == 1, "append() must write xml_migrated=1"

    def test_write_input_row_sets_xml_migrated_1(self, db):
        from services.transcript_service import write_input_row

        rowid = write_input_row("ch-xml", "user", "Input content")
        assert rowid is not None

        row = db.execute(
            "SELECT xml_migrated FROM transcript WHERE id = ?", (rowid,)
        ).fetchone()
        assert row is not None
        assert row[0] == 1, "write_input_row() must write xml_migrated=1"

    def test_write_assistant_row_sets_xml_migrated_1(self, db):
        from services.transcript_service import write_assistant_row

        rowid = write_assistant_row("ch-xml", "<p>Response</p>")
        assert rowid is not None

        row = db.execute(
            "SELECT xml_migrated FROM transcript WHERE id = ?", (rowid,)
        ).fetchone()
        assert row is not None
        assert row[0] == 1, "write_assistant_row() must write xml_migrated=1"


@pytest.mark.unit
class TestOutputServiceXmlWireFrame:
    """enqueue_text() must produce a sanitised `content` string, never `blocks`.

    Regression:
    - If someone re-adds `blocks` to the event payload, the FE renderer breaks.
    - If a wrap-as-<p>-and-escape heuristic is reintroduced, mixed plain-text +
      HTML responses (the model's normal output) get their tags escaped into
      entities and render as literal `<h1>` text in the chat.
    """

    @pytest.fixture
    def svc(self, store):
        from services.output_service import OutputService

        yield OutputService()

    def test_plaintext_response_passes_through(self, svc, store):
        """Plain text without any tags is valid HTML — pass through, no wrapping."""
        output_id = svc.enqueue_text(
            channel="ch-wrap",
            response="Plain text from LLM",
            original_metadata={"uuid": "u1"},
        )
        raw = store.get(f"output:{output_id}")
        assert raw is not None
        frame = json.loads(raw)
        assert frame["metadata"]["content"] == "Plain text from LLM"

    def test_xml_response_passed_through_unchanged(self, svc, store):
        """A response that is already HTML must not be double-wrapped."""
        output_id = svc.enqueue_text(
            channel="ch-pass",
            response="<p>Already XML</p>",
            original_metadata={"uuid": "u2"},
        )
        frame = json.loads(store.get(f"output:{output_id}"))
        assert frame["metadata"]["content"] == "<p>Already XML</p>"

    def test_mixed_text_and_html_passes_through(self, svc, store):
        """Mixed conversational text + HTML tags must render as-is — the bug
        that motivated dropping the is_xml_content heuristic. Tags must remain
        as tags, not be escaped into entities."""
        response = "Alright, Dylan. <h1>What executed</h1> <ul><li>search</li></ul>"
        output_id = svc.enqueue_text(
            channel="ch-mix",
            response=response,
            original_metadata={"uuid": "u4"},
        )
        frame = json.loads(store.get(f"output:{output_id}"))
        content = frame["metadata"]["content"]
        assert "<h1>What executed</h1>" in content
        assert "<ul>" in content and "<li>search</li>" in content
        assert "&lt;h1&gt;" not in content, (
            f"<h1> was escaped to entities — heuristic regressed: {content!r}"
        )

    def test_disallowed_tags_stripped(self, svc, store):
        """sanitize() must strip tags outside the allowlist (e.g. <script>)."""
        output_id = svc.enqueue_text(
            channel="ch-strip",
            response="Hi <script>alert(1)</script><p>safe</p>",
            original_metadata={"uuid": "u5"},
        )
        frame = json.loads(store.get(f"output:{output_id}"))
        content = frame["metadata"]["content"]
        assert "<script>" not in content
        assert "alert(1)" not in content
        assert "<p>safe</p>" in content

    def test_event_payload_has_content_not_blocks(self, svc, store):
        """The SSE event payload dict must contain `content`, not `blocks`."""
        published = []
        real_publish = store.publish

        def spy(ch, msg):
            published.append((ch, msg))
            return real_publish(ch, msg)

        store.publish = spy

        svc.enqueue_text(
            channel="ch-event",
            response="<p>Hello</p>",
            original_metadata={"uuid": "u3"},
        )

        # SSE payload goes to sse:{uuid}; the message is just the output_id.
        # For background (no uuid), payload is the JSON dict. Test with background
        # path to inspect the event payload dict.
        published.clear()
        from unittest.mock import patch
        with patch("api.push.send_push_to_all"):
            svc.enqueue_text(
                channel="ch-event",
                response="<p>Background</p>",
                original_metadata={"source": "proactive"},
            )

        # Find the output:events publish
        output_events = [(ch, msg) for ch, msg in published if ch == "output:events"]
        assert output_events, "Expected a publish to output:events"
        payload = json.loads(output_events[0][1])
        assert "content" in payload, "Event payload must contain 'content'"
        assert "blocks" not in payload, "Event payload must NOT contain 'blocks' (old wire format)"


@pytest.mark.unit
class TestConversationHistoryXmlWireFormat:
    """get_recent_history() returns content strings, never blocks arrays.

    Regression: the old implementation returned `blocks: [...]`. If someone
    adds blocks back (e.g. copying old code), the FE chat boot breaks because
    chat.js now expects `content` only.
    """

    def test_messages_have_content_key_not_blocks(self, db):
        from api.conversation import get_recent_history

        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES ('user', 'assistant', '<p>Hi</p>')"
        )
        db.commit()

        messages, _ = get_recent_history(limit=12, offset=0)
        assert len(messages) >= 1
        for msg in messages:
            assert "content" in msg, "Each message must have a 'content' key"
            assert "blocks" not in msg, (
                "Message must NOT have a 'blocks' key — blocks array is the old wire format"
            )

    def test_content_is_passed_through_verbatim(self, db):
        """The XML string stored in transcript.content is not re-processed."""
        from api.conversation import get_recent_history

        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES ('user', 'assistant', '<p>Verbatim</p>')"
        )
        db.commit()

        messages, _ = get_recent_history(limit=12, offset=0)
        target = next((m for m in messages if "<p>Verbatim</p>" in m.get("content", "")), None)
        assert target is not None, "Content must be returned verbatim, not re-serialized"
