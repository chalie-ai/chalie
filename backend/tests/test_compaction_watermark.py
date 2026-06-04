import pytest
from services import compaction_persistence, transcript_service

pytestmark = pytest.mark.integration


def _clear(db, channel):
    db.execute("DELETE FROM transcript WHERE channel = ?", (channel,))
    db.commit()


def test_get_compaction_reads_transcript_role_compaction(db):
    ch = "test_wm"
    _clear(db, ch)
    transcript_service.write_input_row(ch, "user", "hello one")
    transcript_service.write_input_row(ch, "assistant", "reply one")
    cid = transcript_service.write_input_row(ch, "compaction", "SUMMARY: one happened")

    row = compaction_persistence.get_compaction(ch)
    assert row is not None
    assert row["compacted_text"] == "SUMMARY: one happened"
    assert row["compacted_up_to_id"] == cid  # the row's OWN id is the watermark
    _clear(db, ch)


def test_previous_rows_excludes_through_watermark_and_has_no_limit(db):
    ch = "test_wm2"
    _clear(db, ch)
    for i in range(25):  # >20 — proves the old limit=20 bug is gone
        transcript_service.write_input_row(ch, "user", f"msg {i}")
    cid = transcript_service.write_input_row(ch, "compaction", "checkpoint")
    after = transcript_service.write_input_row(ch, "user", "after compaction")

    rows = transcript_service.get_recent(ch, since_id=cid)
    ids = [r["id"] for r in rows]
    assert after in ids
    assert cid not in ids               # watermark row itself excluded
    assert all(i > cid for i in ids)    # nothing at/below the watermark
    _clear(db, ch)
