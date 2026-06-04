import pytest
from services import compaction_persistence, transcript_service
from services.database_service import get_shared_db_service

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


def test_get_context_limit_reads_declared_max_tokens_capped():
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.providers import MAX_CONTEXT_WINDOW
    from services.provider_db_service import ProviderDbService
    from services.provider_cache_service import ProviderCacheService
    db = get_shared_db_service()
    svc = ProviderDbService(db)
    sel = svc.get_selected_provider()
    if not sel:
        pytest.skip("no active provider in this env")
    pid = sel["id"]
    original = sel.get("max_tokens")
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "hi", None)
    mp.config = UserConfig()
    try:
        with db.connection() as conn:
            conn.execute("UPDATE providers SET max_tokens = 8000 WHERE id = ?", (pid,))
        ProviderCacheService.invalidate()
        assert mp.providers.get_context_limit() == 8000          # declared value honoured
        with db.connection() as conn:
            conn.execute("UPDATE providers SET max_tokens = 999999 WHERE id = ?", (pid,))
        ProviderCacheService.invalidate()
        assert mp.providers.get_context_limit() == MAX_CONTEXT_WINDOW   # capped at 200k
    finally:
        with db.connection() as conn:
            conn.execute("UPDATE providers SET max_tokens = ? WHERE id = ?", (original, pid))
        ProviderCacheService.invalidate()
