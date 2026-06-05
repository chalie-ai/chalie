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


def _seed_selected_ollama(db, max_tokens):
    """Seed an offline Ollama provider and mark it selected, returning its id.

    Ollama's build_request_body / get_context_limit (via declared max_tokens)
    run with zero network, so _fit_request exercises the real provider stack
    without a live model. send_messages is never reached by _fit_request."""
    cur = db.execute(
        "INSERT INTO providers (name, platform, model, host, max_tokens) "
        "VALUES ('fit-test', 'ollama', 'fit-model', 'http://localhost:11434', ?)",
        (max_tokens,),
    )
    pid = cur.lastrowid
    db.execute(
        "INSERT INTO settings (key, value) VALUES ('selected_provider_id', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(pid),),
    )
    db.commit()
    return pid


def _build_send_inputs(mp):
    """Scaffold the exact (system, tools, provider) Providers.send() feeds to
    _fit_request — real stack, no mocks."""
    from abilities._registry import AbilityRegistry
    provider = mp.providers.selected_provider()
    system = mp.config.get_system_prompt(mp)
    tools = AbilityRegistry.build_tools(mp)
    return system, tools, provider


def test_fit_request_trims_oldest_until_request_reserves_headroom(db):
    """Trim-then-compact (design §3.3): when the full request overflows the
    window, _fit_request drops the OLDEST history rows one at a time until the
    request reserves max(10% window, 8k) response headroom, flags
    _compaction_pending for the loop to compact, and TERMINATES (the hang Dylan's
    pivot fixes)."""
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService
    from services.llm_service import estimate_tokens

    ch = "user"
    _clear(db, ch)
    # estimate_tokens counts WORDS (split * 1.3), so rows must be word-rich:
    # 40 rows * ~400 words * 1.3 ≈ 20.8k tok, overflowing the 12k cap.
    n_rows = 40
    big = " ".join(f"w{j}" for j in range(400))
    for i in range(n_rows):
        transcript_service.write_input_row(ch, "user", f"row{i:03d} {big}")
    _seed_selected_ollama(db, 20000)
    ProviderCacheService.invalidate()

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what should I do next?", None)
    mp.config = UserConfig()
    try:
        system, tools, provider = _build_send_inputs(mp)
        window = mp.providers.get_context_limit()
        assert window == 20000
        cap = window - max(int(0.10 * window), 8000)   # 12000

        user = mp.providers._fit_request(system, tools, provider)

        # Had to trim → loop must compact the dropped rows into the checkpoint.
        assert mp._compaction_pending is True
        # Partial trim that terminated (not the whole history, not zero).
        assert 0 < mp._history_drop < n_rows
        # The fitted request honours the response-headroom guarantee.
        body = provider.build_request_body(system, [{"role": "user", "content": user}], tools)
        assert estimate_tokens(body) <= cap
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_fit_request_no_trim_when_request_already_fits(db):
    """When the request already fits, _fit_request drops nothing and leaves
    _compaction_pending False — no needless compaction."""
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

    ch = "user"
    _clear(db, ch)
    for i in range(3):
        transcript_service.write_input_row(ch, "user", f"short {i}")
    _seed_selected_ollama(db, 20000)
    ProviderCacheService.invalidate()

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "hi", None)
    mp.config = UserConfig()
    try:
        system, tools, provider = _build_send_inputs(mp)
        mp.providers._fit_request(system, tools, provider)

        assert mp._compaction_pending is False
        assert mp._history_drop == 0
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_substitute_provider_content_field_uses_mp_providers():
    from configs.channels._common import substitute_provider_content_field, _CONTENT_FIELD_PLACEHOLDER
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "hi", None)
    mp.config = UserConfig()
    # real stack: needs a configured provider whose class declares CONTENT_FIELD_LABEL
    try:
        label = mp.providers.selected_provider().CONTENT_FIELD_LABEL
    except Exception:
        label = None
    if not label:
        pytest.skip("no active provider with a CONTENT_FIELD_LABEL in this env")
    out = substitute_provider_content_field(f"write into {_CONTENT_FIELD_PLACEHOLDER}", mp)
    assert _CONTENT_FIELD_PLACEHOLDER not in out          # placeholder replaced
    assert label in out                                    # replaced with the active provider's real label
