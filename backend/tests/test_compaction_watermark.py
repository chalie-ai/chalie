"""Feature tests for compaction watermark and over-cap detection.

TKT-846 spec change (doc 131): OVER_CAP sentinel is gone, replaced by
RequestOverCapError. Providers is mp-free; send(dto) takes a ProviderApiRequest.
_over_cap() and build_request_body() are deleted; replaced by
providers.measure(dto) + the cap formula. Tests migrated to exercise the real
new production entry points.
"""

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

    OllamaClient.estimate_request_tokens / get_context_limit (via declared
    max_tokens) run with zero network, so the cap check exercises the real
    provider stack without a live model. client.send(dto) is never reached
    by the over-cap path — RequestOverCapError is raised before the call."""
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


def test_measure_true_when_full_request_reaches_threshold(db):
    """Compact-first (canonical step 3): when the FULL request's measured token
    count reaches RS >= window - max(10% window, 8k), providers.measure(dto)
    returns a value at or above the cap so the ACT loop fires compaction BEFORE
    sending — no trimmed/partial-view turn, no API call.

    TKT-846: _over_cap(system, messages, tools, provider) deleted; replaced by
    providers.measure(dto) which exercises the real ProviderClient.estimate_request_tokens
    path. The cap formula is preserved: window - max(int(0.10*window), 8000).
    """
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

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
    mp.thinking_level = "low"
    try:
        window = mp.providers.get_context_limit()
        assert window == 20000
        cap = window - max(int(0.10 * window), 8000)   # 12000
        dto = mp._build_send_dto()
        measured = mp.providers.measure(dto)
        assert measured >= cap, (
            f"Expected measured ({measured}) >= cap ({cap}); "
            f"over-cap check should trigger compaction"
        )
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_measure_false_when_request_fits(db):
    """When the request fits under the cap, providers.measure(dto) returns a
    value below the cap so send() proceeds to the API — no compaction fires.

    TKT-846: _over_cap deleted; replaced by providers.measure(dto) < cap.
    """
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
    mp.thinking_level = "low"
    try:
        window = mp.providers.get_context_limit()
        cap = window - max(int(0.10 * window), 8000)
        dto = mp._build_send_dto()
        measured = mp.providers.measure(dto)
        assert measured < cap, (
            f"Expected measured ({measured}) < cap ({cap}); "
            f"a tiny request must not trigger over-cap"
        )
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_send_raises_request_over_cap_without_calling_provider(db):
    """providers.send(dto) raises RequestOverCapError when the request exceeds
    the cap — the offline provider is never reached (zero network), proving
    compaction fires BEFORE any API call (compact-first).

    TKT-846: OVER_CAP sentinel replaced by RequestOverCapError. Sentinel
    pattern `response is OVER_CAP` becomes `except RequestOverCapError`.
    """
    from services.message_processor import MessageProcessor
    from services.provider_api import RequestOverCapError
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

    ch = "user"
    _clear(db, ch)
    big = " ".join(f"w{j}" for j in range(400))
    for i in range(40):
        transcript_service.write_input_row(ch, "user", f"row{i:03d} {big}")
    _seed_selected_ollama(db, 20000)
    ProviderCacheService.invalidate()

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what should I do next?", None)
    mp.config = UserConfig()
    mp.thinking_level = "low"  # set here because process() sets this; __init__ doesn't
    try:
        # No live model — if send() tried to reach Ollama this would raise
        # a network error, not RequestOverCapError.
        dto = mp._build_send_dto()
        with pytest.raises(RequestOverCapError):
            mp.providers.send(dto)
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_fit_compaction_input_drops_oldest_until_bare_request_fits(db):
    """Canonical step 4.2 (rare fallback): ChatHistoryCompactor shrinks its
    bare {system + prior + get_previous_messages} request by dropping the OLDEST
    message one at a time until it fits the cap. Real provider stack, no model.

    TKT-846: provider.build_request_body(system, [msg], []) removed; compactor
    now calls parent.providers.measure(candidate_dto) via ProviderApiRequest.
    """
    from abilities.chat_history_compactor import ChatHistoryCompactor
    from services.message_processor import MessageProcessor
    from services.system_message_prompt import ChatHistoryCompactionSystemPrompt
    from services.provider_api import ProviderApiRequest, ProviderType, ThinkingLevel
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

    ch = "user"
    _clear(db, ch)
    n_rows = 40
    big = " ".join(f"w{j}" for j in range(400))
    for i in range(n_rows):
        transcript_service.write_input_row(ch, "user", f"row{i:03d} {big}")
    _seed_selected_ollama(db, 20000)
    ProviderCacheService.invalidate()

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "compact", None)
    mp.config = UserConfig()
    try:
        combined = ChatHistoryCompactor._fit_compaction_input(mp, "")
        assert combined is not None
        # Dropped at least one oldest row → the very first row is gone.
        assert "row000" not in combined
        # The newest row always survives the floor.
        assert f"row{n_rows - 1:03d}" in combined
        # The bare (tool-free) compaction request now fits the cap.
        system = ChatHistoryCompactionSystemPrompt().get_prompt()
        candidate_dto = ProviderApiRequest(
            system=system,
            messages=[{"role": "user", "content": combined}],
            type=ProviderType.CHAT,
            tools=None,
            thinking_mode=ThinkingLevel.HIGH,
        )
        window = mp.providers.get_context_limit()
        cap = window - max(int(0.10 * window), 8000)   # 12000
        measured = mp.providers.measure(candidate_dto)
        assert measured <= cap
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_fit_compaction_input_no_drop_when_bare_request_fits(db):
    """When the bare compaction request already fits, _fit_compaction_input keeps
    every message (no drop) and folds in the prior checkpoint."""
    from abilities.chat_history_compactor import ChatHistoryCompactor
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
    MessageProcessor.__init__(mp, "compact", None)
    mp.config = UserConfig()
    try:
        combined = ChatHistoryCompactor._fit_compaction_input(mp, "PRIOR-CHECKPOINT")
        assert combined is not None
        assert "## Previous Summary" in combined          # prior carried forward
        assert "PRIOR-CHECKPOINT" in combined
        for i in range(3):
            assert f"short {i}" in combined                # nothing dropped
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_fit_compaction_input_returns_none_when_no_history(db):
    """No rows past the watermark → nothing to compact → None (no watermark write)."""
    from abilities.chat_history_compactor import ChatHistoryCompactor
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

    ch = "user"
    _clear(db, ch)
    _seed_selected_ollama(db, 20000)
    ProviderCacheService.invalidate()

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "compact", None)
    mp.config = UserConfig()
    try:
        assert ChatHistoryCompactor._fit_compaction_input(mp, "") is None
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
