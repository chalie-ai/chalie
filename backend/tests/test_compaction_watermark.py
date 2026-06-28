import sqlite3
from typing import TYPE_CHECKING, cast

import pytest
from services.transcript_service import Transcript

if TYPE_CHECKING:
    from abilities.chat_history_compactor import _CompactionParent  # noqa: F401

pytestmark = pytest.mark.integration


def _clear(db: sqlite3.Connection, channel: str) -> None:
    db.execute(
        "DELETE FROM tool_calls WHERE transcript_id IN "
        "(SELECT id FROM transcript WHERE channel = ?)",
        (channel,),
    )
    db.execute("DELETE FROM transcript WHERE channel = ?", (channel,))
    db.commit()


def _settled_turn(channel: str, contents: list[str]) -> None:
    """One settled turn through the production writers: a user opener, ``len-2``
    tool-bearing assistant steps (a real tool keeps each below settle0), and a
    no-tool answer (settle0). Every row lands in the MAIN spine the over-cap /
    fit checks read, so the seeded history actually reaches the request DTO."""
    from services.act_trail import ActTrail

    in_id = Transcript.write_input_row(channel, "user", contents[0])
    tid = Transcript.turn_id_of_row(in_id)
    for text in contents[1:-1]:
        step = Transcript.write_assistant_row(channel, text, turn_id=tid)
        ActTrail().record(tool_name="search_files", params={}, result="", transcript_id=step)
    Transcript.write_assistant_row(channel, contents[-1], turn_id=tid)


def test_get_context_limit_reads_declared_max_tokens_capped(db: sqlite3.Connection) -> None:
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.providers import MAX_CONTEXT_WINDOW
    from services.provider_cache_service import ProviderCacheService

    # Seed a real selected provider through the production providers/settings
    # tables (zero network — get_context_limit only reads the declared max_tokens).
    pid = _seed_selected_ollama(db, 8000)
    ProviderCacheService.invalidate()

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "hi", None)
    mp.config = UserConfig()
    try:
        assert mp.providers.get_context_limit() == 8000          # declared value honoured
        db.execute("UPDATE providers SET max_tokens = 999999 WHERE id = ?", (pid,))
        db.commit()
        ProviderCacheService.invalidate()
        assert mp.providers.get_context_limit() == MAX_CONTEXT_WINDOW   # capped at 200k
    finally:
        ProviderCacheService.invalidate()


def _seed_selected_ollama(db: sqlite3.Connection, max_tokens: int) -> int:
    """Seed an offline Ollama provider and mark it selected, returning its id.

    Uses zero network - RequestOverCapError is raised before send(dto) is called."""
    cur = db.execute(
        "INSERT INTO providers (name, platform, model, host, max_tokens) "
        "VALUES ('fit-test', 'ollama', 'fit-model', 'http://localhost:11434', ?)",
        (max_tokens,),
    )
    pid = cast(int, cur.lastrowid)
    db.execute(
        "INSERT INTO settings (key, value) VALUES ('selected_provider_id', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(pid),),
    )
    db.commit()
    return pid


def test_measure_true_when_full_request_reaches_threshold(db: sqlite3.Connection) -> None:
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

    ch = "user"
    _clear(db, ch)
    # estimate_tokens counts WORDS (split * 1.3), so rows must be word-rich:
    # 40 spine rows * ~400 words * 1.3 ≈ 20.8k tok, overflowing the 12k cap. One
    # settled turn carries them all so the whole backlog reaches the request DTO.
    n_rows = 40
    big = " ".join(f"w{j}" for j in range(400))
    _settled_turn(ch, [f"row{i:03d} {big}" for i in range(n_rows)])
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


def test_measure_false_when_request_fits(db: sqlite3.Connection) -> None:
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

    ch = "user"
    _clear(db, ch)
    _settled_turn(ch, [f"short {i}" for i in range(3)])
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


def test_send_raises_request_over_cap_without_calling_provider(db: sqlite3.Connection) -> None:
    """providers.send(dto) raises RequestOverCapError before any API call (zero network)."""
    from services.message_processor import MessageProcessor
    from services.provider_api import RequestOverCapError
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

    ch = "user"
    _clear(db, ch)
    big = " ".join(f"w{j}" for j in range(400))
    _settled_turn(ch, [f"row{i:03d} {big}" for i in range(40)])
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


def test_fit_compaction_input_drops_oldest_until_bare_request_fits(db: sqlite3.Connection) -> None:
    """Rare fallback (step 4.2): compactor drops the oldest message one at a time
    until the bare request fits the cap. Real provider stack, no live model."""
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
    _settled_turn(ch, [f"row{i:03d} {big}" for i in range(n_rows)])
    _seed_selected_ollama(db, 20000)
    ProviderCacheService.invalidate()

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "compact", None)
    mp.config = UserConfig()
    try:
        combined = ChatHistoryCompactor._fit_compaction_input(cast("_CompactionParent", mp), "")
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


def test_fit_compaction_input_no_drop_when_bare_request_fits(db: sqlite3.Connection) -> None:
    """When the bare compaction request already fits, _fit_compaction_input keeps
    every message (no drop) and folds in the prior checkpoint."""
    from abilities.chat_history_compactor import ChatHistoryCompactor
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

    ch = "user"
    _clear(db, ch)
    _settled_turn(ch, [f"short {i}" for i in range(3)])
    _seed_selected_ollama(db, 20000)
    ProviderCacheService.invalidate()

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "compact", None)
    mp.config = UserConfig()
    try:
        combined = ChatHistoryCompactor._fit_compaction_input(cast("_CompactionParent", mp), "PRIOR-CHECKPOINT")
        assert combined is not None
        assert "## Previous Summary" in combined          # prior carried forward
        assert "PRIOR-CHECKPOINT" in combined
        for i in range(3):
            assert f"short {i}" in combined                # nothing dropped
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_fit_compaction_input_returns_none_when_no_history(db: sqlite3.Connection) -> None:
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
        assert ChatHistoryCompactor._fit_compaction_input(cast("_CompactionParent", mp), "") is None
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_substitute_provider_content_field_uses_mp_providers(db: sqlite3.Connection) -> None:
    from configs.channels._common import substitute_provider_content_field, _CONTENT_FIELD_PLACEHOLDER
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.provider_cache_service import ProviderCacheService

    # Seed a real selected Ollama provider — its client class declares
    # CONTENT_FIELD_LABEL = "message.content", the live label the substitution
    # must resolve to (no skip, no network).
    _seed_selected_ollama(db, 20000)
    ProviderCacheService.invalidate()

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "hi", None)
    mp.config = UserConfig()
    try:
        label = mp.providers.selected_provider().CONTENT_FIELD_LABEL
        out = substitute_provider_content_field(f"write into {_CONTENT_FIELD_PLACEHOLDER}", mp)
        assert _CONTENT_FIELD_PLACEHOLDER not in out      # placeholder replaced
        assert label in out                                # replaced with the active provider's real label
        assert label == "message.content"                 # the seeded provider's declared field
    finally:
        ProviderCacheService.invalidate()
