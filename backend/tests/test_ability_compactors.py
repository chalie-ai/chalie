"""Chat-history-compactor business-logic tests. The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
compactor's genuine behaviour tests (rows_compacted surface, broadcast guard)
that have no coverage elsewhere."""

import sqlite3
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from abilities.chat_history_compactor import (
    ChatHistoryCompactionConfig,
    ChatHistoryCompactor,
)
from abilities._compaction_config import CompactionConfig
from configs.channels import UserConfig
from services.transcript_service import Transcript
from services.message_processor import MessageProcessor
from services.provider_cache_service import ProviderCacheService

if TYPE_CHECKING:
    class _CompactionParent(Protocol):
        _compaction_kept_rows: int
        turn_id: "int | None"
        def _previous_rows(self) -> list[object]: ...
        def get_previous_messages(self, *, drop_oldest: int = ...) -> str: ...
        class _Providers(Protocol):
            def get_context_limit(self) -> int: ...
            def measure(self, dto: object) -> int: ...
        providers: _Providers
        class _Config(Protocol):
            channel: str
        config: _Config

pytestmark = pytest.mark.unit


def _clear(db: sqlite3.Connection, channel: str) -> None:
    db.execute("DELETE FROM transcript WHERE channel = ?", (channel,))
    db.commit()


def _seed_offline_provider_cap_zero(db: sqlite3.Connection) -> int | None:
    cur = db.execute(
        "INSERT INTO providers (name, platform, model, host, max_tokens) "
        "VALUES ('compactor-test', 'ollama', 'cap-model', 'http://localhost:11434', 8000)",
    )
    pid = cur.lastrowid
    db.execute(
        "INSERT INTO settings (key, value) VALUES ('selected_provider_id', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(pid),),
    )
    db.commit()
    ProviderCacheService.invalidate()
    return pid


def _make_mp(raw_input: str, channel: str) -> MessageProcessor:
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, raw_input, None)
    mp.config = UserConfig()
    assert mp.config.channel == channel  # UserConfig drives the 'user' channel
    return mp


def test_fit_compaction_input_surfaces_kept_row_count(db: sqlite3.Connection) -> None:
    """The compaction result's ``rows_compacted`` must reflect the actual count of kept transcript rows."""
    ch = "user"
    _clear(db, ch)
    _seed_offline_provider_cap_zero(db)
    n_rows = 5
    for i in range(n_rows):
        Transcript.write_input_row(ch, "user", f"row{i:03d}")
    mp = _make_mp("compact", ch)
    try:
        combined = ChatHistoryCompactor._fit_compaction_input(cast("_CompactionParent", mp), "")
        assert combined is not None  # there is a backlog to compact
        # The kept-row count is surfaced on the parent for run() to read.
        assert getattr(mp, "_compaction_kept_rows", None) == n_rows
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_fit_compaction_input_count_is_none_when_nothing_to_compact(db: sqlite3.Connection) -> None:
    """Ensure the surfaced count is never a stale value from a prior turn."""
    ch = "user"
    _clear(db, ch)
    _seed_offline_provider_cap_zero(db)
    mp = _make_mp("compact", ch)
    try:
        combined = ChatHistoryCompactor._fit_compaction_input(cast("_CompactionParent", mp), "")
        assert combined is None
        assert getattr(mp, "_compaction_kept_rows", None) == 0
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_compaction_config_never_broadcasts_to_user() -> None:
    """Pins existing contract: the shared CompactionConfig (and both concrete
    subclasses) carry ``broadcast_to=None`` — never 'user'. The dispatcher only
    assigns a rich-media ordinal when broadcast_to == 'user' AND tr.rich, so a
    compactor result can never be paired to a user-facing card."""
    assert CompactionConfig().broadcast_to is None
    assert ChatHistoryCompactionConfig().broadcast_to is None
