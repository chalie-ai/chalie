"""Chat-history-compactor business-logic tests. The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
compactor's genuine behaviour tests (rows_compacted surface, broadcast guard)
that have no coverage elsewhere."""

import sqlite3
from typing import TYPE_CHECKING, cast

import pytest

from abilities._compaction_config import CompactionConfig
from abilities.chat_history_compactor import (
    ChatHistoryCompactionConfig,
    ChatHistoryCompactor,
)
from configs.channels import UserConfig
from services.act_trail import ActTrail
from services.message_processor import MessageProcessor
from services.provider_cache_service import ProviderCacheService
from services.transcript_service import Transcript

if TYPE_CHECKING:
    # Reuse the compactor's own parent contract — no duplicate Protocol.
    from abilities.chat_history_compactor import _CompactionParent

pytestmark = pytest.mark.unit


def _clear(db: sqlite3.Connection, channel: str) -> None:
    db.execute(
        "DELETE FROM tool_calls WHERE transcript_id IN "
        "(SELECT id FROM transcript WHERE channel = ?)",
        (channel,),
    )
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
    """The compaction result's ``rows_compacted`` must reflect the actual count of
    kept transcript rows — the settled spine the next read will cut on."""
    ch = "user"
    _clear(db, ch)
    _seed_offline_provider_cap_zero(db)
    # Two settled turns on the spine. Turn one: input + a pre-settle working step
    # (a real tool demotes it below settle0) + the no-tool answer (settle0). Turn
    # two: input + answer. Five rows the main spine surfaces to the compactor.
    in1 = Transcript.write_input_row(ch, "user", "first question")
    t1 = Transcript.turn_id_of_row(in1)
    step = Transcript.write_assistant_row(ch, "a working step", turn_id=t1)
    ActTrail().record(tool_name="search_files", params={}, result="ok", transcript_id=step)
    Transcript.write_assistant_row(ch, "first answer", turn_id=t1)
    in2 = Transcript.write_input_row(ch, "user", "second question")
    Transcript.write_assistant_row(ch, "second answer", turn_id=Transcript.turn_id_of_row(in2))
    mp = _make_mp("compact", ch)
    try:
        combined = ChatHistoryCompactor._fit_compaction_input(cast("_CompactionParent", mp), "")
        assert combined is not None  # there is a settled backlog to compact
        # The kept-row count is surfaced on the parent for run() to read.
        assert getattr(mp, "_compaction_kept_rows", None) == 5
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
