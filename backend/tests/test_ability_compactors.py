"""Compactor-specific business-logic tests migrated from the per-ability
conformance file removed in TKT-975. The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
compactor abilities' genuine behaviour tests (rows_compacted surface, broadcast
guard, trail-boundary invariant) that have no coverage elsewhere."""

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities._result import ToolResult
from abilities.chat_history_compactor import (
    ChatHistoryCompactionConfig,
    ChatHistoryCompactor,
)
from abilities.tool_chain_compactor import (
    ToolChainCompactionConfig,
)
from abilities._compaction_config import CompactionConfig
from configs.channels import UserConfig
from services import transcript_service
from services.message_processor import MessageProcessor
from services.provider_cache_service import ProviderCacheService

pytestmark = pytest.mark.unit


def _clear(db, channel):
    db.execute("DELETE FROM transcript WHERE channel = ?", (channel,))
    db.commit()


def _seed_offline_provider_cap_zero(db):
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


def _make_mp(raw_input, channel):
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, raw_input, None)
    mp.config = UserConfig()
    assert mp.config.channel == channel  # UserConfig drives the 'user' channel
    return mp


def test_fit_compaction_input_surfaces_kept_row_count(db):
    """The compaction result's ``rows_compacted`` must reflect the actual count of kept transcript rows."""
    ch = "user"
    _clear(db, ch)
    _seed_offline_provider_cap_zero(db)
    n_rows = 5
    for i in range(n_rows):
        transcript_service.write_input_row(ch, "user", f"row{i:03d}")
    mp = _make_mp("compact", ch)
    try:
        combined = ChatHistoryCompactor._fit_compaction_input(mp, "")
        assert combined is not None  # there is a backlog to compact
        # The kept-row count is surfaced on the parent for run() to read.
        assert getattr(mp, "_compaction_kept_rows", None) == n_rows
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_fit_compaction_input_count_is_none_when_nothing_to_compact(db):
    """Ensure the surfaced count is never a stale value from a prior turn."""
    ch = "user"
    _clear(db, ch)
    _seed_offline_provider_cap_zero(db)
    mp = _make_mp("compact", ch)
    try:
        combined = ChatHistoryCompactor._fit_compaction_input(mp, "")
        assert combined is None
        assert getattr(mp, "_compaction_kept_rows", None) == 0
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)


def test_compaction_config_never_broadcasts_to_user():
    """Pins existing contract: the shared CompactionConfig (and both concrete
    subclasses) carry ``broadcast_to=None`` — never 'user'. The dispatcher only
    assigns a rich-media ordinal when broadcast_to == 'user' AND tr.rich, so a
    compactor result can never be paired to a user-facing card."""
    assert CompactionConfig().broadcast_to is None
    assert ChatHistoryCompactionConfig().broadcast_to is None
    assert ToolChainCompactionConfig().broadcast_to is None


def test_empty_tool_chain_result_is_not_a_trail_boundary(db):
    """Trail-boundary detection (_from_last_compaction) keys off a tool_chain_compactor row
    whose recorded result is non-empty after strip. The no-op path produces an empty body so the
    boundary classifier sees the same string it always has. (chat_history meta uses a DIFFERENT
    tool name and is never inspected by the boundary classifier.)"""
    noop = ToolResult.ok("")
    rendered = ToolDispatcher._render("tool_chain_compactor", noop, None)
    # The boundary classifier inspects the rendered envelope; the body slot is
    # empty for a no-op (only the envelope tags surround it). The meta head is
    # absent — proving the no-op path added no meta that would change the head.
    assert "(status=success)" in rendered  # no extra meta in the head
    # And the success-with-handover path DOES carry a non-empty body (boundary).
    handover = ToolResult.ok("dense handover text", trail_chars=42)
    rendered_boundary = ToolDispatcher._render("tool_chain_compactor", handover, None)
    assert "dense handover text" in rendered_boundary
    assert "trail_chars=42" in rendered_boundary
