"""Feature tests pinning the ToolResult contract of the two internal compactors.

TKT-919 — VERIFY-THEN-CLOSE-GAP. Most of the ToolResult migration for
``chat_history_compactor`` and ``tool_chain_compactor`` already shipped (both
return :class:`abilities._result.ToolResult` via ``ok()``); these tests pin that
contract and pin the one closed gap: the chat-history SUCCESS result now carries
an honest ``rows_compacted`` count surfaced out of ``_fit_compaction_input``.

Hard constraints (these run in the OFFLINE unit gate):
- No test reaches ``MessageProcessor.process(...)`` — that fires a real LLM call.
  The no-op paths short-circuit BEFORE ``process()`` (empty trail / nothing to
  compact), and ``_fit_compaction_input`` is exercised directly with a cap of 0
  (provider max_tokens 8000 → cap = 8000 - max(800, 8000) = 0), which makes the
  drop loop return on its first iteration WITHOUT calling ``providers.measure``.
- Real ``transcript_service`` writes + the shared ``db`` fixture, mirroring
  ``tests/test_compaction_watermark.py``. Zero mocks.

Several pins are GREEN-at-HEAD by design (the contract already holds) and are
labelled as such; the ``rows_compacted`` pins are RED-at-HEAD (the count was not
surfaced before this ticket).
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities._result import ToolResult
from abilities.chat_history_compactor import (
    ChatHistoryCompactionConfig,
    ChatHistoryCompactor,
)
from abilities.tool_chain_compactor import (
    ToolChainCompactionConfig,
    ToolChainCompactor,
)
from abilities._compaction_config import CompactionConfig
from configs.channels import UserConfig
from services import transcript_service
from services.message_processor import MessageProcessor
from services.provider_cache_service import ProviderCacheService

pytestmark = pytest.mark.unit


# ── Harness ────────────────────────────────────────────────────────────────


def _clear(db, channel):
    db.execute("DELETE FROM transcript WHERE channel = ?", (channel,))
    db.commit()


def _seed_offline_provider_cap_zero(db):
    """Seed + select an offline ollama provider whose declared window forces the
    compaction cap to 0.

    ``cap = window - max(int(0.10*window), 8000)``; with ``max_tokens=8000`` the
    cap is ``8000 - 8000 = 0``. ``_fit_compaction_input`` returns on its first
    loop iteration when ``cap <= 0`` — so the drop loop never calls
    ``providers.measure`` (zero network), keeping the unit gate offline.
    """
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


# ── 1. No-op paths return a canonical ToolResult (GREEN-at-HEAD) ──────────────


def test_tool_chain_no_trail_returns_canonical_ok_empty(db):
    """Pins existing contract: tool_chain_compactor with no trail since the last
    boundary returns ``ToolResult.ok("")`` — success, empty body, no code, no
    rich. uid=None → _has_trail() is False, so run() short-circuits BEFORE any
    MessageProcessor.process() (offline)."""
    mp = _make_mp("compact", "user")
    mp.uid = None  # no transcript row → _has_trail() False → no LLM call
    ability = ToolChainCompactor(mp=mp)

    tr = ability.run({})

    assert isinstance(tr, ToolResult)
    assert tr.status == "success"
    assert tr.body == ""
    assert tr.code is None
    assert tr.rich is None


def test_chat_history_nothing_to_compact_returns_canonical_ok_empty(db):
    """Pins existing contract: chat_history_compactor with no rows past the
    watermark (``_fit_compaction_input`` → None) returns ``ToolResult.ok("")`` —
    success, empty body, no code, no rich. No rows → get_previous_messages('') →
    None is returned before any provider.measure / MessageProcessor.process."""
    ch = "user"
    _clear(db, ch)
    _seed_offline_provider_cap_zero(db)
    mp = _make_mp("compact", ch)
    ability = ChatHistoryCompactor(mp=mp)
    try:
        tr = ability.run({})
    finally:
        ProviderCacheService.invalidate()
        _clear(db, ch)

    assert isinstance(tr, ToolResult)
    assert tr.status == "success"
    assert tr.body == ""
    assert tr.code is None
    assert tr.rich is None


# ── 2. rows_compacted is surfaced out of the fit helper (RED-at-HEAD) ─────────


def test_fit_compaction_input_surfaces_kept_row_count(db):
    """RED-at-HEAD: ``_fit_compaction_input`` must surface the count of kept
    transcript rows folded into the checkpoint so the success result can carry an
    honest ``rows_compacted``. Before this ticket no count was surfaced.

    Offline: provider cap is 0, so the drop loop returns ``combined`` on its first
    iteration with ``drop=0`` (no measure call). With N rows past the watermark
    and no prior checkpoint, every row is kept → count == N."""
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
    """RED-at-HEAD: when there is nothing to compact (helper returns None) the
    surfaced count must be 0 — never a stale value from a prior turn."""
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


# ── 3. No rich, no ordinal anywhere ───────────────────────────────────────────


def test_compaction_config_never_broadcasts_to_user():
    """Pins existing contract: the shared CompactionConfig (and both concrete
    subclasses) carry ``broadcast_to=None`` — never 'user'. The dispatcher only
    assigns a rich-media ordinal when broadcast_to == 'user' AND tr.rich, so a
    compactor result can never be paired to a user-facing card."""
    assert CompactionConfig().broadcast_to is None
    assert ChatHistoryCompactionConfig().broadcast_to is None
    assert ToolChainCompactionConfig().broadcast_to is None


def test_noop_render_carries_no_ordinal_or_span(db):
    """Pins existing contract: the rendered envelope for a no-op compactor result
    carries no rich card / no ordinal span instruction. The dispatcher's _render
    is the single envelope formatter; with rich=None and ordinal=None there is no
    span tag to anchor a card."""
    noop = ToolResult.ok("")
    rendered = ToolDispatcher._render("tool_chain_compactor", noop, None)
    assert "status=success" in rendered
    assert "<span" not in rendered
    assert "rows_compacted" not in rendered  # no-op carries no meta
    assert rendered == (
        "[tool_chain_compactor(status=success)]\n\n[end:tool_chain_compactor]"
    )


# ── 4. Boundary invariant — the empty tool_chain result stays a non-boundary ──


def test_empty_tool_chain_result_is_not_a_trail_boundary(db):
    """The trail-boundary detection (_from_last_compaction) keys off a
    tool_chain_compactor row whose recorded result is non-empty after strip. The
    no-op path is UNCHANGED by this ticket (still ``ok("")``, no meta), so its
    rendered envelope is byte-identical to HEAD and the empty-handover semantics
    cannot flip. This pin asserts the rendered no-op body is empty so the boundary
    classifier sees the same string it always has.

    chat_history meta is keyed off a DIFFERENT tool name and is never inspected by
    the boundary classifier, so the rows_compacted change cannot affect boundaries
    at all."""
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


# ── 5. Legacy-marker sweep — no banned code="error", no dict/str returns ──────


def test_modules_carry_no_banned_error_code_or_legacy_returns():
    """Pins the contract: neither compactor module emits the banned ``code="error"``
    sentinel, and every return in run() is a ToolResult (no bare dict/str)."""
    import inspect

    for module_obj, ability_cls in (
        (ChatHistoryCompactor, ChatHistoryCompactor),
        (ToolChainCompactor, ToolChainCompactor),
    ):
        src = inspect.getsource(inspect.getmodule(module_obj))
        assert 'code="error"' not in src
        assert "code='error'" not in src
        run_src = inspect.getsource(ability_cls.run)
        # Every return in run() goes through ToolResult.ok/err.
        for line in run_src.splitlines():
            stripped = line.strip()
            if stripped.startswith("return "):
                assert "ToolResult." in stripped, (
                    f"non-canonical return in {ability_cls.__name__}.run: {stripped!r}"
                )
