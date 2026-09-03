"""Chat-history-compactor business-logic tests. The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
compactor's genuine behaviour tests (rows_compacted surface, checkpoint write,
broadcast guard) that have no coverage elsewhere.

The ``rows_compacted`` / ``_fit_compaction_input`` coverage that used to live in
this file was removed during the old-spine ``MessageProcessor`` cleanup because
``ChatHistoryCompactor`` still called the deleted ``mp.providers.get_context_limit()``
/ ``mp.providers.measure()`` API. That half of the migration is now done —
``_fit_compaction_input`` correctly calls ``parent.provider_service.context_limit()``
/ (``abilities/chat_history_compactor.py``
lines ~205/226) — so this file restores that coverage against the current
``controllers.message_processor.MessageProcessor`` / ``ProviderService`` surface.

KNOWN, UNFIXED PRODUCTION BUG surfaced by every test below: the SAME
``_fit_compaction_input`` (``abilities/chat_history_compactor.py:207``) and
``ChatHistoryCompactor.run`` (``:173``) still call ``parent._previous_rows()`` /
``parent.get_previous_messages(drop_oldest=...)`` — two methods that do NOT exist
on the real ``MessageProcessor`` (confirmed by direct inspection of
``controllers/message_processor.py`` and empirically by running the tests below).
Their real equivalents are ``mp.transcript_service.read()`` (replaces
``_previous_rows()``) and ``mp.prompt_service.previous_messages(drop_oldest=...)``
(replaces ``get_previous_messages(drop_oldest=...)``); the ``_CompactionParent``
Protocol (TYPE_CHECKING-only) declares the old names as if they were a real
runtime contract, which is exactly what let this drift ship unnoticed. Because
``services/dispatch_service.py``'s ``DispatchService._run()`` wraps every
``ability.run(params)`` in a blanket ``except Exception`` and converts it into an
error ``ToolResult`` instead of propagating, this ``AttributeError`` never
surfaces as a crash in production — the compactor silently no-ops every time,
the watermark never advances, and ``MessageProcessor._step()`` unconditionally
re-dispatches ``chat_history_compactor`` on every subsequent over-cap retry. That
is the "unbounded re-dispatch recursion" defect 6 describes; the provider_service
rename alone did not fix it. These three tests are the correct target-state
regression coverage and will pass once ``_previous_rows`` / ``get_previous_messages``
are re-pointed at ``transcript_service.read()`` / ``prompt_service.previous_messages``
— fixing that is out of scope for this test file."""

import sqlite3
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from abilities._compaction_config import CompactionConfig
from abilities.chat_history_compactor import ChatHistoryCompactionConfig, ChatHistoryCompactor
from configs.channels.user import UserConfig
from contracts.params.chat_history_compactor_params_bag import ChatHistoryCompactorParamsBag
from controllers.message_processor import MessageProcessor
from models.compaction import Compaction
from models.transcript import Transcript
from services.provider_cache_service import ProviderCacheService
from services.provider_db_service import ProviderDbService
from tests._tool_result_harness import built

if TYPE_CHECKING:
    # Reuse the compactor's own parent contract — no duplicate Protocol.
    from abilities.chat_history_compactor import _CompactionParent

pytestmark = pytest.mark.unit

# ProviderService builds its thin transport client through this factory — the
# real network boundary (same seam as test_context_usage_signal.py). Patching it
# is what keeps these tests off ``http://localhost:11434``.
_BUILD_CLIENT = "services.provider_service.build_client"

# A window small enough that the compactor's cap formula
# (``window - max(0.10*window, 8000)``) lands at zero, so ``_fit_compaction_input``
# never sizes a request against — nor sends one to — a live provider.
_OFFLINE_WINDOW = 8000


class _OfflineClient:
    """Stub at the ``build_client`` boundary: reports a fixed 8000-token window
    and a zero size estimate, so the whole compaction-sizing path is hermetic —
    no ``get_context_limit`` HTTP call, no ``send``."""

    def get_context_limit(self) -> int:
        return _OFFLINE_WINDOW



def _seed_offline_provider_cap_zero(db: sqlite3.Connection) -> int:
    """A real, selected ``providers`` row with a pinned window small enough that
    the compactor's cap formula (``window - max(0.10*window, 8000)``) lands at or
    below zero. ``ProviderService.context_limit()`` reads that column straight
    off the row, and the cap<=0 branch means ``_fit_compaction_input`` never has
    to call the also-offline-but-unexercised ``ProviderService.measure()``
    either. Pinning it here rather than leaving it NULL is what keeps the test
    hermetic — an unpinned row would send ``pin_context_window`` to the host."""
    cur = db.execute(
        "INSERT INTO providers (name, platform, model, host, context_window) "
        "VALUES ('compactor-test', 'ollama', 'cap-model', "
        "'http://localhost:11434', 8000)",
    )
    pid = cast(int, cur.lastrowid)
    db.commit()
    ProviderDbService().set_selected_provider(pid)
    ProviderCacheService.invalidate()
    return pid


def _seed_settled_turn(channel: str, user_text: str, assistant_text: str) -> int:
    """One real, fully-settled turn (input row + settled assistant row) via the
    real ``Transcript`` active-record model — the same row shape
    ``TranscriptService._append`` writes for a plain (no tool-call) exchange."""
    turn_id = Transcript.next_turn_id(None, channel)
    Transcript(
        channel=channel, role="user", content=user_text,
        turn_id=turn_id, settled=0, xml_migrated=1,
    ).save()
    Transcript(
        channel=channel, role="assistant", content=assistant_text,
        turn_id=turn_id, settled=1, xml_migrated=1,
    ).save()
    return turn_id


def _make_mp(raw_input: str = "compact") -> MessageProcessor:
    """Bare, inert construction (I2) — no begin(), no thread, no LLM call.
    Exercises the exact real ``MessageProcessor`` / ``provider_service`` surface
    ``ChatHistoryCompactor`` reads."""
    return MessageProcessor(UserConfig(), raw_input=raw_input)


def test_fit_compaction_input_fits_settled_history_under_the_context_cap(db: sqlite3.Connection) -> None:
    """Two real, settled turns on the spine must combine into one compaction
    request body — the sizing half of defect 6 (the old code crashed on the
    deleted ``mp.providers.get_context_limit()``; ``provider_service.context_limit()``
    is the new surface it must use instead)."""
    _seed_offline_provider_cap_zero(db)
    _seed_settled_turn("user", "first question", "first answer")
    _seed_settled_turn("user", "second question", "second answer")
    parent = cast("_CompactionParent", _make_mp())

    with patch(_BUILD_CLIENT, return_value=_OfflineClient()):
        combined = ChatHistoryCompactor._fit_compaction_input(parent, "")

    assert combined is not None
    assert "first question" in combined
    assert "second answer" in combined
    assert parent._compaction_kept_rows == 4  # 2 turns x (input + answer)


def test_fit_compaction_input_returns_none_with_no_backlog(db: sqlite3.Connection) -> None:
    """Should-not-fire path: an empty channel has nothing to compact —
    ``_fit_compaction_input`` must report zero kept rows, never a stale count
    left over from a previous call on the same parent."""
    _seed_offline_provider_cap_zero(db)
    parent = cast("_CompactionParent", _make_mp())

    with patch(_BUILD_CLIENT, return_value=_OfflineClient()):
        combined = ChatHistoryCompactor._fit_compaction_input(parent, "")

    assert combined is None
    assert parent._compaction_kept_rows == 0


def test_run_completes_and_writes_a_checkpoint_without_crashing(db: sqlite3.Connection) -> None:
    """Defect 6, end to end: dispatching the compactor's real ``run()`` against a
    real settled backlog must complete without raising and must advance the MAIN
    watermark by writing a ``Compaction`` row. That checkpoint is what stops
    ``MessageProcessor._step()`` from re-dispatching compaction forever on every
    subsequent over-cap retry (the "unbounded re-dispatch recursion" defect 6
    names) — an exception escaping ``run()`` here is exactly that failure mode,
    since the dispatcher's blanket ``except Exception`` turns it into a silent
    no-op instead of a checkpoint."""
    _seed_offline_provider_cap_zero(db)
    turn_id = _seed_settled_turn("user", "first question", "first answer")
    mp = _make_mp()
    ability = ChatHistoryCompactor(mp)

    with patch(_BUILD_CLIENT, return_value=_OfflineClient()):
        result = ability.run(built(ChatHistoryCompactorParamsBag.from_params({})))

    assert result.status == "success"
    checkpoint = Compaction.latest_main("user")
    assert checkpoint is not None
    assert checkpoint.compacted_up_to == turn_id


def test_compaction_config_never_broadcasts_to_user() -> None:
    """Pins existing contract: the shared CompactionConfig (and both concrete
    subclasses) leave ``RENDERS_HTML`` False. The dispatcher only assigns a
    rich-media ordinal when the channel renders to a human AND tr.rich, so a
    compactor result can never be paired to a user-facing card."""
    assert CompactionConfig().RENDERS_HTML is False
    assert ChatHistoryCompactionConfig().RENDERS_HTML is False
