import sqlite3
from typing import cast

import pytest

from services import compaction_persistence
from services.provider_cache_service import ProviderCacheService
from services.subconscious_worker import SubconsciousWorker

pytestmark = pytest.mark.unit

_CH = "user"


def _clear(db: sqlite3.Connection) -> None:
    db.execute("DELETE FROM transcript WHERE channel = ?", (_CH,))
    db.commit()


def _seed_selected_ollama(db: sqlite3.Connection, max_tokens: int) -> int:
    cur = db.execute(
        "INSERT INTO providers (name, platform, model, host, max_tokens) "
        "VALUES ('idle-compact-test', 'ollama', 'fit-model', 'http://localhost:11434', ?)",
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


def test_idle_gate_compaction_job_is_a_clean_noop_with_no_backlog(db: sqlite3.Connection) -> None:
    """The real idle-gate job no-ops with zero side effects when there is nothing
    past the watermark — and repeating it stays free (the over-cap edge case).

    Drives ``SubconsciousWorker._step_compaction()`` — the exact method wired
    first in ``_tick()`` — which constructs a user-channel MP and calls
    ``mp.compact()``. Asserts the downstream effect end-to-end: no compaction
    watermark is written, the job reports a no-op, and a second run changes
    nothing. The offline provider is never reached, proving no LLM call fired.
    """
    _clear(db)
    _seed_selected_ollama(db, 20000)
    ProviderCacheService.invalidate()
    try:
        # Precondition: a fresh user channel has no compaction watermark.
        assert compaction_persistence.get_compaction(_CH) is None

        worker = SubconsciousWorker()

        # First idle tick: the real production job runs the real compaction path.
        # A clean return (no connection error against the offline provider) is the
        # proof that no LLM generation was attempted.
        first = worker._step_compaction()
        assert first == "no-op: nothing to compact"
        # Downstream effect: nothing written — no watermark folded into transcript.
        assert compaction_persistence.get_compaction(_CH) is None

        # Second idle tick with no new messages: still a no-op, still no writes.
        # This is the user's stated edge case — re-running wastes no resources.
        second = worker._step_compaction()
        assert second == "no-op: nothing to compact"
        assert compaction_persistence.get_compaction(_CH) is None
    finally:
        ProviderCacheService.invalidate()
        _clear(db)
