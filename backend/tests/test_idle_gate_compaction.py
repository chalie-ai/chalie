"""Feature test — idle-gate compaction job (TKT-974).

When the idle gate opens, ``SubconsciousWorker`` fires ``_step_compaction()`` as
the FIRST job in the tick. That job builds a real user-channel
``MessageProcessor`` and runs ONLY its compactors via ``mp.compact()`` — no
``_setup``, no ACT loop, no LLM turn. When there is nothing past the durable
compaction watermark the compactors no-op with zero side effects, so re-running
the job on a later idle tick costs nothing (no LLM call, no DB write).

This drives the real production job method end-to-end against the real DB. No
mocks: the only seam is an offline Ollama provider (declared ``max_tokens`` lets
``providers.get_context_limit()`` resolve with zero network — a generation
attempt would raise a connection error rather than return cleanly, which is
exactly how the test proves no LLM call was made).
"""

import pytest

from services import compaction_persistence, transcript_service
from services.provider_cache_service import ProviderCacheService
from services.subconscious_worker import SubconsciousWorker

pytestmark = pytest.mark.unit

_CH = "user"


def _clear(db):
    db.execute("DELETE FROM transcript WHERE channel = ?", (_CH,))
    db.commit()


def _seed_selected_ollama(db, max_tokens):
    """Seed an offline Ollama provider and mark it selected.

    OllamaClient.get_context_limit reads the declared max_tokens with zero
    network, so the compactor's sizing path runs offline. No model is reachable —
    any actual generation attempt raises a connection error, never a clean
    return. Mirrors the fixture in test_compaction_watermark.py.
    """
    cur = db.execute(
        "INSERT INTO providers (name, platform, model, host, max_tokens) "
        "VALUES ('idle-compact-test', 'ollama', 'fit-model', 'http://localhost:11434', ?)",
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


def test_idle_gate_compaction_job_is_a_clean_noop_with_no_backlog(db):
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
