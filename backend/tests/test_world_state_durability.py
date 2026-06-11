"""Feature tests — TKT-922 worker-gate durability across a process restart.

Acceptance criterion (RED-first): ``last_user_message_at`` lives only in
WorldState's in-memory ``_store`` dict today (services/world_state.py), so a
container restart wipes it and starves the subconscious maintenance worker
(observed 57 ticks vs 3029 skips). It must be persisted durably next to
``subconscious_last_fired_at`` — MemoryStore (fast) + data_graph kind='system'
(durable) — and hydrated on construction, mirroring
``SubconsciousWorker._persist_last_fired`` / ``_load_last_fired_from_storage``.

These tests exercise the REAL production hot path with zero mocks of production
logic:

  * WRITER:  the same ``world_state.absorb(Signal(kind='user_message'))`` call
             that ``api/chat.py`` fires on every user turn.
  * RESTART: a brand-new ``WorldState()`` instance against the SAME database
             file, with the volatile MemoryStore cache emptied — which is
             exactly what a ``docker restart`` does (DB volume survives, the
             in-process dict + MemoryStore die). Building a fresh singleton is
             the faithful in-process model of a process restart, NOT a mock.
  * READER:  ``snapshot()`` on the restarted instance (durable read) and the
             REAL ``SubconsciousWorker._check_gates()`` driven against the
             restarted singleton — never a reimplementation of the gate.

At current HEAD (memory-only WorldState) the restart wipes the value, so the
durable-survival and gate-de-starvation tests FAIL. Once persistence+hydrate
land they pass. The freshness tests assert the value read post-restart is the
durable copy, not the dead dict.
"""

from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

import pytest

from services.time_utils import utc_now
from services.world_state import WorldState, Signal


# ── Restart harness ───────────────────────────────────────────────────────────
#
# A real container restart keeps the DB volume and discards everything in
# process memory: the WorldState singleton's dict AND the MemoryStore cache.
# We model that by constructing a fresh WorldState while pointing the volatile
# fast-cache (MemoryStore) at a brand-new empty instance, leaving only the DB
# (the `db` fixture) as the durable survivor. Nothing about production logic is
# mocked — only the process-lifetime caches are reset, which is the literal
# effect of a restart.


@contextmanager
def _simulated_restart():
    """Yield a freshly-constructed WorldState as if the process had restarted.

    The DB (patched by the `db` fixture) persists. The MemoryStore cache is
    replaced with an empty one so any hydrate is forced to read the durable
    store, proving the value survived in the database — not merely in a cache
    that happened to outlive the test's first WorldState.
    """
    from services.memory_store import MemoryStore

    cold_cache = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=cold_cache), \
         patch('services.memory_client.MemoryClientService.create_connection',
               return_value=cold_cache):
        # Constructing here is the "boot" of the new process.
        yield WorldState()


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def warm_store():
    """Isolated MemoryStore for the pre-restart (warm) process lifetime.

    Mirrors the production fast-cache so ``absorb`` writes through the same
    MemoryClientService path it uses at runtime.
    """
    from services.memory_store import MemoryStore

    _store = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=_store), \
         patch('services.memory_client.MemoryClientService.create_connection',
               return_value=_store):
        yield _store


@pytest.mark.integration
class TestWorldStateDurabilityAcrossRestart:
    """The user-message timestamp must outlive a process restart via the DB."""

    def test_user_message_timestamp_survives_restart(self, db, warm_store):
        """absorb(user_message) then restart → durable snapshot equals the write.

        Drives the production writer (the call api/chat.py makes), simulates a
        restart (fresh WorldState + cold cache, same DB), and asserts the
        restarted instance reads back the persisted timestamp. RED at HEAD:
        memory-only WorldState loses the value on restart.
        """
        when = utc_now() - timedelta(minutes=2)
        # Real production writer — identical to api/chat.py:281.
        WorldState().absorb(
            Signal(source="http_chat", kind="user_message",
                   payload={"text": "remember this"}, received_at=when)
        )

        with _simulated_restart() as restarted:
            survived = restarted.snapshot()["last_user_message_at"]

        assert survived is not None, (
            "last_user_message_at did not survive the restart — it must be "
            "persisted durably (data_graph kind='system'), not memory-only."
        )
        # The durable read must be the value we wrote (second precision — the
        # stored representation round-trips through isoformat/parse_utc).
        assert abs((survived - when).total_seconds()) < 1.0

    def test_durable_value_is_read_from_storage_not_a_live_dict(self, db, warm_store):
        """The restarted instance is genuinely fresh — proves a real hydrate.

        A new WorldState starts with an empty in-memory ``_store``; the only way
        its snapshot can return the timestamp is by hydrating from the durable
        store. This guards against a test that would pass merely because some
        dict outlived the restart.
        """
        when = utc_now() - timedelta(minutes=1)
        writer = WorldState()
        writer.absorb(
            Signal(source="http_chat", kind="user_message",
                   payload={"text": "hi"}, received_at=when)
        )

        with _simulated_restart() as restarted:
            # The restarted instance is a different object than the writer — it
            # shares no in-process dict with it. The only channel for the value
            # to reach it is the durable store. A pass here therefore proves a
            # real hydrate, not a surviving dict.
            assert restarted is not writer
            survived = restarted.snapshot()["last_user_message_at"]

        assert survived is not None
        assert abs((survived - when).total_seconds()) < 1.0

    def test_latest_user_message_wins_after_restart(self, db, warm_store):
        """Two user turns then restart → the durable read is the LAST write.

        Persistence must overwrite, never append/stale — the gate compares the
        most recent user activity.
        """
        old = utc_now() - timedelta(hours=3)
        recent = utc_now() - timedelta(minutes=3)
        ws = WorldState()
        ws.absorb(Signal(source="http_chat", kind="user_message",
                         payload={"text": "first"}, received_at=old))
        ws.absorb(Signal(source="http_chat", kind="user_message",
                         payload={"text": "second"}, received_at=recent))

        with _simulated_restart() as restarted:
            survived = restarted.snapshot()["last_user_message_at"]

        assert survived is not None
        assert abs((survived - recent).total_seconds()) < 1.0, (
            "durable read should reflect the latest user message, not an "
            "earlier one."
        )


@pytest.mark.integration
class TestSubconsciousGateDeStarvationAfterRestart:
    """The real worker gate must see the hydrated timestamp post-restart."""

    @contextmanager
    def _gate_against(self, world_state_instance):
        """Point the module-level world_state singleton (which _check_gates
        imports) at the given instance — modelling that the restarted process
        rebuilds its singletons. Not a mock of gate logic; the real
        ``_check_gates`` runs unchanged."""
        import services.world_state as _ws_mod
        original = _ws_mod.world_state
        _ws_mod.world_state = world_state_instance
        try:
            yield
        finally:
            _ws_mod.world_state = original

    def test_recent_user_message_keeps_gate_user_active_after_restart(self, db, warm_store):
        """A user who spoke 1 min ago is still 'active' after a restart.

        At HEAD the restart wipes last_user_message_at → the user-active gate
        cannot fire and the worker would (wrongly) treat the user as idle. With
        durable hydrate, the REAL ``_check_gates`` returns 'user_active'.
        """
        from services.subconscious_worker import SubconsciousWorker

        WorldState().absorb(
            Signal(source="http_chat", kind="user_message",
                   payload={"text": "still here"},
                   received_at=utc_now() - timedelta(minutes=1))
        )

        with _simulated_restart() as restarted, self._gate_against(restarted):
            worker = SubconsciousWorker()
            gate = worker._check_gates()

        assert gate == "user_active", (
            "after a restart the worker must still see the recent user message "
            f"and skip with 'user_active'; got {gate!r}. A None/already_fired "
            "here means the durable timestamp was lost (starvation)."
        )

    def test_idle_user_message_lets_gate_run_after_restart(self, db, warm_store):
        """A user who last spoke 45 min ago (idle) does NOT block the tick.

        The shouldn't-fire side of the gate: with a hydrated-but-old timestamp,
        the user-active gate must NOT trip, and with no prior fire the tick is
        allowed (``_check_gates`` returns None). This proves the hydrated value
        is compared correctly, not merely 'present → skip'.
        """
        from services.subconscious_worker import SubconsciousWorker

        WorldState().absorb(
            Signal(source="http_chat", kind="user_message",
                   payload={"text": "talked a while ago"},
                   received_at=utc_now() - timedelta(minutes=45))
        )

        with _simulated_restart() as restarted, self._gate_against(restarted):
            # Fresh worker that has never fired (no last_fired persisted).
            worker = SubconsciousWorker()
            assert worker.last_fired_at is None
            gate = worker._check_gates()

        assert gate is None, (
            "an idle (45-min-old) hydrated user message must not trip the "
            f"user-active gate; expected None, got {gate!r}."
        )
