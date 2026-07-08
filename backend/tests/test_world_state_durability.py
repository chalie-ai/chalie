"""Feature tests for  worker-gate durability across a process restart.

``last_user_message_at`` lives only in WorldState's in-memory ``_store`` dict
at HEAD, so a container restart wipes it and starves the subconscious worker
(observed 57 ticks vs 3029 skips). It must be persisted durably next to
``subconscious_last_fired_at`` — MemoryStore (fast) + data_graph kind='machine_state'
(durable) — and hydrated on construction, mirroring
``SubconsciousWorker._persist_last_fired`` / ``_load_last_fired_from_storage``.

At HEAD the durable-survival and gate-de-starvation tests FAIL; they pass once
persistence+hydrate land.
"""

import sqlite3
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import cast
from unittest.mock import patch

import pytest

from services.memory_store import MemoryStore
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
def _simulated_restart() -> Generator[WorldState, None, None]:
    """Yield a freshly-constructed WorldState with a cold (empty) MemoryStore.

    Forces any hydrate to read from the durable DB store, proving the value
    survived in the database and not merely in a cache from the first instance.
    """
    cold_cache = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=cold_cache), \
         patch('services.memory_client.MemoryClientService.create_connection',
               return_value=cold_cache):
        # Constructing here is the "boot" of the new process.
        yield WorldState()


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def warm_store() -> Iterator[MemoryStore]:
    """Isolated MemoryStore for the pre-restart (warm) process lifetime."""
    _store = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=_store), \
         patch('services.memory_client.MemoryClientService.create_connection',
               return_value=_store):
        yield _store


@pytest.mark.integration
class TestWorldStateDurabilityAcrossRestart:
    """The user-message timestamp must outlive a process restart via the DB."""

    def test_user_message_timestamp_survives_restart(self, db: sqlite3.Connection, warm_store: MemoryStore) -> None:
        """absorb(user_message) then restart - durable snapshot equals the write.

        RED at HEAD: memory-only WorldState loses the value on restart.
        """
        when = utc_now() - timedelta(minutes=2)
        # Real production writer — identical to api/chat.py:281.
        WorldState().absorb(
            Signal(source="http_chat", kind="user_message",
                   payload={"text": "remember this"}, received_at=when)
        )

        with _simulated_restart() as restarted:
            survived = cast("datetime | None", restarted.snapshot()["last_user_message_at"])

        assert survived is not None, (
            "last_user_message_at did not survive the restart — it must be "
            "persisted durably (data_graph kind='machine_state'), not memory-only."
        )
        # The durable read must be the value we wrote (second precision — the
        # stored representation round-trips through isoformat/parse_utc).
        assert abs((survived - when).total_seconds()) < 1.0

    def test_durable_value_is_read_from_storage_not_a_live_dict(self, db: sqlite3.Connection, warm_store: MemoryStore) -> None:
        """A new WorldState starts with an empty ``_store``; passing proves a real hydrate.

        Guards against a test that would pass merely because some dict outlived
        the restart rather than because the value was read from durable storage.
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
            survived = cast("datetime | None", restarted.snapshot()["last_user_message_at"])

        assert survived is not None
        assert abs((survived - when).total_seconds()) < 1.0

    def test_latest_user_message_wins_after_restart(self, db: sqlite3.Connection, warm_store: MemoryStore) -> None:
        """Persistence must overwrite, never append/stale - gate compares the most recent write."""
        old = utc_now() - timedelta(hours=3)
        recent = utc_now() - timedelta(minutes=3)
        ws = WorldState()
        ws.absorb(Signal(source="http_chat", kind="user_message",
                         payload={"text": "first"}, received_at=old))
        ws.absorb(Signal(source="http_chat", kind="user_message",
                         payload={"text": "second"}, received_at=recent))

        with _simulated_restart() as restarted:
            survived = cast("datetime | None", restarted.snapshot()["last_user_message_at"])

        assert survived is not None
        assert abs((survived - recent).total_seconds()) < 1.0, (
            "durable read should reflect the latest user message, not an "
            "earlier one."
        )


@pytest.mark.integration
class TestSubconsciousGateDeStarvationAfterRestart:
    """The real worker gate must see the hydrated timestamp post-restart."""

    @contextmanager
    def _gate_against(self, world_state_instance: WorldState) -> Generator[None, None, None]:
        """Point the module-level world_state singleton at the given instance.

        Models the restarted process rebuilding its singletons. The real
        ``_check_gates`` runs unchanged - no gate logic is mocked.
        """
        import services.world_state as _ws_mod
        original = _ws_mod.world_state
        _ws_mod.world_state = world_state_instance
        try:
            yield
        finally:
            _ws_mod.world_state = original

    def test_idle_user_message_lets_gate_run_after_restart(self, db: sqlite3.Connection, warm_store: MemoryStore) -> None:
        """User last spoke 45 min ago: hydrated-but-old timestamp must not trip the gate.

        With no prior fire, ``_check_gates`` must return None, proving the
        hydrated value is compared correctly, not merely 'present, skip'.
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
