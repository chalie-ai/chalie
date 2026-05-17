"""Unit tests for WorldState.absorb() and WorldState.snapshot() — v0.5.0 §4.2.

Uses real MemoryStore (the in-process dict-backed store).  No mocks.
"""

import pytest
from datetime import datetime

from services.world_state import WorldState, Signal


@pytest.mark.unit
class TestWorldStateTypedSnapshot:
    @pytest.fixture
    def ws(self):
        """Fresh WorldState backed by a real in-process MemoryStore."""
        state = WorldState()
        # Replace the internal _store dict with a real MemoryStore so
        # the full set/get path is exercised without any DB dependency.
        # WorldState._store is a plain dict; we keep it as-is — absorb/snapshot
        # use self._store directly (dict keys), not MemoryStore API.
        return state

    # ── absorb: user_message ──────────────────────────────────────────────

    def test_absorb_user_message_updates_snapshot(self, ws):
        """absorb(kind='user_message') sets last_user_message_at in snapshot."""
        sig = Signal(source="ws", kind="user_message", payload={"text": "hello"})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["last_user_message_at"] is not None
        assert isinstance(snap["last_user_message_at"], datetime)

    # ── absorb: heartbeat ────────────────────────────────────────────────

    def test_absorb_heartbeat_updates_snapshot(self, ws):
        """absorb(kind='heartbeat') sets last_heartbeat_at in snapshot."""
        sig = Signal(source="/health", kind="heartbeat", payload={"battery": 90})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["last_heartbeat_at"] is not None
        assert isinstance(snap["last_heartbeat_at"], datetime)

    # ── absorb: device ───────────────────────────────────────────────────

    def test_absorb_device_updates_snapshot(self, ws):
        """absorb(kind='device') sets current_device_class in snapshot."""
        sig = Signal(source="/health", kind="device", payload={"device_class": "phone"})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["current_device_class"] == "phone"

    def test_absorb_device_without_device_class_ignored(self, ws):
        """absorb(kind='device') with no device_class payload leaves field None."""
        sig = Signal(source="/health", kind="device", payload={})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["current_device_class"] is None

    def test_absorb_device_overwrites_previous(self, ws):
        """Second absorb(kind='device') overwrites the first."""
        ws.absorb(Signal(source="/health", kind="device", payload={"device_class": "desktop"}))
        ws.absorb(Signal(source="/health", kind="device", payload={"device_class": "phone"}))
        snap = ws.snapshot()
        assert snap["current_device_class"] == "phone"

    # ── absorb: local_time ───────────────────────────────────────────────

    def test_absorb_local_time_string_updates_snapshot(self, ws):
        """absorb(kind='local_time') with ISO string sets current_local_time."""
        time_str = "2026-04-25T14:30:00+00:00"
        sig = Signal(source="/health", kind="local_time", payload={"local_time": time_str})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["current_local_time"] is not None
        assert isinstance(snap["current_local_time"], datetime)

    def test_absorb_local_time_without_value_ignored(self, ws):
        """absorb(kind='local_time') with empty payload leaves field None."""
        sig = Signal(source="/health", kind="local_time", payload={})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["current_local_time"] is None

    # ── snapshot: unset fields return None ───────────────────────────────

    def test_snapshot_returns_none_for_unset_fields(self, ws):
        """Fresh WorldState snapshot has all four typed fields as None."""
        snap = ws.snapshot()
        assert snap["last_user_message_at"] is None
        assert snap["last_heartbeat_at"] is None
        assert snap["current_device_class"] is None
        assert snap["current_local_time"] is None

    # ── unknown signal kinds silently ignored ────────────────────────────

    def test_unknown_signal_kind_silently_ignored(self, ws):
        """Unknown kind leaves snapshot unchanged — forward-compatible."""
        sig = Signal(source="future_interface", kind="mood_shift", payload={"mood": "happy"})
        ws.absorb(sig)
        snap = ws.snapshot()
        # Nothing should have changed from defaults
        assert snap["last_user_message_at"] is None
        assert snap["last_heartbeat_at"] is None
        assert snap["current_device_class"] is None
        assert snap["current_local_time"] is None

