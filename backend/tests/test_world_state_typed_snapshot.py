"""Tests WorldState with real MemoryStore — no mocks."""

import sqlite3
from collections.abc import Generator
from datetime import datetime
from unittest.mock import patch

import pytest

from services.world_state import WorldState, Signal


@pytest.mark.unit
class TestWorldStateTypedSnapshot:
    @pytest.fixture
    def ws(self, db: sqlite3.Connection) -> Generator[WorldState, None, None]:
        """Cold-booted WorldState: fresh MemoryStore isolates absorb→snapshot per test."""
        from services.memory_store import MemoryStore

        cold_cache = MemoryStore()
        with patch('services.memory_store.get_shared_store', return_value=cold_cache), \
             patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=cold_cache):
            yield WorldState()

    # ── absorb: user_message ──────────────────────────────────────────────

    def test_absorb_user_message_updates_snapshot(self, ws: WorldState) -> None:
        sig = Signal(source="ws", kind="user_message", payload={"text": "hello"})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["last_user_message_at"] is not None
        assert isinstance(snap["last_user_message_at"], datetime)

    # ── absorb: heartbeat ────────────────────────────────────────────────

    def test_absorb_heartbeat_updates_snapshot(self, ws: WorldState) -> None:
        sig = Signal(source="/health", kind="heartbeat", payload={"battery": 90})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["last_heartbeat_at"] is not None
        assert isinstance(snap["last_heartbeat_at"], datetime)

    # ── absorb: device ───────────────────────────────────────────────────

    def test_absorb_device_updates_snapshot(self, ws: WorldState) -> None:
        sig = Signal(source="/health", kind="device", payload={"device_class": "phone"})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["current_device_class"] == "phone"

    def test_absorb_device_without_device_class_ignored(self, ws: WorldState) -> None:
        sig = Signal(source="/health", kind="device", payload={})
        ws.absorb(sig)
        snap = ws.snapshot()
        assert snap["current_device_class"] is None

    def test_absorb_device_overwrites_previous(self, ws: WorldState) -> None:
        ws.absorb(Signal(source="/health", kind="device", payload={"device_class": "desktop"}))
        ws.absorb(Signal(source="/health", kind="device", payload={"device_class": "phone"}))
        snap = ws.snapshot()
        assert snap["current_device_class"] == "phone"

    # ── snapshot: unset fields return None ───────────────────────────────

    def test_snapshot_returns_none_for_unset_fields(self, ws: WorldState) -> None:
        snap = ws.snapshot()
        assert snap["last_user_message_at"] is None
        assert snap["last_heartbeat_at"] is None
        assert snap["current_device_class"] is None

    # ── unknown signal kinds silently ignored ────────────────────────────

    def test_unknown_signal_kind_silently_ignored(self, ws: WorldState) -> None:
        sig = Signal(source="future_interface", kind="mood_shift", payload={"mood": "happy"})
        ws.absorb(sig)
        snap = ws.snapshot()
        # Nothing should have changed from defaults
        assert snap["last_user_message_at"] is None
        assert snap["last_heartbeat_at"] is None
        assert snap["current_device_class"] is None
