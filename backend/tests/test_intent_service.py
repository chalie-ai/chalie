"""
Tests for IntentService — emit, get_pending, acknowledge, resolve.
"""

import json
import pytest

from services.intent_service import CognitiveIntent, IntentService, _BROADCAST_KEY

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_store():
    """Return a real MemoryStore (in-process, no external deps)."""
    from services.memory_store import MemoryStore
    return MemoryStore()


def _make_service(store=None):
    """Return an IntentService backed by a fresh MemoryStore."""
    store = store or _make_store()
    return IntentService(store=store), store


def _make_intent(
    intent_type="execute",
    target_wrapper="wrp_test",
    urgency="normal",
    expires_at=None,
):
    """Build a minimal CognitiveIntent."""
    return CognitiveIntent(
        intent_id="test-intent-001",
        intent_type=intent_type,
        target_wrapper=target_wrapper,
        payload={"action": "do_something"},
        urgency=urgency,
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# CognitiveIntent dataclass
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# IntentService.emit
# ---------------------------------------------------------------------------

class TestEmit:

    def test_emit_stores_in_list_key(self):
        svc, store = _make_service()
        intent = _make_intent()
        svc.emit(intent)

        items = store.lrange("intents:wrp_test", 0, -1)
        assert len(items) == 1
        stored = json.loads(items[0])
        assert stored["intent_id"] == "test-intent-001"

    def test_emit_broadcast_uses_broadcast_key(self):
        svc, store = _make_service()
        intent = _make_intent(target_wrapper=None)
        svc.emit(intent)

        items = store.lrange(f"intents:{_BROADCAST_KEY}", 0, -1)
        assert len(items) == 1

    def test_emit_multiple_intents_accumulate(self):
        svc, store = _make_service()
        for i in range(3):
            intent = CognitiveIntent(
                intent_id=f"id-{i}",
                intent_type="notify",
                target_wrapper="wrp_a",
                payload={"n": i},
            )
            svc.emit(intent)

        items = store.lrange("intents:wrp_a", 0, -1)
        assert len(items) == 3


# ---------------------------------------------------------------------------
# IntentService.get_pending
# ---------------------------------------------------------------------------

class TestGetPending:

    def test_get_pending_returns_pending_intents(self):
        svc, _ = _make_service()
        svc.emit(_make_intent())

        results = svc.get_pending("wrp_test")
        assert len(results) == 1
        assert results[0]["intent_id"] == "test-intent-001"

    def test_get_pending_marks_as_delivered(self):
        svc, store = _make_service()
        svc.emit(_make_intent())
        svc.get_pending("wrp_test")

        raw = store.get("intent:test-intent-001")
        stored = json.loads(raw)
        assert stored["status"] == "delivered"

    def test_get_pending_does_not_return_already_delivered(self):
        svc, _ = _make_service()
        svc.emit(_make_intent())
        svc.get_pending("wrp_test")  # first call → marks delivered

        results = svc.get_pending("wrp_test")  # second call
        assert results == []

    def test_get_pending_includes_broadcast_intents(self):
        svc, _ = _make_service()
        broadcast = _make_intent(target_wrapper=None)
        svc.emit(broadcast)

        results = svc.get_pending("some_other_wrapper")
        assert len(results) == 1
        assert results[0]["target_wrapper"] is None

    def test_get_pending_respects_limit(self):
        svc, _ = _make_service()
        for i in range(5):
            svc.emit(CognitiveIntent(
                intent_id=f"id-{i}",
                intent_type="notify",
                target_wrapper="wrp_x",
                payload={},
            ))

        results = svc.get_pending("wrp_x", limit=3)
        assert len(results) == 3

    def test_get_pending_skips_expired_intents(self):
        from services.time_utils import utc_now
        from datetime import timedelta

        svc, _ = _make_service()
        past = (utc_now() - timedelta(hours=1)).isoformat()
        expired_intent = CognitiveIntent(
            intent_id="expired-001",
            intent_type="execute",
            target_wrapper="wrp_test",
            payload={},
            expires_at=past,
        )
        svc.emit(expired_intent)

        results = svc.get_pending("wrp_test")
        assert results == []


# ---------------------------------------------------------------------------
# IntentService.get_intent
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# IntentService.acknowledge
# ---------------------------------------------------------------------------

class TestAcknowledge:

    def test_acknowledge_sets_status(self):
        svc, store = _make_service()
        svc.emit(_make_intent())
        ok = svc.acknowledge("test-intent-001", "wrp_test")

        assert ok is True
        raw = store.get("intent:test-intent-001")
        stored = json.loads(raw)
        assert stored["status"] == "acknowledged"




# ---------------------------------------------------------------------------
# IntentService.resolve
# ---------------------------------------------------------------------------

class TestResolve:

    def test_resolve_executed(self):
        svc, store = _make_service()
        svc.emit(_make_intent())
        ok = svc.resolve("test-intent-001", {"status": "executed", "result": {"pr_url": "https://example.com"}})

        assert ok is True
        stored = json.loads(store.get("intent:test-intent-001"))
        assert stored["status"] == "executed"
        assert stored["execution_result"]["result"]["pr_url"] == "https://example.com"




