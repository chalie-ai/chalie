"""
Tests for IntentService — emit, get_pending, acknowledge, resolve, expire_stale.
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

class TestCognitiveIntent:

    def test_to_dict_contains_all_fields(self):
        intent = _make_intent()
        d = intent.to_dict()
        assert d["intent_id"] == "test-intent-001"
        assert d["intent_type"] == "execute"
        assert d["target_wrapper"] == "wrp_test"
        assert d["status"] == "pending"
        assert d["urgency"] == "normal"
        assert isinstance(d["payload"], dict)

    def test_to_json_emits_valid_json_with_all_fields(self):
        intent = _make_intent()
        json_str = intent.to_json()
        data = json.loads(json_str)
        assert data["intent_id"] == intent.intent_id
        assert data["payload"] == intent.payload
        assert data["status"] == intent.status

    def test_default_status_is_pending(self):
        intent = _make_intent()
        assert intent.status == "pending"

    def test_default_confidence(self):
        intent = _make_intent()
        assert intent.confidence == pytest.approx(0.5, abs=1e-9)

    def test_broadcast_intent_has_none_target(self):
        intent = _make_intent(target_wrapper=None)
        assert intent.target_wrapper is None


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

    def test_emit_stores_individual_key(self):
        svc, store = _make_service()
        intent = _make_intent()
        svc.emit(intent)

        raw = store.get("intent:test-intent-001")
        assert raw is not None
        stored = json.loads(raw)
        assert stored["intent_type"] == "execute"

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

class TestGetIntent:

    def test_get_intent_returns_stored_intent(self):
        svc, _ = _make_service()
        svc.emit(_make_intent())
        result = svc.get_intent("test-intent-001")
        assert result is not None
        assert result["intent_id"] == "test-intent-001"

    def test_get_intent_returns_none_for_unknown(self):
        svc, _ = _make_service()
        assert svc.get_intent("does-not-exist") is None


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

    def test_acknowledge_returns_false_for_missing(self):
        svc, _ = _make_service()
        ok = svc.acknowledge("nonexistent", "wrp_test")
        assert ok is False


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

    def test_resolve_failed(self):
        svc, store = _make_service()
        svc.emit(_make_intent())
        ok = svc.resolve("test-intent-001", {"status": "failed", "error": "permission denied"})

        assert ok is True
        stored = json.loads(store.get("intent:test-intent-001"))
        assert stored["status"] == "failed"

    def test_resolve_skipped(self):
        svc, store = _make_service()
        svc.emit(_make_intent())
        ok = svc.resolve("test-intent-001", {"status": "skipped", "reason": "user declined"})

        assert ok is True
        stored = json.loads(store.get("intent:test-intent-001"))
        assert stored["status"] == "skipped"

    def test_resolve_returns_false_for_missing(self):
        svc, _ = _make_service()
        ok = svc.resolve("nonexistent", {"status": "executed"})
        assert ok is False

    def test_resolve_unknown_status_defaults_to_executed(self):
        svc, store = _make_service()
        svc.emit(_make_intent())
        ok = svc.resolve("test-intent-001", {"status": "weird_value"})
        assert ok is True
        stored = json.loads(store.get("intent:test-intent-001"))
        assert stored["status"] == "executed"


# ---------------------------------------------------------------------------
# IntentService.expire_stale
# ---------------------------------------------------------------------------

class TestExpireStale:

    def test_expire_stale_expires_past_deadline(self):
        from services.time_utils import utc_now
        from datetime import timedelta

        svc, store = _make_service()
        past = (utc_now() - timedelta(hours=2)).isoformat()
        intent = CognitiveIntent(
            intent_id="stale-001",
            intent_type="notify",
            target_wrapper="wrp_y",
            payload={},
            expires_at=past,
        )
        svc.emit(intent)

        count = svc.expire_stale()
        assert count == 1

        stored = json.loads(store.get("intent:stale-001"))
        assert stored["status"] == "expired"

    def test_expire_stale_ignores_future_deadline(self):
        from services.time_utils import utc_now
        from datetime import timedelta

        svc, store = _make_service()
        future = (utc_now() + timedelta(hours=2)).isoformat()
        intent = CognitiveIntent(
            intent_id="future-001",
            intent_type="notify",
            target_wrapper="wrp_y",
            payload={},
            expires_at=future,
        )
        svc.emit(intent)

        count = svc.expire_stale()
        assert count == 0

        stored = json.loads(store.get("intent:future-001"))
        assert stored["status"] == "pending"

    def test_expire_stale_ignores_no_expiry(self):
        svc, _ = _make_service()
        svc.emit(_make_intent())  # no expires_at

        count = svc.expire_stale()
        assert count == 0

    def test_expire_stale_ignores_already_resolved(self):
        from services.time_utils import utc_now
        from datetime import timedelta

        svc, _ = _make_service()
        past = (utc_now() - timedelta(hours=1)).isoformat()
        intent = CognitiveIntent(
            intent_id="done-001",
            intent_type="execute",
            target_wrapper="wrp_y",
            payload={},
            expires_at=past,
        )
        svc.emit(intent)
        svc.resolve("done-001", {"status": "executed"})  # already terminal

        count = svc.expire_stale()
        assert count == 0  # already executed, not re-expired

    def test_expire_stale_returns_count(self):
        from services.time_utils import utc_now
        from datetime import timedelta

        svc, _ = _make_service()
        past = (utc_now() - timedelta(hours=1)).isoformat()
        for i in range(3):
            svc.emit(CognitiveIntent(
                intent_id=f"stale-{i}",
                intent_type="notify",
                target_wrapper="wrp_z",
                payload={},
                expires_at=past,
            ))

        count = svc.expire_stale()
        assert count == 3
