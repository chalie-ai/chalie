"""Tests that every MemoryStore write for a queue or coordination key sets a TTL."""

import pytest

from services.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _has_ttl(store: MemoryStore, key: str) -> bool:
    for keyspace in (
        store._strings,
        store._lists,
        store._sorted_sets,
        store._sets,
    ):
        if key in keyspace:
            _, expiry = keyspace[key]
            return expiry is not None
    return False


# ---------------------------------------------------------------------------
# 6. intent_service — rpush to intents:{target}
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIntentServiceQueueTTL:
    """After rpush to intents:{target}, expire(intents:{target}, _INTENT_TTL_SECONDS) must fire."""

    def test_emit_sets_ttl_on_intents_list(self) -> None:
        store = MemoryStore()
        from services.intent_service import IntentService, CognitiveIntent

        svc = IntentService(store=store)
        intent = CognitiveIntent(
            intent_id="ttl-test-intent-001",
            intent_type="notify",
            target_wrapper="wrapper-abc",
            payload={"message": "hello"},
        )
        svc.emit(intent)

        list_key = "intents:wrapper-abc"
        assert _has_ttl(store, list_key), f"{list_key} must have a TTL after rpush"

    def test_emit_broadcast_sets_ttl_on_broadcast_list(self) -> None:
        store = MemoryStore()
        from services.intent_service import IntentService, CognitiveIntent, _BROADCAST_KEY

        svc = IntentService(store=store)
        intent = CognitiveIntent(
            intent_id="ttl-broadcast-001",
            intent_type="suggest",
            target_wrapper=None,
            payload={"text": "tip"},
        )
        svc.emit(intent)

        list_key = f"intents:{_BROADCAST_KEY}"
        assert _has_ttl(store, list_key), f"{list_key} must have a TTL for broadcast intents"




