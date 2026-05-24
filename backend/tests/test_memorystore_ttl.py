"""
Unit tests for MemoryStore TTL coverage across services.

Every MemoryStore write that represents a queue or coordination key must set a
TTL so the background reaper can eventually reclaim memory.  These tests verify
that the TTL is applied at the point of the write — not merely that the
operation succeeds.

Pattern used throughout:
- Instantiate the real MemoryStore (no mocking — it IS production).
- Call the method under test with MemoryClientService.create_connection patched
  to return the real store instance.
- Inspect the internal keyspace tuple: (value, expiry_timestamp).
  A non-None expiry confirms the TTL was set.
"""

import pytest
from unittest.mock import patch

from services.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _has_ttl(store: MemoryStore, key: str) -> bool:
    """Return True if *key* exists in any keyspace and has a non-None expiry."""
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
# 5. app_update_service — set(IN_PROGRESS_KEY, "1", ex=3600)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAppUpdateServiceTTL:
    """store.set(IN_PROGRESS_KEY, '1', ex=3600) must set a TTL.

    apply_update deletes IN_PROGRESS_KEY when the update succeeds or fails,
    so we verify the TTL contract by spying on store.set and confirming the
    ex=3600 argument was passed before the key is deleted.
    """

    def test_apply_update_in_progress_key_set_with_ex_3600(self):
        store = MemoryStore()
        set_calls = []
        original_set = store.set

        def spy_set(key, value, **kwargs):
            set_calls.append((key, value, kwargs))
            return original_set(key, value, **kwargs)

        store.set = spy_set

        with patch("services.app_update_service.MemoryClientService.create_connection",
                   return_value=store), \
             patch("services.app_update_service.AppUpdateService.detect_deployment_mode",
                   return_value="installed"), \
             patch("services.app_update_service.AppUpdateService.get_current_version",
                   return_value="0.2.0"), \
             patch("services.app_update_service.AppUpdateService.download_and_validate",
                   side_effect=RuntimeError("abort early for test")):
            from services.app_update_service import AppUpdateService, IN_PROGRESS_KEY
            svc = AppUpdateService()
            svc.apply_update("v9.9.9")

        in_progress_calls = [c for c in set_calls if c[0] == IN_PROGRESS_KEY]
        assert len(in_progress_calls) == 1, "IN_PROGRESS_KEY must be set exactly once"
        assert in_progress_calls[0][2].get("ex") == 3600, \
            f"IN_PROGRESS_KEY must be set with ex=3600, got kwargs={in_progress_calls[0][2]}"


# ---------------------------------------------------------------------------
# 6. intent_service — rpush to intents:{target}
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIntentServiceQueueTTL:
    """After rpush to intents:{target}, expire(intents:{target}, _INTENT_TTL_SECONDS) must fire."""

    def test_emit_sets_ttl_on_intents_list(self):
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

    def test_emit_broadcast_sets_ttl_on_broadcast_list(self):
        """Broadcast intents (target_wrapper=None) must also get a TTL."""
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




