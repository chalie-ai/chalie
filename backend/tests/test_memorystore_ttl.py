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

import json
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
        store._hashes,
        store._sorted_sets,
        store._sets,
    ):
        if key in keyspace:
            _, expiry = keyspace[key]
            return expiry is not None
    return False


# ---------------------------------------------------------------------------
# 1. output_service — enqueue path + re-queue path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOutputServiceQueueTTL:
    """After every lpush to output-queue, expire(output-queue, 3600) must fire.

    The output-queue is written by enqueue_act (primary path) and by the
    re-queue branch inside dequeue() when a type-mismatch is detected.
    enqueue_text does NOT touch output-queue; it uses setex on the output key
    and rpush/expire on notifications:recent.
    """

    @pytest.fixture
    def svc(self):
        store = MemoryStore()
        connections = {"memory": {"topics": {"output_queue": "output-queue"}}}
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store), \
             patch("services.config_service.ConfigService.connections",
                   return_value=connections):
            from services.output_service import OutputService
            return OutputService(), store

    def test_enqueue_act_sets_ttl_on_output_queue(self, svc):
        """enqueue_act calls lpush then expire(output-queue, 3600)."""
        service, store = svc
        service.enqueue_act(topic="t", actions=["search"], channel="web:default:1")
        assert _has_ttl(store, "output-queue"), "output-queue must have a TTL after enqueue_act lpush"

    def test_requeue_sets_ttl_on_output_queue(self, svc):
        """dequeue re-queue path (type mismatch) calls expire on output-queue."""
        service, store = svc

        # Push a TEXT output and an ACT output so that dequeue finds ACT on second pop.
        # brpop pops from the right, so ACT is at the right (last) position.
        text_output = {"id": "out-text", "type": "TEXT", "topic": "t", "metadata": {}}
        act_output = {"id": "out-act", "type": "ACT", "topic": "t",
                      "metadata": {"actions": [], "downstream_mode": "ACT",
                                   "act_history": [], "loop_id": "", "generation_time": 0.0}}
        store.set("output:out-text", json.dumps(text_output))
        store.set("output:out-act", json.dumps(act_output))
        # rpush appends to the right; brpop pops from the right.
        # Push ACT first (left), then TEXT (rightmost) so TEXT is popped first.
        store.rpush("output-queue", "out-act")
        store.rpush("output-queue", "out-text")

        # dequeue asks for ACT: pops out-text (type mismatch) → re-queues it → pops out-act → returns
        result = service.dequeue(output_type="ACT", timeout=2)

        assert result is not None and result["type"] == "ACT"
        assert _has_ttl(store, "output-queue"), "output-queue must have a TTL after re-queue lpush"


# ---------------------------------------------------------------------------
# 2. background_llm_queue — rpush to bg_llm:queue
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBackgroundLLMQueueTTL:
    """After rpush to bg_llm:queue, expire(bg_llm:queue, 600) must fire."""

    def test_enqueue_sets_ttl_on_queue_key(self):
        store = MemoryStore()
        with patch("services.background_llm_queue.MemoryClientService.create_connection",
                   return_value=store):
            from services.background_llm_queue import BackgroundLLMProxy, QUEUE_KEY
            proxy = BackgroundLLMProxy("test-agent")

        # Intercept brpop so the proxy doesn't block indefinitely
        with patch.object(store, "brpop", return_value=None):
            proxy.send_message("sys", "user")

        assert _has_ttl(store, QUEUE_KEY), "bg_llm:queue must have a TTL after rpush"


# ---------------------------------------------------------------------------
# 3. event_bus_service — rpush to event_bus:{event_type}
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEventBusServiceQueueTTL:
    """After rpush to event_bus:<type>, expire(event_bus:<type>, 3600) must fire."""

    def test_emit_sets_ttl_on_event_queue(self):
        store = MemoryStore()
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store):
            from services.event_bus_service import EventBusService, ENCODE_EVENT
            bus = EventBusService()

        bus.emit(ENCODE_EVENT, {"data": "test"})

        queue_key = f"event_bus:{ENCODE_EVENT}"
        assert _has_ttl(store, queue_key), f"{queue_key} must have a TTL after rpush"

    def test_emit_custom_event_type_sets_ttl(self):
        """TTL is applied regardless of the event type string."""
        store = MemoryStore()
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store):
            from services.event_bus_service import EventBusService
            bus = EventBusService()

        bus.emit("custom_event", {"key": "value"})

        assert _has_ttl(store, "event_bus:custom_event")


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
             patch("services.app_update_service.AppUpdateService.backup_database"), \
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


# ---------------------------------------------------------------------------
# 10. dmn_service — zadd to dmn:deliveries
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDMNServiceDeliveryZSetTTL:
    """After zadd to dmn:deliveries, expire(dmn:deliveries, 86400) must fire."""

    def test_record_delivery_sets_ttl_on_deliveries_zset(self):
        store = MemoryStore()
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store), \
             patch("services.database_service.get_shared_db_service"):
            from services.dmn_service import DMNService, _DELIVERY_ZSET
            svc = DMNService.__new__(DMNService)
            svc._store = store

        svc._record_delivery()

        assert _has_ttl(store, _DELIVERY_ZSET), \
            "dmn:deliveries must have TTL=86400 after zadd"


# ---------------------------------------------------------------------------
# 11. api/push.py — store.set(VAPID_KEYS_KEY, ..., ex=86400)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPushAPIVapidKeyTTL:
    """Generated VAPID keys stored via store.set(..., ex=86400) must have a TTL."""

    def test_get_vapid_keys_sets_ttl_on_generated_keys(self):
        store = MemoryStore()
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=store), \
             patch.dict("os.environ", {"VAPID_PUBLIC_KEY": "", "VAPID_PRIVATE_KEY": ""},
                        clear=False):
            # Remove env vars so the generation path fires
            import os
            env_backup = {k: os.environ.pop(k, None)
                          for k in ("VAPID_PUBLIC_KEY", "VAPID_PRIVATE_KEY")}
            try:
                from api.push import _get_vapid_keys, VAPID_KEYS_KEY
                # Ensure no cached key so we hit the generation branch
                store._strings.pop(VAPID_KEYS_KEY, None)
                _get_vapid_keys()
            finally:
                for k, v in env_backup.items():
                    if v is not None:
                        os.environ[k] = v

        assert _has_ttl(store, VAPID_KEYS_KEY), \
            f"{VAPID_KEYS_KEY} must have ex=86400 TTL after key generation"


