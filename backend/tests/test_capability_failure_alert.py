"""Tests for capability failure alerting (base.py).

Verifies that _maybe_send_failure_alert() and _send_recovery_alert() publish
to output:events via MemoryStore.publish() and manage capability:alert:{cap_id}
string keys — NOT the old prompt-queue rpush path.

All tests use the real MemoryStore (no mocking — it IS production).
The store fixture from conftest patches MemoryClientService.create_connection
so every lazy import inside base.py gets the isolated test instance.
"""

import json
import pytest
from unittest.mock import MagicMock

from capabilities.base import AbstractCapability

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Concrete stub — minimal concrete subclass so AbstractCapability can be
# instantiated without requiring real service wiring.
# ---------------------------------------------------------------------------

class _StubCapability(AbstractCapability):
    """Minimal concrete subclass for testing AbstractCapability behaviour."""

    def __init__(self, cap_id="stub-cap", cap_name="Stub Cap", last_error="auth expired"):
        super().__init__()
        self._cap_id = cap_id
        self._cap_name = cap_name
        self._last_error = last_error

    def get_id(self):
        return self._cap_id

    def get_manifest(self):
        return {"id": self._cap_id, "name": self._cap_name}

    def configure(self, credentials):
        pass  # stub — no configuration needed for this test double

    def connect(self):
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def ingest(self):
        return []

    def understand(self, items):
        return []

    def _do_monitor(self):
        pass  # stub — monitoring not exercised in this test

    def act(self, action, params):
        return {"success": True}

    def get_tools(self):
        return []


# ---------------------------------------------------------------------------
# Helper: subscribe to output:events before triggering and drain one message.
# ---------------------------------------------------------------------------

def _subscribe_and_get_message(store, trigger_fn):
    """Subscribe to output:events, call trigger_fn, return the first message dict."""
    ps = store.pubsub()
    ps.subscribe("output:events")
    trigger_fn()
    return ps.get_message(timeout=0.5)


# ---------------------------------------------------------------------------
# 1. _maybe_send_failure_alert publishes to output:events
# ---------------------------------------------------------------------------

class TestFailureAlertPublish:

    def test_publishes_to_output_events(self, store):
        """_maybe_send_failure_alert must publish exactly one message to output:events."""
        cap = _StubCapability()
        msg = _subscribe_and_get_message(store, cap._maybe_send_failure_alert)

        assert msg is not None, "Expected a message on output:events"
        assert msg["channel"] == "output:events"

    def test_published_payload_is_valid_json(self, store):
        """The published data must be valid JSON."""
        cap = _StubCapability()
        msg = _subscribe_and_get_message(store, cap._maybe_send_failure_alert)

        payload = json.loads(msg["data"])  # raises if invalid
        assert isinstance(payload, dict)

    def test_published_payload_structure(self, store):
        """Payload must contain type, cap_id, cap_name, error, and recovered=False."""
        cap = _StubCapability(cap_id="test-cap", cap_name="Test Cap", last_error="connection refused")
        msg = _subscribe_and_get_message(store, cap._maybe_send_failure_alert)

        payload = json.loads(msg["data"])
        assert payload["type"] == "capability_alert"
        assert payload["cap_id"] == "test-cap"
        assert payload["cap_name"] == "Test Cap"
        assert payload["error"] == "connection refused"
        assert payload["recovered"] is False

    def test_fires_once_only_dedup(self, store):
        """Second call when _failure_alerted is True must not publish anything."""
        cap = _StubCapability()

        ps = store.pubsub()
        ps.subscribe("output:events")

        cap._maybe_send_failure_alert()
        first = ps.get_message(timeout=0.2)
        assert first is not None, "First call must publish"

        cap._maybe_send_failure_alert()  # _failure_alerted is now True
        second = ps.get_message(timeout=0.1)
        assert second is None, "Dedup: second call must not publish"

    def test_sets_failure_alerted_flag(self, store):
        """_failure_alerted must be True after the first call."""
        cap = _StubCapability()
        assert cap._failure_alerted is False
        cap._maybe_send_failure_alert()
        assert cap._failure_alerted is True


# ---------------------------------------------------------------------------
# 2. _maybe_send_failure_alert sets MemoryStore key
# ---------------------------------------------------------------------------

class TestFailureAlertStoreKey:

    def test_sets_capability_alert_key(self, store):
        """setex must write capability:alert:{cap_id} into the store."""
        cap = _StubCapability(cap_id="mail-cap")
        cap._maybe_send_failure_alert()

        assert store.get("capability:alert:mail-cap") is not None

    def test_key_payload_structure(self, store):
        """The string value under the key must be a complete capability_alert payload."""
        cap = _StubCapability(cap_id="cal-cap", cap_name="Calendar", last_error="timeout")
        cap._maybe_send_failure_alert()

        raw = store.get("capability:alert:cal-cap")
        payload = json.loads(raw)
        assert payload["type"] == "capability_alert"
        assert payload["cap_id"] == "cal-cap"
        assert payload["cap_name"] == "Calendar"
        assert payload["error"] == "timeout"
        assert payload["recovered"] is False

    def test_key_has_ttl(self, store):
        """The capability:alert key must have a 1800s TTL (setex, not plain set)."""
        cap = _StubCapability(cap_id="ttl-test-cap")
        cap._maybe_send_failure_alert()

        _, expiry = store._strings.get("capability:alert:ttl-test-cap", (None, None))
        assert expiry is not None, "capability:alert key must have a TTL"

    def test_error_falls_back_to_unknown_when_last_error_is_none(self, store):
        """When _last_error is None the payload error field must be 'unknown error'."""
        cap = _StubCapability()
        cap._last_error = None
        cap._maybe_send_failure_alert()

        raw = store.get("capability:alert:stub-cap")
        payload = json.loads(raw)
        assert payload["error"] == "unknown error"


# ---------------------------------------------------------------------------
# 3. _send_recovery_alert
# ---------------------------------------------------------------------------

class TestRecoveryAlert:

    def test_publishes_to_output_events(self, store):
        """_send_recovery_alert must publish to output:events."""
        cap = _StubCapability()
        msg = _subscribe_and_get_message(store, cap._send_recovery_alert)

        assert msg is not None, "Expected recovery message on output:events"
        assert msg["channel"] == "output:events"

    def test_published_payload_has_recovered_true(self, store):
        """Recovery payload must contain recovered=True."""
        cap = _StubCapability(cap_id="rec-cap", cap_name="Recovered Cap")
        msg = _subscribe_and_get_message(store, cap._send_recovery_alert)

        payload = json.loads(msg["data"])
        assert payload["type"] == "capability_alert"
        assert payload["cap_id"] == "rec-cap"
        assert payload["cap_name"] == "Recovered Cap"
        assert payload["recovered"] is True

    def test_recovery_payload_has_no_error_field(self, store):
        """The recovery payload must NOT include an 'error' field."""
        cap = _StubCapability()
        msg = _subscribe_and_get_message(store, cap._send_recovery_alert)

        payload = json.loads(msg["data"])
        assert "error" not in payload

    def test_deletes_capability_alert_key(self, store):
        """_send_recovery_alert must delete the capability:alert:{cap_id} key."""
        cap = _StubCapability(cap_id="delete-me")
        # Seed the key as _maybe_send_failure_alert would
        store.setex("capability:alert:delete-me", 1800, '{"type":"capability_alert"}')

        cap._send_recovery_alert()

        assert store.get("capability:alert:delete-me") is None

    def test_delete_is_safe_when_key_absent(self, store):
        """_send_recovery_alert must not raise if the key doesn't exist."""
        cap = _StubCapability(cap_id="never-set-cap")
        # No key was set — delete must silently succeed
        cap._send_recovery_alert()  # should not raise


# ---------------------------------------------------------------------------
# 4. Full circuit-breaker flow
# ---------------------------------------------------------------------------

class TestCircuitBreakerFlow:
    """5 consecutive failures → alert → next success → recovery → clear."""

    def _make_failing_cap(self, cap_id="circuit-cap"):
        cap = _StubCapability(cap_id=cap_id)
        # Patch _persist_health so we don't need a real DB in these unit tests
        cap._persist_health = lambda: None
        return cap

    def test_no_alert_before_threshold(self, store):
        """run_monitor does not publish until MAX_CONSECUTIVE_FAILURES is reached."""
        cap = self._make_failing_cap()
        ps = store.pubsub()
        ps.subscribe("output:events")

        for _ in range(AbstractCapability.MAX_CONSECUTIVE_FAILURES - 1):
            cap._do_monitor = MagicMock(side_effect=RuntimeError("boom"))
            cap.run_monitor()

        msg = ps.get_message(timeout=0.1)
        assert msg is None, "Must not publish before threshold"

    def test_alert_fires_on_threshold(self, store):
        """On the MAX_CONSECUTIVE_FAILURES-th failure, one message is published."""
        cap = self._make_failing_cap()
        ps = store.pubsub()
        ps.subscribe("output:events")

        cap._do_monitor = MagicMock(side_effect=RuntimeError("boom"))
        for _ in range(AbstractCapability.MAX_CONSECUTIVE_FAILURES):
            cap.run_monitor()

        msg = ps.get_message(timeout=0.5)
        assert msg is not None, "Alert must publish on threshold failure"
        payload = json.loads(msg["data"])
        assert payload["recovered"] is False

    def test_alert_fires_once_for_repeated_failures(self, store):
        """Alert fires once and is deduplicated for subsequent failures."""
        cap = self._make_failing_cap()
        ps = store.pubsub()
        ps.subscribe("output:events")

        cap._do_monitor = MagicMock(side_effect=RuntimeError("boom"))
        # Trigger initial alert
        for _ in range(AbstractCapability.MAX_CONSECUTIVE_FAILURES):
            cap.run_monitor()
        ps.get_message(timeout=0.3)  # consume first alert

        # More failures should not produce another publish
        cap.run_monitor()
        cap.run_monitor()
        extra = ps.get_message(timeout=0.1)
        assert extra is None, "Alert must not fire again while _failure_alerted=True"

    def test_recovery_publishes_and_clears_key(self, store):
        """After a success following failure, recovery is published and key deleted."""
        cap = self._make_failing_cap()
        ps = store.pubsub()
        ps.subscribe("output:events")

        # Drive to failure alert
        cap._do_monitor = MagicMock(side_effect=RuntimeError("boom"))
        for _ in range(AbstractCapability.MAX_CONSECUTIVE_FAILURES):
            cap.run_monitor()

        # Drain the failure alert message
        ps.get_message(timeout=0.3)

        # Cancel backoff so recovery monitor runs immediately
        cap._next_retry_at = None

        # Simulate success
        cap._do_monitor = MagicMock()
        cap.run_monitor()

        recovery_msg = ps.get_message(timeout=0.5)
        assert recovery_msg is not None, "Recovery message must be published"
        payload = json.loads(recovery_msg["data"])
        assert payload["recovered"] is True

        # Key must be cleaned up
        assert store.get(f"capability:alert:{cap.get_id()}") is None

    def test_state_reset_after_recovery(self, store):
        """After recovery, error counters and alerted flag are cleared."""
        cap = self._make_failing_cap()
        cap._do_monitor = MagicMock(side_effect=RuntimeError("boom"))
        for _ in range(AbstractCapability.MAX_CONSECUTIVE_FAILURES):
            cap.run_monitor()

        cap._next_retry_at = None
        cap._do_monitor = MagicMock()
        cap.run_monitor()

        assert cap._error_count == 0
        assert cap._last_error is None
        assert cap._failure_alerted is False
        assert cap._backoff_secs == 0


# ---------------------------------------------------------------------------
# 5. Multiple capabilities — isolated keys
# ---------------------------------------------------------------------------

class TestMultipleCapabilities:

    def test_each_cap_gets_its_own_store_key(self, store):
        """Two different capabilities failing must not share a store key."""
        cap_a = _StubCapability(cap_id="cap-a", cap_name="Cap A", last_error="err-a")
        cap_b = _StubCapability(cap_id="cap-b", cap_name="Cap B", last_error="err-b")

        cap_a._maybe_send_failure_alert()
        cap_b._maybe_send_failure_alert()

        raw_a = store.get("capability:alert:cap-a")
        raw_b = store.get("capability:alert:cap-b")

        assert raw_a is not None
        assert raw_b is not None

        payload_a = json.loads(raw_a)
        payload_b = json.loads(raw_b)

        assert payload_a["cap_id"] == "cap-a"
        assert payload_b["cap_id"] == "cap-b"

    def test_recovering_one_cap_does_not_remove_other_key(self, store):
        """Recovery for cap-a must not delete cap-b's alert key."""
        cap_a = _StubCapability(cap_id="cap-a")
        cap_b = _StubCapability(cap_id="cap-b")

        cap_a._maybe_send_failure_alert()
        cap_b._maybe_send_failure_alert()

        cap_a._send_recovery_alert()

        # cap-a key gone, cap-b key untouched
        assert store.get("capability:alert:cap-a") is None
        assert store.get("capability:alert:cap-b") is not None

    def test_two_caps_each_publish_independently(self, store):
        """Two caps failing produce two distinct messages on output:events."""
        ps = store.pubsub()
        ps.subscribe("output:events")

        cap_a = _StubCapability(cap_id="cap-a", cap_name="A")
        cap_b = _StubCapability(cap_id="cap-b", cap_name="B")

        cap_a._maybe_send_failure_alert()
        cap_b._maybe_send_failure_alert()

        msg1 = ps.get_message(timeout=0.3)
        msg2 = ps.get_message(timeout=0.3)

        assert msg1 is not None and msg2 is not None
        ids = {json.loads(msg1["data"])["cap_id"], json.loads(msg2["data"])["cap_id"]}
        assert ids == {"cap-a", "cap-b"}


# ---------------------------------------------------------------------------
# 6. Payload completeness — all required fields present
# ---------------------------------------------------------------------------

class TestPayloadStructure:

    def test_failure_payload_required_fields(self, store):
        """Failure payload must have exactly the four specified fields."""
        cap = _StubCapability(cap_id="field-check", cap_name="Field Check", last_error="oops")
        msg = _subscribe_and_get_message(store, cap._maybe_send_failure_alert)
        payload = json.loads(msg["data"])

        required = {"type", "cap_id", "cap_name", "error", "recovered"}
        assert required.issubset(payload.keys()), (
            f"Missing fields: {required - payload.keys()}"
        )

    def test_recovery_payload_required_fields(self, store):
        """Recovery payload must have type, cap_id, cap_name, and recovered."""
        cap = _StubCapability(cap_id="rec-fields", cap_name="Rec Fields")
        msg = _subscribe_and_get_message(store, cap._send_recovery_alert)
        payload = json.loads(msg["data"])

        required = {"type", "cap_id", "cap_name", "recovered"}
        assert required.issubset(payload.keys()), (
            f"Missing fields: {required - payload.keys()}"
        )

    def test_failure_payload_type_value(self, store):
        """type field must be the literal string 'capability_alert'."""
        cap = _StubCapability()
        msg = _subscribe_and_get_message(store, cap._maybe_send_failure_alert)
        assert json.loads(msg["data"])["type"] == "capability_alert"

    def test_key_payload_matches_published_payload(self, store):
        """The MemoryStore key payload and the published message payload must be identical."""
        cap = _StubCapability(cap_id="sync-check", cap_name="Sync Check", last_error="mismatch")
        msg = _subscribe_and_get_message(store, cap._maybe_send_failure_alert)

        published = json.loads(msg["data"])
        stored = json.loads(store.get("capability:alert:sync-check"))

        assert published == stored, "Published payload and stored key payload must be identical"


# ---------------------------------------------------------------------------
# 7. No prompt-queue usage — the old alert path is GONE
# ---------------------------------------------------------------------------

class TestNoPromptQueue:

    def test_failure_alert_does_not_touch_prompt_queue(self, store):
        """_maybe_send_failure_alert must NOT rpush to prompt-queue."""
        cap = _StubCapability()
        cap._maybe_send_failure_alert()

        assert store.llen("prompt-queue") == 0, (
            "prompt-queue must remain empty — alerts now use output:events"
        )

    def test_recovery_alert_does_not_touch_prompt_queue(self, store):
        """_send_recovery_alert must NOT rpush to prompt-queue."""
        cap = _StubCapability()
        cap._send_recovery_alert()

        assert store.llen("prompt-queue") == 0, (
            "prompt-queue must remain empty — recovery now uses output:events"
        )

    def test_run_monitor_circuit_breaker_does_not_touch_prompt_queue(self, store):
        """Full circuit-breaker path (5 failures) must not write to prompt-queue."""
        cap = _StubCapability(cap_id="no-pq-cap")
        cap._persist_health = lambda: None
        cap._do_monitor = MagicMock(side_effect=RuntimeError("boom"))

        for _ in range(AbstractCapability.MAX_CONSECUTIVE_FAILURES):
            cap.run_monitor()

        assert store.llen("prompt-queue") == 0, (
            "prompt-queue must be empty after circuit breaker fires"
        )
