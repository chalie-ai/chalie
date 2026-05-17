"""Tests for capability failure alerting (base.py).

Verifies that _maybe_send_failure_alert() and _send_recovery_alert()
broadcast via WebSocketBroker and manage capability:alert:{cap_id} keys.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from capabilities.base import AbstractCapability

pytestmark = pytest.mark.unit


class _StubCapability(AbstractCapability):

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
        pass

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
        pass

    def act(self, action, params):
        return {"success": True}

    def get_tools(self):
        return []


class TestFailureAlertBroadcast:

    def test_broadcast_payload_structure(self, store):
        cap = _StubCapability(cap_id="test-cap", cap_name="Test Cap", last_error="connection refused")
        with patch("capabilities.base.WebSocketBroker") as mock_cls:
            mock_broker = MagicMock()
            mock_cls.return_value = mock_broker
            cap._maybe_send_failure_alert()
            mock_broker.broadcast.assert_called_once()
            payload = mock_broker.broadcast.call_args[0][0]
            assert payload["type"] == "capability_alert"
            assert payload["cap_id"] == "test-cap"
            assert payload["cap_name"] == "Test Cap"
            assert payload["error"] == "connection refused"
            assert payload["recovered"] is False

    def test_fires_once_only_dedup(self, store):
        cap = _StubCapability()
        with patch("capabilities.base.WebSocketBroker") as mock_cls:
            mock_broker = MagicMock()
            mock_cls.return_value = mock_broker
            cap._maybe_send_failure_alert()
            assert mock_broker.broadcast.call_count == 1
            cap._maybe_send_failure_alert()
            assert mock_broker.broadcast.call_count == 1


class TestFailureAlertStoreKey:

    def test_sets_capability_alert_key(self, store):
        cap = _StubCapability(cap_id="mail-cap")
        cap._maybe_send_failure_alert()
        assert store.get("capability:alert:mail-cap") is not None

    def test_key_has_ttl(self, store):
        cap = _StubCapability(cap_id="ttl-test-cap")
        cap._maybe_send_failure_alert()
        _, expiry = store._strings.get("capability:alert:ttl-test-cap", (None, None))
        assert expiry is not None, "capability:alert key must have a TTL"


class TestRecoveryAlert:

    def test_broadcast_payload_has_recovered_true(self, store):
        cap = _StubCapability(cap_id="rec-cap", cap_name="Recovered Cap")
        with patch("capabilities.base.WebSocketBroker") as mock_cls:
            mock_broker = MagicMock()
            mock_cls.return_value = mock_broker
            cap._send_recovery_alert()
            mock_broker.broadcast.assert_called_once()
            payload = mock_broker.broadcast.call_args[0][0]
            assert payload["type"] == "capability_alert"
            assert payload["cap_id"] == "rec-cap"
            assert payload["recovered"] is True

    def test_deletes_capability_alert_key(self, store):
        cap = _StubCapability(cap_id="delete-me")
        store.setex("capability:alert:delete-me", 1800, '{"type":"capability_alert"}')
        cap._send_recovery_alert()
        assert store.get("capability:alert:delete-me") is None


class TestCircuitBreakerFlow:

    def _make_failing_cap(self, cap_id="circuit-cap"):
        cap = _StubCapability(cap_id=cap_id)
        cap._persist_health = lambda: None
        return cap

    def test_no_alert_before_threshold(self, store):
        cap = self._make_failing_cap()
        with patch("capabilities.base.WebSocketBroker") as mock_cls:
            mock_broker = MagicMock()
            mock_cls.return_value = mock_broker
            for _ in range(AbstractCapability.MAX_CONSECUTIVE_FAILURES - 1):
                cap._do_monitor = MagicMock(side_effect=RuntimeError("boom"))
                cap.run_monitor()
            mock_broker.broadcast.assert_not_called()

    def test_alert_fires_on_threshold(self, store):
        cap = self._make_failing_cap()
        with patch("capabilities.base.WebSocketBroker") as mock_cls:
            mock_broker = MagicMock()
            mock_cls.return_value = mock_broker
            cap._do_monitor = MagicMock(side_effect=RuntimeError("boom"))
            for _ in range(AbstractCapability.MAX_CONSECUTIVE_FAILURES):
                cap.run_monitor()
            assert mock_broker.broadcast.call_count == 1
            payload = mock_broker.broadcast.call_args[0][0]
            assert payload["recovered"] is False

    def test_recovery_broadcasts_and_clears_key(self, store):
        cap = self._make_failing_cap()
        with patch("capabilities.base.WebSocketBroker") as mock_cls:
            mock_broker = MagicMock()
            mock_cls.return_value = mock_broker
            cap._do_monitor = MagicMock(side_effect=RuntimeError("boom"))
            for _ in range(AbstractCapability.MAX_CONSECUTIVE_FAILURES):
                cap.run_monitor()
            mock_broker.broadcast.reset_mock()
            cap._next_retry_at = None
            cap._do_monitor = MagicMock()
            cap.run_monitor()
            assert mock_broker.broadcast.call_count == 1
            payload = mock_broker.broadcast.call_args[0][0]
            assert payload["recovered"] is True
        assert store.get(f"capability:alert:{cap.get_id()}") is None


class TestMultipleCapabilities:

    def test_each_cap_gets_its_own_store_key(self, store):
        cap_a = _StubCapability(cap_id="cap-a", cap_name="Cap A", last_error="err-a")
        cap_b = _StubCapability(cap_id="cap-b", cap_name="Cap B", last_error="err-b")
        cap_a._maybe_send_failure_alert()
        cap_b._maybe_send_failure_alert()
        raw_a = store.get("capability:alert:cap-a")
        raw_b = store.get("capability:alert:cap-b")
        assert raw_a is not None
        assert raw_b is not None
        assert json.loads(raw_a)["cap_id"] == "cap-a"
        assert json.loads(raw_b)["cap_id"] == "cap-b"

    def test_recovering_one_cap_does_not_remove_other_key(self, store):
        cap_a = _StubCapability(cap_id="cap-a")
        cap_b = _StubCapability(cap_id="cap-b")
        cap_a._maybe_send_failure_alert()
        cap_b._maybe_send_failure_alert()
        cap_a._send_recovery_alert()
        assert store.get("capability:alert:cap-a") is None
        assert store.get("capability:alert:cap-b") is not None
