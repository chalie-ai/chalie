"""Tests for capability failure alerting (base.py)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from capabilities.base import AbstractCapability

_SIG = "capabilities.signal_bridge.emit_capability_signal"
_MEM = "services.memory_client.MemoryClientService.create_connection"


def _make_cap():
    """Build a mock with the attributes _maybe_send_failure_alert reads."""
    m = MagicMock()
    m._failure_alerted = False
    m._last_error = "auth expired"
    m.get_id.return_value = "test"
    m.get_manifest.return_value = {"name": "Test"}
    m.MAX_CONSECUTIVE_FAILURES = 5
    return m


@pytest.mark.unit
class TestFailureAlert:

    def test_fires_once_and_dedup(self):
        cap = _make_cap()
        ms = MagicMock()
        with patch(_SIG) as sig, patch(_MEM, return_value=ms):
            AbstractCapability._maybe_send_failure_alert(cap)
            assert cap._failure_alerted is True
            sig.assert_called_once()
            assert sig.call_args[0][1] == "capability_failure"
            ms.rpush.assert_called_once()
            AbstractCapability._maybe_send_failure_alert(cap)
            assert sig.call_count == 1

    def test_prompt_content(self):
        cap = _make_cap()
        ms = MagicMock()
        with patch(_SIG), patch(_MEM, return_value=ms):
            AbstractCapability._maybe_send_failure_alert(cap)
        payload = json.loads(ms.rpush.call_args[0][1])
        assert "Test" in payload["prompt"]
        assert "auth expired" in payload["prompt"]
        assert payload["metadata"]["source"] == (
            "capability_health:test"
        )

    def test_reset_after_recovery(self):
        cap = _make_cap()
        ms = MagicMock()
        with patch(_SIG), patch(_MEM, return_value=ms):
            AbstractCapability._maybe_send_failure_alert(cap)
            cap._failure_alerted = False
            AbstractCapability._maybe_send_failure_alert(cap)
        assert ms.rpush.call_count == 2
