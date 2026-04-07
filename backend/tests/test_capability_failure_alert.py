"""Tests for capability failure alerting (base.py).

Rewritten to use a real MemoryStore instead of a MagicMock.
Assertions check actual queue state (llen, lrange) rather than
mock call counts.
"""

import json
import pytest

from capabilities.base import AbstractCapability

pytestmark = pytest.mark.unit


def _make_cap():
    """Return a MagicMock with the attributes _maybe_send_failure_alert reads."""
    from unittest.mock import MagicMock
    m = MagicMock()
    m._failure_alerted = False
    m._last_error = "auth expired"
    m.get_id.return_value = "test"
    m.get_manifest.return_value = {"name": "Test"}
    m.MAX_CONSECUTIVE_FAILURES = 5
    return m


class TestFailureAlert:

    def test_fires_and_pushes_to_queue(self, store):
        """First call writes one item to the prompt-queue in the real store."""
        cap = _make_cap()
        AbstractCapability._maybe_send_failure_alert(cap)

        assert cap._failure_alerted is True
        assert store.llen("prompt-queue") == 1

    def test_dedup_prevents_second_push(self, store):
        """Second call after first is a no-op — queue stays at length 1."""
        cap = _make_cap()
        AbstractCapability._maybe_send_failure_alert(cap)
        AbstractCapability._maybe_send_failure_alert(cap)

        assert store.llen("prompt-queue") == 1

    def test_prompt_content_contains_cap_name_and_error(self, store):
        """The queued payload embeds the capability name and last_error."""
        cap = _make_cap()
        AbstractCapability._maybe_send_failure_alert(cap)

        raw = store.lrange("prompt-queue", 0, -1)[0]
        payload = json.loads(raw)
        assert "Test" in payload["prompt"]
        assert "auth expired" in payload["prompt"]

    def test_metadata_source_matches_cap_id(self, store):
        """metadata.source must be capability_health:<cap_id>."""
        cap = _make_cap()
        AbstractCapability._maybe_send_failure_alert(cap)

        raw = store.lrange("prompt-queue", 0, -1)[0]
        payload = json.loads(raw)
        assert payload["metadata"]["source"] == "capability_health:test"

    def test_reset_after_recovery_allows_second_alert(self, store):
        """After resetting _failure_alerted, a second alert is pushed."""
        cap = _make_cap()
        AbstractCapability._maybe_send_failure_alert(cap)
        assert store.llen("prompt-queue") == 1

        cap._failure_alerted = False
        AbstractCapability._maybe_send_failure_alert(cap)
        assert store.llen("prompt-queue") == 2
