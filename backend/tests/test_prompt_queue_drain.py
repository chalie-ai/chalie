"""Tests for the prompt-queue drain mechanism in capabilities.base.

The MemoryStore ``prompt-queue`` list is the delivery path for all proactive
hooks — they rpush JSON items into the list, and ``_drain_prompt_queue()``
pops them and dispatches via the PromptQueue thread mechanism.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from capabilities.base import _drain_prompt_queue


@pytest.mark.unit
class TestDrainPromptQueue:
    """Unit tests for _drain_prompt_queue()."""

    def _make_store(self, items):
        """Create a fake MemoryStore that returns items then None."""
        store = MagicMock()
        store.lpop = MagicMock(side_effect=list(items) + [None])
        return store

    def _run_drain(self, store, max_items=50):
        """Run _drain_prompt_queue with patched dependencies."""
        mock_queue = MagicMock()
        with patch("services.memory_client.MemoryClientService") as mock_mc, \
             patch("services.prompt_queue.PromptQueue") as mock_pq, \
             patch("workers.digest_worker.digest_worker"):
            mock_mc.create_connection.return_value = store
            mock_pq.return_value = mock_queue
            count = _drain_prompt_queue(max_items=max_items)
        return count, mock_queue

    def test_drains_single_item(self):
        """A single queued item is popped and dispatched to digest_worker."""
        item = json.dumps({
            "prompt": "Hello from morning brief",
            "metadata": {"type": "proactive_drift", "source": "morning_brief"},
        })
        store = self._make_store([item])
        count, queue = self._run_drain(store)

        assert count == 1
        queue.enqueue.assert_called_once_with(
            "Hello from morning brief",
            metadata={"type": "proactive_drift", "source": "morning_brief"},
        )

    def test_drains_multiple_items(self):
        """Multiple queued items are all drained and dispatched."""
        items = [
            json.dumps({"prompt": f"Item {i}", "metadata": {"source": f"hook_{i}"}})
            for i in range(3)
        ]
        store = self._make_store(items)
        count, queue = self._run_drain(store)

        assert count == 3
        assert queue.enqueue.call_count == 3

    def test_empty_queue_returns_zero(self):
        """When prompt-queue is empty, no items are dispatched."""
        store = self._make_store([])
        count, queue = self._run_drain(store)

        assert count == 0
        queue.enqueue.assert_not_called()

    def test_skips_malformed_json(self):
        """Malformed JSON items are skipped without crashing."""
        items = ["not-json", json.dumps({"prompt": "valid", "metadata": {}})]
        store = self._make_store(items)
        count, queue = self._run_drain(store)

        assert count == 1
        queue.enqueue.assert_called_once()

    def test_skips_empty_prompt(self):
        """Items with empty prompt are not dispatched."""
        item = json.dumps({"prompt": "", "metadata": {"source": "test"}})
        store = self._make_store([item])
        count, queue = self._run_drain(store)

        assert count == 0
        queue.enqueue.assert_not_called()

    def test_max_items_limit(self):
        """Drain respects max_items to prevent infinite loop."""
        items = [
            json.dumps({"prompt": f"Item {i}", "metadata": {}})
            for i in range(10)
        ]
        store = MagicMock()
        store.lpop = MagicMock(side_effect=items)

        count, queue = self._run_drain(store, max_items=5)

        assert count == 5
        assert queue.enqueue.call_count == 5

    def test_survives_store_error(self):
        """If MemoryStore raises, drain returns 0 without crashing."""
        with patch("services.memory_client.MemoryClientService") as mock_mc, \
             patch("services.prompt_queue.PromptQueue"), \
             patch("workers.digest_worker.digest_worker"):
            mock_mc.create_connection.side_effect = RuntimeError("boom")
            count = _drain_prompt_queue()

        assert count == 0
