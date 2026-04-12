"""Tests for digest_worker — calculate_context_warmth, image context resolution."""

import json
import pytest
from unittest.mock import MagicMock, patch

from workers.digest_worker import (
    calculate_context_warmth,
    _resolve_image_contexts,
)
pytestmark = pytest.mark.unit


# ── calculate_context_warmth ─────────────────────────────────────────

class TestCalculateContextWarmth:
    """
    warmth = (wm_score + world_score) / 2
    wm_score    = min(working_memory_len / 4, 1.0)
    world_score = 1.0 if world_state_nonempty else 0.0

    Gist score removed in Stream 1 (memory chunker killed).
    """

    def test_all_zeros_returns_zero(self):
        assert calculate_context_warmth(0, False) == 0.0

    def test_all_maxed_returns_one(self):
        result = calculate_context_warmth(8, True)
        assert result == pytest.approx(1.0)

    def test_wm_caps_at_one(self):
        # 8 turns → min(8/4, 1.0) = 1.0, world=False
        result = calculate_context_warmth(8, False)
        expected = (1.0 + 0.0) / 2
        assert result == pytest.approx(expected)

    def test_world_state_true_contributes_half(self):
        result = calculate_context_warmth(0, True)
        expected = (0.0 + 1.0) / 2
        assert result == pytest.approx(expected)

    def test_world_state_false_contributes_zero(self):
        result = calculate_context_warmth(0, False)
        assert result == 0.0

    def test_mixed_inputs(self):
        # wm=2 → 0.5, world=True → 1.0
        result = calculate_context_warmth(2, True)
        expected = (0.5 + 1.0) / 2
        assert result == pytest.approx(expected, abs=0.001)


# ── _resolve_image_contexts (WS4) ────────────────────────────────────

class TestResolveImageContexts:
    """
    _resolve_image_contexts polls MemoryStore for vision analysis results.
    Tests cover: immediate hit, in-flight wait, timeout, JSON error, multi-image.
    """

    def _make_store(self, data: dict):
        """Build a minimal MemoryStore mock with deterministic get()."""
        store = MagicMock()
        store.get.side_effect = lambda key: data.get(key)
        return store

    def _patch_store(self, store):
        """Context manager: patch MemoryClientService.create_connection to return *store*."""
        return patch(
            'services.memory_client.MemoryClientService.create_connection',
            return_value=store,
        )

    def test_returns_empty_list_for_no_ids(self):
        # No IDs — create_connection should never be called (early return)
        result = _resolve_image_contexts([])
        assert result == []

    def test_immediate_hit_returns_context(self):
        ctx = {'description': 'A cat sitting on a mat.', 'ocr_text': ''}
        store = self._make_store({'chat_image_result:abc123': json.dumps(ctx)})
        with self._patch_store(store):
            result = _resolve_image_contexts(['abc123'])
        assert len(result) == 1
        assert result[0]['description'] == 'A cat sitting on a mat.'

    def test_missing_key_times_out_gracefully(self):
        """If the result never appears, an error context is returned (not skipped)."""
        store = self._make_store({})  # nothing in store
        with self._patch_store(store), patch('time.sleep'):  # skip actual sleeping
            result = _resolve_image_contexts(['missing_id'], timeout=0)
        assert len(result) == 1
        assert 'timed out' in result[0]['error']

    def test_invalid_json_returns_error(self):
        store = self._make_store({'chat_image_result:badid': 'not-json{{'})
        with self._patch_store(store):
            result = _resolve_image_contexts(['badid'])
        assert len(result) == 1
        assert 'failed to parse' in result[0]['error']

    def test_multiple_ids_all_resolved(self):
        ctx_a = {'description': 'Image A', 'ocr_text': ''}
        ctx_b = {'description': 'Image B', 'ocr_text': 'hello'}
        store = self._make_store({
            'chat_image_result:id_a': json.dumps(ctx_a),
            'chat_image_result:id_b': json.dumps(ctx_b),
        })
        with self._patch_store(store):
            result = _resolve_image_contexts(['id_a', 'id_b'])
        assert len(result) == 2
        descs = [r['description'] for r in result]
        assert 'Image A' in descs
        assert 'Image B' in descs

    def test_partial_resolution_returns_all_with_error_for_missing(self):
        ctx = {'description': 'Found image', 'ocr_text': ''}
        store = self._make_store({'chat_image_result:found_id': json.dumps(ctx)})
        # Use timeout=5 so found_id resolves on the first poll iteration;
        # missing_id times out but still returns an error context.
        with self._patch_store(store), patch('time.sleep'):
            result = _resolve_image_contexts(['found_id', 'missing_id'], timeout=5)
        assert len(result) == 2
        assert result[0]['description'] == 'Found image'
        assert 'timed out' in result[1]['error']
