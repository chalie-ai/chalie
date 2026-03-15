"""Tests for digest_worker — calculate_context_warmth, NLP signal patterns, ignore branch triage."""

import json
import pytest
from unittest.mock import MagicMock, patch

from workers.digest_worker import (
    calculate_context_warmth,
    _resolve_image_contexts,
)
from services.mode_router_service import (
    GREETING_PATTERNS,
    INTERROGATIVE_WORDS,
    IMPLICIT_REFERENCE,
    POSITIVE_FEEDBACK,
    NEGATIVE_FEEDBACK,
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


# ── NLP signal patterns ──────────────────────────────────────────────

class TestNlpSignalPatterns:

    def test_greeting_match_on_hey(self):
        assert GREETING_PATTERNS.match("hey there") is not None

    def test_greeting_match_on_good_morning(self):
        assert GREETING_PATTERNS.match("good morning") is not None

    def test_greeting_no_match_on_normal_text(self):
        assert GREETING_PATTERNS.match("the weather is nice") is None

    def test_interrogative_match_on_what(self):
        assert INTERROGATIVE_WORDS.search("what is this") is not None

    def test_interrogative_no_match_on_plain_sentence(self):
        assert INTERROGATIVE_WORDS.search("the cat sat") is None

    def test_implicit_reference_match(self):
        assert IMPLICIT_REFERENCE.search("you remember that?") is not None

    def test_implicit_reference_no_match(self):
        assert IMPLICIT_REFERENCE.search("the sky is blue") is None

    def test_question_mark_detection(self):
        assert '?' in "What time is it?"
        assert '?' not in "Tell me the time"

    def test_token_count_via_split(self):
        tokens = "hello world foo".split()
        assert len(tokens) == 3

    def test_positive_feedback_match(self):
        assert POSITIVE_FEEDBACK.search("thanks a lot") is not None

    def test_negative_feedback_match(self):
        assert NEGATIVE_FEEDBACK.search("that's not what I meant") is not None

    def test_information_density_calculation(self):
        tokens = "the the the cat".split()
        unique = len(set(t.lower() for t in tokens))
        density = unique / max(len(tokens), 1)
        # 2 unique / 4 total = 0.5
        assert density == pytest.approx(0.5)


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
        """If the result never appears, the image is skipped (no crash, no blocking)."""
        store = self._make_store({})  # nothing in store
        with self._patch_store(store), patch('time.sleep'):  # skip actual sleeping
            result = _resolve_image_contexts(['missing_id'], timeout=0)
        assert result == []

    def test_invalid_json_is_skipped(self):
        store = self._make_store({'chat_image_result:badid': 'not-json{{'})
        with self._patch_store(store):
            result = _resolve_image_contexts(['badid'])
        assert result == []

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

    def test_partial_resolution_returns_only_found(self):
        ctx = {'description': 'Found image', 'ocr_text': ''}
        store = self._make_store({'chat_image_result:found_id': json.dumps(ctx)})
        # Use timeout=5 so found_id resolves on the first poll iteration;
        # missing_id times out and is skipped.  patch time.sleep to avoid delay.
        with self._patch_store(store), patch('time.sleep'):
            result = _resolve_image_contexts(['found_id', 'missing_id'], timeout=5)
        assert len(result) == 1
        assert result[0]['description'] == 'Found image'
