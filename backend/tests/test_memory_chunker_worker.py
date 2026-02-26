"""Tests for memory_chunker_worker pure-logic helpers — JSON extraction,
emotion signals, reward signals, and micro-preference regex patterns."""

import re
import pytest
from workers.memory_chunker_worker import (
    _extract_json,
    _compute_emotion_signals,
    _compute_reward_signals,
    _MICRO_PREF_PATTERNS,
)


pytestmark = pytest.mark.unit


# ── _extract_json ──────────────────────────────────────────────


class TestExtractJson:

    def test_strips_json_code_fence_when_present(self):
        """```json\n{...}``` fence stripped, inner JSON returned."""
        raw = '```json\n{"key": "value"}\n```'
        result = _extract_json(raw)
        assert result == '{"key": "value"}'

    def test_strips_generic_code_fence_when_no_language_tag(self):
        """``` fence without language tag stripped, inner content returned."""
        raw = '```\n{"items": [1, 2, 3]}\n```'
        result = _extract_json(raw)
        assert result == '{"items": [1, 2, 3]}'

    def test_finds_json_by_brace_scan_when_no_fence(self):
        """Leading prose stripped by scanning for first { and last }."""
        raw = 'Here is the result: {"score": 42} hope that helps'
        result = _extract_json(raw)
        assert result == '{"score": 42}'

    def test_nested_braces_preserved_when_scanning(self):
        """Nested braces survive boundary scan (first { to last })."""
        raw = 'output: {"outer": {"inner": 1}}'
        result = _extract_json(raw)
        assert result == '{"outer": {"inner": 1}}'

    def test_raw_passthrough_when_no_fence_and_no_braces(self):
        """Plain text without fences or braces returned unchanged."""
        raw = 'no json here at all'
        result = _extract_json(raw)
        assert result == raw


# ── _compute_emotion_signals ───────────────────────────────────


def _make_chunk(user_emotion=None, scope=None):
    """Build a minimal memory chunk dict with optional emotion and scope."""
    chunk = {}
    if user_emotion is not None:
        chunk['emotion'] = {'user': user_emotion}
    if scope is not None:
        chunk['scope'] = scope
    return chunk


class TestComputeEmotionSignals:

    def test_joy_above_threshold_when_score_five(self):
        """joy 5/10 (=0.5) exceeds 0.3 threshold, emits warmth and playfulness."""
        chunk = _make_chunk(user_emotion={'joy': 5})
        signals = _compute_emotion_signals(chunk)
        assert signals['warmth'] == pytest.approx(0.25)
        assert signals['playfulness'] == pytest.approx(0.15)

    def test_joy_below_threshold_when_score_two(self):
        """joy 2/10 (=0.2) below 0.3 threshold, no warmth or playfulness emitted."""
        chunk = _make_chunk(user_emotion={'joy': 2})
        signals = _compute_emotion_signals(chunk)
        assert 'warmth' not in signals
        assert 'playfulness' not in signals

    def test_surprise_above_threshold_when_score_six(self):
        """surprise 6/10 (=0.6) exceeds 0.3, emits curiosity = 0.24."""
        chunk = _make_chunk(user_emotion={'surprise': 6})
        signals = _compute_emotion_signals(chunk)
        assert signals['curiosity'] == pytest.approx(0.24)

    def test_anger_above_threshold_when_score_seven(self):
        """anger 7/10 → negative assertiveness and added warmth."""
        chunk = _make_chunk(user_emotion={'anger': 7})
        signals = _compute_emotion_signals(chunk)
        assert signals['assertiveness'] == pytest.approx(-0.21)
        assert signals['warmth'] == pytest.approx(0.14)

    def test_disgust_triggers_negative_assertiveness_when_above_threshold(self):
        """disgust also feeds the negative assertiveness pathway (same as anger)."""
        chunk = _make_chunk(user_emotion={'disgust': 7})
        signals = _compute_emotion_signals(chunk)
        assert signals['assertiveness'] == pytest.approx(-0.21)
        assert signals['warmth'] == pytest.approx(0.14)

    def test_intent_and_confidence_reinforce_assertiveness_when_both_high(self):
        """intent 7/10 + confidence 8/10 → assertiveness +0.2."""
        chunk = _make_chunk(scope={'intent': 7, 'confidence': 8})
        signals = _compute_emotion_signals(chunk)
        assert signals['assertiveness'] == pytest.approx(0.2)

    def test_emotion_scope_emits_emotional_intensity_when_above_threshold(self):
        """scope.emotion 6/10 (=0.6) > 0.4 → emotional_intensity present."""
        chunk = _make_chunk(scope={'emotion': 6})
        signals = _compute_emotion_signals(chunk)
        assert signals['emotional_intensity'] == pytest.approx(0.12)

    def test_empty_chunk_when_no_emotion_or_scope(self):
        """Chunk with no emotion and no scope returns empty signals dict."""
        signals = _compute_emotion_signals({})
        assert signals == {}

    def test_warmth_accumulates_when_anger_and_joy_combined(self):
        """anger 7/10 + joy 5/10 → warmth from both sources added together."""
        chunk = _make_chunk(user_emotion={'anger': 7, 'joy': 5})
        signals = _compute_emotion_signals(chunk)
        # joy warmth: 0.5 * 0.5 = 0.25, anger warmth: +0.7 * 0.2 = +0.14
        assert signals['warmth'] == pytest.approx(0.39)


# ── _compute_reward_signals ────────────────────────────────────


def _make_vectors(**vec_specs):
    """Build vectors dict from keyword args of (current, baseline) tuples."""
    return {
        name: {'current_activation': cur, 'baseline_weight': base}
        for name, (cur, base) in vec_specs.items()
    }


class TestComputeRewardSignals:

    def test_zero_reward_when_below_threshold(self):
        """abs(reward) < 0.1 → empty dict regardless of vectors."""
        vectors = _make_vectors(warmth=(0.7, 0.5))
        signals = _compute_reward_signals(0.0, vectors)
        assert signals == {}

    def test_small_reward_when_below_threshold(self):
        """reward 0.05 → empty dict (still below 0.1)."""
        vectors = _make_vectors(warmth=(0.7, 0.5))
        signals = _compute_reward_signals(0.05, vectors)
        assert signals == {}

    def test_positive_signal_when_reward_positive_and_deviation_positive(self):
        """Positive reward + activation above baseline → reinforcing signal."""
        vectors = _make_vectors(warmth=(0.7, 0.5))
        signals = _compute_reward_signals(0.5, vectors)
        # 0.5 * 0.3 * 1.0 = 0.15
        assert signals['warmth'] == pytest.approx(0.15)

    def test_negative_signal_when_reward_positive_and_deviation_negative(self):
        """Positive reward + activation below baseline → dampening signal."""
        vectors = _make_vectors(warmth=(0.3, 0.5))
        signals = _compute_reward_signals(0.5, vectors)
        # 0.5 * 0.3 * -1.0 = -0.15
        assert signals['warmth'] == pytest.approx(-0.15)

    def test_negative_signal_when_reward_negative_and_deviation_positive(self):
        """Negative reward + activation above baseline → reversal signal."""
        vectors = _make_vectors(warmth=(0.7, 0.5))
        signals = _compute_reward_signals(-0.5, vectors)
        # -0.5 * 0.3 * 1.0 = -0.15
        assert signals['warmth'] == pytest.approx(-0.15)

    def test_vector_skipped_when_deviation_below_threshold(self):
        """abs(deviation) < 0.05 → vector excluded from signals."""
        vectors = _make_vectors(
            warmth=(0.52, 0.5),        # deviation 0.02 < 0.05 → skipped
            curiosity=(0.8, 0.5),      # deviation 0.3 → included
        )
        signals = _compute_reward_signals(0.5, vectors)
        assert 'warmth' not in signals
        assert 'curiosity' in signals
        assert signals['curiosity'] == pytest.approx(0.15)


# ── _MICRO_PREF_PATTERNS (regex matching) ─────────────────────


class TestMicroPrefPatterns:

    def _match_any(self, text: str) -> list:
        """Return all trait_keys matched by _MICRO_PREF_PATTERNS for given text."""
        lowered = text.lower()
        return [
            trait_key
            for pattern, trait_key in _MICRO_PREF_PATTERNS
            if re.search(pattern, lowered)
        ]

    def test_bullet_points_please_matches_prefers_bullet_format(self):
        """'bullet points please' triggers prefers_bullet_format."""
        matched = self._match_any("Can you give me bullet points please?")
        assert 'prefers_bullet_format' in matched

    def test_short_version_matches_prefers_concise(self):
        """'short version' triggers prefers_concise."""
        matched = self._match_any("Give me the short version")
        assert 'prefers_concise' in matched

    def test_elaborate_more_matches_prefers_depth(self):
        """'elaborate more' triggers prefers_depth."""
        matched = self._match_any("Could you elaborate more on that?")
        assert 'prefers_depth' in matched

    def test_challenge_me_matches_enjoys_challenge(self):
        """'challenge me' triggers enjoys_challenge."""
        matched = self._match_any("I want you to challenge me on this idea")
        assert 'enjoys_challenge' in matched

    def test_no_false_positive_when_benign_sentence(self):
        """'the weather is nice' triggers no pattern matches."""
        matched = self._match_any("the weather is nice")
        assert matched == []
