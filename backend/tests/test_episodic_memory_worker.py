"""Tests for episodic_memory_worker — JSON extraction and safe loading."""

import logging
import pytest

from workers.episodic_memory_worker import _extract_json, _safe_json_load


pytestmark = pytest.mark.unit


# ── _extract_json ─────────────────────────────────────────────


class TestExtractJson:

    def test_strips_json_fence_when_fenced_with_json_tag(self):
        """```json ... ``` fence is removed, inner content returned."""
        raw = '```json\n{"key": "value"}\n```'

        result = _extract_json(raw)

        assert result == '{"key": "value"}'

    def test_strips_generic_fence_when_fenced_without_json_tag(self):
        """``` ... ``` fence (no language tag) is removed, inner content returned."""
        raw = '```\n{"items": [1, 2, 3]}\n```'

        result = _extract_json(raw)

        assert result == '{"items": [1, 2, 3]}'

    def test_returns_stripped_text_when_no_fence_present(self):
        """Plain text with no fence markers is returned as-is after stripping whitespace."""
        raw = '  {"plain": true}  '

        result = _extract_json(raw)

        assert result == '{"plain": true}'


# ── _safe_json_load ───────────────────────────────────────────


class TestSafeJsonLoad:

    def test_parses_valid_json_when_plain_string(self):
        """Valid JSON string without fences is parsed into a dict."""
        raw = '{"status": "ok", "count": 42}'

        result = _safe_json_load(raw)

        assert result == {"status": "ok", "count": 42}

    def test_parses_valid_json_when_fenced(self):
        """Valid JSON wrapped in ```json fences is stripped and parsed."""
        raw = '```json\n{"intent": "greeting", "salience": 0.7}\n```'

        result = _safe_json_load(raw)

        assert result == {"intent": "greeting", "salience": 0.7}

    def test_returns_none_when_json_invalid(self, caplog):
        """Malformed JSON returns None without raising an exception."""
        raw = '{not valid json at all'

        with caplog.at_level(logging.ERROR):
            result = _safe_json_load(raw)

        assert result is None
        assert any("[EPISODIC] Failed to parse JSON" in msg for msg in caplog.messages)


# ── Salience Normalization (inline logic) ─────────────────────


class TestSalienceNormalization:
    """Tests for the inline salience formula: max(1, min(10, round(salience_float * 10)))"""

    @staticmethod
    def _normalize(salience_float: float) -> int:
        return max(1, min(10, round(salience_float * 10)))

    def test_midrange_value_when_salience_half(self):
        """0.5 maps to 5."""
        assert self._normalize(0.5) == 5

    def test_clamps_to_minimum_when_salience_very_low(self):
        """0.05 rounds to 1 (0.5 rounds to 0, but clamped to 1)."""
        assert self._normalize(0.05) == 1

    def test_clamps_to_maximum_when_salience_at_ceiling(self):
        """1.0 maps to 10."""
        assert self._normalize(1.0) == 10

    def test_rounds_correctly_when_salience_fractional(self):
        """0.74 * 10 = 7.4, rounds to 7."""
        assert self._normalize(0.74) == 7


# ── Backoff Calculation (inline logic) ────────────────────────


class TestBackoffCalculation:
    """Tests for the inline backoff formula: min(300, 2 ** retry_count)"""

    @staticmethod
    def _backoff(retry_count: int) -> int:
        return min(300, 2 ** retry_count)

    def test_early_retries_use_exponential_delay(self):
        """retry_count=0 yields 1s, retry_count=4 yields 16s."""
        assert self._backoff(0) == 1
        assert self._backoff(4) == 16

    def test_caps_at_300_when_exponential_exceeds_limit(self):
        """retry_count=9 produces 2^9=512, capped to 300."""
        assert self._backoff(9) == 300
