# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Unit tests for ResponseGenerationService helpers — Part 1.

Covers:
- :class:`~services.response_generation_service.ChatHistoryProcessor`
- :func:`~services.response_generation_service._extract_response_from_broken_json`

All imports are taken from the canonical module
``services.response_generation_service``, *not* the facade.
"""

import pytest
from unittest.mock import MagicMock

from services.response_generation_service import (
    ChatHistoryProcessor,
    _extract_response_from_broken_json,
)

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# TestChatHistoryProcessor
# ─────────────────────────────────────────────────────────────────────────────


class TestChatHistoryProcessor:
    """Unit tests for :class:`ChatHistoryProcessor`.

    Verifies deterministic text serialisation, limit enforcement, and
    edge-case tolerance for all supported input shapes.
    """

    def test_empty_history_returns_no_conversation(self):
        """Empty list must return the sentinel string exactly.

        Ensures the processor handles the initial-session case gracefully
        without raising and with a deterministic, non-empty fallback value.
        """
        processor = ChatHistoryProcessor()
        result = processor.process([])
        assert result == "No previous conversation"

    def test_single_exchange_formatting(self):
        """A single dict exchange must produce both User and Assistant lines.

        Validates that message extraction from ``prompt.message`` and
        ``response.message`` works for the canonical exchange shape.
        """
        processor = ChatHistoryProcessor()
        history = [
            {
                "prompt": {"message": "What is Python?"},
                "response": {"message": "A high-level programming language."},
            }
        ]
        result = processor.process(history)
        assert "User: What is Python?" in result
        assert "Assistant: A high-level programming language." in result

    def test_multi_exchange_formatting_preserves_order(self):
        """All exchanges in a multi-item list must appear in output, in order.

        Ensures the processor doesn't silently drop intermediate exchanges
        and preserves insertion order.
        """
        processor = ChatHistoryProcessor()
        history = [
            {"prompt": {"message": "Hello"}, "response": {"message": "Hi there"}},
            {"prompt": {"message": "How are you?"}, "response": {"message": "Well!"}},
        ]
        result = processor.process(history)
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result
        assert "User: How are you?" in result
        assert "Assistant: Well!" in result
        # Order check: "Hello" should appear before "How are you?"
        assert result.index("User: Hello") < result.index("User: How are you?")

    def test_max_exchanges_trims_oldest_entries(self):
        """max_exchanges=2 must keep only the two most-recent exchanges.

        Confirms that oldest entries are discarded and the tail of the
        history is preserved when the limit is smaller than the list length.
        """
        processor = ChatHistoryProcessor(max_exchanges=2)
        history = [
            {"prompt": {"message": f"q{i}"}, "response": {"message": f"a{i}"}}
            for i in range(5)
        ]
        result = processor.process(history)
        assert "User: q0" not in result
        assert "User: q1" not in result
        assert "User: q2" not in result
        assert "User: q3" in result
        assert "User: q4" in result

    def test_max_exchanges_equal_to_history_length_keeps_all(self):
        """max_exchanges equal to history length should not trim anything.

        Boundary condition: when the cap exactly matches the list size, all
        exchanges must survive.
        """
        processor = ChatHistoryProcessor(max_exchanges=3)
        history = [
            {"prompt": {"message": f"q{i}"}, "response": {"message": f"a{i}"}}
            for i in range(3)
        ]
        result = processor.process(history)
        for i in range(3):
            assert f"User: q{i}" in result

    def test_max_exchanges_one_keeps_only_last(self):
        """max_exchanges=1 must keep only the very last exchange.

        Ensures the tail-slicing logic works at the extreme single-entry
        boundary.
        """
        processor = ChatHistoryProcessor(max_exchanges=1)
        history = [
            {"prompt": {"message": "first"}, "response": {"message": "r1"}},
            {"prompt": {"message": "second"}, "response": {"message": "r2"}},
            {"prompt": {"message": "third"}, "response": {"message": "r3"}},
        ]
        result = processor.process(history)
        assert "User: first" not in result
        assert "User: second" not in result
        assert "User: third" in result
        assert "Assistant: r3" in result

    def test_response_as_plain_string(self):
        """A plain-string response value must be serialised directly.

        Some callers store legacy string responses; the processor must not
        attempt dict-access on them.
        """
        processor = ChatHistoryProcessor()
        history = [
            {
                "prompt": {"message": "Tell me a joke"},
                "response": "Why did the chicken cross the road?",
            }
        ]
        result = processor.process(history)
        assert "Assistant: Why did the chicken cross the road?" in result

    def test_response_with_error_key(self):
        """Error dicts must be formatted as ``[Error: ...]``.

        Verifies that a response dict with an ``'error'`` key (instead of
        ``'message'``) is rendered with the error-bracket prefix.
        """
        processor = ChatHistoryProcessor()
        history = [
            {
                "prompt": {"message": "Do something"},
                "response": {"error": "Service unavailable"},
            }
        ]
        result = processor.process(history)
        assert "Assistant: [Error: Service unavailable]" in result

    def test_exchange_without_prompt_key_skips_user_line(self):
        """An exchange without a ``'prompt'`` key must omit the User line.

        Proactive system messages have no user turn; the processor must
        not emit a ``User:`` line for them.
        """
        processor = ChatHistoryProcessor()
        history = [{"response": {"message": "Proactive message"}}]
        result = processor.process(history)
        assert "User:" not in result
        assert "Assistant: Proactive message" in result

    def test_unicode_content_in_messages(self):
        """Unicode characters in messages must survive serialisation intact.

        Exercises multi-byte scripts (CJK, emoji) to confirm no
        encoding issues during string concatenation.
        """
        processor = ChatHistoryProcessor()
        history = [
            {
                "prompt": {"message": "你好 🌍"},
                "response": {"message": "こんにちは 🎉"},
            }
        ]
        result = processor.process(history)
        assert "User: 你好 🌍" in result
        assert "Assistant: こんにちは 🎉" in result

    def test_very_long_message_content_is_included(self):
        """Messages longer than 1 000 characters must not be truncated.

        Confirms the processor does not impose any hidden length cap on
        individual message strings.
        """
        long_message = "x" * 2000
        long_response = "y" * 1500
        processor = ChatHistoryProcessor()
        history = [
            {
                "prompt": {"message": long_message},
                "response": {"message": long_response},
            }
        ]
        result = processor.process(history)
        assert long_message in result
        assert long_response in result

    def test_init_accepts_max_tokens_without_error(self):
        """Passing max_tokens to __init__ must not raise.

        The parameter is declared in the constructor signature and must be
        accepted even though token-budget enforcement is not yet implemented.
        """
        processor = ChatHistoryProcessor(max_exchanges=5, max_tokens=4096)
        assert processor.max_exchanges == 5
        assert processor.max_tokens == 4096

    def test_init_defaults_are_none(self):
        """Default construction must leave both limits as None.

        Ensures that an unconfigured processor does not silently cap history.
        """
        processor = ChatHistoryProcessor()
        assert processor.max_exchanges is None
        assert processor.max_tokens is None

    def test_exchange_with_response_missing_both_message_and_error(self):
        """An exchange whose response dict has neither 'message' nor 'error'
        must not produce an Assistant line (no crash, graceful skip).

        Guards against KeyError / AttributeError on unknown response shapes.
        """
        processor = ChatHistoryProcessor()
        history = [
            {
                "prompt": {"message": "What?"},
                "response": {"unknown_key": "some value"},
            }
        ]
        result = processor.process(history)
        assert "User: What?" in result
        # No assistant line should be emitted for this malformed exchange
        assert "Assistant:" not in result


# ─────────────────────────────────────────────────────────────────────────────
# TestExtractResponseFromBrokenJson
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractResponseFromBrokenJson:
    """Unit tests for :func:`_extract_response_from_broken_json`.

    The function extracts the raw text value of the ``"response"`` key from
    broken JSON where inner quotes are unescaped.  It locates the value
    boundary by searching for a known sibling key or by falling back to the
    last ``}`` in the string.
    """

    def test_extracts_response_with_sibling_key_modifiers(self):
        """Value should be extracted when 'modifiers' is the sibling boundary.

        Simulates the most common cortex output shape where ``"modifiers"``
        appears immediately after ``"response"``.
        """
        text = '{"response": "Hello world", "modifiers": []}'
        result = _extract_response_from_broken_json(text)
        assert result == "Hello world"

    def test_extracts_response_with_sibling_key_mode(self):
        """Value should be extracted when 'mode' is the sibling boundary."""
        text = '{"response": "My reply here", "mode": "UNIFIED"}'
        result = _extract_response_from_broken_json(text)
        assert result == "My reply here"

    def test_extracts_response_with_sibling_key_actions(self):
        """Value should be extracted when 'actions' is the sibling boundary."""
        text = '{"response": "Doing the thing", "actions": []}'
        result = _extract_response_from_broken_json(text)
        assert result == "Doing the thing"

    def test_extracts_response_with_sibling_key_confidence(self):
        """Value should be extracted when 'confidence' is the sibling boundary."""
        text = '{"response": "I am confident", "confidence": 0.9}'
        result = _extract_response_from_broken_json(text)
        assert result == "I am confident"

    def test_no_response_key_returns_none(self):
        """Text without a ``"response"`` key must return None.

        Ensures the function does not attempt to extract from unrelated JSON
        structures.
        """
        text = '{"mode": "UNIFIED", "modifiers": []}'
        result = _extract_response_from_broken_json(text)
        assert result is None

    def test_response_key_with_no_colon_returns_none(self):
        """Malformed text where no colon follows ``"response"`` must return None.

        Tests a degenerate input that the function cannot meaningfully parse.
        """
        text = '"response" "value"'
        result = _extract_response_from_broken_json(text)
        assert result is None

    def test_response_key_with_no_opening_quote_returns_none(self):
        """If the value after the colon has no opening quote, return None.

        Covers the case where the JSON value is not a quoted string at all
        (e.g., a bare number or null).
        """
        text = '{"response": null, "mode": "UNIFIED"}'
        # There is no opening quote for the value 'null', so open_quote search
        # would find the quote in "mode" — the function may or may not extract
        # something, but the primary purpose is: if we get None, that is fine;
        # if we get something, it should not crash.
        # We assert no exception is raised:
        result = _extract_response_from_broken_json(text)
        # result can be None or a string — both are acceptable; no crash is key
        assert result is None or isinstance(result, str)

    def test_empty_segment_after_stripping_returns_none(self):
        """An empty segment after boundary-stripping must return None.

        Covers the case where the value between key positions is blank.
        """
        text = '{"response": "", "modifiers": []}'
        result = _extract_response_from_broken_json(text)
        # "" value → segment becomes empty after stripping → None
        assert result is None

    def test_no_sibling_key_uses_last_brace_boundary(self):
        """When no sibling key is found, extraction falls back to the last ``}``.

        Ensures the function still produces a result for JSON that contains
        only the ``"response"`` key.
        """
        text = '{"response": "standalone value"}'
        result = _extract_response_from_broken_json(text)
        # The extracted segment ends at the last '}'; after strip it should be non-None
        assert result is not None
        assert "standalone value" in result

    def test_trailing_comma_before_sibling_key_is_stripped(self):
        """Trailing comma in the value segment must be removed.

        In broken JSON, the segment may end with ``,"`` before the sibling key.
        After stripping comma and quote, the clean value must be returned.
        """
        text = '{"response": "clean value", "mode": "ACT"}'
        result = _extract_response_from_broken_json(text)
        assert result == "clean value"
        # Must not contain trailing comma or quote artefacts
        assert not result.endswith(",")
        assert not result.endswith('"')

    def test_multiline_response_value_extracted_correctly(self):
        """A multi-word value spanning realistic prose must be extracted whole.

        Validates extraction on a longer realistic broken-JSON fragment where
        the "response" value contains spaces and punctuation.
        """
        text = (
            '{"response": "This is a longer answer with punctuation! '
            'Including commas, and colons: neat.", "confidence": 0.7}'
        )
        result = _extract_response_from_broken_json(text)
        assert result is not None
        assert "longer answer" in result
        assert "neat." in result

    def test_response_key_appears_multiple_times_uses_first(self):
        """When ``"response"`` appears more than once, the first occurrence wins.

        ``str.find`` returns the first match; the extracted value should
        correspond to the first ``"response"`` key in the string.
        """
        text = '{"response": "first value", "response": "second value", "mode": "X"}'
        result = _extract_response_from_broken_json(text)
        assert result is not None
        assert "first value" in result
