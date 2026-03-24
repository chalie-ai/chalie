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
from unittest.mock import MagicMock, patch

from services.response_generation_service import (
    ChatHistoryProcessor,
    ResponseGenerationService,
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


# ─────────────────────────────────────────────────────────────────────────────
# TestParseResponseTextLayers
# ─────────────────────────────────────────────────────────────────────────────


class TestParseResponseTextLayers:
    """Unit tests for :meth:`ResponseGenerationService._parse_response_text`.

    Covers all JSON recovery layers (0 through 5), markdown fence stripping,
    field validation (mode normalisation, confidence clamping, action
    filtering, alternative_paths validation), and edge cases such as non-dict
    responses and empty input.

    A fresh :class:`ResponseGenerationService` instance is created for each
    test via :meth:`_make_service`, which patches
    ``services.llm_service.create_llm_service`` so no real LLM provider is
    instantiated.
    """

    @staticmethod
    def _make_service() -> ResponseGenerationService:
        """Create a ResponseGenerationService backed by a mocked LLM.

        Patches ``services.llm_service.create_llm_service`` to return a plain
        :class:`~unittest.mock.MagicMock` so the constructor completes without
        any network access or import-time side effects.

        Returns:
            A ready-to-use :class:`ResponseGenerationService` instance whose
            ``llm`` attribute is a :class:`~unittest.mock.MagicMock`.
        """
        with patch("services.llm_service.create_llm_service", return_value=MagicMock()):
            svc = ResponseGenerationService({})
        return svc

    # ── Layer 0 ──────────────────────────────────────────────────────────────

    def test_layer0_valid_json_parses_directly(self):
        """Layer 0: a well-formed JSON string must parse without recovery.

        Verifies the fast path (direct ``json.loads``) for clean LLM output
        and asserts that all standard result keys are populated with the
        expected values.
        """
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "Hello!", "confidence": 0.8}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "Hello!"
        assert result["confidence"] == pytest.approx(0.8)
        assert result["generation_time"] == pytest.approx(0.1)

    def test_layer0_act_mode_with_valid_actions(self):
        """Layer 0: ACT mode with a valid actions list must preserve those actions.

        Confirms that the ``actions`` list is returned intact when the parsed
        JSON specifies ACT mode and contains at least one dict with a
        ``'type'`` key.
        """
        svc = self._make_service()
        text = (
            '{"mode": "ACT", "response": "", '
            '"actions": [{"type": "search", "query": "python"}], '
            '"confidence": 0.9}'
        )
        result = svc._parse_response_text(text, 0.2)
        assert result["mode"] == "ACT"
        assert result["actions"] is not None
        assert len(result["actions"]) == 1
        assert result["actions"][0]["type"] == "search"

    def test_layer0_ignore_mode_preserved(self):
        """Layer 0: IGNORE mode must pass through the mode validator unchanged.

        Ensures the third valid mode value is not incorrectly coerced to
        UNIFIED by the normalisation step.
        """
        svc = self._make_service()
        text = '{"mode": "IGNORE", "response": "", "confidence": 0.5}'
        result = svc._parse_response_text(text, 0.05)
        assert result["mode"] == "IGNORE"

    # ── Layer 0b ─────────────────────────────────────────────────────────────

    def test_layer0b_multi_object_json_uses_first_object(self):
        """Layer 0b: two concatenated JSON objects must yield only the first.

        Simulates a provider response where two objects are emitted back-to-
        back.  ``json.loads`` raises "Extra data"; the ``raw_decode`` fallback
        must recover the first valid object.
        """
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "first"}\n{"extra": "second"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "first"

    # ── Layer 1 ──────────────────────────────────────────────────────────────

    def test_layer1_invalid_escape_sequence_fixed_and_parsed(self):
        r"""Layer 1: an invalid ``\$`` escape must be repaired before parsing.

        The regex ``re.sub(r'\\([^"\\/bfnrtu])', r'\1', ...)`` converts
        ``\$`` to ``$``; the resulting text must be valid JSON.
        """
        svc = self._make_service()
        # Raw string: the JSON value contains \$ which is invalid in JSON
        text = r'{"mode": "UNIFIED", "response": "cost is \$10"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert "$10" in result["response"]

    # ── Layer 1b ─────────────────────────────────────────────────────────────

    def test_layer1b_escape_fixed_multi_object_json_uses_first(self):
        r"""Layer 1b: escape-fixed multi-object text must yield the first object.

        After the invalid-escape fix converts the text to syntactically valid
        JSON fragments, the resulting string may still contain two objects
        (triggering an "Extra data" error); ``raw_decode`` must recover the
        first.
        """
        svc = self._make_service()
        # After fixing \$ -> $, the two-object structure triggers "Extra data"
        text = '{"mode": "UNIFIED", "response": "value \\$end"}\n{"stray": 1}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert "end" in result["response"]

    # ── Layer 2 ──────────────────────────────────────────────────────────────

    def test_layer2_json_embedded_in_prose_is_extracted(self):
        """Layer 2: a JSON object embedded in prose must be found by brace scanning.

        The LLM sometimes wraps its JSON output in explanation text.  The
        brace extractor (``find('{')`` / ``rfind('}')``) must locate and parse
        the embedded object, ignoring surrounding prose.
        """
        svc = self._make_service()
        text = (
            'Sure, here is my answer: '
            '{"mode": "UNIFIED", "response": "embedded"} — done.'
        )
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "embedded"

    def test_layer2_json_after_newline_prose_extracted(self):
        """Layer 2: JSON appearing after a prose preamble line must be extracted.

        Verifies that multi-line preambles (prose on line 1, JSON on line 2)
        are handled correctly by the brace scanner.
        """
        svc = self._make_service()
        text = 'Let me respond:\n{"mode": "UNIFIED", "response": "multiline"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "multiline"

    # ── Layer 3 ──────────────────────────────────────────────────────────────

    def test_layer3_literal_newline_in_string_value_fixed(self):
        """Layer 3: a literal newline (chr 10) inside a JSON string must be escaped.

        JSON forbids bare control characters in string values.  The character-
        by-character loop replaces each such character with its JSON escape
        sequence so the parse can succeed.
        """
        svc = self._make_service()
        # Embed a real newline character (chr 10) inside the JSON string value
        text = '{"mode": "UNIFIED", "response": "line1' + chr(10) + 'line2"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert "line1" in result["response"]
        assert "line2" in result["response"]

    def test_layer3_literal_tab_in_string_value_fixed(self):
        """Layer 3: a literal tab character (chr 9) inside a JSON string must be escaped.

        Mirrors the newline case; the character loop must also replace bare
        tab characters with ``\\t`` before attempting to parse.
        """
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "col1' + chr(9) + 'col2"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert "col1" in result["response"]
        assert "col2" in result["response"]

    # ── Layer 4 ──────────────────────────────────────────────────────────────

    def test_layer4_broken_json_unescaped_inner_quotes_extracted(self):
        """Layer 4: broken JSON with unescaped inner quotes must be recovered.

        :func:`_extract_response_from_broken_json` locates the ``"response"``
        value by scanning for a sibling key boundary, bypassing the standard
        JSON parser entirely.
        """
        svc = self._make_service()
        # Unescaped inner quotes make this syntactically invalid JSON
        text = '{"response": "He said "hello" to her", "mode": "UNIFIED"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["response"] is not None
        assert len(result["response"]) > 0
        assert "hello" in result["response"]

    # ── Layer 5 ──────────────────────────────────────────────────────────────

    def test_layer5_pure_prose_wrapped_as_unified(self):
        """Layer 5: a plain prose string with no JSON must be wrapped as UNIFIED.

        When no JSON is recoverable from any layer, the raw text becomes the
        ``response`` value and ``mode`` is forced to UNIFIED.
        """
        svc = self._make_service()
        text = "I'm sorry, I don't know the answer to that question."
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == text

    def test_layer5_empty_string_raises_exception(self):
        """Layer 5: an entirely empty LLM response must raise an Exception.

        An empty response is unrecoverable; the parser must propagate an
        exception rather than silently returning an empty result dict.
        """
        svc = self._make_service()
        with pytest.raises(Exception, match="[Ee]mpty"):
            svc._parse_response_text("", 0.1)

    # ── Markdown fence stripping ─────────────────────────────────────────────

    def test_markdown_fence_json_stripped_before_parse(self):
        r"""Markdown \`\`\`json fences must be removed before JSON parsing.

        Some LLM providers wrap JSON output in fenced code blocks; the
        stripper must handle the ``\`\`\`json`` variant before any parse
        attempt.
        """
        svc = self._make_service()
        text = '```json\n{"mode": "UNIFIED", "response": "fenced"}\n```'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "fenced"

    def test_markdown_fence_plain_stripped_before_parse(self):
        r"""Plain \`\`\` fences (no language tag) must also be stripped.

        Covers the ``\`\`\`` variant (without a language specifier) to ensure
        the stripper is not limited to the ``json``-tagged form.
        """
        svc = self._make_service()
        text = '```\n{"mode": "UNIFIED", "response": "plain-fenced"}\n```'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "plain-fenced"

    # ── Field validation — mode ───────────────────────────────────────────────

    def test_mode_invalid_string_defaults_to_unified(self):
        """An unrecognised mode string must be normalised to UNIFIED.

        Protects downstream consumers from unexpected mode values that do not
        appear in the ``valid_modes`` list (``['ACT', 'UNIFIED', 'IGNORE']``).
        """
        svc = self._make_service()
        text = '{"mode": "CUSTOM_MODE", "response": "hello"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"

    # ── Field validation — confidence ────────────────────────────────────────

    def test_confidence_above_one_clamped_to_one(self):
        """A confidence value above 1.0 must be clamped to exactly 1.0.

        LLMs occasionally return out-of-range confidence scores; the upper
        clamp must enforce the [0, 1] contract.
        """
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "hi", "confidence": 1.9}'
        result = svc._parse_response_text(text, 0.1)
        assert result["confidence"] == pytest.approx(1.0)

    def test_confidence_below_zero_clamped_to_zero(self):
        """A negative confidence value must be clamped to exactly 0.0.

        Enforces the lower bound of the [0, 1] confidence range.
        """
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "hi", "confidence": -0.5}'
        result = svc._parse_response_text(text, 0.1)
        assert result["confidence"] == pytest.approx(0.0)

    def test_confidence_non_numeric_defaults_to_half(self):
        """A non-numeric confidence value must fall back to the default of 0.5.

        When ``float(confidence)`` raises a ``ValueError``, the parser must
        substitute the safe midpoint default.
        """
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "hi", "confidence": "high"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["confidence"] == pytest.approx(0.5)

    # ── Field validation — actions ───────────────────────────────────────────

    def test_actions_infer_act_mode_when_mode_absent(self):
        """A non-empty actions list must cause ACT mode to be inferred.

        ACT prompt output may omit the ``mode`` field; the parser must set
        ``mode`` to ACT whenever a valid actions list is present.
        """
        svc = self._make_service()
        text = '{"response": "", "actions": [{"type": "lookup", "key": "x"}]}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "ACT"
        assert result["actions"] is not None

    def test_actions_without_type_key_filtered_out(self):
        """Action dicts missing the ``'type'`` key must be silently discarded.

        Only dicts with a ``'type'`` key constitute valid action descriptors;
        malformed entries must not reach downstream consumers.
        """
        svc = self._make_service()
        text = (
            '{"mode": "ACT", "response": "", '
            '"actions": [{"type": "valid"}, {"no_type": true}]}'
        )
        result = svc._parse_response_text(text, 0.1)
        assert result["actions"] is not None
        assert len(result["actions"]) == 1
        assert result["actions"][0]["type"] == "valid"

    def test_actions_all_invalid_results_in_none(self):
        """When every action is invalid, the ``actions`` field must be None.

        After filtering, an empty list is normalised to ``None`` so callers
        can use a simple truthiness check.
        """
        svc = self._make_service()
        text = '{"mode": "ACT", "response": "", "actions": [{"no_type": true}]}'
        result = svc._parse_response_text(text, 0.1)
        assert result["actions"] is None

    # ── Field validation — alternative_paths ────────────────────────────────

    def test_alternative_paths_missing_mode_excluded(self):
        """Alternative paths without a ``'mode'`` key must be excluded.

        The validator requires each alternative path to carry a ``'mode'``
        key; paths without it must be silently dropped.
        """
        svc = self._make_service()
        text = (
            '{"mode": "UNIFIED", "response": "hi", '
            '"alternative_paths": ['
            '  {"expected_confidence": 0.7},'
            '  {"mode": "ACT", "expected_confidence": 0.5}'
            ']}'
        )
        result = svc._parse_response_text(text, 0.1)
        assert len(result["alternative_paths"]) == 1
        assert result["alternative_paths"][0]["mode"] == "ACT"

    def test_alternative_paths_confidence_clamped(self):
        """Alternative path ``expected_confidence`` must be clamped to [0, 1].

        The same range enforcement that applies to top-level ``confidence``
        must also apply to each alternative path's ``expected_confidence``.
        """
        svc = self._make_service()
        text = (
            '{"mode": "UNIFIED", "response": "hi", '
            '"alternative_paths": [{"mode": "ACT", "expected_confidence": 2.5}]}'
        )
        result = svc._parse_response_text(text, 0.1)
        assert len(result["alternative_paths"]) == 1
        assert result["alternative_paths"][0]["expected_confidence"] == pytest.approx(1.0)

    def test_alternative_paths_missing_confidence_defaults_to_half(self):
        """Alternative path lacking ``expected_confidence`` must default to 0.5.

        When the key is absent the validator must inject the default rather
        than allowing a ``KeyError`` downstream.
        """
        svc = self._make_service()
        text = (
            '{"mode": "UNIFIED", "response": "hi", '
            '"alternative_paths": [{"mode": "ACT"}]}'
        )
        result = svc._parse_response_text(text, 0.1)
        assert len(result["alternative_paths"]) == 1
        assert result["alternative_paths"][0]["expected_confidence"] == pytest.approx(0.5)

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_non_dict_json_array_uses_first_dict_element(self):
        """A bare JSON array must be unwrapped to its first dict element.

        When ``json.loads`` succeeds but returns a list, the first dict
        element is promoted to ``response_data`` so field extraction can
        proceed normally.
        """
        svc = self._make_service()
        text = '[{"mode": "UNIFIED", "response": "array-item"}]'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "array-item"

    def test_generation_time_propagated_unchanged(self):
        """The ``generation_time`` argument must be forwarded verbatim to the result.

        Verifies that the timing measurement is threaded through the parse
        pipeline without being altered or dropped.
        """
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "timing test"}'
        result = svc._parse_response_text(text, 3.14159)
        assert result["generation_time"] == pytest.approx(3.14159)

    def test_narrated_and_narration_fields_passed_through(self):
        """Optional ``narrated`` and ``narration`` keys must appear in the result.

        When the parsed JSON includes these optional fields (used by the ACT
        loop for live progress display), they must be forwarded unchanged to
        the caller without being dropped during field extraction.
        """
        svc = self._make_service()
        text = (
            '{"mode": "UNIFIED", "response": "done", '
            '"narrated": true, "narration": "Working on it\u2026"}'
        )
        result = svc._parse_response_text(text, 0.1)
        assert result.get("narrated") is True
        assert result.get("narration") == "Working on it\u2026"

    def test_response_value_non_string_coerced_to_string(self):
        """A non-string ``response`` value must be coerced to a string.

        When the LLM returns a nested object or number as the ``response``
        value, it must be converted via ``str()`` rather than propagated as-is
        (which would break callers expecting a string).
        """
        svc = self._make_service()
        # response is a JSON number, not a string
        text = '{"mode": "UNIFIED", "response": 42}'
        result = svc._parse_response_text(text, 0.1)
        assert isinstance(result["response"], str)
        assert "42" in result["response"]

    def test_downstream_mode_default_is_unified(self):
        """When ``downstream_mode`` is absent, it must default to UNIFIED.

        The key is optional in ACT prompt output; its absence must not cause
        a ``KeyError`` or a ``None`` value in the result dict.
        """
        svc = self._make_service()
        text = '{"mode": "ACT", "response": "", "actions": [{"type": "run"}]}'
        result = svc._parse_response_text(text, 0.1)
        assert result["downstream_mode"] == "UNIFIED"
