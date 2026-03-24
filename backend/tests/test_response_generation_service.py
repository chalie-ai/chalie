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
    """Unit tests for :class:`ChatHistoryProcessor`."""

    def test_empty_history_returns_no_conversation(self):
        processor = ChatHistoryProcessor()
        result = processor.process([])
        assert result == "No previous conversation"

    def test_single_exchange_formatting(self):
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
        processor = ChatHistoryProcessor(max_exchanges=3)
        history = [
            {"prompt": {"message": f"q{i}"}, "response": {"message": f"a{i}"}}
            for i in range(3)
        ]
        result = processor.process(history)
        for i in range(3):
            assert f"User: q{i}" in result

    def test_response_as_plain_string(self):
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
        processor = ChatHistoryProcessor()
        history = [{"response": {"message": "Proactive message"}}]
        result = processor.process(history)
        assert "User:" not in result
        assert "Assistant: Proactive message" in result

    def test_exchange_with_response_missing_both_message_and_error(self):
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
    """Unit tests for :func:`_extract_response_from_broken_json`."""

    def test_extracts_response_with_sibling_key_modifiers(self):
        text = '{"response": "Hello world", "modifiers": []}'
        result = _extract_response_from_broken_json(text)
        assert result == "Hello world"

    def test_no_response_key_returns_none(self):
        text = '{"mode": "UNIFIED", "modifiers": []}'
        result = _extract_response_from_broken_json(text)
        assert result is None

    def test_response_key_with_no_colon_returns_none(self):
        text = '"response" "value"'
        result = _extract_response_from_broken_json(text)
        assert result is None

    def test_empty_segment_after_stripping_returns_none(self):
        text = '{"response": "", "modifiers": []}'
        result = _extract_response_from_broken_json(text)
        # "" value -> segment becomes empty after stripping -> None
        assert result is None

    def test_no_sibling_key_uses_last_brace_boundary(self):
        text = '{"response": "standalone value"}'
        result = _extract_response_from_broken_json(text)
        # The extracted segment ends at the last '}'; after strip it should be non-None
        assert result is not None
        assert "standalone value" in result

    def test_multiline_response_value_extracted_correctly(self):
        text = (
            '{"response": "This is a longer answer with punctuation! '
            'Including commas, and colons: neat.", "confidence": 0.7}'
        )
        result = _extract_response_from_broken_json(text)
        assert result is not None
        assert "longer answer" in result
        assert "neat." in result

    def test_response_key_appears_multiple_times_uses_first(self):
        text = '{"response": "first value", "response": "second value", "mode": "X"}'
        result = _extract_response_from_broken_json(text)
        assert result is not None
        assert "first value" in result


# ─────────────────────────────────────────────────────────────────────────────
# TestParseResponseTextLayers
# ─────────────────────────────────────────────────────────────────────────────


class TestParseResponseTextLayers:
    """Unit tests for :meth:`ResponseGenerationService._parse_response_text`."""

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

    # -- Layer 0 --------------------------------------------------------------

    def test_layer0_valid_json_parses_directly(self):
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "Hello!", "confidence": 0.8}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "Hello!"
        assert result["confidence"] == pytest.approx(0.8)
        assert result["generation_time"] == pytest.approx(0.1)

    def test_layer0_act_mode_with_valid_actions(self):
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
        svc = self._make_service()
        text = '{"mode": "IGNORE", "response": "", "confidence": 0.5}'
        result = svc._parse_response_text(text, 0.05)
        assert result["mode"] == "IGNORE"

    # -- Layer 0b -------------------------------------------------------------

    def test_layer0b_multi_object_json_uses_first_object(self):
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "first"}\n{"extra": "second"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "first"

    # -- Layer 1 --------------------------------------------------------------

    def test_layer1_invalid_escape_sequence_fixed_and_parsed(self):
        svc = self._make_service()
        # Raw string: the JSON value contains \$ which is invalid in JSON
        text = r'{"mode": "UNIFIED", "response": "cost is \$10"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert "$10" in result["response"]

    # -- Layer 1b -------------------------------------------------------------

    def test_layer1b_escape_fixed_multi_object_json_uses_first(self):
        svc = self._make_service()
        # After fixing \$ -> $, the two-object structure triggers "Extra data"
        text = '{"mode": "UNIFIED", "response": "value \\$end"}\n{"stray": 1}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert "end" in result["response"]

    # -- Layer 2 --------------------------------------------------------------

    def test_layer2_json_embedded_in_prose_is_extracted(self):
        svc = self._make_service()
        text = (
            'Sure, here is my answer: '
            '{"mode": "UNIFIED", "response": "embedded"} — done.'
        )
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "embedded"

    # -- Layer 3 --------------------------------------------------------------

    def test_layer3_literal_newline_in_string_value_fixed(self):
        svc = self._make_service()
        # Embed a real newline character (chr 10) inside the JSON string value
        text = '{"mode": "UNIFIED", "response": "line1' + chr(10) + 'line2"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert "line1" in result["response"]
        assert "line2" in result["response"]

    # -- Layer 4 --------------------------------------------------------------

    def test_layer4_broken_json_unescaped_inner_quotes_extracted(self):
        svc = self._make_service()
        # Unescaped inner quotes make this syntactically invalid JSON
        text = '{"response": "He said "hello" to her", "mode": "UNIFIED"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["response"] is not None
        assert len(result["response"]) > 0
        assert "hello" in result["response"]

    # -- Layer 5 --------------------------------------------------------------

    def test_layer5_pure_prose_wrapped_as_unified(self):
        svc = self._make_service()
        text = "I'm sorry, I don't know the answer to that question."
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == text

    def test_layer5_empty_string_raises_exception(self):
        svc = self._make_service()
        with pytest.raises(Exception, match="[Ee]mpty"):
            svc._parse_response_text("", 0.1)

    # -- Markdown fence stripping ---------------------------------------------

    def test_markdown_fence_json_stripped_before_parse(self):
        svc = self._make_service()
        text = '```json\n{"mode": "UNIFIED", "response": "fenced"}\n```'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "fenced"

    # -- Field validation: mode -----------------------------------------------

    def test_mode_invalid_string_defaults_to_unified(self):
        svc = self._make_service()
        text = '{"mode": "CUSTOM_MODE", "response": "hello"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"

    # -- Field validation: confidence -----------------------------------------

    def test_confidence_above_one_clamped_to_one(self):
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "hi", "confidence": 1.9}'
        result = svc._parse_response_text(text, 0.1)
        assert result["confidence"] == pytest.approx(1.0)

    def test_confidence_below_zero_clamped_to_zero(self):
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "hi", "confidence": -0.5}'
        result = svc._parse_response_text(text, 0.1)
        assert result["confidence"] == pytest.approx(0.0)

    def test_confidence_non_numeric_defaults_to_half(self):
        svc = self._make_service()
        text = '{"mode": "UNIFIED", "response": "hi", "confidence": "high"}'
        result = svc._parse_response_text(text, 0.1)
        assert result["confidence"] == pytest.approx(0.5)

    # -- Field validation: actions --------------------------------------------

    def test_actions_infer_act_mode_when_mode_absent(self):
        svc = self._make_service()
        text = '{"response": "", "actions": [{"type": "lookup", "key": "x"}]}'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "ACT"
        assert result["actions"] is not None

    def test_actions_without_type_key_filtered_out(self):
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
        svc = self._make_service()
        text = '{"mode": "ACT", "response": "", "actions": [{"no_type": true}]}'
        result = svc._parse_response_text(text, 0.1)
        assert result["actions"] is None

    # -- Field validation: alternative_paths ----------------------------------

    def test_alternative_paths_missing_mode_excluded(self):
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
        svc = self._make_service()
        text = (
            '{"mode": "UNIFIED", "response": "hi", '
            '"alternative_paths": [{"mode": "ACT", "expected_confidence": 2.5}]}'
        )
        result = svc._parse_response_text(text, 0.1)
        assert len(result["alternative_paths"]) == 1
        assert result["alternative_paths"][0]["expected_confidence"] == pytest.approx(1.0)

    def test_alternative_paths_missing_confidence_defaults_to_half(self):
        svc = self._make_service()
        text = (
            '{"mode": "UNIFIED", "response": "hi", '
            '"alternative_paths": [{"mode": "ACT"}]}'
        )
        result = svc._parse_response_text(text, 0.1)
        assert len(result["alternative_paths"]) == 1
        assert result["alternative_paths"][0]["expected_confidence"] == pytest.approx(0.5)

    # -- Edge cases -----------------------------------------------------------

    def test_non_dict_json_array_uses_first_dict_element(self):
        svc = self._make_service()
        text = '[{"mode": "UNIFIED", "response": "array-item"}]'
        result = svc._parse_response_text(text, 0.1)
        assert result["mode"] == "UNIFIED"
        assert result["response"] == "array-item"

    def test_response_value_non_string_coerced_to_string(self):
        svc = self._make_service()
        # response is a JSON number, not a string
        text = '{"mode": "UNIFIED", "response": 42}'
        result = svc._parse_response_text(text, 0.1)
        assert isinstance(result["response"], str)
        assert "42" in result["response"]
