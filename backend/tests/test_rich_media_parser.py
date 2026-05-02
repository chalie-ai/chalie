"""Unit tests for services.rich_media_parser.

parse() is a pure function with no collaborators — no DB, no network, no LLM.
Covers: single card, multi-card, orphan tag, unclosed span, mismatched ordinal,
prose-only (no spans), mixed prose+spans, double-quoted id, empty input.
"""

import pytest

from services.rich_media_parser import parse, _find_payload, _extract_data

pytestmark = pytest.mark.unit


def _tc(tool_name: str, result: str) -> dict:
    """Minimal tool_call row dict."""
    return {"tool_name": tool_name, "params": "{}", "result": result, "ephemeral": 1}


# ── _extract_data ─────────────────────────────────────────────────────────────

class TestExtractData:
    def test_parses_json_head(self):
        result = '{"temp": 12}\n\nInstruction trailer'
        assert _extract_data(result) == {"temp": 12}

    def test_returns_raw_head_on_invalid_json(self):
        result = "not json\n\ntrailer"
        assert _extract_data(result) == "not json"

    def test_no_blank_line_returns_full_string_parsed(self):
        result = '{"temp": 5}'
        assert _extract_data(result) == {"temp": 5}

    def test_empty_string_returns_empty_string(self):
        assert _extract_data("") == ""


# ── _find_payload ─────────────────────────────────────────────────────────────

class TestFindPayload:
    def test_matches_single_quote_syntax(self):
        tc = _tc("weather", '{"loc":"London"}\n\n<span id=\'weather_1\'>')
        result = _find_payload("weather_1", [tc])
        assert result == {"loc": "London"}

    def test_matches_double_quote_syntax(self):
        tc = _tc("weather", '{"loc":"Paris"}\n\n<span id="weather_1">')
        result = _find_payload("weather_1", [tc])
        assert result == {"loc": "Paris"}

    def test_returns_none_for_no_match(self):
        tc = _tc("weather", '{"loc":"Rome"}\n\n<span id=\'weather_1\'>')
        assert _find_payload("weather_2", [tc]) is None

    def test_returns_none_for_empty_list(self):
        assert _find_payload("weather_1", []) is None

    def test_first_match_wins(self):
        tc1 = _tc("weather", '{"loc":"A"}\n\n<span id=\'weather_1\'>')
        tc2 = _tc("weather", '{"loc":"B"}\n\n<span id=\'weather_1\'>')
        result = _find_payload("weather_1", [tc1, tc2])
        assert result == {"loc": "A"}


# ── parse() — core cases ──────────────────────────────────────────────────────

class TestParseNoSpans:
    def test_prose_only_returns_single_text_segment(self):
        segs = parse("Hello there, how can I help?", [])
        assert segs == [{"type": "text", "content": "Hello there, how can I help?"}]

    def test_empty_content_returns_empty_list(self):
        assert parse("", []) == []

    def test_whitespace_only_returns_empty_list(self):
        assert parse("   \n  ", []) == []


class TestParseSingleCard:
    def test_single_span_paired_with_tool_call(self):
        tool_result = '{"location":"London","temperature_c":12}\n\n<span id=\'weather_1\'>'
        tc = _tc("weather", tool_result)
        content = "<span id='weather_1'>It is partly cloudy in London.</span>"
        segs = parse(content, [tc])
        assert len(segs) == 1
        seg = segs[0]
        assert seg["type"] == "rich"
        assert seg["tag"] == "weather_1"
        assert seg["synthesis"] == "It is partly cloudy in London."
        assert seg["payload"] == {"location": "London", "temperature_c": 12}

    def test_double_quoted_id_parsed_correctly(self):
        tool_result = '{"location":"Paris"}\n\n<span id="weather_1">'
        tc = _tc("weather", tool_result)
        content = '<span id="weather_1">Cloudy in Paris.</span>'
        segs = parse(content, [tc])
        assert len(segs) == 1
        assert segs[0]["type"] == "rich"
        assert segs[0]["tag"] == "weather_1"

    def test_prose_before_span(self):
        tool_result = '{"location":"Tokyo"}\n\n<span id=\'weather_1\'>'
        tc = _tc("weather", tool_result)
        content = "Here is the weather. <span id='weather_1'>Clear skies.</span>"
        segs = parse(content, [tc])
        assert len(segs) == 2
        assert segs[0] == {"type": "text", "content": "Here is the weather."}
        assert segs[1]["type"] == "rich"
        assert segs[1]["tag"] == "weather_1"

    def test_prose_after_span(self):
        tool_result = '{"location":"Berlin"}\n\n<span id=\'weather_1\'>'
        tc = _tc("weather", tool_result)
        content = "<span id='weather_1'>Overcast.</span> Stay warm today!"
        segs = parse(content, [tc])
        assert len(segs) == 2
        assert segs[0]["type"] == "rich"
        assert segs[1] == {"type": "text", "content": "Stay warm today!"}

    def test_prose_before_and_after_span(self):
        tool_result = '{"location":"NYC"}\n\n<span id=\'weather_1\'>'
        tc = _tc("weather", tool_result)
        content = "Here you go. <span id='weather_1'>Sunny.</span> Have a great day!"
        segs = parse(content, [tc])
        assert len(segs) == 3
        assert segs[0]["type"] == "text"
        assert segs[1]["type"] == "rich"
        assert segs[2]["type"] == "text"


class TestParseMultiCard:
    def test_two_cards_in_sequence(self):
        tc1 = _tc("weather", '{"location":"London","temperature_c":12}\n\n<span id=\'weather_1\'>')
        tc2 = _tc("weather", '{"location":"Tokyo","temperature_c":22}\n\n<span id=\'weather_2\'>')
        content = (
            "Here is London: <span id='weather_1'>Cloudy.</span> "
            "And here is Tokyo: <span id='weather_2'>Sunny.</span>"
        )
        segs = parse(content, [tc1, tc2])
        assert len(segs) == 4
        assert segs[0] == {"type": "text", "content": "Here is London:"}
        assert segs[1]["type"] == "rich"
        assert segs[1]["tag"] == "weather_1"
        assert segs[1]["payload"] == {"location": "London", "temperature_c": 12}
        assert segs[2] == {"type": "text", "content": "And here is Tokyo:"}
        assert segs[3]["type"] == "rich"
        assert segs[3]["tag"] == "weather_2"
        assert segs[3]["payload"] == {"location": "Tokyo", "temperature_c": 22}

    def test_ordinal_pairs_correctly_across_tool_calls(self):
        # weather_2 in tool_calls before weather_1 — tag name drives pairing, not order
        tc_ord2 = _tc("weather", '{"loc":"B"}\n\n<span id=\'weather_2\'>')
        tc_ord1 = _tc("weather", '{"loc":"A"}\n\n<span id=\'weather_1\'>')
        content = "<span id='weather_1'>A.</span> Then <span id='weather_2'>B.</span>"
        segs = parse(content, [tc_ord2, tc_ord1])
        assert segs[0]["payload"] == {"loc": "A"}
        assert segs[2]["payload"] == {"loc": "B"}


class TestParseOrphanTag:
    def test_orphan_tag_emits_text_segment_with_synthesis(self, caplog):
        import logging
        content = "<span id='weather_1'>Some synthesis here.</span>"
        with caplog.at_level(logging.WARNING, logger="services.rich_media_parser"):
            segs = parse(content, [])
        assert any("orphan" in r.message for r in caplog.records)
        assert len(segs) == 1
        assert segs[0] == {"type": "text", "content": "Some synthesis here."}

    def test_mismatched_ordinal_emits_text_segment(self, caplog):
        import logging
        # Only weather_1 in tool_calls but weather_3 in content
        tc = _tc("weather", '{"loc":"A"}\n\n<span id=\'weather_1\'>')
        content = "<span id='weather_3'>Wrong ordinal.</span>"
        with caplog.at_level(logging.WARNING, logger="services.rich_media_parser"):
            segs = parse(content, [tc])
        assert any("orphan" in r.message for r in caplog.records)
        assert segs[0]["type"] == "text"
        assert segs[0]["content"] == "Wrong ordinal."

    def test_orphan_tag_empty_synthesis_produces_no_segment(self, caplog):
        import logging
        content = "<span id='weather_1'></span>"
        with caplog.at_level(logging.WARNING, logger="services.rich_media_parser"):
            segs = parse(content, [])
        # Empty synthesis → no text segment emitted
        assert not any(s["type"] == "rich" for s in segs)


class TestParseUnclosedSpan:
    def test_unclosed_span_passes_through_as_text(self):
        # Non-greedy regex won't match if </span> is absent
        tc = _tc("weather", '{"loc":"X"}\n\n<span id=\'weather_1\'>')
        content = "Hello <span id='weather_1'>No close"
        segs = parse(content, [tc])
        # Entire content becomes a text segment since the regex doesn't match
        assert len(segs) == 1
        assert segs[0]["type"] == "text"
        assert "Hello" in segs[0]["content"]


class TestParseMixedProseAndSpans:
    def test_rich_segment_synthesis_preserved(self):
        tool_result = '{"temperature_c":5}\n\n<span id=\'weather_1\'>'
        tc = _tc("weather", tool_result)
        content = "Intro. <span id='weather_1'>It is very cold today, dress warmly.</span> Outro."
        segs = parse(content, [tc])
        rich_seg = next(s for s in segs if s["type"] == "rich")
        assert rich_seg["synthesis"] == "It is very cold today, dress warmly."

    def test_multiline_synthesis_preserved(self):
        tool_result = '{"temperature_c":20}\n\n<span id=\'weather_1\'>'
        tc = _tc("weather", tool_result)
        content = "<span id='weather_1'>Line one.\nLine two.</span>"
        segs = parse(content, [tc])
        assert segs[0]["synthesis"] == "Line one.\nLine two."
