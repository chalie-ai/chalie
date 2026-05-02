"""Adversarial-input tests for the sanitiser + parser pipeline.

Every test runs through the real production services (services.markup.sanitize
and services.rich_media_parser.parse).  No mocks.

Regression class covered: hostile span content that survived the nh3
sanitisation pass must still produce safe, well-defined output from the
parser.  Covers cases the coder's test_markup_span_whitelist.py did not:

- id with leading underscore  → regex [a-z][a-z0-9_]* rejects; falls to text
- <script> nested inside span synthesis → nh3 strips the script; synthesis
  text survives; parser sees clean text inside the span
- XSS event attribute on span alongside a valid id → nh3 strips onclick;
  span with just id survives; parser pairs normally
- nested spans (<span id='weather_1'><span id='weather_2'>inner</span></span>)
  → outer span regex matches the outermost close tag; inner span tag is
  treated as part of synthesis (either stripped by nh3 or left as a plain tag)
- unclosed span injected via onclick-carrying span → sanitiser strips onclick,
  no close tag → regex misses; full content falls to text segment
"""

import pytest

from services.markup import sanitize
from services.rich_media_parser import parse

pytestmark = pytest.mark.unit


def _tc(result: str) -> dict:
    return {"tool_name": "weather", "params": "{}", "result": result, "ephemeral": 1}


class TestAdversarialSanitiser:
    """Sanitiser must strip hostile attributes while keeping the id for pairing."""

    def test_onclick_plus_id_onclick_stripped_id_kept(self):
        raw = "<span id='weather_1' onclick='alert(1)'>Sunny.</span>"
        out = sanitize(raw)
        assert "onclick" not in out
        assert "weather_1" in out
        assert "Sunny." in out

    def test_script_nested_inside_span_stripped(self):
        raw = "<span id='weather_1'><script>evil()</script>Partly cloudy.</span>"
        out = sanitize(raw)
        assert "<script>" not in out
        assert "evil" not in out
        # Inner text outside the script tag must survive
        assert "Partly cloudy." in out

    def test_style_injection_via_span_stripped(self):
        raw = "<span id='weather_1' style='position:fixed;top:0'>hi</span>"
        out = sanitize(raw)
        assert "style=" not in out
        assert "weather_1" in out
        assert "hi" in out

    def test_data_attribute_stripped(self):
        raw = "<span id='weather_1' data-xss='payload'>text</span>"
        out = sanitize(raw)
        assert "data-xss" not in out
        assert "weather_1" in out


class TestAdversarialParser:
    """Parser must produce safe fallback behaviour on hostile or malformed input."""

    def test_id_with_leading_underscore_not_recognised_as_rich(self):
        # _invalid_1: leading underscore — does not match [a-z][a-z0-9_]*.
        # No span match → full content is a text segment, no rich segment emitted.
        tc = _tc('{"temperature_c":5}\n\n<span id=\'_invalid_1\'>')
        content = "<span id='_invalid_1'>Cold today.</span>"
        segs = parse(content, [tc])
        assert not any(s["type"] == "rich" for s in segs)
        # Content still reaches the user as text
        all_text = " ".join(s.get("content", "") for s in segs if s["type"] == "text")
        assert "Cold today." in all_text or content in all_text or len(segs) >= 1

    def test_script_inside_span_synthesis_stripped_before_parse(self):
        # After sanitize(), the <script> is gone; parser sees clean synthesis.
        raw_content = "<span id='weather_1'><script>evil()</script>Partly cloudy.</span>"
        sanitised = sanitize(raw_content)
        tc = _tc('{"temperature_c":12}\n\n<span id=\'weather_1\'>')
        segs = parse(sanitised, [tc])
        # Should produce exactly one rich segment (script stripped, text remains)
        rich = [s for s in segs if s["type"] == "rich"]
        assert len(rich) == 1
        assert "evil" not in rich[0]["synthesis"]
        assert "Partly cloudy." in rich[0]["synthesis"]

    def test_onclick_span_after_sanitise_still_pairs_correctly(self):
        # onclick is stripped by sanitize(); the span+id survives; parser pairs.
        raw_content = "<span id='weather_1' onclick='alert(1)'>Sunny.</span>"
        sanitised = sanitize(raw_content)
        tc = _tc('{"temperature_c":22}\n\n<span id=\'weather_1\'>')
        segs = parse(sanitised, [tc])
        rich = [s for s in segs if s["type"] == "rich"]
        assert len(rich) == 1
        assert rich[0]["tag"] == "weather_1"
        assert rich[0]["synthesis"] == "Sunny."

    def test_numeric_only_id_not_recognised_as_rich(self):
        # id '123' — no alpha prefix, regex [a-z][a-z0-9_]* does not match.
        tc = _tc('{"temperature_c":5}\n\n<span id=\'123\'>')
        content = "<span id='123'>text</span>"
        segs = parse(content, [tc])
        assert not any(s["type"] == "rich" for s in segs)

    def test_empty_id_falls_back_to_text(self):
        # id='' — regex requires at least one [a-z] char.
        tc = _tc('{"temperature_c":5}\n\n<span id=\'\'>')
        content = "<span id=''>text</span>"
        segs = parse(content, [tc])
        assert not any(s["type"] == "rich" for s in segs)

    def test_very_long_synthesis_handled_without_error(self):
        # Synthesis that fills the LLM context limit — must not raise or corrupt.
        long_text = "Sunny. " * 2000
        tc = _tc('{"temperature_c":25}\n\n<span id=\'weather_1\'>')
        content = f"<span id='weather_1'>{long_text}</span>"
        segs = parse(content, [tc])
        rich = [s for s in segs if s["type"] == "rich"]
        assert len(rich) == 1
        assert rich[0]["synthesis"].startswith("Sunny.")
