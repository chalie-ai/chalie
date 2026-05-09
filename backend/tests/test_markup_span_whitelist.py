"""Unit tests for the span+id whitelist in services.markup.

Asserts that:
- <span id='weather_N'> tags survive sanitize() so the RichMediaParser can
  read them at the WS-send boundary.
- No other span attributes (class, style, onclick, data-*) survive.
- Existing non-span tags continue to behave as before.
- The id attribute value is preserved verbatim (structural validation happens
  in RichMediaParser, not here).
"""

import pytest

from services.markup import sanitize

pytestmark = pytest.mark.unit


class TestSpanIdWhitelist:
    def test_span_with_id_single_quote_survives(self):
        out = sanitize("<span id='weather_1'>Partly cloudy in London.</span>")
        assert "<span" in out
        assert "weather_1" in out
        assert "Partly cloudy in London." in out

    def test_span_with_id_double_quote_survives(self):
        out = sanitize('<span id="weather_1">Sunny skies.</span>')
        assert "<span" in out
        assert "weather_1" in out
        assert "Sunny skies." in out

    def test_span_id_preserved_for_ordinal_2(self):
        out = sanitize("<span id='weather_2'>Cold front moving in.</span>")
        assert "weather_2" in out

    def test_span_class_stripped(self):
        out = sanitize("<span class='highlight'>text</span>")
        assert "class=" not in out
        # Inner text must survive
        assert "text" in out

    def test_span_style_stripped(self):
        out = sanitize('<span style="color:red">text</span>')
        assert "style=" not in out
        assert "text" in out

    def test_span_onclick_stripped(self):
        out = sanitize("<span onclick='alert(1)'>text</span>")
        assert "onclick" not in out
        assert "text" in out

    def test_span_data_attr_stripped(self):
        out = sanitize("<span data-evil='payload'>text</span>")
        assert "data-evil" not in out
        assert "text" in out

    def test_span_with_multiple_attrs_only_id_kept(self):
        out = sanitize("<span id='weather_1' class='card' style='color:red'>text</span>")
        assert "id=" in out
        assert "weather_1" in out
        assert "class=" not in out
        assert "style=" not in out

    def test_span_without_id_still_renders_inner_text(self):
        out = sanitize("<span>inner text</span>")
        assert "inner text" in out

    def test_nested_span_in_prose_survives_inner_text(self):
        out = sanitize("<p>See <span id='weather_1'>card here</span> for details.</p>")
        assert "weather_1" in out
        assert "card here" in out
        assert "<p>" in out

    def test_script_still_stripped(self):
        out = sanitize("<script>evil()</script>")
        assert "<script>" not in out
        assert "evil" not in out

    def test_anchor_still_stripped(self):
        out = sanitize('<a href="https://x.com">link</a>')
        assert "<a" not in out
        assert "link" in out

    def test_span_and_existing_formatting_coexist(self):
        out = sanitize(
            "<p>Hello <b>world</b>. <span id='weather_1'>Sunny.</span></p>"
        )
        assert "<b>world</b>" in out
        assert "weather_1" in out
        assert "Sunny." in out
