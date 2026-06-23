"""Test <span id='weather_N'> tags survive sanitize() while other span attrs are stripped."""

import pytest

from services.markup import sanitize

pytestmark = pytest.mark.unit


class TestSpanIdWhitelist:
    def test_span_with_id_single_quote_survives(self) -> None:
        out = sanitize("<span id='weather_1'>Partly cloudy in London.</span>")
        assert "<span" in out
        assert "weather_1" in out
        assert "Partly cloudy in London." in out

    def test_span_class_stripped(self) -> None:
        out = sanitize("<span class='highlight'>text</span>")
        assert "class=" not in out
        # Inner text must survive
        assert "text" in out

    def test_span_with_multiple_attrs_only_id_kept(self) -> None:
        out = sanitize("<span id='weather_1' class='card' style='color:red'>text</span>")
        assert "id=" in out
        assert "weather_1" in out
        assert "class=" not in out
        assert "style=" not in out

    def test_span_without_id_still_renders_inner_text(self) -> None:
        out = sanitize("<span>inner text</span>")
        assert "inner text" in out

    def test_nested_span_in_prose_survives_inner_text(self) -> None:
        out = sanitize("<p>See <span id='weather_1'>card here</span> for details.</p>")
        assert "weather_1" in out
        assert "card here" in out
        assert "<p>" in out

    def test_span_and_existing_formatting_coexist(self) -> None:
        out = sanitize(
            "<p>Hello <b>world</b>. <span id='weather_1'>Sunny.</span></p>"
        )
        assert "<b>world</b>" in out
        assert "weather_1" in out
        assert "Sunny." in out
