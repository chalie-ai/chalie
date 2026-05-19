"""Feature tests for ``services.markup`` — the nh3-backed sanitiser chokepoint.

The LLM emits a strict subset of HTML. Every assistant response is run
through ``sanitize()`` before reaching the frontend. Tests here pin the
contract: which tags survive, which get stripped, which URL schemes are
blocked, what plain-text extraction produces.
"""

import pytest

from services.markup import (
    actions_to_xml,
    extract_plaintext,
    sanitize,
)



@pytest.mark.unit
class TestActionsToXml:
    def test_single_action(self):
        result = actions_to_xml([{"label": "Yes", "value": "confirm"}])
        assert result == '<actions><action label="Yes" value="confirm"/></actions>'


@pytest.mark.unit
class TestSanitize:
    def test_keeps_formatting_tags(self):
        out = sanitize("<p>a <b>bold</b> <i>italic</i> <u>under</u></p>")
        assert out == "<p>a <b>bold</b> <i>italic</i> <u>under</u></p>"

    def test_strips_script(self):
        out = sanitize("<p>safe</p><script>alert(1)</script>")
        assert "<script>" not in out
        assert "<p>safe</p>" in out

    def test_keeps_actions_block(self):
        out = sanitize('<p>pick</p><actions><action label="A" value="a"/></actions>')
        assert "<actions>" in out
        assert "<action " in out
        assert 'label="A"' in out
        assert 'value="a"' in out

    def test_keeps_img_with_http_src(self):
        out = sanitize('<img src="https://x.com/c.png" alt="cat"/>')
        assert '<img' in out
        assert 'src="https://x.com/c.png"' in out
        assert 'alt="cat"' in out

    def test_strips_data_uri_image(self):
        out = sanitize('<img src="data:image/png;base64,xxx" alt="bad"/>')
        # data: scheme is not in url_schemes — src must be stripped or img dropped
        assert 'data:' not in out


@pytest.mark.unit
class TestExtractPlaintext:
    def test_strips_simple_tags(self):
        assert extract_plaintext("<p>hello <b>world</b></p>") == "hello world"

    def test_drops_actions_block(self):
        assert (
            extract_plaintext('<p>pick</p><actions><action label="A" value="a"/></actions>')
            == "pick"
        )

    def test_minified_list_items_are_separated(self):
        # Without block-boundary spacing this collapses to ``oneTwoThree`` and
        # phonemizer drops the gibberish tokens — the "lists skipped" TTS bug.
        assert (
            extract_plaintext("<ul><li>one</li><li>two</li><li>three</li></ul>")
            == "one two three"
        )



