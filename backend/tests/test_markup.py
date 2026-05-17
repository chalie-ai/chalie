"""Feature tests for ``services.markup`` — the nh3-backed sanitiser chokepoint.

The LLM emits a strict subset of HTML. Every assistant response is run
through ``sanitize()`` before reaching the frontend. Tests here pin the
contract: which tags survive, which get stripped, which URL schemes are
blocked, what plain-text extraction produces.
"""

import pytest

from services.markup import (
    actions_to_xml,
    escape_attr,
    extract_plaintext,
    sanitize,
)


@pytest.mark.unit
class TestEscapeAttr:
    def test_escapes_ampersand_and_quote_combined(self):
        assert escape_attr('a & b "c"') == "a &amp; b &quot;c&quot;"



@pytest.mark.unit
class TestActionsToXml:
    def test_single_action(self):
        result = actions_to_xml([{"label": "Yes", "value": "confirm"}])
        assert result == '<actions><action label="Yes" value="confirm"/></actions>'

    def test_multiple_actions(self):
        result = actions_to_xml([
            {"label": "Yes", "value": "y"},
            {"label": "No", "value": "n"},
        ])
        assert (
            result
            == '<actions><action label="Yes" value="y"/><action label="No" value="n"/></actions>'
        )

    def test_escapes_attributes(self):
        result = actions_to_xml([{"label": 'Say "hi"', "value": "x<y"}])
        assert 'label="Say &quot;hi&quot;"' in result
        assert 'value="x&lt;y"' in result

@pytest.mark.unit
class TestSanitize:
    def test_keeps_formatting_tags(self):
        out = sanitize("<p>a <b>bold</b> <i>italic</i> <u>under</u></p>")
        assert out == "<p>a <b>bold</b> <i>italic</i> <u>under</u></p>"

    def test_strips_script(self):
        out = sanitize("<p>safe</p><script>alert(1)</script>")
        assert "<script>" not in out
        assert "<p>safe</p>" in out

    def test_strips_anchor_keeps_text(self):
        # The LLM is told never to emit <a>; if it slips through, the tag
        # is stripped but the visible label survives so the user still
        # sees the words.
        out = sanitize('<p>see <a href="https://x.com">docs</a></p>')
        assert "<a" not in out
        assert "docs" in out

    def test_strips_javascript_url_on_anchor(self):
        # Belt-and-braces: even if <a> is somehow allowed in future, the
        # url_schemes allowlist would block javascript: URLs.
        out = sanitize('<p><a href="javascript:alert(1)">x</a></p>')
        assert "javascript:" not in out

    def test_keeps_actions_block(self):
        out = sanitize('<p>pick</p><actions><action label="A" value="a"/></actions>')
        assert "<actions>" in out
        assert "<action " in out
        assert 'label="A"' in out
        assert 'value="a"' in out

    def test_keeps_action_overlay_attrs(self):
        # Overlay-action attributes wired by AppsPanel daemon
        out = sanitize('<actions><action label="L" value="v" execute="cmd" target="overlay"/></actions>')
        assert 'execute="cmd"' in out
        assert 'target="overlay"' in out

    def test_keeps_img_with_http_src(self):
        out = sanitize('<img src="https://x.com/c.png" alt="cat"/>')
        assert '<img' in out
        assert 'src="https://x.com/c.png"' in out
        assert 'alt="cat"' in out

    def test_strips_data_uri_image(self):
        out = sanitize('<img src="data:image/png;base64,xxx" alt="bad"/>')
        # data: scheme is not in url_schemes — src must be stripped or img dropped
        assert 'data:' not in out

    def test_empty_returns_empty(self):
        assert sanitize("") == ""
        assert sanitize(None) == ""


@pytest.mark.unit
class TestExtractPlaintext:
    def test_strips_simple_tags(self):
        assert extract_plaintext("<p>hello <b>world</b></p>") == "hello world"

    def test_drops_actions_block(self):
        assert (
            extract_plaintext('<p>pick</p><actions><action label="A" value="a"/></actions>')
            == "pick"
        )

    def test_collapses_whitespace(self):
        assert extract_plaintext("<p>a   b\n\n  c</p>") == "a b c"

    def test_decodes_entities(self):
        assert extract_plaintext("<p>a &amp; b</p>") == "a & b"

    def test_minified_list_items_are_separated(self):
        # Without block-boundary spacing this collapses to ``oneTwoThree`` and
        # phonemizer drops the gibberish tokens — the "lists skipped" TTS bug.
        assert (
            extract_plaintext("<ul><li>one</li><li>two</li><li>three</li></ul>")
            == "one two three"
        )

    def test_h1_followed_by_paragraph_is_separated(self):
        assert extract_plaintext("<h1>Title</h1><p>Body.</p>") == "Title Body."

    def test_br_separates_inline_text(self):
        assert extract_plaintext("<p>a<br>b<br>c</p>") == "a b c"
