import pytest
from services.markup import escape_text, wrap_text_xml, actions_to_xml, tokenize, Token, extract_plaintext


@pytest.mark.unit
class TestEscapeText:
    def test_escapes_ampersand(self):
        assert escape_text("a & b") == "a &amp; b"

    def test_escapes_lt_gt(self):
        assert escape_text("<script>") == "&lt;script&gt;"

    def test_passes_normal_text(self):
        assert escape_text("hello world") == "hello world"

    def test_empty_string(self):
        assert escape_text("") == ""


@pytest.mark.unit
class TestWrapTextXml:
    def test_wraps_in_paragraph(self):
        assert wrap_text_xml("hello") == "<p>hello</p>"

    def test_escapes_content(self):
        assert wrap_text_xml("a < b") == "<p>a &lt; b</p>"

    def test_empty_returns_empty(self):
        assert wrap_text_xml("") == ""

    def test_strips_outer_whitespace(self):
        assert wrap_text_xml("  hello  ") == "<p>hello</p>"


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

    def test_empty_returns_empty(self):
        assert actions_to_xml([]) == ""


@pytest.mark.unit
class TestTokenize:
    def test_plain_text(self):
        tokens = tokenize("hello world")
        assert tokens == [Token("text", "hello world", {})]

    def test_simple_tag(self):
        tokens = tokenize("<b>hi</b>")
        assert tokens == [
            Token("open", "b", {}),
            Token("text", "hi", {}),
            Token("close", "b", {}),
        ]

    def test_self_closing(self):
        tokens = tokenize('<img src="x" alt="y"/>')
        assert tokens == [Token("void", "img", {"src": "x", "alt": "y"})]

    def test_attributes(self):
        tokens = tokenize('<a href="https://x.com">link</a>')
        assert tokens[0] == Token("open", "a", {"href": "https://x.com"})

    def test_unknown_tag_kept_as_text(self):
        # Strict allowlist: <bogus> rendered as escaped text
        tokens = tokenize("<bogus>x</bogus>")
        assert tokens == [Token("text", "<bogus>x</bogus>", {})]

    def test_unclosed_tag_auto_closed(self):
        tokens = tokenize("<b>unclosed")
        assert tokens == [Token("open", "b", {}), Token("text", "unclosed", {})]
        # Renderer is responsible for auto-closing at EOF; tokenize does not synthesize close

    def test_nested_tags(self):
        tokens = tokenize("<p>a <b>bold</b> c</p>")
        assert len(tokens) == 7
        assert tokens[2] == Token("open", "b", {})

    def test_decodes_entities_in_text(self):
        tokens = tokenize("a &amp; b &lt; c")
        assert tokens == [Token("text", "a & b < c", {})]


@pytest.mark.unit
class TestExtractPlaintext:
    def test_strips_simple_tags(self):
        assert extract_plaintext("<p>hello <b>world</b></p>") == "hello world"

    def test_drops_actions_block(self):
        assert (
            extract_plaintext('<p>pick</p><actions><action label="A" value="a"/></actions>')
            == "pick"
        )

    def test_uses_img_alt(self):
        assert extract_plaintext('<img src="x" alt="a cat"/>') == "a cat"

    def test_omits_img_with_no_alt(self):
        assert extract_plaintext('<img src="x"/>') == ""

    def test_collapses_whitespace(self):
        assert extract_plaintext("<p>a   b\n\n  c</p>") == "a b c"

    def test_empty(self):
        assert extract_plaintext("") == ""
