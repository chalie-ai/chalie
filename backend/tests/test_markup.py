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


    def test_keeps_table_structure(self):
        html = "<table><thead><tr><th>Name</th></tr></thead><tbody><tr><td>Alice</td></tr></tbody></table>"
        out = sanitize(html)
        assert "<table>" in out
        assert "<thead>" in out
        assert "<tbody>" in out
        assert "<tr>" in out
        assert "<th>" in out
        assert "<td>" in out

    def test_keeps_tfoot(self):
        html = "<table><tfoot><tr><td>Total</td></tr></tfoot></table>"
        out = sanitize(html)
        assert "<tfoot>" in out

    def test_strips_table_attributes(self):
        html = '<table class="fancy" style="width:100%"><tr><td colspan="2">x</td></tr></table>'
        out = sanitize(html)
        assert "<table>" in out
        assert "class=" not in out
        assert "style=" not in out
        assert "colspan=" not in out


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
        assert (
            extract_plaintext("<ul><li>one</li><li>two</li><li>three</li></ul>")
            == "one two three"
        )

    def test_table_cells_are_separated(self):
        assert (
            extract_plaintext("<table><tr><td>Alice</td><td>30</td></tr></table>")
            == "Alice 30"
        )



