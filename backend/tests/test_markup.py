import pytest

from services.markup import Markup


@pytest.mark.unit
class TestSanitize:
    def test_keeps_formatting_tags(self) -> None:
        out = Markup.sanitize("<p>a <b>bold</b> <i>italic</i> <u>under</u></p>")
        assert out == "<p>a <b>bold</b> <i>italic</i> <u>under</u></p>"

    def test_strips_script(self) -> None:
        out = Markup.sanitize("<p>safe</p><script>alert(1)</script>")
        assert "<script>" not in out
        assert "<p>safe</p>" in out

    def test_keeps_img_with_http_src(self) -> None:
        out = Markup.sanitize('<img src="https://x.com/c.png" alt="cat"/>')
        assert '<img' in out
        assert 'src="https://x.com/c.png"' in out
        assert 'alt="cat"' in out

    def test_strips_data_uri_image(self) -> None:
        out = Markup.sanitize('<img src="data:image/png;base64,xxx" alt="bad"/>')
        # data: scheme is not in url_schemes — src must be stripped or img dropped
        assert 'data:' not in out

    def test_keeps_table_structure(self) -> None:
        html = "<table><thead><tr><th>Name</th></tr></thead><tbody><tr><td>Alice</td></tr></tbody></table>"
        out = Markup.sanitize(html)
        assert "<table>" in out
        assert "<thead>" in out
        assert "<tbody>" in out
        assert "<tr>" in out
        assert "<th>" in out
        assert "<td>" in out

    def test_keeps_tfoot(self) -> None:
        html = "<table><tfoot><tr><td>Total</td></tr></tfoot></table>"
        out = Markup.sanitize(html)
        assert "<tfoot>" in out

    def test_preserves_table_cell_attrs_strips_unsafe(self) -> None:
        html = '<table class="fancy" style="width:100%"><tr><td colspan="2" style="color:red">x</td></tr></table>'
        out = Markup.sanitize(html)
        assert "<table>" in out
        assert "class=" not in out
        assert "style=" not in out  # stripped from table and td alike
        assert 'colspan="2"' in out  # structural attr preserved


@pytest.mark.unit
class TestExtractPlaintext:
    def test_strips_simple_tags(self) -> None:
        assert Markup.extract_plaintext("<p>hello <b>world</b></p>") == "hello world"

    def test_minified_list_items_are_separated(self) -> None:
        assert (
            Markup.extract_plaintext("<ul><li>one</li><li>two</li><li>three</li></ul>")
            == "one two three"
        )

    def test_table_cells_are_separated(self) -> None:
        assert (
            Markup.extract_plaintext("<table><tr><td>Alice</td><td>30</td></tr></table>")
            == "Alice 30"
        )


@pytest.mark.unit
class TestFormatSanitizesXss:
    """TKT-1492: _format must sanitize HTML at the persist-time boundary so that
    any dangerous markup the LLM emits (or that was in raw text) is stripped
    before it reaches the frontend's v-html renderer."""

    def test_format_strips_onerror_handler(self) -> None:
        from configs.channels.user import UserConfig
        from controllers.message_processor import MessageProcessor

        mp = MessageProcessor(UserConfig())
        payload = '<img src=x onerror=alert(1)>'
        out = mp._format(payload)
        assert "onerror" not in out
        assert "alert" not in out
        # bare <img> tag is fine — only attrs are stripped

    def test_format_strips_script_tags(self) -> None:
        from configs.channels.user import UserConfig
        from controllers.message_processor import MessageProcessor

        mp = MessageProcessor(UserConfig())
        payload = '<script>alert(1)</script>'
        out = mp._format(payload)
        assert "<script" not in out
        assert "alert" not in out
