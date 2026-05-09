"""Voice TTS text-cleanup tests.

Covers the pre-Kokoro text pipeline: HTML extract → markdown strip →
whitespace collapse. The synthesis path itself (Kokoro.create) is the
library's responsibility — we no longer chunk, retry, trim silence, or
detect saturation in our own code, so there is nothing to test here for
that layer beyond the integration smoke check exercised on chalie-dev.
"""

import pytest

from api.voice import _clean_for_tts, _strip_markdown


@pytest.mark.unit
class TestCleanForTtsXml:
    def test_strips_xml_tags(self):
        assert _clean_for_tts("<p>hello <b>world</b></p>") == "hello world"

    def test_drops_actions_block(self):
        assert (
            _clean_for_tts('<p>pick</p><actions><action label="A" value="a"/></actions>')
            == "pick"
        )

    def test_drops_img_entirely(self):
        # ``alt`` is an accessibility label for the visual surface; the spoken
        # TTS path no longer narrates it. Images are programmatic affordances
        # emitted by the harness — narration covers the surrounding prose.
        assert _clean_for_tts('<img src="x" alt="a cat"/>') == ""

    def test_preserves_plain_text(self):
        assert _clean_for_tts("just text") == "just text"

    def test_handles_entities(self):
        assert _clean_for_tts("<p>a &amp; b</p>") == "a & b"

    def test_collapses_whitespace(self):
        assert _clean_for_tts("<p>a   b</p>  <p>c</p>") == "a b c"

    def test_minified_list_items_separated(self):
        # Block-boundary fix lives in services.markup.extract_plaintext —
        # exercise it here too so a regression in the TTS path also fails.
        assert (
            _clean_for_tts("<ul><li>one</li><li>two</li><li>three</li></ul>")
            == "one two three"
        )


@pytest.mark.unit
class TestStripMarkdown:
    def test_italic_asterisk_becomes_word(self):
        # The user's exact reported bug: ``*example*`` was spoken as
        # "asterisk example asterisk" because Kokoro pronounces literal
        # punctuation. The italic marker must vanish.
        assert _strip_markdown("*example*") == "example"

    def test_bold_double_asterisk(self):
        assert _strip_markdown("**bold**") == "bold"

    def test_italic_underscore(self):
        assert _strip_markdown("_italic_") == "italic"

    def test_bold_underscore(self):
        assert _strip_markdown("__bold__") == "bold"

    def test_underscore_preserved_in_identifier(self):
        # ``module_name`` is a single token, not italic emphasis. The
        # underscore-as-italic regex must not strip word-internal underscores.
        assert _strip_markdown("see module_name in code") == "see module_name in code"

    def test_asterisk_math_preserved(self):
        # Space-flanked asterisks are arithmetic, not emphasis. The italic
        # regex requires non-space inside the wrap.
        assert _strip_markdown("2 * 3 = 6") == "2 * 3 = 6"

    def test_inline_code(self):
        assert _strip_markdown("use `print()` here") == "use print() here"

    def test_fenced_code_block(self):
        text = "before\n```python\nprint('hi')\n```\nafter"
        assert "```" not in _strip_markdown(text)
        assert "print('hi')" in _strip_markdown(text)

    def test_markdown_link(self):
        assert _strip_markdown("[click here](https://example.com)") == "click here"

    def test_markdown_image(self):
        assert _strip_markdown("![alt text](img.png)") == "alt text"

    def test_bare_url_dropped(self):
        # Bare URLs read aloud are noise. The synthesiser would say
        # "h-t-t-p-s-colon-slash-slash..." otherwise.
        out = _strip_markdown("see https://example.com/foo for details")
        assert "http" not in out
        assert "example.com" not in out
        assert "details" in out

    def test_header(self):
        assert _strip_markdown("# Title").strip() == "Title"
        assert _strip_markdown("### Sub").strip() == "Sub"

    def test_blockquote(self):
        assert _strip_markdown("> quoted").strip() == "quoted"

    def test_list_bullet(self):
        assert _strip_markdown("- item one").strip() == "item one"
        assert _strip_markdown("* item two").strip() == "item two"

    def test_list_numbered(self):
        assert _strip_markdown("1. first").strip() == "first"

    def test_horizontal_rule_dropped(self):
        text = "above\n\n---\n\nbelow"
        assert "---" not in _strip_markdown(text)


@pytest.mark.unit
class TestCleanForTtsMarkdown:
    """End-to-end via _clean_for_tts (the real entry point)."""

    def test_markdown_stripped_without_html(self):
        assert _clean_for_tts("*example*") == "example"

    def test_markdown_inside_html(self):
        # Both passes run: HTML extract first, then markdown strip.
        assert _clean_for_tts("<p>say *example* now</p>") == "say example now"

    def test_link_with_html_paragraph(self):
        assert (
            _clean_for_tts("<p>see [docs](https://x.com) here</p>")
            == "see docs here"
        )

    def test_real_world_assistant_reply(self):
        text = (
            "<p>Here's the plan:</p>"
            "<ul><li>**install** the package</li>"
            "<li>run `pytest` to verify</li></ul>"
        )
        out = _clean_for_tts(text)
        assert "*" not in out
        assert "`" not in out
        assert "install" in out
        assert "pytest" in out
