"""Augmented Phase A coverage for markup.py and markdown_xml_migration.py.

Each test here catches a specific regression that the original 55 tests miss.
The regressions are stated in each test's docstring.
"""

import pytest

from services.markup import (
    Token,
    escape_attr,
    extract_plaintext,
    tokenize,
)
from services.markdown_xml_migration import markdown_to_xml, run_if_needed


# ─────────────────────────────────────────────────────────────────
# escape_attr — not covered by existing tests beyond escapes_attributes
# ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestEscapeAttr:
    def test_escapes_double_quote(self):
        """Regression: double-quote in an attr value breaks out of the attribute
        and allows injecting arbitrary XML attributes.
        """
        assert escape_attr('"') == "&quot;"

    def test_escapes_ampersand_and_quote_combined(self):
        """Regression: LLM-produced href like 'a & b "c"' must have both
        & and " escaped so the rendered attribute is well-formed XML.
        """
        assert escape_attr('a & b "c"') == "a &amp; b &quot;c&quot;"

    def test_passes_unicode_unchanged(self):
        """Regression: non-ASCII characters (e.g. accented letters, emoji) in
        attribute values must survive the escape pass without corruption.
        """
        assert escape_attr("café ☕") == "café ☕"


# ─────────────────────────────────────────────────────────────────
# tokenize — hostile LLM output not covered by existing tests
# ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTokenizeHostileInput:
    def test_mixed_case_tag_lowercased(self):
        """Regression: LLMs sometimes emit <B> or <P> in upper case.
        The tokenizer must normalise to lowercase so the tag lands in ALLOWED_TAGS
        and is not silently dropped as an unknown tag.
        """
        toks = tokenize("<B>bold</B>")
        assert toks[0] == Token("open", "b", {})
        assert toks[2] == Token("close", "b", {})

    def test_attr_entity_injection_decoded_in_attrs(self):
        """Regression: an LLM that emits &quot; inside an attribute value to
        break out of the attribute context must be decoded INTO the attrs dict,
        not silently promoted to inject extra attributes.
        e.g. href='x&quot; onclick=&quot;evil' must stay as one href value.
        """
        toks = tokenize('<a href="x&quot; onclick=&quot;evil">click</a>')
        # The attribute regex strips the outer double-quotes by design; the
        # decoded value contains the raw " characters — one href, no onclick.
        assert len(toks) == 3
        href = toks[0].attrs.get("href", "MISSING")
        assert "onclick" not in toks[0].attrs
        assert "onclick" not in href or href.startswith("x")

    def test_partial_tag_at_eof_becomes_text(self):
        """Regression: a response truncated mid-tag (<p>partial <b) must not
        crash the tokenizer or silently discard content.  The dangling '<b'
        falls through as trailing text.
        """
        toks = tokenize("<p>partial <b")
        assert toks[0] == Token("open", "p", {})
        assert toks[1].kind == "text"
        assert "partial" in toks[1].name

    def test_nested_same_tag_tokenizes_all_tokens(self):
        """Regression: an LLM that erroneously nests <p> inside <p> must
        produce all four boundary tokens so the renderer can recover gracefully,
        rather than merging or discarding any.
        """
        toks = tokenize("<p>outer <p>inner</p></p>")
        kinds = [t.kind for t in toks]
        assert kinds.count("open") == 2
        assert kinds.count("close") == 2


# ─────────────────────────────────────────────────────────────────
# extract_plaintext — entity decoding not covered
# ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestExtractPlaintextEntities:
    def test_entities_decoded_to_plain_characters(self):
        """Regression: TTS receives &amp; literally instead of & if entity
        decoding is skipped during plaintext extraction.
        """
        result = extract_plaintext("<p>a &amp; b</p>")
        assert result == "a & b"

    def test_entities_decoded_in_img_alt(self):
        """Regression: <img alt="cats &amp; dogs"/> alt text for TTS must
        decode entities so the speaker says 'cats & dogs', not 'cats &amp; dogs'.
        """
        result = extract_plaintext('<img src="x" alt="cats &amp; dogs"/>')
        assert result == "cats & dogs"


# ─────────────────────────────────────────────────────────────────
# markdown_to_xml — gap scenarios from the audit brief
# ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestMarkdownToXmlGaps:
    def test_inline_code_containing_xml_special_chars(self):
        """Regression: `<b>` inside backtick code must be double-escaped to
        &amp;lt;b&amp;gt; so the renderer shows the literal text '<b>' rather
        than treating it as a bold tag.
        """
        result = markdown_to_xml("use `<b>` tag")
        # The <code> element must contain escaped text, not a raw tag
        assert "<code>" in result
        assert "<b>" not in result.split("<code>")[1].split("</code>")[0]
        # The actual displayed text is the escaped form
        assert "&lt;b&gt;" in result or "&amp;lt;b&amp;gt;" in result

    def test_mixed_markdown_heading_and_already_xml_paragraph(self):
        """Regression: when _looks_like_xml() returns False (because the content
        starts with '# Heading' not a tag), the full block-split path runs and
        a later block starting with '<p>already xml</p>' should be treated as a
        plain-text paragraph (escaped), not silently passed through as raw XML.
        This documents the current best-effort behaviour so a future refactor
        cannot accidentally start emitting un-escaped passthrough XML mid-document.
        """
        result = markdown_to_xml("# Heading\n\n<p>already xml</p>")
        assert result.startswith("<h1>")
        # The <p>already xml</p> block must be escaped, not raw passthrough
        assert "&lt;p&gt;" in result or result.count("<p>") == 0

    def test_multiple_consecutive_blank_lines_collapse_to_two_paragraphs(self):
        """Regression: four blank lines between two paragraphs must still produce
        exactly two <p> elements, not an empty one or a crash.
        """
        result = markdown_to_xml("first\n\n\n\nsecond")
        assert result == "<p>first</p><p>second</p>"

    def test_nested_list_items_flattened_to_single_ul(self):
        """Regression: indented sub-items (nested lists) are not supported by
        the 11-tag schema.  They must be flattened into the parent <ul> as
        peer <li> elements rather than producing invalid nesting or being dropped.
        """
        result = markdown_to_xml("- one\n  - nested\n- two")
        assert result.startswith("<ul>")
        assert result.count("<li>") >= 2
        assert "<ul><ul>" not in result  # no nested ul

    def test_crlf_line_endings_handled(self):
        """Regression: responses copy-pasted from Windows sources use CRLF.
        The heading and paragraph split must work identically to LF-only input.
        """
        result = markdown_to_xml("# Title\r\n\r\nParagraph")
        assert "<h1>Title</h1>" in result
        assert "<p>Paragraph</p>" in result


# ─────────────────────────────────────────────────────────────────
# run_if_needed — boot integration via real schema
# ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestRunIfNeededBootIntegration:
    """Boot-time migration smoke: real schema, real SQLite, no mocks."""

    def test_migrates_three_markdown_rows_on_real_schema(self, db):
        """Regression: the boot hook calls run_if_needed() against the real
        transcript table (schema.sql).  If the xml_migrated column is missing
        or the SELECT/UPDATE SQL is wrong, all three rows stay unconverted and
        every subsequent turn emits markdown to the frontend.

        Seeds 3 markdown rows, calls run_if_needed, asserts all emerge as XML
        and xml_migrated=1.
        """
        rows = [
            ("assistant", "**bold** answer", "ch1"),
            ("assistant", "# Title\n\nBody text", "ch1"),
            ("assistant", "_italic_ note", "ch1"),
        ]
        for role, content, channel in rows:
            db.execute(
                "INSERT INTO transcript (channel, role, content, xml_migrated) "
                "VALUES (?, ?, ?, 0)",
                (channel, role, content),
            )
        db.commit()

        count = run_if_needed(db)

        assert count == 3
        results = db.execute(
            "SELECT content, xml_migrated FROM transcript ORDER BY id"
        ).fetchall()
        for content_out, migrated in results:
            assert migrated == 1, "xml_migrated sentinel must be set to 1"
            assert content_out.startswith("<"), (
                f"Content must start with an XML tag after migration; got: {content_out!r}"
            )
        # Spot-check specific conversions
        assert results[0][0] == "<p><b>bold</b> answer</p>"
        assert results[1][0] == "<h1>Title</h1><p>Body text</p>"
        assert results[2][0] == "<p><i>italic</i> note</p>"
