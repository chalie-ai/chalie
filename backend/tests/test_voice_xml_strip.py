"""Voice TTS text-cleanup tests.

Covers the pre-Kokoro text pipeline:
    markdown render → plaintext → URL spoken-host rewrite → whitespace collapse.

Ordinal expansion, acronym letter-spacing, and per-sentence segmentation are
NOT tested here — those transformations are handled by kokoro-onnx / espeak-ng
natively and are not part of the Python-side pipeline.

All tests are @pytest.mark.unit: _clean_for_tts is a pure function with no
collaborators beyond markdown-it-py and extract_plaintext (both pure). No
model files, no network, no DB required.
"""

import pytest

from api.voice import _clean_for_tts


@pytest.mark.unit
class TestCleanForTts:
    """End-to-end _clean_for_tts: markdown/HTML → plaintext → URL rewrite."""

    def test_markdown_header_strips(self):
        assert _clean_for_tts("# My Header") == "My Header"

    def test_markdown_bold_strips(self):
        assert _clean_for_tts("**bold text**") == "bold text"

    def test_markdown_italic_strips(self):
        assert _clean_for_tts("*italic text*") == "italic text"

    def test_markdown_inline_code_strips(self):
        assert _clean_for_tts("use `print()` here") == "use print() here"

    def test_markdown_fenced_code_strips(self):
        result = _clean_for_tts("```python\nprint('hi')\n```")
        assert "```" not in result
        assert "print" in result

    def test_markdown_link_keeps_text(self):
        assert _clean_for_tts("[click here](https://example.com)") == "click here"

    def test_markdown_image_drops_entirely(self):
        # Images are programmatic affordances — alt text is a visual a11y label,
        # not narration. The TTS path has always dropped image content.
        assert _clean_for_tts("![alt text](img.png)") == ""

    def test_markdown_list_items_separated(self):
        result = _clean_for_tts("- item one\n- item two\n- item three")
        assert "item one" in result
        assert "item two" in result
        assert "item three" in result

    def test_html_entities_decode(self):
        assert _clean_for_tts("<p>cats &amp; dogs</p>") == "cats & dogs"
        assert _clean_for_tts("<p>it&#39;s fine</p>") == "it's fine"

    def test_url_rewritten_to_spoken_host(self):
        # http://google.com/123 → "google dot com" (protocol + path stripped,
        # dots spoken so espeak-ng doesn't fuse them into one unintelligible token).
        result = _clean_for_tts("see http://google.com/123 for details")
        assert "http" not in result
        assert "/123" not in result
        assert "google dot com" in result
        assert "details" in result

    def test_url_strips_www_prefix(self):
        result = _clean_for_tts("visit https://www.example.com/page")
        assert "www" not in result
        assert "example dot com" in result

    def test_url_subdomain_is_spoken(self):
        # Multi-segment hosts get every dot spoken, not just the TLD.
        result = _clean_for_tts("docs at https://api.example.co.uk/v2")
        assert "api dot example dot co dot uk" in result

    def test_url_without_path_is_spoken(self):
        result = _clean_for_tts("see https://example.com please")
        assert "example dot com" in result
        assert "please" in result

    def test_bare_url_round_trip_produces_spoken_host(self):
        # Regression guard: a bare URL with no surrounding text still produces
        # the spoken-host form (not an empty string, not the raw URL).
        result = _clean_for_tts("http://google.com/path/to/page")
        assert result == "google dot com"

    def test_plain_text_passthrough(self):
        assert _clean_for_tts("just plain text") == "just plain text"

    def test_whitespace_collapses_across_blocks(self):
        assert _clean_for_tts("<p>a</p><p>b</p>") == "a b"

    def test_minified_list_items_separated(self):
        # Block-boundary spacing is handled by extract_plaintext; regression guard
        # for the incident where <ul><li>A</li><li>B</li></ul> collapsed to "AB".
        result = _clean_for_tts("<ul><li>one</li><li>two</li><li>three</li></ul>")
        assert result == "one two three"

    def test_html_drops_actions_block(self):
        result = _clean_for_tts('<p>pick one</p><actions><action label="A" value="a"/></actions>')
        assert result == "pick one"
