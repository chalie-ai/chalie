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

    def test_url_subdomain_is_spoken(self):
        # Multi-segment hosts get every dot spoken, not just the TLD.
        result = _clean_for_tts("docs at https://api.example.co.uk/v2")
        assert "api dot example dot co dot uk" in result

    def test_bare_url_round_trip_produces_spoken_host(self):
        # Regression guard: a bare URL with no surrounding text still produces
        # the spoken-host form (not an empty string, not the raw URL).
        result = _clean_for_tts("http://google.com/path/to/page")
        assert result == "google dot com"

    def test_whitespace_collapses_across_blocks(self):
        assert _clean_for_tts("<p>a</p><p>b</p>") == "a b"

    def test_minified_list_items_separated(self):
        # Items get a sentence-terminating period so espeak produces a real
        # pause between them. Regression guard for the May 9 incident where
        # <ul><li>A</li><li>B</li></ul> collapsed to "AB", plus the May 11
        # follow-up where the items separated but ran together prosodically.
        result = _clean_for_tts("<ul><li>one</li><li>two</li><li>three</li></ul>")
        assert result == "one. two. three."

    def test_list_item_existing_punctuation_preserved(self):
        # Items already ending in sentence-terminators do not get an extra
        # period stacked on — "A." stays "A.", "B!" stays "B!".
        result = _clean_for_tts("<ul><li>Boil water.</li><li>Add pasta!</li><li>Stir</li></ul>")
        assert result == "Boil water. Add pasta! Stir."

    def test_markdown_list_gets_pauses(self):
        # The markdown path renders `- item` → `<li>item</li>` and must pick
        # up the same terminator injection as raw HTML.
        result = _clean_for_tts("Steps:\n- one\n- two\n- three\n")
        assert result == "Steps: one. two. three."

    def test_html_drops_actions_block(self):
        result = _clean_for_tts('<p>pick one</p><actions><action label="A" value="a"/></actions>')
        assert result == "pick one"
