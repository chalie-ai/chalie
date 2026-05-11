"""Voice TTS text-cleanup tests.

Covers the pre-Kokoro text pipeline: markdown render → nh3 tag strip →
URL spoken-form rewrite → ordinal expansion → acronym expansion → whitespace collapse.

The individual regex chain (_strip_markdown) has been replaced by
markdown-it-py + nh3 (both upstream-tested); these tests verify the
END-TO-END _clean_for_tts and _segment_for_prosody behaviour for the cases
that matter in production:

* Markdown markup varieties strip cleanly
* HTML entities decode to spoken form
* Ordinal integers expand; bare integers do NOT
* Acronym letter-spacing still fires
* Whitespace collapses correctly across block boundaries
* pysbd segmentation handles abbreviations and sentence boundaries correctly
"""

import pytest

from api.voice import _clean_for_tts, _segment_for_prosody


@pytest.mark.unit
class TestCleanForTts:
    """End-to-end _clean_for_tts: markdown → plain text."""

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
        # not narration. The TTS path has always dropped image alt text.
        assert _clean_for_tts("![alt text](img.png)") == ""

    def test_markdown_list_items_separated(self):
        result = _clean_for_tts("- item one\n- item two\n- item three")
        assert "item one" in result
        assert "item two" in result
        assert "item three" in result

    def test_html_entities_decode(self):
        # Entities decoded by extract_plaintext via html.unescape
        assert _clean_for_tts("<p>cats &amp; dogs</p>") == "cats & dogs"
        assert _clean_for_tts("<p>it&#39;s fine</p>") == "it's fine"

    def test_url_rewritten_to_spoken_host(self):
        # http://google.com/123 → "google dot com" (protocol + path stripped,
        # dots spoken so the neural G2P doesn't fuse them into one token).
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
        # Multi-segment hosts get every dot spoken, not just TLD.
        result = _clean_for_tts("docs at https://api.example.co.uk/v2")
        assert "api dot example dot co dot uk" in result

    def test_url_without_path_is_spoken(self):
        result = _clean_for_tts("see https://example.com please")
        assert "example dot com" in result
        assert "please" in result

    def test_plain_text_passthrough(self):
        assert _clean_for_tts("just plain text") == "just plain text"

    def test_whitespace_collapses_across_blocks(self):
        assert _clean_for_tts("<p>a</p><p>b</p>") == "a b"

    def test_minified_list_items_separated(self):
        # Block-boundary spacing is handled by extract_plaintext; regression guard.
        result = _clean_for_tts("<ul><li>one</li><li>two</li><li>three</li></ul>")
        assert result == "one two three"

    def test_html_drops_actions_block(self):
        result = _clean_for_tts('<p>pick one</p><actions><action label="A" value="a"/></actions>')
        assert result == "pick one"

    def test_acronym_expansion(self):
        # 2-3 letter ALL-CAPS tokens are letter-spaced so neural G2P reads them correctly
        result = _clean_for_tts("Use the VM or GB setting")
        assert "V M" in result
        assert "G B" in result

    def test_ordinal_expansion(self):
        assert _clean_for_tts("She finished 1st.") == "She finished first."
        assert _clean_for_tts("21st century") == "twenty-first century"

    def test_bare_integers_not_expanded(self):
        # Phone numbers, version strings, and years must NOT change
        assert "1800" in _clean_for_tts("call 1800 for help")
        assert "3.11.0" in _clean_for_tts("Python 3.11.0 released")


@pytest.mark.unit
class TestSegmentForProsody:
    """Unit tests for pysbd-backed sentence segmentation."""

    def test_abbreviation_not_split(self):
        # "Dr. Smith" is ONE sentence — abbreviation must not trigger a break
        segments = _segment_for_prosody("Dr. Smith arrived. He spoke.")
        texts = [s for s, _ in segments]
        assert len(texts) == 2, f"Expected 2 sentences, got {len(texts)}: {texts}"
        assert "Dr. Smith arrived." in texts[0]

    def test_comma_stays_inside_sentence(self):
        # Comma is part of sentence 1, not a boundary that creates a separate chunk
        segments = _segment_for_prosody("Hello, world. How are you?")
        texts = [s for s, _ in segments]
        assert len(texts) == 2
        assert "Hello, world." in texts[0]
        assert "How are you?" in texts[1]

    def test_empty_string_returns_empty(self):
        assert _segment_for_prosody("") == []

    def test_single_sentence_no_terminal_gets_zero_pause(self):
        segments = _segment_for_prosody("A sentence with no terminal punctuation")
        assert len(segments) == 1
        text, pause = segments[0]
        assert pause == 0.0
        assert "no terminal punctuation" in text

    def test_last_sentence_has_zero_pause(self):
        # The final sentence must never carry a trailing silence pad
        segments = _segment_for_prosody("First sentence. Second sentence.")
        assert len(segments) == 2
        assert segments[-1][1] == 0.0

    def test_period_pause_value(self):
        # Non-final sentence ending in "." should use the period pause (0.28s)
        segments = _segment_for_prosody("One sentence. Two sentences.")
        assert len(segments) == 2
        _, pause = segments[0]
        assert pause == pytest.approx(0.28)

    def test_question_pause_value(self):
        segments = _segment_for_prosody("Are you sure? Yes I am.")
        assert len(segments) == 2
        _, pause = segments[0]
        assert pause == pytest.approx(0.30)
