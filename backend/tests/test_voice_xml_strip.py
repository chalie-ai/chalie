from unittest.mock import patch

import numpy as np
import pytest

from api import voice as voice_module
from api.voice import (
    _chunk_text_for_tts,
    _clean_for_tts,
    _strip_markdown,
    _synthesise_chunk_safe,
)


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


@pytest.mark.unit
class TestChunkTextForTts:
    def test_empty_text(self):
        assert _chunk_text_for_tts("") == []
        assert _chunk_text_for_tts("   ") == []

    def test_short_text_single_chunk(self):
        assert _chunk_text_for_tts("hello world") == ["hello world"]

    def test_no_truncation_total_chars(self):
        # Truncation is the bug we're fixing. The chunker MUST emit every
        # word it received, in order, with no silent drops.
        text = ("This is a longer test. " * 60).strip()  # ~1380 chars
        chunks = _chunk_text_for_tts(text)
        rejoined = " ".join(chunks)
        assert text.split() == rejoined.split()

    def test_every_chunk_within_budget(self):
        text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50).strip()
        for chunk in _chunk_text_for_tts(text):
            assert len(chunk) <= 400, (
                f"Chunk over budget ({len(chunk)} chars): {chunk!r}"
            )

    def test_long_sentence_falls_to_clause_split(self):
        # One sentence, no terminal punctuation, but plenty of commas.
        # Cascading split should land on clause boundaries.
        text = ", ".join(["clause " + str(i) * 30 for i in range(20)]) + "."
        chunks = _chunk_text_for_tts(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 400

    def test_runaway_token_hard_cut(self):
        # Pathological single token > target. Must still produce chunks
        # under budget — the alternative is silent overflow at the model.
        runaway = "x" * 1000
        chunks = _chunk_text_for_tts(runaway)
        assert chunks  # never empty
        for c in chunks:
            assert len(c) <= 400
        # Nothing dropped: rejoined char count matches input.
        assert sum(len(c) for c in chunks) == len(runaway)

    def test_packs_short_sentences(self):
        # Many short sentences should coalesce — we don't want one
        # Kokoro call per period.
        text = " ".join(["Hi." for _ in range(50)])
        chunks = _chunk_text_for_tts(text)
        assert len(chunks) < 50
        for c in chunks:
            assert len(c) <= 400

    def test_single_long_paragraph(self):
        # Reproduces the original truncation case: ~2000 chars of dense
        # prose without paragraph breaks. Must split, must lose nothing.
        text = (
            "The quick brown fox jumps over the lazy dog. " * 50
        ).strip()
        chunks = _chunk_text_for_tts(text)
        assert len(chunks) >= 4
        rejoined = " ".join(chunks)
        assert text.split() == rejoined.split()

    def test_newline_is_primary_boundary(self):
        # Each newline-separated paragraph is short enough to be its own
        # atom — the chunker must respect line boundaries before falling
        # to sentence/clause splits.
        text = "Line one.\nLine two.\nLine three."
        chunks = _chunk_text_for_tts(text)
        # Three short atoms pack into a single chunk under the budget.
        assert len(chunks) == 1
        assert "Line one." in chunks[0]
        assert "Line three." in chunks[0]

    def test_long_paragraphs_separated_by_newlines(self):
        # Two long paragraphs separated by newline. Each individually
        # exceeds the budget so each cascades to sentence-level split,
        # but the boundary between them stays intact (no run-on).
        para1 = ("First topic here. " * 30).strip()
        para2 = ("Second topic here. " * 30).strip()
        text = para1 + "\n\n" + para2
        chunks = _chunk_text_for_tts(text)
        rejoined = " ".join(chunks)
        # Lossless: every word survives.
        assert text.split() == rejoined.split()
        # Budget: every chunk fits.
        for c in chunks:
            assert len(c) <= 400

    def test_blank_lines_collapsed(self):
        text = "Hello.\n\n\n\nWorld."
        chunks = _chunk_text_for_tts(text)
        assert chunks == ["Hello. World."]


@pytest.mark.unit
class TestSynthesiseChunkSafe:
    """Covers the retry-on-failure / retry-on-silent wrapper around
    ``_synthesise_chunk``. The phonemizer occasionally returns truncated
    output for chunks dominated by punctuation/em-dashes, leaving Kokoro
    to produce near-silent audio; a single bad chunk used to take the
    whole synthesis down via ``pool.map``'s exception propagation."""

    def test_success_first_try(self):
        good = np.ones(48_000, dtype=np.float32)
        with patch.object(voice_module, "_synthesise_chunk", return_value=good) as syn:
            audio = _synthesise_chunk_safe((0, "hello world"))
        assert syn.call_count == 1
        np.testing.assert_array_equal(audio, good)

    def test_retries_on_exception_then_succeeds(self):
        good = np.ones(48_000, dtype=np.float32)
        with patch.object(
            voice_module,
            "_synthesise_chunk",
            side_effect=[RuntimeError("phonemizer mismatch"), good],
        ) as syn:
            audio = _synthesise_chunk_safe((1, "tricky chunk"))
        assert syn.call_count == 2
        np.testing.assert_array_equal(audio, good)

    def test_retries_on_silent_then_succeeds(self):
        silent = np.zeros(50, dtype=np.float32)  # below _TTS_MIN_SAMPLES
        good = np.ones(48_000, dtype=np.float32)
        with patch.object(
            voice_module,
            "_synthesise_chunk",
            side_effect=[silent, good],
        ) as syn:
            audio = _synthesise_chunk_safe((2, "—"))
        assert syn.call_count == 2
        np.testing.assert_array_equal(audio, good)

    def test_persistent_failure_returns_silence_pad(self):
        """All attempts exhausted — must NOT raise; must return short silence
        so the surrounding chunks still concatenate and play."""
        with patch.object(
            voice_module,
            "_synthesise_chunk",
            side_effect=RuntimeError("perma-fail"),
        ) as syn:
            audio = _synthesise_chunk_safe((3, "..."))
        assert syn.call_count == voice_module._TTS_MAX_ATTEMPTS
        assert audio.size == voice_module._TTS_FALLBACK_SAMPLES
        assert audio.dtype == np.float32
        assert float(np.abs(audio).max()) == 0.0  # silent pad

    def test_persistent_silent_returns_silence_pad(self):
        silent = np.zeros(10, dtype=np.float32)
        with patch.object(
            voice_module,
            "_synthesise_chunk",
            return_value=silent,
        ) as syn:
            audio = _synthesise_chunk_safe((4, "***"))
        assert syn.call_count == voice_module._TTS_MAX_ATTEMPTS
        assert audio.size == voice_module._TTS_FALLBACK_SAMPLES
