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
        # 0.1 amplitude is well under _TTS_SATURATION_THRESHOLD (0.95) so
        # the new detector treats it as legitimate speech.
        good = np.full(48_000, 0.1, dtype=np.float32)
        with patch.object(voice_module, "_synthesise_chunk", return_value=good) as syn:
            audio = _synthesise_chunk_safe((0, "hello world"))
        assert syn.call_count == 1
        np.testing.assert_array_equal(audio, good)

    def test_retries_on_exception_then_succeeds(self):
        good = np.full(48_000, 0.1, dtype=np.float32)
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
        good = np.full(48_000, 0.1, dtype=np.float32)
        with patch.object(
            voice_module,
            "_synthesise_chunk",
            side_effect=[silent, good],
        ) as syn:
            audio = _synthesise_chunk_safe((2, "—"))
        assert syn.call_count == 2
        np.testing.assert_array_equal(audio, good)

    def test_persistent_failure_falls_back_to_sentence_rescue(self):
        """All attempts exhausted on the whole chunk — rescue splits by
        sentence; each failing sentence becomes a short silence pad. Must
        NOT raise; surrounding chunks still concatenate and play."""
        # Single-sentence chunk → rescue splits into 1 sentence → 1 pad.
        with patch.object(
            voice_module,
            "_synthesise_chunk",
            side_effect=RuntimeError("perma-fail"),
        ) as syn:
            audio = _synthesise_chunk_safe((3, "..."))
        # 2 retry attempts on the whole chunk + 1 rescue attempt on the
        # one sentence = 3 calls total.
        assert syn.call_count == voice_module._TTS_MAX_ATTEMPTS + 1
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
        # 2 retries + 1 rescue sentence (the whole "***" un-splittable string).
        assert syn.call_count == voice_module._TTS_MAX_ATTEMPTS + 1
        assert audio.size == voice_module._TTS_FALLBACK_SAMPLES

    def test_retries_on_saturated_then_succeeds(self):
        """Kokoro DC-saturation (every sample at ±1.0) sounds like silence
        through any DC-blocked playback chain. Detector flags it; retry
        recovers with clean speech."""
        saturated = np.ones(48_000, dtype=np.float32)  # 100% at peak
        good = np.full(48_000, 0.1, dtype=np.float32)  # well under 0.95
        with patch.object(
            voice_module,
            "_synthesise_chunk",
            side_effect=[saturated, good],
        ) as syn:
            audio = _synthesise_chunk_safe((5, "2,097 tests at 0.973."))
        assert syn.call_count == 2
        np.testing.assert_array_equal(audio, good)

    def test_persistent_saturation_triggers_sentence_rescue(self):
        """Whole chunk saturates on every retry → rescue splits into
        sentences → if every sentence still saturates, output is
        N × silence pad (one per sentence)."""
        saturated = np.ones(48_000, dtype=np.float32)
        with patch.object(
            voice_module,
            "_synthesise_chunk",
            return_value=saturated,
        ) as syn:
            audio = _synthesise_chunk_safe((6, "First. Second. Third."))
        # 2 chunk-level retries + 3 sentence-level rescue calls = 5 calls.
        assert syn.call_count == voice_module._TTS_MAX_ATTEMPTS + 3
        # 3 silence pads concatenated.
        assert audio.size == voice_module._TTS_FALLBACK_SAMPLES * 3
        assert float(np.abs(audio).max()) == 0.0

    def test_sentence_rescue_preserves_good_sentences(self):
        """The whole point of sentence rescue: when ONE sentence of a chunk
        breaks Kokoro, the OTHERS still get spoken. Listener loses one
        sentence, not the whole 25-second chunk."""
        saturated = np.ones(48_000, dtype=np.float32)
        good_a = np.full(36_000, 0.1, dtype=np.float32)
        good_c = np.full(24_000, 0.2, dtype=np.float32)
        # Whole-chunk attempts saturate; rescue gets good/sat/good per sentence.
        with patch.object(
            voice_module,
            "_synthesise_chunk",
            side_effect=[saturated, saturated, good_a, saturated, good_c],
        ) as syn:
            audio = _synthesise_chunk_safe(
                (7, "Sentence A here. Bad sentence with 2,097 tokens. Sentence C."),
            )
        assert syn.call_count == 5
        # good_a + pad (saturated rescue) + good_c
        expected_size = 36_000 + voice_module._TTS_FALLBACK_SAMPLES + 24_000
        assert audio.size == expected_size
        # Both good clips' amplitudes must appear in the output.
        assert float(audio[:36_000].max()) == pytest.approx(0.1)
        assert float(audio[-24_000:].max()) == pytest.approx(0.2)


@pytest.mark.unit
class TestIsChunkBroken:
    """Direct coverage for the failure-mode classifier so future Kokoro
    quirks have an obvious place to land."""

    def test_normal_speech_is_good(self):
        rng = np.random.default_rng(0)
        speech = rng.normal(0, 0.1, 48_000).astype(np.float32)
        broken, reason = voice_module._is_chunk_broken(speech)
        assert not broken
        assert reason == ""

    def test_empty_array_is_broken(self):
        broken, reason = voice_module._is_chunk_broken(np.zeros(50, dtype=np.float32))
        assert broken
        assert reason == "empty"

    def test_dc_saturated_is_broken(self):
        broken, reason = voice_module._is_chunk_broken(
            np.ones(48_000, dtype=np.float32),
        )
        assert broken
        assert reason.startswith("saturated(")

    def test_negative_dc_saturated_is_broken(self):
        broken, reason = voice_module._is_chunk_broken(
            -np.ones(48_000, dtype=np.float32),
        )
        assert broken
        assert reason.startswith("saturated(")

    def test_loud_speech_with_brief_clipping_is_good(self):
        """Real speech can clip a few samples on a loud syllable; that's
        fine — only sustained saturation is rejected."""
        rng = np.random.default_rng(1)
        audio = rng.normal(0, 0.2, 48_000).astype(np.float32)
        # 5% of samples briefly at peak — well under 25% threshold.
        idx = rng.choice(48_000, 2_400, replace=False)
        audio[idx] = 1.0
        broken, _ = voice_module._is_chunk_broken(audio)
        assert not broken


@pytest.mark.unit
class TestTrimChunkSilence:
    """Kokoro emits 0.4-0.7s of trailing silence per chunk; concatenating
    a multi-chunk reply produced audible gaps every 3-4s. Trimming brings
    inter-chunk pauses down to a natural ~100ms."""

    def _voiced(self, n=24_000):
        # 1s of "voiced" audio above threshold.
        return np.full(n, 0.1, dtype=np.float32)

    def _silent(self, n=12_000):
        return np.zeros(n, dtype=np.float32)

    def test_trims_long_trailing_silence(self):
        from api.voice import _trim_chunk_silence, _TTS_TRAIL_KEEP_SAMPLES
        audio = np.concatenate([self._voiced(24_000), self._silent(15_000)])
        out = _trim_chunk_silence(audio)
        # Trailing silence capped at _TTS_TRAIL_KEEP_SAMPLES (100ms).
        assert out.size == 24_000 + _TTS_TRAIL_KEEP_SAMPLES

    def test_trims_long_leading_silence(self):
        from api.voice import _trim_chunk_silence, _TTS_LEAD_KEEP_SAMPLES
        audio = np.concatenate([self._silent(8_000), self._voiced(24_000)])
        out = _trim_chunk_silence(audio)
        # Leading silence capped at _TTS_LEAD_KEEP_SAMPLES (50ms).
        assert out.size == _TTS_LEAD_KEEP_SAMPLES + 24_000

    def test_short_silence_passes_through(self):
        # Already under both keep budgets — no trimming.
        from api.voice import _trim_chunk_silence
        audio = np.concatenate([
            np.zeros(800, dtype=np.float32),
            self._voiced(24_000),
            np.zeros(1500, dtype=np.float32),
        ])
        out = _trim_chunk_silence(audio)
        assert out.size == audio.size

    def test_all_silent_unchanged(self):
        # Retry wrapper handles empty-output cases via _TTS_MIN_SAMPLES;
        # trimmer must NOT mutate length here so that pathway still fires.
        from api.voice import _trim_chunk_silence
        audio = np.zeros(5000, dtype=np.float32)
        out = _trim_chunk_silence(audio)
        assert out.size == audio.size

    def test_empty_array_unchanged(self):
        from api.voice import _trim_chunk_silence
        audio = np.zeros(0, dtype=np.float32)
        out = _trim_chunk_silence(audio)
        assert out.size == 0

    def test_voiced_only_unchanged(self):
        from api.voice import _trim_chunk_silence
        audio = self._voiced(48_000)
        out = _trim_chunk_silence(audio)
        assert out.size == audio.size

    def test_preserves_dtype(self):
        from api.voice import _trim_chunk_silence
        audio = np.concatenate([self._voiced(), self._silent(10_000)]).astype(np.float32)
        out = _trim_chunk_silence(audio)
        assert out.dtype == np.float32
