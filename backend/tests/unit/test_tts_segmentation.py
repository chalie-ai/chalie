import pytest

from services.voice_transcript_service import _segment_for_tts, _MAX_TTS_CHUNK_CHARS

# ── helpers ────────────────────────────────────────────────────────────────────

_LIMIT = _MAX_TTS_CHUNK_CHARS


def _word(length: int, char: str = "a") -> str:
    return char * length


def _sentence_of(length: int) -> str:
    # "word word ... word." — fill with 4-char words + spaces then trim/pad.
    assert length >= 2
    body = "word " * (length // 5 + 2)
    # Trim to length-1 then append the period.
    return body[: length - 1].rstrip() + "."


# ── test class ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSegmentForTts:

    # ── trivial / edge inputs ──────────────────────────────────────────────

    def test_empty_string_returns_empty_list(self) -> None:
        assert _segment_for_tts("") == []

    def test_whitespace_only_returns_original_via_fallback(self) -> None:
        # Pure whitespace: all sentences strip to "" and are skipped, so chunks
        # stays empty.  The final guard "return chunks if chunks else [text]"
        # returns [text] to honour the "always non-empty list" contract.
        text = "   \t\n  "
        result = _segment_for_tts(text)
        assert result == [text]

    def test_short_text_returns_single_element_list(self) -> None:
        text = "Hello, world."
        result = _segment_for_tts(text)
        assert result == [text]

    def test_text_exactly_at_limit_stays_as_one_chunk(self) -> None:
        # A single sentence whose length is exactly _MAX_TTS_CHUNK_CHARS must
        # not be split — the boundary is inclusive.
        text = _word(_LIMIT)   # one "word" (no spaces) of exactly the limit
        result = _segment_for_tts(text)
        assert len(result) == 1
        assert result[0] == text

    # ── sentence-level greedy accumulation ────────────────────────────────

    def test_multiple_short_sentences_fit_in_one_chunk(self) -> None:
        # Three short sentences whose combined length is well under the limit.
        text = "Hello. How are you? I am fine."
        result = _segment_for_tts(text)
        assert len(result) == 1
        # The whole text (stripped) comes back as one chunk.
        assert result[0] == text

    def test_sentences_accumulate_greedily_before_flushing(self) -> None:
        # Build two groups of sentences: each group's combined length fits in
        # the limit, but adding any sentence from group 2 to group 1 would
        # overflow.  Expect exactly 2 chunks.
        chunk_budget = _LIMIT - 10          # leave 10 chars headroom
        s1 = _word(chunk_budget // 2) + "."
        s2 = _word(chunk_budget - len(s1) - 1) + "."   # fills rest of budget
        # s3 is long enough that "s1 s2 s3" would overflow but "s3" alone fits.
        s3 = _word(50) + "."
        text = f"{s1} {s2} {s3}"
        result = _segment_for_tts(text)
        assert len(result) == 2
        # First chunk contains the two sentences that fit together.
        assert s1 in result[0]
        assert s2 in result[0]
        assert s3 in result[1]

    def test_sentences_that_exceed_limit_when_combined_split_across_chunks(self) -> None:
        # Each sentence individually fits, but concatenating them overflows.
        s1 = _word(200) + "."
        s2 = _word(200) + "."   # s1 + " " + s2 = 401 > 320
        text = f"{s1} {s2}"
        result = _segment_for_tts(text)
        assert len(result) == 2
        assert result[0] == s1
        assert result[1] == s2

    # ── punctuation preserved with preceding text ──────────────────────────

    def test_sentence_terminator_stays_with_preceding_sentence(self) -> None:
        # The split regex uses a look-behind so the period/! stays with the
        # sentence it ends, not the start of the next chunk.
        long_s1 = _word(280) + "."   # just under limit on its own
        long_s2 = _word(280) + "."
        text = f"{long_s1} {long_s2}"
        result = _segment_for_tts(text)
        assert all(r.endswith(".") for r in result), (
            f"Sentence terminator was lost: {result}"
        )

    # ── clause-level fallback ──────────────────────────────────────────────

    def test_single_long_sentence_splits_on_clause_boundary(self) -> None:
        # One sentence > 320 chars, split by a comma in the middle.
        part_a = _word(200)
        part_b = _word(200)
        sentence = f"{part_a}, {part_b}."
        result = _segment_for_tts(sentence)
        assert len(result) >= 2
        # Reassembling should recover both word-blocks.
        joined = " ".join(result)
        assert part_a in joined
        assert part_b in joined

    # ── whitespace fallback ────────────────────────────────────────────────

    def test_single_long_clause_with_no_punctuation_splits_on_whitespace(self) -> None:
        # Build a clause longer than the limit with no sentence or clause
        # punctuation at all — must fall through to word-level splitting.
        words = [_word(30) for _ in range(20)]   # 20 × 30 chars = 600 chars
        text = " ".join(words)
        result = _segment_for_tts(text)
        assert len(result) >= 2
        # Every chunk must be ≤ the limit.
        for chunk in result:
            assert len(chunk) <= _LIMIT, f"Chunk too long ({len(chunk)}): {chunk!r}"

