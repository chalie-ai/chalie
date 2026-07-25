"""Voice TTS text-cleanup tests."""

import pytest

from services.voice_transcript_service import _clean_for_tts


@pytest.mark.unit
class TestCleanForTts:

    def test_markdown_header_strips(self) -> None:
        assert _clean_for_tts("# My Header") == "My Header"

    def test_markdown_fenced_code_strips(self) -> None:
        result = _clean_for_tts("```python\nprint('hi')\n```")
        assert "```" not in result
        assert "print" in result

    def test_markdown_link_keeps_text(self) -> None:
        assert _clean_for_tts("[click here](https://example.com)") == "click here"

    def test_markdown_image_drops_entirely(self) -> None:
        assert _clean_for_tts("![alt text](img.png)") == ""

    def test_url_rewritten_to_spoken_host(self) -> None:
        result = _clean_for_tts("see http://google.com/123 for details")
        assert "http" not in result
        assert "/123" not in result
        assert "google dot com" in result
        assert "details" in result

    def test_bare_url_round_trip_produces_spoken_host(self) -> None:
        result = _clean_for_tts("http://google.com/path/to/page")
        assert result == "google dot com"

    def test_minified_list_items_separated(self) -> None:
        result = _clean_for_tts("<ul><li>one</li><li>two</li><li>three</li></ul>")
        assert result == "one. two. three."

    def test_list_item_existing_punctuation_preserved(self) -> None:
        result = _clean_for_tts("<ul><li>Boil water.</li><li>Add pasta!</li><li>Stir</li></ul>")
        assert result == "Boil water. Add pasta! Stir."

    def test_markdown_list_gets_pauses(self) -> None:
        result = _clean_for_tts("Steps:\n- one\n- two\n- three\n")
        assert result == "Steps: one. two. three."
