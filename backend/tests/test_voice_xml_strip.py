"""Voice TTS text-cleanup tests."""

import pytest

from api.voice import _clean_for_tts


@pytest.mark.unit
class TestCleanForTts:

    def test_markdown_header_strips(self):
        assert _clean_for_tts("# My Header") == "My Header"

    def test_markdown_fenced_code_strips(self):
        result = _clean_for_tts("```python\nprint('hi')\n```")
        assert "```" not in result
        assert "print" in result

    def test_markdown_link_keeps_text(self):
        assert _clean_for_tts("[click here](https://example.com)") == "click here"

    def test_markdown_image_drops_entirely(self):
        assert _clean_for_tts("![alt text](img.png)") == ""

    def test_url_rewritten_to_spoken_host(self):
        result = _clean_for_tts("see http://google.com/123 for details")
        assert "http" not in result
        assert "/123" not in result
        assert "google dot com" in result
        assert "details" in result

    def test_bare_url_round_trip_produces_spoken_host(self):
        result = _clean_for_tts("http://google.com/path/to/page")
        assert result == "google dot com"

    def test_minified_list_items_separated(self):
        result = _clean_for_tts("<ul><li>one</li><li>two</li><li>three</li></ul>")
        assert result == "one. two. three."

    def test_list_item_existing_punctuation_preserved(self):
        result = _clean_for_tts("<ul><li>Boil water.</li><li>Add pasta!</li><li>Stir</li></ul>")
        assert result == "Boil water. Add pasta! Stir."

    def test_markdown_list_gets_pauses(self):
        result = _clean_for_tts("Steps:\n- one\n- two\n- three\n")
        assert result == "Steps: one. two. three."

    def test_html_drops_actions_block(self):
        result = _clean_for_tts('<p>pick one</p><actions><action label="A" value="a"/></actions>')
        assert result == "pick one"
