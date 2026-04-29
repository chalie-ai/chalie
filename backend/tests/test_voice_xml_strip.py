import pytest

from api.voice import _clean_for_tts


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
