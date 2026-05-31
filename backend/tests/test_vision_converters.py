"""Unit tests for image-block emission in the per-platform message converters."""
import pytest

pytestmark = pytest.mark.unit

_IMG = {"data": "QkFTRTY0", "mime_type": "image/png"}


def test_anthropic_emits_image_block():
    from services.llm_service import _anthropic_convert_messages
    out = _anthropic_convert_messages(
        [{"role": "user", "content": "what is this?", "image": _IMG}]
    )
    assert out == [{
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "QkFTRTY0"}},
        ],
    }]


def test_anthropic_image_only_no_text():
    from services.llm_service import _anthropic_convert_messages
    out = _anthropic_convert_messages([{"role": "user", "content": "", "image": _IMG}])
    assert out[0]["content"] == [
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "QkFTRTY0"}},
    ]


def test_anthropic_text_only_unchanged():
    from services.llm_service import _anthropic_convert_messages
    msg = {"role": "user", "content": "plain text"}
    assert _anthropic_convert_messages([msg]) == [msg]


def test_openai_emits_image_url_block():
    from services.llm_service import _openai_convert_messages
    out = _openai_convert_messages(
        [{"role": "user", "content": "describe", "image": _IMG}]
    )
    assert out == [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,QkFTRTY0"}},
        ],
    }]


def test_openai_text_only_unchanged():
    from services.llm_service import _openai_convert_messages
    msg = {"role": "user", "content": "plain"}
    assert _openai_convert_messages([msg]) == [msg]


def test_gemini_emits_inline_data_part():
    from services.llm_service import _gemini_convert_messages
    out = _gemini_convert_messages(
        [{"role": "user", "content": "look", "image": _IMG}]
    )
    assert out == [{
        "role": "user",
        "parts": [
            {"text": "look"},
            {"inline_data": {"mime_type": "image/png", "data": "QkFTRTY0"}},
        ],
    }]


def test_gemini_text_only_unchanged():
    from services.llm_service import _gemini_convert_messages
    out = _gemini_convert_messages([{"role": "user", "content": "plain"}])
    assert out == [{"role": "user", "parts": [{"text": "plain"}]}]


def test_ollama_emits_images_array():
    from services.ollama_service import _ollama_convert_messages
    out = _ollama_convert_messages(
        [{"role": "user", "content": "what", "image": _IMG}]
    )
    assert out == [{"role": "user", "content": "what", "images": ["QkFTRTY0"]}]


def test_ollama_text_only_unchanged():
    from services.ollama_service import _ollama_convert_messages
    msg = {"role": "user", "content": "plain"}
    assert _ollama_convert_messages([msg]) == [msg]
