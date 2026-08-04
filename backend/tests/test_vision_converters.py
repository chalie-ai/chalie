"""Unit tests for image-block emission in the per-platform message converters."""
from typing import cast

import pytest

pytestmark = pytest.mark.unit

_IMG: dict[str, object] = {"data": "QkFTRTY0", "mime_type": "image/png"}


def test_anthropic_emits_image_block() -> None:
    from services.llm_clients.anthropic import _anthropic_convert_messages
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


def test_anthropic_image_only_no_text() -> None:
    from services.llm_clients.anthropic import _anthropic_convert_messages
    out = _anthropic_convert_messages([{"role": "user", "content": "", "image": _IMG}])
    assert out[0]["content"] == [
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "QkFTRTY0"}},
    ]


def test_anthropic_text_only_unchanged() -> None:
    from services.llm_clients.anthropic import _anthropic_convert_messages
    msg: dict[str, object] = {"role": "user", "content": "plain text"}
    assert _anthropic_convert_messages([msg]) == [msg]


def test_openai_emits_image_url_block() -> None:
    from services.llm_clients.openai_compatible import _openai_convert_messages
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


def test_openai_image_only_no_text() -> None:
    from services.llm_clients.openai_compatible import _openai_convert_messages
    out = _openai_convert_messages([{"role": "user", "content": "", "image": _IMG}])
    assert out[0]["content"] == [
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,QkFTRTY0"}},
    ]


def test_openai_text_only_unchanged() -> None:
    from services.llm_clients.openai_compatible import _openai_convert_messages
    msg: dict[str, object] = {"role": "user", "content": "plain"}
    assert _openai_convert_messages([msg]) == [msg]


def test_openai_assistant_tool_calls_unchanged() -> None:
    # Guards the image branch staying in the FINAL else, after the
    # assistant+tool_calls and tool branches — an image-less assistant
    # tool-call message must pass through its dedicated branch untouched.
    from services.llm_clients.openai_compatible import _openai_convert_messages
    msg: dict[str, object] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "name": "f", "input": {}}],
    }
    out = _openai_convert_messages([msg])
    assert out[0]["role"] == "assistant"
    assert cast(dict[str, object], cast(list[dict[str, object]], out[0]["tool_calls"])[0]["function"])["name"] == "f"
    assert "content" not in out[0] or not isinstance(out[0]["content"], list)


def test_gemini_emits_inline_data_part() -> None:
    from services.llm_clients.gemini import _gemini_convert_messages
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


def test_gemini_image_only_keeps_empty_text_part() -> None:
    # Deliberate divergence from Anthropic/OpenAI: Gemini always emits a
    # leading text part (even ""), then the inline_data part. Documented in
    # message-processing.md; Gemini tolerates the empty text part.
    from services.llm_clients.gemini import _gemini_convert_messages
    out = _gemini_convert_messages([{"role": "user", "content": "", "image": _IMG}])
    assert out == [{
        "role": "user",
        "parts": [
            {"text": ""},
            {"inline_data": {"mime_type": "image/png", "data": "QkFTRTY0"}},
        ],
    }]


def test_gemini_text_only_unchanged() -> None:
    from services.llm_clients.gemini import _gemini_convert_messages
    out = _gemini_convert_messages([{"role": "user", "content": "plain"}])
    assert out == [{"role": "user", "parts": [{"text": "plain"}]}]


def test_ollama_emits_images_array() -> None:
    from services.llm_clients.ollama import _ollama_convert_messages
    out = _ollama_convert_messages(
        [{"role": "user", "content": "what", "image": _IMG}]
    )
    assert out == [{"role": "user", "content": "what", "images": ["QkFTRTY0"]}]


def test_ollama_image_only() -> None:
    from services.llm_clients.ollama import _ollama_convert_messages
    out = _ollama_convert_messages([{"role": "user", "content": "", "image": _IMG}])
    assert out == [{"role": "user", "content": "", "images": ["QkFTRTY0"]}]


def test_ollama_text_only_unchanged() -> None:
    from services.llm_clients.ollama import _ollama_convert_messages
    msg: dict[str, object] = {"role": "user", "content": "plain"}
    assert _ollama_convert_messages([msg]) == [msg]
