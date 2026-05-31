"""Unit tests for vision_service — no network; create_llm_service is mocked."""
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def test_build_vision_config_maps_fields():
    from services import vision_service
    cfg = vision_service.build_vision_config(
        {"platform": "openai", "model": "gpt-4o", "api_key": "sk", "host": None}
    )
    assert cfg["platform"] == "openai"
    assert cfg["model"] == "gpt-4o"
    assert cfg["api_key"] == "sk"
    assert "host" not in cfg          # None host omitted
    assert cfg["timeout"] == 60


def test_send_image_attaches_image_key_and_returns_text():
    from services import vision_service
    fake_llm = MagicMock()
    fake_llm.send_messages.return_value = MagicMock(text="a red square")
    with patch("services.llm_service.create_llm_service", return_value=fake_llm):
        out = vision_service.send_image_with_config(
            {"platform": "ollama", "model": "llava"}, b"\x89PNG...", "what?",
        )
    assert out == "a red square"
    # the message handed to send_messages must carry the image key (base64)
    _system, messages = fake_llm.send_messages.call_args[0][:2]
    assert messages[0]["content"] == "what?"
    assert messages[0]["image"]["mime_type"] == "image/png"
    assert isinstance(messages[0]["image"]["data"], str) and messages[0]["image"]["data"]


def test_send_image_returns_none_on_exception():
    from services import vision_service
    with patch("services.llm_service.create_llm_service", side_effect=RuntimeError("boom")):
        assert vision_service.send_image_with_config({}, b"x", "p") is None
