"""Test that the message processor's _build_send_messages() correctly
assembles its ProviderApiRequest DTO with image attachment from config.get_image."""

import base64
from pathlib import Path
from typing import cast

import pytest

from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def test_build_send_messages_attaches_image_from_get_image(tmp_path: Path) -> None:
    from configs.channels.vision import VisionConfig
    from services.processor_config import ProcessorConfig

    img = tmp_path / "a.png"
    img.write_bytes(_png_bytes())

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what is this", {"image_path": str(img), "mime_type": "image/png"})
    mp.config = VisionConfig(ProcessorConfig.PolicyChannel.CHAT)

    messages = mp._build_send_messages()

    assert messages[0]["role"] == "user"
    assert cast(dict[str, object], messages[0]["image"])["mime_type"] == "image/png"
    assert cast(dict[str, object], messages[0]["image"])["data"] == base64.b64encode(_png_bytes()).decode()


def test_build_send_messages_no_image_when_get_image_returns_none(tmp_path: Path) -> None:
    from configs.channels.vision import VisionConfig
    from services.processor_config import ProcessorConfig

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what is this", {})
    mp.config = VisionConfig(ProcessorConfig.PolicyChannel.CHAT)

    messages = mp._build_send_messages()

    assert messages[0]["role"] == "user"
    assert "image" not in messages[0]
