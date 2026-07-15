"""Test that the message processor's _build_messages() correctly assembles its
ProviderRequest DTO with image attachment from ``PromptService.image()``
(``VisionConfig.get_image`` under the old spine)."""

import base64
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from configs.channels.vision import VisionConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def test_build_send_messages_attaches_image_from_get_image(
    tmp_path: Path, db: sqlite3.Connection
) -> None:

    img = tmp_path / "a.png"
    img.write_bytes(_png_bytes())

    config = VisionConfig(PolicyChannel.CHAT)
    mp = MessageProcessor(
        config, raw_input="what is this",
        metadata={"image_path": str(img), "mime_type": "image/png"},
    )

    messages = mp._build_messages()

    assert messages[0]["role"] == "user"
    assert cast(dict[str, object], messages[0]["image"])["mime_type"] == "image/png"
    assert cast(dict[str, object], messages[0]["image"])["data"] == base64.b64encode(_png_bytes()).decode()


def test_build_send_messages_no_image_when_get_image_returns_none(db: sqlite3.Connection) -> None:

    config = VisionConfig(PolicyChannel.CHAT)
    mp = MessageProcessor(config, raw_input="what is this")

    messages = mp._build_messages()

    assert messages[0]["role"] == "user"
    assert "image" not in messages[0]
