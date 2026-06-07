"""Feature test: the mp's user-message builder attaches an image from
config.get_image when the metadata carries image_path (TKT-838).

TKT-846 spec change (doc 131 resolution #1): Providers is mp-free.
The method that builds the user-message list — previously
Providers._build_user_messages() — moved to MessageProcessor._build_send_messages()
where it is called when assembling the ProviderApiRequest DTO. This test
exercises that real production method (the same entry point _loop uses) and
asserts the image attachment survives into the message dict.
"""

import base64

import pytest

from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def test_build_send_messages_attaches_image_from_get_image(tmp_path):
    """_build_send_messages() on a VisionConfig mp with image_path metadata
    produces a message whose 'image' dict carries the correct mime_type and
    base64-encoded data — proving the DTO construction carries vision payloads."""
    from configs.channels.vision import VisionConfig
    from services.processor_config import ProcessorConfig

    img = tmp_path / "a.png"
    img.write_bytes(_png_bytes())

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what is this", {"image_path": str(img), "mime_type": "image/png"})
    mp.config = VisionConfig(ProcessorConfig.POLICY_CHANNEL.CHAT)

    messages = mp._build_send_messages()

    assert messages[0]["role"] == "user"
    assert messages[0]["image"]["mime_type"] == "image/png"
    assert messages[0]["image"]["data"] == base64.b64encode(_png_bytes()).decode()


def test_build_send_messages_no_image_when_get_image_returns_none(tmp_path):
    """A VisionConfig with no image_path in metadata attaches no image —
    the framework hook returns None and the message stays a bare user message."""
    from configs.channels.vision import VisionConfig
    from services.processor_config import ProcessorConfig

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what is this", {})
    mp.config = VisionConfig(ProcessorConfig.POLICY_CHANNEL.CHAT)

    messages = mp._build_send_messages()

    assert messages[0]["role"] == "user"
    assert "image" not in messages[0]
