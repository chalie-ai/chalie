import base64
import threading

import pytest

from services.message_processor import MessageProcessor
from services.providers import Providers

pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def test_build_user_messages_attaches_image_from_get_image(tmp_path):
    from abilities.vision import VisionConfig
    from services.processor_config import ProcessorConfig

    img = tmp_path / "a.png"
    img.write_bytes(_png_bytes())

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what is this", {"image_path": str(img), "mime_type": "image/png"})
    mp.config = VisionConfig(ProcessorConfig.POLICY_CHANNEL.CHAT)
    mp.uid = None
    mp.current_iteration = 0
    mp.cancel_event = threading.Event()
    mp.thinking_level = "low"
    mp.thinking_override = None

    messages = Providers(mp)._build_user_messages()

    assert messages[0]["role"] == "user"
    assert messages[0]["image"]["mime_type"] == "image/png"
    assert messages[0]["image"]["data"] == base64.b64encode(_png_bytes()).decode()


def test_build_user_messages_no_image_when_get_image_returns_none(tmp_path):
    """A VisionConfig with no image_path in metadata attaches no image — the
    framework hook returns None and the message stays a bare user message."""
    from abilities.vision import VisionConfig
    from services.processor_config import ProcessorConfig

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what is this", {})
    mp.config = VisionConfig(ProcessorConfig.POLICY_CHANNEL.CHAT)
    mp.uid = None
    mp.current_iteration = 0
    mp.cancel_event = threading.Event()
    mp.thinking_level = "low"
    mp.thinking_override = None

    messages = Providers(mp)._build_user_messages()

    assert messages[0]["role"] == "user"
    assert "image" not in messages[0]
