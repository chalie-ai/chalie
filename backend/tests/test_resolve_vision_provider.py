"""Feature test: Providers._resolve branches to the brain's Vision Provider
when the bound config sets uses_vision_provider (TKT-838).

Proves the framework gap #2: a VisionConfig-bound MessageProcessor resolves the
DB-configured vision provider — NOT the globally selected provider. Drives the
real Providers._resolve() against the real test DB, with a real provider row
created by the production ProviderDbService factory (vision probe forced on,
mirroring test_image_context_ocr_only.py), and asserts the resolved LLM service
wraps the VISION provider's model/host.
"""

import threading

import pytest

from services.database_service import get_shared_db_service
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.provider_db_service import ProviderDbService
from services.providers import Providers

pytestmark = pytest.mark.unit


def _make_mp(config) -> MessageProcessor:
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what is this", {})
    mp.config = config
    mp.uid = None
    mp.current_iteration = 0
    mp.cancel_event = threading.Event()
    mp.thinking_level = "low"
    mp.thinking_override = None
    return mp


def test_resolve_uses_vision_provider_not_global(db):
    from abilities.vision import VisionConfig
    from configs.channels.user import UserConfig

    svc = ProviderDbService(get_shared_db_service())

    # The GLOBAL selected provider — a distinct ollama model on a distinct host.
    global_provider = svc.create_provider(
        {
            "name": "global-text",
            "platform": "ollama",
            "model": "qwen3",
            "host": "http://127.0.0.1:2",
            "api_key": "",
        }
    )
    svc.set_selected_provider(global_provider["id"])

    # The VISION provider — different model/host. Force supports_vision=1 (the
    # state a successful probe writes) so it is genuinely resolvable.
    vision_provider = svc.create_provider(
        {
            "name": "probe-vision",
            "platform": "ollama",
            "model": "llava",
            "host": "http://127.0.0.1:1",
            "api_key": "",
        }
    )
    vid = vision_provider["id"]
    db.execute("UPDATE providers SET supports_vision = 1 WHERE id = ?", (vid,))
    db.commit()
    svc.set_vision_provider(vid)
    assert svc.get_vision_provider() is not None

    # A plain UserConfig resolves the GLOBAL provider (control).
    user_mp = _make_mp(UserConfig())
    user_llm = Providers(user_mp)._resolve()
    assert user_llm._service.model == "qwen3"
    assert user_llm._service.host == "http://127.0.0.1:2"

    # A VisionConfig (uses_vision_provider=True) resolves the VISION provider.
    vision_mp = _make_mp(VisionConfig(ProcessorConfig.POLICY_CHANNEL.CHAT))
    vision_llm = Providers(vision_mp)._resolve()
    assert vision_llm._service.model == "llava"
    assert vision_llm._service.host == "http://127.0.0.1:1"


def test_resolve_vision_raises_when_no_vision_provider_configured(db):
    """uses_vision_provider set but no vision provider in the DB → fail loud,
    never silently fall back to the global provider."""
    from abilities.vision import VisionConfig

    svc = ProviderDbService(get_shared_db_service())
    svc.set_vision_provider(None)
    assert svc.get_vision_provider() is None

    vision_mp = _make_mp(VisionConfig(ProcessorConfig.POLICY_CHANNEL.CHAT))
    with pytest.raises(RuntimeError, match="no vision provider"):
        Providers(vision_mp)._resolve()
