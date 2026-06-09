"""Feature test: Providers._resolve branches to the brain's Vision Provider
when dto.type == ProviderType.VISION.

Providers is now mp-free.
_resolve(provider_type) takes a ProviderType instead of reading mp.config.
It returns a ProviderClient, not a LoggingLLMService, so the attribute
access path changes from ._service.model → .model directly on the client.

Proves: a Providers()._resolve(ProviderType.VISION) resolves the DB-configured
vision provider — NOT the globally selected provider. Drives the real Providers
facade against the real test DB, with real provider rows created by the
production ProviderDbService factory.
"""

import pytest

from services.database_service import get_shared_db_service
from services.provider_api import ProviderType
from services.provider_db_service import ProviderDbService
from services.providers import Providers

pytestmark = pytest.mark.unit


def test_resolve_chat_returns_global_provider(db):
    """Providers()._resolve(ProviderType.CHAT) returns the globally selected provider."""
    svc = ProviderDbService(get_shared_db_service())

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

    from services.provider_cache_service import ProviderCacheService
    ProviderCacheService.invalidate()

    client = Providers()._resolve(ProviderType.CHAT)

    # The client is an OllamaClient (ProviderClient subclass) — assert on its fields directly.
    assert client.model == "qwen3"
    assert client.host == "http://127.0.0.1:2"


def test_resolve_vision_returns_vision_provider_not_global(db):
    """Providers()._resolve(ProviderType.VISION) returns the DB vision provider,
    NOT the globally selected provider — confirmed that the two are distinct."""
    svc = ProviderDbService(get_shared_db_service())

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

    from services.provider_cache_service import ProviderCacheService
    ProviderCacheService.invalidate()

    chat_client = Providers()._resolve(ProviderType.CHAT)
    vision_client = Providers()._resolve(ProviderType.VISION)

    # Chat resolves the global provider.
    assert chat_client.model == "qwen3"
    assert chat_client.host == "http://127.0.0.1:2"

    # Vision resolves the DB vision provider — completely different model/host.
    assert vision_client.model == "llava"
    assert vision_client.host == "http://127.0.0.1:1"


def test_resolve_vision_raises_when_no_vision_provider_configured(db):
    """uses_vision_provider set but no vision provider in the DB → fail loud,
    never silently fall back to the global provider."""
    svc = ProviderDbService(get_shared_db_service())
    svc.set_vision_provider(None)
    assert svc.get_vision_provider() is None

    with pytest.raises(RuntimeError, match="no vision provider"):
        Providers()._resolve(ProviderType.VISION)
